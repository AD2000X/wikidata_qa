"""
Entity linker for the Wikidata QA system.

Replaces the hardcoded DISAMBIGUATION_RULES / FALLBACK_QIDS approach
with a fully dynamic entity linking pipeline:

1. Candidate recall    – Wikidata Search API + fuzzy alias expansion
2. Embedding scoring   – Gemini gemini-embedding-001 cosine similarity
3. Context-aware rank  – question context biases disambiguation
4. NIL detection       – rejects when best candidate score is too low

All I/O goes through httpx (sync + async) — no new dependencies.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx

from config import (
    DEFAULT_LANGUAGE,
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT,
    DESCRIPTION_SIMILARITY_WEIGHT,
    ENTITY_RESOLVE_TTL,
    ENTITY_SEARCH_TTL,
    INTENT_CONFIG,
    TYPE_MATCH_BONUS,
    TYPE_MISMATCH_PENALTY,
    TYPE_TTL,
    WIKIDATA_API,
    WIKIDATA_SPARQL,
    # Entity linker settings
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_TTL,
    NIL_SCORE_THRESHOLD,
    EMBEDDING_SIMILARITY_WEIGHT,
    CONTEXT_SIMILARITY_WEIGHT,
)
from gemini_config import GEMINI_API_BASE, GEMINI_API_KEY


# =========================================================
# Exceptions
# =========================================================


class EntityLinkingError(Exception):
    """Base error for entity linking failures."""


class EntitySearchError(EntityLinkingError):
    """Wikidata entity search failed before disambiguation could run."""


class NILEntityError(EntityLinkingError):
    """Entity not found in knowledge base (NIL)."""


# =========================================================
# Data models
# =========================================================


@dataclass(frozen=True, slots=True)
class QIDCandidate:
    qid: str
    label: str
    description: str
    match_type: str = ""
    aliases: Tuple[str, ...] = ()


@dataclass(slots=True)
class LinkingResult:
    candidate: QIDCandidate
    score: float
    is_nil: bool = False
    candidate_scores: List[Dict[str, Any]] = field(default_factory=list)
    recall_source: str = "wikidata_search"


# =========================================================
# Text helpers (shared with wikidata_qa_llm.py)
# =========================================================

import re
import unicodedata


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = text.replace("&", " and ")
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^a-z0-9\s'-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _normalize_entity_alias(text: str) -> str:
    text = _normalize_text(text)
    for prefix in ("the ", "a ", "an "):
        if text.startswith(prefix):
            text = text[len(prefix):]
    return text


_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "for", "to", "is", "what", "who", "where",
    "was", "were", "are", "does", "do", "did", "list", "show", "with", "have", "has",
    "country", "birth", "spouse", "occupation", "capital", "population", "age", "old",
    "humans", "people", "city", "cities",
}


def _tokenize_semantic(text: str) -> Set[str]:
    parts = re.findall(r"[a-z0-9]+", _normalize_text(text))
    return {part for part in parts if part and part not in _STOPWORDS and len(part) > 1}


def _semantic_similarity(text_a: str, text_b: str) -> float:
    left = _tokenize_semantic(text_a)
    right = _tokenize_semantic(text_b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


# =========================================================
# Embedding client (Gemini REST API via httpx)
# =========================================================


def _compute_backoff(attempt: int, base: float = 1.5, jitter_max: float = 0.25) -> float:
    import random
    return (base ** max(0, attempt - 1)) + random.uniform(0.0, jitter_max)


def _should_retry_request(exc: Exception) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)):
        return True
    response = getattr(exc, "response", None)
    if response is not None:
        return getattr(response, "status_code", None) in {408, 409, 425, 429, 500, 502, 503, 504}
    return False


class GeminiEmbeddingClient:
    """Thin wrapper around the Gemini embedContent REST endpoint.

    Uses httpx directly — no SDK dependency.  Supports both sync
    and async calls with retry + caching.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = EMBEDDING_MODEL,
        dimensions: int = EMBEDDING_DIMENSIONS,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = 2,
    ) -> None:
        self.api_key = api_key or GEMINI_API_KEY
        self.model = model
        self.dimensions = dimensions
        self.timeout = timeout
        self.max_retries = max_retries
        self._endpoint = (
            f"{GEMINI_API_BASE}/models/{self.model}:embedContent"
            f"?key={self.api_key}"
        )

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def _build_body(self, text: str, task_type: str = "SEMANTIC_SIMILARITY") -> Dict[str, Any]:
        return {
            "model": f"models/{self.model}",
            "content": {"parts": [{"text": text}]},
            "taskType": task_type,
            "outputDimensionality": self.dimensions,
        }

    # ----- sync -----

    def embed(self, text: str, task_type: str = "SEMANTIC_SIMILARITY") -> List[float]:
        body = self._build_body(text, task_type)
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                with httpx.Client() as client:
                    response = client.post(
                        self._endpoint,
                        json=body,
                        headers={"Content-Type": "application/json"},
                        timeout=self.timeout,
                    )
                    response.raise_for_status()
                    data = response.json()
                    return data["embedding"]["values"]
            except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(_compute_backoff(attempt))
        raise EntityLinkingError(f"Embedding API failed: {last_exc}")

    # ----- async -----

    async def embed_async(
        self,
        text: str,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        task_type: str = "SEMANTIC_SIMILARITY",
    ) -> List[float]:
        body = self._build_body(text, task_type)
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                async with semaphore:
                    response = await client.post(
                        self._endpoint,
                        json=body,
                        headers={"Content-Type": "application/json"},
                        timeout=self.timeout,
                    )
                    response.raise_for_status()
                    data = response.json()
                    return data["embedding"]["values"]
            except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                await asyncio.sleep(_compute_backoff(attempt))
        raise EntityLinkingError(f"Async embedding API failed: {last_exc}")


# =========================================================
# Cosine similarity
# =========================================================


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# =========================================================
# Candidate recall
# =========================================================


def parse_search_payload(payload: Dict[str, Any]) -> List[QIDCandidate]:
    """Parse Wikidata Search API response into candidates."""
    candidates: List[QIDCandidate] = []
    for item in payload.get("search", []):
        aliases_raw = item.get("aliases", [])
        aliases = tuple(
            _normalize_entity_alias(alias if isinstance(alias, str) else alias.get("value", ""))
            for alias in aliases_raw
        )
        candidates.append(
            QIDCandidate(
                qid=item.get("id", ""),
                label=item.get("label", ""),
                description=item.get("description", ""),
                match_type=item.get("match", {}).get("type", ""),
                aliases=tuple(a for a in aliases if a),
            )
        )
    return candidates


class CandidateRecall:
    """Generates entity candidates from multiple sources.

    Sources:
    1. Wikidata Search API (primary) — searches by entity name
    2. Fuzzy recall — searches by normalized/cleaned entity name
       if different from original
    3. Alias expansion — searches by sub-phrases for multi-word entities

    Deduplicates by QID across all sources.
    """

    def __init__(
        self,
        *,
        api_url: str = WIKIDATA_API,
        language: str = DEFAULT_LANGUAGE,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        self.api_url = api_url
        self.language = language
        self.timeout = timeout

    def _search_params(self, query: str, language: str, limit: int) -> Dict[str, Any]:
        return {
            "action": "wbsearchentities",
            "search": query,
            "language": language,
            "format": "json",
            "limit": limit,
        }

    # ----- sync -----

    def recall(
        self,
        entity_name: str,
        client: httpx.Client,
        cache: Any,
        limit: int = 8,
    ) -> List[QIDCandidate]:
        """Multi-source candidate recall with deduplication."""
        seen_qids: Set[str] = set()
        candidates: List[QIDCandidate] = []

        # Source 1: exact search
        exact = self._search_sync(entity_name, client, cache, limit)
        for c in exact:
            if c.qid not in seen_qids:
                seen_qids.add(c.qid)
                candidates.append(c)

        # Source 2: normalised search (if different)
        normalised = _normalize_entity_alias(entity_name)
        if normalised != entity_name.lower().strip() and normalised:
            fuzzy = self._search_sync(normalised, client, cache, limit // 2)
            for c in fuzzy:
                if c.qid not in seen_qids:
                    seen_qids.add(c.qid)
                    candidates.append(c)

        # Source 3: sub-phrase search for multi-word entities
        words = entity_name.strip().split()
        if len(words) >= 3:
            # Try last two words (common for "New York City" → "York City")
            sub = " ".join(words[-2:])
            sub_results = self._search_sync(sub, client, cache, 3)
            for c in sub_results:
                if c.qid not in seen_qids:
                    seen_qids.add(c.qid)
                    candidates.append(c)

        return candidates

    def _search_sync(
        self,
        query: str,
        client: httpx.Client,
        cache: Any,
        limit: int,
    ) -> List[QIDCandidate]:
        cache_key = f"el_search::{self.language}::{limit}::{_normalize_entity_alias(query)}"
        cached = cache.get(cache_key)
        if cached is not None:
            return [QIDCandidate(**item) for item in cached]

        last_exc: Optional[Exception] = None
        for attempt in range(1, DEFAULT_MAX_RETRIES + 1):
            try:
                response = client.get(
                    self.api_url,
                    params=self._search_params(query, self.language, limit),
                    timeout=self.timeout,
                )
                response.raise_for_status()
                candidates = parse_search_payload(response.json())
                cache.set(cache_key, [asdict(c) for c in candidates], ENTITY_SEARCH_TTL)
                return candidates
            except (httpx.HTTPError, ValueError, TypeError) as exc:
                last_exc = exc
                if attempt >= DEFAULT_MAX_RETRIES or not _should_retry_request(exc):
                    break
                time.sleep(_compute_backoff(attempt))

        raise EntitySearchError(f"Wikidata Search API failed for '{query}': {last_exc}")

    # ----- async -----

    async def recall_async(
        self,
        entity_name: str,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        cache: Any,
        limit: int = 8,
    ) -> List[QIDCandidate]:
        seen_qids: Set[str] = set()
        candidates: List[QIDCandidate] = []

        exact = await self._search_async(entity_name, client, semaphore, cache, limit)
        for c in exact:
            if c.qid not in seen_qids:
                seen_qids.add(c.qid)
                candidates.append(c)

        normalised = _normalize_entity_alias(entity_name)
        if normalised != entity_name.lower().strip() and normalised:
            fuzzy = await self._search_async(normalised, client, semaphore, cache, limit // 2)
            for c in fuzzy:
                if c.qid not in seen_qids:
                    seen_qids.add(c.qid)
                    candidates.append(c)

        words = entity_name.strip().split()
        if len(words) >= 3:
            sub = " ".join(words[-2:])
            sub_results = await self._search_async(sub, client, semaphore, cache, 3)
            for c in sub_results:
                if c.qid not in seen_qids:
                    seen_qids.add(c.qid)
                    candidates.append(c)

        return candidates

    async def _search_async(
        self,
        query: str,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        cache: Any,
        limit: int,
    ) -> List[QIDCandidate]:
        cache_key = f"el_search::{self.language}::{limit}::{_normalize_entity_alias(query)}"
        cached = cache.get(cache_key)
        if cached is not None:
            return [QIDCandidate(**item) for item in cached]

        last_exc: Optional[Exception] = None
        for attempt in range(1, DEFAULT_MAX_RETRIES + 1):
            try:
                async with semaphore:
                    response = await client.get(
                        self.api_url,
                        params=self._search_params(query, self.language, limit),
                        timeout=self.timeout,
                    )
                    response.raise_for_status()
                candidates = parse_search_payload(response.json())
                cache.set(cache_key, [asdict(c) for c in candidates], ENTITY_SEARCH_TTL)
                return candidates
            except (httpx.HTTPError, ValueError, TypeError) as exc:
                last_exc = exc
                if attempt >= DEFAULT_MAX_RETRIES or not _should_retry_request(exc):
                    break
                await asyncio.sleep(_compute_backoff(attempt))

        raise EntitySearchError(f"Wikidata Search API failed for '{query}': {last_exc}")


# =========================================================
# Candidate scorer (embedding-based + structural signals)
# =========================================================


class CandidateScorer:
    """Scores candidates using embeddings when available, otherwise deterministic signals."""

    def __init__(self, embedding_client: Optional[GeminiEmbeddingClient]) -> None:
        self.embedding_client = embedding_client

    # ----- sync -----

    def score_candidates(
        self,
        candidates: List[QIDCandidate],
        entity_name: str,
        question: str,
        intent: Optional[str],
        type_fetcher,
        cache: Any,
    ) -> List[Tuple[float, QIDCandidate]]:
        """Score and rank candidates. Returns sorted (score, candidate) pairs."""
        embedding_bundle = self._prepare_embeddings(candidates, entity_name, question, cache)

        scored: List[Tuple[float, QIDCandidate]] = []
        for candidate in candidates[:8]:
            emb_sim = 0.0
            ctx_sim = 0.0
            if embedding_bundle is not None:
                entity_embedding, context_embedding, candidate_embeddings = embedding_bundle
                candidate_embedding = candidate_embeddings[candidate.qid]
                emb_sim = cosine_similarity(entity_embedding, candidate_embedding)
                ctx_sim = cosine_similarity(context_embedding, candidate_embedding)

            types = type_fetcher(candidate.qid) if type_fetcher else set()
            structural = self._structural_score(
                candidate, entity_name, question, intent, types,
            )

            total = (
                EMBEDDING_SIMILARITY_WEIGHT * emb_sim
                + CONTEXT_SIMILARITY_WEIGHT * ctx_sim
                + structural
            )
            scored.append((total, candidate))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    # ----- async -----

    async def score_candidates_async(
        self,
        candidates: List[QIDCandidate],
        entity_name: str,
        question: str,
        intent: Optional[str],
        type_fetcher,
        cache: Any,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
    ) -> List[Tuple[float, QIDCandidate]]:
        embedding_bundle = await self._prepare_embeddings_async(
            candidates, entity_name, question, cache, client, semaphore,
        )

        scored: List[Tuple[float, QIDCandidate]] = []
        for candidate in candidates[:8]:
            emb_sim = 0.0
            ctx_sim = 0.0
            if embedding_bundle is not None:
                entity_embedding, context_embedding, candidate_embeddings = embedding_bundle
                candidate_embedding = candidate_embeddings[candidate.qid]
                emb_sim = cosine_similarity(entity_embedding, candidate_embedding)
                ctx_sim = cosine_similarity(context_embedding, candidate_embedding)

            types = await type_fetcher(candidate.qid) if type_fetcher else set()
            structural = self._structural_score(
                candidate, entity_name, question, intent, types,
            )

            total = (
                EMBEDDING_SIMILARITY_WEIGHT * emb_sim
                + CONTEXT_SIMILARITY_WEIGHT * ctx_sim
                + structural
            )
            scored.append((total, candidate))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    # ----- embedding with cache -----

    def _get_embedding_cached(self, text: str, cache: Any) -> List[float]:
        cache_key = f"emb::{EMBEDDING_DIMENSIONS}::{_normalize_entity_alias(text)}"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
        if self.embedding_client is None or not self.embedding_client.is_configured:
            raise EntityLinkingError("Gemini embedding client is not configured")
        embedding = self.embedding_client.embed(text)
        cache.set(cache_key, embedding, EMBEDDING_TTL)
        return embedding

    async def _get_embedding_cached_async(
        self,
        text: str,
        cache: Any,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
    ) -> List[float]:
        cache_key = f"emb::{EMBEDDING_DIMENSIONS}::{_normalize_entity_alias(text)}"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
        if self.embedding_client is None or not self.embedding_client.is_configured:
            raise EntityLinkingError("Gemini embedding client is not configured")
        embedding = await self.embedding_client.embed_async(text, client, semaphore)
        cache.set(cache_key, embedding, EMBEDDING_TTL)
        return embedding

    def _prepare_embeddings(
        self,
        candidates: List[QIDCandidate],
        entity_name: str,
        question: str,
        cache: Any,
    ) -> Optional[Tuple[List[float], List[float], Dict[str, List[float]]]]:
        if self.embedding_client is None or not self.embedding_client.is_configured:
            return None

        entity_embedding = self._get_embedding_cached(entity_name, cache)
        context_embedding = self._get_embedding_cached(question, cache)
        candidate_embeddings = {
            candidate.qid: self._get_embedding_cached(
                f"{candidate.label}: {candidate.description}",
                cache,
            )
            for candidate in candidates[:8]
        }
        return entity_embedding, context_embedding, candidate_embeddings

    async def _prepare_embeddings_async(
        self,
        candidates: List[QIDCandidate],
        entity_name: str,
        question: str,
        cache: Any,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
    ) -> Optional[Tuple[List[float], List[float], Dict[str, List[float]]]]:
        if self.embedding_client is None or not self.embedding_client.is_configured:
            return None

        entity_embedding = await self._get_embedding_cached_async(
            entity_name, cache, client, semaphore,
        )
        context_embedding = await self._get_embedding_cached_async(
            question, cache, client, semaphore,
        )
        candidate_embeddings: Dict[str, List[float]] = {}
        for candidate in candidates[:8]:
            candidate_embeddings[candidate.qid] = await self._get_embedding_cached_async(
                f"{candidate.label}: {candidate.description}",
                cache, client, semaphore,
            )
        return entity_embedding, context_embedding, candidate_embeddings

    # ----- structural scoring (deterministic, no I/O) -----

    @staticmethod
    def _structural_score(
        candidate: QIDCandidate,
        entity_name: str,
        question: Optional[str],
        intent: Optional[str],
        types: Set[str],
    ) -> float:
        """Deterministic structural scoring signals."""
        score = 0.0
        target = _normalize_entity_alias(entity_name)
        label = _normalize_entity_alias(candidate.label)
        description = candidate.description.lower()
        aliases = {_normalize_entity_alias(a) for a in candidate.aliases}

        if label == target:
            score += 5.0
        elif target in label or label in target:
            score += 3.0

        if candidate.match_type == "label":
            score += 2.0
        elif candidate.match_type == "alias":
            score += 1.0

        if target in aliases:
            score += 2.0

        if intent and intent in INTENT_CONFIG:
            config = INTENT_CONFIG[intent]
            if config.description_hints:
                score += DESCRIPTION_SIMILARITY_WEIGHT * _semantic_similarity(
                    " ".join(config.description_hints),
                    candidate.description,
                )
            if question:
                score += 2.0 * _semantic_similarity(
                    question,
                    f"{candidate.label} {candidate.description}",
                )
            expected = set(config.expected_entity_types)
            if expected:
                if types & expected:
                    score += TYPE_MATCH_BONUS
                elif types:
                    score -= TYPE_MISMATCH_PENALTY

        if "disambiguation page" in description:
            score -= 10.0
        if any(word in description for word in ("film", "album", "song")) and intent in {"age", "capital", "population"}:
            score -= 4.0

        return score


# =========================================================
# Entity Linker (orchestrator)
# =========================================================


class EntityLinker:
    """Full entity linking pipeline: recall → score → NIL detect.

    Replaces WikidataEntityResolver with embedding-based disambiguation,
    context-aware scoring, and NIL detection.  No hardcoded QID maps.

    Usage::

        linker = EntityLinker(client=httpx_client, cache=persistent_cache)
        result = linker.link("Madonna", intent="age", question="How old is Madonna?")
        if result.is_nil:
            print("Entity not found in Wikidata")
        else:
            print(result.candidate.qid)  # "Q1744"
    """

    def __init__(
        self,
        *,
        client: httpx.Client,
        cache: Any,
        api_url: str = WIKIDATA_API,
        language: str = DEFAULT_LANGUAGE,
    ) -> None:
        self.client = client
        self.cache = cache
        self.recall = CandidateRecall(api_url=api_url, language=language)
        self.embedding_client = GeminiEmbeddingClient() if GEMINI_API_KEY else None
        self.scorer = CandidateScorer(self.embedding_client)

    # ----- type fetcher (shared with old resolver) -----

    def fetch_entity_types(self, qid: str) -> Set[str]:
        from wikidata_qa_llm import execute_sparql, parse_type_qids
        cache_key = f"entity_types::{qid}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return set(cached)
        query = f"SELECT ?type WHERE {{ wd:{qid} wdt:P31/wdt:P279* ?type . }} LIMIT 200"
        rows = execute_sparql(query, cache=self.cache, ttl_seconds=TYPE_TTL)
        types = parse_type_qids(rows)
        self.cache.set(cache_key, list(types), TYPE_TTL)
        return types

    async def fetch_entity_types_async(
        self, qid: str, client: httpx.AsyncClient, semaphore: asyncio.Semaphore,
    ) -> Set[str]:
        from wikidata_qa_llm import execute_sparql_async, parse_type_qids
        cache_key = f"entity_types::{qid}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return set(cached)
        query = f"SELECT ?type WHERE {{ wd:{qid} wdt:P31/wdt:P279* ?type . }} LIMIT 200"
        rows = await execute_sparql_async(query, client, semaphore, cache=self.cache, ttl_seconds=TYPE_TTL)
        types = parse_type_qids(rows)
        self.cache.set(cache_key, list(types), TYPE_TTL)
        return types

    # ----- sync link -----

    def link(
        self,
        entity_name: str,
        *,
        intent: Optional[str] = None,
        question: Optional[str] = None,
    ) -> LinkingResult:
        """Full entity linking: recall → score → NIL detect."""
        resolve_key = f"el_resolve::{intent or 'none'}::{_normalize_entity_alias(entity_name)}"
        cached = self.cache.get(resolve_key)
        if cached is not None:
            return LinkingResult(
                candidate=QIDCandidate(**cached["candidate"]),
                score=cached["score"],
                is_nil=cached["is_nil"],
            )

        # Step 1: Candidate recall
        candidates = self.recall.recall(entity_name, self.client, self.cache)
        if not candidates:
            return LinkingResult(
                candidate=QIDCandidate(qid="NIL", label=entity_name, description="no candidates found"),
                score=0.0,
                is_nil=True,
            )

        # Step 2: Score candidates
        question_text = question or entity_name
        scored = self.scorer.score_candidates(
            candidates, entity_name, question_text, intent,
            self.fetch_entity_types, self.cache,
        )

        best_score, best_candidate = scored[0]

        # Step 3: NIL detection
        is_nil = best_score < NIL_SCORE_THRESHOLD

        result = LinkingResult(
            candidate=best_candidate,
            score=best_score,
            is_nil=is_nil,
            candidate_scores=[
                {"qid": c.qid, "label": c.label, "score": round(s, 3)}
                for s, c in scored[:5]
            ],
        )

        # Cache successful links
        if not is_nil:
            self.cache.set(resolve_key, {
                "candidate": asdict(best_candidate),
                "score": best_score,
                "is_nil": False,
            }, ENTITY_RESOLVE_TTL)

        return result

    # ----- async link -----

    async def link_async(
        self,
        entity_name: str,
        *,
        intent: Optional[str] = None,
        question: Optional[str] = None,
        client: httpx.AsyncClient,
        search_semaphore: asyncio.Semaphore,
        sparql_semaphore: asyncio.Semaphore,
        embedding_semaphore: Optional[asyncio.Semaphore] = None,
    ) -> LinkingResult:
        resolve_key = f"el_resolve::{intent or 'none'}::{_normalize_entity_alias(entity_name)}"
        cached = self.cache.get(resolve_key)
        if cached is not None:
            return LinkingResult(
                candidate=QIDCandidate(**cached["candidate"]),
                score=cached["score"],
                is_nil=cached["is_nil"],
            )

        # Step 1: Candidate recall
        candidates = await self.recall.recall_async(
            entity_name, client, search_semaphore, self.cache,
        )
        if not candidates:
            return LinkingResult(
                candidate=QIDCandidate(qid="NIL", label=entity_name, description="no candidates found"),
                score=0.0,
                is_nil=True,
            )

        # Step 2: Score candidates
        question_text = question or entity_name
        emb_sem = embedding_semaphore or asyncio.Semaphore(5)

        async def _fetch_types(qid: str) -> Set[str]:
            return await self.fetch_entity_types_async(qid, client, sparql_semaphore)

        scored = await self.scorer.score_candidates_async(
            candidates, entity_name, question_text, intent,
            _fetch_types, self.cache, client, emb_sem,
        )

        best_score, best_candidate = scored[0]

        # Step 3: NIL detection
        is_nil = best_score < NIL_SCORE_THRESHOLD

        result = LinkingResult(
            candidate=best_candidate,
            score=best_score,
            is_nil=is_nil,
            candidate_scores=[
                {"qid": c.qid, "label": c.label, "score": round(s, 3)}
                for s, c in scored[:5]
            ],
        )

        if not is_nil:
            self.cache.set(resolve_key, {
                "candidate": asdict(best_candidate),
                "score": best_score,
                "is_nil": False,
            }, ENTITY_RESOLVE_TTL)

        return result
