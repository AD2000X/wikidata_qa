from __future__ import annotations

import asyncio
import random
import re
import time
import unicodedata
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import httpx
from diskcache import Cache
from SPARQLWrapper import SPARQLWrapper, JSON as SPARQL_JSON

from config import (
    CACHE_DIR,
    DEFAULT_BACKOFF_BASE,
    DEFAULT_JITTER_MAX,
    DEFAULT_LANGUAGE,
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT,
    DESCRIPTION_SIMILARITY_WEIGHT,
    DISAMBIGUATION_RULES,
    ENTITY_RESOLVE_TTL,
    ENTITY_SEARCH_TTL,
    FALLBACK_QIDS,
    INTENT_CONFIG,
    QID_ACTOR,
    QID_CITY,
    QID_HUMAN,
    SEMANTIC_FALLBACK_THRESHOLD,
    TYPE_MATCH_BONUS,
    TYPE_MISMATCH_PENALTY,
    TYPE_TTL,
    USER_AGENT,
    WIKIDATA_API,
    WIKIDATA_SPARQL,
)

# =========================================================
# Exceptions
# =========================================================


class WikidataError(Exception):
    """Base error for the QA system."""


class IntentNotSupportedError(WikidataError):
    pass


class EntityExtractionError(WikidataError):
    pass


class EntityResolutionError(WikidataError):
    pass


class SPARQLExecutionError(WikidataError):
    pass


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


# =========================================================
# Retry helpers (unified on httpx)
# =========================================================


def compute_backoff_sleep(attempt: int, base: float = DEFAULT_BACKOFF_BASE, jitter_max: float = DEFAULT_JITTER_MAX) -> float:
    return (base ** max(0, attempt - 1)) + random.uniform(0.0, jitter_max)


def _should_retry(exc: Exception) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)):
        return True
    response = getattr(exc, "response", None)
    if response is not None:
        return getattr(response, "status_code", None) in {408, 409, 425, 429, 500, 502, 503, 504}
    return False


def sync_get_with_retry(
    client: httpx.Client,
    url: str,
    *,
    params: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> httpx.Response:
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.get(url, params=params, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response
        except Exception as exc:
            last_exc = exc
            if attempt >= max_retries or not _should_retry(exc):
                break
            time.sleep(compute_backoff_sleep(attempt))
    raise WikidataError(f"HTTP request failed after {max_retries} attempts: {last_exc}")


async def async_get_with_retry(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> httpx.Response:
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            response = await client.get(url, params=params, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response
        except Exception as exc:
            last_exc = exc
            if attempt >= max_retries or not _should_retry(exc):
                break
            await asyncio.sleep(compute_backoff_sleep(attempt))
    raise WikidataError(f"Async HTTP request failed after {max_retries} attempts: {last_exc}")


# =========================================================
# Text helpers
# =========================================================

_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "for", "to", "is", "what", "who", "where",
    "was", "were", "are", "does", "do", "did", "list", "show", "with", "have", "has",
    "country", "birth", "spouse", "occupation", "capital", "population", "age", "old",
    "humans", "people", "city", "cities",
}


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = text.replace("&", " and ")
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^a-z0-9\s'-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_entity_alias(text: str) -> str:
    text = normalize_text(text)
    for prefix in ("the ", "a ", "an "):
        if text.startswith(prefix):
            text = text[len(prefix):]
    for suffix in (" city",):
        if text.endswith(suffix) and len(text) > len(suffix):
            break
    return text


def tokenize_semantic(text: str) -> Set[str]:
    text = normalize_text(text)
    parts = re.findall(r"[a-z0-9]+", text)
    return {p for p in parts if p and p not in _STOPWORDS and len(p) > 1}


def semantic_similarity(text_a: str, text_b: str) -> float:
    a = tokenize_semantic(text_a)
    b = tokenize_semantic(text_b)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# =========================================================
# Caching layer
# =========================================================


class PersistentCache:
    def __init__(self, cache_dir: Path = CACHE_DIR) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache = Cache(str(cache_dir))

    def get(self, key: str) -> Any:
        return self.cache.get(key, default=None)

    def set(self, key: str, value: Any, ttl: int) -> None:
        self.cache.set(key, value, expire=ttl)

    def close(self) -> None:
        self.cache.close()


# =========================================================
# Intent + entity extraction (kept for regex fallback)
# =========================================================


def detect_intent_rule(question: str) -> Optional[str]:
    text = question.strip().lower()
    for intent_name, config in INTENT_CONFIG.items():
        if all(re.search(pattern, text) for pattern in config.keywords):
            return intent_name
    for intent_name, config in INTENT_CONFIG.items():
        if any(re.search(pattern, text) for pattern in config.keywords):
            return intent_name
    return None


def detect_intent_semantic(question: str) -> Optional[str]:
    scored: List[Tuple[float, str]] = []
    q = normalize_text(question)
    for intent_name, config in INTENT_CONFIG.items():
        texts = [config.name.replace("_", " "), *config.examples, *config.description_hints]
        score = max((semantic_similarity(q, t) for t in texts), default=0.0)
        scored.append((score, intent_name))
    scored.sort(reverse=True)
    if scored and scored[0][0] >= SEMANTIC_FALLBACK_THRESHOLD:
        return scored[0][1]
    return None


def detect_intent(question: str) -> str:
    intent = detect_intent_rule(question)
    if intent:
        return intent
    intent = detect_intent_semantic(question)
    if intent:
        return intent
    raise IntentNotSupportedError(f"Unsupported question: {question}")


def extract_entity_name(question: str, intent: str) -> str:
    config = INTENT_CONFIG[intent]
    for pattern in config.entity_patterns:
        match = re.match(pattern, question.strip(), flags=re.IGNORECASE)
        if match:
            entity = match.group("entity").strip(" ?,.!")
            if entity:
                return entity
    raise EntityExtractionError(f"Could not extract entity from question: {question}")


# =========================================================
# Shared parsing (written once, used by sync AND async)
# =========================================================


def parse_search_payload(payload: Dict[str, Any]) -> List[QIDCandidate]:
    """Parse Wikidata Search API response into candidates."""
    candidates: List[QIDCandidate] = []
    for item in payload.get("search", []):
        aliases_raw = item.get("aliases", [])
        aliases = tuple(
            normalize_entity_alias(alias if isinstance(alias, str) else alias.get("value", ""))
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


def parse_sparql_bindings(bindings: Iterable[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Extract flat key-value rows from SPARQL JSON bindings."""
    return [{key: value.get("value", "") for key, value in binding.items()} for binding in bindings]


def parse_type_qids(rows: List[Dict[str, str]]) -> Set[str]:
    """Extract QID suffixes from type-query SPARQL rows."""
    return {row["type"].rsplit("/", 1)[-1] for row in rows if "type" in row}


def score_candidate(
    candidate: QIDCandidate,
    entity_name: str,
    intent: Optional[str],
    question: Optional[str],
    types: Optional[Set[str]] = None,
) -> float:
    """Score a QID candidate for entity disambiguation. Pure logic, no I/O."""
    score = 0.0
    target = normalize_entity_alias(entity_name)
    label = normalize_entity_alias(candidate.label)
    description = candidate.description.lower()
    aliases = {normalize_entity_alias(alias) for alias in candidate.aliases}

    if label == target:
        score += 10.0
    elif target in label or label in target:
        score += 6.0

    if candidate.match_type == "label":
        score += 3.0
    elif candidate.match_type == "alias":
        score += 1.5

    if target in aliases:
        score += 3.0

    if intent and intent in INTENT_CONFIG:
        config = INTENT_CONFIG[intent]
        if config.description_hints:
            score += DESCRIPTION_SIMILARITY_WEIGHT * semantic_similarity(
                " ".join(config.description_hints),
                candidate.description,
            )
        if question:
            score += 2.0 * semantic_similarity(question, f"{candidate.label} {candidate.description}")
        expected = set(config.expected_entity_types)
        if expected:
            types = types or set()
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
# SPARQL execution (sync: SPARQLWrapper, async: httpx)
# =========================================================

_SPARQL_HEADERS = {"Accept": "application/sparql-results+json"}


def _sparql_cache_key(query: str) -> str:
    return f"sparql::{' '.join(query.split())}"


def execute_sparql(
    query: str,
    *,
    cache: PersistentCache,
    ttl_seconds: int,
    endpoint_url: str = WIKIDATA_SPARQL,
) -> List[Dict[str, str]]:
    cache_key = _sparql_cache_key(query)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    sparql = SPARQLWrapper(endpoint_url, agent=USER_AGENT)
    sparql.setQuery(query)
    sparql.setReturnFormat(SPARQL_JSON)
    sparql.setTimeout(DEFAULT_TIMEOUT)

    last_exc: Optional[Exception] = None
    for attempt in range(1, DEFAULT_MAX_RETRIES + 1):
        try:
            result = sparql.query().convert()
            rows = parse_sparql_bindings(result.get("results", {}).get("bindings", []))
            cache.set(cache_key, rows, ttl_seconds)
            return rows
        except Exception as exc:
            last_exc = exc
            if attempt >= DEFAULT_MAX_RETRIES:
                break
            time.sleep(compute_backoff_sleep(attempt))

    raise SPARQLExecutionError(f"SPARQL query failed after {DEFAULT_MAX_RETRIES} attempts: {last_exc}")


async def execute_sparql_async(
    query: str,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    *,
    cache: PersistentCache,
    ttl_seconds: int,
    endpoint_url: str = WIKIDATA_SPARQL,
) -> List[Dict[str, str]]:
    cache_key = _sparql_cache_key(query)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    async with semaphore:
        response = await async_get_with_retry(
            client, endpoint_url,
            params={"query": query, "format": "json"},
            headers=_SPARQL_HEADERS,
        )
    rows = parse_sparql_bindings(response.json().get("results", {}).get("bindings", []))
    cache.set(cache_key, rows, ttl_seconds)
    return rows


# =========================================================
# Entity resolver
# =========================================================


class WikidataEntityResolver:
    def __init__(
        self,
        *,
        client: httpx.Client,
        cache: PersistentCache,
        api_url: str = WIKIDATA_API,
        language: str = DEFAULT_LANGUAGE,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        self.api_url = api_url
        self.language = language
        self.timeout = timeout
        self.client = client
        self.cache = cache
        self.local_qid_map = FALLBACK_QIDS

    # ----- cache key helpers -----

    def _search_cache_key(self, entity_name: str, language: str, limit: int) -> str:
        return f"entity_search::{language}::{limit}::{normalize_entity_alias(entity_name)}"

    def _resolve_cache_key(self, entity_name: str, language: str, intent: Optional[str]) -> str:
        return f"entity_resolve::{language}::{intent or 'none'}::{normalize_entity_alias(entity_name)}"

    @staticmethod
    def _type_cache_key(qid: str) -> str:
        return f"entity_types::{qid}"

    # ----- shared helpers (no I/O) -----

    def _from_local_map(self, entity_name: str, language: Optional[str] = None) -> Optional[QIDCandidate]:
        lang = language or self.language
        mapping = self.local_qid_map.get(lang, {})
        qid = mapping.get(normalize_entity_alias(entity_name))
        if qid:
            return QIDCandidate(qid=qid, label=entity_name, description="local fallback")
        return None

    def _search_params(self, entity_name: str, language: str, limit: int) -> Dict[str, Any]:
        return {
            "action": "wbsearchentities",
            "search": entity_name,
            "language": language,
            "format": "json",
            "limit": limit,
        }

    @staticmethod
    def _type_query(qid: str) -> str:
        return f"SELECT ?type WHERE {{ wd:{qid} wdt:P31/wdt:P279* ?type . }} LIMIT 200"

    def _resolve_from_overrides(self, entity_name: str) -> Optional[QIDCandidate]:
        normalized = normalize_entity_alias(entity_name)
        if normalized in DISAMBIGUATION_RULES:
            return QIDCandidate(qid=DISAMBIGUATION_RULES[normalized], label=entity_name, description="manual override")
        return self._from_local_map(entity_name)

    def _rank_candidates(
        self,
        candidates: List[QIDCandidate],
        entity_name: str,
        intent: Optional[str],
        question: Optional[str],
        type_fetcher,
    ) -> QIDCandidate:
        ranked: List[Tuple[float, QIDCandidate]] = []
        for candidate in candidates[:5]:
            types = type_fetcher(candidate.qid)
            s = score_candidate(candidate, entity_name, intent, question, types)
            ranked.append((s, candidate))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return ranked[0][1]

    async def _rank_candidates_async(
        self,
        candidates: List[QIDCandidate],
        entity_name: str,
        intent: Optional[str],
        question: Optional[str],
        type_fetcher,
    ) -> QIDCandidate:
        ranked: List[Tuple[float, QIDCandidate]] = []
        for candidate in candidates[:5]:
            types = await type_fetcher(candidate.qid)
            s = score_candidate(candidate, entity_name, intent, question, types)
            ranked.append((s, candidate))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return ranked[0][1]

    # ----- sync I/O -----

    def search(self, entity_name: str, language: Optional[str] = None, limit: int = 5) -> List[QIDCandidate]:
        language = language or self.language
        cache_key = self._search_cache_key(entity_name, language, limit)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return [QIDCandidate(**item) for item in cached]

        try:
            response = sync_get_with_retry(
                self.client, self.api_url,
                params=self._search_params(entity_name, language, limit),
                timeout=self.timeout,
            )
            candidates = parse_search_payload(response.json())
        except WikidataError:
            local = self._from_local_map(entity_name, language)
            if local:
                print(f"[WARN] Search API failed for '{entity_name}', using local fallback")
                return [local]
            raise EntityResolutionError(f"Wikidata Search API failed for '{entity_name}'")

        self.cache.set(cache_key, [asdict(c) for c in candidates], ENTITY_SEARCH_TTL)
        return candidates

    def fetch_entity_types(self, qid: str) -> Set[str]:
        cache_key = self._type_cache_key(qid)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return set(cached)

        rows = execute_sparql(self._type_query(qid), cache=self.cache, ttl_seconds=TYPE_TTL)
        types = parse_type_qids(rows)
        self.cache.set(cache_key, list(types), TYPE_TTL)
        return types

    def resolve(self, entity_name: str, intent: Optional[str] = None, question: Optional[str] = None) -> QIDCandidate:
        cache_key = self._resolve_cache_key(entity_name, self.language, intent)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return QIDCandidate(**cached)

        override = self._resolve_from_overrides(entity_name)
        if override:
            self.cache.set(cache_key, asdict(override), ENTITY_RESOLVE_TTL)
            return override

        candidates = self.search(entity_name, self.language, limit=8)
        if not candidates:
            raise EntityResolutionError(f"No Wikidata candidates found for '{entity_name}'")

        best = self._rank_candidates(candidates, entity_name, intent, question, self.fetch_entity_types)
        self.cache.set(cache_key, asdict(best), ENTITY_RESOLVE_TTL)
        return best

    # ----- async I/O (same logic, different transport) -----

    async def search_async(
        self,
        entity_name: str,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        language: Optional[str] = None,
        limit: int = 5,
    ) -> List[QIDCandidate]:
        language = language or self.language
        cache_key = self._search_cache_key(entity_name, language, limit)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return [QIDCandidate(**item) for item in cached]

        try:
            async with semaphore:
                response = await async_get_with_retry(
                    client, self.api_url,
                    params=self._search_params(entity_name, language, limit),
                    timeout=self.timeout,
                )
            candidates = parse_search_payload(response.json())
        except WikidataError:
            local = self._from_local_map(entity_name, language)
            if local:
                print(f"[WARN] Async Search API failed for '{entity_name}', using local fallback")
                return [local]
            raise EntityResolutionError(f"Async Wikidata Search API failed for '{entity_name}'")

        self.cache.set(cache_key, [asdict(c) for c in candidates], ENTITY_SEARCH_TTL)
        return candidates

    async def fetch_entity_types_async(self, qid: str, client: httpx.AsyncClient, semaphore: asyncio.Semaphore) -> Set[str]:
        cache_key = self._type_cache_key(qid)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return set(cached)

        rows = await execute_sparql_async(self._type_query(qid), client, semaphore, cache=self.cache, ttl_seconds=TYPE_TTL)
        types = parse_type_qids(rows)
        self.cache.set(cache_key, list(types), TYPE_TTL)
        return types

    async def resolve_async(
        self,
        entity_name: str,
        intent: Optional[str],
        question: Optional[str],
        client: httpx.AsyncClient,
        search_semaphore: asyncio.Semaphore,
        sparql_semaphore: asyncio.Semaphore,
    ) -> QIDCandidate:
        cache_key = self._resolve_cache_key(entity_name, self.language, intent)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return QIDCandidate(**cached)

        override = self._resolve_from_overrides(entity_name)
        if override:
            self.cache.set(cache_key, asdict(override), ENTITY_RESOLVE_TTL)
            return override

        candidates = await self.search_async(entity_name, client, search_semaphore, self.language, limit=8)
        if not candidates:
            raise EntityResolutionError(f"No Wikidata candidates found for '{entity_name}'")

        async def _fetch_types(qid: str) -> Set[str]:
            return await self.fetch_entity_types_async(qid, client, sparql_semaphore)

        best = await self._rank_candidates_async(candidates, entity_name, intent, question, _fetch_types)
        self.cache.set(cache_key, asdict(best), ENTITY_RESOLVE_TTL)
        return best


# =========================================================
# SPARQL query builder
# =========================================================


def build_sparql(qid: str, intent: str) -> str:
    if intent == "age":
        return f"""
        SELECT ?birth WHERE {{
          wd:{qid} wdt:P569 ?birth .
        }}
        LIMIT 1
        """.strip()

    if intent == "population":
        return f"""
        SELECT ?population ?pointInTime WHERE {{
          wd:{qid} p:P1082 ?statement .
          ?statement ps:P1082 ?population .
          OPTIONAL {{ ?statement pq:P585 ?pointInTime . }}
          OPTIONAL {{ ?statement wikibase:rank ?rank . }}
          FILTER(?rank != wikibase:DeprecatedRank || !BOUND(?rank))
        }}
        ORDER BY DESC(?pointInTime) DESC(?population)
        LIMIT 1
        """.strip()

    if intent == "capital":
        return f"""
        SELECT ?capitalLabel WHERE {{
          wd:{qid} wdt:P36 ?capital .
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{DEFAULT_LANGUAGE}". }}
        }}
        LIMIT 10
        """.strip()

    if intent == "spouse_birth_place":
        return f"""
        SELECT ?placeLabel WHERE {{
          wd:{qid} wdt:P26 ?spouse .
          ?spouse wdt:P19 ?place .
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{DEFAULT_LANGUAGE}". }}
        }}
        LIMIT 10
        """.strip()

    if intent == "birth_country_capital":
        return f"""
        SELECT ?capitalLabel WHERE {{
          wd:{qid} wdt:P19 ?birthPlace .
          ?birthPlace wdt:P17 ?country .
          ?country wdt:P36 ?capital .
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{DEFAULT_LANGUAGE}". }}
        }}
        LIMIT 10
        """.strip()

    if intent == "spouse_occupation":
        return f"""
        SELECT ?occupationLabel WHERE {{
          wd:{qid} wdt:P26 ?spouse .
          ?spouse wdt:P106 ?occupation .
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{DEFAULT_LANGUAGE}". }}
        }}
        LIMIT 10
        """.strip()

    if intent == "cities_in_country":
        return f"""
        SELECT ?cityLabel WHERE {{
          ?city wdt:P31/wdt:P279* wd:{QID_CITY} .
          ?city wdt:P17 wd:{qid} .
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{DEFAULT_LANGUAGE}". }}
        }}
        LIMIT 20
        """.strip()

    if intent == "actors_born_in_place":
        return f"""
        SELECT ?personLabel WHERE {{
          ?person wdt:P31 wd:{QID_HUMAN} .
          ?person wdt:P106 wd:{QID_ACTOR} .
          ?person wdt:P19 wd:{qid} .
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{DEFAULT_LANGUAGE}". }}
        }}
        LIMIT 20
        """.strip()

    if intent == "humans_with_occupation":
        return f"""
        SELECT ?personLabel WHERE {{
          ?person wdt:P31 wd:{QID_HUMAN} .
          ?person wdt:P106 wd:{qid} .
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{DEFAULT_LANGUAGE}". }}
        }}
        LIMIT 20
        """.strip()

    raise IntentNotSupportedError(f"Unsupported intent: {intent}")


# =========================================================
# Formatting
# =========================================================


def compute_age_from_birthdate(birth_value: str, today: Optional[date] = None) -> int:
    if today is None:
        today = date.today()
    dt = datetime.fromisoformat(birth_value.replace("Z", "+00:00"))
    birth = dt.date()
    age = today.year - birth.year
    if (today.month, today.day) < (birth.month, birth.day):
        age -= 1
    return age


def normalize_population(value: str) -> str:
    try:
        return str(int(float(value)))
    except ValueError:
        return value


def unique_join(rows: Sequence[Dict[str, str]], key: str) -> str:
    seen: List[str] = []
    for row in rows:
        value = row.get(key, "").strip()
        if value and value not in seen:
            seen.append(value)
    return ", ".join(seen) if seen else "Unknown"


def format_answer(intent: str, rows: Sequence[Dict[str, str]]) -> str:
    row = rows[0]

    if intent == "age":
        birth = row.get("birth")
        if not birth:
            raise SPARQLExecutionError("Age query returned no birth date")
        return str(compute_age_from_birthdate(birth))

    if intent == "population":
        population = row.get("population")
        if not population:
            raise SPARQLExecutionError("Population query returned no population value")
        return normalize_population(population)

    _LABEL_KEY = {
        "capital": "capitalLabel",
        "birth_country_capital": "capitalLabel",
        "spouse_birth_place": "placeLabel",
        "spouse_occupation": "occupationLabel",
        "cities_in_country": "cityLabel",
        "actors_born_in_place": "personLabel",
        "humans_with_occupation": "personLabel",
    }
    if intent in _LABEL_KEY:
        return unique_join(rows, _LABEL_KEY[intent])

    raise IntentNotSupportedError(f"Unsupported intent: {intent}")


# =========================================================
# Semantic parsing orchestration
#
# Integrates GeminiSemanticParser, RegexSemanticParser,
# FrameValidator, and confidence routing into a single
# function that returns a ParseResult.
# =========================================================


def _parse_with_routing(question: str) -> "ParseResult":
    """Run the full semantic parsing pipeline with confidence routing.

    Flow:
      1. Try Gemini parser
      2. Validate frame (hard errors → regex fallback)
      3. Confidence routing (medium/low → try regex, accept/reject)
      4. Handle unsupported intent
    """
    from semantic_parser import (
        FrameValidator,
        GeminiSemanticParser,
        GeminiAPIError,
        ParseResult,
        RegexSemanticParser,
        SemanticFrame,
        SemanticParserError,
    )
    from gemini_config import CONFIDENCE_HIGH, CONFIDENCE_LOW, CONFIDENCE_REJECT

    regex_parser = RegexSemanticParser()

    # --- Step 1: try Gemini ---
    gemini_frame: Optional[SemanticFrame] = None
    parser_source = "gemini"
    status = "ok"

    try:
        gemini_parser = GeminiSemanticParser()
        gemini_frame = gemini_parser.parse(question)
    except (GeminiAPIError, SemanticParserError) as exc:
        # Technical failure → immediate regex fallback
        print(f"[WARN] Gemini parser failed: {exc}")
        status = "parser_error"
        try:
            frame = regex_parser.parse(question)
            return ParseResult(
                frame=frame,
                parser_source="regex_fallback",
                status="ok",
            )
        except SemanticParserError:
            raise IntentNotSupportedError(f"Both parsers failed for: {question}")

    # --- Step 2: validate frame ---
    hard_errors, soft_warnings = FrameValidator.validate(gemini_frame)

    if hard_errors:
        try:
            frame = regex_parser.parse(question)
            return ParseResult(
                frame=frame,
                parser_source="gemini_with_regex_fallback",
                status="ok",
                validation_errors=hard_errors,
                validation_warnings=soft_warnings,
            )
        except SemanticParserError:
            raise IntentNotSupportedError(
                f"Gemini frame failed validation ({hard_errors}) "
                f"and regex fallback also failed for: {question}"
            )

    # --- Step 3: confidence routing ---
    confidence = gemini_frame.confidence

    if confidence < CONFIDENCE_REJECT and gemini_frame.intent == "unsupported":
        # Very low confidence + unsupported → refuse
        return ParseResult(
            frame=gemini_frame,
            parser_source="gemini",
            status="unsupported",
            validation_warnings=soft_warnings,
        )

    if confidence < CONFIDENCE_LOW:
        # Low confidence → prefer regex
        try:
            frame = regex_parser.parse(question)
            return ParseResult(
                frame=frame,
                parser_source="gemini_with_regex_fallback",
                status="ok",
                validation_warnings=soft_warnings,
            )
        except SemanticParserError:
            # Regex also failed → accept Gemini but mark ambiguous
            return ParseResult(
                frame=gemini_frame,
                parser_source="gemini",
                status="ambiguous",
                validation_warnings=soft_warnings,
            )

    if confidence < CONFIDENCE_HIGH:
        # Medium confidence → try regex, fall back to Gemini
        try:
            frame = regex_parser.parse(question)
            return ParseResult(
                frame=frame,
                parser_source="gemini_with_regex_fallback",
                status="ok",
                validation_warnings=soft_warnings,
            )
        except SemanticParserError:
            return ParseResult(
                frame=gemini_frame,
                parser_source="gemini",
                status="ok",
                validation_warnings=soft_warnings,
            )

    # High confidence → use Gemini directly
    return ParseResult(
        frame=gemini_frame,
        parser_source="gemini",
        status="ok",
        validation_warnings=soft_warnings,
    )


async def _parse_with_routing_async(
    question: str,
    client: httpx.AsyncClient,
    gemini_semaphore: asyncio.Semaphore,
) -> "ParseResult":
    """Async version of _parse_with_routing."""
    from semantic_parser import (
        FrameValidator,
        GeminiSemanticParser,
        GeminiAPIError,
        ParseResult,
        RegexSemanticParser,
        SemanticFrame,
        SemanticParserError,
    )
    from gemini_config import CONFIDENCE_HIGH, CONFIDENCE_LOW, CONFIDENCE_REJECT

    regex_parser = RegexSemanticParser()

    gemini_frame: Optional[SemanticFrame] = None
    parser_source = "gemini"
    status = "ok"

    try:
        gemini_parser = GeminiSemanticParser()
        gemini_frame = await gemini_parser.parse_async(question, client, gemini_semaphore)
    except (GeminiAPIError, SemanticParserError) as exc:
        print(f"[WARN] Gemini async parser failed: {exc}")
        status = "parser_error"
        try:
            frame = regex_parser.parse(question)
            return ParseResult(
                frame=frame,
                parser_source="regex_fallback",
                status="ok",
            )
        except SemanticParserError:
            raise IntentNotSupportedError(f"Both parsers failed for: {question}")

    hard_errors, soft_warnings = FrameValidator.validate(gemini_frame)

    if hard_errors:
        try:
            frame = regex_parser.parse(question)
            return ParseResult(
                frame=frame,
                parser_source="gemini_with_regex_fallback",
                status="ok",
                validation_errors=hard_errors,
                validation_warnings=soft_warnings,
            )
        except SemanticParserError:
            raise IntentNotSupportedError(
                f"Gemini frame failed validation ({hard_errors}) "
                f"and regex fallback also failed for: {question}"
            )

    confidence = gemini_frame.confidence

    if confidence < CONFIDENCE_REJECT and gemini_frame.intent == "unsupported":
        return ParseResult(
            frame=gemini_frame,
            parser_source="gemini",
            status="unsupported",
            validation_warnings=soft_warnings,
        )

    if confidence < CONFIDENCE_LOW:
        try:
            frame = regex_parser.parse(question)
            return ParseResult(
                frame=frame,
                parser_source="gemini_with_regex_fallback",
                status="ok",
                validation_warnings=soft_warnings,
            )
        except SemanticParserError:
            return ParseResult(
                frame=gemini_frame,
                parser_source="gemini",
                status="ambiguous",
                validation_warnings=soft_warnings,
            )

    if confidence < CONFIDENCE_HIGH:
        try:
            frame = regex_parser.parse(question)
            return ParseResult(
                frame=frame,
                parser_source="gemini_with_regex_fallback",
                status="ok",
                validation_warnings=soft_warnings,
            )
        except SemanticParserError:
            return ParseResult(
                frame=gemini_frame,
                parser_source="gemini",
                status="ok",
                validation_warnings=soft_warnings,
            )

    return ParseResult(
        frame=gemini_frame,
        parser_source="gemini",
        status="ok",
        validation_warnings=soft_warnings,
    )


# =========================================================
# High-level QA
# =========================================================


class WikidataQA:
    """Orchestrator: owns the shared httpx.Client, cache, and resolver.

    The ``ask`` method now uses a two-stage semantic parser:
    1. GeminiSemanticParser (LLM-based, structured output)
    2. RegexSemanticParser  (deterministic fallback)

    All downstream modules (entity resolver, SPARQL builder,
    formatter) remain unchanged.
    """

    def __init__(
        self,
        *,
        cache: Optional[PersistentCache] = None,
        client: Optional[httpx.Client] = None,
        resolver: Optional[WikidataEntityResolver] = None,
        use_llm_parser: bool = True,
    ) -> None:
        self.cache = cache or PersistentCache()
        self._owns_client = client is None
        self.client = client or httpx.Client(headers={"User-Agent": USER_AGENT})
        self.resolver = resolver or WikidataEntityResolver(client=self.client, cache=self.cache)
        self.use_llm_parser = use_llm_parser

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def ask(self, question: str, endpoint: str = WIKIDATA_SPARQL) -> Dict[str, Any]:
        from semantic_parser import ParseResult, RegexSemanticParser, SemanticParserError
        from resolution_reconciler import ResolutionReconciler
        from execution_verifier import ExecutionVerifier

        # --- Step 1-3: Semantic parsing with routing ---
        if self.use_llm_parser:
            parse_result = _parse_with_routing(question)
        else:
            # Legacy mode: regex only
            regex_parser = RegexSemanticParser()
            try:
                frame = regex_parser.parse(question)
                parse_result = ParseResult(frame=frame, parser_source="regex_fallback", status="ok")
            except SemanticParserError:
                raise IntentNotSupportedError(f"Unsupported question: {question}")

        frame = parse_result.frame

        # --- Step 4: Unsupported check ---
        if parse_result.status == "unsupported" or frame.intent == "unsupported":
            return {
                "question": question,
                "intent": "unsupported",
                "entity_name": frame.entity_text,
                "answer": None,
                "semantic_frame": asdict(frame),
                "parser_source": parse_result.parser_source,
                "status": "unsupported",
                "validation_errors": parse_result.validation_errors,
                "validation_warnings": parse_result.validation_warnings,
            }

        # --- Step 5: Entity resolution (unchanged) ---
        candidate = self.resolver.resolve(
            frame.entity_text,
            intent=frame.intent,
            question=question,
        )

        # --- Step 6: Resolution reconciliation ---
        resolved_types = self.resolver.fetch_entity_types(candidate.qid)
        reconciliation = ResolutionReconciler.reconcile(
            parser_type_hint=frame.entity_type_hint,
            resolved_types=resolved_types,
            intent_expected_types=set(INTENT_CONFIG[frame.intent].expected_entity_types),
        )

        # --- Step 7: SPARQL compilation (unchanged) ---
        sparql = build_sparql(candidate.qid, frame.intent)
        ttl = INTENT_CONFIG[frame.intent].ttl_seconds

        # --- Step 8: SPARQL execution (unchanged) ---
        rows = execute_sparql(sparql, cache=self.cache, ttl_seconds=ttl, endpoint_url=endpoint)

        # --- Step 9: Execution verification ---
        verification = ExecutionVerifier.verify(frame.intent, rows)
        if verification.hard_errors:
            raise SPARQLExecutionError(
                f"Execution verification failed for '{question}': "
                f"{verification.hard_errors}"
            )

        # --- Step 10: Answer formatting (unchanged) ---
        answer = format_answer(frame.intent, rows)

        # --- Step 11: Return with full audit trail ---
        return {
            "question": question,
            "intent": frame.intent,
            "entity_name": frame.entity_text,
            "resolved_qid": candidate.qid,
            "resolved_label": candidate.label,
            "resolved_description": candidate.description,
            "sparql": sparql,
            "answer": answer,
            "raw_rows": rows,
            # audit fields
            "semantic_frame": asdict(frame),
            "parser_source": parse_result.parser_source,
            "status": parse_result.status,
            "validation_errors": parse_result.validation_errors,
            "validation_warnings": parse_result.validation_warnings,
            "reconciliation": asdict(reconciliation),
            "verification": {
                "is_valid": verification.is_valid,
                "hard_errors": verification.hard_errors,
                "warnings": verification.warnings,
            },
        }

    async def ask_async(
        self,
        question: str,
        client: httpx.AsyncClient,
        search_semaphore: asyncio.Semaphore,
        sparql_semaphore: asyncio.Semaphore,
        gemini_semaphore: Optional[asyncio.Semaphore] = None,
    ) -> Dict[str, Any]:
        from semantic_parser import ParseResult, RegexSemanticParser, SemanticParserError
        from resolution_reconciler import ResolutionReconciler
        from execution_verifier import ExecutionVerifier

        # --- Step 1-3: Semantic parsing with routing ---
        if self.use_llm_parser and gemini_semaphore is not None:
            parse_result = await _parse_with_routing_async(question, client, gemini_semaphore)
        else:
            regex_parser = RegexSemanticParser()
            try:
                frame = regex_parser.parse(question)
                parse_result = ParseResult(frame=frame, parser_source="regex_fallback", status="ok")
            except SemanticParserError:
                raise IntentNotSupportedError(f"Unsupported question: {question}")

        frame = parse_result.frame

        # --- Step 4: Unsupported check ---
        if parse_result.status == "unsupported" or frame.intent == "unsupported":
            return {
                "question": question,
                "intent": "unsupported",
                "entity_name": frame.entity_text,
                "answer": None,
                "semantic_frame": asdict(frame),
                "parser_source": parse_result.parser_source,
                "status": "unsupported",
                "validation_errors": parse_result.validation_errors,
                "validation_warnings": parse_result.validation_warnings,
            }

        # --- Step 5: Entity resolution (unchanged) ---
        candidate = await self.resolver.resolve_async(
            frame.entity_text,
            frame.intent,
            question,
            client,
            search_semaphore,
            sparql_semaphore,
        )

        # --- Step 6: Resolution reconciliation ---
        resolved_types = await self.resolver.fetch_entity_types_async(
            candidate.qid, client, sparql_semaphore,
        )
        reconciliation = ResolutionReconciler.reconcile(
            parser_type_hint=frame.entity_type_hint,
            resolved_types=resolved_types,
            intent_expected_types=set(INTENT_CONFIG[frame.intent].expected_entity_types),
        )

        # --- Step 7: SPARQL compilation (unchanged) ---
        sparql = build_sparql(candidate.qid, frame.intent)
        ttl = INTENT_CONFIG[frame.intent].ttl_seconds

        # --- Step 8: SPARQL execution (unchanged) ---
        rows = await execute_sparql_async(
            sparql, client, sparql_semaphore,
            cache=self.cache, ttl_seconds=ttl,
        )

        # --- Step 9: Execution verification ---
        verification = ExecutionVerifier.verify(frame.intent, rows)
        if verification.hard_errors:
            raise SPARQLExecutionError(
                f"Execution verification failed for '{question}': "
                f"{verification.hard_errors}"
            )

        # --- Step 10: Answer formatting (unchanged) ---
        answer = format_answer(frame.intent, rows)

        # --- Step 11: Return with full audit trail ---
        return {
            "question": question,
            "intent": frame.intent,
            "entity_name": frame.entity_text,
            "resolved_qid": candidate.qid,
            "resolved_label": candidate.label,
            "resolved_description": candidate.description,
            "sparql": sparql,
            "answer": answer,
            "raw_rows": rows,
            "semantic_frame": asdict(frame),
            "parser_source": parse_result.parser_source,
            "status": parse_result.status,
            "validation_errors": parse_result.validation_errors,
            "validation_warnings": parse_result.validation_warnings,
            "reconciliation": asdict(reconciliation),
            "verification": {
                "is_valid": verification.is_valid,
                "hard_errors": verification.hard_errors,
                "warnings": verification.warnings,
            },
        }


async def ask_batch(
    questions: List[str],
    *,
    search_concurrency: int = 2,
    sparql_concurrency: int = 1,
    gemini_concurrency: int = 5,
    use_llm_parser: bool = True,
) -> List[Dict[str, Any]]:
    qa = WikidataQA(use_llm_parser=use_llm_parser)
    search_semaphore = asyncio.Semaphore(search_concurrency)
    sparql_semaphore = asyncio.Semaphore(sparql_concurrency)
    gemini_semaphore = asyncio.Semaphore(gemini_concurrency)
    limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)

    async with httpx.AsyncClient(limits=limits, headers={"User-Agent": USER_AGENT}) as client:
        tasks = [
            qa.ask_async(q, client, search_semaphore, sparql_semaphore, gemini_semaphore)
            for q in questions
        ]
        return await asyncio.gather(*tasks)


def ask(question: str, endpoint: str = WIKIDATA_SPARQL, use_llm_parser: bool = True) -> str:
    with WikidataQA(use_llm_parser=use_llm_parser) as qa:
        result = qa.ask(question, endpoint=endpoint)
        if result.get("answer") is None:
            raise IntentNotSupportedError(f"Unsupported question: {question}")
        return result["answer"]


__all__ = [
    "ask",
    "ask_batch",
    "WikidataQA",
    "WikidataError",
    "IntentNotSupportedError",
    "EntityExtractionError",
    "EntityResolutionError",
    "SPARQLExecutionError",
]
