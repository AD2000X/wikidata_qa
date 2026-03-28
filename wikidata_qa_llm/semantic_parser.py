"""
Semantic parser layer for the Wikidata QA system.

Provides two parser implementations behind a common Protocol:

- RegexSemanticParser  – wraps the existing regex pipeline (zero new deps)
- GeminiSemanticParser – calls Gemini 2.5 Flash-Lite for structured output

Also includes FrameValidator for hard-error / soft-warning checks.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple

import httpx

from config import INTENT_CONFIG, QID_CITY, QID_COUNTRY, QID_HUMAN, QID_OCCUPATION
from gemini_config import (
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_REJECT,
    GEMINI_API_BASE,
    GEMINI_API_KEY,
    GEMINI_MAX_RETRIES,
    GEMINI_MODEL,
    GEMINI_TIMEOUT,
    build_gemini_request_body,
)

# =========================================================
# Exceptions
# =========================================================


class SemanticParserError(Exception):
    """Base error for semantic parsing failures."""


class GeminiAPIError(SemanticParserError):
    """Gemini API returned an error or timed out."""


class FrameValidationError(SemanticParserError):
    """Semantic frame failed hard validation."""


# =========================================================
# Data models
# =========================================================

# Allowed values (kept in sync with gemini_config schema)
ALLOWED_INTENTS: Set[str] = {
    "age", "population", "capital",
    "spouse_birth_place", "birth_country_capital", "spouse_occupation",
    "cities_in_country", "actors_born_in_place", "humans_with_occupation",
    "unsupported",
}

ALLOWED_TYPE_HINTS: Set[str] = {
    "person", "city", "country", "place", "occupation", "unknown",
}

# Maps expected_entity_types QIDs → human-readable type hints
_QID_TO_HINT: Dict[str, str] = {
    QID_HUMAN: "person",
    QID_CITY: "city",
    QID_COUNTRY: "country",
    QID_OCCUPATION: "occupation",
}

# Hard type-mismatch rules: (intent, entity_type_hint) pairs that are
# clearly contradictory.  Anything not listed here is at most a warning.
_HARD_MISMATCH: Set[Tuple[str, str]] = {
    ("age", "city"),
    ("age", "country"),
    ("age", "place"),
    ("age", "occupation"),
    ("population", "person"),
    ("population", "occupation"),
    ("capital", "person"),
    ("capital", "city"),
    ("capital", "occupation"),
    ("spouse_birth_place", "city"),
    ("spouse_birth_place", "country"),
    ("spouse_birth_place", "occupation"),
    ("birth_country_capital", "city"),
    ("birth_country_capital", "country"),
    ("birth_country_capital", "occupation"),
    ("spouse_occupation", "city"),
    ("spouse_occupation", "country"),
    ("spouse_occupation", "occupation"),
    ("cities_in_country", "person"),
    ("cities_in_country", "city"),
    ("cities_in_country", "occupation"),
    ("actors_born_in_place", "person"),
    ("actors_born_in_place", "occupation"),
    ("humans_with_occupation", "person"),
    ("humans_with_occupation", "city"),
    ("humans_with_occupation", "country"),
    ("humans_with_occupation", "place"),
}


@dataclass(frozen=True, slots=True)
class SemanticFrame:
    intent: str
    entity_text: str
    entity_type_hint: str = "unknown"
    confidence: float = 1.0


@dataclass(slots=True)
class ParseResult:
    frame: SemanticFrame
    parser_source: str = "gemini"
    status: str = "ok"
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)


# =========================================================
# SemanticParser protocol
# =========================================================


class SemanticParser(Protocol):
    def parse(self, question: str) -> SemanticFrame: ...


# =========================================================
# Regex-based parser (wraps existing pipeline)
# =========================================================


class RegexSemanticParser:
    """Wraps the existing detect_intent / extract_entity_name pipeline.

    Produces a SemanticFrame with confidence derived from how the
    intent was matched:
      - entity_pattern regex match → 0.9
      - keyword / semantic fallback → 0.7
    """

    def parse(self, question: str) -> SemanticFrame:
        from wikidata_qa_llm import (
            detect_intent_rule,
            detect_intent_semantic,
            extract_entity_name,
        )

        intent = detect_intent_rule(question)
        confidence = 0.9 if intent else 0.0

        if not intent:
            intent = detect_intent_semantic(question)
            confidence = 0.7 if intent else 0.0

        if not intent:
            raise SemanticParserError(f"Regex parser: unsupported question: {question}")

        try:
            entity_text = extract_entity_name(question, intent)
        except Exception as exc:
            raise SemanticParserError(f"Regex parser: entity extraction failed: {exc}") from exc

        entity_type_hint = self._infer_type_hint(intent)

        return SemanticFrame(
            intent=intent,
            entity_text=entity_text,
            entity_type_hint=entity_type_hint,
            confidence=confidence,
        )

    @staticmethod
    def _infer_type_hint(intent: str) -> str:
        """Derive entity_type_hint from INTENT_CONFIG.expected_entity_types."""
        config = INTENT_CONFIG.get(intent)
        if not config or not config.expected_entity_types:
            return "unknown"
        first_qid = config.expected_entity_types[0]
        return _QID_TO_HINT.get(first_qid, "unknown")


# =========================================================
# Gemini-based parser
# =========================================================


def _compute_backoff(attempt: int, base: float = 1.5, jitter_max: float = 0.25) -> float:
    import random
    return (base ** max(0, attempt - 1)) + random.uniform(0.0, jitter_max)


class GeminiSemanticParser:
    """Calls Gemini 2.5 Flash-Lite for structured semantic frame extraction.

    Uses httpx for HTTP (no extra dependencies beyond what wikidata_qa_adv
    already requires).  Constrained decoding is enforced via
    ``responseMimeType`` + ``responseSchema`` at the API level.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = GEMINI_MODEL,
        timeout: int = GEMINI_TIMEOUT,
        max_retries: int = GEMINI_MAX_RETRIES,
    ) -> None:
        self.api_key = api_key or GEMINI_API_KEY
        if not self.api_key:
            raise GeminiAPIError(
                "GEMINI_API_KEY environment variable is not set. "
                "Get a free key at https://aistudio.google.com/"
            )
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self._endpoint = (
            f"{GEMINI_API_BASE}/models/{self.model}:generateContent"
            f"?key={self.api_key}"
        )

    # ----- sync -----

    def parse(self, question: str) -> SemanticFrame:
        body = build_gemini_request_body(question)
        payload = self._call_sync(body)
        return self._extract_frame(payload)

    def _call_sync(self, body: Dict[str, Any]) -> Dict[str, Any]:
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
                    return response.json()
            except Exception as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(_compute_backoff(attempt))
        raise GeminiAPIError(f"Gemini API failed after {self.max_retries} attempts: {last_exc}")

    # ----- async -----

    async def parse_async(
        self,
        question: str,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
    ) -> SemanticFrame:
        body = build_gemini_request_body(question)
        payload = await self._call_async(body, client, semaphore)
        return self._extract_frame(payload)

    async def _call_async(
        self,
        body: Dict[str, Any],
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
    ) -> Dict[str, Any]:
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
                    return response.json()
            except Exception as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                await asyncio.sleep(_compute_backoff(attempt))
        raise GeminiAPIError(f"Gemini async API failed after {self.max_retries} attempts: {last_exc}")

    # ----- response parsing -----

    @staticmethod
    def _extract_frame(payload: Dict[str, Any]) -> SemanticFrame:
        """Extract SemanticFrame from Gemini API response JSON.

        Handles the nested structure:
        payload -> candidates[0] -> content -> parts[0] -> text (JSON string)
        """
        try:
            candidates = payload.get("candidates", [])
            if not candidates:
                raise GeminiAPIError("Gemini response contains no candidates")

            text = candidates[0]["content"]["parts"][0]["text"]
            data = json.loads(text)

            intent = data.get("intent", "unsupported")
            entity_text = data.get("entity_text", "")
            entity_type_hint = data.get("entity_type_hint", "unknown")
            confidence = data.get("confidence", 0.0)

            # Clamp confidence to [0, 1]
            confidence = max(0.0, min(1.0, float(confidence)))

            # Normalise enum values
            if intent not in ALLOWED_INTENTS:
                intent = "unsupported"
                confidence = min(confidence, 0.3)
            if entity_type_hint not in ALLOWED_TYPE_HINTS:
                entity_type_hint = "unknown"

            return SemanticFrame(
                intent=intent,
                entity_text=entity_text.strip(),
                entity_type_hint=entity_type_hint,
                confidence=confidence,
            )
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise GeminiAPIError(f"Failed to parse Gemini response: {exc}") from exc


# =========================================================
# Frame validator
# =========================================================


class FrameValidator:
    """Validates a SemanticFrame, returning hard errors and soft warnings.

    entity_type_hint is treated as a weak signal from the parser.
    Only clearly contradictory (intent, type_hint) pairs are hard errors.
    Everything else is at most a warning — the real type verification
    happens in ResolutionReconciler after entity resolution.
    """

    @staticmethod
    def validate(frame: SemanticFrame) -> Tuple[List[str], List[str]]:
        """Return (hard_errors, soft_warnings)."""
        errors: List[str] = []
        warnings: List[str] = []

        # --- hard errors ---

        if frame.intent not in ALLOWED_INTENTS:
            errors.append("invalid_intent")

        if frame.intent != "unsupported" and not frame.entity_text.strip():
            errors.append("empty_entity")

        if (frame.intent, frame.entity_type_hint) in _HARD_MISMATCH:
            errors.append("hard_type_mismatch")

        # --- soft warnings ---

        if frame.entity_type_hint == "unknown" and frame.intent != "unsupported":
            warnings.append("soft_type_mismatch")

        if frame.confidence < CONFIDENCE_LOW:
            warnings.append("low_confidence")
        elif frame.confidence < CONFIDENCE_HIGH:
            warnings.append("medium_confidence")

        return errors, warnings
