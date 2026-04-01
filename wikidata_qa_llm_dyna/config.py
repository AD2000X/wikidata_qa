"""
Configuration for the Wikidata QA system.

All constants and intent definitions live here.
Hardcoded QID mappings (DISAMBIGUATION_RULES, FALLBACK_QIDS) have been
removed — entity resolution is now fully dynamic via the EntityLinker.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

# =========================================================
# Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / ".wikidata_cache"

# =========================================================
# API endpoints & HTTP settings
# =========================================================

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
USER_AGENT = f"Wikidata-Ask-Bot/8.0 ({sys.platform}) Python/{sys.version_info[0]}.{sys.version_info[1]}"
DEFAULT_TIMEOUT = 10
DEFAULT_LANGUAGE = "en"
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_BASE = 1.5
DEFAULT_JITTER_MAX = 0.25

# =========================================================
# Cache TTLs (seconds)
# =========================================================

ENTITY_SEARCH_TTL = 60 * 60 * 24 * 30
ENTITY_RESOLVE_TTL = 60 * 60 * 24 * 30
SPARQL_TTL_STABLE = 60 * 60 * 24 * 30
SPARQL_TTL_DYNAMIC = 60 * 60 * 24
TYPE_TTL = 60 * 60 * 24 * 30

# =========================================================
# Scoring thresholds & weights (structural signals)
# =========================================================

SEMANTIC_FALLBACK_THRESHOLD = 0.22
DESCRIPTION_SIMILARITY_WEIGHT = 2.5
TYPE_MATCH_BONUS = 6.0
TYPE_MISMATCH_PENALTY = 8.0

# =========================================================
# Entity linker settings
# =========================================================

EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 768
EMBEDDING_TTL = 60 * 60 * 24 * 30
EMBEDDING_SIMILARITY_WEIGHT = 8.0
CONTEXT_SIMILARITY_WEIGHT = 5.0
NIL_SCORE_THRESHOLD = 3.0

# =========================================================
# Wikidata type QIDs
# =========================================================

QID_HUMAN = "Q5"
QID_CITY = "Q515"
QID_COUNTRY = "Q6256"
QID_OCCUPATION = "Q28640"
QID_ACTOR = "Q33999"

# =========================================================
# Intent configuration
# =========================================================


@dataclass(frozen=True, slots=True)
class IntentConfig:
    name: str
    keywords: Tuple[str, ...]
    entity_patterns: Tuple[str, ...]
    property_id: str
    answer_kind: str
    ttl_seconds: int = SPARQL_TTL_STABLE
    examples: Tuple[str, ...] = ()
    expected_entity_types: Tuple[str, ...] = ()
    description_hints: Tuple[str, ...] = ()


INTENT_CONFIG: Dict[str, IntentConfig] = {
    "age": IntentConfig(
        name="age",
        keywords=(r"\b(age|old)\b",),
        entity_patterns=(
            r"^\s*how\s+old\s+is\s+(?P<entity>.+?)\s*\??\s*$",
            r"^\s*what\s+age\s+is\s+(?P<entity>.+?)\s*\??\s*$",
        ),
        property_id="P569",
        answer_kind="age",
        ttl_seconds=SPARQL_TTL_DYNAMIC,
        examples=("how old is tom cruise", "what age is madonna"),
        expected_entity_types=(QID_HUMAN,),
        description_hints=("person", "human", "actor", "singer", "politician"),
    ),
    "population": IntentConfig(
        name="population",
        keywords=(r"\bpopulation\b",),
        entity_patterns=(
            r"^\s*what\s+is\s+the\s+population\s+of\s+(?P<entity>.+?)\s*\??\s*$",
            r"^\s*population\s+of\s+(?P<entity>.+?)\s*\??\s*$",
        ),
        property_id="P1082",
        answer_kind="population",
        ttl_seconds=SPARQL_TTL_DYNAMIC,
        examples=("what is the population of london", "population of new york"),
        expected_entity_types=(QID_CITY, QID_COUNTRY),
        description_hints=("city", "country", "capital", "town", "state"),
    ),
    "capital": IntentConfig(
        name="capital",
        keywords=(r"\bcapital\b",),
        entity_patterns=(
            r"^\s*what\s+is\s+the\s+capital\s+of\s+(?P<entity>.+?)\s*\??\s*$",
        ),
        property_id="P36",
        answer_kind="label_list",
        examples=("what is the capital of japan", "what is the capital of france"),
        expected_entity_types=(QID_COUNTRY,),
        description_hints=("country", "state", "region"),
    ),
    "spouse_birth_place": IntentConfig(
        name="spouse_birth_place",
        keywords=(r"\bspouse\b", r"\bborn\b"),
        entity_patterns=(
            r"^\s*where\s+was\s+(?P<entity>.+?)'s\s+spouse\s+born\s*\??\s*$",
            r"^\s*what\s+is\s+the\s+birthplace\s+of\s+(?P<entity>.+?)'s\s+spouse\s*\??\s*$",
        ),
        property_id="P26->P19",
        answer_kind="label_list",
        examples=("where was tom hanks's spouse born", "where was barack obama's spouse born"),
        expected_entity_types=(QID_HUMAN,),
        description_hints=("person", "human", "actor", "politician"),
    ),
    "birth_country_capital": IntentConfig(
        name="birth_country_capital",
        keywords=(r"\bcapital\b", r"\bborn\b"),
        entity_patterns=(
            r"^\s*what\s+is\s+the\s+capital\s+of\s+(?P<entity>.+?)'s\s+birth\s+country\s*\??\s*$",
            r"^\s*what\s+is\s+the\s+capital\s+of\s+the\s+country\s+where\s+(?P<entity>.+?)\s+was\s+born\s*\??\s*$",
        ),
        property_id="P19->P17->P36",
        answer_kind="label_list",
        examples=("what is the capital of tom cruise's birth country", "what is the capital of the country where taylor swift was born"),
        expected_entity_types=(QID_HUMAN,),
        description_hints=("person", "human", "actor", "singer"),
    ),
    "spouse_occupation": IntentConfig(
        name="spouse_occupation",
        keywords=(r"\bspouse\b", r"\boccupation\b"),
        entity_patterns=(
            r"^\s*what\s+occupation\s+does\s+(?P<entity>.+?)'s\s+spouse\s+have\s*\??\s*$",
            r"^\s*what\s+is\s+(?P<entity>.+?)'s\s+spouse's\s+occupation\s*\??\s*$",
        ),
        property_id="P26->P106",
        answer_kind="label_list",
        examples=("what occupation does tom cruise's spouse have", "what occupation does barack obama's spouse have"),
        expected_entity_types=(QID_HUMAN,),
        description_hints=("person", "human", "actor", "politician"),
    ),
    "cities_in_country": IntentConfig(
        name="cities_in_country",
        keywords=(r"\bcities\b|\bcity\b", r"\bin\b"),
        entity_patterns=(
            r"^\s*(?:list|show|what\s+are)\s+(?:the\s+)?cities\s+in\s+(?P<entity>.+?)\s*\??\s*$",
        ),
        property_id="Q515 in country",
        answer_kind="label_list",
        examples=("list cities in japan", "what are the cities in france"),
        expected_entity_types=(QID_COUNTRY,),
        description_hints=("country",),
    ),
    "actors_born_in_place": IntentConfig(
        name="actors_born_in_place",
        keywords=(r"\bactors\b|\bactor\b", r"\bborn\b"),
        entity_patterns=(
            r"^\s*(?:list|show|which|what)\s+(?:actors|actor)\s+born\s+in\s+(?P<entity>.+?)\s*\??\s*$",
        ),
        property_id="Q33999 born in place",
        answer_kind="label_list",
        examples=("actors born in london", "list actors born in new york"),
        expected_entity_types=(QID_CITY, QID_COUNTRY),
        description_hints=("city", "country", "place"),
    ),
    "humans_with_occupation": IntentConfig(
        name="humans_with_occupation",
        keywords=(r"\bhumans\b|\bpeople\b", r"\boccupation\b|\bwho are\b"),
        entity_patterns=(
            r"^\s*(?:list|show|which|what)\s+(?:humans|people)\s+with\s+occupation\s+(?P<entity>.+?)\s*\??\s*$",
            r"^\s*who\s+are\s+(?P<entity>.+?)\s*\??\s*$",
        ),
        property_id="Q5 with occupation",
        answer_kind="label_list",
        examples=("list humans with occupation actor", "who are actors"),
        expected_entity_types=(QID_OCCUPATION,),
        description_hints=("occupation", "profession", "job", "role"),
    ),
}
