"""
Gemini API configuration for the semantic parser.

Model settings, JSON schema for constrained decoding,
and few-shot prompt for intent classification + entity extraction.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

# =========================================================
# API settings
# =========================================================

GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL: str = "gemini-2.5-flash-lite"
GEMINI_API_BASE: str = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_TIMEOUT: int = 15
GEMINI_MAX_RETRIES: int = 2

# =========================================================
# Confidence routing thresholds
# =========================================================

CONFIDENCE_HIGH: float = 0.75
CONFIDENCE_LOW: float = 0.40
CONFIDENCE_REJECT: float = 0.20

# =========================================================
# JSON Schema for constrained structured output
#
# This schema is passed to Gemini via response_schema.
# The model MUST output exactly these fields with the
# allowed enum values.  relation / answer_format are
# NOT included — they are derived from intent by the
# deterministic compiler layer.
# =========================================================

SEMANTIC_FRAME_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "intent": {
            "type": "string",
            "enum": [
                "age",
                "population",
                "capital",
                "spouse_birth_place",
                "birth_country_capital",
                "spouse_occupation",
                "cities_in_country",
                "actors_born_in_place",
                "humans_with_occupation",
                "unsupported",
            ],
        },
        "entity_text": {
            "type": "string",
        },
        "entity_type_hint": {
            "type": "string",
            "enum": ["person", "city", "country", "place", "occupation", "unknown"],
        },
        "confidence": {
            "type": "number",
        },
    },
    "required": ["intent", "entity_text", "entity_type_hint", "confidence"],
}

# =========================================================
# System prompt
# =========================================================

SYSTEM_PROMPT: str = """\
You are a semantic parser for a Wikidata question-answering system.
Given a natural language question, output ONLY a JSON object matching the provided schema.

## Intent definitions

- age: asking how old a person is (maps to date of birth)
- population: asking for the population of a city or country
- capital: asking for the capital of a country
- spouse_birth_place: asking where a person's spouse was born (two-hop: person → spouse → birth place)
- birth_country_capital: asking for the capital of the country where a person was born (three-hop: person → birth place → country → capital)
- spouse_occupation: asking what occupation a person's spouse has (two-hop: person → spouse → occupation)
- cities_in_country: asking to list cities in a country
- actors_born_in_place: asking which actors were born in a place
- humans_with_occupation: asking to list people with a specific occupation
- unsupported: the question does not match any of the above intents

## Entity type hint rules

- person: the entity is a human being (actor, singer, politician, etc.)
- city: the entity is a city or town
- country: the entity is a country or sovereign state
- place: the entity is a geographic location that is not clearly a city or country
- occupation: the entity is a job or profession (actor, physicist, etc.)
- unknown: you are not sure what type the entity is

## Confidence calibration

- 0.9 to 1.0: you are certain about both intent and entity
- 0.6 to 0.8: you are fairly confident but the question is slightly ambiguous
- 0.3 to 0.5: the question is ambiguous or borderline
- 0.0 to 0.2: the question likely does not match any supported intent

## Critical rules

- For multi-hop intents (spouse_birth_place, birth_country_capital, spouse_occupation), the entity_text is the ROOT person, not the spouse or place.
- If the question asks about a person's OWN birth place, birth country, or occupation (without mentioning spouse), return unsupported — only spouse-related or country-capital chains are supported.
- Never invent intents outside the allowed set.
- If uncertain, prefer unsupported with low confidence over guessing.\
"""

# =========================================================
# Few-shot examples
#
# Covers: normal single-hop, multi-hop, minimal pairs,
# negation/unsupported, ambiguous-but-handleable.
# =========================================================

FEW_SHOT_EXAMPLES: List[Dict[str, str]] = [
    # --- single-hop: age ---
    {
        "question": "How old is Tom Cruise?",
        "answer": '{"intent":"age","entity_text":"Tom Cruise","entity_type_hint":"person","confidence":0.95}',
    },
    # --- single-hop: population ---
    {
        "question": "What is the population of London?",
        "answer": '{"intent":"population","entity_text":"London","entity_type_hint":"city","confidence":0.95}',
    },
    # --- single-hop: capital ---
    {
        "question": "What is the capital of Japan?",
        "answer": '{"intent":"capital","entity_text":"Japan","entity_type_hint":"country","confidence":0.95}',
    },
    # --- multi-hop: spouse_birth_place ---
    {
        "question": "Where was Tom Hanks's spouse born?",
        "answer": '{"intent":"spouse_birth_place","entity_text":"Tom Hanks","entity_type_hint":"person","confidence":0.92}',
    },
    # --- multi-hop: birth_country_capital ---
    {
        "question": "What is the capital of the country where Einstein was born?",
        "answer": '{"intent":"birth_country_capital","entity_text":"Einstein","entity_type_hint":"person","confidence":0.90}',
    },
    # --- multi-hop: spouse_occupation ---
    {
        "question": "What is Barack Obama's spouse's occupation?",
        "answer": '{"intent":"spouse_occupation","entity_text":"Barack Obama","entity_type_hint":"person","confidence":0.93}',
    },
    # --- minimal pair: own birth place → unsupported ---
    {
        "question": "Where was Tom Hanks born?",
        "answer": '{"intent":"unsupported","entity_text":"Tom Hanks","entity_type_hint":"person","confidence":0.15}',
    },
    # --- list: cities_in_country ---
    {
        "question": "List cities in France",
        "answer": '{"intent":"cities_in_country","entity_text":"France","entity_type_hint":"country","confidence":0.93}',
    },
    # --- list: actors_born_in_place ---
    {
        "question": "Which actors were born in London?",
        "answer": '{"intent":"actors_born_in_place","entity_text":"London","entity_type_hint":"city","confidence":0.92}',
    },
    # --- ambiguous but handleable: humans_with_occupation ---
    {
        "question": "Who are physicists?",
        "answer": '{"intent":"humans_with_occupation","entity_text":"physicists","entity_type_hint":"occupation","confidence":0.80}',
    },
    # --- unsupported: vague ---
    {
        "question": "Tell me a fact about London",
        "answer": '{"intent":"unsupported","entity_text":"London","entity_type_hint":"city","confidence":0.10}',
    },
    # --- unsupported: opinion ---
    {
        "question": "What does Tom Cruise think about politics?",
        "answer": '{"intent":"unsupported","entity_text":"Tom Cruise","entity_type_hint":"person","confidence":0.08}',
    },
]


def build_gemini_messages(question: str) -> List[Dict[str, Any]]:
    """Build the messages array for the Gemini API call.

    Returns a list of role/parts dicts suitable for the
    ``contents`` field of a generateContent request.
    """
    contents: List[Dict[str, Any]] = []
    for example in FEW_SHOT_EXAMPLES:
        contents.append({"role": "user", "parts": [{"text": example["question"]}]})
        contents.append({"role": "model", "parts": [{"text": example["answer"]}]})
    contents.append({"role": "user", "parts": [{"text": question}]})
    return contents


def build_gemini_request_body(question: str) -> Dict[str, Any]:
    """Build the full JSON request body for Gemini generateContent."""
    return {
        "system_instruction": {
            "parts": [{"text": SYSTEM_PROMPT}],
        },
        "contents": build_gemini_messages(question),
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": SEMANTIC_FRAME_SCHEMA,
            "temperature": 0.1,
            "maxOutputTokens": 256,
        },
    }
