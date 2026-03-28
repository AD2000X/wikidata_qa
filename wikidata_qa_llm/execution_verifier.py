"""
Execution verifier for the Wikidata QA system.

Checks SPARQL query results for structural validity, type correctness,
cardinality expectations, and basic plausibility.

This layer sits between SPARQL execution and answer formatting.
Hard errors prevent formatting; warnings are logged in the audit trail.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Sequence

# =========================================================
# Expected result structure per intent
# =========================================================

# Maps intent → (required_field, expected_cardinality)
# cardinality: "single" = expect 1 row, "multiple" = expect >= 1 row
_INTENT_EXPECTATIONS: Dict[str, Dict[str, str]] = {
    "age": {"field": "birth", "cardinality": "single"},
    "population": {"field": "population", "cardinality": "single"},
    "capital": {"field": "capitalLabel", "cardinality": "single"},
    "spouse_birth_place": {"field": "placeLabel", "cardinality": "multiple"},
    "birth_country_capital": {"field": "capitalLabel", "cardinality": "single"},
    "spouse_occupation": {"field": "occupationLabel", "cardinality": "multiple"},
    "cities_in_country": {"field": "cityLabel", "cardinality": "multiple"},
    "actors_born_in_place": {"field": "personLabel", "cardinality": "multiple"},
    "humans_with_occupation": {"field": "personLabel", "cardinality": "multiple"},
}


# =========================================================
# Data model
# =========================================================


@dataclass(slots=True)
class VerificationResult:
    is_valid: bool = True
    hard_errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# =========================================================
# Verifier
# =========================================================


class ExecutionVerifier:
    """Validates SPARQL result rows before answer formatting.

    Checks performed:
    - empty_result: no rows returned
    - missing_required_field: expected field not in first row
    - invalid_date_value: birth date cannot be parsed (age intent)
    - invalid_numeric_value: population cannot be converted to number
    - negative_age: computed age is negative
    - negative_population: population value is negative or zero
    - cardinality_violation: row count vs expectation mismatch (warning)
    - multiple_candidates_returned: single-cardinality intent got > 1
      distinct values (warning)
    """

    @staticmethod
    def verify(intent: str, rows: Sequence[Dict[str, str]]) -> VerificationResult:
        result = VerificationResult()
        expectation = _INTENT_EXPECTATIONS.get(intent)

        if not expectation:
            # Unknown intent — cannot verify, pass through
            result.warnings.append("unknown_intent_no_verification")
            return result

        required_field = expectation["field"]
        cardinality = expectation["cardinality"]

        # --- empty result ---
        if not rows:
            result.is_valid = False
            result.hard_errors.append("empty_result")
            return result

        first_row = rows[0]

        # --- missing required field ---
        if required_field not in first_row:
            result.is_valid = False
            result.hard_errors.append(f"missing_required_field:{required_field}")
            return result

        value = first_row[required_field]

        # --- intent-specific value checks ---
        if intent == "age":
            _check_date_value(value, result)

        elif intent == "population":
            _check_numeric_value(value, result)

        # --- cardinality checks (warnings only) ---
        if cardinality == "single" and len(rows) > 1:
            # Check if multiple rows have distinct values for the key field
            distinct_values = {r.get(required_field, "") for r in rows}
            distinct_values.discard("")
            if len(distinct_values) > 1:
                result.warnings.append("multiple_candidates_returned")

        if cardinality == "multiple" and len(rows) == 1:
            result.warnings.append("cardinality_violation:expected_multiple_got_single")

        return result


# =========================================================
# Value check helpers
# =========================================================


def _check_date_value(value: str, result: VerificationResult) -> None:
    """Validate that a birth date string is parseable and yields a sensible age."""
    if not value.strip():
        result.is_valid = False
        result.hard_errors.append("invalid_date_value:empty")
        return

    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        from datetime import date
        today = date.today()
        birth = dt.date()
        age = today.year - birth.year
        if (today.month, today.day) < (birth.month, birth.day):
            age -= 1
        if age < 0:
            result.warnings.append("suspicious_value:negative_age")
    except (ValueError, TypeError):
        result.is_valid = False
        result.hard_errors.append("invalid_date_value:parse_error")


def _check_numeric_value(value: str, result: VerificationResult) -> None:
    """Validate that a population string is a valid positive number."""
    if not value.strip():
        result.is_valid = False
        result.hard_errors.append("invalid_numeric_value:empty")
        return

    try:
        num = float(value)
        if num < 0:
            result.warnings.append("suspicious_value:negative_population")
        elif num == 0:
            result.warnings.append("suspicious_value:zero_population")
    except (ValueError, TypeError):
        result.is_valid = False
        result.hard_errors.append("invalid_numeric_value:parse_error")
