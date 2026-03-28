"""
Tests for the LLM-assisted semantic parsing pipeline.

Covers: FrameValidator, ResolutionReconciler, ExecutionVerifier,
RegexSemanticParser, and the legacy (regex-only) flow.

Gemini integration tests require GEMINI_API_KEY and are skipped
if the key is not set.
"""

from __future__ import annotations

import os
import sys

from semantic_parser import (
    FrameValidator,
    RegexSemanticParser,
    SemanticFrame,
    SemanticParserError,
)
from resolution_reconciler import ResolutionReconciler
from execution_verifier import ExecutionVerifier


# =========================================================
# FrameValidator tests
# =========================================================


def test_validator_valid_frame():
    frame = SemanticFrame(intent="age", entity_text="Tom Cruise", entity_type_hint="person", confidence=0.95)
    errors, warnings = FrameValidator.validate(frame)
    assert errors == [], f"Expected no errors, got: {errors}"
    assert warnings == [], f"Expected no warnings, got: {warnings}"
    print("[PASS] test_validator_valid_frame")


def test_validator_invalid_intent():
    frame = SemanticFrame(intent="weather", entity_text="London", entity_type_hint="city", confidence=0.9)
    errors, warnings = FrameValidator.validate(frame)
    assert "invalid_intent" in errors
    print("[PASS] test_validator_invalid_intent")


def test_validator_empty_entity():
    frame = SemanticFrame(intent="age", entity_text="", entity_type_hint="person", confidence=0.9)
    errors, warnings = FrameValidator.validate(frame)
    assert "empty_entity" in errors
    print("[PASS] test_validator_empty_entity")


def test_validator_empty_entity_allowed_for_unsupported():
    frame = SemanticFrame(intent="unsupported", entity_text="", entity_type_hint="unknown", confidence=0.1)
    errors, warnings = FrameValidator.validate(frame)
    assert "empty_entity" not in errors
    print("[PASS] test_validator_empty_entity_allowed_for_unsupported")


def test_validator_hard_type_mismatch():
    frame = SemanticFrame(intent="age", entity_text="London", entity_type_hint="city", confidence=0.9)
    errors, warnings = FrameValidator.validate(frame)
    assert "hard_type_mismatch" in errors
    print("[PASS] test_validator_hard_type_mismatch")


def test_validator_soft_type_mismatch():
    frame = SemanticFrame(intent="age", entity_text="Jordan", entity_type_hint="unknown", confidence=0.8)
    errors, warnings = FrameValidator.validate(frame)
    assert errors == [], f"Expected no errors, got: {errors}"
    assert "soft_type_mismatch" in warnings
    print("[PASS] test_validator_soft_type_mismatch")


def test_validator_low_confidence_warning():
    frame = SemanticFrame(intent="capital", entity_text="France", entity_type_hint="country", confidence=0.3)
    errors, warnings = FrameValidator.validate(frame)
    assert "low_confidence" in warnings
    print("[PASS] test_validator_low_confidence_warning")


def test_validator_medium_confidence_warning():
    frame = SemanticFrame(intent="capital", entity_text="France", entity_type_hint="country", confidence=0.6)
    errors, warnings = FrameValidator.validate(frame)
    assert "medium_confidence" in warnings
    print("[PASS] test_validator_medium_confidence_warning")


# =========================================================
# ResolutionReconciler tests
# =========================================================


def test_reconciler_compatible():
    result = ResolutionReconciler.reconcile(
        parser_type_hint="person",
        resolved_types={"Q5", "Q33999"},  # human + actor
        intent_expected_types={"Q5"},  # expects human
    )
    assert result.is_compatible is True
    assert result.warnings == []
    assert result.corrected_entity_type_hint == "person"
    print("[PASS] test_reconciler_compatible")


def test_reconciler_type_intent_tension():
    result = ResolutionReconciler.reconcile(
        parser_type_hint="person",
        resolved_types={"Q6256"},  # country
        intent_expected_types={"Q5"},  # expects human
    )
    assert result.is_compatible is False
    assert "type_intent_tension" in result.warnings
    assert result.corrected_entity_type_hint == "country"
    print("[PASS] test_reconciler_type_intent_tension")


def test_reconciler_unverified_type():
    result = ResolutionReconciler.reconcile(
        parser_type_hint="person",
        resolved_types=set(),  # no types (local fallback)
        intent_expected_types={"Q5"},
    )
    assert "unverified_type" in result.warnings
    assert result.corrected_entity_type_hint == "person"  # keeps parser hint
    print("[PASS] test_reconciler_unverified_type")


def test_reconciler_parser_hint_mismatch():
    result = ResolutionReconciler.reconcile(
        parser_type_hint="person",
        resolved_types={"Q515", "Q6256"},  # city + country
        intent_expected_types={"Q515", "Q6256"},  # expects city/country
    )
    assert result.is_compatible is True
    assert "parser_hint_mismatch" in result.warnings
    print("[PASS] test_reconciler_parser_hint_mismatch")


# =========================================================
# ExecutionVerifier tests
# =========================================================


def test_verifier_valid_age():
    rows = [{"birth": "1962-07-03T00:00:00Z"}]
    result = ExecutionVerifier.verify("age", rows)
    assert result.is_valid is True
    assert result.hard_errors == []
    print("[PASS] test_verifier_valid_age")


def test_verifier_empty_result():
    result = ExecutionVerifier.verify("age", [])
    assert result.is_valid is False
    assert "empty_result" in result.hard_errors
    print("[PASS] test_verifier_empty_result")


def test_verifier_missing_field():
    rows = [{"wrong_field": "value"}]
    result = ExecutionVerifier.verify("age", rows)
    assert result.is_valid is False
    assert any("missing_required_field" in e for e in result.hard_errors)
    print("[PASS] test_verifier_missing_field")


def test_verifier_invalid_date():
    rows = [{"birth": "not-a-date"}]
    result = ExecutionVerifier.verify("age", rows)
    assert result.is_valid is False
    assert any("invalid_date_value" in e for e in result.hard_errors)
    print("[PASS] test_verifier_invalid_date")


def test_verifier_valid_population():
    rows = [{"population": "8799728"}]
    result = ExecutionVerifier.verify("population", rows)
    assert result.is_valid is True
    print("[PASS] test_verifier_valid_population")


def test_verifier_invalid_population():
    rows = [{"population": "not-a-number"}]
    result = ExecutionVerifier.verify("population", rows)
    assert result.is_valid is False
    assert any("invalid_numeric_value" in e for e in result.hard_errors)
    print("[PASS] test_verifier_invalid_population")


def test_verifier_cardinality_warning():
    rows = [{"capitalLabel": "Tokyo"}, {"capitalLabel": "Kyoto"}]
    result = ExecutionVerifier.verify("capital", rows)
    assert result.is_valid is True
    assert "multiple_candidates_returned" in result.warnings
    print("[PASS] test_verifier_cardinality_warning")


def test_verifier_single_for_multiple():
    rows = [{"cityLabel": "Tokyo"}]
    result = ExecutionVerifier.verify("cities_in_country", rows)
    assert result.is_valid is True
    assert "cardinality_violation:expected_multiple_got_single" in result.warnings
    print("[PASS] test_verifier_single_for_multiple")


# =========================================================
# RegexSemanticParser tests
# =========================================================


def test_regex_parser_age():
    parser = RegexSemanticParser()
    frame = parser.parse("How old is Tom Cruise?")
    assert frame.intent == "age"
    assert frame.entity_text == "Tom Cruise"
    assert frame.entity_type_hint == "person"
    assert frame.confidence == 0.9
    print("[PASS] test_regex_parser_age")


def test_regex_parser_population():
    parser = RegexSemanticParser()
    frame = parser.parse("What is the population of London?")
    assert frame.intent == "population"
    assert frame.entity_text == "London"
    assert frame.confidence == 0.9
    print("[PASS] test_regex_parser_population")


def test_regex_parser_capital():
    parser = RegexSemanticParser()
    frame = parser.parse("What is the capital of France?")
    assert frame.intent == "capital"
    assert frame.entity_text == "France"
    assert frame.entity_type_hint == "country"
    print("[PASS] test_regex_parser_capital")


def test_regex_parser_unsupported():
    parser = RegexSemanticParser()
    try:
        parser.parse("Tell me a joke")
        assert False, "Should have raised"
    except SemanticParserError:
        pass
    print("[PASS] test_regex_parser_unsupported")


# =========================================================
# Gemini integration tests (require API key)
# =========================================================


def test_gemini_parser_age():
    """Requires GEMINI_API_KEY environment variable."""
    from semantic_parser import GeminiSemanticParser
    parser = GeminiSemanticParser()
    frame = parser.parse("How old is Madonna?")
    assert frame.intent == "age", f"Expected 'age', got '{frame.intent}'"
    assert "madonna" in frame.entity_text.lower(), f"Expected 'Madonna' in entity_text, got '{frame.entity_text}'"
    assert frame.confidence > 0.5
    print(f"[PASS] test_gemini_parser_age (confidence={frame.confidence:.2f})")


def test_gemini_parser_multi_hop():
    """Requires GEMINI_API_KEY environment variable."""
    from semantic_parser import GeminiSemanticParser
    parser = GeminiSemanticParser()
    frame = parser.parse("Where was Tom Hanks's spouse born?")
    assert frame.intent == "spouse_birth_place", f"Expected 'spouse_birth_place', got '{frame.intent}'"
    assert "tom hanks" in frame.entity_text.lower()
    print(f"[PASS] test_gemini_parser_multi_hop (confidence={frame.confidence:.2f})")


def test_gemini_parser_unsupported():
    """Requires GEMINI_API_KEY environment variable."""
    from semantic_parser import GeminiSemanticParser
    parser = GeminiSemanticParser()
    frame = parser.parse("What does Tom Cruise think about politics?")
    assert frame.intent == "unsupported", f"Expected 'unsupported', got '{frame.intent}'"
    print(f"[PASS] test_gemini_parser_unsupported (confidence={frame.confidence:.2f})")


# =========================================================
# Legacy flow test (regex-only mode)
# =========================================================


def test_legacy_flow():
    """Test that use_llm_parser=False preserves original behaviour."""
    from wikidata_qa_llm import ask
    answer = ask("What is the capital of France?", use_llm_parser=False)
    assert answer.lower() == "paris", f"Expected 'Paris', got '{answer}'"
    print("[PASS] test_legacy_flow")


# =========================================================
# Runner
# =========================================================


if __name__ == "__main__":
    print("=" * 60)
    print("Unit tests (no API key required)")
    print("=" * 60)

    # Validator
    test_validator_valid_frame()
    test_validator_invalid_intent()
    test_validator_empty_entity()
    test_validator_empty_entity_allowed_for_unsupported()
    test_validator_hard_type_mismatch()
    test_validator_soft_type_mismatch()
    test_validator_low_confidence_warning()
    test_validator_medium_confidence_warning()

    # Reconciler
    test_reconciler_compatible()
    test_reconciler_type_intent_tension()
    test_reconciler_unverified_type()
    test_reconciler_parser_hint_mismatch()

    # Verifier
    test_verifier_valid_age()
    test_verifier_empty_result()
    test_verifier_missing_field()
    test_verifier_invalid_date()
    test_verifier_valid_population()
    test_verifier_invalid_population()
    test_verifier_cardinality_warning()
    test_verifier_single_for_multiple()

    # Regex parser
    test_regex_parser_age()
    test_regex_parser_population()
    test_regex_parser_capital()
    test_regex_parser_unsupported()

    print()
    print("=" * 60)
    print("Integration tests (require network)")
    print("=" * 60)

    test_legacy_flow()

    print()
    if os.environ.get("GEMINI_API_KEY"):
        print("=" * 60)
        print("Gemini integration tests")
        print("=" * 60)
        test_gemini_parser_age()
        test_gemini_parser_multi_hop()
        test_gemini_parser_unsupported()
    else:
        print("[SKIP] Gemini tests: GEMINI_API_KEY not set")

    print()
    print("All tests complete.")
