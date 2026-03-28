"""
Resolution reconciler for the Wikidata QA system.

Sits between entity resolution and SPARQL execution.
Compares the parser's entity_type_hint (a guess) against
the resolver's actual P31/P279 types (ground truth from Wikidata)
and the intent's expected entity types.

This layer does NOT hard-reject.  It produces warnings and
a corrected type hint for the audit trail.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

from config import (
    QID_CITY,
    QID_COUNTRY,
    QID_HUMAN,
    QID_OCCUPATION,
    QID_ACTOR,
)

# =========================================================
# Type QID → hint mapping
# =========================================================

_TYPE_QID_TO_HINT = {
    QID_HUMAN: "person",
    QID_CITY: "city",
    QID_COUNTRY: "country",
    QID_OCCUPATION: "occupation",
    QID_ACTOR: "occupation",
}

# Broader category groups for fuzzy matching.
# If resolved types include any QID in the set, the hint applies.
_HINT_QID_GROUPS = {
    "person": {QID_HUMAN},
    "city": {QID_CITY},
    "country": {QID_COUNTRY},
    "occupation": {QID_OCCUPATION, QID_ACTOR},
    "place": {QID_CITY, QID_COUNTRY},
}


# =========================================================
# Data model
# =========================================================


@dataclass(slots=True)
class ReconciliationResult:
    is_compatible: bool = True
    warnings: List[str] = field(default_factory=list)
    corrected_entity_type_hint: str = "unknown"


# =========================================================
# Reconciler
# =========================================================


class ResolutionReconciler:
    """Compares parser hint, resolved types, and intent expectation.

    Usage::

        result = ResolutionReconciler.reconcile(
            parser_type_hint="person",
            resolved_types={"Q5", "Q515", ...},
            intent_expected_types={"Q5"},
        )
    """

    @staticmethod
    def reconcile(
        parser_type_hint: str,
        resolved_types: Set[str],
        intent_expected_types: Tuple[str, ...] | Set[str],
    ) -> ReconciliationResult:
        expected = set(intent_expected_types)
        result = ReconciliationResult()

        # --- derive corrected hint from resolved types ---
        result.corrected_entity_type_hint = _derive_hint(resolved_types, parser_type_hint)

        if not resolved_types:
            # Resolver used local fallback / override — no real types available
            result.warnings.append("unverified_type")
            result.corrected_entity_type_hint = parser_type_hint
            return result

        if not expected:
            # Intent has no expected types (shouldn't happen with current config)
            return result

        if resolved_types & expected:
            # Perfect match: resolved types overlap with what the intent expects
            result.is_compatible = True
        else:
            # No overlap — flag tension but do NOT hard-reject.
            # The resolver's own scoring already accounts for type matching,
            # so this situation usually means either (a) the parser guessed a
            # borderline intent or (b) the entity is genuinely ambiguous.
            result.is_compatible = False
            result.warnings.append("type_intent_tension")

        # --- check parser hint vs resolved types ---
        if parser_type_hint != "unknown":
            hint_qids = _HINT_QID_GROUPS.get(parser_type_hint, set())
            if hint_qids and not (resolved_types & hint_qids):
                result.warnings.append("parser_hint_mismatch")

        return result


def _derive_hint(resolved_types: Set[str], fallback: str) -> str:
    """Map resolved QID types to a human-readable hint string."""
    for qid, hint in _TYPE_QID_TO_HINT.items():
        if qid in resolved_types:
            return hint
    return fallback
