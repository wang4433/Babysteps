"""Sim-free tests for babysteps.stage5.conditions (the registry)."""
from __future__ import annotations

from babysteps.stage5.conditions import (
    CONDITION_REGISTRY, CONDITIONS, available_conditions, deferred_conditions,
)


def test_six_conditions_in_canonical_order():
    assert CONDITIONS == (
        "same_intent_retry",
        "random_factor_local_edit",
        "vlm_free_replan",
        "vlm_diagnosis_local_edit",
        "shared_revision_policy",
        "oracle_single_slot",
    )
    assert set(CONDITION_REGISTRY) == set(CONDITIONS)


def test_shared_policy_is_deferred_not_runnable():
    # corrections #8: the proposed shared policy is registered as deferred, not
    # a runtime stub that raises during registry use.
    assert deferred_conditions() == ("shared_revision_policy",)
    assert "shared_revision_policy" not in available_conditions()
    assert CONDITION_REGISTRY["shared_revision_policy"].runnable is False


def test_available_conditions_are_the_five_runnable():
    assert available_conditions() == (
        "same_intent_retry",
        "random_factor_local_edit",
        "vlm_free_replan",
        "vlm_diagnosis_local_edit",
        "oracle_single_slot",
    )


def test_condition_kinds():
    kinds = {c: CONDITION_REGISTRY[c].kind for c in CONDITIONS}
    assert kinds["same_intent_retry"] == "identity"
    assert kinds["random_factor_local_edit"] == "paired"
    assert kinds["vlm_free_replan"] == "free_replan"
    assert kinds["vlm_diagnosis_local_edit"] == "paired"
    assert kinds["oracle_single_slot"] == "oracle"
