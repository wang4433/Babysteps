"""Stage-5 unified main-table — the comparison conditions (PURE, sim-free).

The six conditions of the elevated research target
(``redesign_failure_paradigm.md`` → "Required evaluation conditions"). Each is
described by a :class:`ConditionSpec` whose ``kind`` tells the evaluator how to
run it (see ``scripts/stage5_unified_maintable_eval.py``):

* ``identity``    — retry the initial intent unchanged (no attribution/edit).
* ``paired``      — an ``Attributor`` selects a factor, a ``RevisionPolicy``
                    selects the typed value (the honest decomposition).
* ``free_replan`` — one joint VLM call regenerates the whole intent.
* ``oracle``      — privileged upper bound (oracle factor + oracle value).

``shared_revision_policy`` is the proposed method (build-order step 2). It is
registered as **deferred / not runnable** — there is no shared checkpoint yet,
so the evaluator records it as deferred and never instantiates a policy for it
(corrections #8: not a runtime stub that raises during registry use)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    kind: str            # identity | paired | free_replan | oracle
    runnable: bool = True
    note: str = ""


# Canonical order = the order rows appear in the main table.
CONDITIONS: tuple[str, ...] = (
    "same_intent_retry",
    "random_factor_local_edit",
    "vlm_free_replan",
    "vlm_diagnosis_local_edit",
    "shared_revision_policy",
    "oracle_single_slot",
)

CONDITION_REGISTRY: dict[str, ConditionSpec] = {
    "same_intent_retry": ConditionSpec(
        "same_intent_retry", "identity",
        note="failure-agnostic retry control"),
    "random_factor_local_edit": ConditionSpec(
        "random_factor_local_edit", "paired",
        note="RandomAttributor + RandomCandidatePolicy"),
    "vlm_free_replan": ConditionSpec(
        "vlm_free_replan", "free_replan",
        note="live VLM regenerates the full intent (broad-replan baseline)"),
    "vlm_diagnosis_local_edit": ConditionSpec(
        "vlm_diagnosis_local_edit", "paired",
        note="VLMAttributor + PerTaskEditorAdapter (factorization ablation)"),
    "shared_revision_policy": ConditionSpec(
        "shared_revision_policy", "paired", runnable=False,
        note="proposed shared, task-general policy — build-order step 2"),
    "oracle_single_slot": ConditionSpec(
        "oracle_single_slot", "oracle",
        note="OracleAttributor + OracleValuePolicy (upper bound only)"),
}


def available_conditions() -> tuple[str, ...]:
    """Conditions runnable today (excludes the deferred shared policy)."""
    return tuple(c for c in CONDITIONS if CONDITION_REGISTRY[c].runnable)


def deferred_conditions() -> tuple[str, ...]:
    return tuple(c for c in CONDITIONS if not CONDITION_REGISTRY[c].runnable)
