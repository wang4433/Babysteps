"""Structured failure detection + attribution.

`build_failure_packet` turns an AttemptResult into a FailurePacket whose
`failure_predicate` is derived from a strict precedence rule (most specific
first). `attribute_failure` then maps that predicate onto the implicated
intent factor via a small rule table.

Both functions are pure. They're the place where "failure as observation"
becomes "structured intent-revision signal" — i.e., where the core BABYSTEPS
claim is encoded. Future stages replace `attribute_failure` with a learned
classifier; the rule table stays as the analytic upper bound.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from babysteps.schemas import (
    INTENT_FIELDS,
    AttemptResult,
    FailurePacket,
    Intent,
    SceneState,
)

# Predicate → (primary wrong_factor, ordered tuple of factors safe to revise).
# Tuple-of-factors-to-revise is intentionally a tuple, not a set: ordering
# encodes "try this revision first, fall back to that one" for future
# multi-candidate revisers.
FAILURE_TO_FACTOR: dict[str, tuple[str, tuple[str, ...]]] = {
    "approach_blocked":   ("approach_direction", ("approach_direction", "contact_region")),
    "direction_error":    ("approach_direction", ("approach_direction",)),
    "contact_failure":    ("contact_region",     ("contact_region",)),
    "no_motion":          ("approach_direction", ("approach_direction", "contact_region")),
    "goal_not_satisfied": ("goal_state",         ("goal_state",)),
    # Sub-project B (PickCube): grasp_slip → contact_region is wrong (the
    # chosen gripper-axis is slip-prone for this cube). embodiment_mapping
    # is also in `revise` to permit future operators that switch grasp
    # primitives; Stage-0's contact_substitution does not touch it.
    "grasp_slip":         ("contact_region",     ("contact_region", "embodiment_mapping")),
    # Sub-project D (TurnFaucet): constraint_violation fires when the gripper
    # contacted a non-articulating link and tried to actuate it (collision=True,
    # object_moved=False). Revise both constraint_region and contact_region —
    # the two-factor pair that constraint_introduction operates over.
    "constraint_violation": ("constraint_region", ("constraint_region", "contact_region")),
}


@dataclass(frozen=True)
class Attribution:
    semantic_failure: bool
    wrong_factor: Optional[str]
    freeze: tuple[str, ...]
    revise: tuple[str, ...]


# ---------- helpers ---------------------------------------------------- #


def _displacement(a: AttemptResult) -> float:
    init = np.asarray(a.initial_obj_xy, dtype=np.float64)
    final = np.asarray(a.final_obj_xy, dtype=np.float64)
    return float(np.linalg.norm(final - init))


def _direction_alignment(a: AttemptResult) -> Optional[float]:
    """cos(motion, goal − initial) ∈ [-1, 1]; None if either vector is zero."""
    init = np.asarray(a.initial_obj_xy, dtype=np.float64)
    final = np.asarray(a.final_obj_xy, dtype=np.float64)
    goal = np.asarray(a.goal_xy, dtype=np.float64)
    motion = final - init
    intended = goal - init
    mn = np.linalg.norm(motion)
    iv = np.linalg.norm(intended)
    if mn < 1e-9 or iv < 1e-9:
        return None
    return float(np.dot(motion, intended) / (mn * iv))


# ---------- public API ------------------------------------------------- #


def build_failure_packet(
    intent: Intent, attempt: AttemptResult, scene: SceneState,
) -> FailurePacket:
    """Derive the structured FailurePacket. Predicate precedence: most
    specific first (success → planner_failed → constraint_violation →
    grasp_slip → contact_failure → no_motion → direction_error →
    goal_not_satisfied).
    """
    et = {
        "reached_contact": bool(attempt.reached_contact),
        "object_moved":    bool(attempt.object_moved),
        "collision":       bool(attempt.collision),
        "planner_failed":  bool(attempt.planner_failed),
        "grasp_slip":      bool(attempt.grasp_slip),
    }
    disp = _displacement(attempt)
    align = _direction_alignment(attempt)

    if attempt.success:
        predicate = "none"
    elif attempt.planner_failed:
        predicate = "approach_blocked"
    elif attempt.collision and not attempt.object_moved:
        # Sub-project D: constraint_violation when the gripper contacted a
        # non-articulating link and tried to actuate it. The env_runner
        # marks this case by setting collision=True (Stage-0 proxy for
        # "touched something that didn't move"). More specific than
        # grasp_slip / contact_failure / no_motion.
        predicate = "constraint_violation"
    elif attempt.grasp_slip:
        # Sub-project B: grasp_slip is more specific than the contact /
        # motion / direction predicates below — it carries the strong
        # signal that the gripper DID reach the cube but lost grip. Goes
        # right after planner_failed in the precedence.
        predicate = "grasp_slip"
    elif not attempt.reached_contact:
        predicate = "contact_failure"
    elif not attempt.object_moved:
        predicate = "no_motion"
    elif align is not None and align < 0:
        predicate = "direction_error"
    else:
        predicate = "goal_not_satisfied"

    # `scene` is currently unused in detection — kept on the signature for
    # future stages (e.g., to incorporate constraint-violation checks).
    _ = scene
    return FailurePacket(
        chosen_intent=intent,
        execution_trace=et,
        failure_predicate=predicate,
        object_displacement=disp,
        direction_alignment=align,
    )


def attribute_failure(fp: FailurePacket) -> Attribution:
    """Rule-based predicate → wrong_factor mapping. Stage-0 analytic baseline."""
    if fp.failure_predicate == "none":
        return Attribution(
            semantic_failure=False,
            wrong_factor=None,
            freeze=INTENT_FIELDS,
            revise=(),
        )
    if fp.failure_predicate not in FAILURE_TO_FACTOR:
        raise ValueError(
            f"no attribution rule for failure_predicate {fp.failure_predicate!r}"
        )
    wrong_factor, revise = FAILURE_TO_FACTOR[fp.failure_predicate]
    freeze = tuple(f for f in INTENT_FIELDS if f not in revise)
    return Attribution(
        semantic_failure=True,
        wrong_factor=wrong_factor,
        freeze=freeze,
        revise=revise,
    )
