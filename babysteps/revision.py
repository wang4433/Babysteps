"""Factor-local intent revision.

`revise_intent` produces a new Intent that differs from the input in EXACTLY
ONE field — the field the attribution rule named. Every other factor is
copied byte-identical from the input. This invariant is what makes Stage 0
"factor-local" rather than "generic retry"; the summarizer's
`non_regression_score` audits it on every revised episode.

Stage 0 implements:
  * `approach_substitution` — for wrong_factor=="approach_direction"
    (Sub-project A / PushCube).
  * `contact_substitution` — for wrong_factor=="contact_region"
    (Sub-project B / PickCube).
  * `goal_refinement` — for wrong_factor=="goal_state"
    (Sub-project C / StackCube; strict-extension: cube_at_target →
    cubeA_on_cubeB only).
  * `constraint_introduction` — for wrong_factor=="constraint_region"
    (Sub-project D / TurnFaucet; strict-extension: (none, faucet_base) →
    (faucet_base_static, handle_grip) only; the only Stage-0 operator
    that revises 2 factors at once).

Other wrong_factors raise `NotImplementedError` — honest about what is and
isn't validated.
"""
from __future__ import annotations

from dataclasses import replace

from babysteps.envs.scene import OPPOSITE_APPROACH, ORTHOGONAL_FACE
from babysteps.failure import Attribution
from babysteps.schemas import (
    APPROACH_DIRECTIONS,
    CONTACT_REGIONS,
    INTENT_FIELDS,
    Intent,
    Revision,
    SceneState,
)

# Order in which the reviser searches for an unblocked approach. The opposite
# of the current approach is tried first (the canonical "approach_substitution"
# move), then the remaining cardinals, then "from_above" as a last resort.
_CARDINAL_FALLBACK_ORDER: tuple[str, ...] = (
    "from_minus_x", "from_plus_x", "from_minus_y", "from_plus_y",
)

# Fallback order for contact_substitution if both current and its orthogonal
# are blocked. Searches the four cardinal faces in a deterministic order.
_FACE_FALLBACK_ORDER: tuple[str, ...] = (
    "minus_x_face", "plus_x_face", "minus_y_face", "plus_y_face",
)


def _pick_unblocked_approach(current: str, blocked: tuple[str, ...]) -> str:
    blocked_set = set(blocked)
    # First preference: the geometric opposite of the current approach.
    if current in OPPOSITE_APPROACH:
        cand = OPPOSITE_APPROACH[current]
        if cand not in blocked_set and cand != current:
            return cand
    # Next: any other cardinal not currently used or blocked.
    for cand in _CARDINAL_FALLBACK_ORDER:
        if cand != current and cand not in blocked_set:
            return cand
    # Last resort: top approach (whitelisted in APPROACH_DIRECTIONS).
    assert "from_above" in APPROACH_DIRECTIONS
    return "from_above"


def _pick_unblocked_face(current: str, blocked: tuple[str, ...]) -> str:
    """Stage-0 contact_substitution choice: prefer the 90°-orthogonal face
    (i.e., rotate the gripper axis), then fall back to any unblocked
    cardinal face."""
    blocked_set = set(blocked)
    # First preference: 90°-rotated gripper axis.
    if current in ORTHOGONAL_FACE:
        cand = ORTHOGONAL_FACE[current]
        if cand not in blocked_set and cand != current:
            return cand
    # Next: any other cardinal face that is not the current and not blocked.
    for cand in _FACE_FALLBACK_ORDER:
        if cand != current and cand not in blocked_set:
            return cand
    # Stage-0 has no further fallback (no "any_face" wildcard); if every
    # cardinal is blocked, the executor scene is over-constrained.
    raise RuntimeError(
        f"no unblocked contact_region available: current={current!r}, "
        f"blocked={sorted(blocked_set)!r}. Stage-0 has no further fallback."
    )


def revise_intent(
    intent: Intent, attribution: Attribution, scene: SceneState,
) -> tuple[Intent, Revision]:
    """Return (revised_intent, Revision record). Dispatches on
    `attribution.wrong_factor`. Stage-0 supports approach_direction,
    contact_region, goal_state, and constraint_region; other factors raise
    NotImplementedError.
    """
    if attribution.wrong_factor is None:
        raise ValueError(
            "revise_intent called with attribution.wrong_factor=None; "
            "the failure was not semantic — nothing to revise."
        )

    if attribution.wrong_factor == "approach_direction":
        old = intent.approach_direction
        new = _pick_unblocked_approach(old, scene.blocked_sides)
        revised = replace(intent, approach_direction=new)
        frozen = tuple(f for f in INTENT_FIELDS if f != "approach_direction")
        rev_record = Revision(
            operator="approach_substitution",
            factor="approach_direction",
            old_value=old,
            new_value=new,
            frozen_factors=frozen,
        )
        return revised, rev_record

    if attribution.wrong_factor == "contact_region":
        old = intent.contact_region
        if old not in CONTACT_REGIONS:
            raise ValueError(
                f"contact_substitution: current contact_region {old!r} not in "
                f"CONTACT_REGIONS"
            )
        new = _pick_unblocked_face(old, scene.blocked_sides)
        revised = replace(intent, contact_region=new)
        frozen = tuple(f for f in INTENT_FIELDS if f != "contact_region")
        rev_record = Revision(
            operator="contact_substitution",
            factor="contact_region",
            old_value=old,
            new_value=new,
            frozen_factors=frozen,
        )
        return revised, rev_record

    if attribution.wrong_factor == "goal_state":
        # Stage-0's goal_refinement is a strict-extension operator:
        # cube_at_target → cubeA_on_cubeB only. Other goal_state transitions
        # are deferred (per spec §6 of
        # docs/superpowers/specs/2026-05-17-stage0-stackcube-c-design.md).
        if intent.goal_state != "cube_at_target":
            raise NotImplementedError(
                f"goal_refinement does not handle transitions from "
                f"goal_state {intent.goal_state!r}. (Stage-0 supports only "
                f"the cube_at_target → cubeA_on_cubeB refinement per "
                f"docs/superpowers/specs/2026-05-17-stage0-stackcube-c-design.md §6)"
            )
        old = intent.goal_state
        new = "cubeA_on_cubeB"
        revised = replace(intent, goal_state=new)
        frozen = tuple(f for f in INTENT_FIELDS if f != "goal_state")
        rev_record = Revision(
            operator="goal_refinement",
            factor="goal_state",
            old_value=old,
            new_value=new,
            frozen_factors=frozen,
        )
        return revised, rev_record

    if attribution.wrong_factor == "constraint_region":
        if (intent.constraint_region != "none"
                or intent.contact_region != "faucet_base"):
            raise NotImplementedError(
                f"constraint_introduction does not handle transitions "
                f"from (constraint_region={intent.constraint_region!r}, "
                f"contact_region={intent.contact_region!r}). (Stage-0 "
                f"supports only the (none, faucet_base) → "
                f"(faucet_base_static, handle_grip) refinement per "
                f"docs/superpowers/specs/2026-05-17-stage0-turnfaucet-d-design.md §6)"
            )
        revised = replace(
            intent,
            constraint_region="faucet_base_static",
            contact_region="handle_grip",
        )
        frozen = tuple(
            f for f in INTENT_FIELDS
            if f not in ("constraint_region", "contact_region")
        )
        rev_record = Revision(
            operator="constraint_introduction",
            factor="constraint_region",   # primary factor for audit
            old_value="none",
            new_value="faucet_base_static",
            frozen_factors=frozen,
        )
        return revised, rev_record

    if attribution.wrong_factor == "embodiment_mapping":
        # Spec §6: pure single-factor swap. Only the
        # grasp_turn → poke_turn transition is supported in Stage-0.
        if intent.embodiment_mapping != "proxy_contact_to_franka_grasp_turn":
            raise NotImplementedError(
                f"embodiment_substitution handles only "
                f"grasp_turn → poke_turn (got {intent.embodiment_mapping!r}). "
                f"See docs/superpowers/specs/"
                f"2026-05-18-stage0-turnfaucet-embodiment-design.md §6"
            )
        revised = replace(
            intent, embodiment_mapping="proxy_contact_to_franka_poke_turn",
        )
        frozen = tuple(f for f in INTENT_FIELDS if f != "embodiment_mapping")
        rev = Revision(
            operator="embodiment_substitution",
            factor="embodiment_mapping",
            old_value="proxy_contact_to_franka_grasp_turn",
            new_value="proxy_contact_to_franka_poke_turn",
            frozen_factors=frozen,
        )
        return revised, rev

    raise NotImplementedError(
        f"Stage-0 reviser handles 'approach_direction', 'contact_region', "
        f"'goal_state', and 'constraint_region'; got "
        f"{attribution.wrong_factor!r}. (Other factors are reserved for "
        f"later sub-projects — see "
        f"docs/superpowers/specs/2026-05-17-stage0-four-scene-roadmap-design.md §6)"
    )
