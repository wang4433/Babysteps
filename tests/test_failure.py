"""Tests for babysteps.failure — packet builder + rule-based attribution.

The packet builder maps an AttemptResult onto a structured FailurePacket whose
`failure_predicate` is determined by a strict precedence (most specific first).
The attribution mapping (predicate → wrong_factor) is the Stage-0 rule table.
"""
from __future__ import annotations

import pytest

from babysteps.failure import (
    FAILURE_TO_FACTOR,
    Attribution,
    attribute_failure,
    build_failure_packet,
)
from babysteps.schemas import AttemptResult, FailurePacket, Intent, SceneState


def _intent() -> Intent:
    return Intent(
        goal_state="cube_at_target",
        object_motion="translate_+x",
        contact_region="minus_x_face",
        approach_direction="from_minus_x",
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_push",
    )


def _scene() -> SceneState:
    return SceneState(
        cube_xy=(0.0, 0.0),
        cube_z=0.02,
        goal_xy=(0.2, 0.0),
        tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0),
        blocked_sides=("from_minus_x",),
    )


def _attempt(**overrides) -> AttemptResult:
    base = dict(
        initial_obj_xy=(0.0, 0.0),
        final_obj_xy=(0.0, 0.0),
        goal_xy=(0.2, 0.0),
        reached_contact=False,
        object_moved=False,
        planner_failed=False,
        collision=False,
        grasp_slip=False,
        rollout_log_path=None,
        success=False,
    )
    base.update(overrides)
    return AttemptResult(**base)  # type: ignore[arg-type]


# ---------- Predicate precedence ---------------------------------------- #


def test_build_failure_packet_success_none():
    a = _attempt(reached_contact=True, object_moved=True,
                 final_obj_xy=(0.2, 0.0), success=True)
    fp = build_failure_packet(_intent(), a, _scene())
    assert fp.failure_predicate == "none"


def test_build_failure_packet_planner_failed_blocks_everything():
    """planner_failed beats every other signal: it means the skill never ran."""
    a = _attempt(planner_failed=True, reached_contact=False, object_moved=False)
    fp = build_failure_packet(_intent(), a, _scene())
    assert fp.failure_predicate == "approach_blocked"


def test_build_failure_packet_contact_failure():
    a = _attempt(reached_contact=False, object_moved=False, planner_failed=False)
    fp = build_failure_packet(_intent(), a, _scene())
    assert fp.failure_predicate == "contact_failure"


def test_build_failure_packet_no_motion():
    a = _attempt(reached_contact=True, object_moved=False)
    fp = build_failure_packet(_intent(), a, _scene())
    assert fp.failure_predicate == "no_motion"


def test_build_failure_packet_direction_error_when_opposite():
    """Cube moved in -x while goal is at +x → direction_alignment < 0."""
    a = _attempt(
        reached_contact=True, object_moved=True,
        final_obj_xy=(-0.1, 0.0),
    )
    fp = build_failure_packet(_intent(), a, _scene())
    assert fp.failure_predicate == "direction_error"
    assert fp.direction_alignment is not None
    assert fp.direction_alignment < 0


def test_build_failure_packet_goal_not_satisfied_when_short():
    a = _attempt(
        reached_contact=True, object_moved=True,
        final_obj_xy=(0.05, 0.0),       # in the right direction but didn't reach
    )
    fp = build_failure_packet(_intent(), a, _scene())
    assert fp.failure_predicate == "goal_not_satisfied"


def test_build_failure_packet_grasp_slip_beats_contact_failure():
    """Sub-project B: grasp_slip is more specific than the contact/motion
    predicates that follow it in the precedence rule. The gripper DID
    reach the cube; lift failed because grip was lost."""
    a = _attempt(
        reached_contact=True, object_moved=False, grasp_slip=True,
    )
    fp = build_failure_packet(_intent(), a, _scene())
    assert fp.failure_predicate == "grasp_slip"


def test_build_failure_packet_planner_failed_still_beats_grasp_slip():
    """planner_failed is the most specific predicate — even if grasp_slip
    is also flagged, the skill never ran."""
    a = _attempt(planner_failed=True, grasp_slip=True)
    fp = build_failure_packet(_intent(), a, _scene())
    assert fp.failure_predicate == "approach_blocked"


def test_build_failure_packet_carries_displacement():
    a = _attempt(reached_contact=True, object_moved=True, final_obj_xy=(0.1, 0.0))
    fp = build_failure_packet(_intent(), a, _scene())
    assert fp.object_displacement == pytest.approx(0.1)


def test_build_failure_packet_execution_trace_keys():
    a = _attempt(planner_failed=True)
    fp = build_failure_packet(_intent(), a, _scene())
    assert set(fp.execution_trace.keys()) == {
        "reached_contact", "object_moved", "collision",
        "planner_failed", "grasp_slip",
    }


def test_build_failure_packet_returns_failure_packet_type():
    a = _attempt(planner_failed=True)
    fp = build_failure_packet(_intent(), a, _scene())
    assert isinstance(fp, FailurePacket)


# ---------- Attribution rule table ------------------------------------- #


def test_failure_to_factor_table_covers_every_predicate_we_emit():
    expected_predicates = {
        "approach_blocked", "direction_error", "contact_failure",
        "no_motion", "goal_not_satisfied",
        "grasp_slip",       # Sub-project B
    }
    assert expected_predicates <= set(FAILURE_TO_FACTOR.keys())


def test_attribute_failure_grasp_slip_implicates_contact_region():
    """Sub-project B: grasp_slip → contact_region is wrong (the chosen
    gripper-axis is slip-prone). embodiment_mapping is also in `revise`
    for future operators; Stage-0's contact_substitution leaves it alone."""
    attr = attribute_failure(_make_fp("grasp_slip"))
    assert attr.semantic_failure is True
    assert attr.wrong_factor == "contact_region"
    assert "contact_region" in attr.revise
    assert "embodiment_mapping" in attr.revise
    assert "contact_region" not in attr.freeze


def _make_fp(pred: str) -> FailurePacket:
    return FailurePacket(
        chosen_intent=_intent(),
        execution_trace={
            "reached_contact": False, "object_moved": False,
            "collision": False, "planner_failed": True, "grasp_slip": False,
        },
        failure_predicate=pred,
        object_displacement=None,
        direction_alignment=None,
    )


def test_attribute_failure_approach_blocked():
    attr = attribute_failure(_make_fp("approach_blocked"))
    assert attr.semantic_failure is True
    assert attr.wrong_factor == "approach_direction"
    assert "approach_direction" in attr.revise
    # Frozen must NOT include the revised factor.
    assert "approach_direction" not in attr.freeze


def test_attribute_failure_direction_error():
    attr = attribute_failure(_make_fp("direction_error"))
    assert attr.wrong_factor == "approach_direction"


def test_attribute_failure_contact_failure():
    attr = attribute_failure(_make_fp("contact_failure"))
    assert attr.wrong_factor == "contact_region"


def test_attribute_failure_no_motion():
    attr = attribute_failure(_make_fp("no_motion"))
    assert attr.wrong_factor in {"approach_direction", "contact_region"}


def test_attribute_failure_goal_not_satisfied():
    attr = attribute_failure(_make_fp("goal_not_satisfied"))
    assert attr.wrong_factor == "goal_state"


def test_attribute_failure_none_is_not_semantic():
    attr = attribute_failure(_make_fp("none"))
    assert attr.semantic_failure is False
    assert attr.wrong_factor is None
    assert attr.revise == ()


def test_attribute_failure_returns_attribution_dataclass():
    attr = attribute_failure(_make_fp("approach_blocked"))
    assert isinstance(attr, Attribution)
    assert isinstance(attr.freeze, tuple)
    assert isinstance(attr.revise, tuple)


# ---------- Sub-project D: constraint_violation -------------------- #


def test_constraint_violation_predicate_fires_for_collision_no_motion():
    """When attempt.collision=True AND attempt.object_moved=False (and
    not planner_failed), the failure_predicate is 'constraint_violation'.
    This is more specific than the no_motion predicate which would
    otherwise fire."""
    from babysteps.failure import build_failure_packet
    from babysteps.schemas import AttemptResult, Intent, SceneState

    intent = Intent(
        goal_state="faucet_turned", object_motion="turn",
        contact_region="faucet_base", approach_direction="from_above",
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_turn",
    )
    scene = SceneState(
        cube_xy=(0.1, 0.0), cube_z=0.1, goal_xy=(0.1, 0.0),
        tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0),
        blocked_sides=(),
    )
    attempt = AttemptResult(
        initial_obj_xy=(0.1, 0.0), final_obj_xy=(0.1, 0.0),
        goal_xy=(0.1, 0.0),
        reached_contact=True, object_moved=False,
        planner_failed=False, collision=True, grasp_slip=False,
        rollout_log_path=None, success=False,
    )
    fp = build_failure_packet(intent, attempt, scene)
    assert fp.failure_predicate == "constraint_violation"


def test_attribute_failure_constraint_violation_to_constraint_region():
    """constraint_violation predicate maps to wrong_factor=
    'constraint_region' with revise=(constraint_region, contact_region)
    — the two-factor revision pair."""
    from babysteps.failure import attribute_failure
    from babysteps.schemas import FailurePacket, Intent

    intent = Intent(
        goal_state="faucet_turned", object_motion="turn",
        contact_region="faucet_base", approach_direction="from_above",
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_turn",
    )
    fp = FailurePacket(
        chosen_intent=intent,
        execution_trace={"reached_contact": True, "object_moved": False,
                         "collision": True, "planner_failed": False,
                         "grasp_slip": False},
        failure_predicate="constraint_violation",
        object_displacement=0.0, direction_alignment=None,
    )
    attribution = attribute_failure(fp)
    assert attribution.wrong_factor == "constraint_region"
    assert "constraint_region" in attribution.revise
    assert "contact_region" in attribution.revise
    assert "constraint_region" not in attribution.freeze
    assert "contact_region" not in attribution.freeze
