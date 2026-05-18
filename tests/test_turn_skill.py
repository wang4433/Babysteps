"""Tests for babysteps/skills/turn.py — pure TurnSkill geometry."""
from __future__ import annotations

import numpy as np
import pytest

from babysteps.schemas import Intent, SceneState


def _scene(handle_xy=(0.10, 0.0), base_xy=(0.05, 0.0), handle_z=0.10,
           axis_xy=(0.0, 1.0)):
    return SceneState(
        cube_xy=handle_xy,
        cube_z=handle_z,
        goal_xy=handle_xy,
        tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0),
        blocked_sides=(),
        extra={
            "handle_xy": handle_xy,
            "handle_z": handle_z,
            "faucet_base_xy": base_xy,
            "faucet_base_z": 0.0,
            "target_joint_axis_xy": axis_xy,
        },
    )


def _intent(contact_region="handle_grip", constraint_region="faucet_base_static"):
    return Intent(
        goal_state="faucet_turned",
        object_motion="turn",
        contact_region=contact_region,
        approach_direction="from_above",
        constraint_region=constraint_region,
        embodiment_mapping="proxy_contact_to_franka_turn",
    )


def test_compile_returns_turnskill_instance():
    from babysteps.skills.turn import TurnSkill, compile_intent_to_turn_skill
    skill = compile_intent_to_turn_skill(_intent(), _scene())
    assert isinstance(skill, TurnSkill)


def test_handle_grip_waypoints_target_handle_xy():
    from babysteps.skills.turn import compile_intent_to_turn_skill
    scene = _scene(handle_xy=(0.12, 0.04))
    skill = compile_intent_to_turn_skill(_intent("handle_grip"), scene)
    assert skill.waypoints.shape == (4, 7)
    # First three waypoints' xy == handle xy
    for i in range(3):
        assert skill.waypoints[i, 0] == pytest.approx(0.12)
        assert skill.waypoints[i, 1] == pytest.approx(0.04)


def test_faucet_base_waypoints_target_base_xy():
    from babysteps.skills.turn import compile_intent_to_turn_skill
    scene = _scene(base_xy=(0.06, -0.02))
    skill = compile_intent_to_turn_skill(_intent("faucet_base"), scene)
    assert skill.waypoints.shape == (4, 7)
    for i in range(3):
        assert skill.waypoints[i, 0] == pytest.approx(0.06)
        assert skill.waypoints[i, 1] == pytest.approx(-0.02)


def test_pull_waypoint_offset_perpendicular_to_joint_axis():
    """Waypoint 3 is contact_xy + perpendicular(axis_xy) * TURN_PULL_DISTANCE_M.
    Pulling along axis_xy would generate no torque on the handle; the
    perpendicular (90° CCW of axis_xy in xy) is the tangential direction
    that induces rotation."""
    from babysteps.skills.turn import (
        TURN_PULL_DISTANCE_M, compile_intent_to_turn_skill,
    )
    # axis along +x: perpendicular CCW = +y
    scene = _scene(handle_xy=(0.10, 0.05), axis_xy=(1.0, 0.0))
    skill = compile_intent_to_turn_skill(_intent("handle_grip"), scene)
    pull = skill.waypoints[3]
    assert pull[0] == pytest.approx(0.10)
    assert pull[1] == pytest.approx(0.05 + TURN_PULL_DISTANCE_M)


def test_pull_waypoint_default_direction_for_vertical_axis():
    """When axis_xy ~ 0 (vertical joint, common for faucets), the
    perpendicular is undefined and the skill falls back to +y."""
    from babysteps.skills.turn import (
        TURN_PULL_DISTANCE_M, compile_intent_to_turn_skill,
    )
    scene = _scene(handle_xy=(0.10, 0.05), axis_xy=(0.0, 0.0))
    skill = compile_intent_to_turn_skill(_intent("handle_grip"), scene)
    pull = skill.waypoints[3]
    assert pull[0] == pytest.approx(0.10)
    assert pull[1] == pytest.approx(0.05 + TURN_PULL_DISTANCE_M)


def test_compile_raises_on_unknown_contact_region():
    from babysteps.skills.turn import compile_intent_to_turn_skill
    scene = _scene()
    # minus_x_face is a cube-task contact_region; not handled by TurnSkill.
    bad_intent = Intent(
        goal_state="faucet_turned", object_motion="turn",
        contact_region="minus_x_face", approach_direction="from_above",
        constraint_region="faucet_base_static",
        embodiment_mapping="proxy_contact_to_franka_turn",
    )
    with pytest.raises(ValueError) as exc:
        compile_intent_to_turn_skill(bad_intent, scene)
    assert "minus_x_face" in str(exc.value)


def test_skill_exposes_contact_region_and_axis():
    from babysteps.skills.turn import compile_intent_to_turn_skill
    scene = _scene(axis_xy=(0.6, 0.8))
    skill = compile_intent_to_turn_skill(_intent("handle_grip"), scene)
    assert skill.contact_region == "handle_grip"
    assert skill.target_joint_axis_xy == pytest.approx((0.6, 0.8))


def test_turn_skill_has_mode_gripper_schedule_sign_fields():
    """TurnSkill must carry per-mode dispatch metadata so generic phase
    loops (runner/render) can iterate without hardcoded grasp assumptions."""
    import numpy as np
    from babysteps.skills.turn import TurnSkill
    wp = np.zeros((3, 7), dtype=np.float64)
    skill = TurnSkill(
        waypoints=wp,
        contact_region="handle_grip",
        target_joint_axis_xy=(0.0, 1.0),
        mode="poke",
        gripper_schedule=(-1.0, -1.0, -1.0),
        sign=+1,
    )
    assert skill.mode == "poke"
    assert skill.gripper_schedule == (-1.0, -1.0, -1.0)
    assert skill.sign == +1
    assert len(skill.gripper_schedule) == len(skill.waypoints)
