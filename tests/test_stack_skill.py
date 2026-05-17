"""Tests for babysteps/skills/stack.py — pure waypoint geometry."""
from __future__ import annotations

import numpy as np
import pytest

from babysteps.schemas import Intent, SceneState


def _scene(cubeA_xy=(0.0, 0.0), cubeB_xy=(0.10, 0.0), cubeB_z=0.02):
    return SceneState(
        cube_xy=cubeA_xy,
        cube_z=0.02,
        goal_xy=cubeB_xy,
        tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0),
        blocked_sides=(),
        extra={
            "cubeB_xy": cubeB_xy,
            "cubeB_z": cubeB_z,
            "cubeB_top_z": cubeB_z + 0.04,
        },
    )


def _intent(goal_state="cubeA_on_cubeB"):
    return Intent(
        goal_state=goal_state,
        object_motion="place_on" if goal_state == "cubeA_on_cubeB" else "translate_+x",
        contact_region="minus_x_face",
        approach_direction="from_above",
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_pick_and_place",
    )


def test_compile_returns_stackskill_instance():
    from babysteps.skills.stack import StackSkill, compile_intent_to_stack_skill
    skill = compile_intent_to_stack_skill(_intent(), _scene())
    assert isinstance(skill, StackSkill)


def test_cubeA_on_cubeB_has_five_waypoints():
    from babysteps.skills.stack import compile_intent_to_stack_skill
    skill = compile_intent_to_stack_skill(_intent("cubeA_on_cubeB"), _scene())
    assert skill.waypoints.shape == (5, 7)


def test_cube_at_target_has_four_waypoints():
    from babysteps.skills.stack import compile_intent_to_stack_skill
    skill = compile_intent_to_stack_skill(_intent("cube_at_target"), _scene())
    assert skill.waypoints.shape == (4, 7)


def test_cubeA_on_cubeB_final_waypoint_is_above_cubeB_top():
    """The place_on waypoint puts the TCP at cubeB_top_z + CUBE_HALF_SIZE +
    PLACE_CLEARANCE_M so cubeA settles on top after gripper release."""
    from babysteps.skills.stack import (
        CUBE_HALF_SIZE, PLACE_CLEARANCE_M, compile_intent_to_stack_skill,
    )
    scene = _scene(cubeB_xy=(0.12, 0.05), cubeB_z=0.02)
    skill = compile_intent_to_stack_skill(_intent("cubeA_on_cubeB"), scene)
    final = skill.waypoints[-1]
    expected_z = scene.extra["cubeB_top_z"] + CUBE_HALF_SIZE + PLACE_CLEARANCE_M
    assert final[0] == pytest.approx(0.12)
    assert final[1] == pytest.approx(0.05)
    assert final[2] == pytest.approx(expected_z)


def test_cube_at_target_final_waypoint_is_low_at_cubeB_xy():
    """The translate-release waypoint puts the TCP at cubeB.xy at low z
    (cubeA_z + DESCEND_CLEARANCE_M) — cubeA collides with cubeB and scatters."""
    from babysteps.skills.stack import (
        DESCEND_CLEARANCE_M, compile_intent_to_stack_skill,
    )
    scene = _scene(cubeB_xy=(0.12, 0.05), cubeB_z=0.02)
    skill = compile_intent_to_stack_skill(_intent("cube_at_target"), scene)
    final = skill.waypoints[-1]
    expected_z = scene.cube_z + DESCEND_CLEARANCE_M
    assert final[0] == pytest.approx(0.12)
    assert final[1] == pytest.approx(0.05)
    assert final[2] == pytest.approx(expected_z)


def test_first_waypoint_is_above_cubeA():
    """Both compile paths start with approach above cubeA at travel_z."""
    from babysteps.skills.stack import compile_intent_to_stack_skill
    scene = _scene(cubeA_xy=(-0.05, 0.03))
    for goal in ("cube_at_target", "cubeA_on_cubeB"):
        skill = compile_intent_to_stack_skill(_intent(goal), scene)
        wp0 = skill.waypoints[0]
        assert wp0[0] == pytest.approx(-0.05)
        assert wp0[1] == pytest.approx(0.03)
        assert wp0[2] == pytest.approx(0.25)   # travel_z from tcp_start_pose


def test_grasp_waypoint_is_at_cubeA_z():
    """Waypoint 2 (zero-indexed) is the grasp_close — at cubeA's actual z."""
    from babysteps.skills.stack import compile_intent_to_stack_skill
    scene = _scene()
    skill = compile_intent_to_stack_skill(_intent("cubeA_on_cubeB"), scene)
    grasp = skill.waypoints[2]
    assert grasp[2] == pytest.approx(scene.cube_z)


def test_quaternion_columns_come_from_tcp_start_pose():
    """Columns 3:7 of every waypoint hold the TCP's starting quaternion."""
    from babysteps.skills.stack import compile_intent_to_stack_skill
    scene = _scene()
    skill = compile_intent_to_stack_skill(_intent("cubeA_on_cubeB"), scene)
    tcp_q = np.asarray(scene.tcp_start_pose[3:7])
    for i in range(skill.waypoints.shape[0]):
        assert np.allclose(skill.waypoints[i, 3:7], tcp_q)


def test_compile_raises_on_unknown_goal_state():
    """Goal states outside the C-supported set raise ValueError."""
    from babysteps.skills.stack import compile_intent_to_stack_skill
    scene = _scene()
    # cube_lifted_at_target is a PickCube goal_state; not handled by stack skill.
    bad_intent = Intent(
        goal_state="cube_lifted_at_target",
        object_motion="lift_up",
        contact_region="minus_x_face",
        approach_direction="from_above",
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_grasp",
    )
    with pytest.raises(ValueError) as exc:
        compile_intent_to_stack_skill(bad_intent, scene)
    assert "cube_lifted_at_target" in str(exc.value)


def test_skill_exposes_cubeA_z_and_cubeB_top_z():
    """The compiled skill carries the geometry the env_runner needs."""
    from babysteps.skills.stack import compile_intent_to_stack_skill
    scene = _scene(cubeB_xy=(0.08, 0.0), cubeB_z=0.025)
    skill = compile_intent_to_stack_skill(_intent("cubeA_on_cubeB"), scene)
    assert skill.cubeA_z == pytest.approx(scene.cube_z)
    assert skill.cubeB_top_z == pytest.approx(0.025 + 0.04)
    assert skill.goal_state == "cubeA_on_cubeB"
