"""Tests for babysteps.skills.pick (PickSkillCompiler + waypoint geometry).

Pure module — no simulator, no ManiSkill. PickCube's controlled Stage-0
failure (grasp_slip) is an execution-time decision in the env_runner; the
compiler always returns a skill (in contrast to PushSkill).
"""
from __future__ import annotations

import numpy as np
import pytest

from babysteps.schemas import CONTACT_REGIONS, Intent, SceneState
from babysteps.skills.pick import (
    DESCEND_CLEARANCE_M,
    PickSkill,
    build_pick_waypoints,
    compile_intent_to_pick_skill,
)


# ---------- Helpers ------------------------------------------------------ #


def _pick_intent(contact: str = "minus_x_face") -> Intent:
    return Intent(
        goal_state="cube_lifted_at_target",
        object_motion="lift_up",
        contact_region=contact,
        approach_direction="from_above",
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_grasp",
    )


def _scene(
    cube_xy: tuple[float, float] = (0.0, 0.0),
    goal_xy: tuple[float, float] = (0.1, 0.05),
    blocked: tuple[str, ...] = (),
) -> SceneState:
    return SceneState(
        cube_xy=cube_xy,
        cube_z=0.02,
        goal_xy=goal_xy,
        tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0),
        blocked_sides=blocked,
    )


# ---------- PickSkill compilation --------------------------------------- #


def test_compile_returns_pickskill_for_each_cardinal_face():
    """The compiler returns a PickSkill for every CONTACT_REGIONS value
    — the slip-blocked design lives in the env_runner, not here."""
    scene = _scene()
    for face in CONTACT_REGIONS:
        intent = _pick_intent(contact=face)
        skill = compile_intent_to_pick_skill(intent, scene)
        assert isinstance(skill, PickSkill)
        assert skill.contact_region == face
        assert skill.waypoints.shape == (4, 7)


def test_compile_never_returns_none_even_when_contact_blocked():
    """Stage-0 controlled-failure mechanism for PickCube is the env_runner's
    grasp_slip, not a compile-time None. This is the key contrast to
    PushSkill (compile_intent_to_push_skill returns None for blocked
    approach)."""
    intent = _pick_intent(contact="minus_x_face")
    scene = _scene(blocked=("minus_x_face",))   # demonstrated contact blocked
    skill = compile_intent_to_pick_skill(intent, scene)
    assert isinstance(skill, PickSkill)         # NOT None
    assert skill.contact_region == "minus_x_face"


def test_compile_preserves_cube_z():
    intent = _pick_intent()
    scene = _scene()
    skill = compile_intent_to_pick_skill(intent, scene)
    assert skill.cube_z == pytest.approx(scene.cube_z)


# ---------- build_pick_waypoints geometry ------------------------------- #


def test_waypoints_shape_and_quat_preserved():
    intent = _pick_intent()
    scene = _scene()
    wp = build_pick_waypoints(scene, intent)
    assert wp.shape == (4, 7)
    np.testing.assert_allclose(
        wp[:, 3:7], np.tile(scene.tcp_start_pose[3:7], (4, 1))
    )


def test_waypoint_0_approach_above_cube_at_travel_z():
    intent = _pick_intent()
    scene = _scene(cube_xy=(0.05, -0.03))
    wp = build_pick_waypoints(scene, intent)
    np.testing.assert_allclose(wp[0, 0:2], scene.cube_xy)
    assert wp[0, 2] == pytest.approx(scene.tcp_start_pose[2])


def test_waypoint_1_descend_above_cube_with_clearance():
    intent = _pick_intent()
    scene = _scene()
    wp = build_pick_waypoints(scene, intent)
    np.testing.assert_allclose(wp[1, 0:2], scene.cube_xy)
    assert wp[1, 2] == pytest.approx(scene.cube_z + DESCEND_CLEARANCE_M)


def test_waypoint_2_grasp_close_at_cube_z():
    intent = _pick_intent()
    scene = _scene()
    wp = build_pick_waypoints(scene, intent)
    np.testing.assert_allclose(wp[2, 0:2], scene.cube_xy)
    assert wp[2, 2] == pytest.approx(scene.cube_z)


def test_waypoint_3_lift_above_goal_at_travel_z():
    intent = _pick_intent()
    scene = _scene(goal_xy=(0.12, 0.04))
    wp = build_pick_waypoints(scene, intent)
    np.testing.assert_allclose(wp[3, 0:2], scene.goal_xy)
    assert wp[3, 2] == pytest.approx(scene.tcp_start_pose[2])


def test_waypoints_geometry_independent_of_contact_region():
    """The four waypoints do not depend on contact_region — the
    gripper-axis rotation that differentiates faces lives in the env_runner.
    This is what makes Stage-0 unit tests deterministic across faces."""
    scene = _scene()
    wp_x = build_pick_waypoints(scene, _pick_intent(contact="minus_x_face"))
    wp_y = build_pick_waypoints(scene, _pick_intent(contact="minus_y_face"))
    np.testing.assert_array_equal(wp_x, wp_y)


def test_waypoints_descend_then_grasp_then_lift_z_monotone():
    """Sanity: z descends from travel → cube+clearance → cube, then lifts
    back to travel. Catches accidental sign flips in the geometry."""
    intent = _pick_intent()
    scene = _scene()
    wp = build_pick_waypoints(scene, intent)
    assert wp[0, 2] > wp[1, 2] > wp[2, 2]
    assert wp[3, 2] > wp[2, 2]


# ---------- Validation -------------------------------------------------- #


def test_compile_raises_on_invalid_contact_region(monkeypatch):
    """Schema-bypassing callers get a clear error. (Intent's __post_init__
    normally blocks this — the test simulates a future caller that
    constructs an Intent-shaped object outside the schema.)"""
    intent = _pick_intent()
    bad = object.__new__(Intent)   # bypass __post_init__
    object.__setattr__(bad, "goal_state", intent.goal_state)
    object.__setattr__(bad, "object_motion", intent.object_motion)
    object.__setattr__(bad, "contact_region", "not_a_face")
    object.__setattr__(bad, "approach_direction", intent.approach_direction)
    object.__setattr__(bad, "constraint_region", intent.constraint_region)
    object.__setattr__(bad, "embodiment_mapping", intent.embodiment_mapping)
    scene = _scene()
    with pytest.raises(ValueError, match="contact_region"):
        compile_intent_to_pick_skill(bad, scene)
