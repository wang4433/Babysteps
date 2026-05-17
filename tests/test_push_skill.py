"""Tests for babysteps.skills.push (PushSkillCompiler + waypoint geometry) and
babysteps.envs.scene (face/approach/motion helpers).

These are pure modules — no simulator, no ManiSkill. They are what makes
factor-local revision honest for PushCube: the physical push uses
contact_region, while approach_direction is a semantic feasibility filter.
"""
from __future__ import annotations

import numpy as np
import pytest

from babysteps.envs.scene import (
    OPPOSITE_APPROACH,
    approach_to_unit,
    direction_to_face,
    face_to_approach,
    face_to_push_unit,
    goal_direction_to_motion,
)
from babysteps.skills.push import (
    APPROACH_STANDOFF_M,
    CUBE_HALF_SIZE,
    PRE_CONTACT_STANDOFF,
    PUSH_TRAVEL_MAX_M,
    PUSH_TRAVEL_SCALE,
    PushSkill,
    build_push_waypoints,
    compile_intent_to_push_skill,
)
from babysteps.schemas import Intent, SceneState


# ---------- Scene helpers ------------------------------------------------ #


def test_direction_to_face_plus_x():
    # Goal is +x of the cube → contact the cube's -x face (push it toward +x).
    assert direction_to_face(np.array([1.0, 0.0])) == "minus_x_face"


def test_direction_to_face_minus_x():
    assert direction_to_face(np.array([-1.0, 0.0])) == "plus_x_face"


def test_direction_to_face_plus_y():
    assert direction_to_face(np.array([0.0, 1.0])) == "minus_y_face"


def test_direction_to_face_minus_y():
    assert direction_to_face(np.array([0.0, -1.0])) == "plus_y_face"


def test_direction_to_face_snaps_dominant_axis():
    # Goal is (1, 0.1) — dominant axis +x.
    assert direction_to_face(np.array([1.0, 0.1])) == "minus_x_face"


def test_face_to_approach_pairs():
    assert face_to_approach("minus_x_face") == "from_minus_x"
    assert face_to_approach("plus_x_face") == "from_plus_x"
    assert face_to_approach("minus_y_face") == "from_minus_y"
    assert face_to_approach("plus_y_face") == "from_plus_y"


def test_face_to_push_unit_directions():
    # Contact the -x face → push the cube toward +x.
    np.testing.assert_allclose(face_to_push_unit("minus_x_face"), [1.0, 0.0])
    np.testing.assert_allclose(face_to_push_unit("plus_x_face"), [-1.0, 0.0])
    np.testing.assert_allclose(face_to_push_unit("minus_y_face"), [0.0, 1.0])
    np.testing.assert_allclose(face_to_push_unit("plus_y_face"), [0.0, -1.0])


def test_goal_direction_to_motion():
    assert goal_direction_to_motion(np.array([0.2, 0.0])) == "translate_+x"
    assert goal_direction_to_motion(np.array([-0.2, 0.0])) == "translate_-x"
    assert goal_direction_to_motion(np.array([0.0, 0.2])) == "translate_+y"
    assert goal_direction_to_motion(np.array([0.0, -0.2])) == "translate_-y"
    # Snaps on dominant axis.
    assert goal_direction_to_motion(np.array([0.2, 0.05])) == "translate_+x"


def test_opposite_approach_pairs():
    assert OPPOSITE_APPROACH["from_minus_x"] == "from_plus_x"
    assert OPPOSITE_APPROACH["from_plus_x"] == "from_minus_x"
    assert OPPOSITE_APPROACH["from_minus_y"] == "from_plus_y"
    assert OPPOSITE_APPROACH["from_plus_y"] == "from_minus_y"


# ---------- PushSkill compilation --------------------------------------- #


def _correct_intent() -> Intent:
    return Intent(
        goal_state="cube_at_target",
        object_motion="translate_+x",
        contact_region="minus_x_face",
        approach_direction="from_minus_x",
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_push",
    )


def _scene(blocked: tuple[str, ...] = ()) -> SceneState:
    return SceneState(
        cube_xy=(0.0, 0.0),
        cube_z=0.02,
        goal_xy=(0.2, 0.0),
        tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0),
        blocked_sides=blocked,
    )


def test_compile_blocked_returns_none():
    intent = _correct_intent()
    scene = _scene(blocked=("from_minus_x",))
    assert compile_intent_to_push_skill(intent, scene) is None


def test_compile_unblocked_returns_pushskill():
    intent = _correct_intent()
    scene = _scene(blocked=())
    skill = compile_intent_to_push_skill(intent, scene)
    assert isinstance(skill, PushSkill)
    assert skill.waypoints.shape == (4, 7)
    assert skill.contact_region == "minus_x_face"


def test_compile_unblocked_with_irrelevant_block():
    """approach_direction != blocked entry → still compiles."""
    intent = _correct_intent()  # from_minus_x
    scene = _scene(blocked=("from_plus_y",))
    skill = compile_intent_to_push_skill(intent, scene)
    assert isinstance(skill, PushSkill)


# ---------- build_push_waypoints geometry ------------------------------- #


def test_waypoints_quat_preserved():
    intent = _correct_intent()
    scene = _scene()
    wp = build_push_waypoints(scene, intent)
    np.testing.assert_allclose(wp[:, 3:7], np.tile(scene.tcp_start_pose[3:7], (4, 1)))


def test_waypoints_approach_on_approach_direction_side_at_travel_height():
    intent = _correct_intent()    # approach_direction = from_minus_x
    scene = _scene()              # cube_xy=(0,0), tcp_z=0.25
    wp = build_push_waypoints(scene, intent)
    # Waypoint 0 (wide approach): on the -x side of the cube, at travel z.
    np.testing.assert_allclose(wp[0, 0], -APPROACH_STANDOFF_M)
    np.testing.assert_allclose(wp[0, 1], 0.0)
    assert wp[0, 2] == pytest.approx(scene.tcp_start_pose[2])


def test_waypoints_pre_contact_behind_cube_at_travel_height():
    intent = _correct_intent()    # push_unit = (+1, 0)
    scene = _scene()              # cube_xy=(0,0), tcp_z=0.25
    wp = build_push_waypoints(scene, intent)
    # Waypoint 1 (pre-contact above): behind cube along -push_unit, at travel z.
    expected_xy = -1.0 * (CUBE_HALF_SIZE + PRE_CONTACT_STANDOFF)
    np.testing.assert_allclose(wp[1, 0], expected_xy)
    np.testing.assert_allclose(wp[1, 1], 0.0)
    assert wp[1, 2] == pytest.approx(scene.tcp_start_pose[2])


def test_waypoints_descend_to_push_height():
    intent = _correct_intent()
    scene = _scene()
    wp = build_push_waypoints(scene, intent)
    # Waypoint 2 (descend): same xy as wp 1, z = cube_z.
    np.testing.assert_allclose(wp[2, 0:2], wp[1, 0:2])
    assert wp[2, 2] == pytest.approx(scene.cube_z)


def test_waypoints_push_distance_scaled_and_capped():
    intent = _correct_intent()
    scene = _scene()  # cube at (0,0), goal at (0.2,0) → dist 0.2
    wp = build_push_waypoints(scene, intent)
    expected_travel = min(PUSH_TRAVEL_SCALE * 0.2, PUSH_TRAVEL_MAX_M)
    # Waypoint 3 (push end): cube_xy + push_unit * expected_travel, at push z.
    np.testing.assert_allclose(wp[3, 0], expected_travel)
    np.testing.assert_allclose(wp[3, 1], 0.0)
    assert wp[3, 2] == pytest.approx(scene.cube_z)


def test_waypoints_push_phase_invariant_under_approach_change():
    """Factor-local invariant (relaxed): the physical push (descend + push-end,
    wp[2..]) depends only on contact_region. Changing approach_direction is
    allowed to reshape the approach (wp[0]) but must NOT touch the push."""
    scene = _scene()
    intent_a = _correct_intent()                                 # from_minus_x
    intent_b = Intent(
        goal_state="cube_at_target",
        object_motion="translate_+x",
        contact_region="minus_x_face",       # SAME contact
        approach_direction="from_above",     # DIFFERENT approach
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_push",
    )
    wp_a = build_push_waypoints(scene, intent_a)
    wp_b = build_push_waypoints(scene, intent_b)
    # Push waypoints (descend, push-end) are byte-identical.
    np.testing.assert_array_equal(wp_a[2:], wp_b[2:])
    # Pre-contact-high (wp[1]) also unchanged — it sits on the contact-region
    # side of the cube, not the approach side.
    np.testing.assert_array_equal(wp_a[1], wp_b[1])


def test_waypoints_approach_reflects_approach_direction():
    """The wide approach waypoint (wp[0]) is placed on the side named by
    approach_direction. Opposite approaches put wp[0] on opposite sides."""
    scene = _scene()
    intent_minus_x = _correct_intent()         # approach=from_minus_x
    intent_plus_x = Intent(
        goal_state="cube_at_target",
        object_motion="translate_+x",
        contact_region="minus_x_face",          # SAME contact
        approach_direction="from_plus_x",       # OPPOSITE approach
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_push",
    )
    wp_minus = build_push_waypoints(scene, intent_minus_x)
    wp_plus = build_push_waypoints(scene, intent_plus_x)
    # wp[0] x has opposite sign for opposite approaches, same magnitude.
    assert wp_minus[0, 0] < 0.0
    assert wp_plus[0, 0] > 0.0
    np.testing.assert_allclose(wp_minus[0, 0], -wp_plus[0, 0])
    # Push (wp[2..]) must still be identical — only approach changed.
    np.testing.assert_array_equal(wp_minus[2:], wp_plus[2:])


def test_approach_to_unit_pairs():
    np.testing.assert_allclose(approach_to_unit("from_minus_x"), [-1.0, 0.0])
    np.testing.assert_allclose(approach_to_unit("from_plus_x"),  [ 1.0, 0.0])
    np.testing.assert_allclose(approach_to_unit("from_minus_y"), [ 0.0, -1.0])
    np.testing.assert_allclose(approach_to_unit("from_plus_y"),  [ 0.0,  1.0])
    np.testing.assert_allclose(approach_to_unit("from_above"),   [ 0.0,  0.0])
