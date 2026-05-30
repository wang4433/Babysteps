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


def _intent(contact_region="handle_grip", constraint_region="none"):
    return Intent(
        goal_state="faucet_turned",
        object_motion="turn",
        contact_region=contact_region,
        approach_direction="from_above",
        constraint_region=constraint_region,
        embodiment_mapping="proxy_contact_to_franka_grasp_turn",
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


def _make_grasp_intent():
    from babysteps.schemas import Intent
    return Intent(
        goal_state="faucet_turned",
        object_motion="turn",
        contact_region="handle_grip",
        approach_direction="from_above",
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_grasp_turn",
    )


def _make_scene():
    from babysteps.schemas import SceneState
    return SceneState(
        cube_xy=(0.05, 0.02), cube_z=0.10, goal_xy=(0.05, 0.02),
        tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0),
        blocked_sides=(),
        extra={"handle_xy": (0.05, 0.02), "handle_z": 0.10,
               "target_joint_axis_xy": (0.0, 1.0)},
    )


def test_compile_grasp_turn_returns_grasp_mode_4_waypoints():
    from babysteps.skills.turn import compile_intent_to_turn_skill
    skill = compile_intent_to_turn_skill(_make_grasp_intent(), _make_scene())
    assert skill.mode == "grasp"
    assert len(skill.waypoints) == 4
    assert skill.gripper_schedule == (1.0, 1.0, -1.0, -1.0)
    assert skill.contact_region == "handle_grip"


def test_compile_deprecated_turn_token_falls_back_to_grasp():
    """Per spec §7: deprecated 'proxy_contact_to_franka_turn' compiles to
    the grasp variant for backward-compat with old diag scripts."""
    from babysteps.schemas import Intent
    from babysteps.skills.turn import compile_intent_to_turn_skill
    intent = Intent(
        goal_state="faucet_turned", object_motion="turn",
        contact_region="handle_grip", approach_direction="from_above",
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_turn",
    )
    skill = compile_intent_to_turn_skill(intent, _make_scene())
    assert skill.mode == "grasp"


def test_compile_unknown_embodiment_raises():
    from babysteps.schemas import Intent
    from babysteps.skills.turn import compile_intent_to_turn_skill
    intent = Intent(
        goal_state="faucet_turned", object_motion="turn",
        contact_region="handle_grip", approach_direction="from_above",
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_push",  # wrong embodiment for turn task
    )
    import pytest
    with pytest.raises(ValueError, match="unsupported embodiment_mapping"):
        compile_intent_to_turn_skill(intent, _make_scene())


def _make_poke_intent():
    from babysteps.schemas import Intent
    return Intent(
        goal_state="faucet_turned",
        object_motion="turn",
        contact_region="handle_grip",
        approach_direction="from_above",
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_poke_turn",
    )


def test_compile_poke_returns_poke_mode_3_waypoints():
    from babysteps.skills.turn import compile_intent_to_turn_skill
    skill = compile_intent_to_turn_skill(_make_poke_intent(), _make_scene())
    assert skill.mode == "poke"
    assert len(skill.waypoints) == 3
    assert skill.gripper_schedule == (-1.0, -1.0, -1.0)
    assert skill.contact_region == "handle_grip"
    assert skill.sign == +1


def test_compile_poke_sign_negative_flips_sweep_direction():
    import numpy as np
    from babysteps.skills.turn import compile_intent_to_turn_skill
    skill_pos = compile_intent_to_turn_skill(_make_poke_intent(), _make_scene(), sign=+1)
    skill_neg = compile_intent_to_turn_skill(_make_poke_intent(), _make_scene(), sign=-1)
    # pre_xy and post_xy should mirror through handle_xy.
    handle_xy = np.array(_make_scene().extra["handle_xy"])
    pre_pos  = skill_pos.waypoints[1, 0:2]
    pre_neg  = skill_neg.waypoints[1, 0:2]
    post_pos = skill_pos.waypoints[2, 0:2]
    post_neg = skill_neg.waypoints[2, 0:2]
    # pre points are on opposite sides of handle_xy.
    np.testing.assert_allclose(pre_pos - handle_xy, -(pre_neg - handle_xy), atol=1e-9)
    np.testing.assert_allclose(post_pos - handle_xy, -(post_neg - handle_xy), atol=1e-9)


def test_compile_poke_requires_handle_grip_contact_region():
    from babysteps.schemas import Intent
    from babysteps.skills.turn import compile_intent_to_turn_skill
    intent = Intent(
        goal_state="faucet_turned", object_motion="turn",
        contact_region="faucet_base",  # deprecated D token, still in whitelist
        approach_direction="from_above", constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_poke_turn",
    )
    import pytest
    with pytest.raises(ValueError, match="poke_turn requires contact_region='handle_grip'"):
        compile_intent_to_turn_skill(intent, _make_scene())


def test_compile_poke_z_above_handle_for_finger_dangle():
    from babysteps.skills.turn import compile_intent_to_turn_skill
    skill = compile_intent_to_turn_skill(_make_poke_intent(), _make_scene())
    # contact_z (waypoint 1 and 2) is handle_z + _POKE_HEIGHT_ABOVE_M.
    assert skill.waypoints[1, 2] == 0.10 + 0.04
    assert skill.waypoints[2, 2] == 0.10 + 0.04


# --- v1-geometry port: scene.extra['poke_*'] overrides ---------------------
#
# The GPU runner precomputes the v1 poke geometry (OBB handle centre + true
# circular tangent cross(joint_axis_3d, radius_3d)) and threads it through
# scene.extra. When those keys are present the compiler MUST use them instead
# of the target_link_pos + perp(axis_xy) heuristic. When absent (sim-free
# callers, snapshots) the compiler keeps its original behaviour — that
# back-compat is what every test above this point exercises.


def _make_poke_scene_with_precomputed_geometry(
    *, handle_xy=(0.05, 0.02), handle_z=0.10,
    poke_handle_xy=(0.12, -0.03), poke_handle_z=0.17,
    poke_tangent_xy=(1.0, 0.0), axis_xy=(0.0, 0.0),
):
    from babysteps.schemas import SceneState
    return SceneState(
        cube_xy=handle_xy, cube_z=handle_z, goal_xy=handle_xy,
        tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0),
        blocked_sides=(),
        extra={
            "handle_xy": handle_xy,
            "handle_z": handle_z,
            "target_joint_axis_xy": axis_xy,
            "poke_handle_xy": poke_handle_xy,
            "poke_handle_z": poke_handle_z,
            "poke_tangent_xy": poke_tangent_xy,
        },
    )


def test_compile_poke_uses_precomputed_handle_and_tangent():
    """With poke_* keys present the sweep is centred on poke_handle_xy at
    poke_handle_z, swept along poke_tangent_xy — NOT target_link_pos /
    perp(axis_xy). axis_xy=(0,0) would degenerate the heuristic to +y; the
    precomputed tangent (+x here) must win."""
    import numpy as np
    from babysteps.skills.turn import (
        _POKE_LATERAL_OFFSET_M, _POKE_SWEEP_DISTANCE_M, _POKE_HEIGHT_ABOVE_M,
        compile_intent_to_turn_skill,
    )
    scene = _make_poke_scene_with_precomputed_geometry(
        poke_handle_xy=(0.12, -0.03), poke_handle_z=0.17,
        poke_tangent_xy=(1.0, 0.0), axis_xy=(0.0, 0.0),
    )
    skill = compile_intent_to_turn_skill(_make_poke_intent(), scene, sign=+1)
    handle = np.array([0.12, -0.03])
    tangent = np.array([1.0, 0.0])
    # contact_z uses poke_handle_z, not handle_z.
    assert skill.waypoints[1, 2] == pytest.approx(0.17 + _POKE_HEIGHT_ABOVE_M)
    assert skill.waypoints[2, 2] == pytest.approx(0.17 + _POKE_HEIGHT_ABOVE_M)
    # pre_xy = handle - tangent*offset ; post_xy = handle + tangent*sweep.
    np.testing.assert_allclose(
        skill.waypoints[1, 0:2], handle - tangent * _POKE_LATERAL_OFFSET_M, atol=1e-9)
    np.testing.assert_allclose(
        skill.waypoints[2, 0:2], handle + tangent * _POKE_SWEEP_DISTANCE_M, atol=1e-9)
    # approach waypoint shares the pre xy.
    np.testing.assert_allclose(skill.waypoints[0, 0:2], skill.waypoints[1, 0:2], atol=1e-9)


def test_compile_poke_precomputed_tangent_not_renormalized():
    """The precomputed tangent is used verbatim (matches v1, which projects a
    3D-unit tangent to xy and does NOT re-normalise in 2D). A non-unit xy
    tangent therefore scales the lateral offset / sweep distance."""
    import numpy as np
    from babysteps.skills.turn import (
        _POKE_SWEEP_DISTANCE_M, compile_intent_to_turn_skill,
    )
    tangent = (0.6, 0.0)  # |xy| = 0.6 < 1, as for a tilted joint axis
    scene = _make_poke_scene_with_precomputed_geometry(
        poke_handle_xy=(0.0, 0.0), poke_tangent_xy=tangent,
    )
    skill = compile_intent_to_turn_skill(_make_poke_intent(), scene, sign=+1)
    np.testing.assert_allclose(
        skill.waypoints[2, 0:2],
        np.array(tangent) * _POKE_SWEEP_DISTANCE_M, atol=1e-9)


def test_compile_poke_precomputed_sign_negative_flips_sweep():
    import numpy as np
    from babysteps.skills.turn import compile_intent_to_turn_skill
    scene = _make_poke_scene_with_precomputed_geometry(poke_handle_xy=(0.12, -0.03))
    sp = compile_intent_to_turn_skill(_make_poke_intent(), scene, sign=+1)
    sn = compile_intent_to_turn_skill(_make_poke_intent(), scene, sign=-1)
    handle = np.array([0.12, -0.03])
    np.testing.assert_allclose(
        sp.waypoints[1, 0:2] - handle, -(sn.waypoints[1, 0:2] - handle), atol=1e-9)
    np.testing.assert_allclose(
        sp.waypoints[2, 0:2] - handle, -(sn.waypoints[2, 0:2] - handle), atol=1e-9)
