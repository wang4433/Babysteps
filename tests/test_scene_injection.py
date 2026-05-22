"""Pure injection geometry — sim-free."""
import numpy as np
import pytest

from babysteps.envs.scene import (
    cubeA_to_cubeB_motion,
    injected_cube_xy,
)


@pytest.mark.parametrize("motion,expected_sign", [
    ("translate_+x", (1.0, 0.0)),
    ("translate_-x", (-1.0, 0.0)),
    ("translate_+y", (0.0, 1.0)),
    ("translate_-y", (0.0, -1.0)),
])
def test_injected_cube_pushes_toward_goal_in_target_motion(motion, expected_sign):
    goal = (0.30, 0.10)
    dist = 0.12
    cube = injected_cube_xy(goal, dist, motion)
    vec = np.array(goal) - np.array(cube)
    assert np.linalg.norm(vec) == pytest.approx(dist)
    unit = vec / np.linalg.norm(vec)
    assert unit == pytest.approx(np.array(expected_sign), abs=1e-9)


def test_injected_cube_motion_roundtrips_through_goal_direction():
    from babysteps.envs.scene import goal_direction_to_motion
    goal = (0.25, -0.05)
    for motion in ("translate_+x", "translate_-x", "translate_+y", "translate_-y"):
        cube = injected_cube_xy(goal, 0.1, motion)
        vec = np.array(goal) - np.array(cube)
        assert goal_direction_to_motion(vec) == motion


def test_injected_cube_rejects_non_cardinal_motion():
    with pytest.raises(ValueError):
        injected_cube_xy((0.0, 0.0), 0.1, "lift_up")


def test_cubeA_to_cubeB_motion_matches_displacement_snap():
    assert cubeA_to_cubeB_motion((0.0, 0.0), (0.01, 0.20)) == "translate_+y"
    assert cubeA_to_cubeB_motion((0.0, 0.0), (-0.20, 0.01)) == "translate_-x"
