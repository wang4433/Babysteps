"""Stack skill compiler — turns a StackCube Intent into an executable StackSkill.

This module is pure: no simulator, no I/O. It encodes the Sub-project C
design (docs/superpowers/specs/2026-05-17-stage0-stackcube-c-design.md):

1. **Compile dispatches on intent.goal_state.** Two supported values:
   - `cube_at_target`  → 4-waypoint trajectory (pick + low-z release at
     cubeB.xy). Cube collides with cubeB and scatters — the deliberately
     under-specified Stage-0 demo outcome.
   - `cubeA_on_cubeB`  → 5-waypoint trajectory (pick + lift over cubeB +
     descend onto cubeB top + release). Successful stack.

2. **Geometry is symmetric across the two goal_states for phases 0-2.**
   The difference is in the final phase(s): translate-and-drop-low vs
   lift-and-descend-onto-cubeB. This keeps the geometric tests
   parameterizable and makes the failure narrative read directly off
   the skill's waypoint count.

3. **No slip mechanism.** Unlike PickSkill (which never returns None
   because slip is execution-time), StackSkill also never returns None;
   the failure is purely from the wrong waypoints. ValueError is raised
   only when intent.goal_state is outside the C-supported set.

Geometry constants are calibrated for ManiSkill StackCube-v1
(cube half-size 0.02; pd_ee_delta_pose normalization).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from babysteps.schemas import Intent, SceneState

# StackCube cube half-size — matches ManiSkill's self.cube_half_size.
CUBE_HALF_SIZE: float = 0.02

# Vertical gap above cubeA_z at the pre-close descend waypoint. Gives the
# PD controller a soft landing before grasp_close.
DESCEND_CLEARANCE_M: float = 0.02

# Vertical gap above cubeB's top at the place_on waypoint. Small enough
# that cubeA settles onto cubeB without overshooting.
PLACE_CLEARANCE_M: float = 0.005


@dataclass(frozen=True)
class StackSkill:
    """A compiled stack-and-place trajectory.

    `waypoints` is (N, 7) where N=4 for cube_at_target and N=5 for
    cubeA_on_cubeB. Columns are [x, y, z, qx, qy, qz, qw]. The
    quaternion is the TCP's starting orientation (cubeA is grasped with
    the default gripper rotation; the runner overrides only the gripper
    open/close command, not the orientation).

    `cubeA_z` and `cubeB_top_z` are exposed for the env_runner's success
    checks (e.g., confirming the lift cleared cubeB).
    """

    waypoints: np.ndarray
    cubeA_z: float
    cubeB_top_z: float
    goal_state: str


def _build_translate_waypoints(scene: SceneState) -> np.ndarray:
    """4 waypoints: approach above cubeA, descend, grasp, translate-release
    at cubeB.xy at low z. Used for the under-specified cube_at_target intent;
    cubeA collides with cubeB on release and scatters."""
    cubeA_xy = np.asarray(scene.cube_xy, dtype=np.float64)
    cubeB_xy = np.asarray(scene.extra["cubeB_xy"], dtype=np.float64)
    tcp = np.asarray(scene.tcp_start_pose, dtype=np.float64)
    travel_z = float(tcp[2])
    cubeA_z = float(scene.cube_z)

    wp = np.zeros((4, 7), dtype=np.float64)
    wp[0, 0:2] = cubeA_xy
    wp[0, 2] = travel_z
    wp[1, 0:2] = cubeA_xy
    wp[1, 2] = cubeA_z + DESCEND_CLEARANCE_M
    wp[2, 0:2] = cubeA_xy
    wp[2, 2] = cubeA_z
    wp[3, 0:2] = cubeB_xy
    wp[3, 2] = cubeA_z + DESCEND_CLEARANCE_M
    wp[:, 3:7] = tcp[3:7]
    return wp


def _build_place_on_waypoints(scene: SceneState) -> np.ndarray:
    """5 waypoints: approach above cubeA, descend, grasp, lift over cubeB,
    descend onto cubeB top. Used for the cubeA_on_cubeB intent; cubeA
    settles onto cubeB after the gripper releases."""
    cubeA_xy = np.asarray(scene.cube_xy, dtype=np.float64)
    cubeB_xy = np.asarray(scene.extra["cubeB_xy"], dtype=np.float64)
    cubeB_top_z = float(scene.extra["cubeB_top_z"])
    tcp = np.asarray(scene.tcp_start_pose, dtype=np.float64)
    travel_z = float(tcp[2])
    cubeA_z = float(scene.cube_z)

    wp = np.zeros((5, 7), dtype=np.float64)
    wp[0, 0:2] = cubeA_xy
    wp[0, 2] = travel_z
    wp[1, 0:2] = cubeA_xy
    wp[1, 2] = cubeA_z + DESCEND_CLEARANCE_M
    wp[2, 0:2] = cubeA_xy
    wp[2, 2] = cubeA_z
    wp[3, 0:2] = cubeB_xy
    wp[3, 2] = travel_z
    wp[4, 0:2] = cubeB_xy
    wp[4, 2] = cubeB_top_z + CUBE_HALF_SIZE + PLACE_CLEARANCE_M
    wp[:, 3:7] = tcp[3:7]
    return wp


def compile_intent_to_stack_skill(
    intent: Intent, scene: SceneState,
) -> StackSkill:
    """Returns a StackSkill ready for the env_runner.

    Dispatches on `intent.goal_state`:
      - cube_at_target  → 4-waypoint translate-and-release (under-specified)
      - cubeA_on_cubeB  → 5-waypoint pick-and-place (correct)

    Raises ValueError for any other goal_state (defensive; Intent's
    whitelist already enforces the vocabulary, so this fires only on
    callers passing a goal_state not in the C-supported subset
    e.g. cube_lifted_at_target from PickCube).
    """
    cubeB_top_z = float(scene.extra["cubeB_top_z"])
    if intent.goal_state == "cube_at_target":
        return StackSkill(
            waypoints=_build_translate_waypoints(scene),
            cubeA_z=float(scene.cube_z),
            cubeB_top_z=cubeB_top_z,
            goal_state="cube_at_target",
        )
    if intent.goal_state == "cubeA_on_cubeB":
        return StackSkill(
            waypoints=_build_place_on_waypoints(scene),
            cubeA_z=float(scene.cube_z),
            cubeB_top_z=cubeB_top_z,
            goal_state="cubeA_on_cubeB",
        )
    raise ValueError(
        f"compile_intent_to_stack_skill: goal_state must be one of "
        f"{{'cube_at_target', 'cubeA_on_cubeB'}}, got {intent.goal_state!r}"
    )
