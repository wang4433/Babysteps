"""Push skill compiler — turns an Intent into either an executable PushSkill
or `None` (blocked).

This module is pure: no simulator, no I/O. It encodes two design properties:

1. **Feasibility check is semantic.** `compile_intent_to_push_skill` returns
   `None` iff `intent.approach_direction in scene.blocked_sides`. That `None`
   is the Stage-0 "controlled semantic failure" mechanism — it propagates as
   `planner_failed=True` in the AttemptResult downstream.

2. **Waypoint geometry depends only on contact_region (not approach_direction).**
   The TCP path is computed from the cube face that will be touched, with the
   physical push aimed straight at the goal. Decoupling these two factors is
   what makes factor-local revision honest: revising approach_direction alone
   changes the feasibility outcome without changing the physical push.

Geometry constants are Pick4Pass-calibrated for PushCube-v1's
pd_ee_delta_pose normalization (pos_upper=0.1 m).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from babysteps.envs.scene import SceneState, face_to_push_unit
from babysteps.schemas import Intent

# Push geometry — Pick4Pass-calibrated for PushCube-v1.
CUBE_HALF_SIZE: float = 0.02
PRE_CONTACT_STANDOFF: float = 0.005
PUSH_TRAVEL_SCALE: float = 0.6
PUSH_TRAVEL_MAX_M: float = 0.15


@dataclass(frozen=True)
class PushSkill:
    """A compiled, ready-to-execute push.

    `waypoints` is (3, 7): rows are pre-contact-high, descend, push-end;
    columns are [x, y, z, qx, qy, qz, qw]. Quaternion is held at the TCP's
    starting orientation across all three rows (Pick4Pass pattern).
    """

    waypoints: np.ndarray
    cube_z: float
    contact_region: str


def build_push_waypoints(scene: SceneState, intent: Intent) -> np.ndarray:
    """Pure geometry — three TCP waypoints for an open-loop push.

    Push direction is `face_to_push_unit(intent.contact_region)`. Push travel
    length is `min(PUSH_TRAVEL_SCALE * dist(cube, goal), PUSH_TRAVEL_MAX_M)`.
    """
    cube_xy = np.asarray(scene.cube_xy, dtype=np.float64)
    goal_xy = np.asarray(scene.goal_xy, dtype=np.float64)
    tcp = np.asarray(scene.tcp_start_pose, dtype=np.float64)
    travel_z = float(tcp[2])
    push_z = float(scene.cube_z)
    push_unit = face_to_push_unit(intent.contact_region)

    standoff = CUBE_HALF_SIZE + PRE_CONTACT_STANDOFF
    pre_contact_xy = cube_xy - push_unit * standoff
    cube_to_goal = float(np.linalg.norm(goal_xy - cube_xy))
    push_travel = min(PUSH_TRAVEL_SCALE * cube_to_goal, PUSH_TRAVEL_MAX_M)
    push_end_xy = cube_xy + push_unit * push_travel

    wp = np.zeros((3, 7), dtype=np.float64)
    wp[0, 0:2] = pre_contact_xy
    wp[0, 2] = travel_z
    wp[1, 0:2] = pre_contact_xy
    wp[1, 2] = push_z
    wp[2, 0:2] = push_end_xy
    wp[2, 2] = push_z
    wp[:, 3:7] = tcp[3:7]
    return wp


def compile_intent_to_push_skill(intent: Intent, scene: SceneState) -> Optional[PushSkill]:
    """Returns a PushSkill ready for the env_runner, or None when the
    intent's approach_direction is blocked. None propagates as
    planner_failed=True downstream (failure_predicate "approach_blocked")."""
    if intent.approach_direction in scene.blocked_sides:
        return None
    return PushSkill(
        waypoints=build_push_waypoints(scene, intent),
        cube_z=float(scene.cube_z),
        contact_region=intent.contact_region,
    )
