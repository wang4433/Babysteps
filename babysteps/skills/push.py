"""Push skill compiler — turns an Intent into either an executable PushSkill
or `None` (blocked).

This module is pure: no simulator, no I/O. It encodes two design properties:

1. **Feasibility check is semantic.** `compile_intent_to_push_skill` returns
   `None` iff `intent.approach_direction in scene.blocked_sides`. That `None`
   is the Stage-0 "controlled semantic failure" mechanism — it propagates as
   `planner_failed=True` in the AttemptResult downstream.

2. **Approach vs. push are decoupled.** `approach_direction` chooses the
   *route* the EE takes to reach the pre-contact pose (waypoint 0, the wide
   high standoff). `contact_region` chooses where the EE *touches* the cube
   and therefore the physical push direction (waypoints 2 and 3). This
   decoupling is what makes factor-local revision honest *and* visible:
   revising approach_direction alone reshapes only the approach path, while
   the push itself is unchanged.

Geometry constants are Pick4Pass-calibrated for PushCube-v1's
pd_ee_delta_pose normalization (pos_upper=0.1 m).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from babysteps.envs.scene import SceneState, approach_to_unit, face_to_push_unit
from babysteps.schemas import Intent

# Push geometry — Pick4Pass-calibrated for PushCube-v1.
CUBE_HALF_SIZE: float = 0.02
PRE_CONTACT_STANDOFF: float = 0.005
PUSH_TRAVEL_SCALE: float = 0.6
PUSH_TRAVEL_MAX_M: float = 0.15
# Wide-approach offset: how far from the cube centre the high "approach"
# waypoint sits along the approach_direction unit vector. 0.10 m matches the
# pd_ee_delta_pose pos_upper, so the prop controller reaches it in one step
# from far away — short detour for matching approach, visible arc for
# opposite approach.
APPROACH_STANDOFF_M: float = 0.10


@dataclass(frozen=True)
class PushSkill:
    """A compiled, ready-to-execute push.

    `waypoints` is (4, 7): rows are
      0. wide approach (high, on approach_direction side),
      1. pre-contact (high, on contact_region side),
      2. pre-contact (descended to push_z),
      3. push-end (descended, target of the push);
    columns are [x, y, z, qx, qy, qz, qw]. Quaternion is held at the TCP's
    starting orientation across all four rows (Pick4Pass pattern).
    """

    waypoints: np.ndarray
    cube_z: float
    contact_region: str
    # Stage-5 4-way fix: target gripper YAW (deg, relative to the resting EE
    # yaw) so the closed-gripper push face is perpendicular to the push
    # direction. 0 for x-axis pushes (the gripper rests face-along-x); 90 for
    # y-axis pushes. The runner P-controls action[5] toward this ONLY when
    # constructed with orient_control=True — default-off keeps the committed
    # +x data path byte-identical (a y-push without orient_control squirts the
    # cube ~85deg sideways; see job 10966492 / reports/stage5/diag_pushcube_ypush).
    push_yaw_deg: float = 0.0


def build_push_waypoints(scene: SceneState, intent: Intent) -> np.ndarray:
    """Pure geometry — four TCP waypoints for an open-loop push.

    Approach waypoint (wp[0]) is placed by `approach_to_unit(approach_direction)`.
    Push waypoints (wp[1..3]) are placed by `face_to_push_unit(contact_region)`
    — the physical push is identical for any approach.
    """
    cube_xy = np.asarray(scene.cube_xy, dtype=np.float64)
    goal_xy = np.asarray(scene.goal_xy, dtype=np.float64)
    tcp = np.asarray(scene.tcp_start_pose, dtype=np.float64)
    travel_z = float(tcp[2])
    push_z = float(scene.cube_z)
    push_unit = face_to_push_unit(intent.contact_region)
    approach_unit = approach_to_unit(intent.approach_direction)

    standoff = CUBE_HALF_SIZE + PRE_CONTACT_STANDOFF
    pre_contact_xy = cube_xy - push_unit * standoff
    approach_xy = cube_xy + approach_unit * APPROACH_STANDOFF_M
    cube_to_goal = float(np.linalg.norm(goal_xy - cube_xy))
    push_travel = min(PUSH_TRAVEL_SCALE * cube_to_goal, PUSH_TRAVEL_MAX_M)
    push_end_xy = cube_xy + push_unit * push_travel

    wp = np.zeros((4, 7), dtype=np.float64)
    wp[0, 0:2] = approach_xy
    wp[0, 2] = travel_z
    wp[1, 0:2] = pre_contact_xy
    wp[1, 2] = travel_z
    wp[2, 0:2] = pre_contact_xy
    wp[2, 2] = push_z
    wp[3, 0:2] = push_end_xy
    wp[3, 2] = push_z
    wp[:, 3:7] = tcp[3:7]
    return wp


def compile_intent_to_push_skill(intent: Intent, scene: SceneState) -> Optional[PushSkill]:
    """Returns a PushSkill ready for the env_runner, or None when the
    intent's approach_direction is blocked. None propagates as
    planner_failed=True downstream (failure_predicate "approach_blocked")."""
    if intent.approach_direction in scene.blocked_sides:
        return None
    push_unit = face_to_push_unit(intent.contact_region)
    # y-axis push needs the gripper yawed 90deg so its flat face is normal to
    # travel; x-axis push uses the resting (0deg) face. The face is symmetric
    # mod 180deg, so we are free to pick the rotation SIGN — and the Franka
    # wrist can complete -90 for a +y push but stalls partway on +90 (job
    # 10966514: +y reached only ~30deg of +90); -y completes +90. So rotate
    # toward the reachable side: +y -> -90, -y -> +90.
    if abs(push_unit[1]) > abs(push_unit[0]):
        push_yaw_deg = -90.0 if push_unit[1] > 0 else 90.0
    else:
        push_yaw_deg = 0.0
    return PushSkill(
        waypoints=build_push_waypoints(scene, intent),
        cube_z=float(scene.cube_z),
        contact_region=intent.contact_region,
        push_yaw_deg=push_yaw_deg,
    )
