"""Pick skill compiler — turns an Intent into an executable PickSkill.

This module is pure: no simulator, no I/O. It encodes the Sub-project B
design properties:

1. **Slip is execution-time.** Unlike PushSkill (which returns None when
   the demonstrated approach is blocked), PickSkill ALWAYS returns a
   skill. PickCube's controlled Stage-0 failure is `grasp_slip` — the
   gripper closes successfully, lifts, then loses grip. That decision
   is made by `PickCubeEnvRunner` from `scene.blocked_sides`, NOT here.

2. **Geometry is contact-region-independent.** The four waypoints
   describe a top-down grasp trajectory (approach high → descend →
   grasp close at cube_z → lift to goal_xy at travel_z). The
   gripper-axis rotation that differentiates `minus_x_face` from
   `minus_y_face` contacts is realised by the env_runner's gripper
   joint command, not by the EE position. PickSkill exposes
   `contact_region` for the runner to consult.

Geometry constants are Pick4Pass-style — calibrated for PickCube-v1's
pd_ee_delta_pose normalization (pos_upper=0.1 m).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from babysteps.schemas import CONTACT_REGIONS, Intent, SceneState

# Pick geometry — Stage-0 baseline.
DESCEND_CLEARANCE_M: float = 0.02
"""Vertical gap above cube_z at the pre-close descend waypoint. Gives the
PD controller a soft landing before grasp_close, so the gripper does not
overshoot the cube and bounce."""

LIFT_TCP_OFFSET_M: float = 0.015
"""Vertical offset between the TCP frame and the grasped cube's centre during
the lift (TCP sits ~15 mm above the cube centre — measured from real PickCube-v1
rollouts). The lift waypoint adds this to the 3D goal height so the grasped
cube's *centre* lands at goal_z (PickCube-v1 success needs the cube at goal_pos,
~0.025 tol), not the arm's travel height."""


@dataclass(frozen=True)
class PickSkill:
    """A compiled, ready-to-execute pick-and-lift.

    `waypoints` is (4, 7): rows are
      0. approach (high, above cube_xy),
      1. pre-contact descend (above cube_xy, cube_z + clearance),
      2. grasp_close (above cube_xy, cube_z — runner closes gripper here),
      3. lift to goal (above goal_xy, travel_z);
    columns are [x, y, z, qx, qy, qz, qw]. Quaternion is held at the TCP's
    starting orientation — the runner overrides per-axis gripper rotation
    from `contact_region`.

    `contact_region` is one of CONTACT_REGIONS and tells the runner which
    gripper-axis to align (minus_x_face / plus_x_face → x-aligned;
    minus_y_face / plus_y_face → y-aligned, i.e. 90° around z).
    """

    waypoints: np.ndarray
    cube_z: float
    contact_region: str


def build_pick_waypoints(scene: SceneState, intent: Intent) -> np.ndarray:
    """Pure geometry — four TCP waypoints for a top-down pick-and-lift.

    The trajectory is contact-region-independent: PickCube's Stage-0
    distinction between contact_region values is realised by the
    env_runner's gripper-axis rotation, not by EE position. This keeps
    the geometric tests deterministic across the four cardinal faces.
    """
    cube_xy = np.asarray(scene.cube_xy, dtype=np.float64)
    goal_xy = np.asarray(scene.goal_xy, dtype=np.float64)
    tcp = np.asarray(scene.tcp_start_pose, dtype=np.float64)
    travel_z = float(tcp[2])
    cube_z = float(scene.cube_z)

    wp = np.zeros((4, 7), dtype=np.float64)
    wp[0, 0:2] = cube_xy
    wp[0, 2] = travel_z
    wp[1, 0:2] = cube_xy
    wp[1, 2] = cube_z + DESCEND_CLEARANCE_M
    wp[2, 0:2] = cube_xy
    wp[2, 2] = cube_z
    wp[3, 0:2] = goal_xy
    # Lift target: if the real 3D goal height is known (extra['goal_z'], set by
    # the env_runner), aim so the grasped cube's centre lands at goal_z; the
    # TCP rides LIFT_TCP_OFFSET_M above the cube centre. Otherwise (sim-free /
    # fake-env callers) fall back to the arm's travel height.
    goal_z = scene.extra.get("goal_z")
    wp[3, 2] = (float(goal_z) + LIFT_TCP_OFFSET_M) if goal_z is not None else travel_z
    wp[:, 3:7] = tcp[3:7]
    return wp


def compile_intent_to_pick_skill(intent: Intent, scene: SceneState) -> PickSkill:
    """Returns a PickSkill ready for the env_runner.

    Always returns a skill — unlike PushSkill, the Stage-0 grasp_slip
    failure is detected at execution time by the env_runner consulting
    `scene.blocked_sides`, not at compile time.

    Raises ValueError if `intent.contact_region` is not one of the four
    cardinal faces (defensive check; Intent's whitelist already enforces
    this, so this fires only on malformed callers bypassing the schema).
    """
    if intent.contact_region not in CONTACT_REGIONS:
        raise ValueError(
            f"contact_region must be one of {sorted(CONTACT_REGIONS)}, "
            f"got {intent.contact_region!r}"
        )
    return PickSkill(
        waypoints=build_pick_waypoints(scene, intent),
        cube_z=float(scene.cube_z),
        contact_region=intent.contact_region,
    )
