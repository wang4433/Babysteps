"""Turn skill compiler — Sub-project D (TurnFaucet) approach-grip-pull.

Pure geometry compiler. Dispatches on intent.contact_region:
  - handle_grip  → waypoints target scene.extra["handle_xy"]
                   (the rotating switch link's centroid)
  - faucet_base  → waypoints target scene.extra["faucet_base_xy"]
                   (the static body's Stage-0 approximation)

The waypoint count is always 4: approach high above contact,
descend with clearance, grip (close gripper at contact_z), then
pull along target_joint_axis_xy for TURN_PULL_DISTANCE_M.

The skill never returns None — failure (when contact_region was
faucet_base) is detected at execution time by the env_runner setting
collision=True. ValueError fires only when intent.contact_region is
outside the D-supported subset (e.g., a cardinal cube face).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from babysteps.schemas import Intent, SceneState

DESCEND_CLEARANCE_M: float = 0.03
TURN_PULL_DISTANCE_M: float = 0.05
# TCP needs to sit ABOVE the handle's centroid so the gripper fingers
# wrap around the handle when they close (TCP at handle_z would put the
# fingertips below the handle). Empirically tuned to ~one finger length
# for the panda_wristcam.
GRIP_OFFSET_M: float = 0.02


@dataclass(frozen=True)
class TurnSkill:
    """A compiled approach trajectory for the TurnFaucet task.

    waypoints is (N, 7). Columns are [x, y, z, qx, qy, qz, qw]. N varies
    by mode: grasp uses 4 (approach, descend, grip, pull), poke uses 3
    (approach, descend-lateral, sweep). Runner/render phase loops MUST
    iterate based on len(waypoints) + gripper_schedule, never on a
    hardcoded 4-phase grasp shape.

    mode is "grasp" | "poke" and is dispatched on intent.embodiment_mapping
    by compile_intent_to_turn_skill (§7 of the spec).

    gripper_schedule[i] is the gripper command for waypoint i:
    +1.0 = open, -1.0 = closed.

    sign is poke-only (+1 or -1). For poke, the runner's auto-sign
    two-trial loop picks the winning sign per seed.
    """
    waypoints: np.ndarray
    contact_region: str
    target_joint_axis_xy: tuple[float, float]
    mode: str
    gripper_schedule: tuple[float, ...]
    sign: int = +1


def compile_intent_to_turn_skill(
    intent: Intent, scene: SceneState,
) -> TurnSkill:
    if intent.contact_region == "handle_grip":
        contact_xy = scene.extra["handle_xy"]
        contact_z = scene.extra["handle_z"]
    elif intent.contact_region == "faucet_base":
        contact_xy = scene.extra["faucet_base_xy"]
        contact_z = scene.extra["faucet_base_z"]
    else:
        raise ValueError(
            f"compile_intent_to_turn_skill: contact_region must be one of "
            f"{{'faucet_base', 'handle_grip'}}, got {intent.contact_region!r}"
        )

    contact_xy = np.asarray(contact_xy, dtype=np.float64)
    axis_xy = np.asarray(scene.extra["target_joint_axis_xy"],
                          dtype=np.float64)
    # Pull PERPENDICULAR to the joint axis (in the xy plane). The earlier
    # code pulled ALONG axis_xy, which is parallel to the rotation axis
    # and generates no torque on the handle. For axis along +z
    # (axis_xy ≈ 0, common for most faucets), the rotation is in the xy
    # plane and any in-plane direction induces rotation; default to +y so
    # the gripper sweeps forward (CCW around the faucet base when the
    # base is left of the handle).
    axis_norm = float(np.linalg.norm(axis_xy))
    if axis_norm < 1e-3:
        pull_dir_xy = np.array([0.0, 1.0])
    else:
        # 90° CCW rotation of the axis projection gives the tangent.
        pull_dir_xy = np.array([-axis_xy[1], axis_xy[0]]) / axis_norm
    pull_xy = contact_xy + pull_dir_xy * TURN_PULL_DISTANCE_M
    tcp = np.asarray(scene.tcp_start_pose, dtype=np.float64)
    travel_z = float(tcp[2])

    # Grip is at contact_z + GRIP_OFFSET_M so the gripper fingers wrap
    # around the contact point, not below it.
    grip_z = contact_z + GRIP_OFFSET_M
    wp = np.zeros((4, 7), dtype=np.float64)
    wp[0, 0:2] = contact_xy
    wp[0, 2] = travel_z
    wp[1, 0:2] = contact_xy
    wp[1, 2] = grip_z + DESCEND_CLEARANCE_M
    wp[2, 0:2] = contact_xy
    wp[2, 2] = grip_z
    wp[3, 0:2] = pull_xy
    wp[3, 2] = grip_z
    wp[:, 3:7] = tcp[3:7]

    return TurnSkill(
        waypoints=wp,
        contact_region=intent.contact_region,
        target_joint_axis_xy=(float(axis_xy[0]), float(axis_xy[1])),
    )
