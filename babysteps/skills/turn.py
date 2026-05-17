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


@dataclass(frozen=True)
class TurnSkill:
    """A compiled approach-grip-pull trajectory.

    waypoints is (4, 7). Columns are [x, y, z, qx, qy, qz, qw].
    contact_region is one of {"faucet_base", "handle_grip"} and is
    used by the env_runner for failure attribution.
    target_joint_axis_xy is the xy projection of the rotating joint's
    axis, used to direct the pull stroke.
    """
    waypoints: np.ndarray
    contact_region: str
    target_joint_axis_xy: tuple[float, float]


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
    pull_xy = contact_xy + axis_xy * TURN_PULL_DISTANCE_M
    tcp = np.asarray(scene.tcp_start_pose, dtype=np.float64)
    travel_z = float(tcp[2])

    wp = np.zeros((4, 7), dtype=np.float64)
    wp[0, 0:2] = contact_xy
    wp[0, 2] = travel_z
    wp[1, 0:2] = contact_xy
    wp[1, 2] = contact_z + DESCEND_CLEARANCE_M
    wp[2, 0:2] = contact_xy
    wp[2, 2] = contact_z
    wp[3, 0:2] = pull_xy
    wp[3, 2] = contact_z
    wp[:, 3:7] = tcp[3:7]

    return TurnSkill(
        waypoints=wp,
        contact_region=intent.contact_region,
        target_joint_axis_xy=(float(axis_xy[0]), float(axis_xy[1])),
    )
