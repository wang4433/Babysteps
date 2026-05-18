"""Turn skill compiler — Sub-project D (TurnFaucet) embodiment dispatch.

Pure geometry compiler. compile_intent_to_turn_skill dispatches on
intent.embodiment_mapping (per
docs/superpowers/specs/2026-05-18-stage0-turnfaucet-embodiment-design.md §7):

  - proxy_contact_to_franka_grasp_turn → _compile_grasp (4 waypoints,
    OPEN→CLOSED schedule, perpendicular tangent pull)
  - proxy_contact_to_franka_poke_turn  → _compile_poke (3 waypoints,
    closed-gripper lateral sweep, sign parameter for auto-sign retry)
  - proxy_contact_to_franka_turn       → _compile_grasp (deprecated
    token, kept in whitelist; falls through to grasp behavior)

ValueError fires when intent.embodiment_mapping is not one of these.
The waypoint count varies by mode — callers (env_runner, render module)
MUST iterate based on len(skill.waypoints) + skill.gripper_schedule,
not on a hardcoded shape.
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
    intent: Intent, scene: SceneState, sign: int = +1,
) -> TurnSkill:
    """Dispatches on intent.embodiment_mapping per spec §7.

    grasp_turn (and deprecated proxy_contact_to_franka_turn) → _compile_grasp
    poke_turn → _compile_poke(sign=sign)
    anything else → ValueError
    """
    if intent.embodiment_mapping == "proxy_contact_to_franka_grasp_turn":
        return _compile_grasp(intent, scene)
    if intent.embodiment_mapping == "proxy_contact_to_franka_poke_turn":
        return _compile_poke(intent, scene, sign=sign)
    if intent.embodiment_mapping == "proxy_contact_to_franka_turn":
        # Deprecated token still in whitelist — preserve behavioral parity for
        # old diag scripts (which may use contact_region="faucet_base").
        # Removal happens in the schema cleanup commit (T8).
        return _compile_grasp_compat(intent, scene)
    raise ValueError(
        f"compile_intent_to_turn_skill: unsupported embodiment_mapping "
        f"{intent.embodiment_mapping!r}"
    )


def _compile_grasp(intent: Intent, scene: SceneState) -> TurnSkill:
    """Grasp-mode TurnSkill (4 waypoints, OPEN→CLOSED schedule).

    Approach above handle, descend with clearance, close gripper at
    grip_z, pull tangentially. The geometry (perpendicular pull,
    GRIP_OFFSET_M, grip_z) was previously inline in
    compile_intent_to_turn_skill.
    """
    if intent.contact_region != "handle_grip":
        raise ValueError(
            f"_compile_grasp: contact_region must be 'handle_grip', "
            f"got {intent.contact_region!r}"
        )
    contact_xy = np.asarray(scene.extra["handle_xy"], dtype=np.float64)
    contact_z = float(scene.extra["handle_z"])
    axis_xy = np.asarray(scene.extra["target_joint_axis_xy"], dtype=np.float64)
    axis_norm = float(np.linalg.norm(axis_xy))
    if axis_norm < 1e-3:
        pull_dir_xy = np.array([0.0, 1.0])
    else:
        pull_dir_xy = np.array([-axis_xy[1], axis_xy[0]]) / axis_norm
    pull_xy = contact_xy + pull_dir_xy * TURN_PULL_DISTANCE_M
    tcp = np.asarray(scene.tcp_start_pose, dtype=np.float64)
    travel_z = float(tcp[2])
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
        contact_region="handle_grip",
        target_joint_axis_xy=(float(axis_xy[0]), float(axis_xy[1])),
        mode="grasp",
        gripper_schedule=(1.0, 1.0, -1.0, -1.0),
        sign=+1,
    )


def _compile_grasp_compat(intent: Intent, scene: SceneState) -> TurnSkill:
    """Grasp-mode TurnSkill with backward-compat contact_region dispatch.

    Used only by the deprecated 'proxy_contact_to_franka_turn' token so that
    old diag scripts which pass contact_region="faucet_base" continue to work
    until the schema cleanup commit (T8) migrates them to the canonical
    grasp_turn token + contact_region="handle_grip".

    Supports:
      contact_region="handle_grip" → handle_xy / handle_z
      contact_region="faucet_base" → faucet_base_xy / faucet_base_z

    Produces identical geometry and TurnSkill fields as _compile_grasp.
    """
    if intent.contact_region == "handle_grip":
        contact_xy = np.asarray(scene.extra["handle_xy"], dtype=np.float64)
        contact_z = float(scene.extra["handle_z"])
    elif intent.contact_region == "faucet_base":
        contact_xy = np.asarray(scene.extra["faucet_base_xy"], dtype=np.float64)
        contact_z = float(scene.extra["faucet_base_z"])
    else:
        raise ValueError(
            f"_compile_grasp_compat: contact_region must be one of "
            f"{{'faucet_base', 'handle_grip'}}, got {intent.contact_region!r}"
        )
    axis_xy = np.asarray(scene.extra["target_joint_axis_xy"], dtype=np.float64)
    axis_norm = float(np.linalg.norm(axis_xy))
    if axis_norm < 1e-3:
        pull_dir_xy = np.array([0.0, 1.0])
    else:
        pull_dir_xy = np.array([-axis_xy[1], axis_xy[0]]) / axis_norm
    pull_xy = contact_xy + pull_dir_xy * TURN_PULL_DISTANCE_M
    tcp = np.asarray(scene.tcp_start_pose, dtype=np.float64)
    travel_z = float(tcp[2])
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
        mode="grasp",
        gripper_schedule=(1.0, 1.0, -1.0, -1.0),
        sign=+1,
    )


def _compile_poke(intent: Intent, scene: SceneState, sign: int) -> TurnSkill:
    raise NotImplementedError("_compile_poke implemented in Task 4")
