"""Real ManiSkill TurnFaucet-v1 env_runner.

Mirrors babysteps/envs/stackcube_runner.py with these differences:
- Reads target_link_pos (handle xyz) and target_joint_axis (3D axis)
  from obs.extra. Faucet base xy approximated as (handle_xy - (0.05, 0))
  for Stage-0; the real body root has variable per-model geometry.
- 4-phase trajectory (approach, descend, grip, pull). Gripper schedule
  always [OPEN, OPEN, CLOSED, CLOSED].
- Reports collision=True (Stage-0 proxy for constraint_violation)
  when contact_region=faucet_base AND not info["success"]. Otherwise
  collision=False.

Requires partnet_mobility_faucet asset download (see CLAUDE.md)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from babysteps.schemas import AttemptResult, Intent, SceneState
from babysteps.skills.turn import compile_intent_to_turn_skill


_POS_SCALE: float = 0.1
_PHASE_TOL_M: float = 0.015
_MAX_CONTROL_STEPS: int = 400
_GRIPPER_OPEN: float = 1.0
_GRIPPER_CLOSED: float = -1.0


def _to_np(x):
    arr = x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)
    return arr[0] if arr.ndim == 2 else arr


def _raw_to_xyzw(raw_pose) -> np.ndarray:
    raw = np.asarray(raw_pose, dtype=np.float64)
    return np.concatenate([raw[0:3], raw[4:7], raw[3:4]])


def _read_obs(obs):
    """(tcp_xyzw, handle_xyz, joint_axis_xyz) from TurnFaucet obs."""
    tcp = _raw_to_xyzw(_to_np(obs["extra"]["tcp_pose"]))
    handle_xyz = _to_np(obs["extra"]["target_link_pos"]).astype(np.float64)
    axis_xyz = _to_np(obs["extra"]["target_joint_axis"]).astype(np.float64)
    return tcp, handle_xyz, axis_xyz


def _prop_action(tcp_xyzw, target_xyz, gripper_cmd):
    pos_err = target_xyz - tcp_xyzw[0:3]
    action = np.zeros(7, dtype=np.float32)
    action[0:3] = np.clip(pos_err / _POS_SCALE, -1.0, 1.0).astype(np.float32)
    action[6] = np.float32(gripper_cmd)
    return action


class TurnFaucetEnvRunner:
    """Real ManiSkill TurnFaucet-v1 runner."""

    def __init__(self) -> None:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401 — registers TurnFaucet-v1

        self._env = gym.make(
            "TurnFaucet-v1",
            obs_mode="state_dict",
            control_mode="pd_ee_delta_pose",
            sim_backend="cpu",
        )
        self._last_seed: Optional[int] = None

    def reset(self, seed: int) -> SceneState:
        self._last_seed = int(seed)
        obs, _info = self._env.reset(seed=int(seed))
        tcp, handle_xyz, axis_xyz = _read_obs(obs)
        handle_xy = (float(handle_xyz[0]), float(handle_xyz[1]))
        handle_z = float(handle_xyz[2])
        base_xy = (handle_xy[0] - 0.05, handle_xy[1])  # Stage-0 approximation
        base_z = 0.0
        axis_xy = (float(axis_xyz[0]), float(axis_xyz[1]))
        return SceneState(
            cube_xy=handle_xy,
            cube_z=handle_z,
            goal_xy=handle_xy,
            tcp_start_pose=tuple(float(v) for v in tcp),  # type: ignore[arg-type]
            blocked_sides=(),
            extra={
                "handle_xy": handle_xy,
                "handle_z": handle_z,
                "faucet_base_xy": base_xy,
                "faucet_base_z": base_z,
                "target_joint_axis_xy": axis_xy,
            },
        )

    def run(
        self,
        intent: Intent,
        scene: SceneState,
        *,
        rollout_log_path: Optional[Path] = None,
    ) -> AttemptResult:
        skill = compile_intent_to_turn_skill(intent, scene)
        if self._last_seed is None:
            raise RuntimeError("TurnFaucetEnvRunner.run called before reset()")
        obs, _info = self._env.reset(seed=int(self._last_seed))
        tcp0, handle_xyz0, _axis0 = _read_obs(obs)
        initial_obj_xy = (float(handle_xyz0[0]), float(handle_xyz0[1]))

        targets = [np.asarray(wp[0:3], dtype=np.float64) for wp in skill.waypoints]
        phase_gripper = (_GRIPPER_OPEN, _GRIPPER_OPEN, _GRIPPER_CLOSED, _GRIPPER_CLOSED)

        trajectory: list[tuple[float, float]] = []
        phase_idx = 0
        reached_contact = False
        success = False
        for _step in range(_MAX_CONTROL_STEPS):
            tcp, handle_xyz, _axis = _read_obs(obs)
            trajectory.append((float(handle_xyz[0]), float(handle_xyz[1])))
            target = targets[phase_idx]
            if np.linalg.norm(target - tcp[0:3]) < _PHASE_TOL_M:
                phase_idx += 1
                if phase_idx >= len(targets):
                    break
                target = targets[phase_idx]
            # Contact heuristic: TCP near the chosen contact point.
            if phase_idx >= 1:
                cxy = (np.asarray(scene.extra["handle_xy"], dtype=np.float64)
                       if intent.contact_region == "handle_grip"
                       else np.asarray(scene.extra["faucet_base_xy"], dtype=np.float64))
                dxy = float(np.linalg.norm(tcp[0:2] - cxy))
                if dxy < 0.04:
                    reached_contact = True
            action = _prop_action(tcp, target, phase_gripper[phase_idx])
            obs, _r, terminated, truncated, info = self._env.step(action)
            term = bool(_to_np(terminated).item()) if hasattr(terminated, "cpu") else bool(terminated)
            trunc = bool(_to_np(truncated).item()) if hasattr(truncated, "cpu") else bool(truncated)
            succ_field = info.get("success", False) if hasattr(info, "get") else False
            success = bool(_to_np(succ_field).item()) if hasattr(succ_field, "cpu") else bool(succ_field)
            if success or term or trunc:
                break

        _tcp_f, handle_xyz_f, _axis_f = _read_obs(obs)
        final_obj_xy = (float(handle_xyz_f[0]), float(handle_xyz_f[1]))
        trajectory.append(final_obj_xy)

        object_moved = (
            float(np.linalg.norm(np.asarray(final_obj_xy) - np.asarray(initial_obj_xy)))
            > 0.005
        )

        # Stage-0 constraint_violation proxy:
        # if contact_region was faucet_base AND faucet didn't rotate,
        # mark collision=True so build_failure_packet emits the
        # constraint_violation predicate.
        collision = (intent.contact_region == "faucet_base" and not success)

        if rollout_log_path is not None:
            rollout_log_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                rollout_log_path,
                trajectory_xy=np.asarray(trajectory, dtype=np.float64),
                initial_obj_xy=np.asarray(initial_obj_xy, dtype=np.float64),
                final_obj_xy=np.asarray(final_obj_xy, dtype=np.float64),
                goal_xy=np.asarray(scene.goal_xy, dtype=np.float64),
            )

        return AttemptResult(
            initial_obj_xy=initial_obj_xy,
            final_obj_xy=final_obj_xy,
            goal_xy=scene.goal_xy,
            reached_contact=bool(reached_contact),
            object_moved=bool(object_moved),
            planner_failed=False,
            collision=bool(collision),
            grasp_slip=False,
            rollout_log_path=str(rollout_log_path) if rollout_log_path else None,
            success=bool(success),
            trajectory_xy=tuple(trajectory),
        )

    def close(self) -> None:
        try:
            self._env.close()
        except Exception:
            pass
