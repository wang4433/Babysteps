"""Real ManiSkill StackCube-v1 env_runner.

Mirrors babysteps/envs/pickcube_runner.py's structure. The key
differences:

  - StackCube obs has cubeA_pose and cubeB_pose (no goal_pos).
  - The waypoint count is goal_state-dependent: 4 phases for
    cube_at_target (translate-and-drop), 5 for cubeA_on_cubeB
    (pick-and-place). The runner reads skill.waypoints.shape[0] and
    builds the gripper schedule accordingly.
  - Per-phase gripper schedule:
      4 phases (cube_at_target):  [OPEN, OPEN, CLOSED, OPEN]
                                   (release at translate-release)
      5 phases (cubeA_on_cubeB):  [OPEN, OPEN, CLOSED, CLOSED, OPEN]
                                   (release only at place_on)
  - No slip mechanism; no blocked_sides logic. Stage-0's controlled
    failure for StackCube is purely from the under-specified
    goal_state (handled by the skill compiler's branch selection).

Note on Gilbreth: requires a GPU/Vulkan node (same as PushCube/PickCube
runners)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from babysteps.schemas import AttemptResult, Intent, SceneState
from babysteps.skills.stack import (
    CUBE_HALF_SIZE,
    compile_intent_to_stack_skill,
)


# Phase-control constants — match PickCubeEnvRunner's PD calibration.
_POS_SCALE: float = 0.1
_PHASE_TOL_M: float = 0.015
_MAX_CONTROL_STEPS: int = 400         # matches PickCubeEnvRunner

_GRIPPER_OPEN: float = 1.0
_GRIPPER_CLOSED: float = -1.0


def _to_np(x):
    arr = x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)
    return arr[0] if arr.ndim == 2 else arr


def _raw_to_xyzw(raw_pose) -> np.ndarray:
    raw = np.asarray(raw_pose, dtype=np.float64)
    return np.concatenate([raw[0:3], raw[4:7], raw[3:4]])


def _read_obs(
    obs,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, float]:
    """(tcp_xyzw, cubeA_xy, cubeA_z, cubeB_xy, cubeB_z) from StackCube obs.

    StackCube-v1's obs.extra has cubeA_pose, cubeB_pose, tcp_pose. No
    goal_pos — the "goal" is implicit (cubeB.xy + cube_height)."""
    tcp = _raw_to_xyzw(_to_np(obs["extra"]["tcp_pose"]))
    cubeA_full = _to_np(obs["extra"]["cubeA_pose"])
    cubeA_xy = cubeA_full[0:2].astype(np.float64)
    cubeA_z = float(cubeA_full[2])
    cubeB_full = _to_np(obs["extra"]["cubeB_pose"])
    cubeB_xy = cubeB_full[0:2].astype(np.float64)
    cubeB_z = float(cubeB_full[2])
    return tcp, cubeA_xy, cubeA_z, cubeB_xy, cubeB_z


def _prop_action(
    tcp_xyzw: np.ndarray, target_xyz: np.ndarray, gripper_cmd: float,
) -> np.ndarray:
    pos_err = target_xyz - tcp_xyzw[0:3]
    action = np.zeros(7, dtype=np.float32)
    action[0:3] = np.clip(pos_err / _POS_SCALE, -1.0, 1.0).astype(np.float32)
    action[6] = np.float32(gripper_cmd)
    return action


def _gripper_at_cubeA(
    tcp: np.ndarray, cubeA_xy: np.ndarray, cubeA_z: float,
    *, threshold: float = 0.04,
) -> bool:
    dxy = float(np.linalg.norm(tcp[0:2] - np.asarray(cubeA_xy, dtype=np.float64)))
    dz = abs(float(tcp[2]) - float(cubeA_z))
    return dxy < threshold and dz < threshold


class StackCubeEnvRunner:
    """Real ManiSkill StackCube-v1 runner.

    Lazy-imports mani_skill on construction. Holds one gym env across
    multiple run(...) calls; each run internally resets to the captured
    seed before executing the compiled stack trajectory."""

    def __init__(self) -> None:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401 — registers StackCube-v1

        self._env = gym.make(
            "StackCube-v1",
            obs_mode="state_dict",
            control_mode="pd_ee_delta_pose",
            sim_backend="cpu",
        )
        self._last_seed: Optional[int] = None

    def reset(self, seed: int) -> SceneState:
        self._last_seed = int(seed)
        obs, _info = self._env.reset(seed=int(seed))
        tcp, cubeA_xy, cubeA_z, cubeB_xy, cubeB_z = _read_obs(obs)
        cubeB_top_z = cubeB_z + 2 * CUBE_HALF_SIZE
        return SceneState(
            cube_xy=(float(cubeA_xy[0]), float(cubeA_xy[1])),
            cube_z=cubeA_z,
            # Convenience: scene.goal_xy = cubeB.xy so existing
            # scene-reading callers (metrics computers, frame-by-frame
            # render utilities) work without StackCube-specific branches.
            goal_xy=(float(cubeB_xy[0]), float(cubeB_xy[1])),
            tcp_start_pose=tuple(float(v) for v in tcp),  # type: ignore[arg-type]
            blocked_sides=(),
            extra={
                "cubeB_xy": (float(cubeB_xy[0]), float(cubeB_xy[1])),
                "cubeB_z": cubeB_z,
                "cubeB_top_z": cubeB_top_z,
            },
        )

    def run(
        self,
        intent: Intent,
        scene: SceneState,
        *,
        rollout_log_path: Optional[Path] = None,
        rollout_seed: Optional[int] = None,
    ) -> AttemptResult:
        # rollout_seed: EnvRunner fresh-seed-per-attempt protocol. StackCube
        # resets from the episode seed (layout fixed) with a deterministic
        # controller; accepted for protocol conformance — see
        # PushCubeEnvRunner.run for the rationale.
        skill = compile_intent_to_stack_skill(intent, scene)
        # StackSkill never returns None — defensive only.

        if self._last_seed is None:
            raise RuntimeError("StackCubeEnvRunner.run called before reset()")
        obs, _info = self._env.reset(seed=int(self._last_seed))
        _tcp0, cubeA_xy0, _cubeA_z0, _cubeB_xy0, _cubeB_z0 = _read_obs(obs)
        initial_obj_xy = (float(cubeA_xy0[0]), float(cubeA_xy0[1]))

        targets: list[np.ndarray] = [
            np.asarray(wp[0:3], dtype=np.float64) for wp in skill.waypoints
        ]

        n_phases = len(targets)
        if n_phases == 4:
            # cube_at_target: [approach, descend, grasp, translate-release]
            phase_gripper = (
                _GRIPPER_OPEN, _GRIPPER_OPEN, _GRIPPER_CLOSED, _GRIPPER_OPEN,
            )
        elif n_phases == 5:
            # cubeA_on_cubeB: [approach, descend, grasp, lift, place_on]
            phase_gripper = (
                _GRIPPER_OPEN, _GRIPPER_OPEN, _GRIPPER_CLOSED,
                _GRIPPER_CLOSED, _GRIPPER_OPEN,
            )
        else:
            raise RuntimeError(
                f"StackCubeEnvRunner: unexpected waypoint count {n_phases}; "
                "expected 4 (cube_at_target) or 5 (cubeA_on_cubeB)"
            )

        trajectory: list[tuple[float, float]] = []
        phase_idx = 0
        reached_contact = False
        success = False

        for _step in range(_MAX_CONTROL_STEPS):
            tcp, cubeA_xy, _cubeA_z, _cubeB_xy, _cubeB_z = _read_obs(obs)
            trajectory.append((float(cubeA_xy[0]), float(cubeA_xy[1])))
            target = targets[phase_idx]
            if np.linalg.norm(target - tcp[0:3]) < _PHASE_TOL_M:
                phase_idx += 1
                if phase_idx >= n_phases:
                    break
                target = targets[phase_idx]
            # Contact heuristic: TCP near cubeA, post-approach.
            if phase_idx >= 1:
                reached_contact = reached_contact or _gripper_at_cubeA(
                    tcp, cubeA_xy, skill.cubeA_z,
                )
            action = _prop_action(tcp, target, phase_gripper[phase_idx])
            obs, _r, terminated, truncated, info = self._env.step(action)
            term = bool(_to_np(terminated).item()) if hasattr(terminated, "cpu") else bool(terminated)
            trunc = bool(_to_np(truncated).item()) if hasattr(truncated, "cpu") else bool(truncated)
            succ_field = info.get("success", False) if hasattr(info, "get") else False
            success = bool(_to_np(succ_field).item()) if hasattr(succ_field, "cpu") else bool(succ_field)
            if success or term or trunc:
                break

        _tcp_f, final_cubeA_xy, _, _, _ = _read_obs(obs)
        final_obj_xy = (float(final_cubeA_xy[0]), float(final_cubeA_xy[1]))
        trajectory.append(final_obj_xy)

        object_moved = (
            float(np.linalg.norm(
                np.asarray(final_obj_xy) - np.asarray(initial_obj_xy)
            )) > 0.005
        )

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
            collision=False,
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
