"""Real ManiSkill TurnFaucet-v1 env_runner — embodiment_substitution version.

Generic phase loop driven by len(skill.waypoints) + skill.gripper_schedule.
No hardcoded 4-phase grasp assumptions. run() dispatches single-trial
(grasp_turn) vs two-trial auto-sign (poke_turn) per spec §8.

Requires partnet_mobility_faucet asset download (see CLAUDE.md)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from babysteps.schemas import AttemptResult, Intent, SceneState
from babysteps.skills.turn import compile_intent_to_turn_skill


_POS_SCALE: float = 0.1
_PHASE_TOL_M: float = 0.015
_GRASP_PHASE_TOL_M: float = 0.025
_GRIP_MIN_STEPS: int = 15
_MAX_CONTROL_STEPS: int = 400
_POKE_PROBE_STEPS: int = 80
_POKE_PROBE_MIN_PROGRESS: float = 0.4   # fraction of needed_delta required by probe


def _to_np(x):
    arr = x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)
    return arr[0] if arr.ndim == 2 else arr


def _safe_bool(x) -> bool:
    """Safe bool from a (possibly batched torch) tensor."""
    if hasattr(x, "cpu"):
        x = x.cpu().numpy()
    arr = np.asarray(x)
    return bool(arr.item() if arr.ndim > 0 else arr)


def _raw_to_xyzw(raw_pose) -> np.ndarray:
    raw = np.asarray(raw_pose, dtype=np.float64)
    return np.concatenate([raw[0:3], raw[4:7], raw[3:4]])


def _read_obs(obs):
    """(tcp_xyzw, handle_xyz, joint_axis_xyz)."""
    tcp = _raw_to_xyzw(_to_np(obs["extra"]["tcp_pose"]))
    handle_xyz = _to_np(obs["extra"]["target_link_pos"]).astype(np.float64)
    axis_xyz = _to_np(obs["extra"]["target_joint_axis"]).astype(np.float64)
    return tcp, handle_xyz, axis_xyz


def _read_faucet_qpos(env) -> float:
    """env.unwrapped.target_switch_link.joint.qpos as a python float."""
    return float(_to_np(env.unwrapped.target_switch_link.joint.qpos).item())


def _read_needed_delta(env) -> float:
    """target_angle - current qpos, both via env.unwrapped."""
    env_u = env.unwrapped
    target_angle = float(_to_np(env_u.target_angle).item())
    return target_angle - _read_faucet_qpos(env)


def _prop_action(tcp_xyzw, target_xyz, gripper_cmd):
    pos_err = target_xyz - tcp_xyzw[0:3]
    action = np.zeros(7, dtype=np.float32)
    action[0:3] = np.clip(pos_err / _POS_SCALE, -1.0, 1.0).astype(np.float32)
    action[6] = np.float32(gripper_cmd)
    return action


@dataclass(frozen=True)
class _TrialOutcome:
    success: bool
    reached_contact: bool
    object_moved: bool
    qpos_extremum_signed_progress: float
    initial_obj_xy: tuple[float, float]
    final_obj_xy: tuple[float, float]
    trajectory_xy: tuple[tuple[float, float], ...]


def _execute_skill(env, skill, *, seed, needed_delta, contact_xy, max_steps):
    """One full execution. Generic over len(skill.waypoints) and
    skill.gripper_schedule. Grasp-mode dwell at phase index 2 requires
    _GRIP_MIN_STEPS before advancing; poke mode has no dwell.

    Per spec §8.2: object_moved is derived from qpos delta (NOT handle xy
    delta) because target_link_pos sweeps the arc as the joint rotates
    but qpos is the direct signal of articulation motion.
    """
    obs, _ = env.reset(seed=int(seed))
    n_phases = len(skill.waypoints)
    assert len(skill.gripper_schedule) == n_phases, \
        "gripper_schedule length must match waypoints length"
    targets = [np.asarray(wp[0:3], dtype=np.float64) for wp in skill.waypoints]
    grip_phase = 2 if skill.mode == "grasp" and n_phases >= 3 else -1
    phase_tol = tuple(
        _GRASP_PHASE_TOL_M if i == grip_phase else _PHASE_TOL_M
        for i in range(n_phases)
    )

    tcp0, handle_xyz0, _ = _read_obs(obs)
    initial_xy = (float(handle_xyz0[0]), float(handle_xyz0[1]))
    initial_qpos = _read_faucet_qpos(env)

    trajectory: list[tuple[float, float]] = []
    phase_idx, steps_in_phase = 0, 0
    reached_contact, success = False, False
    qpos_extremum = initial_qpos

    for _ in range(max_steps):
        tcp, handle_xyz, _ = _read_obs(obs)
        trajectory.append((float(handle_xyz[0]), float(handle_xyz[1])))
        target = targets[phase_idx]
        reached = np.linalg.norm(target - tcp[0:3]) < phase_tol[phase_idx]
        advance = reached and (
            phase_idx != grip_phase or steps_in_phase >= _GRIP_MIN_STEPS
        )
        if advance:
            phase_idx += 1
            steps_in_phase = 0
            if phase_idx >= n_phases:
                break
            target = targets[phase_idx]
        else:
            steps_in_phase += 1
        if phase_idx >= 1 and np.linalg.norm(tcp[0:2] - contact_xy) < 0.04:
            reached_contact = True
        action = _prop_action(tcp, target, skill.gripper_schedule[phase_idx])
        obs, _r, terminated, truncated, info = env.step(action)
        qpos = _read_faucet_qpos(env)
        if needed_delta > 0:
            qpos_extremum = max(qpos_extremum, qpos)
        else:
            qpos_extremum = min(qpos_extremum, qpos)
        success = _safe_bool(info.get("success", False))
        if success or _safe_bool(terminated) or _safe_bool(truncated):
            break

    final_xy = trajectory[-1] if trajectory else initial_xy
    progress = (qpos_extremum - initial_qpos) / max(abs(needed_delta), 1e-6)
    object_moved = abs(qpos_extremum - initial_qpos) > 0.05  # rad

    return _TrialOutcome(
        success=success, reached_contact=reached_contact, object_moved=object_moved,
        qpos_extremum_signed_progress=progress,
        initial_obj_xy=initial_xy, final_obj_xy=final_xy,
        trajectory_xy=tuple(trajectory),
    )


class TurnFaucetEnvRunner:
    """Real ManiSkill TurnFaucet-v1 runner. See run() for dispatch logic."""

    def __init__(self) -> None:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401
        self._env = gym.make(
            "TurnFaucet-v1",
            obs_mode="state_dict",
            control_mode="pd_ee_delta_pose",
            sim_backend="gpu",   # CPU IK is broken for this env
        )
        self._last_seed: Optional[int] = None

    def reset(self, seed: int) -> SceneState:
        self._last_seed = int(seed)
        obs, _ = self._env.reset(seed=int(seed))
        tcp, handle_xyz, axis_xyz = _read_obs(obs)
        handle_xy = (float(handle_xyz[0]), float(handle_xyz[1]))
        handle_z = float(handle_xyz[2])
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
                "target_joint_axis_xy": axis_xy,
            },
        )

    def run(self, intent: Intent, scene: SceneState, *,
            rollout_log_path: Optional[Path] = None) -> AttemptResult:
        seed = self._last_seed
        if seed is None:
            raise RuntimeError("TurnFaucetEnvRunner.run called before reset()")
        contact_xy = np.asarray(scene.extra["handle_xy"], dtype=np.float64)
        needed_delta = _read_needed_delta(self._env)

        if intent.embodiment_mapping in (
            "proxy_contact_to_franka_grasp_turn",
            "proxy_contact_to_franka_turn",   # deprecated; same single-trial behavior
        ):
            skill = compile_intent_to_turn_skill(intent, scene)
            outcome = _execute_skill(
                self._env, skill, seed=seed, needed_delta=needed_delta,
                contact_xy=contact_xy, max_steps=_MAX_CONTROL_STEPS,
            )
            return self._outcome_to_attempt_result(outcome, scene, rollout_log_path)

        if intent.embodiment_mapping != "proxy_contact_to_franka_poke_turn":
            raise ValueError(
                f"TurnFaucetEnvRunner.run: unsupported embodiment_mapping "
                f"{intent.embodiment_mapping!r}"
            )

        # Poke: auto-sign two-trial. Each trial does its own env.reset(seed) so
        # the sign retry is a true counterfactual (identical faucet config).
        # Probe with sign=+1 is a truncated preview to decide direction.
        # If probe makes >= _POKE_PROBE_MIN_PROGRESS, rerun a full trial with
        # sign=+1 from fresh reset (captured trajectory reflects a complete
        # attempt). If probe falls short, run sign=-1 at full budget and pick
        # the better of the two by progress.
        skill_pos = compile_intent_to_turn_skill(intent, scene, sign=+1)
        probe = _execute_skill(
            self._env, skill_pos, seed=seed, needed_delta=needed_delta,
            contact_xy=contact_xy, max_steps=_POKE_PROBE_STEPS,
        )
        if probe.success:
            return self._outcome_to_attempt_result(probe, scene, rollout_log_path)
        if probe.qpos_extremum_signed_progress >= _POKE_PROBE_MIN_PROGRESS:
            full_pos = _execute_skill(
                self._env, skill_pos, seed=seed, needed_delta=needed_delta,
                contact_xy=contact_xy, max_steps=_MAX_CONTROL_STEPS,
            )
            return self._outcome_to_attempt_result(full_pos, scene, rollout_log_path)
        skill_neg = compile_intent_to_turn_skill(intent, scene, sign=-1)
        full_neg = _execute_skill(
            self._env, skill_neg, seed=seed, needed_delta=needed_delta,
            contact_xy=contact_xy, max_steps=_MAX_CONTROL_STEPS,
        )
        if (full_neg.success
                or full_neg.qpos_extremum_signed_progress
                    > probe.qpos_extremum_signed_progress):
            return self._outcome_to_attempt_result(full_neg, scene, rollout_log_path)
        full_pos = _execute_skill(
            self._env, skill_pos, seed=seed, needed_delta=needed_delta,
            contact_xy=contact_xy, max_steps=_MAX_CONTROL_STEPS,
        )
        return self._outcome_to_attempt_result(full_pos, scene, rollout_log_path)

    def _outcome_to_attempt_result(
        self, outcome: _TrialOutcome, scene: SceneState,
        rollout_log_path: Optional[Path],
    ) -> AttemptResult:
        if rollout_log_path is not None:
            rollout_log_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                rollout_log_path,
                trajectory_xy=np.asarray(outcome.trajectory_xy, dtype=np.float64),
                initial_obj_xy=np.asarray(outcome.initial_obj_xy, dtype=np.float64),
                final_obj_xy=np.asarray(outcome.final_obj_xy, dtype=np.float64),
                goal_xy=np.asarray(scene.goal_xy, dtype=np.float64),
            )
        return AttemptResult(
            initial_obj_xy=outcome.initial_obj_xy,
            final_obj_xy=outcome.final_obj_xy,
            goal_xy=scene.goal_xy,
            reached_contact=outcome.reached_contact,
            object_moved=outcome.object_moved,
            planner_failed=False,
            collision=False,
            grasp_slip=False,
            rollout_log_path=str(rollout_log_path) if rollout_log_path else None,
            success=outcome.success,
            trajectory_xy=outcome.trajectory_xy,
        )

    def close(self) -> None:
        try:
            self._env.close()
        except Exception:
            pass
