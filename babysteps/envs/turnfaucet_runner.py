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


def _compute_poke_geometry(env, obs):
    """Port of scripts/_diag_tf_poke5.py::compute_geometry — the empirically
    validated v1 poke geometry. Returns (handle_xy, handle_z, tangent_xy) or
    None when it cannot be computed (caller then keeps the target_link_pos +
    perp(axis_xy) fallback).

    Two things this recovers over the obs-only path:
      - handle position = the OBB *centre* of the switch-link handle mesh
        (target_link_pos is the link-frame origin, often offset from the
        graspable handle).
      - tangent = the true circular tangent cross(joint_axis_3d, radius_3d)
        projected to xy, where radius = handle_centre - joint_anchor. This is
        the direction the handle actually traces; perp(axis_xy) only matches
        it for a purely-horizontal joint axis and degenerates for vertical
        axes. The 3D-unit tangent is projected to xy WITHOUT 2D
        re-normalisation, matching v1 (a tilted axis shortens the xy sweep).

    GPU-side only: imports the mani_skill trimesh helper lazily and reads the
    physx switch-link mesh, so this never executes in the sim-free package.
    """
    switch_link = getattr(getattr(env, "unwrapped", env), "target_switch_link", None)
    # Only a real physx articulation link carries _objs + pose; the sim-free
    # stub env in tests/ has neither. Gate here so the sim-free render test
    # never imports mani_skill or touches the mesh path — it falls back to the
    # perp(axis_xy) heuristic, exactly as before this port.
    if (switch_link is None or not hasattr(switch_link, "_objs")
            or not hasattr(switch_link, "pose")):
        return None
    from mani_skill.utils.geometry.trimesh_utils import get_component_mesh

    comp = switch_link._objs[0]
    mesh_local = get_component_mesh(comp, to_world_frame=False)
    if mesh_local is None:
        return None
    obb_local = mesh_local.bounding_box_oriented
    link_T_batched = switch_link.pose.to_transformation_matrix()
    link_T = (
        link_T_batched[0].cpu().numpy() if hasattr(link_T_batched, "cpu")
        else np.asarray(link_T_batched)[0]
    )
    obb_T_world = link_T @ np.array(obb_local.primitive.transform)
    handle_center = obb_T_world[:3, 3]

    joint_anchor = _to_np(switch_link.joint.get_global_pose().p).astype(np.float64)
    joint_axis = _to_np(obs["extra"]["target_joint_axis"]).astype(np.float64)
    jn = float(np.linalg.norm(joint_axis))
    if jn < 1e-6:
        return None
    joint_axis = joint_axis / jn
    radius = handle_center - joint_anchor
    tangent_3d = np.cross(joint_axis, radius)
    tn = float(np.linalg.norm(tangent_3d))
    if tn < 1e-4:
        return None
    tangent_3d = tangent_3d / tn

    handle_xy = (float(handle_center[0]), float(handle_center[1]))
    handle_z = float(handle_center[2])
    tangent_xy = (float(tangent_3d[0]), float(tangent_3d[1]))
    return handle_xy, handle_z, tangent_xy


def _poke_geometry_extra(env, obs) -> dict:
    """scene.extra patch with the v1 poke geometry keys, or empty when the
    mesh/axis path is unavailable. Shared by the runner, the render module,
    and the Phase-3 diagnostic so all three feed the compiler identically."""
    geo = _compute_poke_geometry(env, obs)
    if geo is None:
        return {}
    handle_xy, handle_z, tangent_xy = geo
    return {
        "poke_handle_xy": handle_xy,
        "poke_handle_z": handle_z,
        "poke_tangent_xy": tangent_xy,
    }


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

    def __init__(self, render_mode: Optional[str] = None) -> None:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401
        kwargs: dict = dict(
            obs_mode="state_dict",
            control_mode="pd_ee_delta_pose",
            sim_backend="gpu",   # CPU IK is broken for this env
        )
        if render_mode is not None:
            # Optional: when set (e.g. "rgb_array") the env allocates a render
            # camera. Default None preserves the data-collection path
            # byte-for-byte (matches all previously-collected runs).
            kwargs["render_mode"] = render_mode
        self._env = gym.make("TurnFaucet-v1", **kwargs)
        self._last_seed: Optional[int] = None

    def reset(self, seed: int) -> SceneState:
        self._last_seed = int(seed)
        obs, _ = self._env.reset(seed=int(seed))
        tcp, handle_xyz, axis_xyz = _read_obs(obs)
        handle_xy = (float(handle_xyz[0]), float(handle_xyz[1]))
        handle_z = float(handle_xyz[2])
        axis_xy = (float(axis_xyz[0]), float(axis_xyz[1]))
        extra = {
            "handle_xy": handle_xy,
            "handle_z": handle_z,
            "target_joint_axis_xy": axis_xy,
        }
        # v1 poke geometry (OBB handle centre + true circular tangent). Used by
        # the poke compiler when present; grasp mode keeps the obs-only handle.
        extra.update(_poke_geometry_extra(self._env, obs))
        return SceneState(
            cube_xy=handle_xy,
            cube_z=handle_z,
            goal_xy=handle_xy,
            tcp_start_pose=tuple(float(v) for v in tcp),  # type: ignore[arg-type]
            blocked_sides=(),
            extra=extra,
        )

    def run(self, intent: Intent, scene: SceneState, *,
            rollout_log_path: Optional[Path] = None,
            rollout_seed: Optional[int] = None,  # see EnvRunner protocol docstring
            ) -> AttemptResult:
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
