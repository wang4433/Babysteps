"""Real ManiSkill PickCube-v1 env_runner.

Mirrors `babysteps/envs/pushcube_runner.py`'s structure: open-loop
proportional EE control, 4 phases this time (approach → descend →
grasp_close → lift). The 7-dim pd_ee_delta_pose action carries the
gripper command in `action[6]`: +1 = open, -1 = closed.

Stage-0 controlled-failure mechanism:
  At the start of the lift phase, the runner checks
  `intent.contact_region in scene.blocked_sides`. If True, the gripper
  is sent the OPEN command (`action[6] = +1`) instead of CLOSED during
  the lift, so the cube falls back to the table. The resulting
  AttemptResult carries `grasp_slip=True, reached_contact=True,
  object_moved=False, success=False`, which the rule-based attribution
  maps to `wrong_factor = "contact_region"`.

Note on Gilbreth: requires a GPU/Vulkan node (same as PushCubeEnvRunner)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from babysteps.skills.pick import compile_intent_to_pick_skill
from babysteps.schemas import AttemptResult, Intent, SceneState

# Phase-control constants — match PushCubeEnvRunner's PD calibration so the
# two runners feel the same to the orchestration code.
_POS_SCALE: float = 0.1
_PHASE_TOL_M: float = 0.015
_MAX_CONTROL_STEPS: int = 400         # +100 vs PushCube to allow the lift

# PickCube-v1's default TimeLimit truncates at 50 steps — too short once the
# settle dwell below is added. Override at gym.make so the trajectory finishes.
_MAX_EPISODE_STEPS: int = 200

# Hold the cube at the 3D goal with the gripper closed so the arm goes static
# (PickCube-v1 success = cube at goal_pos AND robot static) — a real settle,
# not widened tolerances.
_SETTLE_DWELL_STEPS: int = 25

_GRIPPER_OPEN: float = 1.0
_GRIPPER_CLOSED: float = -1.0


def _to_np(x):
    arr = x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)
    return arr[0] if arr.ndim == 2 else arr


def _raw_to_xyzw(raw_pose) -> np.ndarray:
    raw = np.asarray(raw_pose, dtype=np.float64)
    return np.concatenate([raw[0:3], raw[4:7], raw[3:4]])


def _read_obs(obs) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return (tcp_xyzw, cube_xy, goal_xy, cube_z) from a PickCube obs.

    PickCube-v1's obs has the same keys as PushCube-v1 for the relevant
    fields (tcp_pose, obj_pose, goal_pos). The 3D goal is read as 2D xy
    here because Stage-0 only exposes 2D in SceneState.goal_xy; the
    runner adds a fixed lift_z for the lift waypoint."""
    tcp = _raw_to_xyzw(_to_np(obs["extra"]["tcp_pose"]))
    cube_full = _to_np(obs["extra"]["obj_pose"])
    cube_xy = cube_full[0:2].astype(np.float64)
    cube_z = float(cube_full[2])
    goal_xy = _to_np(obs["extra"]["goal_pos"])[0:2].astype(np.float64)
    return tcp, cube_xy, goal_xy, cube_z


def _prop_action(
    tcp_xyzw: np.ndarray, target_xyz: np.ndarray, gripper_cmd: float,
) -> np.ndarray:
    """Proportional normalized action toward target_xyz with explicit
    gripper command (unlike PushCube where the gripper is held closed)."""
    pos_err = target_xyz - tcp_xyzw[0:3]
    action = np.zeros(7, dtype=np.float32)
    action[0:3] = np.clip(pos_err / _POS_SCALE, -1.0, 1.0).astype(np.float32)
    action[6] = np.float32(gripper_cmd)
    return action


class PickCubeEnvRunner:
    """Real ManiSkill PickCube-v1 runner.

    Lazy-imports `mani_skill.envs` on construction. Holds one gym env
    across multiple `run(...)` calls; each `run` internally resets to
    the captured seed before executing the compiled pick.
    """

    def __init__(self, render_mode: Optional[str] = None) -> None:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401 — registers PickCube-v1

        kwargs: dict = dict(
            obs_mode="state_dict",
            control_mode="pd_ee_delta_pose",
            sim_backend="cpu",
            max_episode_steps=_MAX_EPISODE_STEPS,
        )
        if render_mode is not None:
            kwargs["render_mode"] = render_mode
        self._env = gym.make("PickCube-v1", **kwargs)
        self._last_seed: Optional[int] = None

    def reset(self, seed: int) -> SceneState:
        self._last_seed = int(seed)
        obs, _info = self._env.reset(seed=int(seed))
        tcp, cube_xy, goal_xy, cube_z = _read_obs(obs)
        # PickCube-v1's goal is a 3D point in the air; SceneState.goal_xy keeps
        # only the xy, so expose the goal height via extra['goal_z'] for the
        # pick skill's lift waypoint (the cube must reach the 3D goal_pos).
        goal_z = float(_to_np(obs["extra"]["goal_pos"])[2])
        return SceneState(
            cube_xy=(float(cube_xy[0]), float(cube_xy[1])),
            cube_z=cube_z,
            goal_xy=(float(goal_xy[0]), float(goal_xy[1])),
            tcp_start_pose=tuple(float(v) for v in tcp),  # type: ignore[arg-type]
            blocked_sides=(),
            extra={"goal_z": goal_z},
        )

    def run(
        self,
        intent: Intent,
        scene: SceneState,
        *,
        rollout_log_path: Optional[Path] = None,
        rollout_seed: Optional[int] = None,  # see EnvRunner protocol docstring
    ) -> AttemptResult:
        skill = compile_intent_to_pick_skill(intent, scene)
        # PickSkill never returns None — the Stage-0 controlled failure
        # is execution-time slip, not compile-time block.

        if self._last_seed is None:
            raise RuntimeError("PickCubeEnvRunner.run called before reset()")
        obs, _info = self._env.reset(seed=int(self._last_seed))
        _tcp0, cube_xy0, _goal0, _cube_z0 = _read_obs(obs)
        initial_obj_xy = (float(cube_xy0[0]), float(cube_xy0[1]))
        initial_cube_z = _cube_z0

        # Four waypoint phase targets (xyz only).
        targets: list[np.ndarray] = [
            np.asarray(wp[0:3], dtype=np.float64) for wp in skill.waypoints
        ]

        # Per-phase gripper command schedule:
        #   0 approach        → open
        #   1 descend         → open
        #   2 grasp_close     → closed
        #   3 lift            → closed UNLESS contact_region in blocked_sides
        #                       (Stage-0 controlled grasp_slip)
        slip = intent.contact_region in scene.blocked_sides
        lift_gripper = _GRIPPER_OPEN if slip else _GRIPPER_CLOSED
        phase_gripper = [
            _GRIPPER_OPEN, _GRIPPER_OPEN, _GRIPPER_CLOSED, lift_gripper,
        ]

        trajectory: list[tuple[float, float]] = []
        phase_idx = 0
        dwelling = False
        dwell_remaining = 0
        # Only the final lift phase settles — hold at the 3D goal with the
        # gripper closed so the arm goes static. Earlier phases advance on
        # arrival.
        dwell_len = [0, 0, 0, _SETTLE_DWELL_STEPS]
        reached_contact = False
        success = False

        for _step in range(_MAX_CONTROL_STEPS):
            tcp, cube_xy, _, cube_z_now = _read_obs(obs)
            trajectory.append((float(cube_xy[0]), float(cube_xy[1])))
            target = targets[phase_idx]
            if not dwelling and np.linalg.norm(target - tcp[0:3]) < _PHASE_TOL_M:
                if dwell_len[phase_idx] > 0:
                    dwelling = True
                    dwell_remaining = dwell_len[phase_idx]
                else:
                    phase_idx += 1
                    if phase_idx >= len(targets):
                        break
                    target = targets[phase_idx]
            # Contact heuristic: TCP near cube AND not yet in lift phase.
            if phase_idx >= 1:
                reached_contact = reached_contact or _gripper_at_cube(
                    tcp, cube_xy, skill.cube_z,
                )
            action = _prop_action(tcp, target, phase_gripper[phase_idx])
            obs, _r, terminated, truncated, info = self._env.step(action)
            term = bool(_to_np(terminated).item()) if hasattr(terminated, "cpu") else bool(terminated)
            trunc = bool(_to_np(truncated).item()) if hasattr(truncated, "cpu") else bool(truncated)
            succ_field = info.get("success", False) if hasattr(info, "get") else False
            success = bool(_to_np(succ_field).item()) if hasattr(succ_field, "cpu") else bool(succ_field)
            if dwelling:
                dwell_remaining -= 1
                if dwell_remaining <= 0:
                    dwelling = False
                    phase_idx += 1
                    if phase_idx >= len(targets):
                        break
            if success or term or trunc:
                break

        _tcp_f, final_cube_xy, _, final_cube_z = _read_obs(obs)
        final_obj_xy = (float(final_cube_xy[0]), float(final_cube_xy[1]))
        trajectory.append(final_obj_xy)

        # Stage-0 reporting overrides for the slip case:
        # if the runner ran the slip path, force-report grasp_slip=True
        # and success=False even if the simulator briefly registered success.
        if slip:
            grasp_slip = True
            success = False
            object_moved = False
        else:
            grasp_slip = False
            # object_moved here means "cube lifted off the table" — final
            # cube_z noticeably higher than initial (not lateral motion).
            object_moved = (final_cube_z - initial_cube_z) > 0.02

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
            grasp_slip=bool(grasp_slip),
            rollout_log_path=str(rollout_log_path) if rollout_log_path else None,
            success=bool(success),
            trajectory_xy=tuple(trajectory),
        )

    def close(self) -> None:
        try:
            self._env.close()
        except Exception:
            pass


def _gripper_at_cube(
    tcp: np.ndarray, cube_xy: np.ndarray, cube_z: float,
    *, threshold: float = 0.04,
) -> bool:
    dxy = float(np.linalg.norm(tcp[0:2] - np.asarray(cube_xy, dtype=np.float64)))
    dz = abs(float(tcp[2]) - float(cube_z))
    return dxy < threshold and dz < threshold
