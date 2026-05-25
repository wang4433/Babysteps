"""Real ManiSkill PushCube-v1 env_runner.

This is the ONLY module that imports `mani_skill`. The 3-phase open-loop
proportional EE control loop is ported from the Pick4Pass reference
`Code/scripts/run_pushcube_loop.py::_run_one_push`, simplified to consume a
pre-compiled `PushSkill` (the skill compiler is in
`babysteps.skills.push.compile_intent_to_push_skill`).

Note on Gilbreth: this requires a GPU/Vulkan-capable node (e.g.
`salloc --gres=gpu:1`). The login node fails at SAPIEN's Vulkan device init.
The unit tests, schemas, and fake env_runner do NOT need ManiSkill and run
on the login node.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np

from babysteps.skills.push import compile_intent_to_push_skill
from babysteps.schemas import AttemptResult, Intent, SceneState

# PD-tracking / phase-control constants. Matches Pick4Pass calibration for
# PushCube-v1's pd_ee_delta_pose (pos_upper=0.1 m).
_POS_SCALE: float = 0.1
_PHASE_TOL_M: float = 0.015
_MAX_CONTROL_STEPS: int = 300


def _to_np(x):
    """Batched ManiSkill tensor (1, N) → flat numpy (N,)."""
    arr = x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)
    return arr[0] if arr.ndim == 2 else arr


def _raw_to_xyzw(raw_pose) -> np.ndarray:
    """ManiSkill pose [x,y,z,qw,qx,qy,qz] → [x,y,z,qx,qy,qz,qw]."""
    raw = np.asarray(raw_pose, dtype=np.float64)
    return np.concatenate([raw[0:3], raw[4:7], raw[3:4]])


def _read_obs(obs) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return (tcp_xyzw, cube_xy, goal_xy, cube_z) from a PushCube obs."""
    tcp = _raw_to_xyzw(_to_np(obs["extra"]["tcp_pose"]))
    cube_full = _to_np(obs["extra"]["obj_pose"])
    cube_xy = cube_full[0:2].astype(np.float64)
    cube_z = float(cube_full[2])
    goal_xy = _to_np(obs["extra"]["goal_pos"])[0:2].astype(np.float64)
    return tcp, cube_xy, goal_xy, cube_z


def _prop_action(tcp_xyzw: np.ndarray, target_xyz: np.ndarray) -> np.ndarray:
    """Proportional normalized action toward target_xyz. Gripper kept closed."""
    pos_err = target_xyz - tcp_xyzw[0:3]
    action = np.zeros(7, dtype=np.float32)
    action[0:3] = np.clip(pos_err / _POS_SCALE, -1.0, 1.0).astype(np.float32)
    action[6] = np.float32(-1.0)
    return action


class PushCubeEnvRunner:
    """Real ManiSkill PushCube-v1 runner.

    Lazy-imports `mani_skill.envs` on construction. Holds one gym env across
    multiple `run(...)` calls; each `run` internally resets to the captured
    seed before executing the compiled push.
    """

    def __init__(self) -> None:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401 — registers PushCube-v1

        self._env = gym.make(
            "PushCube-v1",
            obs_mode="state_dict",
            control_mode="pd_ee_delta_pose",
            sim_backend="cpu",
        )
        self._last_seed: Optional[int] = None
        self._pending_motion: Optional[str] = None

    def set_injection(self, target_motion: Optional[str]) -> None:
        """Set (or clear with None) the target object_motion for the NEXT
        reset. The driver calls this per episode before run_episode; reset
        moves the GOAL so cube→goal points along target_motion, keeping the
        cube at its native (reachable) pose. (Cube-move was tried first but the
        +x-tuned push controller can't push from displaced cube positions; see
        the Stage-4 injection diagnostics.)"""
        self._pending_motion = target_motion

    def _reset_with_injection(self, seed: int):
        """env.reset(seed), then (re)apply the pending cube-pose injection.

        CRITICAL: this must run after EVERY self._env.reset — both the
        standalone reset() and run()'s internal re-reset. env.reset returns the
        cube to its native pose, so without re-applying here, run() would
        execute the (injected-scene) push waypoints against the un-injected
        native layout and the gripper would miss the cube entirely (it never
        moves). Idempotent: re-injecting an already-injected scene reproduces
        the same pose — the cube is already exactly push_dist from the goal.
        Returns the post-injection (obs, tcp, cube_xy, goal_xy, cube_z)."""
        obs, _info = self._env.reset(seed=int(seed))
        tcp, cube_xy, goal_xy, cube_z = _read_obs(obs)
        if self._pending_motion is not None:
            from babysteps.envs.scene import motion_to_unit
            # Goal-move: keep the cube at its native (reachable) pose, place the
            # goal push_dist away along the target motion so cube→goal points in
            # that direction. push_dist = native cube↔goal distance.
            push_dist = float(np.linalg.norm(goal_xy - cube_xy))
            unit = motion_to_unit(self._pending_motion)
            new_goal = (float(cube_xy[0]) + push_dist * float(unit[0]),
                        float(cube_xy[1]) + push_dist * float(unit[1]))
            import sapien
            gr = self._env.unwrapped.goal_region  # PushCube-v1 goal site
            gpose = gr.pose.sp if hasattr(gr.pose, "sp") else gr.pose
            gr.set_pose(sapien.Pose(
                p=[new_goal[0], new_goal[1], float(gpose.p[2])],
                q=list(gpose.q),
            ))
            obs = self._env.unwrapped.get_obs()
            tcp, cube_xy, goal_xy, cube_z = _read_obs(obs)
        return obs, tcp, cube_xy, goal_xy, cube_z

    def reset(self, seed: int) -> SceneState:
        """Reset and return the SceneState. blocked_sides is always ()
        — the privileged blocked-sides flag is set by the caller."""
        self._last_seed = int(seed)
        _obs, tcp, cube_xy, goal_xy, cube_z = self._reset_with_injection(seed)
        return SceneState(
            cube_xy=(float(cube_xy[0]), float(cube_xy[1])),
            cube_z=cube_z,
            goal_xy=(float(goal_xy[0]), float(goal_xy[1])),
            tcp_start_pose=tuple(float(v) for v in tcp),  # type: ignore[arg-type]
            blocked_sides=(),
        )

    def run(
        self,
        intent: Intent,
        scene: SceneState,
        *,
        rollout_log_path: Optional[Path] = None,
        rollout_seed: Optional[int] = None,  # see EnvRunner protocol docstring
    ) -> AttemptResult:
        """Execute one push attempt for `intent` under `scene` (`scene` carries
        blocked_sides). If the intent is blocked, returns planner_failed without
        stepping the env."""
        skill = compile_intent_to_push_skill(intent, scene)
        if skill is None:
            # Deliberate divergence from the render path: collection labels
            # this as planner_failed=True without stepping the env — fast,
            # and the right attribution for the schema. The render-path
            # equivalent (babysteps/render/pushcube.py) spawns a physical
            # red wall and steps until the arm stalls, because reviewers
            # see the MP4. Do not unify.
            return AttemptResult(
                initial_obj_xy=scene.cube_xy,
                final_obj_xy=scene.cube_xy,
                goal_xy=scene.goal_xy,
                reached_contact=False,
                object_moved=False,
                planner_failed=True,
                collision=False,
                grasp_slip=False,
                rollout_log_path=None,
                success=False,
                trajectory_xy=(),
            )

        if self._last_seed is None:
            raise RuntimeError("PushCubeEnvRunner.run called before reset()")
        # Re-apply the injection: env.reset returns the cube to its native pose,
        # so without this the (injected-scene) waypoints would act on the native
        # layout and miss the cube (see _reset_with_injection).
        obs, tcp, cube_xy0, goal_xy, cube_z = self._reset_with_injection(self._last_seed)
        initial_obj_xy = (float(cube_xy0[0]), float(cube_xy0[1]))

        # Three waypoint phase targets, each a 3-vec xyz.
        targets: list[np.ndarray] = []
        for wp in skill.waypoints:
            targets.append(np.asarray(wp[0:3], dtype=np.float64))

        trajectory: list[tuple[float, float]] = []
        phase_idx = 0
        reached_contact = False
        success = False

        for _step in range(_MAX_CONTROL_STEPS):
            tcp, cube_xy, _, _ = _read_obs(obs)
            trajectory.append((float(cube_xy[0]), float(cube_xy[1])))
            target = targets[phase_idx]
            if np.linalg.norm(target - tcp[0:3]) < _PHASE_TOL_M:
                phase_idx += 1
                if phase_idx >= len(targets):
                    break
                target = targets[phase_idx]
            # Cube has been touched in the descend/push phases (≥ 1).
            if phase_idx >= 1:
                reached_contact = reached_contact or _cube_in_contact_range(
                    tcp, cube_xy, skill.cube_z,
                )
            action = _prop_action(tcp, target)
            obs, _r, terminated, truncated, info = self._env.step(action)
            term = bool(_to_np(terminated).item()) if hasattr(terminated, "cpu") else bool(terminated)
            trunc = bool(_to_np(truncated).item()) if hasattr(truncated, "cpu") else bool(truncated)
            succ_field = info.get("success", False) if hasattr(info, "get") else False
            success = bool(_to_np(succ_field).item()) if hasattr(succ_field, "cpu") else bool(succ_field)
            if success or term or trunc:
                break

        _, final_cube_xy, _, _ = _read_obs(obs)
        final_obj_xy = (float(final_cube_xy[0]), float(final_cube_xy[1]))
        trajectory.append(final_obj_xy)
        object_moved = (
            math.hypot(final_obj_xy[0] - initial_obj_xy[0],
                       final_obj_xy[1] - initial_obj_xy[1]) > 0.005
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


def _cube_in_contact_range(
    tcp: np.ndarray, cube_xy: np.ndarray, cube_z: float,
    *, threshold: float = 0.03,
) -> bool:
    """Crude contact heuristic: TCP within `threshold` of cube xy AND
    within `threshold` of cube z."""
    dxy = float(np.linalg.norm(tcp[0:2] - np.asarray(cube_xy, dtype=np.float64)))
    dz = abs(float(tcp[2]) - float(cube_z))
    return dxy < threshold and dz < threshold
