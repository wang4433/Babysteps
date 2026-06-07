"""Real ManiSkill PokeCube-v1 env_runner — the grasp+poke controller.

PokeCube is the SECOND ``contact_region`` family (build-order step 3): the
Franka must GRASP a peg lying on the table and POKE the cube to a goal region.
Unlike PushCube (gripper fingers contact the cube face directly), here the peg
TIP contacts the face — same contact_region semantics, different execution
physics. That shared factor + different physics is what makes a frozen shared
RevisionPolicy's transfer to PokeCube a real leave-one-task-family-out test.

Why hand-rolled proportional EE control (not mplib motion planning):
``mplib`` is broken in this env (toppra's Cython extension was built against
numpy 1.x; the env ships numpy 2.x), so the official PandaArmMotionPlanningSolver
route is unavailable without a risky numpy downgrade that would threaten the
DINOv2 / V-JEPA / torch stack. So this mirrors PickCubeEnvRunner's phased
``pd_ee_delta_pose`` loop instead: approach → descend → grasp → lift → rotate →
realign → lower → poke(settle), with the gripper command in ``action[6]``
(+1 open, -1 closed) and the peg yaw P-controlled via ``action[5]`` (reusing
PushCubeEnvRunner's calibration) so non-+x pokes can reorient the peg to lead
with its head along the travel direction.

Geometry (all derived from ``intent.contact_region``):
  poke_dir = face_to_push_unit(contact_region)   # cube travel direction
  peg head (after yaw reorientation) leads at  TCP_xy + poke_dir * peg_half_length
  prepoke TCP = cube_xy - poke_dir * (cube_half + gap + peg_half_length)
  poke   TCP = goal_xy - poke_dir * (cube_half + peg_half_length - overshoot)

The goal is read from the ``goal_region`` ACTOR pose, NOT obs['extra']['goal_pos']
(in PokeCube-v1 that obs field is a quirk that returns the peg position). Goal-move
injection mirrors PushCubeEnvRunner.set_injection: the goal_region is re-placed so
cube->goal points along the requested motion, keeping the cube at its native pose.

Constants are deliberately exposed as module-level tunables — the grasp/poke
heights and travel are GPU-calibrated by scripts/stage5_pokecube_killgate.py.

Note on Gilbreth: requires a GPU/Vulkan node (same as the other runners).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np

from babysteps.envs.scene import face_to_push_unit
from babysteps.schemas import AttemptResult, Intent, SceneState

# --- PD / phase constants (match PushCube/PickCube calibration) ----------- #
_POS_SCALE: float = 0.1
_PHASE_TOL_M: float = 0.015
_MAX_CONTROL_STEPS: int = 500
# PokeCube-v1's default TimeLimit truncates at 50 steps — far too short for
# grasp+rotate+poke. Override at gym.make (PickCubeEnvRunner does the same).
_MAX_EPISODE_STEPS: int = 350

# Yaw P-control: action[5] is world-z yaw at ~ -2.24 deg/(step*unit) (PushCube
# calibration, job 10966502). Saturates beyond _YAW_K_DEG degrees of error.
_YAW_K_DEG: float = 20.0
_YAW_TOL_DEG: float = 6.0          # peg considered reoriented within this

# --- grasp + poke geometry (GPU-calibrated tunables) ---------------------- #
_Z_SAFE: float = 0.08              # carry height (peg clears table + cube)
_GRASP_Z_OFFSET: float = 0.0       # TCP z at grasp = peg_center_z + this
_POKE_Z_OFFSET: float = 0.0        # TCP z at poke = peg_center_z + this
_CONTACT_GAP: float = 0.015        # head standoff from the cube face pre-poke
# Aim the cube to land at goal CENTER (overshoot 0). A positive overshoot pushes
# the cube past goal and, worse, lengthens the poke stroke into reach-edge
# territory for far seeds (poke endpoint ~0.92 m from base at cube_x~0.32); the
# cube only needs to enter goal_radius (0.05), so center-aim is the reach-safe
# default. GPU-calibrated.
_POKE_OVERSHOOT: float = 0.0       # extra travel past nominal (cube lags head)
_CUBE_HALF: float = 0.02           # PokeCube cube_half_size
_PEG_HALF_LENGTH: float = 0.12     # PokeCube peg_half_length (head offset)

_GRIPPER_OPEN: float = 1.0
_GRIPPER_CLOSED: float = -1.0

# dwell schedule per phase (steps to hold once the waypoint is reached)
_GRASP_DWELL: int = 12
_ROTATE_DWELL: int = 6
_SETTLE_DWELL: int = 25


def _wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def _yaw_deg(tcp_xyzw: np.ndarray) -> float:
    """World-z yaw (deg) of the EE from a [x,y,z,qx,qy,qz,qw] pose (same scipy
    'xyz' euler convention PushCubeEnvRunner calibrated action[5] against)."""
    from scipy.spatial.transform import Rotation as _R
    q = np.asarray(tcp_xyzw, dtype=np.float64)
    return float(_R.from_quat([q[3], q[4], q[5], q[6]]).as_euler("xyz", degrees=True)[2])


def _to_np(x):
    arr = x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)
    return arr[0] if arr.ndim == 2 else arr


def _raw_to_xyzw(raw_pose) -> np.ndarray:
    """ManiSkill pose [x,y,z,qw,qx,qy,qz] → [x,y,z,qx,qy,qz,qw]."""
    raw = np.asarray(raw_pose, dtype=np.float64)
    return np.concatenate([raw[0:3], raw[4:7], raw[3:4]])


def _poke_yaw_offset_deg(poke_dir: np.ndarray) -> float:
    """EE yaw delta (deg) that aligns the peg AXIS with poke_dir's axis.

    The peg is geometrically symmetric (build_twocolor_peg, identical tips,
    symmetric length), so it can lead with EITHER tip — which tip is selected
    by the poke_dir*peg_half_length head offset, NOT by the yaw. We therefore
    fold the heading mod 180° into (-90, 90], exactly as PushCube's skill
    compiler folds push_yaw_deg ("the face is symmetric mod 180°", job
    10966514): x-axis pokes use 0°, y-axis pokes use -90°. This avoids the
    ~180° wrist flip that exceeds joint-7's range / hits the _wrap180
    singularity for the -x direction."""
    a = math.degrees(math.atan2(float(poke_dir[1]), float(poke_dir[0])))
    return float((a + 90.0) % 180.0 - 90.0)


class PokeCubeEnvRunner:
    """Real ManiSkill PokeCube-v1 grasp+poke runner.

    Lazy-imports ``mani_skill.envs`` on construction. Holds one gym env across
    multiple ``run(...)`` calls; each ``run`` re-resets to the captured seed
    (re-applying any pending goal-move injection) before executing the poke.

    After each ``run`` the simulator's terminal diagnostics are stashed on
    ``self.last_diag`` (is_peg_grasped / is_cube_placed / head_to_cube_dist /
    cube_displacement) for the kill-gate harness to read without widening the
    AttemptResult schema.
    """

    def __init__(
        self,
        render_mode: Optional[str] = None,
        *,
        capture_wrist_rgb: bool = False,
    ) -> None:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401 — registers PokeCube-v1

        self._capture_wrist_rgb = bool(capture_wrist_rgb)
        kwargs: dict = dict(
            obs_mode="state_dict",
            control_mode="pd_ee_delta_pose",
            sim_backend="cpu",
            max_episode_steps=_MAX_EPISODE_STEPS,
        )
        if render_mode is not None:
            kwargs["render_mode"] = render_mode
        if capture_wrist_rgb:
            # First-person execution view (panda_wristcam). EXECUTION-side only —
            # never fed to the demo->intent encoder. See PushCubeEnvRunner.
            kwargs["robot_uids"] = "panda_wristcam"
            kwargs["sensor_configs"] = dict(width=512, height=512)
        self._env = gym.make("PokeCube-v1", **kwargs)
        self._last_seed: Optional[int] = None
        self._pending_motion: Optional[str] = None
        self.last_diag: dict = {}

    # ------------------------------------------------------------------ #
    # reset + goal-move injection (mirrors PushCubeEnvRunner)
    # ------------------------------------------------------------------ #
    def set_injection(self, target_motion: Optional[str]) -> None:
        """Set (or clear with None) the target object_motion for the NEXT reset.
        reset moves the GOAL region so cube->goal points along target_motion,
        keeping the cube at its native (reachable) pose. None keeps the native
        +x goal."""
        self._pending_motion = target_motion

    def _goal_region(self):
        return self._env.unwrapped.goal_region

    def _read_state(self, obs):
        """Return (tcp_xyzw, peg_xy, peg_z, cube_xy, cube_z, goal_xy).

        Goal comes from the goal_region ACTOR (obs['extra']['goal_pos'] is the
        PokeCube quirk that returns the peg position)."""
        tcp = _raw_to_xyzw(_to_np(obs["extra"]["tcp_pose"]))
        peg = _to_np(obs["extra"]["peg_pose"])
        cube = _to_np(obs["extra"]["cube_pose"])
        gr = self._goal_region()
        gpose = gr.pose.sp if hasattr(gr.pose, "sp") else gr.pose
        goal_p = np.asarray(gpose.p, dtype=np.float64).reshape(-1)
        return (
            tcp,
            peg[0:2].astype(np.float64), float(peg[2]),
            cube[0:2].astype(np.float64), float(cube[2]),
            goal_p[0:2],
        )

    def _reset_with_injection(self, seed: int):
        obs, _info = self._env.reset(seed=int(seed))
        tcp, peg_xy, peg_z, cube_xy, cube_z, goal_xy = self._read_state(obs)
        if self._pending_motion is not None:
            from babysteps.envs.scene import motion_to_unit
            # Goal-move: keep the cube native, re-place the goal push_dist away
            # along the target motion. push_dist = native cube<->goal distance.
            push_dist = float(np.linalg.norm(goal_xy - cube_xy))
            unit = motion_to_unit(self._pending_motion)
            new_goal = (float(cube_xy[0]) + push_dist * float(unit[0]),
                        float(cube_xy[1]) + push_dist * float(unit[1]))
            import sapien
            gr = self._goal_region()
            gpose = gr.pose.sp if hasattr(gr.pose, "sp") else gr.pose
            gr.set_pose(sapien.Pose(
                p=[new_goal[0], new_goal[1], float(gpose.p[2])],
                q=list(gpose.q),
            ))
            obs = self._env.unwrapped.get_obs()
            tcp, peg_xy, peg_z, cube_xy, cube_z, goal_xy = self._read_state(obs)
        return obs, tcp, peg_xy, peg_z, cube_xy, cube_z, goal_xy

    def reset(self, seed: int) -> SceneState:
        self._last_seed = int(seed)
        _obs, tcp, peg_xy, peg_z, cube_xy, cube_z, goal_xy = \
            self._reset_with_injection(seed)
        return SceneState(
            cube_xy=(float(cube_xy[0]), float(cube_xy[1])),
            cube_z=cube_z,
            goal_xy=(float(goal_xy[0]), float(goal_xy[1])),
            tcp_start_pose=tuple(float(v) for v in tcp),  # type: ignore[arg-type]
            blocked_sides=(),
            extra={"peg_xy": (float(peg_xy[0]), float(peg_xy[1])),
                   "peg_z": float(peg_z)},
        )

    # ------------------------------------------------------------------ #
    # run: grasp the peg, reorient, poke the cube to goal
    # ------------------------------------------------------------------ #
    def run(
        self,
        intent: Intent,
        scene: SceneState,
        *,
        rollout_log_path: Optional[Path] = None,
        rollout_seed: Optional[int] = None,
    ) -> AttemptResult:
        if self._last_seed is None:
            raise RuntimeError("PokeCubeEnvRunner.run called before reset()")
        obs, tcp, peg_xy, peg_z, cube_xy, cube_z, goal_xy = \
            self._reset_with_injection(self._last_seed)
        initial_obj_xy = (float(cube_xy[0]), float(cube_xy[1]))

        poke_dir = face_to_push_unit(intent.contact_region)  # cube travel unit
        yaw_offset = _poke_yaw_offset_deg(poke_dir)
        resting_yaw = _yaw_deg(tcp)
        target_yaw = resting_yaw + yaw_offset

        grasp_z = peg_z + _GRASP_Z_OFFSET
        poke_z = peg_z + _POKE_Z_OFFSET
        prepoke_xy = cube_xy - poke_dir * (_CUBE_HALF + _CONTACT_GAP + _PEG_HALF_LENGTH)
        poke_xy = goal_xy - poke_dir * (_CUBE_HALF + _PEG_HALF_LENGTH - _POKE_OVERSHOOT)

        # phase: (target_xyz, gripper, use_target_yaw, dwell_steps, name)
        peg3 = np.array([peg_xy[0], peg_xy[1]], dtype=np.float64)
        phases = [
            (np.array([peg3[0], peg3[1], _Z_SAFE]), _GRIPPER_OPEN,   False, 0, "approach"),
            (np.array([peg3[0], peg3[1], grasp_z]), _GRIPPER_OPEN,   False, 0, "descend"),
            (np.array([peg3[0], peg3[1], grasp_z]), _GRIPPER_CLOSED, False, _GRASP_DWELL, "grasp"),
            (np.array([peg3[0], peg3[1], _Z_SAFE]), _GRIPPER_CLOSED, False, 0, "lift"),
            (np.array([peg3[0], peg3[1], _Z_SAFE]), _GRIPPER_CLOSED, True,  _ROTATE_DWELL, "rotate"),
            (np.array([prepoke_xy[0], prepoke_xy[1], _Z_SAFE]), _GRIPPER_CLOSED, True, 0, "realign"),
            (np.array([prepoke_xy[0], prepoke_xy[1], poke_z]), _GRIPPER_CLOSED, True, 0, "lower"),
            (np.array([poke_xy[0], poke_xy[1], poke_z]),  _GRIPPER_CLOSED, True, _SETTLE_DWELL, "poke"),
        ]

        trajectory: list[tuple[float, float]] = []
        phase_idx = 0
        dwelling = False
        dwell_remaining = 0
        reached_contact = False
        success = False
        cube_placed = False        # latches True once cube enters goal_radius
        last_info: dict = {}

        capture_wrist = self._capture_wrist_rgb and rollout_log_path is not None
        wrist_frames: list[np.ndarray] = []
        if capture_wrist:
            from babysteps.render.common import render_wrist_frame
            wrist_frames.append(render_wrist_frame(self._env))

        for _step in range(_MAX_CONTROL_STEPS):
            tcp, _peg_xy, _peg_z, cube_now, _cube_z, _goal = self._read_state(obs)
            trajectory.append((float(cube_now[0]), float(cube_now[1])))
            target, gripper, use_yaw, dwell_len, _name = phases[phase_idx]

            advance = False
            if not dwelling:
                pos_ok = np.linalg.norm(target - tcp[0:3]) < _PHASE_TOL_M
                yaw_ok = (not use_yaw) or abs(_wrap180(target_yaw - _yaw_deg(tcp))) < _YAW_TOL_DEG
                if pos_ok and yaw_ok:
                    if dwell_len > 0:
                        dwelling = True
                        dwell_remaining = dwell_len
                    else:
                        advance = True
            if advance:
                phase_idx += 1
                if phase_idx >= len(phases):
                    break
                continue

            # contact heuristic: only the lower/poke phases (peg at poke height,
            # not the carry-height realign) — and require the head near the cube
            # in BOTH xy AND z, so contact_rate isn't just reach-rate.
            if phase_idx >= 6:
                head_xy = tcp[0:2] + poke_dir * _PEG_HALF_LENGTH
                if (float(np.linalg.norm(head_xy - cube_now)) < (_CUBE_HALF + 0.04)
                        and abs(float(tcp[2]) - poke_z) < 0.03):
                    reached_contact = True

            action = np.zeros(7, dtype=np.float32)
            # Reach de-risk: once the cube is inside goal_radius, STOP driving
            # toward poke_xy (which sits near the reach edge for far seeds) and
            # just hold — so the arm decelerates and is_robot_static fires,
            # letting success register. The cube only needs to be in the radius,
            # not the gripper at the last 0.015 m of an unreachable target.
            if not cube_placed:
                action[0:3] = np.clip((target - tcp[0:3]) / _POS_SCALE, -1.0, 1.0)
            action[6] = np.float32(gripper)
            if use_yaw and not cube_placed:
                err = _wrap180(target_yaw - _yaw_deg(tcp))
                action[5] = np.float32(np.clip(-err / _YAW_K_DEG, -1.0, 1.0))

            obs, _r, terminated, truncated, info = self._env.step(action)
            last_info = info
            term = bool(_to_np(terminated).item()) if hasattr(terminated, "cpu") else bool(terminated)
            trunc = bool(_to_np(truncated).item()) if hasattr(truncated, "cpu") else bool(truncated)
            succ_field = info.get("success", False) if hasattr(info, "get") else False
            success = bool(_to_np(succ_field).item()) if hasattr(succ_field, "cpu") else bool(succ_field)
            placed_field = info.get("is_cube_placed", False) if hasattr(info, "get") else False
            if (bool(_to_np(placed_field).item()) if hasattr(placed_field, "cpu")
                    else bool(placed_field)):
                cube_placed = True   # latch: arm now holds + settles → static
            if capture_wrist:
                wrist_frames.append(render_wrist_frame(self._env))

            if dwelling:
                dwell_remaining -= 1
                if dwell_remaining <= 0:
                    dwelling = False
                    phase_idx += 1
                    if phase_idx >= len(phases):
                        break
            if success or term or trunc:
                break

        _tcp_f, _peg_f, _peg_zf, final_cube, _cz, _g = self._read_state(obs)
        final_obj_xy = (float(final_cube[0]), float(final_cube[1]))
        trajectory.append(final_obj_xy)
        object_moved = (
            math.hypot(final_obj_xy[0] - initial_obj_xy[0],
                       final_obj_xy[1] - initial_obj_xy[1]) > 0.005
        )

        # terminal sim diagnostics for the kill-gate (not in AttemptResult).
        def _diag(key):
            v = last_info.get(key) if hasattr(last_info, "get") else None
            if v is None:
                return None
            arr = _to_np(v)
            if not hasattr(arr, "dtype"):
                return v
            return bool(arr.item()) if np.issubdtype(arr.dtype, np.bool_) \
                else float(arr.item())
        self.last_diag = {
            "is_peg_grasped": _diag("is_peg_grasped"),
            "is_cube_placed": _diag("is_cube_placed"),
            "head_to_cube_dist": _diag("head_to_cube_dist"),
            "cube_displacement": float(math.hypot(
                final_obj_xy[0] - initial_obj_xy[0],
                final_obj_xy[1] - initial_obj_xy[1])),
            "contact_region": intent.contact_region,
        }

        if rollout_log_path is not None:
            rollout_log_path.parent.mkdir(parents=True, exist_ok=True)
            save_kwargs: dict = dict(
                trajectory_xy=np.asarray(trajectory, dtype=np.float64),
                initial_obj_xy=np.asarray(initial_obj_xy, dtype=np.float64),
                final_obj_xy=np.asarray(final_obj_xy, dtype=np.float64),
                goal_xy=np.asarray(scene.goal_xy, dtype=np.float64),
            )
            if wrist_frames:
                save_kwargs["wrist_rgb"] = np.stack(wrist_frames).astype(np.uint8)
            np.savez(rollout_log_path, **save_kwargs)

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
