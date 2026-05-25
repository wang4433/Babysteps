# scripts/stage5_render_demo_frames.py
"""Stage-5 P1 — re-render demo frames per seed (GPU).

For each seed in an existing varied-intent ``samples.jsonl``, re-runs the
oracle demo on the real ManiSkill env, captures one (H, W, 3) uint8 frame
per control step via :func:`babysteps.render.common.render_frame`, and
saves the stack as
``datasets/stage5/varied_intent/<task>/frames/seed_NNNN.npz``.

The demo is deterministic (same seed + same scripted oracle program), so
this adds no new ground truth: it produces a frame cache keyed to the
existing ``samples.jsonl`` records. Downstream (S3) consumes these
``.npz`` files to cache DINOv2 features.

Reuses three things from the codebase rather than duplicating logic:

* The per-seed cube-pose injection used by ``stage4_collect_varied.py``
  (PushCube only). The object_motion target comes from a local
  reconstruction of ``stratified_seed_plan`` (the same call
  ``stage4_collect_varied._collect_pushcube`` makes) — *not* from the
  source record's ``execution.initial_intent.object_motion``. That
  observed-motion field is derived from the demo trajectory and can drift
  from the originally-injected target when the cube barely moves (e.g.
  seed 19: injected ``translate_-x`` but observed motion snapped to
  ``translate_-y`` due to float noise — see commit ``c9a5426`` bug fix).
* :func:`babysteps.skills.push.build_push_waypoints` and
  :func:`babysteps.skills.stack.compile_intent_to_stack_skill` — the
  same waypoint compilers the real env_runners use.
* :func:`babysteps.render.common.render_frame` — the third-person RGB
  capture path used by the Stage-0 MP4 set.

Example::

    python scripts/stage5_render_demo_frames.py \\
        --jsonl datasets/stage4/varied_intent/PushCube-v1/samples.jsonl \\
        --out-dir datasets/stage5/varied_intent/PushCube-v1/frames/

GPU/Vulkan node required (login-node Vulkan device init fails).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from babysteps.envs.task_registry import get_task_entry  # noqa: E402
from babysteps.render.common import (  # noqa: E402
    PHASE_TOL_M,
    PUSHCUBE_MAX_CONTROL_STEPS,
    STACKCUBE_MAX_CONTROL_STEPS,
    prop_action,
    read_obs,
    render_frame,
    to_np,
)
from babysteps.schemas import EpisodeRecord, SceneState  # noqa: E402
from babysteps.stage4.collection_plan import stratified_seed_plan  # noqa: E402


# PushCube injection plan — must mirror scripts/stage4_collect_varied.py
# (`_PUSHCUBE_DIRS` and the `stratified_seed_plan(..., per_class=10,
# seed_start=0)` call in `_collect_pushcube`). We reconstruct the plan
# here instead of reading the observed motion from
# `execution.initial_intent.object_motion` because the latter is derived
# from the demo trajectory and can drift from the original injection
# target when the cube barely moves (commit c9a5426 bug — see module
# docstring).
_PUSHCUBE_DIRS = ("translate_+x", "translate_-x")
_PUSHCUBE_INJECTION_BY_SEED: dict[int, str] = dict(
    stratified_seed_plan(_PUSHCUBE_DIRS, episodes_per_class=10, seed_start=0)
)


# !!! Drift hazard: these mirror StackCubeEnvRunner constants by value
# (babysteps/envs/stackcube_runner.py: _MAX_EPISODE_STEPS, _GRASP_DWELL_STEPS,
# _SETTLE_DWELL_STEPS, _GRIPPER_OPEN, _GRIPPER_CLOSED, and the per-phase
# gripper schedules). If you tune the runner's PD/dwell/gripper timings,
# update these too or cached frames will diverge from the production rollout.
# StackCube TimeLimit override — matches StackCubeEnvRunner.
_STACKCUBE_MAX_EPISODE_STEPS = 200

# Gripper schedule for StackCube — matches the dispatch in
# StackCubeEnvRunner.run(). The demo is always cubeA_on_cubeB (5 phases)
# since the oracle correct intent is the pick-and-place branch.
_GRIPPER_OPEN = 1.0
_GRIPPER_CLOSED = -1.0
_STACKCUBE_5PHASE_APPROACH_GRIP = (
    _GRIPPER_OPEN, _GRIPPER_OPEN, _GRIPPER_OPEN,
    _GRIPPER_CLOSED, _GRIPPER_CLOSED,
)
_STACKCUBE_5PHASE_DWELL_GRIP = (
    _GRIPPER_OPEN, _GRIPPER_OPEN, _GRIPPER_CLOSED,
    _GRIPPER_CLOSED, _GRIPPER_OPEN,
)
# Dwell lengths (steps): hold the waypoint while the gripper finishes
# closing on the grasp / while cubeA settles after release. Matches
# StackCubeEnvRunner.
_GRASP_DWELL_STEPS = 15
_SETTLE_DWELL_STEPS = 25
_STACKCUBE_5PHASE_DWELL_LEN = (
    0, 0, _GRASP_DWELL_STEPS, 0, _SETTLE_DWELL_STEPS,
)


# ---------- per-seed helpers (sim-touching) ---------------------------- #


def _seed_from_record(rec: dict) -> int:
    """Extract the int seed from an episode_id like 'pushcube_varied_seed_0012'."""
    return int(rec["episode_id"].split("_")[-1])


def _read_stackcube_obs(obs):
    """(tcp_xyzw, cubeA_xy, cubeA_z, cubeB_xy, cubeB_z) from a StackCube obs."""
    raw = to_np(obs["extra"]["tcp_pose"])
    raw = np.asarray(raw, dtype=np.float64)
    tcp = np.concatenate([raw[0:3], raw[4:7], raw[3:4]])
    cubeA_full = np.asarray(to_np(obs["extra"]["cubeA_pose"]), dtype=np.float64)
    cubeA_xy = cubeA_full[0:2]
    cubeA_z = float(cubeA_full[2])
    cubeB_full = np.asarray(to_np(obs["extra"]["cubeB_pose"]), dtype=np.float64)
    cubeB_xy = cubeB_full[0:2]
    cubeB_z = float(cubeB_full[2])
    return tcp, cubeA_xy, cubeA_z, cubeB_xy, cubeB_z


def _pushcube_inject_goal(env, seed: int, object_motion: str):
    """Replicate PushCubeEnvRunner._reset_with_injection for the render env.

    PushCube varied-intent collection moves the GOAL (not the cube) so the
    cube→goal direction matches the target object_motion. We re-do that
    injection on the render env so the cached frames depict the same scene
    layout as the original episode.
    """
    import sapien
    from babysteps.envs.scene import motion_to_unit

    obs, _info = env.reset(seed=int(seed))
    _, cube_xy, goal_xy, _ = read_obs(obs)
    push_dist = float(np.linalg.norm(goal_xy - cube_xy))
    unit = motion_to_unit(object_motion)
    new_goal = (
        float(cube_xy[0]) + push_dist * float(unit[0]),
        float(cube_xy[1]) + push_dist * float(unit[1]),
    )
    gr = env.unwrapped.goal_region
    gpose = gr.pose.sp if hasattr(gr.pose, "sp") else gr.pose
    gr.set_pose(sapien.Pose(
        p=[new_goal[0], new_goal[1], float(gpose.p[2])],
        q=list(gpose.q),
    ))
    return env.unwrapped.get_obs()


def _capture_pushcube_demo(
    env, adapter, seed: int, object_motion: Optional[str],
) -> np.ndarray:
    """Re-run the PushCube oracle demo on `seed` and return a (T,H,W,3) uint8 stack.

    Mirrors ``babysteps.episode.generate_proxy_demo`` for PushCube: builds the
    SceneState from the post-injection obs (or the native reset when
    ``object_motion is None``), asks the adapter for the oracle correct intent,
    compiles waypoints via ``build_push_waypoints``, then steps the env
    open-loop. Captures one third-person RGB frame per step.

    When ``object_motion`` is None, no goal injection is performed and the
    env is reset natively (`env.reset(seed)`). This matches M2a's held-out
    `generate_proxy_demo` flow on eval seeds (no stratified injection).
    """
    from babysteps.skills.push import build_push_waypoints

    if object_motion is None:
        # Native reset, no goal injection — matches M2a eval's
        # generate_proxy_demo path on held-out seeds.
        obs, _info = env.reset(seed=int(seed))
    else:
        obs = _pushcube_inject_goal(env, seed, object_motion)
    tcp, cube_xy0, goal_xy, cube_z = read_obs(obs)
    scene = SceneState(
        cube_xy=(float(cube_xy0[0]), float(cube_xy0[1])),
        cube_z=cube_z,
        goal_xy=(float(goal_xy[0]), float(goal_xy[1])),
        tcp_start_pose=tuple(float(v) for v in tcp),  # type: ignore[arg-type]
        blocked_sides=(),
    )
    correct = adapter.oracle_correct_intent(scene)
    waypoints = build_push_waypoints(scene, correct)
    targets = [np.asarray(wp[0:3], dtype=np.float64) for wp in waypoints]

    frames: list[np.ndarray] = [render_frame(env)]
    phase_idx = 0
    for _ in range(PUSHCUBE_MAX_CONTROL_STEPS):
        tcp, _, _, _ = read_obs(obs)
        target = targets[phase_idx]
        if np.linalg.norm(target - tcp[0:3]) < PHASE_TOL_M:
            phase_idx += 1
            if phase_idx >= len(targets):
                break
            target = targets[phase_idx]
        action = prop_action(tcp, target, gripper_cmd=-1.0)
        obs, _r, term, trunc, info = env.step(action)
        frames.append(render_frame(env))
        term_b = bool(to_np(term).item()) if hasattr(term, "cpu") else bool(term)
        trunc_b = bool(to_np(trunc).item()) if hasattr(trunc, "cpu") else bool(trunc)
        succ = info.get("success", False) if hasattr(info, "get") else False
        succ_b = bool(to_np(succ).item()) if hasattr(succ, "cpu") else bool(succ)
        if succ_b or term_b or trunc_b:
            break
    return np.stack(frames, axis=0)


def _capture_stackcube_demo(env, adapter, seed: int) -> np.ndarray:
    """Re-run the StackCube oracle demo on `seed` and return a (T,H,W,3) uint8 stack.

    Mirrors ``babysteps.episode.generate_proxy_demo`` for StackCube: builds the
    SceneState from the native reset (no injection — StackCube uses native
    resets binned by cubeA→cubeB direction), asks the adapter for the oracle
    correct intent (cubeA_on_cubeB → 5 waypoint phases with grasp/settle
    dwells matching StackCubeEnvRunner.run).
    """
    from babysteps.skills.stack import (
        CUBE_HALF_SIZE,
        compile_intent_to_stack_skill,
    )

    obs, _info = env.reset(seed=int(seed))
    tcp, cubeA_xy0, cubeA_z, cubeB_xy, cubeB_z = _read_stackcube_obs(obs)
    cubeB_top_z = cubeB_z + 2 * CUBE_HALF_SIZE
    scene = SceneState(
        cube_xy=(float(cubeA_xy0[0]), float(cubeA_xy0[1])),
        cube_z=cubeA_z,
        goal_xy=(float(cubeB_xy[0]), float(cubeB_xy[1])),
        tcp_start_pose=tuple(float(v) for v in tcp),  # type: ignore[arg-type]
        blocked_sides=(),
        extra={
            "cubeB_xy": (float(cubeB_xy[0]), float(cubeB_xy[1])),
            "cubeB_z": cubeB_z,
            "cubeB_top_z": cubeB_top_z,
        },
    )
    correct = adapter.oracle_correct_intent(scene)
    skill = compile_intent_to_stack_skill(correct, scene)
    targets = [np.asarray(wp[0:3], dtype=np.float64) for wp in skill.waypoints]
    n_phases = len(targets)
    if n_phases != 5:
        raise RuntimeError(
            f"_capture_stackcube_demo: expected 5 oracle phases "
            f"(cubeA_on_cubeB), got {n_phases}"
        )

    frames: list[np.ndarray] = [render_frame(env)]
    phase_idx = 0
    dwelling = False
    dwell_remaining = 0
    for _ in range(STACKCUBE_MAX_CONTROL_STEPS):
        tcp, _cubeA_xy, _cubeA_z, _, _ = _read_stackcube_obs(obs)
        target = targets[phase_idx]
        if not dwelling and np.linalg.norm(target - tcp[0:3]) < PHASE_TOL_M:
            if _STACKCUBE_5PHASE_DWELL_LEN[phase_idx] > 0:
                dwelling = True
                dwell_remaining = _STACKCUBE_5PHASE_DWELL_LEN[phase_idx]
            else:
                phase_idx += 1
                if phase_idx >= n_phases:
                    break
                target = targets[phase_idx]
        grip = (
            _STACKCUBE_5PHASE_DWELL_GRIP[phase_idx] if dwelling
            else _STACKCUBE_5PHASE_APPROACH_GRIP[phase_idx]
        )
        action = prop_action(tcp, target, gripper_cmd=grip)
        obs, _r, term, trunc, info = env.step(action)
        frames.append(render_frame(env))
        term_b = bool(to_np(term).item()) if hasattr(term, "cpu") else bool(term)
        trunc_b = bool(to_np(trunc).item()) if hasattr(trunc, "cpu") else bool(trunc)
        succ = info.get("success", False) if hasattr(info, "get") else False
        succ_b = bool(to_np(succ).item()) if hasattr(succ, "cpu") else bool(succ)
        if dwelling:
            dwell_remaining -= 1
            if dwell_remaining <= 0:
                dwelling = False
                phase_idx += 1
                if phase_idx >= n_phases:
                    break
        if succ_b or term_b or trunc_b:
            break
    return np.stack(frames, axis=0)


# ---------- env construction ------------------------------------------- #


def _make_env(task: str):
    """Build a ManiSkill env in render mode for `task`.

    Matches scripts/render_stage0_maniskill.py's setup: state_dict obs,
    pd_ee_delta_pose control, cpu backend, rgb_array render. No
    panda_wristcam — the demo phase only needs the third-person view.
    """
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401 — registers tasks

    kwargs = dict(
        obs_mode="state_dict",
        control_mode="pd_ee_delta_pose",
        sim_backend="cpu",
        render_mode="rgb_array",
    )
    if task == "StackCube-v1":
        kwargs["max_episode_steps"] = _STACKCUBE_MAX_EPISODE_STEPS
    return gym.make(task, **kwargs)


# ---------- CLI -------------------------------------------------------- #


def _load_records(jsonl: Path, limit: Optional[int]) -> list[dict]:
    with jsonl.open() as f:
        records = [
            EpisodeRecord.from_jsonl_line(line).to_dict()
            for line in f if line.strip()
        ]
    if limit is not None:
        records = records[:limit]
    return records


def _capture_one(env, adapter, task: str, rec: dict) -> tuple[int, np.ndarray]:
    seed = _seed_from_record(rec)
    if task == "PushCube-v1":
        # Source of truth is the stratified collection plan, NOT
        # `execution.initial_intent.object_motion` (observed motion can drift
        # from the injection target when the cube barely moves — see module
        # docstring / commit c9a5426 bug).
        try:
            motion = _PUSHCUBE_INJECTION_BY_SEED[seed]
        except KeyError:
            raise RuntimeError(
                f"seed {seed} is not in the stratified PushCube injection plan; "
                f"can't reproduce the original demo. Update the plan reconstruction "
                f"to match the collection seed range."
            )
        frames = _capture_pushcube_demo(env, adapter, seed, motion)
    elif task == "StackCube-v1":
        frames = _capture_stackcube_demo(env, adapter, seed)
    else:
        raise NotImplementedError(
            f"stage5_render_demo_frames: task {task!r} not supported "
            f"(supported: PushCube-v1, StackCube-v1)"
        )
    if frames.shape[0] == 0:
        raise RuntimeError(f"demo produced 0 frames for seed {seed}")
    return seed, frames


def _capture_one_native(env, adapter, task: str, seed: int) -> tuple[int, np.ndarray]:
    """Capture one demo with NATIVE env reset (no stratified injection).

    Used by the --seed-range mode for held-out eval cuts where the source of
    truth is M2a's `generate_proxy_demo` flow (no injection). Currently only
    PushCube-v1 is supported in this mode; StackCube native capture would be
    identical to `_capture_stackcube_demo` (which already uses a native reset)
    if/when needed.
    """
    if task == "PushCube-v1":
        frames = _capture_pushcube_demo(env, adapter, seed, object_motion=None)
    elif task == "StackCube-v1":
        frames = _capture_stackcube_demo(env, adapter, seed)
    else:
        raise NotImplementedError(
            f"stage5_render_demo_frames: task {task!r} not supported in "
            f"--seed-range mode (supported: PushCube-v1, StackCube-v1)"
        )
    if frames.shape[0] == 0:
        raise RuntimeError(f"demo produced 0 frames for seed {seed}")
    return seed, frames


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    # Mutually exclusive seed sources: stratified collection (training cut)
    # vs explicit range (eval cut, no injection — matches M2a's native-reset
    # generate_proxy_demo flow on held-out seeds).
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--jsonl", type=Path, default=None,
                     help="Source varied-intent samples.jsonl (training cut "
                          "with stratified injection per _PUSHCUBE_INJECTION_BY_SEED).")
    grp.add_argument("--seed-range", type=str, default=None,
                     help="Inclusive seed range A-B for native-reset rendering "
                          "(matches M2a's held-out generate_proxy_demo path; "
                          "no injection). Example: --seed-range 100-149.")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Output directory for seed_NNNN.npz files.")
    p.add_argument("--limit", type=int, default=None,
                   help="Optional cap on number of seeds (smoke test).")
    p.add_argument("--task", type=str, default="PushCube-v1",
                   help="Task id (only used by --seed-range; --jsonl derives "
                        "task from records).")
    args = p.parse_args(argv)

    if args.jsonl is not None:
        records = _load_records(args.jsonl, args.limit)
        if not records:
            print("no records to render", file=sys.stderr)
            return 1

        task = records[0]["task"]
        if not all(r["task"] == task for r in records):
            print(f"mixed tasks in {args.jsonl}; aborting", file=sys.stderr)
            return 2

        entry = get_task_entry(task)
        adapter = entry.adapter_cls()
        args.out_dir.mkdir(parents=True, exist_ok=True)

        env = _make_env(task)
        try:
            for rec in records:
                seed, frames = _capture_one(env, adapter, task, rec)
                out = args.out_dir / f"seed_{seed:04d}.npz"
                np.savez_compressed(out, frames=frames)
                print(
                    f"wrote {out} (T={frames.shape[0]}, "
                    f"H={frames.shape[1]}, W={frames.shape[2]})",
                    flush=True,
                )
        finally:
            try:
                env.close()
            except Exception:
                pass
            adapter.close()
        return 0

    # --seed-range mode: NATIVE reset (no injection), matches M2a's
    # generate_proxy_demo flow for held-out eval seeds.
    lo_str, hi_str = args.seed_range.split("-")
    seeds = list(range(int(lo_str), int(hi_str) + 1))
    if args.limit is not None:
        seeds = seeds[:args.limit]
    if not seeds:
        print(f"empty --seed-range {args.seed_range}", file=sys.stderr)
        return 1

    task = args.task
    entry = get_task_entry(task)
    adapter = entry.adapter_cls()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    env = _make_env(task)
    try:
        for seed in seeds:
            seed_out, frames = _capture_one_native(env, adapter, task, seed)
            out = args.out_dir / f"seed_{seed_out:04d}.npz"
            np.savez_compressed(out, frames=frames)
            print(
                f"wrote {out} (T={frames.shape[0]}, "
                f"H={frames.shape[1]}, W={frames.shape[2]})",
                flush=True,
            )
    finally:
        try:
            env.close()
        except Exception:
            pass
        adapter.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
