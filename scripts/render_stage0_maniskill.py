"""Render Stage-0 episodes as ManiSkill RGB MP4s — multi-task dispatcher.

For each seed, runs the full BABYSTEPS loop (demo proxy → blocked attempt
→ revised retry) for the chosen task, capturing `env.render()` frames per
phase, and writes one MP4 per phase to `<out_dir>/videos_maniskill/`.

Tasks:
  --task PushCube-v1 (default) — phase 2 is held-still (planner_failed).
  --task PickCube-v1            — phase 2 actually executes the failing
                                  lift so the grasp_slip is visible.

This script needs Vulkan. On the Gilbreth login node it works via Mesa's
software Vulkan rasterizer (lavapipe) — slow but real. On a GPU compute
node it uses the NVIDIA Vulkan ICD and runs much faster.

Per-task render flows live in babysteps/render/{pushcube,pickcube}.py;
this script is just the orchestration over a `--task` choice.

Output goes to <repo>/renders/<task>/videos_maniskill/ by default (all
project renders live under renders/); pass --out_dir only to override.

Recommended invocation on a GPU node (PickCube):

    cd /scratch/gilbreth/wang4433/babysteps
    conda activate handover
    LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" \\
      python scripts/render_stage0_maniskill.py \\
        --task PickCube-v1 --n_episodes 2 --seed_start 0
    # → renders/pickcube/videos_maniskill/
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Make the project root importable without `pip install -e .`.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from babysteps.envs.task_registry import TASK_REGISTRY, get_task_entry  # noqa: E402
from babysteps.render import get_render_fn  # noqa: E402
from babysteps.render.common import annotate_frame, save_mp4  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--task", type=str, default="PushCube-v1",
        choices=sorted(TASK_REGISTRY.keys()),
    )
    p.add_argument(
        "--out_dir", type=Path, default=None,
        help="Output root. Default: <repo>/renders/<task> (e.g. renders/pushcube). "
             "All project renders live under renders/ — pass this only to override.",
    )
    p.add_argument("--n_episodes", type=int, default=5)
    p.add_argument("--seed_start", type=int, default=0)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument(
        "--mode", type=str, default="clutter",
        choices=["clutter", "natural"],
        help=(
            "clutter: Stage-0 obstacle-based blocked-approach render (default; "
            "phase 2 spawns a clutter object on the approach side). "
            "natural: paper-figure mode — no obstacle, phase 2 executes a "
            "single-factor misgrounded intent so the cube goes the wrong way "
            "naturally. PushCube-v1 only for now."
        ),
    )
    p.add_argument(
        "--demo-source", type=str, default="scripted",
        choices=["scripted", "official"],
        help=(
            "scripted (default): the 1_demo phase is the babysteps skill "
            "compiler executing the oracle intent (current behavior). "
            "official: replace the 1_demo clip with ManiSkill's official "
            "Panda oracle demo, rendered third-person. Only the demo phase "
            "changes; attempt/retry are untouched. PushCube/PickCube/StackCube."
        ),
    )
    p.add_argument(
        "--official-source", type=str, default="state_replay",
        choices=["state_replay", "solver"],
        help=(
            "When --demo-source official: state_replay (default) teleports "
            "through the downloaded trajectory.h5 env_states (needs no mplib); "
            "solver runs the official motion planner live (needs a working "
            "mplib). Both render third-person and never read recorded actions."
        ),
    )
    p.add_argument(
        "--official-stride", type=int, default=1,
        help="Frame stride for --demo-source official state_replay (1 = full).",
    )
    args = p.parse_args(argv)

    OFFICIAL_DEMO_TASKS = {"PushCube-v1", "PickCube-v1", "StackCube-v1"}
    if args.demo_source == "official" and args.task not in OFFICIAL_DEMO_TASKS:
        print(
            f"--demo-source official supports {sorted(OFFICIAL_DEMO_TASKS)}; "
            f"got {args.task!r}.",
            file=sys.stderr,
        )
        return 2

    try:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401 — registers PushCube-v1 / PickCube-v1
    except ImportError as exc:
        print(
            f"ManiSkill import failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2

    entry = get_task_entry(args.task)
    if args.mode == "natural":
        # Natural-failure paper-figure render — wired for PushCube only.
        # Other tasks need their own opposite-factor injections (see
        # redesign_failure_paradigm.md §"Five Tasks").
        if args.task != "PushCube-v1":
            print(
                f"--mode natural only supports PushCube-v1 right now "
                f"(got {args.task!r}); other tasks: TODO.",
                file=sys.stderr,
            )
            return 2
        from babysteps.render.pushcube import render_natural_failure_episode
        render_fn = render_natural_failure_episode
        # Phase keys for the natural mode differ from the clutter mode —
        # "attempt" rather than "attempt_blocked".
        phase_to_suffix = [
            ("demo",    "1_demo"),
            ("attempt", "2_attempt_wrong_intent"),
            ("retry",   "3_retry"),
        ]
        videos_subdir = "videos_paper_figure"
    else:
        render_fn = get_render_fn(args.task)
        phase_to_suffix = [
            ("demo",            "1_demo"),
            ("attempt_blocked", "2_attempt_blocked"),
            ("retry",           "3_retry"),
        ]
        videos_subdir = "videos_maniskill"
    adapter = entry.adapter_cls()
    # Default every render under the project's renders/ tree. The first token
    # of episode_id_prefix is the curated per-task dir name (renders/CLAUDE.md):
    # pushcube_blocked_approach → renders/pushcube, crossview_grounding →
    # renders/crossview, etc.
    out_dir = args.out_dir or (_ROOT / "renders" / entry.episode_id_prefix.split("_")[0])
    # Timestamp each render run so re-runs don't overwrite prior outputs and a
    # reviewer can tell at a glance which set of MP4s belongs to which run.
    # Format: YYYY-MM-DD_HHMMSS (sorts chronologically, no shell-hostile chars).
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    try:
        videos_dir = out_dir / videos_subdir / run_timestamp
        videos_dir.mkdir(parents=True, exist_ok=True)
        print(f"writing {args.n_episodes} episode(s) × 3 phases to {videos_dir}",
              flush=True)

        # Tasks whose render module captures the execution phases from the
        # first-person panda_wristcam `hand_camera`. For these we swap in the
        # wristcam Panda (a kinematically identical Panda subclass that mounts
        # a gripper camera) and upscale the wrist sensor for a watchable MP4.
        # The demo phase still uses env.render() (third-person external camera).
        WRIST_VIEW_TASKS = {"PushCube-v1"}
        extra_kwargs = {}
        if args.task in WRIST_VIEW_TASKS:
            extra_kwargs = dict(
                robot_uids="panda_wristcam",
                sensor_configs=dict(width=512, height=512),
            )

        env = gym.make(
            adapter.gym_env_id,
            obs_mode="state_dict",
            control_mode="pd_ee_delta_pose",
            sim_backend="cpu",
            render_mode="rgb_array",
            **extra_kwargs,
        )
        try:
            for i in range(args.n_episodes):
                seed = args.seed_start + i
                episode_id = f"{entry.episode_id_prefix}_seed_{seed:04d}"
                print(f"[{i + 1}/{args.n_episodes}] {episode_id}", flush=True)

                frames, titles = render_fn(env, adapter, seed=seed, fps=args.fps)

                # Optionally replace ONLY the demo clip with the official
                # ManiSkill oracle demo (rendered third-person via its own
                # isolated env). Attempt/retry stay on the babysteps env.
                if args.demo_source == "official":
                    from babysteps.render.official_demo import official_demo_frames
                    # The dispatcher forwards `seed` itself; kw carries only
                    # source-specific extras (stride for state_replay).
                    kw = {}
                    if args.official_source == "state_replay":
                        kw = dict(stride=args.official_stride)
                    official_frames = official_demo_frames(
                        adapter.gym_env_id, seed,
                        source=args.official_source, **kw,
                    )
                    frames["demo"] = official_frames
                    base_title, _ = titles["demo"]
                    titles["demo"] = (
                        base_title,
                        f"official ManiSkill oracle demo "
                        f"({args.official_source}, third-person)",
                    )
                    print(
                        f"   demo: official {args.official_source} "
                        f"({len(official_frames)} frames)"
                    )

                for phase_name, mp4_suffix in phase_to_suffix:
                    title, subtitle = titles[phase_name]
                    annotated = [
                        annotate_frame(fr, title, subtitle)
                        for fr in frames[phase_name]
                    ]
                    out_path = videos_dir / f"{episode_id}__{mp4_suffix}.mp4"
                    save_mp4(annotated, out_path, args.fps)
                    kb = out_path.stat().st_size // 1024
                    print(f"   wrote {out_path.name}  ({kb} KB)")
        finally:
            env.close()
    finally:
        adapter.close()

    print(f"\nDone. MP4s in {videos_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
