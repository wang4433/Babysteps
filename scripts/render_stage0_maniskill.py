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

Recommended invocation on a GPU node (PickCube):

    cd /scratch/gilbreth/wang4433/babysteps
    conda activate handover
    OUT_DIR=/scratch/gilbreth/wang4433/render_pickcube
    LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" \\
      python scripts/render_stage0_maniskill.py \\
        --task PickCube-v1 --out_dir "$OUT_DIR" \\
        --n_episodes 2 --seed_start 0
"""
from __future__ import annotations

import argparse
import sys
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
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--n_episodes", type=int, default=5)
    p.add_argument("--seed_start", type=int, default=0)
    p.add_argument("--fps", type=int, default=20)
    args = p.parse_args(argv)

    try:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401 — registers PushCube-v1 / PickCube-v1
    except Exception as exc:
        print(
            f"ManiSkill import failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2

    entry = get_task_entry(args.task)
    render_fn = get_render_fn(args.task)
    adapter = entry.adapter_cls()

    videos_dir = args.out_dir / "videos_maniskill"
    videos_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(
        args.task,
        obs_mode="state_dict",
        control_mode="pd_ee_delta_pose",
        sim_backend="cpu",
        render_mode="rgb_array",
    )

    try:
        for i in range(args.n_episodes):
            seed = args.seed_start + i
            episode_id = f"{entry.episode_id_prefix}_seed_{seed:04d}"
            print(f"[{i + 1}/{args.n_episodes}] {episode_id}", flush=True)

            frames, titles = render_fn(env, adapter, seed=seed, fps=args.fps)

            for phase_name, mp4_suffix in [
                ("demo",            "1_demo"),
                ("attempt_blocked", "2_attempt_blocked"),
                ("retry",           "3_retry"),
            ]:
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
        adapter.close()

    print(f"\nDone. MP4s in {videos_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
