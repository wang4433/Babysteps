#!/usr/bin/env python
"""Render the PushCube baseline contrast as MP4s (GPU/Vulkan).

For each seed, emits four clips showing the same demo + blocked attempt, then
two retries side-by-side:

    <prefix>_seed_NNNN__1_demo.mp4                 proxy demonstration
    <prefix>_seed_NNNN__2_attempt_blocked.mp4      initial (blocked) attempt
    <prefix>_seed_NNNN__3a_retry_selective.mp4     babysteps_selective → success
    <prefix>_seed_NNNN__3b_retry_full_replan.mp4   full_replan_analogue → wrong-way push

Both retries use the real policies (babysteps.policies), so the clip shows the
measured behaviour: selective preserves contact_region and pushes toward the
goal; full_replan_analogue's collateral contact_region edit pushes the wrong
way and fails. PushCube only — it is the one main-table task with a sibling
editable factor (contact_region) for the collateral edit to break.

Needs Vulkan. On the Gilbreth login node it falls back to Mesa lavapipe (slow);
on a GPU node it uses the NVIDIA Vulkan ICD.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the project root importable without `pip install -e .`.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from babysteps.envs.task_registry import get_task_entry  # noqa: E402
from babysteps.render.common import annotate_frame, save_mp4  # noqa: E402
from babysteps.render.pushcube import render_baseline_contrast  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--n_episodes", type=int, default=2)
    p.add_argument("--seed_start", type=int, default=0)
    p.add_argument("--fps", type=int, default=20)
    args = p.parse_args(argv)

    try:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401 — registers PushCube-v1
    except ImportError as exc:
        print(f"ManiSkill import failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    entry = get_task_entry("PushCube-v1")
    adapter = entry.adapter_cls()
    try:
        videos_dir = args.out_dir / "videos_maniskill"
        videos_dir.mkdir(parents=True, exist_ok=True)

        env = gym.make(
            adapter.gym_env_id,
            obs_mode="state_dict",
            control_mode="pd_ee_delta_pose",
            sim_backend="cpu",
            render_mode="rgb_array",
        )
        try:
            for i in range(args.n_episodes):
                seed = args.seed_start + i
                episode_id = f"{entry.episode_id_prefix}_contrast_seed_{seed:04d}"
                print(f"[{i + 1}/{args.n_episodes}] {episode_id}", flush=True)

                frames, titles = render_baseline_contrast(
                    env, adapter, seed=seed, fps=args.fps,
                )
                for phase_name, mp4_suffix in [
                    ("demo",              "1_demo"),
                    ("attempt_blocked",   "2_attempt_blocked"),
                    ("retry_selective",   "3a_retry_selective"),
                    ("retry_full_replan", "3b_retry_full_replan"),
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
    finally:
        adapter.close()

    print(f"\nDone. MP4s in {videos_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
