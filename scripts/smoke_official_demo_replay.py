"""P0 smoke: render official ManiSkill demos via state-replay (GPU/Vulkan).

Proves the unblocked Scope-A path end to end: load an official
``trajectory.h5``, teleport through its recorded ``env_states`` (never the
``actions`` channel), film the third-person render camera, and write an MP4
+ a mid-solve PNG per task. This is the spec's P0 gate ("saved PNG of the
Panda mid-solve").

Runs on a GPU node only (rendering needs Vulkan). The h5/json reads and the
firewall are already validated sim-free in tests/.

Usage:
    python scripts/smoke_official_demo_replay.py \
        --tasks PushCube-v1 PickCube-v1 StackCube-v1 \
        --seed 0 --stride 2 --out-dir renders/official_demo_smoke
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Make the project root importable without `pip install -e .` (mirrors
# render_stage0_maniskill.py — the script lives in scripts/, so sys.path[0]
# is scripts/, not the repo root).
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from babysteps.render.common import save_mp4  # noqa: E402
from babysteps.render.official_demo import replay_official_state_frames  # noqa: E402


def _run_one(task: str, seed: int, stride: int, out_dir: Path, fps: int) -> dict:
    frames, meta = replay_official_state_frames(task, seed=seed, stride=stride)
    if not frames:
        raise RuntimeError(f"{task}: state-replay produced 0 frames")

    arr0 = np.asarray(frames[0])
    if arr0.dtype != np.uint8 or arr0.ndim != 3 or arr0.shape[-1] != 3:
        raise RuntimeError(
            f"{task}: bad frame shape/dtype {arr0.shape}/{arr0.dtype}; "
            "expected (H, W, 3) uint8"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = out_dir / f"{task}_seed_{seed:04d}__official_replay.mp4"
    save_mp4(frames, mp4_path, fps)

    # Mid-solve PNG (the P0 gate artifact).
    mid = frames[len(frames) // 2]
    png_path = out_dir / f"{task}_seed_{seed:04d}__midframe.png"
    try:
        from PIL import Image

        Image.fromarray(np.asarray(mid)).save(png_path)
    except Exception as exc:  # pragma: no cover - GPU-only path
        print(f"  [warn] PNG save failed ({exc}); MP4 still written")
        png_path = None

    return {
        "task": task,
        "episode_seed": meta["episode_seed"],
        "n_states": meta["n_states"],
        "n_frames": len(frames),
        "frame_shape": tuple(arr0.shape),
        "mp4": str(mp4_path),
        "png": str(png_path) if png_path else None,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--tasks",
        nargs="+",
        default=["PushCube-v1", "PickCube-v1", "StackCube-v1"],
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/home/wang4433/scratch/babysteps/renders/official_demo_smoke"),
    )
    args = p.parse_args(argv)

    results = []
    for task in args.tasks:
        print(f"=== {task} (seed {args.seed}) ===")
        info = _run_one(task, args.seed, args.stride, args.out_dir, args.fps)
        results.append(info)
        print(
            f"  states={info['n_states']} frames={info['n_frames']} "
            f"shape={info['frame_shape']}"
        )
        print(f"  mp4: {info['mp4']}")
        print(f"  png: {info['png']}")

    print("\nP0 smoke OK — official demos render third-person via state-replay.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
