"""Stage-5 P1 — extract and cache DINOv2 features per seed (GPU, one-off).

Reads frame stacks written by scripts/stage5_render_demo_frames.py,
runs extract_vision_features (default DINOv2 ViT-B/14, cls_mean pool),
and writes (768,) float32 features alongside as seed_NNNN_dinov2.npy.

The model is loaded once per process and cached at module level
(babysteps.stage4.vision_features._MODEL_CACHE), so the per-seed cost
is just the forward pass on (T, 3, 224, 224).

Example::

    python scripts/stage5_cache_dinov2.py \\
        --frames-dir datasets/stage5/varied_intent/PushCube-v1/frames/ \\
        --out-dir datasets/stage5/varied_intent/PushCube-v1/features/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from babysteps.stage4.vision_features import extract_vision_features  # noqa: E402


def select_frames(frames: list, mode: str) -> list:
    """Subset a clip's frames before encoding (final-state pooling).

    `goal_state` is a FINAL-STATE factor: the whole-clip mean dilutes it (the
    StackCube grounding result — spatial_mean over {first,last} clears 0.90 while
    over-all-frames caps ~0.77). Mirrors
    `scripts.stage5_goal_state_probe.clip_pool_frame_indices`:
      all        -> every frame (deployed default; trajectory/contact factors)
      final      -> the last frame only
      first_last -> first + last (the validated goal_state pooling)
      last5      -> the final 5 frames
    """
    if mode == "all" or not frames:
        return frames
    n = len(frames)
    last = n - 1
    if mode == "final":
        return [frames[last]]
    if mode == "first_last":
        return [frames[0], frames[last]]
    if mode == "last5":
        return frames[max(0, n - 5):]
    raise ValueError(f"unknown --frame-select {mode!r}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frames-dir", type=Path, default=None,
                   help="Directory of seed_NNNN.npz frame stacks.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output directory for seed_NNNN_dinov2.npy.")
    p.add_argument("--encoder", type=str, default="dinov2_vitb14")
    p.add_argument("--pool", type=str, default="cls_mean")
    p.add_argument("--frame-select", type=str, default="all",
                   choices=("all", "final", "first_last", "last5"),
                   help="Subset frames before encoding (final-state pooling for "
                        "goal_state; default 'all' = whole clip, byte-identical to "
                        "prior caches). Pair with --pool spatial_mean + a distinct "
                        "--feature-suffix for the goal_state pack.")
    p.add_argument("--resolution", type=int, default=224,
                   help="Square resize before encoding (224 default; 384/512 for "
                        "hi-res frame-encoder controls — must divide the patch size).")
    p.add_argument("--feature-suffix", type=str, default="dinov2",
                   help="Output filename suffix seed_NNNN_<suffix>.npy (default "
                        "'dinov2'; e.g. 'dinov3l384' for a resolution control).")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--check", action="store_true",
                   help="Load DINOv2 once and exit (smoke).")
    args = p.parse_args(argv)

    if args.check:
        from babysteps.stage4.vision_features import _load_dinov2
        m = _load_dinov2(args.encoder, args.device)
        print(f"loaded {args.encoder} on {args.device}: "
              f"{sum(p.numel() for p in m.parameters()):,} params")
        return 0

    if args.frames_dir is None or args.out_dir is None:
        p.error("--frames-dir and --out-dir are required unless --check is set")

    frame_files = sorted(args.frames_dir.glob("seed_*.npz"))
    if not frame_files:
        print(f"no seed_*.npz under {args.frames_dir}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for fp in frame_files:
        out = args.out_dir / f"{fp.stem}_{args.feature_suffix}.npy"
        if out.exists():                       # idempotent: standby-preempt safe
            continue
        frames = list(np.load(fp)["frames"])  # list[(H, W, 3) uint8]
        frames = select_frames(frames, args.frame_select)
        z = extract_vision_features(
            frames,
            encoder=args.encoder, pool=args.pool, device=args.device,
            resolution=args.resolution,
        )
        np.save(out, z)
        print(f"wrote {out} (shape={z.shape}, dtype={z.dtype})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
