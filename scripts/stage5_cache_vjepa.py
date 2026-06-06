"""Stage-5 — extract and cache V-JEPA 2.1 clip features per seed (GPU, one-off).

Reads frame stacks written by scripts/stage5_render_demo_frames.py, runs the
V-JEPA 2.1 video encoder (default ViT-L/16 @ 384, token-mean pool), and writes
(1024,) float32 features alongside as seed_NNNN_<suffix>.npy.

V-JEPA 2.1 is a *clip* encoder (one (B,C,T,H,W) forward), the temporal lever
for StackCube object_motion that frame-mean-pooled DINOv2 cannot read (0.685 @
n=200). The probe (scripts/stage5_p1_g1_cert.py --feature-suffix) is identical
across encoders, so the V-JEPA number is apples-to-apples with the DINOv2 cell.

--shuffle-frame-order is the decisive control: it permutes each clip's sampled
frames in time (per-seed fixed RNG) before the forward pass. If ordered ≈
shuffled-time ≈ 0.685, no temporal signal is used (boundary closed); if ordered
≫ both, the lift is genuinely temporal (a 2nd hard latent task).

Weights download once (~5 GB for ViT-L) from dl.fbaipublicfiles.com into the
torch hub cache — pin TORCH_HOME to scratch in the sbatch. The repo's own hub
URL is a localhost stub, so vision_features._load_vjepa loads the real weights.

Example::

    python scripts/stage5_cache_vjepa.py \\
        --frames-dir datasets/stage5/object_relation_n200/StackCube-v1/frames/ \\
        --out-dir   datasets/stage5/object_relation_n200/StackCube-v1/features/ \\
        --encoder vjepa2_1_vitl16
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


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frames-dir", type=Path, default=None,
                   help="Directory of seed_NNNN.npz frame stacks.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output directory for seed_NNNN_<suffix>.npy.")
    p.add_argument("--encoder", type=str, default="vjepa2_1_vitl16")
    p.add_argument("--feature-suffix", type=str, default="vjepa21",
                   help="Filename suffix; --shuffle-frame-order appends '_shuf'.")
    p.add_argument("--n-frames", type=int, default=None,
                   help="Clip length (default: encoder spec, 64).")
    p.add_argument("--crop", type=int, default=None,
                   help="Crop size (default: encoder spec, 384).")
    p.add_argument("--shuffle-frame-order", action="store_true",
                   help="Temporal-order control: permute each clip's frames "
                        "(per-seed fixed RNG) before encoding.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--check", action="store_true",
                   help="Load the encoder once and exit (download + API smoke).")
    args = p.parse_args(argv)

    if args.check:
        from babysteps.stage4.vision_features import _load_vjepa
        m = _load_vjepa(args.encoder, args.device)
        n = sum(p.numel() for p in m.parameters())
        print(f"loaded {args.encoder} on {args.device}: {n:,} params "
              f"(embed_dim={getattr(m, 'embed_dim', '?')})")
        return 0

    if args.frames_dir is None or args.out_dir is None:
        p.error("--frames-dir and --out-dir are required unless --check is set")

    frame_files = sorted(args.frames_dir.glob("seed_*.npz"))
    if not frame_files:
        print(f"no seed_*.npz under {args.frames_dir}", file=sys.stderr)
        return 1

    suffix = args.feature_suffix + ("_shuf" if args.shuffle_frame_order else "")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    n_written = n_skipped = 0
    for fp in frame_files:
        out = args.out_dir / f"{fp.stem}_{suffix}.npy"
        if out.exists():                       # idempotent: standby-preempt safe
            n_skipped += 1
            continue
        frames = list(np.load(fp)["frames"])   # list[(H, W, 3) uint8]
        if args.shuffle_frame_order:
            # Per-seed deterministic permutation (seed from the filename digits).
            seed = int("".join(c for c in fp.stem if c.isdigit()) or "0")
            frames = [frames[i] for i in np.random.default_rng(seed).permutation(len(frames))]
        z = extract_vision_features(
            frames,
            encoder=args.encoder, device=args.device,
            vjepa_n_frames=args.n_frames, vjepa_crop=args.crop,
        )
        np.save(out, z)
        n_written += 1
        print(f"wrote {out} (shape={z.shape}, dtype={z.dtype})")
    print(f"done: {n_written} written, {n_skipped} skipped (already present)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
