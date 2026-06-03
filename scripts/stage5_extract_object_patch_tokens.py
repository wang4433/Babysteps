"""Stage-5 Step-2 — extract object-LOCAL DINOv2 patch tokens for StackCube.

Consumes the cached third-person demo frames
(``datasets/stage5/varied_intent/StackCube-v1/frames/seed_NNNN.npz``, key
``frames`` of shape ``(T, 512, 512, 3)`` uint8) and, per seed, dumps:

  * ``patch_start`` / ``patch_end`` : (256, 768) DINOv2 ``x_norm_patchtokens``
        for the FIRST and LAST frame. The first frame is the *resting*
        configuration that defines the ``object_motion`` label
        (``goal_direction_to_motion(cubeB_init - cubeA_init)``); the last frame
        is kept only for the optional start/end rung.
  * ``centroid_A_start`` / ``centroid_B_start`` (and ``_end``) : (u, v) pixel
        colour-blob centroids of cubeA (red) / cubeB (green), from
        ``babysteps.stage5.object_blobs`` — PIXEL-derived, no sim privilege.
        NaN where detection failed.
  * ``global_spatial_mean`` : (768,) mean over ALL frames' patch tokens —
        reproduces the frozen-DINOv2 spatial_mean feature that scored 0.42 on
        ``object_motion`` (``reports/stage5/p1_vision_g1``), as a sanity column.

GPU is used only for the DINOv2 forward pass — this script touches **no
ManiSkill / Vulkan** (it reads pre-rendered frames), so it imports cleanly on
the login node (torch/vision is imported lazily inside ``main``).

The pooling radius / rung choices are deferred to the CPU probe
(``stage5_object_relation_probe.py``), which reads this npz — so radius sweeps
need no GPU re-run.

Example::

    python scripts/stage5_extract_object_patch_tokens.py \\
        --frames-dir datasets/stage5/varied_intent/StackCube-v1/frames \\
        --jsonl      datasets/stage4/varied_intent/StackCube-v1/samples.jsonl \\
        --out        datasets/stage5/object_relation/StackCube-v1/object_tokens.npz
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Pure imports only at module load (keeps login-node / test import sim-free AND
# torch-free). The heavy torch/vision import is lazy, inside main().
from babysteps.schemas import EpisodeRecord  # noqa: E402
from babysteps.stage5.object_blobs import detect_cube_centroids  # noqa: E402

_NAN2 = np.array([np.nan, np.nan], dtype=np.float32)


def _calibration_image(prow: int, pcol: int, *, grid: int = 16, patch: int = 14,
                       bright: int = 255, base: int = 120) -> np.ndarray:
    """A (grid*patch)^2 gray image with one bright patch cell at (prow, pcol).

    Used to verify DINOv2's patch-token flattening order at runtime. The image
    is exactly the backbone input size (224 = 16*14) so the bright square
    aligns to a single patch cell with no resize ambiguity.
    """
    r = grid * patch
    img = np.full((r, r, 3), base, np.uint8)
    img[prow * patch:(prow + 1) * patch, pcol * patch:(pcol + 1) * patch] = bright
    return img


def _expected_flat(prow: int, pcol: int, *, grid: int = 16) -> int:
    """Row-major flat patch index (must match object_blobs.patch_rc_to_flat)."""
    return prow * grid + pcol


def _assert_row_major_layout(model, device: str, *, grid: int = 16,
                             prow: int = 6, pcol: int = 10) -> None:
    """Abort unless DINOv2 x_norm_patchtokens are row-major (prow*grid + pcol).

    Object-local pooling assumes a bright patch at image cell (prow, pcol) lands
    at flat token index ``prow*grid + pcol``. We verify by DIFFERENCING a
    bright-patch image against a gray baseline: DINOv2's input-independent
    high-norm 'artifact' tokens cancel, so the single most-changed token is the
    bright patch. If its index disagrees with row-major, patch selection would
    be geometrically wrong — so we hard-fail before extracting anything.
    """
    import torch
    from babysteps.stage4.vision_features import _preprocess_frames

    patch = 14
    base = np.full((grid * patch, grid * patch, 3), 120, np.uint8)
    stim = _calibration_image(prow, pcol, grid=grid, patch=patch)
    with torch.no_grad():
        xb = _preprocess_frames([base], resolution=grid * patch).to(device)
        xs = _preprocess_frames([stim], resolution=grid * patch).to(device)
        tb = model.forward_features(xb)["x_norm_patchtokens"][0]
        ts = model.forward_features(xs)["x_norm_patchtokens"][0]
    delta = (ts - tb).norm(dim=1).cpu().numpy()
    got = int(delta.argmax())
    exp = _expected_flat(prow, pcol, grid=grid)
    if got != exp:
        raise RuntimeError(
            f"DINOv2 patch layout is NOT row-major: a bright patch at image cell "
            f"(row={prow}, col={pcol}) should change token flat-index {exp}, but "
            f"the most-changed token is flat {got} (row={got // grid}, "
            f"col={got % grid}). Object-local patch selection would be wrong — "
            f"aborting before extraction.")
    print(f"  [layout OK] row-major verified: bright cell (r{prow},c{pcol}) "
          f"-> token flat {exp}")


def _load_labels(jsonl: Path) -> dict[int, str]:
    """{seed: object_motion} from a varied-intent records jsonl."""
    labels: dict[int, str] = {}
    with jsonl.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = EpisodeRecord.from_jsonl_line(line).to_dict()
            sd = int(rec["episode_id"].split("_")[-1])
            labels[sd] = rec["execution"]["initial_intent"]["object_motion"]
    return labels


def _seed_of(path: str) -> int:
    return int(os.path.basename(path).split("_")[1].split(".")[0])


def _centroid_xy(c: tuple[float, float] | None) -> np.ndarray:
    return _NAN2.copy() if c is None else np.asarray(c, dtype=np.float32)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames-dir", type=Path, required=True,
                   help="dir of seed_NNNN.npz frame stacks")
    p.add_argument("--jsonl", type=Path, required=True,
                   help="varied-intent records jsonl for object_motion labels")
    p.add_argument("--out", type=Path, required=True, help="output .npz path")
    p.add_argument("--encoder", default="dinov2_vitb14")
    p.add_argument("--device", default="cuda")
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--grid", type=int, default=16)
    p.add_argument("--batch", type=int, default=16,
                   help="frames per forward_features call (memory guard)")
    p.add_argument("--blob-thresh", type=int, default=40)
    args = p.parse_args(argv)

    # Lazy heavy imports — only when actually running (not at module import).
    import torch
    from babysteps.stage4.vision_features import _load_dinov2, _preprocess_frames

    labels = _load_labels(args.jsonl)
    files = sorted(glob.glob(str(args.frames_dir / "seed_*.npz")), key=_seed_of)
    if not files:
        print(f"no seed_*.npz under {args.frames_dir}", file=sys.stderr)
        return 1

    model = _load_dinov2(args.encoder, args.device)
    _assert_row_major_layout(model, args.device, grid=args.grid)
    expected_n = args.grid * args.grid

    seeds: list[int] = []
    y: list[str] = []
    patch_start, patch_end = [], []
    cAs, cBs, cAe, cBe = [], [], [], []
    glob_mean = []
    skipped = []

    for fp in files:
        sd = _seed_of(fp)
        if sd not in labels:
            skipped.append(sd)
            continue
        frames = np.load(fp)["frames"]  # (T, H, W, 3) uint8
        T = frames.shape[0]

        # DINOv2 patch tokens for all T frames (chunked to bound memory).
        chunks = []
        with torch.no_grad():
            for s in range(0, T, args.batch):
                x = _preprocess_frames(
                    list(frames[s:s + args.batch]), resolution=224,
                ).to(args.device)
                pt = model.forward_features(x)["x_norm_patchtokens"]  # (b, N, d)
                chunks.append(pt.detach().to("cpu"))
        patches = torch.cat(chunks, dim=0)  # (T, N, d)
        if patches.shape[1] != expected_n:
            raise RuntimeError(
                f"seed {sd}: got {patches.shape[1]} patches, expected "
                f"{expected_n} (grid={args.grid}); check resolution/patch size")

        glob_mean.append(patches.mean(dim=(0, 1)).numpy().astype(np.float32))
        patch_start.append(patches[0].numpy().astype(np.float32))
        patch_end.append(patches[-1].numpy().astype(np.float32))

        cs = detect_cube_centroids(frames[0], thresh=args.blob_thresh)
        ce = detect_cube_centroids(frames[-1], thresh=args.blob_thresh)
        cAs.append(_centroid_xy(cs["cubeA"])); cBs.append(_centroid_xy(cs["cubeB"]))
        cAe.append(_centroid_xy(ce["cubeA"])); cBe.append(_centroid_xy(ce["cubeB"]))

        seeds.append(sd); y.append(labels[sd])
        miss = [k for k, v in cs.items() if v is None]
        print(f"  seed {sd:04d}  T={T}  label={labels[sd]:12s}"
              f"  start-miss={miss if miss else '-'}")

    if not seeds:
        print("no frame/label intersection", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        seeds=np.asarray(seeds, dtype=np.int64),
        labels=np.asarray(y),
        patch_start=np.stack(patch_start),       # (n, 256, 768)
        patch_end=np.stack(patch_end),
        centroid_A_start=np.stack(cAs),           # (n, 2) pixel (u,v), NaN if missing
        centroid_B_start=np.stack(cBs),
        centroid_A_end=np.stack(cAe),
        centroid_B_end=np.stack(cBe),
        global_spatial_mean=np.stack(glob_mean),  # (n, 768) reproduces 0.42 feature
        grid=np.int64(args.grid),
        img_size=np.int64(args.img_size),
        encoder=np.asarray(args.encoder),
    )
    n_missA = int(np.isnan(np.stack(cAs)).any(axis=1).sum())
    n_missB = int(np.isnan(np.stack(cBs)).any(axis=1).sum())
    print(f"\nwrote {args.out}  (n={len(seeds)}; start-frame blob misses: "
          f"cubeA={n_missA}, cubeB={n_missB}; skipped no-label seeds={skipped})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
