"""Stage-5 B.2 — train a 4-way vision-grounded LatentPack for PushCube.

The committed PushCube pack (`models/stage5/p1_vision/PushCube-v1`) was trained
on the x-axis-only varied-intent cut, so its `contact_region` / `object_motion`
centroids cover only 2 / 3 classes. The 4-way natural loop
(`scripts/stage5_natural_loop_eval.py --axes xy`) needs all four cardinal faces,
so the learned residual ReviseHead has a 4-class latent slot space to decode
against. This script builds those 4 vision-grounded centroids from demo clips
rendered in all four directions:

    scripts/stage5_render_demo_frames.py --four-way-range A-B --out-dir FRAMES
    scripts/stage5_cache_dinov2.py --frames-dir FRAMES --out-dir FEATURES
    scripts/stage5_train_4way_pack.py --features-dir FEATURES \
        --labels FRAMES/labels.json --out-dir models/stage5/p1_vision_4way/PushCube-v1

Labels come from `labels.json` (emitted by the --four-way-range render), NOT an
EpisodeRecord jsonl — the 4-way cut is a demo-only render with no failure/revise
records. The per-factor centroids are the class means of the trained IntentHead's
G, identical to `stage5_p1_train_pack.py`. A placeholder non-residual ReviseHead
is saved so the pack round-trips through `load_latent_pack`; the *residual* head
(the one the latent_learned reviser uses) is trained separately by
`stage5_train_residual_revise_head.py`.

Sim-free CPU-only (consumes cached features); ~30s wall.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from babysteps.schemas import INTENT_FIELDS  # noqa: E402
from babysteps.stage4.intent_head import (  # noqa: E402
    IntentHead, nested_cv_probe_one_factor, train_intent_head_joint,
)
from babysteps.stage4.latent_policy import LatentPack, save_latent_pack  # noqa: E402
from babysteps.stage4.revise_head import ReviseHead  # noqa: E402
from babysteps.stage4.slot_decode import build_factor_centroids, decode_slot  # noqa: E402

# Factors carried by a PushCube demo clip (the others are constant / invisible).
_FOUR_WAY_FACTORS = ("object_motion", "contact_region", "approach_direction")


def _load_labels(path: Path) -> dict[int, dict]:
    payload = json.loads(path.read_text())
    return {int(s): lab for s, lab in payload["seeds"].items()}


def _load_features(seeds: list[int], features_dir: Path, suffix: str) -> np.ndarray:
    Zs = []
    for s in seeds:
        Zs.append(np.load(features_dir / f"seed_{s:04d}_{suffix}.npy"))
    return np.stack(Zs).astype(np.float32)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features-dir", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True,
                   help="labels.json from --four-way-range render.")
    p.add_argument("--feature-suffix", type=str, default="dinov2",
                   help="Feature filename suffix seed_NNNN_<suffix>.npy "
                        "(default dinov2; vjepa21 for the V-JEPA pack).")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--d-slot", type=int, default=32)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--n-epochs-intent", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--standardize", action="store_true",
                   help="Fit a StandardScaler on Z, train the IntentHead on "
                        "standardized features, and persist mean/scale to "
                        "scaler.npz. ESSENTIAL for V-JEPA (feature norm ~43 vs "
                        "DINOv2 ~24 -> un-normalized Adam@1e-2 underfits, ~0.54 "
                        "vs ~0.86); ~no-op for DINOv2. VisionIntentExtractor "
                        "applies the same scaler at inference.")
    p.add_argument("--cv", action="store_true",
                   help="Also report held-out IntentHead-CV probe accuracy per "
                        "factor (separability sanity before downstream eval).")
    p.add_argument("--factors", type=str, default=None,
                   help="Comma-separated subset of grounded factors (default: all "
                        f"of {_FOUR_WAY_FACTORS}). Train a per-VIEW pack that "
                        "grounds only the factors that view sees (dual-camera "
                        "setup): --factors contact_region for the contact view, "
                        "--factors object_motion,approach_direction for the global "
                        "view.")
    args = p.parse_args(argv)

    if args.factors is None:
        grounded = set(_FOUR_WAY_FACTORS)
    else:
        grounded = {f.strip() for f in args.factors.split(",") if f.strip()}
        bad = grounded - set(_FOUR_WAY_FACTORS)
        if bad:
            print(f"--factors {sorted(bad)} not in groundable set "
                  f"{_FOUR_WAY_FACTORS}", file=sys.stderr)
            return 2
    print(f"grounding factors: {sorted(grounded)}")

    labels = _load_labels(args.labels)
    seeds = sorted(labels.keys())
    # Keep only seeds whose features exist (resume-tolerant).
    seeds = [s for s in seeds
             if (args.features_dir / f"seed_{s:04d}_{args.feature_suffix}.npy").exists()]
    if not seeds:
        print("no features found for any labelled seed", file=sys.stderr)
        return 1
    Z = _load_features(seeds, args.features_dir, args.feature_suffix)  # raw (CV input)
    print(f"loaded {len(seeds)} seeds / Z={Z.shape}")

    # StandardScaler (persisted so VisionIntentExtractor applies the same
    # transform at inference). Fit on the full Z; the deployed pack's IntentHead
    # + centroids live in standardized-feature space. The CV check below uses the
    # RAW Z with leak-free per-fold standardization, so its number reflects the
    # same standardized model honestly.
    scaler_mean = np.zeros(Z.shape[1], dtype=np.float32)
    scaler_scale = np.ones(Z.shape[1], dtype=np.float32)
    Z_train = Z
    if args.standardize:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(Z)
        scaler_mean = scaler.mean_.astype(np.float32)
        scaler_scale = scaler.scale_.astype(np.float32)
        Z_train = ((Z - scaler_mean) / scaler_scale).astype(np.float32)
        print(f"  standardized: feat mean~{scaler_mean.mean():.2f} "
              f"scale~{scaler_scale.mean():.2f}")

    encoders: dict[int, LabelEncoder] = {}
    labels_per_factor: dict[int, tuple[np.ndarray, int]] = {}
    for fi, factor in enumerate(INTENT_FIELDS):
        if factor not in grounded:
            continue
        vals = [labels[s][factor] for s in seeds]
        present = sorted(set(vals))
        if len(present) < 2:
            print(f"  factor {factor}: constant ({present}); skipped")
            continue
        enc = LabelEncoder().fit(present)
        y = enc.transform(vals).astype(np.int64)
        encoders[fi] = enc
        labels_per_factor[fi] = (y, len(enc.classes_))
        _, counts = np.unique(y, return_counts=True)
        print(f"  factor {factor}: classes={list(enc.classes_)} counts={counts.tolist()}")

    if args.cv:
        for fi, (y, _n) in labels_per_factor.items():
            res = nested_cv_probe_one_factor(
                Z, y, factor_idx=fi, n_factors=len(INTENT_FIELDS),
                d_slot=args.d_slot, seed=args.seed,
                standardize_input=args.standardize)
            print(f"  CV[{INTENT_FIELDS[fi]:18s}] probe={res['probe_acc_mean']:.3f}"
                  f"±{res['probe_acc_std']:.3f} majority={res['majority_class_acc']:.3f} "
                  f"shuffled={res['shuffled_features_acc']:.3f}")

    head = IntentHead(z_dim=Z_train.shape[1], n_factors=len(INTENT_FIELDS),
                      d_slot=args.d_slot, hidden=args.hidden, seed=args.seed)
    train_intent_head_joint(
        head, Z_train, labels_per_factor,
        n_epochs=args.n_epochs_intent, lr=args.lr, seed=args.seed,
    )
    head.eval()
    with torch.no_grad():
        G = head(torch.from_numpy(Z_train)).numpy()
    centroids = build_factor_centroids(
        G, {fi: y for fi, (y, _) in labels_per_factor.items()},
    )

    # Training-set nearest-centroid accuracy (necessary separability check).
    for fi, (y, _n) in labels_per_factor.items():
        pred = np.array([decode_slot(G[b, fi], centroids[fi])
                         for b in range(G.shape[0])])
        acc = float((pred == y).mean())
        print(f"  train nearest-centroid[{INTENT_FIELDS[fi]:18s}] = {acc:.3f}")

    # Placeholder non-residual ReviseHead so the pack round-trips; the residual
    # head used by the latent_learned reviser is trained separately.
    revise = ReviseHead(d_slot=args.d_slot, hidden=args.hidden, seed=args.seed)
    label_tokens = {fi: tuple(enc.classes_) for fi, enc in encoders.items()
                    if fi in centroids}
    pack = LatentPack(
        intent_head=head, revise_head=revise,
        centroids=centroids, label_tokens=label_tokens,
        attribution_head=None,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_latent_pack(pack, args.out_dir)
    if args.standardize:
        np.savez(args.out_dir / "scaler.npz", mean=scaler_mean, scale=scaler_scale)
        print(f"  saved scaler.npz (StandardScaler mean/scale)")
    print(f"saved 4-way LatentPack to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
