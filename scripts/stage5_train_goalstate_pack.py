"""Stage-5 — train a goal_state vision-grounded LatentPack for StackCube.

goal_state is a FINAL-STATE factor: with a retract demo render (gripper cleared
from the final frame) + first_last pooling, frozen DINOv2 separates the two goal
configs at IntentHead-CV 0.920 (n=300; reports/stage5/goal_state_retract). This
script trains the deployed pack from those features:

    # 1. dump the both-class retract first_last features (the validated set)
    python scripts/stage5_goal_state_probe.py --task StackCube-v1 --mode clip-pool \\
        --seeds 0-149 --retract --dump-features FEATS --dump-pool first_last \\
        --feature-suffix dinov2_fl --out-dir reports/stage5/goal_state_retract_pack
    # 2. train the goal_state pack from FEATS/labels.json
    python scripts/stage5_train_goalstate_pack.py --features-dir FEATS \\
        --feature-suffix dinov2_fl --out-dir models/stage5/goalstate/StackCube-v1 --cv

Unlike the 4-way PushCube pack (object_motion/contact_region/approach_direction),
this grounds exactly ONE factor — goal_state (classes cube_at_target,
cubeA_on_cubeB) — from per-(seed,class) features. A placeholder non-residual
ReviseHead is saved so the pack round-trips through load_latent_pack; the
goal_state revision in the loop uses the deterministic goal_refinement operator
(revision.py) and/or a learned head trained separately.

Sim-free CPU-only (consumes cached features); ~10s wall.
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

_GOAL_STATE_IDX = INTENT_FIELDS.index("goal_state")


def _load_dump(features_dir: Path, suffix: str) -> tuple[np.ndarray, list[str], list[str]]:
    """Read labels.json + per-stem features. Returns (Z (N,d), tokens, stems)."""
    payload = json.loads((features_dir / "labels.json").read_text())
    if payload.get("factor") != "goal_state":
        raise ValueError(f"labels.json factor={payload.get('factor')!r}, expected goal_state")
    labels = payload["labels"]
    stems = sorted(labels.keys())
    Z, tokens, kept = [], [], []
    for stem in stems:
        fp = features_dir / f"{stem}_{suffix}.npy"
        if not fp.exists():
            continue
        Z.append(np.load(fp))
        tokens.append(labels[stem])
        kept.append(stem)
    if not Z:
        raise FileNotFoundError(f"no '{suffix}' features for any stem in {features_dir}")
    return np.stack(Z).astype(np.float32), tokens, kept


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features-dir", type=Path, required=True,
                   help="Dir with labels.json + seed_NNNN_<class>_<suffix>.npy "
                        "(from stage5_goal_state_probe.py --dump-features).")
    p.add_argument("--feature-suffix", type=str, default="dinov2_fl")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--d-slot", type=int, default=32)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--n-epochs-intent", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--standardize", action="store_true",
                   help="Fit + persist a StandardScaler (scaler.npz); the "
                        "VisionIntentExtractor applies it at inference.")
    p.add_argument("--cv", action="store_true",
                   help="Report held-out IntentHead-CV probe accuracy for goal_state.")
    args = p.parse_args(argv)

    Z, tokens, stems = _load_dump(args.features_dir, args.feature_suffix)
    present = sorted(set(tokens))
    if len(present) < 2:
        print(f"goal_state is constant ({present}); need both classes to train a "
              "pack", file=sys.stderr)
        return 1
    enc = LabelEncoder().fit(present)
    y = enc.transform(tokens).astype(np.int64)
    _, counts = np.unique(y, return_counts=True)
    print(f"loaded {len(stems)} samples / Z={Z.shape}; goal_state classes="
          f"{list(enc.classes_)} counts={counts.tolist()}")

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

    if args.cv:
        res = nested_cv_probe_one_factor(
            Z, y, factor_idx=_GOAL_STATE_IDX, n_factors=len(INTENT_FIELDS),
            d_slot=args.d_slot, seed=args.seed, standardize_input=args.standardize)
        print(f"  CV[goal_state] probe={res['probe_acc_mean']:.3f}"
              f"±{res['probe_acc_std']:.3f} majority={res['majority_class_acc']:.3f} "
              f"shuffled={res['shuffled_features_acc']:.3f}")

    head = IntentHead(z_dim=Z_train.shape[1], n_factors=len(INTENT_FIELDS),
                      d_slot=args.d_slot, hidden=args.hidden, seed=args.seed)
    train_intent_head_joint(
        head, Z_train, {_GOAL_STATE_IDX: (y, len(enc.classes_))},
        n_epochs=args.n_epochs_intent, lr=args.lr, seed=args.seed)
    head.eval()
    with torch.no_grad():
        G = head(torch.from_numpy(Z_train)).numpy()
    centroids = build_factor_centroids(G, {_GOAL_STATE_IDX: y})

    pred = np.array([decode_slot(G[b, _GOAL_STATE_IDX], centroids[_GOAL_STATE_IDX])
                     for b in range(G.shape[0])])
    print(f"  train nearest-centroid[goal_state] = {float((pred == y).mean()):.3f}")

    revise = ReviseHead(d_slot=args.d_slot, hidden=args.hidden, seed=args.seed)
    label_tokens = {_GOAL_STATE_IDX: tuple(enc.classes_)}
    pack = LatentPack(
        intent_head=head, revise_head=revise,
        centroids=centroids, label_tokens=label_tokens, attribution_head=None,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_latent_pack(pack, args.out_dir)
    if args.standardize:
        np.savez(args.out_dir / "scaler.npz", mean=scaler_mean, scale=scaler_scale)
        print("  saved scaler.npz")
    print(f"saved goal_state LatentPack to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
