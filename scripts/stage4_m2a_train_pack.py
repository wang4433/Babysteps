"""Stage-4 M2a — train + save a LatentPack for one task.

Trains IntentHead jointly across all non-trivial factors on the FULL
varied-intent cut for a task (no CV split — this is a deployment train,
not a cert; the cert numbers came from `scripts/stage4_m2a_a2_eval.py`).
Then builds per-factor centroids and trains ReviseHead on the
counterfactual (g_pre, fp_vec) → centroid[revised_class] pairs.

Saves the resulting `LatentPack` via
`babysteps.stage4.latent_policy.save_latent_pack` so the A3 eval script
can load + run latent revision in `episode.run_episode`.

Sim-free, CPU-only, ~15s wall per task.

Example::

    python scripts/stage4_m2a_train_pack.py \\
        --jsonl datasets/stage4/varied_intent/PushCube-v1/samples.jsonl \\
        --out-dir models/stage4/m2a/PushCube-v1/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from babysteps.schemas import INTENT_FIELDS, EpisodeRecord  # noqa: E402
from babysteps.stage4.features import extract_episode_features  # noqa: E402
from babysteps.stage4.intent_head import (  # noqa: E402
    IntentHead, train_intent_head_joint,
)
from babysteps.stage4.latent_policy import (  # noqa: E402
    LatentPack, save_latent_pack,
)
from babysteps.stage4.revise_head import (  # noqa: E402
    FP_VECTOR_DIM, ReviseHead,
    train_revise_head_l2, vectorize_failure_packet,
)
from babysteps.stage4.slot_decode import build_factor_centroids  # noqa: E402


def _load_records(path: Path) -> list[dict]:
    with path.open() as f:
        return [EpisodeRecord.from_jsonl_line(l).to_dict()
                for l in f if l.strip()]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Train + save a Stage-4 M2a LatentPack.")
    p.add_argument("--jsonl", type=Path, required=True,
                   help="One task's varied-intent samples.jsonl.")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--d-slot", type=int, default=16)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--n-epochs-intent", type=int, default=300)
    p.add_argument("--n-epochs-revise", type=int, default=600)
    p.add_argument("--lr", type=float, default=1e-2)
    args = p.parse_args(argv)

    records = _load_records(args.jsonl)
    print(f"loaded {len(records)} records from {args.jsonl}")
    Z = np.stack([extract_episode_features(r) for r in records]).astype(np.float32)

    # Build per-factor encoders + labels including any revision targets so
    # the encoder is exposed to ALL classes that may need centroids.
    encoders: dict[int, LabelEncoder] = {}
    labels_per_factor: dict[int, tuple[np.ndarray, int]] = {}
    for fi, factor in enumerate(INTENT_FIELDS):
        present_vals = set(r["execution"]["initial_intent"][factor] for r in records)
        # Add revision new_value AND old_value too (defensive)
        for r in records:
            rv = r.get("revision")
            if rv and rv["factor"] == factor:
                present_vals.add(rv["new_value"])
                present_vals.add(rv["old_value"])
        present_vals = sorted(present_vals)
        if len(present_vals) < 2:
            continue
        enc = LabelEncoder().fit(present_vals)
        y = enc.transform(
            [r["execution"]["initial_intent"][factor] for r in records]
        ).astype(np.int64)
        encoders[fi] = enc
        labels_per_factor[fi] = (y, len(enc.classes_))
    print(f"non-trivial supervised factors: "
          f"{[INTENT_FIELDS[fi] for fi in labels_per_factor]}")

    # 1. Joint train IntentHead
    head = IntentHead(z_dim=Z.shape[1], n_factors=len(INTENT_FIELDS),
                      d_slot=args.d_slot, hidden=args.hidden, seed=args.seed)
    train_intent_head_joint(
        head, Z, labels_per_factor,
        n_epochs=args.n_epochs_intent, lr=args.lr, seed=args.seed,
    )
    print(f"joint-trained IntentHead ({args.n_epochs_intent} epochs)")

    # 2. Build centroids
    head.eval()
    with torch.no_grad():
        G = head(torch.from_numpy(Z)).numpy()
    centroids = build_factor_centroids(
        G, {fi: y for fi, (y, _) in labels_per_factor.items()},
    )
    print(f"built centroids for factors "
          f"{[INTENT_FIELDS[fi] for fi in centroids]}: "
          f"{ {INTENT_FIELDS[fi]: list(c.keys()) for fi, c in centroids.items()} }")

    # 3. Train ReviseHead on certable counterfactual pairs
    revisions = []
    for i, r in enumerate(records):
        rv = r.get("revision")
        if not rv or rv["factor"] not in INTENT_FIELDS:
            continue
        fi = INTENT_FIELDS.index(rv["factor"])
        if fi not in encoders:
            continue
        try:
            new_class = int(encoders[fi].transform([rv["new_value"]])[0])
        except ValueError:
            continue
        if fi not in centroids or new_class not in centroids[fi]:
            continue
        revisions.append({
            "i": i, "fi": fi, "new_class": new_class,
            "fp_vec": vectorize_failure_packet(r),
        })

    revise = ReviseHead(d_slot=args.d_slot, hidden=args.hidden, seed=args.seed)
    if revisions:
        g_pre = np.stack([G[rv["i"], rv["fi"]] for rv in revisions]).astype(np.float32)
        fp = np.stack([rv["fp_vec"] for rv in revisions]).astype(np.float32)
        g_tgt = np.stack([centroids[rv["fi"]][rv["new_class"]]
                          for rv in revisions]).astype(np.float32)
        train_revise_head_l2(
            revise, g_pre, fp, g_tgt,
            n_epochs=args.n_epochs_revise, lr=args.lr, seed=args.seed,
        )
        print(f"trained ReviseHead on {len(revisions)} certable revision pairs")
    else:
        print("WARNING: 0 certable revisions; ReviseHead is at random init")

    # 4. Pack + save
    label_tokens = {fi: tuple(enc.classes_) for fi, enc in encoders.items()
                    if fi in centroids}
    pack = LatentPack(
        intent_head=head, revise_head=revise,
        centroids=centroids, label_tokens=label_tokens,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_latent_pack(pack, args.out_dir)
    print(f"saved LatentPack to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
