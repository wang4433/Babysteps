"""Stage-5 B.2 — train the residual-conditioned ReviseHead for the latent loop.

The natural-loop ablation showed the goal-relative RESIDUAL (goal - final_cube,
observable at exec) is the load-bearing feedback: the hand rule
`feedback_residual` recovers 92.5% of 4-way mismatches vs 22.5% for
displacement-vector-only (reports/stage5/natural_loop). B.1 proved (sim-free) a
residual-conditioned ReviseHead can learn this. This script trains that head on
REAL seed-mismatch rollout tuples so the `latent_learned` reviser is a learned
slot-local edit in the vision-grounded latent space, NOT the hand-coded
`direction_to_face` rule.

Inputs:
  --pack-dir   a 4-way LatentPack (stage5_train_4way_pack.py): supplies the
               vision-grounded contact_region centroids + label_tokens.
  --tuples     tuples.jsonl from `stage5_natural_loop_eval.py --dump-tuples`:
               one {demo_face, correct_face, residual_xy, failure_predicate} per
               mismatched training episode. `correct_face` is a sim-derived
               TRAINING label (allowed off the demo->intent path, CLAUDE.md inv #4);
               `residual_xy` is the non-privileged execution feedback the head
               consumes at inference.

Target: head(centroid[demo_face], residual) -> centroid[correct_face], decoded
by nearest-centroid to the corrected face. Saves revise_head_residual.pt next to
the pack. Sim-free CPU-only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from babysteps.schemas import INTENT_FIELDS  # noqa: E402
from babysteps.stage4.latent_policy import load_latent_pack  # noqa: E402
from babysteps.stage4.revise_head import (  # noqa: E402
    FP_VECTOR_DIM_RESIDUAL, ReviseHead, save_revise_head, train_revise_head_l2,
    vectorize_failure_packet_residual,
)
from babysteps.stage4.slot_decode import decode_slot  # noqa: E402

_CONTACT_IDX = INTENT_FIELDS.index("contact_region")


def _load_tuples(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pack-dir", type=Path, required=True)
    p.add_argument("--tuples", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None,
                   help="Output .pt (default: <pack-dir>/revise_head_residual.pt).")
    p.add_argument("--factor", default="contact_region",
                   help="Revised factor whose slot the head edits.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--n-epochs", type=int, default=800)
    p.add_argument("--lr", type=float, default=1e-2)
    args = p.parse_args(argv)

    factor_idx = INTENT_FIELDS.index(args.factor)
    pack = load_latent_pack(args.pack_dir)
    if factor_idx not in pack.centroids:
        print(f"pack has no centroids for {args.factor}", file=sys.stderr)
        return 2
    centroids = pack.centroids[factor_idx]
    tokens = pack.label_tokens[factor_idx]
    tok2cls = {t: i for i, t in enumerate(tokens)}
    d_slot = int(next(iter(centroids.values())).shape[0])
    print(f"pack {args.pack_dir}: factor={args.factor} d_slot={d_slot} "
          f"classes={list(tokens)}")

    tuples = _load_tuples(args.tuples)
    g_pre, fp, g_tgt, y = [], [], [], []
    skipped = 0
    for t in tuples:
        df, cf = t.get("demo_face"), t.get("correct_face")
        if df not in tok2cls or cf not in tok2cls:
            skipped += 1
            continue
        rec = {"revision": {"factor": args.factor},
               "failure_packet": {
                   "failure_predicate": t.get("failure_predicate") or "direction_error"}}
        g_pre.append(centroids[tok2cls[df]])
        fp.append(vectorize_failure_packet_residual(rec, t["residual_xy"]))
        g_tgt.append(centroids[tok2cls[cf]])
        y.append(tok2cls[cf])
    if not g_pre:
        print("no usable tuples (all faces outside the pack vocab)", file=sys.stderr)
        return 1
    g_pre = np.stack(g_pre).astype(np.float32)
    fp = np.stack(fp).astype(np.float32)
    g_tgt = np.stack(g_tgt).astype(np.float32)
    y = np.asarray(y)
    print(f"trained on {len(g_pre)} tuples ({skipped} skipped); "
          f"fp_dim={fp.shape[1]} (expect {FP_VECTOR_DIM_RESIDUAL})")

    head = ReviseHead(d_slot=d_slot, fp_dim=FP_VECTOR_DIM_RESIDUAL,
                      hidden=args.hidden, seed=args.seed)
    train_revise_head_l2(head, g_pre, fp, g_tgt,
                         n_epochs=args.n_epochs, lr=args.lr, seed=args.seed)

    # Train-set nearest-centroid accuracy (sanity; held-out is the loop eval).
    import torch
    head.eval()
    with torch.no_grad():
        out = head(torch.tensor(g_pre), torch.tensor(fp)).numpy()
    pred = np.array([decode_slot(out[i], centroids) for i in range(len(out))])
    print(f"train nearest-centroid acc = {float((pred == y).mean()):.3f}")

    out_path = args.out or (args.pack_dir / "revise_head_residual.pt")
    save_revise_head(head, out_path)
    print(f"saved residual ReviseHead to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
