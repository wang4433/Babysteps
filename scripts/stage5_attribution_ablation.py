"""Stage-5 — distilled attribution: input-ablation table (build-order step 2+4).

Sim-free (CPU torch, no GPU/Vulkan, no gitignored artifacts). Produces the
contact-region BASELINE + input ablation the user ordered:

* Step 2 (residual-only baseline): a head TRAINED residual-only solves clean
  PokeCube contact_region failures (the positional cue) — but collapses on the
  hard negatives, motivating the multimodal interface.
* Step 4 (input ablation): ONE head trained with modality dropout, evaluated
  under each modality MASK (residual-only / traj-only / ctx-only / res+ctx /
  multimodal), split by clean vs hard-negative.

Headline (honest framing): residual (and trajectory) CANNOT tell a Class-A hard
negative (misread object_motion) from a clean contact_region failure with a
byte-identical residual/face/trajectory — they collapse to 0.000 on the hard
negatives. The symbolic context modality (object_motion + contact_region tokens)
restores it. This proves a positional/residual shortcut is INSUFFICIENT and
symbolic intent context is NECESSARY.

It does NOT prove multimodal FUSION: ctx_only also reaches 1.000, because a
misread object_motion token IS the defining feature of the hard-negative class
BY CONSTRUCTION, so (object_motion, contact_region) is a perfect label separator.
Once ctx is present no modality combination is needed. Genuine fusion (no single
modality clean) lives in the GPU/real-pixels regime where intent tokens are
INFERRED, not given. This is a contact-region diagnostic proof-of-concept on 12
synthetic geometries, NOT general or learned attribution (see
attribution_dataset.py honesty note).

Example::

    python scripts/stage5_attribution_ablation.py \\
        --out reports/stage5/attribution_ablation/results.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from babysteps.stage5.attribution_dataset import make_dataset  # noqa: E402
from babysteps.stage5.attribution_head import (  # noqa: E402
    evaluate_attribution,
    train_attribution_head,
)

_EVAL_MASKS = ("residual_only", "traj_only", "ctx_only", "res_traj",
               "res_ctx", "multimodal")
# Honest single-arm baselines: heads TRAINED on one fixed mask. ctx_only is
# included so the table directly shows context SUFFICIENCY (expected ~1.000),
# making the "ctx alone solves it, not fusion" point explicit rather than implied.
_FIXED_ARMS = ("residual_only", "ctx_only", "res_ctx", "multimodal")


def run_ablation(*, n_per_case: int, noise: float, epochs: int,
                 dropout: float, seed: int) -> dict:
    train = make_dataset(n_per_case=n_per_case, noise=noise, seed=seed)
    test = make_dataset(n_per_case=max(8, n_per_case // 4), noise=noise,
                        seed=seed + 100)

    # ONE head, modality dropout -> evaluated under every mask (step 4).
    pooled = train_attribution_head(
        train, modality_dropout=dropout, epochs=epochs, seed=seed)
    dropout_table = {
        m: evaluate_attribution(pooled, test, m) for m in _EVAL_MASKS}

    # Honest fixed-arm baselines (step 2 residual-only is the key one).
    fixed_table = {}
    for arm in _FIXED_ARMS:
        head = train_attribution_head(
            train, fixed_mask=arm, epochs=epochs, seed=seed)
        fixed_table[arm] = evaluate_attribution(head, test, arm)

    return {
        "config": {"n_per_case": n_per_case, "noise": noise, "epochs": epochs,
                   "modality_dropout": dropout, "seed": seed,
                   "n_train": len(train), "n_test": len(test)},
        "dropout_head_by_mask": dropout_table,
        "fixed_arm_baselines": fixed_table,
    }


def _fmt(x) -> str:
    return f"{x:.3f}" if isinstance(x, (int, float)) else str(x)


def _print_table(result: dict) -> None:
    print("\n=== ONE head (modality dropout), evaluated under each mask ===")
    print(f"{'mask':<14} {'overall':>8} {'clean':>8} {'hardneg':>8}")
    for m, r in result["dropout_head_by_mask"].items():
        bk = r["by_kind"]
        print(f"{m:<14} {_fmt(r['accuracy']):>8} "
              f"{_fmt(bk.get('clean')):>8} "
              f"{_fmt(bk.get('hardneg_objmotion')):>8}")
    print("\n=== Fixed-arm baselines (head TRAINED on one mask) ===")
    print(f"{'trained-arm':<14} {'overall':>8} {'clean':>8} {'hardneg':>8}")
    for m, r in result["fixed_arm_baselines"].items():
        bk = r["by_kind"]
        print(f"{m:<14} {_fmt(r['accuracy']):>8} "
              f"{_fmt(bk.get('clean')):>8} "
              f"{_fmt(bk.get('hardneg_objmotion')):>8}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-per-case", type=int, default=64)
    p.add_argument("--noise", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path,
                   default=Path("reports/stage5/attribution_ablation/results.json"))
    args = p.parse_args()

    result = run_ablation(n_per_case=args.n_per_case, noise=args.noise,
                          epochs=args.epochs, dropout=args.dropout,
                          seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    _print_table(result)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
