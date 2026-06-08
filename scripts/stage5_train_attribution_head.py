"""Stage-5 — train + save the distilled multimodal AttributionHead (step 4).

Sim-free (CPU torch). Trains ONE head with modality dropout on the PokeCube LOTO
geometry (clean contact_region failures + Class-A object_motion hard negatives)
and saves a checkpoint for ``DistilledAttributor.from_pack`` -> the recovery-gate
GPU run (``stage5_pokecube_maintable_eval.py --distilled-head``). The head is
tiny; the checkpoint lives under ``models/`` (gitignored).

On the DEPLOYED PokeCube loop the failures are clean contact_region
mis-groundings (object_motion correct), so the head attributes contact_region
(via the inconsistent (object_motion, contact_region) context) and the shared
scorer recovers. The hard negatives are a stress probe, not the deployed
distribution (see attribution_dataset honesty note).

Example::

    python scripts/stage5_train_attribution_head.py \\
        --out models/stage5/attribution_head/head.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from babysteps.stage5.attribution_dataset import make_dataset  # noqa: E402
from babysteps.stage5.attribution_head import (  # noqa: E402
    evaluate_attribution,
    save_attribution_head,
    train_attribution_head,
)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-per-case", type=int, default=64)
    p.add_argument("--noise", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--default-mask", default="multimodal",
                   help="mask DistilledAttributor uses at inference.")
    p.add_argument("--no-hardneg", action="store_true",
                   help="train on clean contact_region only (residual-shortcut "
                        "baseline; will FAIL the hard-negative split).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path,
                   default=Path("models/stage5/attribution_head/head.pt"))
    args = p.parse_args(argv)

    train = make_dataset(n_per_case=args.n_per_case, noise=args.noise,
                         include_hardneg=not args.no_hardneg, seed=args.seed)
    head = train_attribution_head(train, modality_dropout=args.dropout,
                                  epochs=args.epochs, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_attribution_head(head, args.out, default_mask=args.default_mask)

    test = make_dataset(n_per_case=16, noise=args.noise, seed=args.seed + 100)
    print(f"=== trained on {len(train)} examples (hardneg="
          f"{not args.no_hardneg}); sanity on held-out noise ===")
    for m in ("residual_only", "multimodal"):
        r = evaluate_attribution(head, test, m)
        print(f"  {m:<13} acc={r['accuracy']:.3f} "
              f"clean={r['by_kind'].get('clean')} "
              f"hardneg={r['by_kind'].get('hardneg_objmotion')}")
    print(f"saved -> {args.out}  (default_mask={args.default_mask})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
