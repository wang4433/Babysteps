"""Stage-4 M2a Gate G1 — IntentHead probe recoverability CLI.

Trains a fresh IntentHead per outer fold of each (task, factor) cell on the
existing varied-intent cut, then runs the same `babysteps.stage4.report`
three-way cert on the held-out G-probe accuracies. Output shape matches
`scripts/stage4_probe_schema_recoverability.py` so the cert headline is
directly comparable.

This is the M2a Stage-A1 entrance gate per
`docs/superpowers/specs/2026-05-23-stage4-m2-slot-encoder-design.md`.

Example::

    python scripts/stage4_m2a_g1_cert.py \\
        --jsonl datasets/stage4/varied_intent/PushCube-v1/samples.jsonl \\
        --jsonl datasets/stage4/varied_intent/StackCube-v1/samples.jsonl \\
        --out-dir reports/stage4/m2a_intent_head_g1/ --seed 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from babysteps.schemas import INTENT_FIELDS, EpisodeRecord  # noqa: E402
from babysteps.stage4.features import extract_episode_features  # noqa: E402
from babysteps.stage4.intent_head import nested_cv_probe_one_factor  # noqa: E402
from babysteps.stage4.report import (  # noqa: E402
    GATE_THRESHOLD,
    build_report,
    markdown_table,
)

_PRIMARY = [
    _ROOT / "datasets/stage4/varied_intent" / task / "samples.jsonl"
    for task in ("PushCube-v1", "StackCube-v1")
]


def _load_records(paths: list[Path]) -> list[dict]:
    records: list[dict] = []
    for path in paths:
        with Path(path).open() as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(EpisodeRecord.from_jsonl_line(line).to_dict())
    return records


def _probe_rows(
    records: list[dict], *,
    n_factors: int, d_slot: int, n_epochs: int, lr: float, seed: int,
) -> list[dict]:
    """One trained-encoder probe per (task, factor); Z built once per task."""
    by_task: dict[str, list[dict]] = {}
    for rec in records:
        by_task.setdefault(rec["task"], []).append(rec)

    rows: list[dict] = []
    for task in sorted(by_task):
        recs = by_task[task]
        Z = np.stack([extract_episode_features(r) for r in recs])
        for factor_idx, factor in enumerate(INTENT_FIELDS):
            labels = [r["execution"]["initial_intent"][factor] for r in recs]
            y = LabelEncoder().fit_transform(labels)
            out = nested_cv_probe_one_factor(
                Z, y,
                factor_idx=factor_idx,
                n_factors=n_factors,
                d_slot=d_slot,
                n_epochs=n_epochs,
                lr=lr,
                seed=seed,
            )
            out["task"] = task
            out["factor"] = factor
            rows.append(out)
    return rows


def _render_markdown(report: dict, *, n_epochs: int, lr: float, d_slot: int) -> str:
    g = report["gate"]
    header = [
        "# Stage-4 M2a — IntentHead G1 (probe recoverability)",
        "",
        (f"Input Z: 20-dim handcrafted demo features "
         f"(`babysteps/stage4/features.py`)."),
        (f"IntentHead: F=6, d_slot={d_slot}, hidden=64, n_epochs={n_epochs}, "
         f"lr={lr}."),
        (f"Outer CV: per-fold IntentHead training; frozen LogisticRegression "
         f"on G_train, evaluated on G_test."),
        "",
        (f"Gate: geometric cells must reach probe_acc_mean >= "
         f"{GATE_THRESHOLD:.2f} AND beat chance & shuffled each by "
         f"{g['margin']:.2f}."),
        (f"Cells: {g['n_total']} total | {g['n_geometric']} geometric "
         f"({g['n_passing']} pass / {g['n_failing']} fail) | "
         f"{g['n_label_identity']} label-identity | {g['n_trivial']} "
         f"trivially constant."),
        "",
    ]
    md = "\n".join(header) + "\n" + markdown_table(report)
    if g["n_failing"]:
        failing = ", ".join(f"{t}/{f}" for t, f in g["failing_cells"])
        md += f"\n**Failing geometric cells:** {failing}\n"
    return md


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Stage-4 M2a G1 IntentHead probe cert.")
    p.add_argument("--jsonl", action="append", type=Path,
                   help="EpisodeRecord JSONL path (repeatable). "
                        "Default = the two primary varied-intent cuts.")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--d-slot", type=int, default=16)
    p.add_argument("--n-factors", type=int, default=6)
    args = p.parse_args(argv)

    paths = args.jsonl if args.jsonl else _PRIMARY
    records = _load_records(paths)
    rows = _probe_rows(
        records,
        n_factors=args.n_factors, d_slot=args.d_slot,
        n_epochs=args.n_epochs, lr=args.lr, seed=args.seed,
    )
    report = build_report(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "schema_recoverability.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (args.out_dir / "schema_recoverability.md").write_text(
        _render_markdown(
            report, n_epochs=args.n_epochs, lr=args.lr, d_slot=args.d_slot,
        ) + "\n"
    )

    g = report["gate"]
    print(f"wrote {args.out_dir}/schema_recoverability.{{json,md}}")
    print(f"G1: {g['n_passing']} pass / {g['n_failing']} fail "
          f"(of {g['n_geometric']} geometric); "
          f"{g['n_label_identity']} label-identity, {g['n_trivial']} trivial")
    if g["n_failing"]:
        failing = ", ".join(f"{t}/{f}" for t, f in g["failing_cells"])
        print(f"WARNING: {g['n_failing']} geometric cell(s) below G1: {failing}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
