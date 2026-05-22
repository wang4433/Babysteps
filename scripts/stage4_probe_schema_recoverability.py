"""Stage-4 Milestone 1 — schema-recoverability probe CLI.

Reads one or more Stage-0 `samples.jsonl` files, groups episodes by task, and
for each of the six discrete intent factors (schemas.INTENT_FIELDS) trains a
linear probe to predict the factor's initial-intent label from demo-evidence
features alone. Writes a per-task per-factor accuracy table as JSON (machine)
and Markdown (human) into `--out-dir`.

Firewall: features come only from DemoEvidence-shaped fields (see
babysteps/stage4/features.py); the label `y` is read from
execution.initial_intent[factor]. Probes are trained per task, never pooled
across tasks (factor value sets differ per task).

Example:
    python scripts/stage4_probe_schema_recoverability.py \\
        --jsonl datasets/stage0_baselines/babysteps_selective/PushCube-v1/samples.jsonl \\
        --jsonl datasets/stage0_baselines/babysteps_selective/PickCube-v1/samples.jsonl \\
        --jsonl datasets/stage0_baselines/babysteps_selective/StackCube-v1/samples.jsonl \\
        --out-dir reports/stage4/schema_recoverability/ --seed 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder

# Make the project root importable without `pip install -e .` (mirrors
# scripts/stage0_summarize.py).
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from babysteps.schemas import INTENT_FIELDS, EpisodeRecord  # noqa: E402
from babysteps.stage4.features import extract_episode_features  # noqa: E402
from babysteps.stage4.probe import train_probe  # noqa: E402
from babysteps.stage4.report import (  # noqa: E402
    GATE_THRESHOLD,
    build_report,
    markdown_table,
)

# Default = the three primary babysteps_selective baselines (Task 6 Step 1).
_PRIMARY = [
    _ROOT / "datasets/stage0_baselines/babysteps_selective" / task / "samples.jsonl"
    for task in ("PushCube-v1", "PickCube-v1", "StackCube-v1")
]


def _load_records(paths: list[Path]) -> list[dict]:
    """Round-trip each JSONL line through the snapshot guard into a dict."""
    records: list[dict] = []
    for path in paths:
        with Path(path).open() as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(EpisodeRecord.from_jsonl_line(line).to_dict())
    return records


def _probe_rows(records: list[dict], *, seed: int) -> list[dict]:
    """One probe per (task, factor); features built once per task."""
    by_task: dict[str, list[dict]] = {}
    for rec in records:
        by_task.setdefault(rec["task"], []).append(rec)

    rows: list[dict] = []
    for task in sorted(by_task):
        recs = by_task[task]
        X = np.stack([extract_episode_features(r) for r in recs])
        for factor in INTENT_FIELDS:
            labels = [r["execution"]["initial_intent"][factor] for r in recs]
            y = LabelEncoder().fit_transform(labels)
            out = train_probe(X, y, seed=seed)
            out["task"] = task
            out["factor"] = factor
            rows.append(out)
    return rows


def _render_markdown(report: dict) -> str:
    g = report["gate"]
    header = [
        "# Stage-4 Schema-Recoverability Probe",
        "",
        f"Gate: non-trivial cells must reach probe_acc_mean >= {GATE_THRESHOLD:.2f}.",
        (f"Cells: {g['n_total']} total | {g['n_passing']} pass | "
         f"{g['n_failing']} fail | {g['n_trivial']} trivially constant."),
        "",
    ]
    md = "\n".join(header) + "\n" + markdown_table(report)
    if g["n_failing"]:
        failing = ", ".join(f"{t}/{f}" for t, f in g["failing_cells"])
        md += f"\n**Failing cells (need a notes.md explanation):** {failing}\n"
    return md


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Stage-4 M1 schema-recoverability probe."
    )
    p.add_argument(
        "--jsonl", action="append", type=Path,
        help="EpisodeRecord JSONL path (repeatable). "
             "Default = the three primary babysteps_selective files.",
    )
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    paths = args.jsonl if args.jsonl else _PRIMARY
    records = _load_records(paths)
    rows = _probe_rows(records, seed=args.seed)
    report = build_report(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "schema_recoverability.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (args.out_dir / "schema_recoverability.md").write_text(
        _render_markdown(report) + "\n"
    )

    g = report["gate"]
    print(f"wrote {args.out_dir}/schema_recoverability.{{json,md}}")
    print(f"gate: {g['n_passing']} pass / {g['n_failing']} fail / "
          f"{g['n_trivial']} trivial (of {g['n_total']} cells)")
    if g["n_failing"]:
        failing = ", ".join(f"{t}/{f}" for t, f in g["failing_cells"])
        print(
            f"WARNING: {g['n_failing']} non-trivial cell(s) below "
            f"{GATE_THRESHOLD:.2f}: {failing}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
