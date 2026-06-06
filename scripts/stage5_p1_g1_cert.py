"""Stage-5 P1 Gate G1 — IntentHead probe recoverability on DINOv2 features.

Mirrors scripts/stage4_m2a_g1_cert.py but consumes cached DINOv2
features (Z = (768,) float32 per seed) in place of the 20-dim
handcrafted vector. Same nested-CV protocol, same three-way report
schema, same gate threshold.

Example::

    python scripts/stage5_p1_g1_cert.py \\
        --jsonl datasets/stage4/varied_intent/PushCube-v1/samples.jsonl \\
        --features-dir datasets/stage5/varied_intent/PushCube-v1/features/ \\
        --jsonl datasets/stage4/varied_intent/StackCube-v1/samples.jsonl \\
        --features-dir datasets/stage5/varied_intent/StackCube-v1/features/ \\
        --out-dir reports/stage5/p1_vision_g1/ --seed 0
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
from babysteps.stage4.intent_head import nested_cv_probe_one_factor  # noqa: E402
from babysteps.stage4.report import (  # noqa: E402
    GATE_THRESHOLD,
    build_report,
    markdown_table,
)


def _seed_from_record(rec: dict) -> int:
    return int(rec["episode_id"].split("_")[-1])


def _load_one_task(
    jsonl: Path, features_dir: Path, *, feature_suffix: str = "dinov2"
) -> tuple[list[dict], np.ndarray]:
    """Load records and stack their cached encoder features in jsonl order.

    `feature_suffix` selects the cached file (`seed_NNNN_<suffix>.npy`) so the
    same probe runs apples-to-apples across encoders (dinov2 / vjepa21 / ...).
    """
    records: list[dict] = []
    with jsonl.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(EpisodeRecord.from_jsonl_line(line).to_dict())
    Z_rows: list[np.ndarray] = []
    for rec in records:
        seed = _seed_from_record(rec)
        feat = features_dir / f"seed_{seed:04d}_{feature_suffix}.npy"
        Z_rows.append(np.load(feat))
    Z = np.stack(Z_rows).astype(np.float32)
    return records, Z


def _probe_rows(
    records: list[dict], Z: np.ndarray, *,
    n_factors: int, d_slot: int, hidden: int, n_epochs: int, lr: float, seed: int,
    standardize: bool = False,
) -> list[dict]:
    """One trained-encoder probe per factor on the supplied Z."""
    rows: list[dict] = []
    task = records[0]["task"]
    for factor_idx, factor in enumerate(INTENT_FIELDS):
        labels = [r["execution"]["initial_intent"][factor] for r in records]
        y = LabelEncoder().fit_transform(labels)
        out = nested_cv_probe_one_factor(
            Z, y,
            factor_idx=factor_idx, n_factors=n_factors,
            d_slot=d_slot, n_epochs=n_epochs, lr=lr, seed=seed,
            standardize_input=standardize,
        )
        out["task"] = task
        out["factor"] = factor
        out["_z_dim"] = int(Z.shape[1])
        rows.append(out)
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jsonl", type=Path, action="append", required=True,
                   help="One varied-intent samples.jsonl; repeat per task.")
    p.add_argument("--features-dir", type=Path, action="append", required=True,
                   help="Matching DINOv2 features directory; repeat per task.")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--d-slot", type=int, default=32)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--n-epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--pool", type=str, default="spatial_mean",
                   help="Pool strategy used by stage5_cache_dinov2.py to produce "
                        "the cached features (informational for the report header).")
    p.add_argument("--feature-suffix", type=str, default="dinov2",
                   help="Cached-feature filename suffix (seed_NNNN_<suffix>.npy); "
                        "e.g. 'vjepa21' for the V-JEPA 2.1 run.")
    p.add_argument("--encoder-label", type=str, default="DINOv2 ViT-B/14",
                   help="Encoder name for the report header.")
    p.add_argument("--no-narrative", action="store_true",
                   help="Omit the DINOv2-specific falsification-log/interpretation "
                        "prose (use for non-DINOv2 encoders; hand-author the "
                        "comparison/verdict separately).")
    p.add_argument("--standardize", action="store_true",
                   help="StandardScale the IntentHead input Z (train-fold fit) — "
                        "fair across encoders with differing feature norms; the "
                        "committed default (off) leaves prior numbers unchanged. "
                        "See reports/stage5/vjepa_object_motion/FINDINGS.md.")
    args = p.parse_args(argv)

    if len(args.jsonl) != len(args.features_dir):
        print("--jsonl and --features-dir must be repeated in matching pairs",
              file=sys.stderr)
        return 1

    all_rows: list[dict] = []
    for jl, fd in zip(args.jsonl, args.features_dir):
        records, Z = _load_one_task(jl, fd, feature_suffix=args.feature_suffix)
        rows = _probe_rows(
            records, Z,
            n_factors=6, d_slot=args.d_slot, hidden=args.hidden,
            n_epochs=args.n_epochs, lr=args.lr, seed=args.seed,
            standardize=args.standardize,
        )
        all_rows.extend(rows)

    report = build_report(all_rows)
    gate_pass = report["gate"]["n_failing"] == 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2))
    # Pool / dim come from S3's cache (--pool flag of stage5_cache_dinov2.py).
    # Pull dim from the actual feature shape so the header can't drift.
    z_dim_used = (
        int(all_rows[0]["_z_dim"]) if all_rows and "_z_dim" in all_rows[0] else "?"
    )
    md_parts = [
        f"# Stage-5 P1 — Vision-grounded G1 ({args.encoder_label})",
        "",
        f"Input Z: {z_dim_used}-dim {args.encoder_label} features "
        f"({args.pool} pool over demo frames).",
        f"IntentHead: F=6, d_slot={args.d_slot}, hidden={args.hidden}, "
        f"n_epochs={args.n_epochs}, lr={args.lr}.",
        "Outer CV: per-fold IntentHead training; frozen LogisticRegression "
        "on G_train, evaluated on G_test.",
        "",
        markdown_table(report),
        "",
        f"**Gate:** all geometric cells >= {GATE_THRESHOLD:.0%} "
        f"(margin {report['gate']['margin']:.0%} over majority & shuffled) -> "
        f"**{'PASS' if gate_pass else 'FAIL'}**",
        "",
        f"Cells: {report['gate']['n_total']} total | "
        f"{report['gate']['n_geometric']} geometric "
        f"({report['gate']['n_passing']} pass / "
        f"{report['gate']['n_failing']} fail) | "
        f"{report['gate']['n_label_identity']} label-identity | "
        f"{report['gate']['n_trivial']} trivially constant.",
    ]
    narrative = [
        "",
        "## Falsification log",
        "",
        "Three pooling ablations were pre-registered against `_pool_cls` "
        "(`babysteps/stage4/vision_features.py`). Only one (spatial_mean) is "
        "cached on disk in `datasets/stage5/varied_intent/<task>/features/`; "
        "the others were tried in sequence and the cache was overwritten "
        "between runs. Numbers below are from those runs (PushCube and "
        "StackCube `object_motion` rows of the gate table).",
        "",
        "| pool | dim | PushCube object_motion | StackCube object_motion |",
        "| --- | --- | --- | --- |",
        "| cls_mean | 768 | 0.95 ± 0.22 (weak — near-binary labels) | "
        "0.30 ± 0.22 (FAIL) |",
        "| cls_first_last | 1536 | 0.95 ± 0.22 (clean PASS) | "
        "0.23 ± 0.09 (below chance — d ≫ n at d=1536, n=40) |",
        "| **spatial_mean** | **768** | **0.95 ± 0.22** | "
        "**0.42 ± 0.10** (best of failed; no d-vs-n pathology) |",
        "",
        "## Interpretation",
        "",
        "PushCube `object_motion` recovers cleanly under any pool, including "
        "`cls_mean` — but PushCube's 3-class label set collapses to a near-"
        "binary +x/-x split (one outlier seed produces the third class), so "
        "the 0.95 is a weaker pass than the headline suggests.",
        "",
        "StackCube `object_motion` is the cleaner test: 4 balanced classes "
        "(10/10/10/10), labels defined by the **relative** direction from "
        "cubeA to cubeB. Both cubes are visible at the first and last frames "
        "(cubeA just translates on top of cubeB), so `cls_first_last`'s "
        "between-frame delta does not isolate a single moving object. "
        "Increasing dim to 1536 with n=40 pushes the linear probe below "
        "chance.",
        "",
        "`spatial_mean` is the best of the three ablations: 768-dim avoids "
        "the d ≫ n pathology, mean-pooled patch tokens retain more local "
        "structure than CLS alone. It still falls to 0.42 — well below the "
        "0.90 gate.",
        "",
        "**Falsifiable finding:** Frozen DINOv2 with a linear nested-CV "
        "probe supports single-object motion direction when temporal "
        "endpoints are exposed (PushCube `cls_first_last` 0.95 PASS), but "
        "fails on two-object relational direction under n=40 (StackCube "
        "under all three pools, best 0.42). The bottleneck is not only "
        "temporal pooling; it is object-centric relational representation, "
        "possibly compounded by data scale.",
        "",
        "## Implications for S5",
        "",
        "S5 scope narrows to PushCube end-to-end only. StackCube is logged "
        "as an open relational-factor failure of frozen-feature G1, not "
        "papered over with mixed per-task pooling. The remaining "
        "ablations spec §6 listed (R3M, encoder swap) are not pursued — "
        "they would not address the relational-representation bottleneck "
        "diagnosed here.",
    ]
    if not args.no_narrative:
        md_parts.extend(narrative)
    md = "\n".join(md_parts)
    if not gate_pass:
        failing = ", ".join(f"{t}/{f}" for t, f in report["gate"]["failing_cells"])
        md += f"\n\n**Failing geometric cells:** {failing}\n"
    (args.out_dir / "report.md").write_text(md + "\n")
    print(md)
    return 0 if gate_pass else 2


if __name__ == "__main__":
    sys.exit(main())
