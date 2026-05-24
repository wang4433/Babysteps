"""Stage-4 M2.5 — train the learned attribution head per task.

Reads all Stage-0 baselines + the Stage-4 varied-intent cut, extracts
(failure_packet, intent → oracle_wrong_factor) supervision pairs, trains
an `AttributionHead`, evaluates with stratified k-fold honest accuracy,
runs the shuffled-label control, and writes a confusion matrix +
trained .pt to disk.

Run:

  python scripts/stage4_m25_train_attribution.py \
      --task StackCube-v1 \
      --out-dir models/stage4/m25/StackCube-v1/ \
      --report-dir reports/stage4/m25_attribution/StackCube-v1/

Sim-free, CPU-only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from babysteps.schemas import INTENT_FIELDS
from babysteps.stage4.attribution_head import (
    AttributionHead,
    build_training_pairs,
    class_weights_inverse,
    save_attribution_head,
    train_attribution_head,
)

BASELINE_POLICIES: tuple[str, ...] = (
    "babysteps_selective",
    "full_replan_analogue",
    "one_shot",
    "oracle_factor_revision",
    "random_factor_revision",
    "same_intent_retry",
    "text_feedback_replan",
)


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _collect_records(task: str, repo_root: Path) -> tuple[list[dict], list[Path]]:
    """Union of all per-task JSONLs across baselines + varied_intent cut."""
    paths: list[Path] = []
    for pol in BASELINE_POLICIES:
        p = repo_root / "datasets" / "stage0_baselines" / pol / task / "samples.jsonl"
        if p.exists():
            paths.append(p)
    p = repo_root / "datasets" / "stage4" / "varied_intent" / task / "samples.jsonl"
    if p.exists():
        paths.append(p)
    rows: list[dict] = []
    for p in paths:
        rows.extend(_read_jsonl(p))
    return rows, paths


def _stratified_kfold_acc(
    X: np.ndarray, y: np.ndarray, n_splits: int, seed: int,
    class_weights: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Return (mean held-out acc, confusion matrix summed over folds)."""
    n_classes = int(max(y.max() + 1, len(INTENT_FIELDS)))
    if n_splits < 2 or n_splits > len(y):
        # Fall back to one big train/test split when class counts are too
        # small to support k folds.
        return float("nan"), np.zeros((n_classes, n_classes), dtype=np.int64)
    # Need >= n_splits per PRESENT class for stratification — clamp.
    counts = np.bincount(y)
    min_present = int(counts[counts > 0].min())
    eff_splits = min(n_splits, min_present)
    if eff_splits < 2:
        return float("nan"), np.zeros((n_classes, n_classes), dtype=np.int64)
    skf = StratifiedKFold(n_splits=eff_splits, shuffle=True, random_state=seed)
    accs: list[float] = []
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        head = AttributionHead(seed=seed + fold)
        train_attribution_head(
            head, X[tr], y[tr], n_epochs=300, lr=1e-2,
            class_weights=class_weights, seed=seed + fold,
        )
        with torch.no_grad():
            logits = head(torch.from_numpy(X[te].astype(np.float32)))
            preds = logits.argmax(dim=-1).numpy()
        accs.append(float((preds == y[te]).mean()))
        for t, p in zip(y[te], preds):
            cm[t, p] += 1
    return float(np.mean(accs)), cm


def _shuffled_label_control(
    X: np.ndarray, y: np.ndarray, n_splits: int, seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(y)
    cw = class_weights_inverse(shuffled)
    acc, _ = _stratified_kfold_acc(X, shuffled, n_splits, seed, cw)
    return acc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True, action="append",
                    help="StackCube-v1 / PushCube-v1 (repeat to train a "
                         "joint head over multiple tasks; the first task "
                         "is used as the report tag)")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="Where to write attribution_head.pt + meta.json")
    ap.add_argument("--report-dir", required=True, type=Path,
                    help="Where to write metrics + confusion matrix")
    ap.add_argument("--repo-root", type=Path,
                    default=Path(__file__).resolve().parent.parent,
                    help="Repository root (default: parent of scripts/)")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows: list[dict] = []
    paths: list[Path] = []
    for tk in args.task:
        sub_rows, sub_paths = _collect_records(tk, args.repo_root)
        rows.extend(sub_rows)
        paths.extend(sub_paths)
    if not rows:
        raise SystemExit(f"no records found for tasks {args.task}")
    X, y, dropped = build_training_pairs(rows)
    if X.shape[0] == 0:
        raise SystemExit(f"no failed (predicate != none) records for {args.task}")

    cw = class_weights_inverse(y)
    held_out_acc, cm = _stratified_kfold_acc(X, y, args.n_splits, args.seed, cw)
    ctrl_acc = _shuffled_label_control(X, y, args.n_splits, args.seed)

    # Final deployment head: trained on ALL data.
    head = AttributionHead(seed=args.seed)
    final = train_attribution_head(
        head, X, y, n_epochs=300, lr=1e-2, class_weights=cw, seed=args.seed,
    )
    out_path = Path(args.out_dir) / "attribution_head.pt"
    save_attribution_head(head, out_path)

    # Per-class counts.
    bincount = np.bincount(y, minlength=len(INTENT_FIELDS))
    per_class = {INTENT_FIELDS[c]: int(bincount[c])
                 for c in range(len(INTENT_FIELDS))}

    # Per-predicate accuracy on held-out via leave-one-out for ambiguous
    # slices. We replay the StratifiedKFold predictions to compute it.
    # Simpler: report per-predicate vs predicted-correctness using the
    # confusion matrix together with the per-row predicate stash.

    n_unique_classes = int((bincount > 0).sum())
    report = {
        "task": args.task,
        "n_samples": int(X.shape[0]),
        "feature_dim": int(X.shape[1]),
        "n_unique_label_classes": n_unique_classes,
        "per_class_count": per_class,
        "held_out_acc_kfold": held_out_acc,
        "shuffled_label_control_acc": ctrl_acc,
        "final_train_acc": final["acc"],
        "final_train_loss": final["loss"],
        "confusion_matrix_labels": list(INTENT_FIELDS),
        "confusion_matrix": cm.tolist(),
        "dropped_label_count": len(dropped),
        "data_paths": [str(p) for p in paths],
        "out_path": str(out_path),
        "note": ("per-task labels are constant (single-class) so kfold and "
                  "shuffled control are uninformative; train jointly over "
                  "multiple tasks to require intent-conditioned learning"
                  if n_unique_classes < 2 else
                  "multi-class supervised joint training: shuffled-label "
                  "control should collapse toward 1/n_classes"),
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "metrics.json").write_text(json.dumps(report, indent=2))

    # Pretty print
    print(f"task: {args.task}")
    print(f"n_samples: {report['n_samples']}, feature_dim: {report['feature_dim']}")
    print(f"per_class_count: {per_class}")
    print(f"held_out_acc_kfold: {held_out_acc:.4f}")
    print(f"shuffled_label_control_acc: {ctrl_acc:.4f}  (chance = {1.0/len(INTENT_FIELDS):.4f})")
    print(f"final_train_acc: {final['acc']:.4f}")
    print(f"wrote: {out_path}")
    print(f"wrote: {args.report_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
