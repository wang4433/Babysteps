"""Dataset-level metrics + Markdown/JSON report writer.

`compute_metrics` aggregates a list of `EpisodeRecord`s into the rows from
goal.md §"Required Metrics". `write_report` mirrors the Pick4Pass MVP report
format (`Code/checkpoints/pushcube_mvp/report.md`).

Acceptance gate: `delta_pp >= 10.0` (retry success rate minus initial-attempt
success rate, in percentage points) — Pick4Pass M-BABY-1's bar, kept here so
the bar travels with the data contract rather than the schema.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from babysteps.schemas import EpisodeRecord


ACCEPTANCE_DELTA_PP_THRESHOLD: float = 10.0


def _safe_div(a: float, b: float) -> float:
    return a / b if b > 0 else 0.0


def compute_metrics(records: Iterable[EpisodeRecord]) -> dict:
    records_list = list(records)
    n_total = len(records_list)

    n_initial_success = sum(
        1 for r in records_list if r.metrics.get("initial_success") is True
    )
    n_with_revision = sum(1 for r in records_list if r.revision is not None)
    n_retry_success = sum(
        1 for r in records_list
        if r.metrics.get("retry_success") is True
    )
    n_final_success = n_initial_success + n_retry_success

    initial_rate = _safe_div(n_initial_success, n_total)
    retry_rate = _safe_div(n_retry_success, n_with_revision)
    final_rate = _safe_div(n_final_success, n_total)
    delta_pp = (retry_rate - initial_rate) * 100.0

    # Per-revision diagnostics.
    n_attribution_evaluated = 0
    n_attribution_correct = 0
    n_frozen_preserved = 0
    n_frozen_evaluated = 0
    n_non_regression_clean = 0
    for r in records_list:
        if r.revision is None:
            continue
        m = r.metrics
        if m.get("factor_attribution_correct") is not None:
            n_attribution_evaluated += 1
            if m["factor_attribution_correct"] is True:
                n_attribution_correct += 1
        if m.get("frozen_factors_preserved") is not None:
            n_frozen_evaluated += 1
            if m["frozen_factors_preserved"] is True:
                n_frozen_preserved += 1
        # Non-regression: revised exactly the predicted factor.
        wrong = m.get("wrong_factor_predicted")
        changed = tuple(m.get("factors_changed", ()))
        if wrong is not None and changed == (wrong,):
            n_non_regression_clean += 1

    nr_score = _safe_div(n_non_regression_clean, n_with_revision)
    frozen_rate = _safe_div(n_frozen_preserved, n_frozen_evaluated)
    attribution_acc = _safe_div(n_attribution_correct, n_attribution_evaluated)
    unnecessary_rate = (1.0 - frozen_rate) if n_frozen_evaluated > 0 else 0.0

    # num_attempts_to_success — mean over episodes that ultimately succeeded.
    succ_attempts = [
        r.metrics.get("num_attempts_to_success", 0) for r in records_list
        if r.metrics.get("initial_success") is True
        or r.metrics.get("retry_success") is True
    ]
    mean_num_attempts = (
        sum(succ_attempts) / len(succ_attempts) if succ_attempts else 0.0
    )

    passed = round(delta_pp, 10) >= ACCEPTANCE_DELTA_PP_THRESHOLD

    return {
        "n_total": n_total,
        "n_initial_success": n_initial_success,
        "n_with_revision": n_with_revision,
        "n_retry_success": n_retry_success,
        "n_final_success": n_final_success,
        "initial_attempt_success_rate": initial_rate,
        "retry_success_rate": retry_rate,
        "final_success_rate": final_rate,
        "delta_pp": delta_pp,
        "num_attempts_to_success_mean": mean_num_attempts,
        "intent_factor_attribution_accuracy": attribution_acc,
        "non_regression_score": nr_score,
        "frozen_factor_preservation_rate": frozen_rate,
        "unnecessary_factor_change_rate": unnecessary_rate,
        "revision_success_rate": retry_rate,
        "passed_acceptance": bool(passed),
        "acceptance_threshold_pp": ACCEPTANCE_DELTA_PP_THRESHOLD,
    }


def _markdown_report(metrics: dict) -> str:
    pass_str = "PASS" if metrics["passed_acceptance"] else "FAIL"
    rows = [
        ("Total episodes",                       metrics["n_total"]),
        ("Initial success",                      metrics["n_initial_success"]),
        ("With revision",                        metrics["n_with_revision"]),
        ("Retry success (of revised)",           metrics["n_retry_success"]),
        ("Final success",                        metrics["n_final_success"]),
        ("Initial attempt success rate",         f"{metrics['initial_attempt_success_rate']:.2f}"),
        ("Retry success rate (over revisions)",  f"{metrics['retry_success_rate']:.2f}"),
        ("Final success rate",                   f"{metrics['final_success_rate']:.2f}"),
        ("Delta (pp)",                           f"{metrics['delta_pp']:.1f}"),
        ("Num attempts to success (mean)",       f"{metrics['num_attempts_to_success_mean']:.2f}"),
        ("Intent factor attribution accuracy",   f"{metrics['intent_factor_attribution_accuracy']:.2f}"),
        ("Non-regression score",                 f"{metrics['non_regression_score']:.2f}"),
        ("Frozen factor preservation rate",      f"{metrics['frozen_factor_preservation_rate']:.2f}"),
        ("Unnecessary factor change rate",       f"{metrics['unnecessary_factor_change_rate']:.2f}"),
        ("passed_acceptance",                    metrics["passed_acceptance"]),
    ]
    body = "\n".join(f"| {k} | {v} |" for k, v in rows)
    return (
        "# BABYSTEPS Stage 0 — PushCube Blocked-Approach Report\n\n"
        f"Acceptance: **{pass_str}** (delta_pp >= "
        f"{metrics['acceptance_threshold_pp']})\n\n"
        "| Metric | Value |\n"
        "|--------|-------|\n"
        f"{body}\n"
    )


def write_report(metrics: dict, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    (out_dir / "report.md").write_text(_markdown_report(metrics))
