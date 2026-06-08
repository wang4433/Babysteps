"""Stage-5 unified main-table — per-condition metrics + aggregation (PURE).

One source of truth for the table's columns, shared by PushCube and StackCube so
the two tasks are directly comparable. No env / GPU / VLM imports — operates on
``Intent`` values + scalars produced by the evaluator.

Metric families (``redesign_failure_paradigm.md`` → "Fast and better"):

* recovery        — final success (overall) + success on initial-fail episodes;
* selectivity     — preservation, unnecessary/harmful change rates,
                    ``edit_cardinality`` (reused from
                    :mod:`babysteps.stage5.selectivity`, the single definition);
* decision latency — split into ``diagnosis`` (attribution), ``revision``
                    (value policy), ``joint_reasoning`` (one combined VLM call,
                    free-replan only), and ``total_decision`` = their sum;
* rollout latency  — env time for the retry;
* compute cost     — VLM calls + generated/input tokens.

The latency split exists because an 8B VLM dominates wall-clock: lumping
free-replan's single joint call under "revision" would bias the
diagnosis-vs-revision comparison (corrections #7)."""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

from babysteps.schemas import INTENT_FIELDS, Intent
from babysteps.stage5.selectivity import selectivity_metrics

# Numeric per-episode keys aggregated as means.
_MEAN_KEYS: tuple[str, ...] = (
    "preservation",
    "unnecessary_changes_rate",
    "harmful_changes_rate",
    "edit_cardinality",
    "diagnosis_latency_s",
    "revision_latency_s",
    "joint_reasoning_latency_s",
    "total_decision_latency_s",
    "rollout_latency_s",
    "vlm_calls",
    "vlm_gen_tokens",
    "vlm_input_tokens",
)


def per_condition_metrics(
    *,
    initial: Intent,
    revised: Optional[Intent],
    gt: Intent,
    implicated_factor: str,
    initial_success: bool,
    retry_success: Optional[bool],
    factor_menu: tuple[str, ...] = INTENT_FIELDS,
    diagnosis_latency_s: float = 0.0,
    revision_latency_s: float = 0.0,
    joint_reasoning_latency_s: float = 0.0,
    rollout_latency_s: float = 0.0,
    vlm_cost: Optional[Mapping] = None,
    attribution_correct: Optional[bool] = None,
) -> dict:
    """Metrics for ONE (episode, condition).

    ``retry_success is None`` means no revision was attempted (initial already
    succeeded) → ``final_success`` falls back to ``initial_success``. Selectivity
    is measured against ``implicated_factor`` + ``gt`` so all conditions are
    comparable (revised may be ``initial`` for same-intent, or ``None`` for a
    parse-fail)."""
    sel = selectivity_metrics(
        initial=initial, revised=revised, gt=gt,
        implicated_factor=implicated_factor, factor_menu=factor_menu,
    )
    final_success = (bool(retry_success) if retry_success is not None
                     else bool(initial_success))
    cost = dict(vlm_cost) if vlm_cost else {}
    total_decision = (float(diagnosis_latency_s) + float(revision_latency_s)
                      + float(joint_reasoning_latency_s))
    row = {
        "initial_success": bool(initial_success),
        "retry_success": (None if retry_success is None
                          else bool(retry_success)),
        "final_success": final_success,
        "diagnosis_latency_s": float(diagnosis_latency_s),
        "revision_latency_s": float(revision_latency_s),
        "joint_reasoning_latency_s": float(joint_reasoning_latency_s),
        "total_decision_latency_s": total_decision,
        "rollout_latency_s": float(rollout_latency_s),
        "vlm_calls": int(cost.get("n_calls", 0)),
        "vlm_gen_tokens": int(cost.get("gen_tokens", 0)),
        "vlm_input_tokens": int(cost.get("input_tokens", 0)),
        "attribution_correct": attribution_correct,
        **sel,
    }
    return row


def _mean(vals: Sequence[float]) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def aggregate_condition(rows: list[dict]) -> dict:
    """Aggregate one condition's per-episode rows into the table cell."""
    n = len(rows)
    fails = [r for r in rows if not r["initial_success"]]
    out: dict = {
        "n": n,
        "n_initial_fail": len(fails),
        "recovery_overall": _mean([r["final_success"] for r in rows]),
        "recovery_on_initial_fail": _mean([r["final_success"] for r in fails]),
    }
    for k in _MEAN_KEYS:
        out[k + "_mean"] = _mean([r.get(k) for r in rows])
    attr = [r["attribution_correct"] for r in rows
            if r.get("attribution_correct") is not None]
    out["attribution_accuracy"] = (sum(bool(a) for a in attr) / len(attr)
                                   if attr else None)
    return out


def aggregate(rows_by_condition: Mapping[str, list[dict]]) -> dict:
    """Aggregate every condition's rows. Keys preserve insertion order."""
    return {cond: aggregate_condition(rows)
            for cond, rows in rows_by_condition.items()}
