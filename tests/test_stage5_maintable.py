"""Sim-free tests for babysteps.stage5.maintable (pure metrics)."""
from __future__ import annotations

from dataclasses import replace

from babysteps.schemas import INTENT_FIELDS, Intent
from babysteps.stage5.maintable import (
    aggregate, aggregate_condition, per_condition_metrics,
)

GT = Intent(
    goal_state="cube_at_target",
    object_motion="translate_+x",
    contact_region="plus_x_face",
    approach_direction="from_plus_x",
    constraint_region="none",
    embodiment_mapping="proxy_contact_to_franka_push",
)
INITIAL = replace(GT, contact_region="minus_x_face")  # wrong contact face


def test_latency_split_total_is_sum():
    row = per_condition_metrics(
        initial=INITIAL, revised=GT, gt=GT, implicated_factor="contact_region",
        initial_success=False, retry_success=True,
        diagnosis_latency_s=0.1, revision_latency_s=0.2,
        joint_reasoning_latency_s=0.0, rollout_latency_s=0.05)
    assert abs(row["total_decision_latency_s"] - 0.3) < 1e-9
    assert row["rollout_latency_s"] == 0.05
    assert row["final_success"] is True


def test_free_replan_joint_latency_not_in_revision():
    # corrections #7: a joint VLM call is reported under joint_reasoning, and
    # total_decision still sums it — without polluting diagnosis/revision.
    row = per_condition_metrics(
        initial=INITIAL, revised=GT, gt=GT, implicated_factor="contact_region",
        initial_success=False, retry_success=True,
        joint_reasoning_latency_s=0.5)
    assert row["diagnosis_latency_s"] == 0.0
    assert row["revision_latency_s"] == 0.0
    assert row["joint_reasoning_latency_s"] == 0.5
    assert row["total_decision_latency_s"] == 0.5


def test_edit_cardinality_measured_exactly():
    # corrections #5: measure exact cardinality, no ordering assumption.
    one = per_condition_metrics(
        initial=INITIAL, revised=GT, gt=GT, implicated_factor="contact_region",
        initial_success=False, retry_success=True)
    assert one["edit_cardinality"] == 1
    noop = per_condition_metrics(
        initial=INITIAL, revised=INITIAL, gt=GT,
        implicated_factor="contact_region",
        initial_success=False, retry_success=False)
    assert noop["edit_cardinality"] == 0
    parsefail = per_condition_metrics(
        initial=INITIAL, revised=None, gt=GT,
        implicated_factor="contact_region",
        initial_success=False, retry_success=None)
    assert parsefail["edit_cardinality"] == 0
    # A multi-factor rewrite (free-replan style) counts every changed slot.
    multi = replace(GT, contact_region="minus_y_face",
                    approach_direction="from_minus_y")
    m = per_condition_metrics(
        initial=INITIAL, revised=multi, gt=GT,
        implicated_factor="contact_region",
        initial_success=False, retry_success=False)
    assert m["edit_cardinality"] == 2  # contact_region + approach_direction


def test_vlm_cost_passthrough():
    row = per_condition_metrics(
        initial=INITIAL, revised=GT, gt=GT, implicated_factor="contact_region",
        initial_success=False, retry_success=True,
        vlm_cost={"n_calls": 2, "gen_tokens": 80, "input_tokens": 300,
                  "latency_s": 1.0})
    assert row["vlm_calls"] == 2
    assert row["vlm_gen_tokens"] == 80
    assert row["vlm_input_tokens"] == 300


def test_aggregate_recovery_overall_and_on_initial_fail():
    rows = [
        # initial success → final success, no revision.
        per_condition_metrics(initial=GT, revised=GT, gt=GT,
                              implicated_factor="contact_region",
                              initial_success=True, retry_success=None),
        per_condition_metrics(initial=GT, revised=GT, gt=GT,
                              implicated_factor="contact_region",
                              initial_success=True, retry_success=None),
        # initial fail, recovered.
        per_condition_metrics(initial=INITIAL, revised=GT, gt=GT,
                              implicated_factor="contact_region",
                              initial_success=False, retry_success=True),
        # initial fail, not recovered.
        per_condition_metrics(initial=INITIAL, revised=INITIAL, gt=GT,
                              implicated_factor="contact_region",
                              initial_success=False, retry_success=False),
    ]
    agg = aggregate_condition(rows)
    assert agg["n"] == 4 and agg["n_initial_fail"] == 2
    assert abs(agg["recovery_overall"] - 0.75) < 1e-9     # 3/4
    assert abs(agg["recovery_on_initial_fail"] - 0.5) < 1e-9  # 1/2


def test_aggregate_preserves_condition_keys():
    rows = [per_condition_metrics(
        initial=INITIAL, revised=GT, gt=GT, implicated_factor="contact_region",
        initial_success=False, retry_success=True)]
    out = aggregate({"a": rows, "b": rows})
    assert list(out.keys()) == ["a", "b"]
