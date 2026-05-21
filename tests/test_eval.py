"""Tests for babysteps.eval — dataset-level metrics + report writer."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from babysteps.eval import (
    ACCEPTANCE_DELTA_PP_THRESHOLD,
    compute_metrics,
    write_report,
)
from babysteps.schemas import CLAIM_BOUNDARY, EpisodeRecord


def _record(
    *,
    episode_id: str,
    initial_success: bool,
    retry_success: bool | None,
    failure_type: str = "approach_blocked",
    wrong_factor_predicted: str | None = "approach_direction",
    oracle_wrong_factor: str = "approach_direction",
    factors_changed: tuple[str, ...] = ("approach_direction",),
    attribution_correct: bool | None = True,
    frozen_preserved: bool | None = True,
) -> EpisodeRecord:
    num_attempts = 1 if initial_success else 2
    metrics = {
        "initial_success": initial_success,
        "retry_success": retry_success,
        "num_attempts_to_success": num_attempts,
        "failure_type": failure_type if not initial_success else "none",
        "wrong_factor_predicted": wrong_factor_predicted if not initial_success else None,
        "oracle_wrong_factor": oracle_wrong_factor,
        "factor_attribution_correct": attribution_correct if not initial_success else None,
        "factors_changed": list(factors_changed) if not initial_success else [],
        "frozen_factors_preserved": frozen_preserved if not initial_success else None,
    }
    return EpisodeRecord(
        episode_id=episode_id,
        stage="stage_0",
        task="PushCube-v1",
        claim_boundary=CLAIM_BOUNDARY,
        demo={}, execution={"success": initial_success},
        failure_packet={"failure_predicate": "none" if initial_success else failure_type,
                        "wrong_factor": None if initial_success else wrong_factor_predicted,
                        "oracle_wrong_factor": oracle_wrong_factor},
        revision=None if initial_success else {"operator": "approach_substitution",
                                                "factor": "approach_direction",
                                                "old_value": "x", "new_value": "y",
                                                "frozen_factors": []},
        retry=None if initial_success else {"success": retry_success, "num_retries": 1},
        metrics=metrics,
    )


def test_compute_metrics_all_recovered():
    """5 episodes: all blocked then recovered → delta_pp == 100."""
    records = [
        _record(episode_id=f"e_{i}", initial_success=False, retry_success=True)
        for i in range(5)
    ]
    m = compute_metrics(records)
    assert m["n_total"] == 5
    assert m["n_initial_success"] == 0
    assert m["n_with_revision"] == 5
    assert m["n_retry_success"] == 5
    assert m["initial_attempt_success_rate"] == pytest.approx(0.0)
    assert m["retry_success_rate"] == pytest.approx(1.0)
    assert m["delta_pp"] == pytest.approx(100.0)
    assert m["non_regression_score"] == pytest.approx(1.0)
    assert m["frozen_factor_preservation_rate"] == pytest.approx(1.0)
    assert m["intent_factor_attribution_accuracy"] == pytest.approx(1.0)
    assert m["passed_acceptance"] is True


def test_compute_metrics_mixed():
    records = [
        _record(episode_id="ok", initial_success=True, retry_success=None),
        _record(episode_id="recovered", initial_success=False, retry_success=True),
        _record(episode_id="still_failed", initial_success=False, retry_success=False),
    ]
    m = compute_metrics(records)
    assert m["n_total"] == 3
    assert m["n_initial_success"] == 1
    assert m["n_with_revision"] == 2
    assert m["n_retry_success"] == 1
    assert m["final_success_rate"] == pytest.approx(2 / 3)
    assert m["initial_attempt_success_rate"] == pytest.approx(1 / 3)
    assert m["retry_success_rate"] == pytest.approx(0.5)


def test_compute_metrics_non_regression_score_detects_extra_change():
    """If a revision changes two factors when only one was attributed,
    non_regression_score drops."""
    bad = _record(
        episode_id="bad", initial_success=False, retry_success=True,
        factors_changed=("approach_direction", "contact_region"),
        frozen_preserved=False,
    )
    good = _record(
        episode_id="good", initial_success=False, retry_success=True,
        factors_changed=("approach_direction",), frozen_preserved=True,
    )
    m = compute_metrics([bad, good])
    assert m["non_regression_score"] == pytest.approx(0.5)
    assert m["frozen_factor_preservation_rate"] == pytest.approx(0.5)


def test_compute_metrics_attribution_accuracy():
    correct = _record(
        episode_id="correct", initial_success=False, retry_success=True,
        attribution_correct=True,
    )
    wrong = _record(
        episode_id="wrong", initial_success=False, retry_success=False,
        attribution_correct=False,
    )
    m = compute_metrics([correct, wrong])
    assert m["intent_factor_attribution_accuracy"] == pytest.approx(0.5)


def test_compute_metrics_empty_list():
    m = compute_metrics([])
    assert m["n_total"] == 0
    assert m["initial_attempt_success_rate"] == 0.0
    assert m["retry_success_rate"] == 0.0
    assert m["passed_acceptance"] is False


def test_acceptance_threshold_matches_pick4pass():
    assert ACCEPTANCE_DELTA_PP_THRESHOLD == 10.0


def test_write_report_creates_md_and_json(tmp_path: Path):
    records = [_record(episode_id="x", initial_success=False, retry_success=True)]
    metrics = compute_metrics(records)
    write_report(metrics, tmp_path)
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "report.json").exists()
    report_json = json.loads((tmp_path / "report.json").read_text())
    assert report_json["delta_pp"] == pytest.approx(100.0)
    md = (tmp_path / "report.md").read_text()
    assert "BABYSTEPS Stage 0" in md
    assert "PASS" in md or "passed_acceptance" in md


def test_write_report_title_reflects_task_from_records(tmp_path: Path):
    """The report title should say PickCube when records carry task=PickCube-v1."""
    rec = _record(episode_id="x", initial_success=False, retry_success=True)
    # Override the task field — _record() defaults to PushCube-v1.
    from dataclasses import replace
    rec = replace(rec, task="PickCube-v1")
    metrics = compute_metrics([rec])
    write_report(metrics, tmp_path)
    md = (tmp_path / "report.md").read_text()
    title_line = md.splitlines()[0]
    assert title_line == "# BABYSTEPS Stage 0 — PickCube Report", (
        f"unexpected title line: {title_line!r}\nfull head:\n{md[:200]}"
    )


def test_write_report_title_pushcube_default(tmp_path: Path):
    """Backward-compat: PushCube records still yield a PushCube title."""
    rec = _record(episode_id="x", initial_success=False, retry_success=True)
    metrics = compute_metrics(rec for rec in [rec])
    write_report(metrics, tmp_path)
    md = (tmp_path / "report.md").read_text()
    title_line = md.splitlines()[0]
    assert title_line == "# BABYSTEPS Stage 0 — PushCube Report", (
        f"unexpected title line: {title_line!r}"
    )


def _baseline_record(*, correct_fixed, harmful, n_should, n_preserved,
                     initial_success=False, retry_success=True, revised=True):
    from babysteps.schemas import EpisodeRecord
    metrics = {
        "initial_success": initial_success,
        "retry_success": retry_success,
        "num_attempts_to_success": 2 if retry_success else 2,
        "wrong_factor_predicted": "contact_region",
        "oracle_wrong_factor": "contact_region",
        "factor_attribution_correct": True,
        "factors_changed": ["contact_region"],
        "frozen_factors_preserved": True,
        "correct_factor_fixed": correct_fixed,
        "harmful_revision": harmful,
        "n_should_preserve": n_should,
        "n_preserved": n_preserved,
    }
    return EpisodeRecord(
        episode_id="x", stage="stage_0", task="PushCube-v1",
        claim_boundary="third_person_demo_proxy_not_human_demo",
        demo={}, execution={}, failure_packet={},
        revision={} if revised else None,
        retry={"success": retry_success} if revised else None,
        metrics=metrics)


def test_compute_metrics_baseline_columns():
    recs = [
        _baseline_record(correct_fixed=True, harmful=False, n_should=1, n_preserved=1),
        _baseline_record(correct_fixed=True, harmful=True, n_should=2, n_preserved=0),
    ]
    m = compute_metrics(recs)
    assert m["correct_factor_fixed_rate"] == 1.0
    assert m["harmful_revision_rate"] == 0.5
    # preserved/should = (1 + 0) / (1 + 2)
    assert abs(m["frozen_preservation_rate_gt"] - (1 / 3)) < 1e-9


def test_compute_metrics_baseline_columns_absent_when_no_keys():
    # Records without baseline keys → rates default to 0.0, evaluated count 0.
    from babysteps.schemas import EpisodeRecord
    rec = EpisodeRecord(
        episode_id="x", stage="stage_0", task="PushCube-v1",
        claim_boundary="third_person_demo_proxy_not_human_demo",
        demo={}, execution={}, failure_packet={}, revision={}, retry={},
        metrics={"initial_success": False, "retry_success": True,
                 "factors_changed": ["contact_region"]})
    m = compute_metrics([rec])
    assert m["n_baseline_evaluated"] == 0
    assert m["correct_factor_fixed_rate"] == 0.0


def test_compute_comparison_table_shape_and_order(tmp_path):
    from babysteps.eval import compute_comparison_table, write_comparison_table
    # metrics_by_method_task[method][task] = a compute_metrics() dict.
    fake_metrics = {"final_success_rate": 0.9, "retry_success_rate": 0.9,
                    "correct_factor_fixed_rate": 1.0,
                    "frozen_preservation_rate_gt": 1.0,
                    "harmful_revision_rate": 0.0,
                    "num_attempts_to_success_mean": 1.8}
    methods = ["one_shot", "babysteps_selective", "full_replan_analogue"]
    tasks = ["PushCube-v1", "PickCube-v1", "StackCube-v1"]
    by = {mth: {t: dict(fake_metrics) for t in tasks} for mth in methods}
    table = compute_comparison_table(by, methods=methods, tasks=tasks)
    assert [row["method"] for row in table["rows"]] == methods
    # each row has a per-task value + a mean for each column
    assert "mean" in table["rows"][0]["final_success_rate"]
    out = tmp_path / "table.md"
    write_comparison_table(table, out)
    assert out.exists() and "babysteps_selective" in out.read_text()
