"""Stage-4 report aggregation + Markdown rendering."""


def _row(task, factor, acc, n_unique=2, trivial=False):
    return {
        "task": task, "factor": factor,
        "probe_acc_mean": acc, "probe_acc_std": 0.0,
        "majority_class_acc": 0.5, "shuffled_features_acc": 0.5,
        "n_unique_labels": n_unique, "n_episodes": 24,
        "trivially_constant": trivial,
    }


def test_build_report_groups_by_task_then_factor():
    from babysteps.stage4.report import build_report
    rows = [
        _row("PushCube-v1", "approach_direction", 0.95),
        _row("PushCube-v1", "goal_state", 1.0, n_unique=1, trivial=True),
        _row("PickCube-v1", "contact_region", 0.80),
    ]
    rep = build_report(rows)
    assert "PushCube-v1" in rep["by_task"]
    assert "PickCube-v1" in rep["by_task"]
    assert rep["gate"]["n_total"] == 3
    assert rep["gate"]["n_trivial"] == 1
    assert rep["gate"]["n_passing"] == 1   # 0.95 ≥ 0.90
    assert rep["gate"]["n_failing"] == 1   # 0.80 < 0.90
    assert ("PickCube-v1", "contact_region") in rep["gate"]["failing_cells"]


def test_markdown_table_contains_task_headers_and_numbers():
    from babysteps.stage4.report import build_report, markdown_table
    rep = build_report([_row("PushCube-v1", "approach_direction", 0.95)])
    md = markdown_table(rep)
    assert "PushCube-v1" in md
    assert "approach_direction" in md
    assert "0.95" in md
