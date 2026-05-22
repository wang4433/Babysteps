"""Stage-4 report aggregation + Markdown rendering (three-way gate)."""


def _row(task, factor, acc, *, n_unique=2, trivial=False,
         majority=0.5, shuffled=0.5):
    return {
        "task": task, "factor": factor,
        "probe_acc_mean": acc, "probe_acc_std": 0.0,
        "majority_class_acc": majority, "shuffled_features_acc": shuffled,
        "n_unique_labels": n_unique, "n_episodes": 40,
        "trivially_constant": trivial,
    }


def test_cell_class_is_three_way():
    from babysteps.stage4.report import build_report
    rows = [
        _row("StackCube-v1", "object_motion", 0.95),
        _row("PushCube-v1", "object_motion", 0.80),
        _row("PushCube-v1", "contact_region", 1.0),
        _row("PushCube-v1", "approach_direction", 1.0),
        _row("StackCube-v1", "goal_state", 1.0, n_unique=1, trivial=True),
        _row("PushCube-v1", "constraint_region", 1.0, n_unique=1, trivial=True),
    ]
    rep = build_report(rows)
    cls = {(t, f): rep["by_task"][t][f]["cell_class"]
           for t in rep["by_task"] for f in rep["by_task"][t]}
    assert cls[("StackCube-v1", "object_motion")] == "geometric"
    assert cls[("PushCube-v1", "object_motion")] == "geometric"
    assert cls[("PushCube-v1", "contact_region")] == "label_identity"
    assert cls[("PushCube-v1", "approach_direction")] == "label_identity"
    assert cls[("StackCube-v1", "goal_state")] == "trivially_constant"
    assert cls[("PushCube-v1", "constraint_region")] == "trivially_constant"

    g = rep["gate"]
    assert g["n_geometric"] == 2
    assert g["n_passing"] == 1
    assert g["n_failing"] == 1
    assert ("PushCube-v1", "object_motion") in g["failing_cells"]
    assert g["n_label_identity"] == 2
    assert g["n_trivial"] == 2


def test_geometric_pass_requires_margin_over_baselines():
    from babysteps.stage4.report import build_report
    rows = [_row("StackCube-v1", "object_motion", 0.92,
                 majority=0.40, shuffled=0.88)]
    rep = build_report(rows)
    assert rep["by_task"]["StackCube-v1"]["object_motion"]["gate"] == "FAIL"
    assert rep["gate"]["n_passing"] == 0


def test_markdown_table_has_class_column():
    from babysteps.stage4.report import build_report, markdown_table
    rep = build_report([_row("StackCube-v1", "object_motion", 0.95)])
    md = markdown_table(rep)
    assert "StackCube-v1" in md
    assert "object_motion" in md
    assert "geometric" in md
    assert "0.95" in md
