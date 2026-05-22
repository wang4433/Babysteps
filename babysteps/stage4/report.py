"""Stage-4 per-task per-factor report aggregation + Markdown rendering.

A `(task, factor)` cell PASSES the recoverability gate iff it is not
trivially constant and its mean probe accuracy clears GATE_THRESHOLD. The
threshold matches goal.md §"Stage 4: Success Criteria" (probe recoverability
>= 90%). Trivially-constant cells (one label for the whole task) are reported
but excluded from the gate.
"""
from __future__ import annotations

# Per-cell keys carried from a probe output into the machine-readable report.
_CELL_KEYS: tuple[str, ...] = (
    "n_episodes",
    "n_unique_labels",
    "majority_class_acc",
    "shuffled_features_acc",
    "probe_acc_mean",
    "probe_acc_std",
    "trivially_constant",
)

GATE_THRESHOLD: float = 0.90


def _cell_gate(cell: dict) -> str:
    """One of 'trivial' / 'PASS' / 'FAIL' for a single cell."""
    if cell["trivially_constant"]:
        return "trivial"
    return "PASS" if cell["probe_acc_mean"] >= GATE_THRESHOLD else "FAIL"


def build_report(rows: list[dict]) -> dict:
    """Aggregate annotated probe outputs into a nested report + gate summary.

    `rows` is a list of `train_probe(...)` outputs, each annotated with a
    `task` and a `factor` key. Returns::

        {
          "by_task": {task: {factor: {<cell metrics> + "gate"}}},
          "gate": {threshold, n_total, n_trivial, n_passing, n_failing,
                   failing_cells: [(task, factor), ...]},
        }
    """
    by_task: dict[str, dict[str, dict]] = {}
    n_total = n_trivial = n_passing = n_failing = 0
    failing_cells: list[tuple[str, str]] = []

    for row in rows:
        task = row["task"]
        factor = row["factor"]
        cell = {k: row[k] for k in _CELL_KEYS}
        cell["gate"] = _cell_gate(cell)
        by_task.setdefault(task, {})[factor] = cell

        n_total += 1
        if cell["trivially_constant"]:
            n_trivial += 1
        elif cell["probe_acc_mean"] >= GATE_THRESHOLD:
            n_passing += 1
        else:
            n_failing += 1
            failing_cells.append((task, factor))

    gate = {
        "threshold": GATE_THRESHOLD,
        "n_total": n_total,
        "n_trivial": n_trivial,
        "n_passing": n_passing,
        "n_failing": n_failing,
        "failing_cells": failing_cells,
    }
    return {"by_task": by_task, "gate": gate}


def markdown_table(report: dict) -> str:
    """Render `report['by_task']` as one Markdown table per task (humans)."""
    header = (
        "| factor | n_unique | n_episodes | majority | shuffled "
        "| probe ± std | gate |"
    )
    rule = "| --- | --- | --- | --- | --- | --- | --- |"

    lines: list[str] = []
    by_task = report["by_task"]
    for task in sorted(by_task):
        lines.append(f"### {task}")
        lines.append("")
        lines.append(header)
        lines.append(rule)
        factors = by_task[task]
        for factor in sorted(factors):
            c = factors[factor]
            probe_pm = f"{c['probe_acc_mean']:.2f} ± {c['probe_acc_std']:.2f}"
            lines.append(
                f"| {factor} | {c['n_unique_labels']} | {c['n_episodes']} "
                f"| {c['majority_class_acc']:.2f} | {c['shuffled_features_acc']:.2f} "
                f"| {probe_pm} | {c['gate']} |"
            )
        lines.append("")
    return "\n".join(lines)
