"""Stage-4 per-task per-factor report aggregation + Markdown rendering.

Each (task, factor) cell is classified three ways (spec §4):

  * trivially_constant — one label for the whole task. Excluded from the gate.
  * label_identity     — recoverable without trajectory geometry because the
                         factor is fed in as a feature one-hot (contact_region
                         ← contact_region_label, goal_state ← final_state) or
                         is a deterministic function of one (PushCube
                         approach_direction = face_to_approach(contact_region)).
                         Reported but NOT counted toward the geometric headline.
  * geometric          — recoverable only from trajectory geometry (today:
                         object_motion). GATED: probe_acc_mean ≥ GATE_THRESHOLD
                         AND clears majority + shuffled baselines by GATE_MARGIN.
"""
from __future__ import annotations

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
GATE_MARGIN: float = 0.10

# (task, factor) pairs that are label-identity. "*" matches any task.
_LABEL_IDENTITY: frozenset[tuple[str, str]] = frozenset({
    ("*", "contact_region"),
    ("*", "goal_state"),
    ("PushCube-v1", "approach_direction"),
})


def _is_label_identity(task: str, factor: str) -> bool:
    return (("*", factor) in _LABEL_IDENTITY
            or (task, factor) in _LABEL_IDENTITY)


def _cell_class(task: str, factor: str, cell: dict) -> str:
    if cell["trivially_constant"] or cell["n_unique_labels"] <= 1:
        return "trivially_constant"
    if _is_label_identity(task, factor):
        return "label_identity"
    return "geometric"


def _geometric_pass(cell: dict) -> bool:
    acc = cell["probe_acc_mean"]
    return (
        acc >= GATE_THRESHOLD
        and acc >= cell["majority_class_acc"] + GATE_MARGIN
        and acc >= cell["shuffled_features_acc"] + GATE_MARGIN
    )


def _cell_gate(task: str, factor: str, cell: dict) -> str:
    klass = cell["cell_class"]
    if klass == "trivially_constant":
        return "trivial"
    if klass == "label_identity":
        return "label_identity"
    return "PASS" if _geometric_pass(cell) else "FAIL"


def build_report(rows: list[dict]) -> dict:
    """Aggregate annotated probe outputs into a nested report + gate summary.

    Returns::

        {
          "by_task": {task: {factor: {<cell metrics> + cell_class + gate}}},
          "gate": {threshold, margin, n_total, n_trivial, n_label_identity,
                   n_geometric, n_passing, n_failing, failing_cells},
        }

    n_passing / n_failing count GEOMETRIC cells only — label-identity and
    trivially-constant cells never count as a gate pass.
    """
    by_task: dict[str, dict[str, dict]] = {}
    n_total = n_trivial = n_label_identity = n_geometric = 0
    n_passing = n_failing = 0
    failing_cells: list[tuple[str, str]] = []

    for row in rows:
        task = row["task"]
        factor = row["factor"]
        cell = {k: row[k] for k in _CELL_KEYS}
        cell["cell_class"] = _cell_class(task, factor, cell)
        cell["gate"] = _cell_gate(task, factor, cell)
        by_task.setdefault(task, {})[factor] = cell

        n_total += 1
        if cell["cell_class"] == "trivially_constant":
            n_trivial += 1
        elif cell["cell_class"] == "label_identity":
            n_label_identity += 1
        else:
            n_geometric += 1
            if cell["gate"] == "PASS":
                n_passing += 1
            else:
                n_failing += 1
                failing_cells.append((task, factor))

    gate = {
        "threshold": GATE_THRESHOLD,
        "margin": GATE_MARGIN,
        "n_total": n_total,
        "n_trivial": n_trivial,
        "n_label_identity": n_label_identity,
        "n_geometric": n_geometric,
        "n_passing": n_passing,
        "n_failing": n_failing,
        "failing_cells": failing_cells,
    }
    return {"by_task": by_task, "gate": gate}


def markdown_table(report: dict) -> str:
    """Render `report['by_task']` as one Markdown table per task (humans)."""
    header = (
        "| factor | class | n_unique | n_episodes | majority | shuffled "
        "| probe ± std | gate |"
    )
    rule = "| --- | --- | --- | --- | --- | --- | --- | --- |"

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
                f"| {factor} | {c['cell_class']} | {c['n_unique_labels']} "
                f"| {c['n_episodes']} | {c['majority_class_acc']:.2f} "
                f"| {c['shuffled_features_acc']:.2f} | {probe_pm} | {c['gate']} |"
            )
        lines.append("")
    return "\n".join(lines)
