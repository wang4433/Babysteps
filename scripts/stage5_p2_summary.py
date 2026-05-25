"""Stage-5 P2 — cross-task summary of VLM attribution results.

Reads reports/stage5/p2_vlm_attribution/<task>/{c1_results.json,c2_results.json}
and emits reports/stage5/p2_vlm_attribution/summary.{json,md}.

Usage::
    python scripts/stage5_p2_summary.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
REPORT_ROOT = _ROOT / "reports" / "stage5" / "p2_vlm_attribution"
TASKS = ("PushCube-v1", "PickCube-v1", "StackCube-v1")


def main(argv: list[str] | None = None) -> int:
    rows = []
    for task in TASKS:
        td = REPORT_ROOT / task
        if not (td / "c1_results.json").exists():
            print(f"SKIP {task}: no c1_results.json", file=sys.stderr)
            continue
        c1 = json.loads((td / "c1_results.json").read_text())
        c2 = json.loads((td / "c2_results.json").read_text())
        rows.append({"task": task, "c1": c1, "c2": c2})

    summary = {"per_task": []}
    for r in rows:
        c1s, c2s = r["c1"]["summary"], r["c2"]["summary"]
        c1_acc = c1s.get("attribution_correct_rate")
        rule_acc = r["c1"].get("rule_table_accuracy")
        c1_pres = c1s.get("frozen_factor_preserved_rate")
        c2_pres = c2s.get("frozen_factor_preserved_rate")
        c1_succ = c1s.get("final_success_rate")
        c2_succ = c2s.get("final_success_rate")
        summary["per_task"].append({
            "task": r["task"],
            "c1_attribution_acc": c1_acc,
            "rule_table_acc": rule_acc,
            "c1_frozen_preservation": c1_pres,
            "c2_frozen_preservation": c2_pres,
            "c1_final_success": c1_succ,
            "c2_final_success": c2_succ,
            "delta_pres_pp": ((c1_pres - c2_pres) * 100
                              if (c1_pres is not None and c2_pres is not None)
                              else None),
            "delta_success_pp": ((c1_succ - c2_succ) * 100
                                 if (c1_succ is not None and c2_succ is not None)
                                 else None),
        })
    (REPORT_ROOT / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    # Markdown dashboard.
    lines = ["# Stage-5 P2 VLM Attribution — Cross-task Summary", ""]
    lines.append(
        "| task | C1 attr acc | rule-table | C1 pres | C2 pres | "
        "Δpres pp | C1 succ | C2 succ | Δsucc pp |"
    )
    lines.append(
        "|---|---|---|---|---|---|---|---|---|"
    )
    for r in summary["per_task"]:
        def fmt(v):
            return f"{v:.3f}" if isinstance(v, float) else "—"

        def fpp(v):
            return f"{v:+.1f}" if isinstance(v, float) else "—"

        lines.append("| " + " | ".join([
            r["task"], fmt(r["c1_attribution_acc"]),
            fmt(r["rule_table_acc"]),
            fmt(r["c1_frozen_preservation"]),
            fmt(r["c2_frozen_preservation"]),
            fpp(r["delta_pres_pp"]),
            fmt(r["c1_final_success"]),
            fmt(r["c2_final_success"]),
            fpp(r["delta_success_pp"]),
        ]) + " |")
    lines.append("")
    lines.append("## Gates")
    lines.append("")
    for r in summary["per_task"]:
        gate_acc = (r["c1_attribution_acc"] is not None
                    and r["rule_table_acc"] is not None
                    and r["c1_attribution_acc"] >= r["rule_table_acc"])
        gate_pres = (r["delta_pres_pp"] is not None
                     and r["delta_pres_pp"] > 0)
        gate_succ = (r["delta_success_pp"] is not None
                     and r["delta_success_pp"] >= -5)
        lines.append(
            f"- **{r['task']}**: "
            f"attr {'PASS' if gate_acc else 'FAIL'} · "
            f"pres {'PASS' if gate_pres else 'FAIL'} · "
            f"succ {'PASS' if gate_succ else 'FAIL'}"
        )
    (REPORT_ROOT / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {REPORT_ROOT}/summary.{{json,md}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
