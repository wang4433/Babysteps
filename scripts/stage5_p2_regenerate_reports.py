"""Stage-5 P2 — regenerate per-task report.md from c1_results.json + c2_results.json.

Standalone helper: re-runs the report writer using both JSON files on disk,
without re-invoking the VLM. Used after a single-condition rerun (e.g. when
job 10806755 overwrote report.md with C2-only sections because it ran with
--conditions c2 against the pre-fix eval driver).

Usage::
    python scripts/stage5_p2_regenerate_reports.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.stage5_p2_vlm_eval import _write_report_md  # noqa: E402

REPORT_ROOT = _ROOT / "reports" / "stage5" / "p2_vlm_attribution"
TASKS = ("PushCube-v1", "PickCube-v1", "StackCube-v1")


def main() -> int:
    for task in TASKS:
        td = REPORT_ROOT / task
        c1_path = td / "c1_results.json"
        c2_path = td / "c2_results.json"
        if not c1_path.exists():
            print(f"SKIP {task}: missing c1_results.json", file=sys.stderr)
            continue
        c1 = json.loads(c1_path.read_text())
        c2 = (json.loads(c2_path.read_text())
              if c2_path.exists() else None)
        _write_report_md(
            td / "report.md", task,
            c1.get("rule_table_accuracy"),
            c1["per_episode"],
            c2["per_episode"] if c2 else None,
        )
        print(f"wrote {td / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
