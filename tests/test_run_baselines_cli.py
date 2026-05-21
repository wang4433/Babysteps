"""Smoke test for scripts/run_baselines.py sweep runner CLI.

Runs 2 seeds × all methods × PushCube-v1 on the fake env and asserts the
comparison table files are produced with the expected method names.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def test_run_baselines_fake_env_smoke(tmp_path):
    # 2 seeds × all methods × PushCube on the fake env → table is produced.
    out = tmp_path / "sweep"
    proc = subprocess.run(
        [sys.executable, "scripts/run_baselines.py",
         "--tasks", "PushCube-v1", "--methods", "all",
         "--n_episodes", "2", "--seed_start", "0",
         "--out_dir", str(out), "--fake-env"],
        capture_output=True, text=True,
        cwd=Path(__file__).resolve().parent.parent,
    )
    assert proc.returncode == 0, proc.stderr
    table_md = out / "comparison_table.md"
    table_json = out / "comparison_table.json"
    assert table_md.exists()
    assert table_json.exists()
    data = json.loads(table_json.read_text())
    methods = [r["method"] for r in data["rows"]]
    assert "babysteps_selective" in methods
    assert "full_replan_analogue" in methods
