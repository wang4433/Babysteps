"""Acceptance-gate test for Sub-project B (PickCube).

Drives `stage0_collect.main` end-to-end with --fake-env --task PickCube-v1
and asserts the produced report.json shows delta_pp >= 10 (the
BABYSTEPS Stage-0 acceptance bar from Pick4Pass M-BABY-1).

This is the fake-env analogue of the spec's Section 3 acceptance item
5 ('the report.md summarizer reports delta_pp >= 10 between revised-
retry success rate and initial-attempt success rate on PickCube').
The fake runner is deterministic and the FAILURE_TO_FACTOR / revision
pipeline is the same code-path the real runner uses, so this test
proves the orchestration meets the bar; the real-sim version is the
GPU spot-check (manual, in CLAUDE.md)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_pickcube_fake_env_meets_delta_pp_gate(tmp_path: Path, collect_main):
    out_dir = tmp_path / "out"
    rc = collect_main([
        "--task", "PickCube-v1",
        "--fake-env",
        "--out_dir", str(out_dir),
        "--n_episodes", "5",
        "--seed_start", "0",
    ])
    # rc==0 means the script's own acceptance check passed; we re-assert
    # independently below to surface the actual numbers in pytest output.
    report = json.loads((out_dir / "report.json").read_text())
    assert report["delta_pp"] >= 10.0, (
        f"PickCube fake-env delta_pp = {report['delta_pp']:.1f} "
        f"(threshold 10.0). Initial rate {report['initial_attempt_success_rate']:.2f}, "
        f"retry rate {report['retry_success_rate']:.2f}, n_with_revision="
        f"{report['n_with_revision']}, n_retry_success={report['n_retry_success']}."
    )
    assert report["passed_acceptance"] is True
    assert rc == 0


def test_pushcube_fake_env_meets_delta_pp_gate(tmp_path: Path, collect_main):
    """Regression: PushCube must also pass the gate via the same CLI path."""
    out_dir = tmp_path / "out"
    rc = collect_main([
        "--task", "PushCube-v1",
        "--fake-env",
        "--out_dir", str(out_dir),
        "--n_episodes", "5",
        "--seed_start", "0",
    ])
    report = json.loads((out_dir / "report.json").read_text())
    assert report["delta_pp"] >= 10.0, (
        f"PushCube fake-env delta_pp = {report['delta_pp']:.1f} "
        f"(threshold 10.0). Initial rate {report['initial_attempt_success_rate']:.2f}, "
        f"retry rate {report['retry_success_rate']:.2f}, n_with_revision="
        f"{report['n_with_revision']}, n_retry_success={report['n_retry_success']}."
    )
    assert report["passed_acceptance"] is True
    assert rc == 0


def test_stackcube_fake_env_meets_delta_pp_gate(tmp_path: Path, collect_main):
    """Sub-project C acceptance: StackCube fake-env should achieve
    delta_pp >= 10 (Pick4Pass M-BABY-1 bar). With the deterministic
    FakeStackCubeEnvRunner, all 5 seeds follow under-specified-goal →
    goal_refinement → cubeA_on_cubeB → success. Expected: 100.0."""
    out_dir = tmp_path / "out"
    rc = collect_main([
        "--task", "StackCube-v1",
        "--fake-env",
        "--out_dir", str(out_dir),
        "--n_episodes", "5",
        "--seed_start", "0",
    ])
    report = json.loads((out_dir / "report.json").read_text())
    assert report["delta_pp"] >= 10.0, (
        f"StackCube fake-env delta_pp = {report['delta_pp']:.1f} "
        f"(threshold 10.0). Initial rate {report['initial_attempt_success_rate']:.2f}, "
        f"retry rate {report['retry_success_rate']:.2f}, n_with_revision="
        f"{report['n_with_revision']}, n_retry_success={report['n_retry_success']}."
    )
    assert report["passed_acceptance"] is True
    assert rc == 0


def test_turnfaucet_fake_env_meets_delta_pp_gate(tmp_path: Path, collect_main):
    """Sub-project D acceptance: TurnFaucet fake-env should achieve
    delta_pp >= 10. With FakeTurnFaucetEnvRunner's deterministic
    outcome (success iff handle_grip + faucet_base_static), all 5
    seeds follow under-specified → constraint_introduction → success
    arc, yielding delta_pp = 100.0."""
    out_dir = tmp_path / "out"
    rc = collect_main([
        "--task", "TurnFaucet-v1",
        "--fake-env",
        "--out_dir", str(out_dir),
        "--n_episodes", "5",
        "--seed_start", "0",
    ])
    report = json.loads((out_dir / "report.json").read_text())
    assert report["delta_pp"] >= 10.0, (
        f"TurnFaucet fake-env delta_pp = {report['delta_pp']:.1f}. "
        f"Initial {report['initial_attempt_success_rate']:.2f}, "
        f"retry {report['retry_success_rate']:.2f}, n_with_revision="
        f"{report['n_with_revision']}, n_retry_success={report['n_retry_success']}."
    )
    assert report["passed_acceptance"] is True
    assert rc == 0
