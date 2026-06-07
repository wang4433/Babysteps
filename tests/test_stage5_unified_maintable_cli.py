"""Sim-free end-to-end tests for scripts/stage5_unified_maintable_eval.py.

Drives the unified evaluator's --fake --mock path on BOTH tasks (no GPU/Vulkan):
asserts all runnable conditions are produced, the deferred shared policy is
reported, the upper bound recovers, the failure-agnostic retry does not (in the
deterministic fake), and the diagnosis/revision/joint-reasoning latency split is
routed correctly (corrections #6/#7)."""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


@pytest.fixture(scope="module")
def umt():
    return importlib.import_module("stage5_unified_maintable_eval")


@pytest.mark.parametrize("task,seeds", [
    ("StackCube-v1", "200-205"),
    ("PushCube-v1", "100-105"),
    ("PokeCube-v1", "100-105"),  # step 3: second contact_region family
])
def test_cli_fake_mock_runs_both_tasks(umt, tmp_path, task, seeds):
    out = tmp_path / task
    rc = umt.main(["--task", task, "--fake", "--mock",
                   "--eval-seeds", seeds, "--out-dir", str(out)])
    assert rc == 0
    res = json.loads((out / "results.json").read_text())
    summ = res["summary"]
    # All five runnable conditions present; shared policy reported deferred.
    for cond in ("same_intent_retry", "random_factor_local_edit",
                 "vlm_free_replan", "vlm_diagnosis_local_edit",
                 "oracle_single_slot"):
        assert cond in summ
    assert res["deferred_conditions"] == ["shared_revision_policy"]
    # Deterministic fake: every initial attempt fails, the oracle upper bound
    # recovers all, the failure-agnostic retry recovers none (corrections #6 —
    # asserted only in the deterministic fixture).
    assert summ["oracle_single_slot"]["recovery_on_initial_fail"] == 1.0
    assert summ["same_intent_retry"]["recovery_on_initial_fail"] == 0.0
    # Upper bound + same-intent are single-slot/no-op (edit cardinality ≤ 1).
    assert summ["oracle_single_slot"]["edit_cardinality_mean"] <= 1.0
    assert summ["same_intent_retry"]["edit_cardinality_mean"] == 0.0


def test_latency_split_routing(umt):
    """vlm_diagnosis_local_edit charges DIAGNOSIS latency; vlm_free_replan
    charges JOINT_REASONING latency — they must not be conflated."""
    from babysteps.stage5.vlm_attribute import MockVLMClient
    adapter, runner = umt._make_runner_adapter("StackCube-v1", fake=True)
    args = SimpleNamespace(fake=True)
    spec = umt._build_stackcube_spec(args, runner, adapter)
    vlm = MockVLMClient(constrained_response="goal_state",
                        synthetic_latency_s=0.01, synthetic_gen_tokens=5)
    res = umt.run_eval(spec, runner, vlm, seeds=[200, 201, 202],
                       conditions=["vlm_diagnosis_local_edit",
                                   "vlm_free_replan"])
    diag = res["summary"]["vlm_diagnosis_local_edit"]
    repl = res["summary"]["vlm_free_replan"]
    assert diag["diagnosis_latency_s_mean"] > 0.0
    assert diag["joint_reasoning_latency_s_mean"] == 0.0
    assert repl["joint_reasoning_latency_s_mean"] > 0.0
    assert repl["diagnosis_latency_s_mean"] == 0.0
    # vlm_diagnosis with the correct factor edits exactly one slot (goal_state).
    assert diag["edit_cardinality_mean"] == 1.0
    assert diag["attribution_accuracy"] == 1.0


def test_deferred_condition_rejected_by_cli(umt, tmp_path):
    with pytest.raises(SystemExit):
        umt.main(["--task", "StackCube-v1", "--fake", "--mock",
                  "--conditions", "shared_revision_policy",
                  "--out-dir", str(tmp_path / "x")])
