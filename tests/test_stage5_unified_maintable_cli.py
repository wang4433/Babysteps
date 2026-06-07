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


def _synthetic_scorer_pack(path):
    """A tiny untrained SharedScorer saved to `path` (NOT the gitignored
    models/) — exercises the --scorer wiring without a trained checkpoint."""
    from babysteps.stage5.shared_revision_policy import (
        EFAIL_DIM, FACTOR_ORDER, GI_DIM_DEFAULT, SharedScorer,
        build_value_vocab, save_shared_scorer,
    )
    vocab = build_value_vocab()
    scorer = SharedScorer(vocab_size=len(vocab) + 1, n_factors=len(FACTOR_ORDER),
                          d_gi=GI_DIM_DEFAULT, d_efail=EFAIL_DIM, seed=0)
    save_shared_scorer(scorer, vocab, path)
    return path


def test_shared_policy_runs_with_synthetic_scorer(umt, tmp_path):
    """Supplying --scorer ENABLES shared_revision_policy as the 6th condition:
    it runs, is no longer reported deferred, and obeys the single-slot invariant
    (an untrained scorer may abstain or flip one face — never multi-slot)."""
    scorer = _synthetic_scorer_pack(tmp_path / "synthetic.pt")
    out = tmp_path / "poke6"
    rc = umt.main(["--task", "PokeCube-v1", "--fake", "--mock",
                   "--eval-seeds", "0-5", "--scorer", str(scorer),
                   "--conditions",
                   "same_intent_retry,random_factor_local_edit,vlm_free_replan,"
                   "vlm_diagnosis_local_edit,shared_revision_policy,"
                   "oracle_single_slot",
                   "--out-dir", str(out)])
    assert rc == 0
    res = json.loads((out / "results.json").read_text())
    summ = res["summary"]
    assert "shared_revision_policy" in summ
    assert res["deferred_conditions"] == []          # enabled, not deferred
    srp = summ["shared_revision_policy"]
    assert srp["n"] == 6
    assert srp["edit_cardinality_mean"] <= 1.0       # single-slot invariant
    # Shares VLM attribution with vlm_diagnosis_local_edit (cost is charged).
    assert srp["vlm_calls_mean"] >= 1.0


def test_shared_policy_recovers_with_trained_scorer(umt, tmp_path):
    """A scorer trained on the residual->face rule recovers the deterministic
    fake failures via the shared-policy condition (oracle attribution path
    through run_eval, isolating the VALUE policy)."""
    import numpy as np
    import torch
    from babysteps.stage5.shared_revision_policy import (
        EFAIL_DIM, FACTOR_ORDER, GI_DIM_DEFAULT, SharedScorer,
        build_value_vocab, save_shared_scorer, _efail_vec,
    )
    # Train a minimal scorer on synthetic contact_region (residual-sign -> face)
    # tuples so the shared-policy condition can actually correct.
    vocab = build_value_vocab()
    scorer = SharedScorer(vocab_size=len(vocab) + 1, n_factors=len(FACTOR_ORDER),
                          d_gi=GI_DIM_DEFAULT, d_efail=EFAIL_DIM, seed=0)
    faces = {"minus_x_face": (1.0, 0.0), "plus_x_face": (-1.0, 0.0),
             "minus_y_face": (0.0, 1.0), "plus_y_face": (0.0, -1.0)}
    fidx = FACTOR_ORDER.index("contact_region")
    opt = torch.optim.Adam(scorer.parameters(), lr=5e-3)
    rng = np.random.default_rng(0)
    for _ in range(400):
        # sample a goal direction (residual) + a wrong current face; the correct
        # candidate is the face whose push aligns with the residual.
        correct = list(faces)[rng.integers(4)]
        resid = np.array(faces[correct]) * 0.1
        ef = torch.tensor(_efail_vec("direction_error", tuple(resid))).unsqueeze(0)
        cand = list(faces)
        cur = cand[rng.integers(4)]
        gi = torch.zeros(1, GI_DIM_DEFAULT)
        b = len(cand)
        logits = scorer(
            gi.expand(b, -1), ef.expand(b, -1),
            torch.full((b,), fidx), torch.full((b,), vocab[("contact_region", cur)]),
            torch.tensor([vocab[("contact_region", c)] for c in cand]))
        target = torch.tensor(cand.index(correct))
        loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
        opt.zero_grad(); loss.backward(); opt.step()
    save_shared_scorer(scorer, vocab, tmp_path / "trained.pt")

    # Drive run_eval directly with oracle attribution would bypass the VLM; use
    # the mock VLM pinned to contact_region so attribution is correct.
    from babysteps.stage5.vlm_attribute import MockVLMClient
    from babysteps.stage5.shared_revision_policy import SharedScorerPolicy
    adapter, runner = umt._make_runner_adapter("PokeCube-v1", fake=True)
    args = SimpleNamespace(fake=True)
    spec = umt._build_contact_region_spec("PokeCube-v1", args, runner, adapter)
    vlm = MockVLMClient(constrained_response="contact_region")
    policy = SharedScorerPolicy.from_pack(tmp_path / "trained.pt")
    res = umt.run_eval(spec, runner, vlm, seeds=list(range(8)),
                       conditions=["shared_revision_policy", "oracle_single_slot"],
                       shared_policy=policy)
    srp = res["summary"]["shared_revision_policy"]
    orc = res["summary"]["oracle_single_slot"]
    # The trained shared scorer recovers the fake failures up to the oracle.
    assert srp["recovery_on_initial_fail"] == orc["recovery_on_initial_fail"]
    assert srp["recovery_on_initial_fail"] >= 0.9
    assert srp["edit_cardinality_mean"] <= 1.0
