"""Sim-free tests for scripts/stage5_train_shared_policy.py pure helpers.

Validates the pooled-tuple normalization + that ONE SharedScorer trained on
pooled PushCube + StackCube tuples drives BOTH tasks (multi-task capability) —
the step-2 deliverable. No GPU / no gitignored artifacts; synthetic tuples +
synthetic g_i lookup only.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from babysteps.stage5.revision_policy import (
    FailureEvidence, RevisionRequest, candidates_for,
)
from babysteps.stage5.shared_revision_policy import (
    GI_DIM_DEFAULT, SharedScorerPolicy, build_value_vocab,
)

trainmod = importlib.import_module("stage5_train_shared_policy")

FACES = ("minus_x_face", "plus_x_face", "minus_y_face", "plus_y_face")
_DIRS = {"plus_x_face": (1.0, 0.0), "minus_x_face": (-1.0, 0.0),
         "plus_y_face": (0.0, 1.0), "minus_y_face": (0.0, -1.0)}


def test_normalize_pushcube_and_stackcube_tuples():
    push = trainmod.normalize_tuple({
        "demo_face": "minus_x_face", "correct_face": "plus_x_face",
        "residual_xy": [1.0, 0.0], "failure_predicate": "direction_error"})
    assert push["task"] == "PushCube-v1" and push["factor"] == "contact_region"
    assert push["current"] == "minus_x_face" and push["correct"] == "plus_x_face"
    assert set(push["candidates"]) == set(FACES)

    stack = trainmod.normalize_tuple({
        "task": "StackCube-v1", "factor": "goal_state",
        "current_value": "cube_at_target", "correct_value": "cubeA_on_cubeB",
        "failure_predicate": "goal_not_satisfied", "residual_xy": None,
        "candidates": ["cube_at_target", "cubeA_on_cubeB"]})
    assert stack["factor"] == "goal_state" and stack["residual_xy"] is None
    assert stack["correct"] == "cubeA_on_cubeB"


def test_build_training_rows_attaches_gi_and_drops_bad_targets():
    def gi_lookup(task, factor, current):
        return np.ones(32, dtype=np.float32) if task == "PushCube-v1" else None
    tuples = [
        {"demo_face": "minus_x_face", "correct_face": "plus_x_face",
         "residual_xy": [1.0, 0.0], "failure_predicate": "direction_error"},
        # correct not in candidates → dropped.
        {"task": "StackCube-v1", "factor": "goal_state",
         "current_value": "cube_at_target", "correct_value": "not_a_token",
         "failure_predicate": "goal_not_satisfied", "residual_xy": None,
         "candidates": ["cube_at_target", "cubeA_on_cubeB"]},
    ]
    rows = trainmod.build_training_rows(tuples, gi_lookup)
    assert len(rows) == 1
    assert rows[0]["gi"] is not None and rows[0]["task"] == "PushCube-v1"


def test_pooled_training_drives_both_tasks():
    """One checkpoint, pooled PushCube residual-choice + StackCube coverage."""
    def gi_lookup(task, factor, current):
        # deterministic synthetic per-token g_i for PushCube; None for StackCube.
        if task != "PushCube-v1":
            return None
        v = np.zeros(32, dtype=np.float32)
        v[FACES.index(current)] = 1.0
        return v

    push_tuples = [
        {"demo_face": "minus_x_face", "correct_face": "plus_x_face",
         "residual_xy": [1.0, 0.0], "failure_predicate": "direction_error"},
        {"demo_face": "plus_x_face", "correct_face": "minus_x_face",
         "residual_xy": [-1.0, 0.0], "failure_predicate": "direction_error"},
        {"demo_face": "minus_y_face", "correct_face": "plus_y_face",
         "residual_xy": [0.0, 1.0], "failure_predicate": "direction_error"},
        {"demo_face": "plus_y_face", "correct_face": "minus_y_face",
         "residual_xy": [0.0, -1.0], "failure_predicate": "direction_error"},
    ]
    stack_tuples = [
        {"task": "StackCube-v1", "factor": "goal_state",
         "current_value": "cube_at_target", "correct_value": "cubeA_on_cubeB",
         "failure_predicate": "goal_not_satisfied", "residual_xy": None,
         "candidates": ["cube_at_target", "cubeA_on_cubeB"]},
    ]
    vocab = build_value_vocab()
    rows = trainmod.build_training_rows(push_tuples + stack_tuples, gi_lookup)
    scaler = trainmod._fit_scaler(rows)
    scorer = trainmod.train_shared_scorer(
        rows, vocab, d_gi=GI_DIM_DEFAULT, epochs=500, seed=0, scaler=scaler)
    pol = SharedScorerPolicy(scorer, vocab, scaler=scaler)

    # PushCube: the one checkpoint recovers the correct face.
    for t in push_tuples:
        dec = pol.decide(RevisionRequest(
            factor="contact_region", current_value=t["demo_face"],
            candidates=FACES,
            e_fail=FailureEvidence("direction_error", t["residual_xy"]),
            g_i=gi_lookup("PushCube-v1", "contact_region", t["demo_face"])))
        assert dec.new_value == t["correct_face"]

    # StackCube: the SAME checkpoint flips the under-specified goal.
    dec = pol.decide(RevisionRequest(
        factor="goal_state", current_value="cube_at_target",
        candidates=("cube_at_target", "cubeA_on_cubeB"),
        e_fail=FailureEvidence("goal_not_satisfied", None), g_i=None))
    assert dec.new_value == "cubeA_on_cubeB"


def _fixed_wrong(correct: str) -> str:
    """A wrong current face that is NOT simply OPP[correct] (so the label can't
    be read off `current` alone — the residual must carry the rule)."""
    return "minus_x_face" if correct != "minus_x_face" else "plus_x_face"


def _loto_pokecube_trial(seed: int) -> float:
    """Train ONE shared scorer (via the real trainer) on pooled PushCube
    contact_region (residual->face) + StackCube goal_state coverage, freeze it,
    and evaluate a HELD-OUT PokeCube contact_region family — same residual->face
    rule, different g_i family offset (poke unseen in training). Returns held-out
    accuracy. The rule lives in the residual; g_i is a pure distractor (offset
    0.5 + noise, no scaler), mirroring the proven _loto_trial so transfer is
    robust. This is the sim-free proxy of the real step-5 LOTO cell."""
    rng = np.random.default_rng(seed)
    push_off = rng.normal(0, 0.5, size=32).astype(np.float32)
    poke_off = rng.normal(0, 0.5, size=32).astype(np.float32)

    rows = []  # train_shared_scorer rows: factor/current/correct/candidates/...
    for correct, d in _DIRS.items():
        for _ in range(8):
            res = [d[0] + rng.normal(0, 0.05), d[1] + rng.normal(0, 0.05)]
            gi = push_off + rng.normal(0, 0.3, size=32).astype(np.float32)
            rows.append({"task": "PushCube-v1", "factor": "contact_region",
                         "current": _fixed_wrong(correct), "correct": correct,
                         "candidates": list(FACES), "predicate": "direction_error",
                         "residual_xy": res, "gi": gi})
    rows.append({"task": "StackCube-v1", "factor": "goal_state",
                 "current": "cube_at_target", "correct": "cubeA_on_cubeB",
                 "candidates": ["cube_at_target", "cubeA_on_cubeB"],
                 "predicate": "goal_not_satisfied", "residual_xy": None,
                 "gi": None})

    vocab = build_value_vocab()
    scorer = trainmod.train_shared_scorer(
        rows, vocab, d_gi=GI_DIM_DEFAULT, epochs=300, lr=1e-2, wd=1e-3,
        seed=seed, scaler=None)
    pol = SharedScorerPolicy(scorer, vocab, scaler=None)

    ok = tot = 0
    for correct, d in _DIRS.items():
        for _ in range(8):
            res = [d[0] + rng.normal(0, 0.05), d[1] + rng.normal(0, 0.05)]
            gi = poke_off + rng.normal(0, 0.3, size=32).astype(np.float32)
            dec = pol.decide(RevisionRequest(
                factor="contact_region", current_value=_fixed_wrong(correct),
                candidates=candidates_for("PokeCube-v1", "contact_region"),
                e_fail=FailureEvidence("direction_error", res), g_i=gi))
            ok += (dec.new_value == correct)
            tot += 1
    return ok / tot


def test_loto_pooled_train_holds_out_pokecube_family():
    accs = [_loto_pokecube_trial(s) for s in (0, 1, 2)]
    mean_acc = sum(accs) / len(accs)
    assert mean_acc > 0.55, (
        f"held-out PokeCube mean acc {mean_acc:.2f} (per-seed {accs}) not "
        f"clearly above chance 0.25 — residual->face rule did not transfer")
