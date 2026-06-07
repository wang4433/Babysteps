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

from babysteps.stage5.revision_policy import FailureEvidence, RevisionRequest
from babysteps.stage5.shared_revision_policy import (
    GI_DIM_DEFAULT, SharedScorerPolicy, build_value_vocab,
)

trainmod = importlib.import_module("stage5_train_shared_policy")

FACES = ("minus_x_face", "plus_x_face", "minus_y_face", "plus_y_face")


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
