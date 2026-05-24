"""Sim-free tests for the Stage-4 M2.5 AttributionHead.

Five categories per the design spec:
1. shape  2. determinism  3. overfit-tiny separability  4. shuffled-label
collapse  5. predict-roundtrip / save-load.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from babysteps.schemas import INTENT_FIELDS
from babysteps.stage4.attribution_features import (
    FEATURE_DIM,
    vectorize_attribution_input,
)
from babysteps.stage4.attribution_head import (
    AttributionHead,
    N_CLASSES,
    build_training_pairs,
    class_weights_inverse,
    load_attribution_head,
    save_attribution_head,
    train_attribution_head,
)


# --------------------------- 1. shape ---------------------------------- #


def test_forward_shape_is_n_by_n_classes():
    head = AttributionHead(seed=0)
    out = head(torch.zeros(7, FEATURE_DIM))
    assert out.shape == (7, N_CLASSES)


def test_n_classes_matches_intent_fields():
    assert N_CLASSES == len(INTENT_FIELDS)


# --------------------------- 2. determinism ---------------------------- #


def test_same_seed_produces_bit_identical_weights():
    h1 = AttributionHead(seed=42)
    h2 = AttributionHead(seed=42)
    for p1, p2 in zip(h1.parameters(), h2.parameters()):
        assert torch.equal(p1, p2)


def test_different_seed_produces_different_weights():
    h1 = AttributionHead(seed=1)
    h2 = AttributionHead(seed=2)
    diffs = [not torch.equal(p1, p2) for p1, p2 in zip(
        h1.parameters(), h2.parameters())]
    assert any(diffs)


def test_forward_is_deterministic_in_eval_mode():
    head = AttributionHead(seed=0)
    head.eval()
    x = torch.randn(5, FEATURE_DIM)
    out1 = head(x)
    out2 = head(x)
    assert torch.equal(out1, out2)


# --------------------------- 3. overfits-tiny -------------------------- #


def test_overfits_tiny_separable_synthetic():
    """6 classes × 4 samples each, linearly separable by a one-hot feature."""
    rng = np.random.default_rng(0)
    n_per = 4
    X = np.zeros((6 * n_per, FEATURE_DIM), dtype=np.float64)
    y = np.zeros(6 * n_per, dtype=np.int64)
    # Class c gets feature[c] = 1 ± small noise.
    for c in range(6):
        for k in range(n_per):
            i = c * n_per + k
            X[i, c] = 1.0
            X[i] += rng.normal(0, 0.05, FEATURE_DIM)
            y[i] = c
    head = AttributionHead(seed=0)
    metrics = train_attribution_head(head, X, y, n_epochs=300, lr=1e-2, seed=0)
    assert metrics["acc"] >= 0.95, metrics


# --------------------------- 4. shuffled-label collapse ---------------- #


def test_shuffled_labels_collapse_to_near_chance_on_holdout():
    """If labels are random, the head cannot do better than chance on held out."""
    rng = np.random.default_rng(0)
    n = 60
    X = rng.normal(0, 1, (n, FEATURE_DIM))
    y_true = rng.integers(0, N_CLASSES, n)
    # Train on first 40 with SHUFFLED labels, test on remaining 20.
    perm = rng.permutation(40)
    y_train = y_true[:40][perm]  # shuffled labels (loses structure)
    head = AttributionHead(seed=0)
    train_attribution_head(head, X[:40], y_train, n_epochs=300, lr=1e-2, seed=0)
    with torch.no_grad():
        logits = head(torch.from_numpy(X[40:].astype(np.float32)))
        preds = logits.argmax(dim=-1).numpy()
    acc = float((preds == y_true[40:]).mean())
    # Chance is 1/6 ≈ 0.167. Allow a slack of +0.20 = 0.367 upper bound.
    assert acc <= 0.40, f"shuffled-label control failed: held-out acc={acc:.3f}"


# --------------------------- 5. predict / save-load -------------------- #


def test_predict_factor_returns_intent_field_string():
    """predict_factor returns a string from INTENT_FIELDS."""
    head = AttributionHead(seed=0)
    fp = {
        "failure_predicate": "approach_blocked",
        "execution_trace": {"reached_contact": False, "object_moved": False,
                             "collision": False, "planner_failed": True,
                             "grasp_slip": False},
        "object_displacement": 0.0,
        "direction_alignment": None,
    }
    intent = {
        "goal_state": "cube_at_target", "object_motion": "translate_+x",
        "contact_region": "plus_x_face", "approach_direction": "from_plus_x",
        "constraint_region": "none",
        "embodiment_mapping": "proxy_contact_to_franka_push",
    }
    out = head.predict_factor(fp, intent)
    assert out in INTENT_FIELDS


def test_save_load_roundtrip_preserves_weights(tmp_path: Path):
    h1 = AttributionHead(seed=42)
    path = tmp_path / "head.pt"
    save_attribution_head(h1, path)
    h2 = load_attribution_head(path)
    for p1, p2 in zip(h1.parameters(), h2.parameters()):
        assert torch.equal(p1, p2)
    # Forward outputs are bit-identical too.
    x = torch.randn(3, FEATURE_DIM)
    h1.eval(); h2.eval()
    assert torch.equal(h1(x), h2(x))


# --------------------------- training data plumbing -------------------- #


def test_build_training_pairs_skips_success_records():
    records = [
        {  # success record — should be skipped
            "failure_packet": {"failure_predicate": "none",
                                "execution_trace": {},
                                "object_displacement": 0.0,
                                "direction_alignment": None,
                                "oracle_wrong_factor": None},
            "execution": {"initial_intent": {
                "goal_state": "cube_at_target", "object_motion": "translate_+x",
                "contact_region": "plus_x_face",
                "approach_direction": "from_plus_x",
                "constraint_region": "none",
                "embodiment_mapping": "proxy_contact_to_franka_push"}},
        },
        {  # real failure
            "failure_packet": {"failure_predicate": "approach_blocked",
                                "execution_trace": {"reached_contact": False,
                                                     "object_moved": False,
                                                     "collision": False,
                                                     "planner_failed": True,
                                                     "grasp_slip": False},
                                "object_displacement": 0.0,
                                "direction_alignment": None,
                                "oracle_wrong_factor": "approach_direction"},
            "execution": {"initial_intent": {
                "goal_state": "cube_at_target", "object_motion": "translate_+x",
                "contact_region": "plus_x_face",
                "approach_direction": "from_plus_x",
                "constraint_region": "none",
                "embodiment_mapping": "proxy_contact_to_franka_push"}},
        },
    ]
    X, y, _dropped = build_training_pairs(records)
    assert X.shape == (1, FEATURE_DIM)
    assert y.tolist() == [INTENT_FIELDS.index("approach_direction")]


def test_class_weights_inverse_basic_property():
    y = np.array([0, 0, 0, 1, 2])
    w = class_weights_inverse(y, n_classes=6)
    # Heavier weight on the rarer classes.
    assert w[1] > w[0]
    assert w[2] > w[0]
    # All weights positive.
    assert (w > 0).all()
