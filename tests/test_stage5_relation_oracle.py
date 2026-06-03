"""Sim-free tests for the Stage-5 relation oracle-ceiling probe.

Pins the pure logic of ``scripts/stage5_relation_oracle_probe.py`` (feature
ladder, parameter-free rule, direct LR probe, gate) and the importability of
``scripts/stage5_extract_cube_positions.py`` — all on synthetic data, no
dataset files and no GPU/Vulkan.
"""
from __future__ import annotations

import importlib

import numpy as np


def _probe_mod():
    return importlib.import_module("scripts.stage5_relation_oracle_probe")


def test_extractor_imports_sim_free():
    """The GPU extractor must import on the login node (lazy sim import)."""
    import sys

    mod = importlib.import_module("scripts.stage5_extract_cube_positions")
    assert hasattr(mod, "main")
    assert "mani_skill" not in sys.modules, (
        "stage5_extract_cube_positions imported mani_skill at module load; "
        "the sim import must stay lazy (inside _make_env)."
    )


def test_build_features_shapes_and_relation():
    mod = _probe_mod()
    A0 = np.array([[0.0, 0.0], [1.0, 2.0], [-1.0, 0.5]])
    B = np.array([[0.3, 0.0], [1.0, 5.0], [-4.0, 0.5]])
    feats = mod.build_features(A0, B)
    assert set(feats) == {
        "A0 (cubeA start)", "B (cubeB)", "[A0;B] (concat)", "B-A0 (relative)",
    }
    assert feats["A0 (cubeA start)"].shape == (3, 2)
    assert feats["B (cubeB)"].shape == (3, 2)
    assert feats["[A0;B] (concat)"].shape == (3, 4)
    assert feats["B-A0 (relative)"].shape == (3, 2)
    # The relative feature is exactly B - A0.
    np.testing.assert_allclose(feats["B-A0 (relative)"], (B - A0).astype(np.float32))
    # All features float32 (probe input contract).
    assert all(f.dtype == np.float32 for f in feats.values())


def test_rule_accuracy_matches_label_definition():
    """The dominant-axis rule on (B-A0) snaps with the label's own helper.

    When labels ARE the snapped (B-A0) direction, accuracy is 1.0; flipping
    one label drops it by exactly 1/n. This pins the well-posedness framing:
    the StackCube label IS `goal_direction_to_motion(cubeB - cubeA)`, so the
    rule re-derives it by construction (a noiseless, parameter-free function
    of resting positions) — not a non-circular recovery.
    """
    mod = _probe_mod()
    from babysteps.envs.scene import goal_direction_to_motion

    A0 = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    B = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    y = [goal_direction_to_motion(B[i] - A0[i]) for i in range(4)]
    assert mod._rule_accuracy(A0, B, y) == 1.0

    y_bad = list(y)
    y_bad[0] = "translate_-x" if y[0] != "translate_-x" else "translate_+x"
    assert abs(mod._rule_accuracy(A0, B, y_bad) - 0.75) < 1e-9


def test_direct_lr_probe_relation_beats_position_alone():
    """On a synthetic scene where the label = dominant axis of (B-A0), the
    relative feature is recoverable while cubeA-position-alone is ~chance.

    In-miniature version of the ladder's one real lesson: the signal lives in
    the *relation*, not in either object's absolute position (the recovery
    itself is near-tautological — the relation is the label's own input).
    """
    mod = _probe_mod()
    from babysteps.envs.scene import goal_direction_to_motion

    rng = np.random.default_rng(0)
    n = 60
    A0 = rng.normal(0, 0.1, size=(n, 2))
    # Offsets along the four cardinal directions, balanced, so the label is a
    # clean function of (B-A0) but A0 alone is pure noise.
    dirs = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=float)
    offs = dirs[np.arange(n) % 4] * 0.2
    B = A0 + offs
    y_str = [goal_direction_to_motion(B[i] - A0[i]) for i in range(n)]
    classes = sorted(set(y_str))
    y = np.array([classes.index(v) for v in y_str], dtype=np.int64)

    feats = mod.build_features(A0.astype(np.float32), B.astype(np.float32))
    rel = mod._direct_lr_probe(feats["B-A0 (relative)"], y, seed=0)
    a0only = mod._direct_lr_probe(feats["A0 (cubeA start)"], y, seed=0)

    assert rel["probe_acc_mean"] > 0.90          # relation recovers the label
    assert a0only["probe_acc_mean"] < 0.55        # position alone ~chance (4-way)
    assert rel["probe_acc_mean"] > a0only["probe_acc_mean"] + 0.3
    # cert keys present for the report aggregator
    assert set(rel) >= {
        "probe_acc_mean", "probe_acc_std", "majority_class_acc",
        "shuffled_features_acc", "n_episodes", "n_unique_labels",
    }


def test_gate_thresholds():
    mod = _probe_mod()
    assert mod._gate(0.95, 0.34, 0.30) == "PASS"
    assert mod._gate(0.88, 0.34, 0.30) == "FAIL"   # below 0.90
    assert mod._gate(0.95, 0.90, 0.30) == "FAIL"   # margin over majority < 0.10
    assert mod._gate(0.95, 0.34, 0.90) == "FAIL"   # margin over shuffled < 0.10
