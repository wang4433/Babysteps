"""Stage-4 probe smoke tests on synthetic linearly-separable data."""

import numpy as np


def _linearly_separable(n: int = 60, d: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    w = rng.standard_normal(d)
    y = (X @ w > 0).astype(int)
    return X, y


def test_probe_recovers_linear_relationship():
    from babysteps.stage4.probe import train_probe
    X, y = _linearly_separable()
    out = train_probe(X, y, seed=0)
    assert out["n_unique_labels"] == 2
    assert out["probe_acc_mean"] > 0.85
    assert out["shuffled_features_acc"] < out["probe_acc_mean"]


def test_probe_handles_constant_label():
    from babysteps.stage4.probe import train_probe
    X = np.zeros((10, 4))
    y = np.ones(10, dtype=int)
    out = train_probe(X, y)
    assert out["n_unique_labels"] == 1
    assert out["probe_acc_mean"] == 1.0
    assert out["trivially_constant"] is True


def test_probe_majority_class_baseline_is_correct():
    from babysteps.stage4.probe import train_probe
    X = np.zeros((10, 3))
    y = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
    out = train_probe(X, y)
    assert out["majority_class_acc"] == 0.7
