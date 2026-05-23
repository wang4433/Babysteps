"""Stage-4 M2a IntentHead module tests.

Sim-free: CPU-only torch. Holds the shape/determinism contract and the
two cert-honesty guards (synthetic perfect signal → probe recovers;
shuffled-labels training → probe collapses to chance).
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_intent_head_shape_is_BFD():
    from babysteps.stage4.intent_head import IntentHead
    head = IntentHead(z_dim=20, n_factors=6, d_slot=16, seed=0)
    z = torch.zeros(7, 20)
    out = head(z)
    assert out.shape == (7, 6, 16)
    assert out.dtype == torch.float32


def test_intent_head_determinism_same_seed_same_output():
    from babysteps.stage4.intent_head import IntentHead
    z = torch.randn(5, 20, generator=torch.Generator().manual_seed(42))
    a = IntentHead(z_dim=20, n_factors=6, d_slot=16, seed=0)(z)
    b = IntentHead(z_dim=20, n_factors=6, d_slot=16, seed=0)(z)
    torch.testing.assert_close(a, b)


def test_intent_head_runs_on_cpu_explicitly():
    """No CUDA dependency: the module must build, forward, and backward on CPU."""
    from babysteps.stage4.intent_head import IntentHead
    head = IntentHead(z_dim=20, n_factors=6, d_slot=16, seed=0)
    for p in head.parameters():
        assert p.device.type == "cpu"
    z = torch.randn(3, 20)
    out = head(z)
    out.sum().backward()  # gradient path works


def _synthetic_dataset(n_per_class: int = 12, n_classes: int = 4, seed: int = 0):
    """A 20-dim Z with a clean class signal in dims [0:2]: each class is a
    distinct cluster on a circle (this is what `[sin, cos]` of angle gives us
    after the 2026-05-23 fix). The other 18 dims are low-magnitude i.i.d.
    noise — small enough not to swamp the signal under a 64-hidden MLP +
    LogisticRegression probe on ~40 training points.
    """
    rng = np.random.default_rng(seed)
    centers = np.stack([
        [np.cos(2 * np.pi * c / n_classes), np.sin(2 * np.pi * c / n_classes)]
        for c in range(n_classes)
    ])
    Z, y = [], []
    for c in range(n_classes):
        Z.append(centers[c] + 0.05 * rng.standard_normal((n_per_class, 2)))
        y.extend([c] * n_per_class)
    Z = np.concatenate(Z, axis=0)
    noise = rng.standard_normal((Z.shape[0], 18)) * 0.10
    Z_full = np.concatenate([Z, noise], axis=1).astype(np.float32)
    return Z_full, np.array(y, dtype=np.int64)


def test_g1_recovers_perfect_synthetic_signal():
    """A synthetic Z with a clean per-class cluster in its first 2 dims should
    train an IntentHead whose latent slot recovers the label at >=0.90 under
    held-out 5-fold linear probing — the canonical 'G1 PASS' shape.
    """
    from babysteps.stage4.intent_head import nested_cv_probe_one_factor
    Z, y = _synthetic_dataset(n_per_class=12, n_classes=4, seed=0)
    out = nested_cv_probe_one_factor(
        Z, y,
        factor_idx=0,
        n_factors=6,
        d_slot=16,
        n_epochs=200,
        lr=1e-2,
        seed=0,
    )
    assert out["n_unique_labels"] == 4
    assert out["probe_acc_mean"] >= 0.90, out


def test_g1_collapses_on_shuffled_labels():
    """Training IntentHead on shuffled labels must not memorize: held-out
    probe on the resulting G must NOT clear chance + a 0.10 margin.

    Cert-honesty guard: if this test fails, the encoder is leaking some
    signal even on random labels (e.g. probe overfits the train fold).
    """
    from babysteps.stage4.intent_head import nested_cv_probe_one_factor
    Z, y = _synthetic_dataset(n_per_class=12, n_classes=4, seed=0)
    rng = np.random.default_rng(0)
    y_shuf = y.copy()
    rng.shuffle(y_shuf)
    out = nested_cv_probe_one_factor(
        Z, y_shuf,
        factor_idx=0,
        n_factors=6,
        d_slot=16,
        n_epochs=200,
        lr=1e-2,
        seed=0,
    )
    chance = 1.0 / 4
    assert out["probe_acc_mean"] < chance + 0.10, out


def test_trivially_constant_factor_short_circuits():
    """Factors with one unique label must short-circuit, matching probe.py's
    behavior (acc=1.0, trivially_constant=True), so the cert table treats
    them identically whether features or G is the input."""
    from babysteps.stage4.intent_head import nested_cv_probe_one_factor
    Z = np.random.RandomState(0).randn(20, 20).astype(np.float32)
    y = np.zeros(20, dtype=np.int64)
    out = nested_cv_probe_one_factor(
        Z, y,
        factor_idx=0,
        n_factors=6,
        d_slot=16,
        n_epochs=10,
        lr=1e-2,
        seed=0,
    )
    assert out["trivially_constant"] is True
    assert out["probe_acc_mean"] == 1.0
