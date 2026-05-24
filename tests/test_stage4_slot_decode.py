"""Stage-4 M2a slot_decode tests.

Sim-free, numpy-only. The centroid lookup is the canonical M2a discrete
decoder: a learned latent slot vector is mapped back to a discrete
Stage-0 factor value via nearest-centroid in slot space.
"""
from __future__ import annotations

import numpy as np


def test_build_centroids_shape_and_keys():
    from babysteps.stage4.slot_decode import build_factor_centroids
    rng = np.random.default_rng(0)
    G = rng.standard_normal((30, 6, 16)).astype(np.float32)
    labels = {0: np.array([0, 1, 2] * 10, dtype=np.int64),
              3: np.array([0, 1] * 15, dtype=np.int64)}
    centroids = build_factor_centroids(G, labels)
    assert set(centroids.keys()) == {0, 3}
    assert set(centroids[0].keys()) == {0, 1, 2}
    assert set(centroids[3].keys()) == {0, 1}
    for c_dict in centroids.values():
        for vec in c_dict.values():
            assert vec.shape == (16,)
            assert vec.dtype == np.float32


def test_decode_slot_returns_nearest_centroid():
    from babysteps.stage4.slot_decode import decode_slot
    centroids = {
        0: np.array([1.0, 0.0], dtype=np.float32),
        1: np.array([0.0, 1.0], dtype=np.float32),
        2: np.array([-1.0, 0.0], dtype=np.float32),
    }
    assert decode_slot(np.array([0.9, 0.1], dtype=np.float32), centroids) == 0
    assert decode_slot(np.array([0.1, 0.9], dtype=np.float32), centroids) == 1
    assert decode_slot(np.array([-0.7, 0.0], dtype=np.float32), centroids) == 2


def test_decode_slot_handles_single_class():
    """Trivially-constant factor: one centroid → always returns it."""
    from babysteps.stage4.slot_decode import decode_slot
    centroids = {7: np.array([1.0, 2.0, 3.0], dtype=np.float32)}
    assert decode_slot(np.array([0.0, 0.0, 0.0], dtype=np.float32),
                       centroids) == 7
    assert decode_slot(np.array([100.0, -50.0, 0.5], dtype=np.float32),
                       centroids) == 7


def test_centroid_round_trip_on_training_data():
    """For each training episode, decoding G[i, fi] against the
    factor-fi centroids must return the training label more often than not.
    With clear class clusters, train accuracy should be 1.00 by
    construction (a sample is always nearer its own class centroid than
    another class's centroid when classes are linearly separable in
    slot space).
    """
    from babysteps.stage4.slot_decode import (
        build_factor_centroids,
        decode_G,
    )
    rng = np.random.default_rng(0)
    # 3 well-separated clusters in slot[0]; trivial slot[1].
    n = 30
    G = np.zeros((n, 2, 4), dtype=np.float32)
    y0 = np.array([0] * 10 + [1] * 10 + [2] * 10, dtype=np.int64)
    centers = np.array([[+5, 0, 0, 0], [0, +5, 0, 0], [0, 0, +5, 0]],
                       dtype=np.float32)
    for i in range(n):
        G[i, 0] = centers[y0[i]] + 0.1 * rng.standard_normal(4).astype(np.float32)
    labels = {0: y0}
    centroids = build_factor_centroids(G, labels)
    decoded = decode_G(G, centroids)
    assert decoded[0].shape == (n,)
    train_acc = float(np.mean(decoded[0] == y0))
    assert train_acc == 1.0, train_acc


def test_decode_G_skips_unsupervised_factors():
    """decode_G must only return labels for the factors present in
    `centroids` (the trivially-constant ones are not supervised)."""
    from babysteps.stage4.slot_decode import build_factor_centroids, decode_G
    rng = np.random.default_rng(0)
    G = rng.standard_normal((10, 6, 16)).astype(np.float32)
    labels = {2: np.zeros(10, dtype=np.int64) + np.arange(10) % 2}
    centroids = build_factor_centroids(G, labels)
    decoded = decode_G(G, centroids)
    assert set(decoded.keys()) == {2}
