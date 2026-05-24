"""Stage-4 M2a slot_decode — per-factor centroid lookup.

Maps a learned latent slot vector `g_i ∈ R^{d_slot}` back to a discrete
Stage-0 factor value via nearest-centroid in slot space. The centroids
are the mean of training slot vectors per class:

    centroid[factor=i][class=c] = mean_{n : y_n^i == c}( G[n, i] )

No learned decoder weights — keeps the "learned latent / rule-based
schema" boundary clean and makes the discrete intent always recoverable
without retraining when the schema gains tokens (additive-schema rule).

Used by `scripts/stage4_m2a_a2_eval.py` (this milestone) and will be
re-used inside the latent revision loop in A3 (decode revised `g̃_i`
back to a Stage-0 factor value).
"""
from __future__ import annotations

import numpy as np


def build_factor_centroids(
    G: np.ndarray,
    labels_per_factor: dict[int, np.ndarray],
) -> dict[int, dict[int, np.ndarray]]:
    """Per-factor mean slot vector per class.

    `G`: (B, F, d_slot) trained slot tensor (numpy, float32).
    `labels_per_factor`: {factor_idx: y_train (B,)} for each supervised
    factor. Trivially-constant factors should be supplied OR omitted —
    if omitted, that factor has no centroids and `decode_G` will skip it.

    Returns: nested dict {factor_idx: {class_int: centroid (d_slot,)}}.
    """
    if G.ndim != 3:
        raise ValueError(f"G must be (B, F, d_slot); got {G.shape}")
    centroids: dict[int, dict[int, np.ndarray]] = {}
    for fi, y in labels_per_factor.items():
        slot = G[:, fi]  # (B, d_slot)
        per_class: dict[int, np.ndarray] = {}
        for c in np.unique(y):
            mask = (y == c)
            per_class[int(c)] = slot[mask].mean(axis=0).astype(np.float32)
        centroids[int(fi)] = per_class
    return centroids


def decode_slot(
    g: np.ndarray,
    factor_centroids: dict[int, np.ndarray],
) -> int:
    """Nearest-centroid lookup for ONE slot.

    `g`: (d_slot,) slot vector.
    `factor_centroids`: {class_int: centroid (d_slot,)} for this factor.

    Returns the class int. Ties are broken by the lowest class index
    (deterministic). With one centroid this trivially returns that class.
    """
    classes = sorted(factor_centroids.keys())
    if len(classes) == 1:
        return classes[0]
    dists = np.array(
        [np.linalg.norm(g - factor_centroids[c]) for c in classes],
        dtype=np.float32,
    )
    return classes[int(np.argmin(dists))]


def decode_G(
    G: np.ndarray,
    centroids: dict[int, dict[int, np.ndarray]],
) -> dict[int, np.ndarray]:
    """Decode all supervised slots of a batch of G.

    `G`: (B, F, d_slot). `centroids`: output of `build_factor_centroids`.
    Returns {factor_idx: decoded_labels (B,)} — only for factors with
    centroids (others are silently skipped, matching the "trivially-
    constant factors have no recoverable signal" convention).
    """
    if G.ndim != 3:
        raise ValueError(f"G must be (B, F, d_slot); got {G.shape}")
    out: dict[int, np.ndarray] = {}
    for fi, per_class in centroids.items():
        decoded = np.array(
            [decode_slot(G[b, fi], per_class) for b in range(G.shape[0])],
            dtype=np.int64,
        )
        out[fi] = decoded
    return out
