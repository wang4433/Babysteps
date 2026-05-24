"""Stage-4 M2a ReviseHead tests.

Holds the slot-local revision interface: single-slot input, single-slot
output, enforced by the forward signature. Plus the G2-mechanical
preservation guard: `apply_revision` only writes the implicated slot.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")


# ---------- vectorize_failure_packet ------------------------------------- #

def test_vectorize_failure_packet_one_hot_shape():
    from babysteps.stage4.revise_head import vectorize_failure_packet
    from babysteps.schemas import INTENT_FIELDS, FAILURE_PREDICATES
    fake = {
        "revision": {"factor": "approach_direction"},
        "failure_packet": {"failure_predicate": "approach_blocked"},
    }
    v = vectorize_failure_packet(fake)
    assert v.shape == (len(INTENT_FIELDS) + len(FAILURE_PREDICATES),)
    assert v.dtype == np.float32
    assert v.sum() == pytest.approx(2.0)  # two one-hots


def test_vectorize_failure_packet_factor_position_matches_INTENT_FIELDS():
    from babysteps.stage4.revise_head import vectorize_failure_packet
    from babysteps.schemas import INTENT_FIELDS
    for expected_idx, factor in enumerate(INTENT_FIELDS):
        v = vectorize_failure_packet({
            "revision": {"factor": factor},
            "failure_packet": {"failure_predicate": "approach_blocked"},
        })
        assert v[expected_idx] == 1.0, (factor, expected_idx)


# ---------- ReviseHead --------------------------------------------------- #

def test_revise_head_shape_single_slot_in_single_slot_out():
    from babysteps.stage4.revise_head import ReviseHead
    head = ReviseHead(d_slot=16, fp_dim=15, seed=0)
    g = torch.zeros(7, 16)
    fp = torch.zeros(7, 15)
    out = head(g, fp)
    assert out.shape == (7, 16)
    assert out.dtype == torch.float32


def test_revise_head_refuses_full_G_tensor():
    """The whole point of ReviseHead is the slot-local interface: passing
    a (B, F, d_slot) tensor must fail (the caller must slice to ONE slot
    first). This is `goal.md` §"Stage 4 / Architecture" invariant 1
    enforced at the type-signature level.
    """
    from babysteps.stage4.revise_head import ReviseHead
    head = ReviseHead(d_slot=16, fp_dim=15, seed=0)
    G_full = torch.zeros(3, 6, 16)
    fp = torch.zeros(3, 15)
    with pytest.raises((ValueError, AssertionError, RuntimeError)):
        head(G_full, fp)


def test_revise_head_determinism():
    from babysteps.stage4.revise_head import ReviseHead
    g = torch.randn(5, 16, generator=torch.Generator().manual_seed(7))
    fp = torch.randn(5, 15, generator=torch.Generator().manual_seed(11))
    a = ReviseHead(d_slot=16, fp_dim=15, seed=0)(g, fp)
    b = ReviseHead(d_slot=16, fp_dim=15, seed=0)(g, fp)
    torch.testing.assert_close(a, b)


def test_revise_head_learns_synthetic_mapping_to_centroid():
    """Synthetic: a 3-class slot space; given a wrong-class g_pre and the
    correct factor one-hot (the 'fp'), ReviseHead should learn to map to
    the right-class centroid.

    This is the cert-honest version of "the revision interface can edit
    in slot space" — the learned mapping is just point-to-centroid;
    L2 error on held-out pairs should be small relative to the
    inter-class distance.
    """
    from babysteps.stage4.revise_head import (
        ReviseHead, train_revise_head_l2,
    )
    rng = np.random.default_rng(0)
    n_classes = 3
    d_slot = 8
    fp_dim = 5  # n_classes target one-hot (slim) + 2 unused
    centroids = np.eye(n_classes, d_slot, dtype=np.float32) * 5.0
    # Pairs: each sample has a random wrong g_pre (jittered other centroid)
    # and an fp one-hot encoding the target class.
    n = 90
    g_pre = np.zeros((n, d_slot), dtype=np.float32)
    fp = np.zeros((n, fp_dim), dtype=np.float32)
    g_tgt = np.zeros((n, d_slot), dtype=np.float32)
    for i in range(n):
        src = i % n_classes
        tgt = (src + 1) % n_classes
        g_pre[i] = centroids[src] + 0.1 * rng.standard_normal(d_slot)
        fp[i, tgt] = 1.0
        g_tgt[i] = centroids[tgt]
    # train / test split
    cut = 60
    head = ReviseHead(d_slot=d_slot, fp_dim=fp_dim, hidden=32, seed=0)
    train_revise_head_l2(
        head,
        g_pre[:cut], fp[:cut], g_tgt[:cut],
        n_epochs=400, lr=1e-2, seed=0,
    )
    head.eval()
    with torch.no_grad():
        g_pred = head(
            torch.from_numpy(g_pre[cut:]),
            torch.from_numpy(fp[cut:]),
        ).numpy()
    # The mean test L2 error should be much smaller than the inter-class
    # distance (which is ~ ||centroid[i] - centroid[j]|| = sqrt(2) * 5 ≈ 7).
    err = np.linalg.norm(g_pred - g_tgt[cut:], axis=1).mean()
    assert err < 1.0, err


# ---------- apply_revision (G2 mechanical) ------------------------------ #

def test_apply_revision_only_changes_implicated_slot():
    """G2 mechanical guarantee: editing slot `i` must NOT mutate slots
    `j ≠ i`. The non-target slots must be bit-identical after the call.
    """
    from babysteps.stage4.revise_head import ReviseHead, apply_revision
    rng = np.random.default_rng(0)
    G = torch.from_numpy(
        rng.standard_normal((4, 6, 16)).astype(np.float32)
    )
    fp = torch.from_numpy(
        rng.standard_normal((4, 15)).astype(np.float32)
    )
    head = ReviseHead(d_slot=16, fp_dim=15, seed=0)

    target_idx = 3
    G_revised = apply_revision(G, target_idx, fp, head)

    assert G_revised.shape == G.shape
    for j in range(6):
        if j == target_idx:
            # The target slot SHOULD differ from input (in general; with
            # a randomly-initialized head it almost surely will).
            continue
        torch.testing.assert_close(G_revised[:, j], G[:, j], atol=0, rtol=0)


def test_apply_revision_does_not_mutate_input_tensor():
    """The input G must be left untouched (no aliasing)."""
    from babysteps.stage4.revise_head import ReviseHead, apply_revision
    G = torch.zeros(2, 6, 16, requires_grad=False)
    G_clone = G.clone()
    fp = torch.zeros(2, 15)
    head = ReviseHead(d_slot=16, fp_dim=15, seed=0)
    _ = apply_revision(G, 0, fp, head)
    torch.testing.assert_close(G, G_clone, atol=0, rtol=0)
