"""Sim-free proof for Stage-5 Step B: a RESIDUAL-conditioned ReviseHead learns
the corrective direction, and a non-residual head structurally cannot.

The natural-loop ablation showed the goal-relative residual is the load-bearing
feedback (4-way: reverse/disp-vec 22.5% vs residual 92.5%). Here we prove the
*learned* head can consume it: trained on (wrong slot, factor+predicate one-hot +
residual) -> correct-direction centroid, it decodes the right direction. The
control — the committed fp_dim head with NO residual — sees the SAME
(slot, factor, predicate) mapping to several different correct directions, so it
cannot do better than guessing. CPU/torch only; no sim.
"""
from __future__ import annotations

import numpy as np
import torch

from babysteps.stage4.revise_head import (
    FP_VECTOR_DIM, FP_VECTOR_DIM_RESIDUAL, ReviseHead,
    train_revise_head_l2, vectorize_failure_packet,
    vectorize_failure_packet_residual,
)

_D_SLOT = 8
_DIRS = ["translate_+x", "translate_-x", "translate_+y", "translate_-y"]
_UNIT = {"translate_+x": (1.0, 0.0), "translate_-x": (-1.0, 0.0),
         "translate_+y": (0.0, 1.0), "translate_-y": (0.0, -1.0)}


def _centroids():
    # 4 well-separated target slot vectors (scaled one-hots in the first 4 dims).
    C = np.zeros((4, _D_SLOT), dtype=np.float32)
    for i in range(4):
        C[i, i] = 3.0
    return C


def _record():
    return {"revision": {"factor": "contact_region"},
            "failure_packet": {"failure_predicate": "direction_error"}}


def _nearest(C, vecs):
    # argmin L2 to the 4 centroids for each row of vecs (N, d)
    d = np.linalg.norm(vecs[:, None, :] - C[None, :, :], axis=-1)
    return d.argmin(axis=1)


def _build_dataset(n_per_pair=30, noise=0.25, seed=0):
    rng = np.random.default_rng(seed)
    C = _centroids()
    g_pre, fp_res, fp_base, y = [], [], [], []
    for di, demo in enumerate(_DIRS):
        for ei, exec_dir in enumerate(_DIRS):
            if di == ei:
                continue  # only mismatches need revision
            for _ in range(n_per_pair):
                gp = C[di] + rng.normal(0, noise, _D_SLOT).astype(np.float32)
                g_pre.append(gp)
                fp_res.append(vectorize_failure_packet_residual(_record(), _UNIT[exec_dir]))
                fp_base.append(vectorize_failure_packet(_record()))
                y.append(ei)
    return (np.asarray(g_pre, np.float32), np.asarray(fp_res, np.float32),
            np.asarray(fp_base, np.float32), np.asarray(y), C)


def _accuracy(head, C, g_pre, fp, y):
    head.eval()
    with torch.no_grad():
        out = head(torch.tensor(g_pre), torch.tensor(fp)).numpy()
    return float((_nearest(C, out) == y).mean())


def test_residual_head_learns_direction_nonresidual_cannot():
    g_pre, fp_res, fp_base, y, C = _build_dataset(seed=0)
    n = len(y)
    rng = np.random.default_rng(1)
    idx = rng.permutation(n)
    tr, te = idx[: int(0.8 * n)], idx[int(0.8 * n):]
    Ct = torch.tensor(C)

    # Residual-conditioned head: target = the correct-direction centroid.
    head_res = ReviseHead(d_slot=_D_SLOT, fp_dim=FP_VECTOR_DIM_RESIDUAL, hidden=64, seed=0)
    train_revise_head_l2(head_res, g_pre[tr], fp_res[tr], C[y[tr]], n_epochs=500, lr=1e-2)
    acc_res = _accuracy(head_res, C, g_pre[te], fp_res[te], y[te])

    # Control: committed head with NO residual — same (slot, factor, predicate)
    # maps to 3 different correct directions, so it cannot separate them.
    head_base = ReviseHead(d_slot=_D_SLOT, fp_dim=FP_VECTOR_DIM, hidden=64, seed=0)
    train_revise_head_l2(head_base, g_pre[tr], fp_base[tr], C[y[tr]], n_epochs=500, lr=1e-2)
    acc_base = _accuracy(head_base, C, g_pre[te], fp_base[te], y[te])

    assert acc_res >= 0.9, f"residual head should learn direction, got {acc_res:.2f}"
    assert acc_base <= 0.5, f"non-residual head should be near-blind, got {acc_base:.2f}"
    assert acc_res - acc_base >= 0.4


def test_residual_vector_dims():
    v = vectorize_failure_packet_residual(_record(), (0.3, -0.4))
    assert v.shape == (FP_VECTOR_DIM_RESIDUAL,)
    # last 2 entries are the UNIT residual direction
    tail = v[-2:]
    assert abs(float(np.linalg.norm(tail)) - 1.0) < 1e-5
    # zero residual stays zero (no division blow-up)
    v0 = vectorize_failure_packet_residual(_record(), (0.0, 0.0))
    assert float(np.linalg.norm(v0[-2:])) == 0.0
