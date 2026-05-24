"""Stage-4 M2a ReviseHead — slot-local intent revision.

`ReviseHead(g_i, fp) → g̃_i` consumes EXACTLY ONE slot intent and the
vectorized failure packet, and emits EXACTLY ONE revised slot intent.
The type signature is the single-factor-revision invariant
(`goal.md` §"Stage 4 / Architecture" invariant 1): the network has no
way to read the full G or write to multiple slots — that's enforced by
the forward(...) shapes, not by hope.

`apply_revision(G, factor_idx, fp, head)` is the M2a wrapper that
writes only the implicated slot's column of G. The G2-mechanical
preservation guarantee (other slots unchanged after a single-slot edit)
is asserted in `tests/test_stage4_revise_head.py`.

Training (M2a A2): targets are the per-class centroids built by
`babysteps.stage4.slot_decode.build_factor_centroids` — given a wrong
`g_pre` and the failure-packet vector indicating the implicated factor +
predicate, ReviseHead learns to map onto the post-revision class's
centroid. Loss is L2.

Sim-free, CPU-only torch.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from babysteps.schemas import FAILURE_PREDICATES, INTENT_FIELDS

_FACTOR_INDEX: dict[str, int] = {f: i for i, f in enumerate(INTENT_FIELDS)}
_PREDICATE_ORDER: tuple[str, ...] = tuple(sorted(FAILURE_PREDICATES))
_PREDICATE_INDEX: dict[str, int] = {p: i for i, p in enumerate(_PREDICATE_ORDER)}

# Total failure-packet vector dim: F (=6) factor one-hot + |FP| (=9) predicate one-hot
FP_VECTOR_DIM: int = len(INTENT_FIELDS) + len(FAILURE_PREDICATES)


def vectorize_failure_packet(record: dict) -> np.ndarray:
    """Build the (F + |FAILURE_PREDICATES|) one-hot vector for ReviseHead.

    Reads:
      - `record["revision"]["factor"]`   → which Stage-0 factor was wrong.
      - `record["failure_packet"]["failure_predicate"]` → which predicate fired.

    Both are categorical with fixed whitelists (see babysteps.schemas).
    Returns a float32 numpy vector of length `FP_VECTOR_DIM`, with
    exactly two 1.0 entries.
    """
    v = np.zeros(FP_VECTOR_DIM, dtype=np.float32)
    factor = record["revision"]["factor"]
    predicate = record["failure_packet"]["failure_predicate"]
    v[_FACTOR_INDEX[factor]] = 1.0
    v[len(INTENT_FIELDS) + _PREDICATE_INDEX[predicate]] = 1.0
    return v


class ReviseHead(nn.Module):
    """Map (g_slot, fp) -> g_slot_revised.

    Forward enforces a single-slot interface:
      - `g_slot` must be (B, d_slot); a (B, F, d_slot) tensor is REFUSED.
      - `fp`     must be (B, fp_dim).
    Output: (B, d_slot). There is no path through this module to write
    to any slot other than the one whose vector was passed in.
    """

    def __init__(
        self,
        *,
        d_slot: int = 16,
        fp_dim: int = FP_VECTOR_DIM,
        hidden: int = 64,
        seed: int = 0,
    ):
        super().__init__()
        torch.manual_seed(seed)
        self.d_slot = d_slot
        self.fp_dim = fp_dim
        self.net = nn.Sequential(
            nn.Linear(d_slot + fp_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_slot),
        )

    def forward(self, g_slot: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
        if g_slot.ndim != 2 or g_slot.shape[-1] != self.d_slot:
            raise ValueError(
                f"ReviseHead requires g_slot of shape (B, d_slot={self.d_slot}); "
                f"got {tuple(g_slot.shape)}. Pass ONE slot, not the full G."
            )
        if fp.ndim != 2 or fp.shape[-1] != self.fp_dim:
            raise ValueError(
                f"ReviseHead requires fp of shape (B, fp_dim={self.fp_dim}); "
                f"got {tuple(fp.shape)}"
            )
        if g_slot.shape[0] != fp.shape[0]:
            raise ValueError(
                f"batch mismatch: g_slot {tuple(g_slot.shape)} vs fp {tuple(fp.shape)}"
            )
        return self.net(torch.cat([g_slot, fp], dim=-1))


def train_revise_head_l2(
    head: ReviseHead,
    g_pre: np.ndarray,
    fp: np.ndarray,
    g_target: np.ndarray,
    *,
    n_epochs: int = 300,
    lr: float = 1e-2,
    seed: int = 0,
) -> None:
    """Train `head` in place with L2 loss to a target slot vector.

    Inputs are numpy float32. Targets `g_target` are typically per-class
    centroids of the post-revision class in the trained slot space.
    """
    torch.manual_seed(seed)
    g_pre_t = torch.tensor(g_pre, dtype=torch.float32)
    fp_t = torch.tensor(fp, dtype=torch.float32)
    g_tgt_t = torch.tensor(g_target, dtype=torch.float32)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    head.train()
    for _ in range(n_epochs):
        pred = head(g_pre_t, fp_t)
        loss = F.mse_loss(pred, g_tgt_t)
        opt.zero_grad()
        loss.backward()
        opt.step()


def apply_revision(
    G: torch.Tensor,
    factor_idx: int,
    fp: torch.Tensor,
    head: ReviseHead,
) -> torch.Tensor:
    """Return a copy of G with ONLY slot `factor_idx` revised.

    The other slots are bit-identical to the input (G2-mechanical
    guarantee). The input tensor `G` is not mutated.
    """
    if G.ndim != 3:
        raise ValueError(f"G must be (B, F, d_slot); got {tuple(G.shape)}")
    if not (0 <= factor_idx < G.shape[1]):
        raise ValueError(
            f"factor_idx={factor_idx} out of range for F={G.shape[1]}"
        )
    head.eval()
    with torch.no_grad():
        g_slot = G[:, factor_idx]
        g_revised = head(g_slot, fp)
    out = G.clone()
    out[:, factor_idx] = g_revised
    return out
