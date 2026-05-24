"""Stage-4 M2.5 — learned attribution head.

A small MLP classifier `(FailurePacket, Intent) → wrong_factor ∈ INTENT_FIELDS`
that replaces the rule-based `babysteps.failure.attribute_failure` for the
`latent_revision` policy. Sim-free, CPU-only, deterministic.

Trained per-task; the trained head is loaded into `LatentPack.attribution_head`
(optional field), and the `latent_revision_factory` overrides
`ctx.attribution.wrong_factor` with `head.predict(fp, intent)` before
running ReviseHead. The single-factor revision invariant is preserved by
construction: one slot, one ReviseHead call, one Revision row.

Source of supervision: `oracle_wrong_factor` on every failed Stage-0
episode JSON (the analytic upper bound `goal.md` §"Stage 4 / Data
Dependencies" defines as the teacher signal).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from babysteps.schemas import INTENT_FIELDS
from babysteps.stage4.attribution_features import (
    FEATURE_DIM,
    vectorize_attribution_input,
)

N_CLASSES: int = len(INTENT_FIELDS)


class AttributionHead(nn.Module):
    """Linear → GELU → Linear over (failure_packet, intent) features.

    Output is a logits tensor of shape (B, N_CLASSES). `predict_factor`
    returns the argmax mapped through INTENT_FIELDS.
    """

    def __init__(
        self,
        *,
        in_dim: int = FEATURE_DIM,
        hidden: int = 64,
        n_classes: int = N_CLASSES,
        seed: int = 0,
    ):
        super().__init__()
        torch.manual_seed(seed)
        self.in_dim = in_dim
        self.hidden = hidden
        self.n_classes = n_classes
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.no_grad()
    def predict_factor(self, fp_fields, intent) -> str:
        """Return the predicted wrong_factor string."""
        x = torch.from_numpy(
            vectorize_attribution_input(fp_fields, intent).astype(np.float32)
        ).unsqueeze(0)
        logits = self.forward(x)
        idx = int(torch.argmax(logits, dim=-1).item())
        return INTENT_FIELDS[idx]


# ---- Training -------------------------------------------------------- #


def train_attribution_head(
    head: AttributionHead,
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_epochs: int = 300,
    lr: float = 1e-2,
    class_weights: Optional[np.ndarray] = None,
    seed: int = 0,
) -> dict:
    """Full-batch cross-entropy training. Returns final {loss, acc}."""
    if X.ndim != 2 or X.shape[1] != head.in_dim:
        raise ValueError(f"X shape mismatch: {X.shape} vs in_dim={head.in_dim}")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError(f"y shape mismatch: {y.shape}")
    torch.manual_seed(seed)
    x_t = torch.from_numpy(X.astype(np.float32))
    y_t = torch.from_numpy(y.astype(np.int64))
    w_t = (torch.from_numpy(class_weights.astype(np.float32))
           if class_weights is not None else None)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    head.train()
    last_loss = float("nan")
    last_acc = float("nan")
    for _ in range(n_epochs):
        opt.zero_grad()
        logits = head(x_t)
        loss = F.cross_entropy(logits, y_t, weight=w_t)
        loss.backward()
        opt.step()
        last_loss = float(loss.item())
        with torch.no_grad():
            last_acc = float((logits.argmax(dim=-1) == y_t).float().mean())
    head.eval()
    return {"loss": last_loss, "acc": last_acc}


# ---- Persistence ----------------------------------------------------- #


def save_attribution_head(head: AttributionHead, path: Path) -> None:
    """Save state-dict + init kwargs to `path` (a single .pt file)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": head.state_dict(),
        "init": {
            "in_dim": head.in_dim,
            "hidden": head.hidden,
            "n_classes": head.n_classes,
        },
    }, path)


def load_attribution_head(path: Path) -> AttributionHead:
    blob = torch.load(Path(path), weights_only=False)
    head = AttributionHead(**blob["init"], seed=0)
    head.load_state_dict(blob["state_dict"])
    head.eval()
    return head


# ---- Data loading from disk records ---------------------------------- #


def build_training_pairs(
    records: Sequence[dict],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract (X, y, dropped_predicates) from a list of episode dicts.

    Skips records without a real failure (`failure_predicate == "none"`)
    or without an `oracle_wrong_factor` label. Returns label codes in
    INTENT_FIELDS order.
    """
    xs: list[np.ndarray] = []
    ys: list[int] = []
    dropped: list[str] = []
    for r in records:
        fp = r.get("failure_packet")
        exe = r.get("execution") or {}
        intent = exe.get("initial_intent")
        if not fp or not intent:
            continue
        pred = fp.get("failure_predicate")
        if pred == "none" or pred is None:
            continue
        label = fp.get("oracle_wrong_factor")
        if label is None or label not in INTENT_FIELDS:
            dropped.append(str(label))
            continue
        try:
            xs.append(vectorize_attribution_input(fp, intent))
        except ValueError as e:
            dropped.append(str(e))
            continue
        ys.append(INTENT_FIELDS.index(label))
    if not xs:
        return (
            np.zeros((0, FEATURE_DIM), dtype=np.float64),
            np.zeros((0,), dtype=np.int64),
            dropped,
        )
    return np.stack(xs, axis=0), np.asarray(ys, dtype=np.int64), dropped


def class_weights_inverse(y: np.ndarray, n_classes: int = N_CLASSES) -> np.ndarray:
    """Per-class weights = N_total / (n_classes * count_c). Mirrors sklearn."""
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    n = float(len(y))
    w = np.ones(n_classes, dtype=np.float64)
    for c in range(n_classes):
        if counts[c] > 0:
            w[c] = n / (n_classes * counts[c])
    return w
