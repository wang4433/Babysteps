"""Stage-5 — the shared, task-general RevisionPolicy (build-order step 2).

ONE checkpoint, shared parameters across tasks. This realises the target's
dynamic candidate scorer (``redesign_failure_paradigm.md`` → "Shared policy
interface"). The target form is
``score(v) = MLP(g_i, e_fail, z, embed(factor), embed(candidate_v))``; the
**step-2 realised scorer narrows it to**

    score(v) = MLP(g_i, e_fail, embed(factor), embed(current), embed(candidate_v))

scored per candidate, softmax across the (variable-size) candidate set,
single-slot write.

**``z`` (observable object/task context) is intentionally NOT consumed at
step 2.** No observable-context featurizer exists yet, and a task-blind ``z``
encoder fed zeros would be dead capacity (and a careless one is a task-id leak
channel). ``RevisionRequest.z`` is kept as the reserved interface slot; wiring a
real ``z`` encoder is an additive change (a new ``d_z`` input block) made only
when a documented, task-blind featurizer exists. ``decide`` therefore never
reads ``req.z`` today — the narrowing is explicit, not an oversight. It is **not** a pooled IntentHead — the design workflow's
judge panel showed that a pooled slot space is cosmetic here (no factor is
shared between PushCube ``contact_region`` and StackCube ``goal_state``, so a
pooled trunk would just bake in task identification). The genuinely shared,
generalisation-ready component is the scorer over a **shared factor ontology +
shared value embeddings**; the per-task IntentHead packs are left untouched and
``g_i`` is an optional, standardised, non-load-bearing input.

No-leakage boundary (the same one ``revision_policy.RevisionRequest`` enforces):
the policy reads ONLY ``req.{factor, current_value, candidates, e_fail, g_i}`` —
never the task id, gt, oracle factor, scene, or full intent. Honest scope: at
the 2-task step the *factor name itself* + the ``e_fail`` signature are indirect
task proxies, so a pooled checkpoint shows **multi-task capability, not
leave-one-task-family-out generalisation**. The architecture stays
generalisation-ready (task-blind scorer, full-schema embeddings, no task id);
the synthetic leave-one-family-out + leakage-probe unit tests demonstrate the
task-id-free path on the login node before any step-3 task exists.

Torch lives here (mirroring ``residual_reviser.py``); ``revision_policy.py``
stays pure/torch-free and keeps its raising ``SharedRevisionPolicy`` placeholder
as the deferred-contract marker. ``conditions.shared_revision_policy`` stays
``runnable=False`` until a trained checkpoint exists (step 5).

Sim-free, CPU-only torch.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import torch
import torch.nn as nn

from babysteps.schemas import INTENT_FIELDS
from babysteps.stage4.revise_head import _PREDICATE_ORDER
from babysteps.stage5.revision_policy import (
    RevisionDecision,
    RevisionRequest,
    _FACTOR_VOCAB,
)

# Pinned factor ontology: the 6 core factors + the additive 7th. 7 rows so step-3
# / step-5 never shift indices (only the 6 reachable via the compiler are used
# today). Task id is NEVER an input — only the diagnosed factor name.
FACTOR_ORDER: tuple[str, ...] = INTENT_FIELDS + ("direction_grounding",)
_FACTOR_INDEX: dict[str, int] = {f: i for i, f in enumerate(FACTOR_ORDER)}

# e_fail vector layout (12-d): predicate one-hot (reuses the frozen
# revise_head._PREDICATE_ORDER, len 9) + unit residual (2) + residual-present bit
# (1). The present bit TELLS the MLP the residual is absent (StackCube) rather
# than forcing it to infer that zeros mean "missing".
_PREDICATE_INDEX: dict[str, int] = {p: i for i, p in enumerate(_PREDICATE_ORDER)}
EFAIL_DIM: int = len(_PREDICATE_ORDER) + 2 + 1

D_SLOT_DEFAULT: int = 32        # both committed packs use d_slot=32
GI_DIM_DEFAULT: int = D_SLOT_DEFAULT + 1   # slot + present bit


def build_value_vocab() -> dict[tuple[str, str], int]:
    """Shared (factor, token) → index map over the FULL schema union, so a
    held-out task family's correct value is embeddable (LOTO stays structurally
    selectable). Index 0 is reserved for UNK / out-of-vocab. Keying by
    (factor, token) avoids cross-factor collisions (e.g. ``none`` namespaces).
    Deterministic order: factors in FACTOR_ORDER, tokens sorted."""
    vocab: dict[tuple[str, str], int] = {}
    idx = 1
    for factor in FACTOR_ORDER:
        for token in sorted(_FACTOR_VOCAB[factor]):
            vocab[(factor, token)] = idx
            idx += 1
    return vocab


def _efail_vec(predicate: Optional[str], residual_xy) -> np.ndarray:
    """(predicate one-hot ++ unit residual ++ residual-present bit) → float32[12].
    Tolerates a missing predicate (all-zero block) and a ``None`` / zero
    residual (zeros + present-bit 0). Does NOT mutate the load-bearing
    ``vectorize_failure_packet_residual`` and does NOT re-encode the factor."""
    v = np.zeros(EFAIL_DIM, dtype=np.float32)
    if predicate in _PREDICATE_INDEX:
        v[_PREDICATE_INDEX[predicate]] = 1.0
    if residual_xy is not None:
        res = np.asarray(residual_xy, dtype=np.float32).reshape(-1)[:2]
        n = float(np.linalg.norm(res))
        if n > 1e-9:
            v[len(_PREDICATE_ORDER):len(_PREDICATE_ORDER) + 2] = res / n
            v[-1] = 1.0
    return v


def _gi_vec(g_i, scaler: Optional[Mapping], dim: int = GI_DIM_DEFAULT) -> np.ndarray:
    """(standardised slot ++ present bit) → float32[dim]. ``g_i is None`` →
    zeros + present-bit 0. ``scaler`` is ``{"mean", "scale"}`` (persisted in the
    checkpoint) so the raw 32-d slot magnitude cannot drown the embeddings."""
    out = np.zeros(dim, dtype=np.float32)
    if g_i is not None:
        slot = np.asarray(g_i, dtype=np.float32).reshape(-1)
        if scaler is not None:
            mean = np.asarray(scaler["mean"], dtype=np.float32)
            scale = np.asarray(scaler["scale"], dtype=np.float32)
            slot = (slot - mean) / np.where(scale > 1e-8, scale, 1.0)
        k = min(len(slot), dim - 1)
        out[:k] = slot[:k]
        out[-1] = 1.0
    return out


class SharedScorer(nn.Module):
    """Score ONE (factor, current_value, candidate) pair given e_fail + optional
    g_i. Shared across tasks; no task id; ``z`` is NOT an input at step 2 (see
    the module docstring — reserved, additive when a real featurizer exists).
    Forward is batched over candidates."""

    def __init__(self, *, vocab_size: int, n_factors: int = len(FACTOR_ORDER),
                 d_gi: int = GI_DIM_DEFAULT, d_efail: int = EFAIL_DIM,
                 d_factor: int = 8, d_value: int = 16, hidden: int = 128,
                 seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.vocab_size = vocab_size
        self.n_factors = n_factors
        self.d_gi = d_gi
        self.d_efail = d_efail
        self.d_factor = d_factor
        self.d_value = d_value
        self.hidden = hidden
        self.factor_emb = nn.Embedding(n_factors, d_factor)
        self.value_emb = nn.Embedding(vocab_size, d_value)
        d_in = d_gi + d_efail + d_factor + d_value + d_value
        self.mlp = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(),
            nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, g_i: torch.Tensor, e_fail: torch.Tensor,
                factor_idx: torch.Tensor, current_idx: torch.Tensor,
                cand_idx: torch.Tensor) -> torch.Tensor:
        """All inputs batched over candidates (B). Returns (B,) logits."""
        fe = self.factor_emb(factor_idx)
        cve = self.value_emb(current_idx)
        cae = self.value_emb(cand_idx)
        x = torch.cat([g_i, e_fail, fe, cve, cae], dim=-1)
        return self.mlp(x).squeeze(-1)


class SharedScorerPolicy:
    """``RevisionPolicy`` impl wrapping a :class:`SharedScorer`. One checkpoint
    serves all tasks. Never re-chooses the factor; returns ``new_value=None``
    (abstain/keep) when the scorer ranks the current value highest."""
    name = "shared_revision_policy"

    def __init__(self, scorer: SharedScorer,
                 value_vocab: dict[tuple[str, str], int],
                 factor_order: tuple[str, ...] = FACTOR_ORDER,
                 scaler: Optional[Mapping] = None) -> None:
        self.scorer = scorer.eval()
        self.value_vocab = dict(value_vocab)
        self.factor_order = tuple(factor_order)
        self._factor_index = {f: i for i, f in enumerate(self.factor_order)}
        self.scaler = scaler

    def decide(self, req: RevisionRequest) -> RevisionDecision:
        # NOTE: req.z (observable context) is intentionally NOT read at step 2 —
        # the interface is explicitly narrowed (see module docstring).
        factor = req.factor
        # In-vocab candidates only; a genuine out-of-schema hallucination is
        # dropped. current_value → UNK (0) if OOV.
        in_vocab = [c for c in req.candidates
                    if (factor, c) in self.value_vocab]
        if factor not in self._factor_index or not in_vocab:
            return self._fallback(req)

        gi = _gi_vec(req.g_i, self.scaler, dim=self.scorer.d_gi)
        ef = _efail_vec(req.e_fail.predicate if req.e_fail else None,
                        req.e_fail.residual_xy if req.e_fail else None)
        b = len(in_vocab)
        g_i_t = torch.tensor(gi, dtype=torch.float32).expand(b, -1)
        ef_t = torch.tensor(ef, dtype=torch.float32).expand(b, -1)
        fidx = torch.full((b,), self._factor_index[factor], dtype=torch.long)
        cur = torch.full(
            (b,), self.value_vocab.get((factor, req.current_value), 0),
            dtype=torch.long)
        cand = torch.tensor(
            [self.value_vocab[(factor, c)] for c in in_vocab], dtype=torch.long)
        with torch.no_grad():
            logits = self.scorer(g_i_t, ef_t, fidx, cur, cand)
            probs = torch.softmax(logits, dim=0)
        w = int(torch.argmax(probs).item())
        winner = in_vocab[w]
        conf = float(probs[w].item())
        # Abstain: ranking the current value highest means "keep" — this is how
        # an uninformative e_fail or a mis-diagnosis avoids a forced harmful flip.
        new_value = None if winner == req.current_value else winner
        return RevisionDecision(
            factor=factor, new_value=new_value, confidence=conf,
            telemetry={"policy": self.name, "path": "scored"})

    def _fallback(self, req: RevisionRequest) -> RevisionDecision:
        # OOV factor / all-OOV candidate set → ABSTAIN (new_value=None), never
        # emit an out-of-schema token. Emitting a non-schema candidate would be
        # an invalid edit (it would fail Intent validation in the compiler), so
        # the safe action when the scorer cannot ground any typed candidate is
        # to keep the current value and let the failure surface as a non-repair.
        return RevisionDecision(
            factor=req.factor, new_value=None, confidence=0.0,
            telemetry={"policy": self.name, "path": "fallback_abstain"})

    @classmethod
    def from_pack(cls, path) -> "SharedScorerPolicy":
        """Load a trained checkpoint (state_dict + vocab + factor_order +
        scaler). Used by the GPU/step-5 run; gitignored artifact."""
        scorer, vocab, factor_order, scaler = load_shared_scorer(path)
        return cls(scorer, vocab, factor_order=factor_order, scaler=scaler)


def save_shared_scorer(scorer: SharedScorer,
                       value_vocab: dict[tuple[str, str], int], path,
                       *, factor_order: tuple[str, ...] = FACTOR_ORDER,
                       scaler: Optional[Mapping] = None) -> None:
    blob = {
        "state_dict": scorer.state_dict(),
        "init": {
            "vocab_size": scorer.vocab_size, "n_factors": scorer.n_factors,
            "d_gi": scorer.d_gi, "d_efail": scorer.d_efail,
            "d_factor": scorer.d_factor, "d_value": scorer.d_value,
            "hidden": scorer.hidden,
        },
        # store vocab as an idx-ordered list of [factor, token] (UNK=0 omitted).
        "value_vocab": [[f, t] for (f, t), _ in sorted(
            value_vocab.items(), key=lambda kv: kv[1])],
        "factor_order": list(factor_order),
        "scaler": (None if scaler is None else {
            "mean": np.asarray(scaler["mean"], dtype=np.float32),
            "scale": np.asarray(scaler["scale"], dtype=np.float32)}),
    }
    torch.save(blob, Path(path))


def load_shared_scorer(path):
    blob = torch.load(Path(path), weights_only=False)
    scorer = SharedScorer(**blob["init"], seed=0)
    scorer.load_state_dict(blob["state_dict"])
    scorer.eval()
    vocab = {(f, t): i + 1 for i, (f, t) in enumerate(blob["value_vocab"])}
    return scorer, vocab, tuple(blob["factor_order"]), blob.get("scaler")
