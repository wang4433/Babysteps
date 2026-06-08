"""Stage-5 — distilled MULTIMODAL attribution head (build-order step 4).

A tiny CPU-torch head that predicts WHICH revisable factor is wrong, from
multiple OBSERVABLE modalities — NOT residual alone. The user's mandate: design
the interface multimodal from the start so a residual-only ~1:1 positional rule
is never mislabeled as general attribution. Residual-only is just one
modality MASK / ablation arm of this one head.

Modalities (ordered ``MODALITY_ORDER``):

* ``res``  — expected-vs-actual state residual, encoded by the DEPLOYED
  ``_efail_vec`` (predicate one-hot ++ UNIT residual ++ present bit, 12-d).
  Unit-normalised, so this modality sees residual DIRECTION only (the positional
  cue); magnitude lives in ``traj``.
* ``traj`` — action/trajectory history, summarised to fixed length from the
  observable cube path (net displacement, path length, straightness).
* ``obs``  — observation history (frozen-encoder frame features). ALWAYS masked
  off sim-free (``d_obs=0``); the GPU hook is additive and must never pull
  gitignored ``models/`` into the login-node suite.
* ``ctx``  — task/goal context: TASK-BLIND one-hots of the inferred
  ``object_motion`` + ``contact_region`` tokens (NEVER a task id — the same
  leakage boundary as ``shared_revision_policy``). This is the modality that
  separates a Class-A hard negative (misread object_motion) from a clean
  contact_region failure with an identical residual.

The MASK is the mechanism that makes residual-only / history-only / multimodal a
single code path: a masked modality contributes zero to the fused
representation and the mask bits are concatenated so "absent" is distinguishable
from "genuinely zero".

``DistilledAttributor`` wraps the head behind the existing
``revision_policy.Attributor`` Protocol (``name='distilled'``) so it drops into
the unified evaluator's ``attributor_override`` exactly like ``oracle``/``vlm``.

Sim-free, CPU-only torch (mirrors ``residual_reviser.py`` /
``shared_revision_policy.py``: torch lives here, ``revision_policy.py`` stays
torch-free).
"""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from babysteps.schemas import Intent
from babysteps.stage5.revision_policy import (
    ZERO_COST,
    AttributionObs,
    AttributionResult,
)
from babysteps.stage5.shared_revision_policy import (
    EFAIL_DIM,
    FACTOR_ORDER,
    _efail_vec,
)

# ---- modality layout ------------------------------------------------------ #

MODALITY_ORDER: tuple[str, ...] = ("res", "traj", "obs", "ctx")

# ctx one-hot vocab (task-blind): the 3 reachable cardinal motions used by the
# PokeCube family + a 4th cardinal for completeness, and the 4 cube faces.
_CTX_MOTIONS: tuple[str, ...] = (
    "translate_+x", "translate_-x", "translate_+y", "translate_-y")
_CTX_FACES: tuple[str, ...] = (
    "minus_x_face", "plus_x_face", "minus_y_face", "plus_y_face")

D_RES: int = EFAIL_DIM           # 12
D_TRAJ: int = 5                  # net_dx, net_dy, path_len, disp_norm, straightness
D_CTX: int = len(_CTX_MOTIONS) + len(_CTX_FACES)   # 8

# Named mask presets (one code path; the ablation knob).
MASK_PRESETS: dict[str, tuple[int, int, int, int]] = {
    "residual_only": (1, 0, 0, 0),
    "traj_only": (0, 1, 0, 0),
    "ctx_only": (0, 0, 0, 1),
    "res_traj": (1, 1, 0, 0),
    "res_ctx": (1, 0, 0, 1),
    "traj_ctx": (0, 1, 0, 1),
    "multimodal": (1, 1, 0, 1),      # default sim-free (obs off)
    "multimodal_gpu": (1, 1, 1, 1),  # obs on (needs d_obs>0 + real features)
}


# ---- feature builders ----------------------------------------------------- #

def build_residual_feat(predicate: Optional[str], residual_xy) -> np.ndarray:
    """Residual modality: the DEPLOYED 12-d e_fail encoding (delegates to
    ``shared_revision_policy._efail_vec``). Unit-normalised residual + predicate
    one-hot + present bit — direction only, by design."""
    return _efail_vec(predicate, residual_xy)


def build_traj_feat(trajectory_xy: Sequence[Sequence[float]]) -> np.ndarray:
    """Trajectory modality: fixed-length summary of the observable cube path.

    [net_dx, net_dy, path_length, displacement_norm, straightness]. Empty / 1-pt
    paths -> zeros (mask it off upstream if truly absent)."""
    v = np.zeros(D_TRAJ, dtype=np.float32)
    if trajectory_xy is None:
        return v
    pts = np.asarray(trajectory_xy, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 2:
        return v
    net = pts[-1] - pts[0]
    steps = np.diff(pts, axis=0)
    path_len = float(np.linalg.norm(steps, axis=1).sum())
    disp = float(np.linalg.norm(net))
    v[0], v[1] = float(net[0]), float(net[1])
    v[2] = path_len
    v[3] = disp
    v[4] = disp / path_len if path_len > 1e-9 else 0.0
    return v


def build_ctx_feat(initial_intent: Optional[Intent],
                   factor_menu: tuple[str, ...] = ()) -> np.ndarray:
    """Context modality: task-blind one-hots of object_motion + contact_region.

    NEVER encodes a task id. ``factor_menu`` is accepted for interface symmetry
    but not embedded (its length is constant within a task family)."""
    v = np.zeros(D_CTX, dtype=np.float32)
    if initial_intent is None:
        return v
    om = getattr(initial_intent, "object_motion", None)
    cr = getattr(initial_intent, "contact_region", None)
    if om in _CTX_MOTIONS:
        v[_CTX_MOTIONS.index(om)] = 1.0
    if cr in _CTX_FACES:
        v[len(_CTX_MOTIONS) + _CTX_FACES.index(cr)] = 1.0
    return v


def features_from_example(ex) -> dict[str, np.ndarray]:
    """Modality feature dict from an ``attribution_dataset.Example``."""
    return {
        "res": build_residual_feat(ex.predicate, ex.residual_xy),
        "traj": build_traj_feat(ex.trajectory_xy),
        "ctx": build_ctx_feat(ex.initial_intent, ex.factor_menu),
    }


def features_from_obs(obs: AttributionObs) -> dict[str, np.ndarray]:
    """Modality feature dict from an ``AttributionObs`` (additive fields)."""
    return {
        "res": build_residual_feat(obs.failure_predicate,
                                   getattr(obs, "residual_xy", None)),
        "traj": build_traj_feat(getattr(obs, "trajectory_xy", ()) or ()),
        "ctx": build_ctx_feat(obs.initial_intent, obs.factor_menu),
    }


# ---- the head ------------------------------------------------------------- #

class AttributionHead(nn.Module):
    """Per-modality encoders + explicit modality-presence mask -> factor logits.

    ``forward(feats, mask)``: ``feats`` keys ``res``/``traj``/``ctx`` (and
    ``obs`` iff ``d_obs>0``) are float tensors ``(B, d_k)``; ``mask`` is
    ``(B, 4)`` in {0,1} ordered ``MODALITY_ORDER``. A masked modality's encoder
    output is zeroed (no contribution, no gradient); the mask bits are appended
    to the fused vector so the classifier can tell "absent" from "zero"."""

    def __init__(self, *, d_res: int = D_RES, d_traj: int = D_TRAJ,
                 d_obs: int = 0, d_ctx: int = D_CTX, d_embed: int = 16,
                 hidden: int = 64, n_factors: int = len(FACTOR_ORDER),
                 seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.d_res, self.d_traj, self.d_obs, self.d_ctx = d_res, d_traj, d_obs, d_ctx
        self.d_embed, self.hidden, self.n_factors = d_embed, hidden, n_factors
        self.enc_res = nn.Linear(d_res, d_embed)
        self.enc_traj = nn.Linear(d_traj, d_embed)
        self.enc_obs = nn.Linear(d_obs, d_embed) if d_obs > 0 else None
        self.enc_ctx = nn.Linear(d_ctx, d_embed)
        fuse_dim = 4 * d_embed + 4
        self.cls = nn.Sequential(
            nn.Linear(fuse_dim, hidden), nn.GELU(),
            nn.Linear(hidden, n_factors),
        )

    def forward(self, feats: Mapping[str, torch.Tensor],
                mask: torch.Tensor) -> torch.Tensor:
        b = mask.shape[0]
        e_res = self.enc_res(feats["res"]) * mask[:, 0:1]
        e_traj = self.enc_traj(feats["traj"]) * mask[:, 1:2]
        if self.enc_obs is not None and "obs" in feats:
            e_obs = self.enc_obs(feats["obs"]) * mask[:, 2:3]
        else:
            e_obs = torch.zeros(b, self.d_embed, dtype=e_res.dtype)
        e_ctx = self.enc_ctx(feats["ctx"]) * mask[:, 3:4]
        x = torch.cat([e_res, e_traj, e_obs, e_ctx, mask.to(e_res.dtype)], dim=-1)
        return self.cls(x)


def resolve_mask(mask) -> tuple[int, int, int, int]:
    """Accept a preset name or a 4-tuple; return the 4-tuple."""
    if isinstance(mask, str):
        if mask not in MASK_PRESETS:
            raise KeyError(f"unknown mask preset {mask!r}; "
                           f"choose from {sorted(MASK_PRESETS)}")
        return MASK_PRESETS[mask]
    t = tuple(int(x) for x in mask)
    if len(t) != 4:
        raise ValueError(f"mask must have 4 entries (res,traj,obs,ctx), got {t}")
    return t  # type: ignore[return-value]


def _stack(feats_list: list[dict[str, np.ndarray]], d_obs: int
           ) -> dict[str, torch.Tensor]:
    out = {
        "res": torch.tensor(np.stack([f["res"] for f in feats_list])),
        "traj": torch.tensor(np.stack([f["traj"] for f in feats_list])),
        "ctx": torch.tensor(np.stack([f["ctx"] for f in feats_list])),
    }
    if d_obs > 0:
        out["obs"] = torch.tensor(np.stack(
            [f.get("obs", np.zeros(d_obs, dtype=np.float32))
             for f in feats_list]))
    return out


# ---- attributor wrapper --------------------------------------------------- #

class DistilledAttributor:
    """``Attributor`` impl wrapping a frozen :class:`AttributionHead`.

    Mirrors ``VLMAttributor`` / ``OracleAttributor``: loaded once, no per-episode
    model build. Sim-free (``cost = ZERO_COST``); the only cost is the CPU forward
    pass timed into ``latency_s``. The argmax is restricted to ``obs.factor_menu``
    so the head never emits a factor (e.g. ``direction_grounding``) absent from
    the menu.

    SCOPE (do not overclaim): this replaces the VLM's factor-DIAGNOSIS step, not
    its intent inference. It consumes the already-inferred intent tokens
    (``obs.initial_intent`` feeds the ctx modality) plus observable execution
    feedback. The latency comparison against the ~3.05 s VLM full replan is fair
    only when intent inference happens separately (decoded from the demo / cached
    upstream), which is exactly the deployed loop's structure."""
    name = "distilled"

    def __init__(self, head: AttributionHead, *,
                 default_mask=(1, 1, 0, 1),
                 factor_order: tuple[str, ...] = FACTOR_ORDER) -> None:
        if len(factor_order) != head.n_factors:
            raise ValueError(
                f"factor_order length {len(factor_order)} != head.n_factors "
                f"{head.n_factors}; the menu mask would misalign with the logits")
        self.head = head.eval()
        self.default_mask = resolve_mask(default_mask)
        self.factor_order = tuple(factor_order)

    def attribute(self, obs: AttributionObs) -> AttributionResult:
        feats = features_from_obs(obs)
        ftens = _stack([feats], self.head.d_obs)
        mask = torch.tensor([self.default_mask], dtype=torch.float32)
        t0 = perf_counter()
        with torch.no_grad():
            logits = self.head(ftens, mask)[0]
            # Restrict to the menu: mask out-of-menu factors to -inf.
            menu = set(obs.factor_menu)
            neg = torch.full_like(logits, float("-inf"))
            allowed = torch.tensor(
                [1.0 if f in menu else 0.0 for f in self.factor_order])
            logits = torch.where(allowed.bool(), logits, neg)
            probs = torch.softmax(logits, dim=0)
        dt = perf_counter() - t0
        idx = int(torch.argmax(probs).item())
        factor = self.factor_order[idx]
        conf = float(probs[idx].item())
        return AttributionResult(factor=factor, confidence=conf,
                                 latency_s=dt, cost=dict(ZERO_COST))

    @classmethod
    def from_pack(cls, path, *, default_mask=None) -> "DistilledAttributor":
        head, cfg = load_attribution_head(path)
        mask = default_mask if default_mask is not None \
            else cfg.get("default_mask", (1, 1, 0, 1))
        return cls(head, default_mask=mask,
                   factor_order=tuple(cfg.get("factor_order", FACTOR_ORDER)))


# ---- training / evaluation (sim-free, CPU) -------------------------------- #

def train_attribution_head(
    examples, *, modality_dropout: float = 0.5, fixed_mask=None,
    epochs: int = 400, lr: float = 1e-2, d_obs: int = 0, d_embed: int = 16,
    hidden: int = 64, seed: int = 0,
) -> AttributionHead:
    """Train a head over ``examples`` with cross-entropy on the FACTOR_ORDER
    index of ``true_factor``.

    Train-time MODALITY DROPOUT (per-row Bernoulli mask, always keep >=1) makes
    ONE weight set serve every eval mask. Pass ``fixed_mask`` to train a single
    arm instead (e.g. an honest residual-only baseline). ``obs`` is never
    dropped-in here (sim-free, ``d_obs=0``).

    INTERPRETATION (do not overclaim): on the synthetic PokeCube geometry the
    ``ctx`` modality alone reaches 1.000 (context SUFFICIENCY) because the
    (object_motion, contact_region) token pair is a perfect label separator BY
    CONSTRUCTION — a misread object_motion token IS the definition of a Class-A
    hard negative. So a high ``multimodal`` score here demonstrates that a
    positional/residual shortcut is INSUFFICIENT and symbolic intent context is
    NECESSARY — it does NOT demonstrate multimodal FUSION (no modality
    combination is needed once ctx is present) or generalisation (only 12 base
    geometries). Genuine fusion requires a regime where no single modality is
    clean — the GPU/real-pixels setting where intent tokens are INFERRED (hence
    imperfect), not given. See ``attribution_dataset`` honesty note."""
    head = AttributionHead(d_obs=d_obs, d_embed=d_embed, hidden=hidden, seed=seed)
    feats_list = [features_from_example(e) for e in examples]
    ftens = _stack(feats_list, d_obs)
    factor_idx = {f: i for i, f in enumerate(FACTOR_ORDER)}
    labels = torch.tensor([factor_idx[e.true_factor] for e in examples],
                          dtype=torch.long)
    n = len(examples)
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    base = resolve_mask(fixed_mask) if fixed_mask is not None else None
    # obs is always off sim-free (d_obs=0) -> never sample it in dropout.
    droppable = [0, 1, 3]  # res, traj, ctx
    head.train()
    for _ in range(epochs):
        if base is not None:
            mask = torch.tensor(np.tile(base, (n, 1)), dtype=torch.float32)
        else:
            m = np.zeros((n, 4), dtype=np.float32)
            keep = rng.random((n, len(droppable))) > modality_dropout
            for j, slot in enumerate(droppable):
                m[:, slot] = keep[:, j]
            # guarantee >=1 modality per row
            empty = m[:, droppable].sum(axis=1) == 0
            if empty.any():
                forced = rng.integers(0, len(droppable), size=int(empty.sum()))
                m[np.where(empty)[0], np.array(droppable)[forced]] = 1.0
            mask = torch.tensor(m, dtype=torch.float32)
        logits = head(ftens, mask)
        loss = ce(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
    head.eval()
    return head


def evaluate_attribution(head: AttributionHead, examples, mask) -> dict:
    """Attribution accuracy under a fixed ``mask`` (preset name or 4-tuple),
    overall and split by ``true_factor`` and by example ``kind``."""
    m = resolve_mask(mask)
    feats_list = [features_from_example(e) for e in examples]
    ftens = _stack(feats_list, head.d_obs)
    mt = torch.tensor(np.tile(m, (len(examples), 1)), dtype=torch.float32)
    head.eval()
    with torch.no_grad():
        pred = torch.argmax(head(ftens, mt), dim=1).tolist()
    correct, by_factor, by_kind = 0, {}, {}
    for e, p in zip(examples, pred):
        ok = (FACTOR_ORDER[p] == e.true_factor)
        correct += int(ok)
        by_factor.setdefault(e.true_factor, [0, 0])
        by_factor[e.true_factor][0] += int(ok)
        by_factor[e.true_factor][1] += 1
        kind = e.meta.get("kind", "?")
        by_kind.setdefault(kind, [0, 0])
        by_kind[kind][0] += int(ok)
        by_kind[kind][1] += 1
    return {
        "mask": m,
        "accuracy": correct / max(1, len(examples)),
        "n": len(examples),
        "by_factor": {k: v[0] / max(1, v[1]) for k, v in by_factor.items()},
        "by_kind": {k: v[0] / max(1, v[1]) for k, v in by_kind.items()},
    }


# ---- save / load ---------------------------------------------------------- #

def save_attribution_head(head: AttributionHead, path, *,
                          default_mask=(1, 1, 0, 1),
                          factor_order: tuple[str, ...] = FACTOR_ORDER) -> None:
    blob = {
        "state_dict": head.state_dict(),
        "init": {"d_res": head.d_res, "d_traj": head.d_traj,
                 "d_obs": head.d_obs, "d_ctx": head.d_ctx,
                 "d_embed": head.d_embed, "hidden": head.hidden,
                 "n_factors": head.n_factors},
        "default_mask": list(resolve_mask(default_mask)),
        "factor_order": list(factor_order),
    }
    torch.save(blob, Path(path))


def load_attribution_head(path):
    blob = torch.load(Path(path), weights_only=False)
    head = AttributionHead(**blob["init"], seed=0)
    head.load_state_dict(blob["state_dict"])
    head.eval()
    cfg = {"default_mask": tuple(blob.get("default_mask", (1, 1, 0, 1))),
           "factor_order": tuple(blob.get("factor_order", FACTOR_ORDER))}
    return head, cfg
