"""Stage-4 M2a — latent-revision RetryPolicy.

Wraps a trained `(IntentHead, ReviseHead, centroids, label_tokens)` bundle
into a `RetryPolicy` callable that plugs into `babysteps.episode.run_episode`
identically to the M3 procedural baselines (one_shot, babysteps_selective,
…). The single-factor revision invariant comes from the slot-local
ReviseHead interface and the per-factor centroid `slot_decode`; the policy
itself just wires them up against the live `RetryContext`.

A2 deliverable; will be reused by the A3 G4/G5 Δpp eval (sim rollouts of
this policy vs. babysteps_selective on the same seeds).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace as dc_replace
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from babysteps.policies import RetryContext, RetryPolicy
from babysteps.schemas import INTENT_FIELDS, Intent, Revision
from babysteps.stage4.intent_head import IntentHead
from babysteps.stage4.revise_head import (
    FP_VECTOR_DIM, ReviseHead, apply_revision, vectorize_failure_packet,
)
from babysteps.stage4.slot_decode import decode_slot


@dataclass(frozen=True)
class LatentPack:
    """Container for everything `latent_revision_factory` needs at inference.

    Attributes
    ----------
    intent_head : IntentHead
        Trained encoder Z → G (one per task in M2a).
    revise_head : ReviseHead
        Trained slot-local editor (g_slot, fp_vec) → g_slot_revised.
    centroids : dict[int, dict[int, np.ndarray]]
        `{factor_idx: {class_int: centroid_vec}}` — output of
        `babysteps.stage4.slot_decode.build_factor_centroids`.
    label_tokens : dict[int, tuple[str, ...]]
        Per-factor (factor_idx → ordered token tuple). `class_int=i`
        corresponds to `label_tokens[factor_idx][i]`. This is what
        lets the decoded integer round-trip back to a Stage-0 schema
        string.
    """
    intent_head: IntentHead
    revise_head: ReviseHead
    centroids: dict[int, dict[int, np.ndarray]]
    label_tokens: dict[int, tuple[str, ...]]


def _build_revision(ctx: RetryContext, *, factor: str,
                    new_value: str) -> tuple[Intent, Revision]:
    """Replace exactly one factor on `ctx.initial_intent`."""
    revised = dc_replace(ctx.initial_intent, **{factor: new_value})
    revision = Revision(
        operator="latent_revision",
        factor=factor,
        old_value=getattr(ctx.initial_intent, factor),
        new_value=new_value,
        frozen_factors=tuple(f for f in INTENT_FIELDS if f != factor),
    )
    return revised, revision


def latent_revision_factory(pack: LatentPack) -> RetryPolicy:
    """Return a `RetryPolicy` closure over the trained M2a pack."""

    def _policy(ctx: RetryContext) -> Optional[tuple[Intent, Revision]]:
        wrong_factor = ctx.attribution.wrong_factor
        if wrong_factor is None or wrong_factor not in INTENT_FIELDS:
            return None  # nothing to revise

        # Without demo features we cannot run the encoder; fall back to a
        # no-op revision (keep the initial intent) rather than crash. This
        # mirrors how same_intent_retry behaves under no information.
        if ctx.demo_features is None:
            return ctx.initial_intent, Revision(
                operator="latent_revision",
                factor=wrong_factor,
                old_value=getattr(ctx.initial_intent, wrong_factor),
                new_value=getattr(ctx.initial_intent, wrong_factor),
                frozen_factors=tuple(f for f in INTENT_FIELDS if f != wrong_factor),
            )

        factor_idx = INTENT_FIELDS.index(wrong_factor)
        # No centroid bank for this factor → no slot region to decode
        # against. Return the initial intent unchanged (the cert reports
        # this as an uncertable case; see m2a_a2 notes).
        if factor_idx not in pack.centroids:
            return ctx.initial_intent, Revision(
                operator="latent_revision",
                factor=wrong_factor,
                old_value=getattr(ctx.initial_intent, wrong_factor),
                new_value=getattr(ctx.initial_intent, wrong_factor),
                frozen_factors=tuple(f for f in INTENT_FIELDS if f != wrong_factor),
            )

        # Encoder forward → G (1, F, d_slot)
        with torch.no_grad():
            z_t = torch.from_numpy(np.asarray(ctx.demo_features,
                                                 dtype=np.float32)).unsqueeze(0)
            G = pack.intent_head(z_t)
        # Vectorize the failure packet using the same encoding the
        # ReviseHead was trained on (factor one-hot + predicate one-hot).
        # ctx.failure_predicate may be None for adapters that did not
        # plumb it; in that case use a fake record with an empty
        # predicate that vectorize_failure_packet will raise on — and
        # we catch + fall back to a no-op revision.
        predicate = ctx.failure_predicate or "none"
        fake_rec = {
            "revision": {"factor": wrong_factor},
            "failure_packet": {"failure_predicate": predicate},
        }
        try:
            fp_vec = vectorize_failure_packet(fake_rec)
        except (KeyError, ValueError):
            return ctx.initial_intent, Revision(
                operator="latent_revision",
                factor=wrong_factor,
                old_value=getattr(ctx.initial_intent, wrong_factor),
                new_value=getattr(ctx.initial_intent, wrong_factor),
                frozen_factors=tuple(f for f in INTENT_FIELDS if f != wrong_factor),
            )
        fp_t = torch.from_numpy(fp_vec).unsqueeze(0)
        G_revised = apply_revision(G, factor_idx, fp_t, pack.revise_head)
        revised_slot_np = G_revised[0, factor_idx].numpy()
        decoded_int = decode_slot(revised_slot_np, pack.centroids[factor_idx])
        new_value = pack.label_tokens[factor_idx][decoded_int]
        return _build_revision(ctx, factor=wrong_factor, new_value=new_value)

    return _policy


# ---------- Save / load ------------------------------------------------- #

def save_latent_pack(pack: LatentPack, out_dir: Path) -> None:
    """Persist the pack to a directory.

    Layout:
      out_dir/intent_head.pt   - torch state dict + init kwargs
      out_dir/revise_head.pt   - torch state dict + init kwargs
      out_dir/centroids.npz    - flat arrays keyed by `f{fi}_c{ci}`
      out_dir/meta.json        - centroid index + label tokens
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # IntentHead
    torch.save({
        "state_dict": pack.intent_head.state_dict(),
        "init": {
            "z_dim": pack.intent_head.net[0].in_features,
            "n_factors": pack.intent_head.n_factors,
            "d_slot": pack.intent_head.d_slot,
            "hidden": pack.intent_head.net[0].out_features,
        },
    }, out_dir / "intent_head.pt")
    # ReviseHead
    torch.save({
        "state_dict": pack.revise_head.state_dict(),
        "init": {
            "d_slot": pack.revise_head.d_slot,
            "fp_dim": pack.revise_head.fp_dim,
            "hidden": pack.revise_head.net[0].out_features,
        },
    }, out_dir / "revise_head.pt")
    # Centroids (flat arrays, indexed via meta.json)
    arrays: dict[str, np.ndarray] = {}
    centroid_index: dict[str, list[list[int]]] = {}
    for fi, per_class in pack.centroids.items():
        centroid_index[str(fi)] = []
        for ci, vec in per_class.items():
            key = f"f{fi}_c{ci}"
            arrays[key] = vec
            centroid_index[str(fi)].append([ci, key])
    np.savez(out_dir / "centroids.npz", **arrays)
    (out_dir / "meta.json").write_text(json.dumps({
        "centroid_index": centroid_index,
        "label_tokens": {str(fi): list(toks)
                          for fi, toks in pack.label_tokens.items()},
    }, indent=2))


def load_latent_pack(in_dir: Path) -> LatentPack:
    in_dir = Path(in_dir)
    # IntentHead
    ih_blob = torch.load(in_dir / "intent_head.pt", weights_only=False)
    intent_head = IntentHead(**ih_blob["init"], seed=0)
    intent_head.load_state_dict(ih_blob["state_dict"])
    intent_head.eval()
    # ReviseHead
    rh_blob = torch.load(in_dir / "revise_head.pt", weights_only=False)
    revise_head = ReviseHead(**rh_blob["init"], seed=0)
    revise_head.load_state_dict(rh_blob["state_dict"])
    revise_head.eval()
    # Centroids
    arrs = np.load(in_dir / "centroids.npz")
    meta = json.loads((in_dir / "meta.json").read_text())
    centroids: dict[int, dict[int, np.ndarray]] = {}
    for fi_str, pairs in meta["centroid_index"].items():
        fi = int(fi_str)
        centroids[fi] = {}
        for ci, key in pairs:
            centroids[fi][int(ci)] = arrs[key]
    label_tokens = {int(fi): tuple(toks)
                    for fi, toks in meta["label_tokens"].items()}
    return LatentPack(
        intent_head=intent_head, revise_head=revise_head,
        centroids=centroids, label_tokens=label_tokens,
    )
