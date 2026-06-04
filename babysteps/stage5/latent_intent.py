"""Stage-5 — latent intent decoding (the "latent input" sever).

Instead of reading the hand-authored discrete intent JSON, the initial
intent is *decoded from demo-view vision*: third-person demo features Z
-> trained IntentHead -> latent slot tensor G -> per-factor
nearest-centroid codebook -> discrete tokens. This is the same decode
path the latent ReviseHead uses at retry time
(`babysteps.stage4.slot_decode.decode_G`), applied to the *initial*
intent so that the JSON factors are used only for supervision
(centroid training) and evaluation, never as privileged method input.

The VLM/diagnoser is deliberately outside this module: it may choose the
failed slot name after first-person execution/failure evidence, but it
does not create G and does not choose the repaired value.

Honesty boundary
----------------
* Factors that genuinely VARY in the training cut have a centroid bank
  (`pack.centroids[fi]`) and are decoded from the latent slots — this is
  the perceptual signal.
* Factors that are trivially CONSTANT in the cut have no centroid bank and
  are filled from `base_intent`. Those values are a *task constant* (the
  one value the task ever uses), not per-episode information. Decoding
  them would be a one-class tautology, so we do not pretend to.
* The discrete `Intent` is the OUTPUT contract (and the downstream eval
  reference), never the input to the encoder.

Sim-free: numpy + CPU torch only. No simulator, no I/O here.
"""
from __future__ import annotations

from dataclasses import replace as dc_replace

import numpy as np
import torch

from babysteps.schemas import INTENT_FIELDS, Intent
from babysteps.stage4.latent_policy import LatentPack
from babysteps.stage4.revise_head import (
    apply_revision, vectorize_failure_packet,
)
from babysteps.stage4.slot_decode import decode_G, decode_slot


def encode_G(pack: LatentPack, z: np.ndarray) -> np.ndarray:
    """Demo-view frozen-encoder features Z -> G of shape (1, F, d_slot)."""
    z_arr = np.asarray(z, dtype=np.float32).reshape(1, -1)
    with torch.no_grad():
        G = pack.intent_head(torch.from_numpy(z_arr))
    return G.detach().numpy()


def decode_latent_factors(pack: LatentPack, z: np.ndarray) -> dict[str, str]:
    """Decode the tokens the latent G supports (factors with a centroid bank).

    Returns ``{factor_name: token}`` only for factors that have centroids
    (i.e. genuinely vary in the cut). Trivially-constant factors are
    absent — the caller fills them from a task constant.
    """
    G = encode_G(pack, z)
    decoded = decode_G(G, pack.centroids)  # {factor_idx: (1,) int64}
    out: dict[str, str] = {}
    for fi, labels in decoded.items():
        cls = int(labels[0])
        out[INTENT_FIELDS[fi]] = pack.label_tokens[fi][cls]
    return out


def build_latent_intent(
    pack: LatentPack, z: np.ndarray, base_intent: Intent,
) -> Intent:
    """Latent-decoded Intent: decodable factors from G, the rest from base.

    `base_intent` supplies ONLY the trivially-constant (non-centroid)
    factors — a task constant, not per-episode privileged information.
    Every factor that genuinely varies in the cut is overwritten by the
    latent decode.
    """
    decoded = decode_latent_factors(pack, z)
    return dc_replace(base_intent, **decoded)


def latent_factor_names(pack: LatentPack) -> tuple[str, ...]:
    """The factor names this pack decodes from vision (has a centroid bank)."""
    return tuple(INTENT_FIELDS[fi] for fi in sorted(pack.centroids.keys()))


def latent_slot_edit(
    pack: LatentPack,
    z: np.ndarray,
    initial_intent: Intent,
    wrong_factor: str,
    failure_predicate: str | None,
) -> Intent:
    """Latent slot-local repair (the "latent edit" sever, Sever B).

    The learned ReviseHead edits EXACTLY the implicated slot of G (the
    single-factor invariant is enforced by `apply_revision`, which writes
    back only that column), and the edited slot is decoded to a token via
    the same nearest-centroid codebook. This is the latent-space analog of
    the discrete `revision.revise_intent` operator — the VLM supplies only
    the factor NAME; the value comes from the learned editor, never a JSON
    field.

    Returns a new `Intent` with exactly `wrong_factor` changed, or the
    unchanged `initial_intent` when the factor has no centroid bank (no
    perceptual region to edit toward — disclosed by the caller).
    """
    if wrong_factor not in INTENT_FIELDS:
        return initial_intent
    factor_idx = INTENT_FIELDS.index(wrong_factor)
    if factor_idx not in pack.centroids:
        return initial_intent
    G = torch.from_numpy(encode_G(pack, z))
    fp_vec = vectorize_failure_packet({
        "revision": {"factor": wrong_factor},
        "failure_packet": {"failure_predicate": failure_predicate or "none"},
    })
    fp_t = torch.from_numpy(fp_vec).unsqueeze(0)
    G_rev = apply_revision(G, factor_idx, fp_t, pack.revise_head)
    decoded_int = decode_slot(G_rev[0, factor_idx].numpy(), pack.centroids[factor_idx])
    new_value = pack.label_tokens[factor_idx][decoded_int]
    return dc_replace(initial_intent, **{wrong_factor: new_value})
