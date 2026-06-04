"""Sim-free tests for the Stage-5 latent-input sever (latent_intent.py).

Builds a tiny synthetic LatentPack (no on-disk artifact, no GPU) and
verifies the decode wiring:

  * decodable factors are taken from the latent slots (nearest-centroid),
  * trivially-constant factors are preserved from the base intent,
  * the decode is deterministic,
  * the honesty boundary holds: a factor with no centroid bank is NEVER
    sourced from the encoder (it can only come from base).
"""
from __future__ import annotations

import numpy as np
import torch

from babysteps.schemas import INTENT_FIELDS, Intent
from babysteps.stage4.intent_head import IntentHead
from babysteps.stage4.revise_head import ReviseHead
from babysteps.stage4.latent_policy import LatentPack
from babysteps.stage5.latent_intent import (
    build_latent_intent, decode_latent_factors, latent_factor_names,
    latent_slot_edit,
)

_Z_DIM = 8
_D_SLOT = 4

# Real schema tokens so the produced Intent passes any field validation.
_OBJECT_MOTION_IDX = INTENT_FIELDS.index("object_motion")
_OBJECT_MOTION_TOKENS = ("translate_+x", "translate_-x")

_BASE = Intent(
    goal_state="cube_at_target",
    object_motion="translate_+x",     # will be overwritten by the decode
    contact_region="minus_x_face",
    approach_direction="from_minus_x",
    constraint_region="none",
    embodiment_mapping="proxy_contact_to_franka_push",
)


def _synthetic_pack(z_a: np.ndarray, z_b: np.ndarray) -> LatentPack:
    """Pack whose object_motion slot decodes z_a->token0, z_b->token1.

    Centroids are the IntentHead's actual slot outputs on z_a / z_b, so
    nearest-centroid on z_a returns class 0 and on z_b returns class 1 —
    exactly how build_factor_centroids works on real data.
    """
    head = IntentHead(z_dim=_Z_DIM, n_factors=len(INTENT_FIELDS),
                      d_slot=_D_SLOT, hidden=8, seed=0)
    head.eval()
    with torch.no_grad():
        g_a = head(torch.from_numpy(z_a.reshape(1, -1))).numpy()[0]
        g_b = head(torch.from_numpy(z_b.reshape(1, -1))).numpy()[0]
    centroids = {
        _OBJECT_MOTION_IDX: {
            0: g_a[_OBJECT_MOTION_IDX].astype(np.float32),
            1: g_b[_OBJECT_MOTION_IDX].astype(np.float32),
        }
    }
    label_tokens = {_OBJECT_MOTION_IDX: _OBJECT_MOTION_TOKENS}
    revise = ReviseHead(d_slot=_D_SLOT, seed=0)
    return LatentPack(
        intent_head=head, revise_head=revise,
        centroids=centroids, label_tokens=label_tokens,
    )


def _two_zs() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    z_a = rng.standard_normal(_Z_DIM).astype(np.float32)
    z_b = rng.standard_normal(_Z_DIM).astype(np.float32)
    return z_a, z_b


def test_latent_factor_names_lists_only_centroid_factors():
    z_a, z_b = _two_zs()
    pack = _synthetic_pack(z_a, z_b)
    assert latent_factor_names(pack) == ("object_motion",)


def test_decode_picks_the_matching_class():
    z_a, z_b = _two_zs()
    pack = _synthetic_pack(z_a, z_b)
    assert decode_latent_factors(pack, z_a) == {"object_motion": "translate_+x"}
    assert decode_latent_factors(pack, z_b) == {"object_motion": "translate_-x"}


def test_build_latent_intent_overwrites_decodable_keeps_base():
    z_a, z_b = _two_zs()
    pack = _synthetic_pack(z_a, z_b)
    # z_b decodes object_motion -> translate_-x, overwriting the base's +x.
    out = build_latent_intent(pack, z_b, _BASE)
    assert out.object_motion == "translate_-x"
    # Every non-decodable factor is preserved bit-for-bit from base.
    for f in INTENT_FIELDS:
        if f == "object_motion":
            continue
        assert getattr(out, f) == getattr(_BASE, f), f


def test_decode_is_deterministic():
    z_a, z_b = _two_zs()
    pack = _synthetic_pack(z_a, z_b)
    first = build_latent_intent(pack, z_a, _BASE)
    second = build_latent_intent(pack, z_a, _BASE)
    assert first == second


def test_constant_factor_never_sourced_from_encoder():
    """Honesty guard: a factor with no centroid bank cannot be decoded.

    Even if the base carries a value, decode_latent_factors must not emit
    that factor (it has no perceptual signal); build_latent_intent can
    only get it from base. We prove the encoder output for that factor is
    irrelevant by changing z and confirming the constant factor is stable.
    """
    z_a, z_b = _two_zs()
    pack = _synthetic_pack(z_a, z_b)
    decoded_a = decode_latent_factors(pack, z_a)
    decoded_b = decode_latent_factors(pack, z_b)
    # Only object_motion is ever in the decode dict.
    assert set(decoded_a) == {"object_motion"}
    assert set(decoded_b) == {"object_motion"}
    # contact_region (no centroid) stays the base value regardless of z.
    assert build_latent_intent(pack, z_a, _BASE).contact_region == "minus_x_face"
    assert build_latent_intent(pack, z_b, _BASE).contact_region == "minus_x_face"


def test_latent_slot_edit_changes_only_the_implicated_factor():
    z_a, z_b = _two_zs()
    pack = _synthetic_pack(z_a, z_b)
    edited = latent_slot_edit(
        pack, z_a, _BASE, "object_motion", failure_predicate="approach_blocked",
    )
    # Exactly one factor (the implicated one) may differ from base.
    changed = tuple(f for f in INTENT_FIELDS
                    if getattr(edited, f) != getattr(_BASE, f))
    assert changed in ((), ("object_motion",))
    # The new value is a valid token from the codebook.
    assert edited.object_motion in _OBJECT_MOTION_TOKENS


def test_latent_slot_edit_noop_when_factor_has_no_centroid():
    z_a, z_b = _two_zs()
    pack = _synthetic_pack(z_a, z_b)
    # contact_region has no centroid bank → no perceptual region to edit.
    edited = latent_slot_edit(
        pack, z_a, _BASE, "contact_region", failure_predicate="grasp_slip",
    )
    assert edited == _BASE


def test_latent_slot_edit_is_deterministic():
    z_a, z_b = _two_zs()
    pack = _synthetic_pack(z_a, z_b)
    a = latent_slot_edit(pack, z_b, _BASE, "object_motion", "approach_blocked")
    b = latent_slot_edit(pack, z_b, _BASE, "object_motion", "approach_blocked")
    assert a == b


def test_run_episode_honors_initial_intent_provider():
    """Sever A in the recovery-gate harness: run_episode uses the provider's
    intent for attempt-1 instead of the scripted/demo-derived one."""
    from dataclasses import replace as dc_replace

    from babysteps.episode import run_episode
    from babysteps.envs.pushcube_adapter import PushCubeAdapter
    from tests.conftest import FakeEnvRunner

    fake = FakeEnvRunner()

    class _StubAdapter(PushCubeAdapter):
        def make_env_runner(self):
            return fake

    captured: dict = {}

    def provider(seed: int, scripted: Intent) -> Intent:
        captured["scripted"] = scripted
        captured["seed"] = seed
        return dc_replace(scripted, approach_direction="from_plus_x")

    adapter = _StubAdapter()
    try:
        rec = run_episode(
            episode_id="t", seed=1, adapter=adapter,
            initial_intent_provider=provider,
        )
    finally:
        adapter.close()

    # The provider was called with (seed, full scripted Intent), and the
    # recorded attempt-1 intent reflects the provider's output.
    assert captured["seed"] == 1
    assert isinstance(captured["scripted"], Intent)
    assert rec.execution["initial_intent"]["approach_direction"] == "from_plus_x"


def test_run_episode_default_ignores_provider_path():
    """Default (no provider) leaves attempt-1 as the scripted intent — guards
    the byte-identical default path."""
    from babysteps.episode import run_episode
    from babysteps.envs.pushcube_adapter import PushCubeAdapter
    from tests.conftest import FakeEnvRunner

    fake = FakeEnvRunner()

    class _StubAdapter(PushCubeAdapter):
        def make_env_runner(self):
            return fake

    adapter = _StubAdapter()
    try:
        rec = run_episode(episode_id="t", seed=1, adapter=adapter)
    finally:
        adapter.close()
    # Scripted PushCube intent always has these constant factors; the point is
    # the episode ran without a provider and produced a normal record.
    assert rec.execution["initial_intent"]["goal_state"] == "cube_at_target"
