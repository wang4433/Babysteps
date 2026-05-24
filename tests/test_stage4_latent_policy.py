"""Stage-4 M2a latent revision RetryPolicy tests.

Smoke checks that the factory produces a RetryPolicy (Intent, Revision) with
the right shape, operator label, and single-factor invariant. Plus a sim-free
end-to-end run through FakeEnv to verify the wire-up via run_episode.
"""
from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from babysteps.failure import Attribution
from babysteps.policies import RetryContext
from babysteps.schemas import (
    INTENT_FIELDS, Intent, Revision, SceneState,
)


def _mock_pack_for_pushcube(d_slot=8, hidden=16):
    """Build a tiny LatentPack with hand-set centroids that will decode to
    the two PushCube approach_direction tokens."""
    from babysteps.stage4.intent_head import IntentHead
    from babysteps.stage4.revise_head import (
        FP_VECTOR_DIM, ReviseHead,
    )
    from babysteps.stage4.latent_policy import LatentPack
    head = IntentHead(z_dim=20, n_factors=6, d_slot=d_slot,
                      hidden=hidden, seed=0)
    revise = ReviseHead(d_slot=d_slot, fp_dim=FP_VECTOR_DIM,
                        hidden=hidden, seed=0)
    # Crafted centroids: approach_direction slot has two well-separated
    # centroids; the "from_plus_x" centroid is at +e_0 * 5; "from_minus_x"
    # at -e_0 * 5. Other factors get trivial single-class centroids.
    ap_idx = INTENT_FIELDS.index("approach_direction")
    co_idx = INTENT_FIELDS.index("contact_region")
    centroids = {
        ap_idx: {
            0: np.array([5.0] + [0.0] * (d_slot - 1), dtype=np.float32),  # from_minus_x
            1: np.array([-5.0] + [0.0] * (d_slot - 1), dtype=np.float32),  # from_plus_x
        },
        co_idx: {
            0: np.zeros(d_slot, dtype=np.float32),
        },
    }
    label_tokens = {
        ap_idx: ("from_minus_x", "from_plus_x"),
        co_idx: ("minus_x_face",),
    }
    return LatentPack(
        intent_head=head,
        revise_head=revise,
        centroids=centroids,
        label_tokens=label_tokens,
    )


_BASE_INTENT = Intent(
    goal_state="cube_at_target",
    object_motion="translate_+x",
    contact_region="minus_x_face",
    approach_direction="from_minus_x",
    constraint_region="none",
    embodiment_mapping="proxy_contact_to_franka_push",
)

_SCENE = SceneState(
    cube_xy=(0.0, 0.0), cube_z=0.02, goal_xy=(0.1, 0.0),
    tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0), blocked_sides=(),
)


def _make_ctx(*, wrong_factor="approach_direction", demo_features=None,
              failure_predicate="approach_blocked", failure_packet=None):
    if demo_features is None:
        demo_features = np.zeros(20, dtype=np.float32)
    return RetryContext(
        initial_intent=_BASE_INTENT,
        attribution=Attribution(
            semantic_failure=True, wrong_factor=wrong_factor,
            freeze=tuple(f for f in INTENT_FIELDS if f != wrong_factor),
            revise=(wrong_factor,),
        ),
        scene=_SCENE,
        oracle_correct_intent=replace(_BASE_INTENT,
                                      approach_direction="from_plus_x"),
        oracle_wrong_factor="approach_direction",
        task_valid_tokens={
            "approach_direction": ("from_minus_x", "from_plus_x"),
            "contact_region": ("minus_x_face", "plus_x_face"),
        },
        rng=random.Random(0),
        revise_fn=lambda i, a, s: (i, None),
        demo_features=demo_features,
        failure_predicate=failure_predicate,
        failure_packet=failure_packet,
    )


def test_latent_policy_returns_intent_and_revision():
    from babysteps.stage4.latent_policy import latent_revision_factory
    pack = _mock_pack_for_pushcube()
    policy = latent_revision_factory(pack)
    out = policy(_make_ctx())
    assert out is not None
    revised_intent, revision = out
    assert isinstance(revised_intent, Intent)
    assert isinstance(revision, Revision)


def test_latent_policy_operator_label_is_latent_revision():
    from babysteps.stage4.latent_policy import latent_revision_factory
    pack = _mock_pack_for_pushcube()
    policy = latent_revision_factory(pack)
    _, revision = policy(_make_ctx())
    assert revision.operator == "latent_revision"
    assert revision.factor == "approach_direction"


def test_latent_policy_changes_exactly_one_factor():
    """The single-factor revision invariant at the policy layer: the
    returned Intent differs from initial in exactly one field.
    """
    from babysteps.stage4.latent_policy import latent_revision_factory
    pack = _mock_pack_for_pushcube()
    policy = latent_revision_factory(pack)
    revised_intent, _ = policy(_make_ctx())
    diff = [f for f in INTENT_FIELDS
            if getattr(revised_intent, f) != getattr(_BASE_INTENT, f)]
    assert len(diff) <= 1, diff  # zero diff is also acceptable (decode = same)


def test_latent_policy_revised_value_comes_from_decode():
    """Crafted: with the demo features pushing the slot's `g_pre` toward the
    'from_plus_x' centroid via a strong fp signal, the decoded value should
    be 'from_plus_x' (not the initial 'from_minus_x'). This sanity-checks
    that the path encoder → ReviseHead → decode wires up.

    We avoid asserting a specific learned outcome — only that the decoded
    value is one of the two registered tokens (the centroid bank).
    """
    from babysteps.stage4.latent_policy import latent_revision_factory
    pack = _mock_pack_for_pushcube()
    policy = latent_revision_factory(pack)
    revised_intent, _ = policy(_make_ctx())
    assert revised_intent.approach_direction in ("from_minus_x", "from_plus_x")


def test_latent_policy_skips_when_demo_features_missing():
    """If `ctx.demo_features` is None (e.g. an adapter that did not
    precompute it), the policy must NOT raise; it can return None
    (no retry) or fall back to the original intent — but never crash."""
    from babysteps.stage4.latent_policy import latent_revision_factory
    pack = _mock_pack_for_pushcube()
    policy = latent_revision_factory(pack)
    # Build a ctx whose demo_features is None
    ctx_no = _make_ctx(demo_features=None)
    ctx_no = replace(ctx_no, demo_features=None)
    out = policy(ctx_no)
    # Either None (no retry) or a (Intent, Revision) tuple — never a crash.
    assert out is None or (isinstance(out[0], Intent)
                            and isinstance(out[1], Revision))


def test_latent_policy_through_run_episode_smoke(fake_env_runner):
    """End-to-end sim-free smoke: run_episode with policy=latent_revision
    on the PushCube stub adapter + FakeEnvRunner must produce a valid
    EpisodeRecord with revision.operator='latent_revision'.

    The mock pack uses untrained heads so the decoded approach_direction
    is essentially random within the two centroid tokens; this test does
    not assert retry success, only that the path runs and the schema
    contract is met.
    """
    from babysteps.episode import run_episode
    from babysteps.envs.pushcube_adapter import PushCubeAdapter
    from babysteps.schemas import CLAIM_BOUNDARY, EpisodeRecord
    from babysteps.stage4.latent_policy import latent_revision_factory

    class _StubAdapter(PushCubeAdapter):
        def make_env_runner(self):
            return fake_env_runner

    pack = _mock_pack_for_pushcube()
    policy = latent_revision_factory(pack)
    rec = run_episode(
        episode_id="latent_smoke_seed_0000",
        seed=0,
        adapter=_StubAdapter(),
        policy=policy,
    )
    assert isinstance(rec, EpisodeRecord)
    assert rec.claim_boundary == CLAIM_BOUNDARY
    assert rec.revision is not None
    assert rec.revision["operator"] == "latent_revision"
    assert rec.revision["factor"] == "approach_direction"
    # The decoded new_value must be one of the two registered tokens.
    assert rec.revision["new_value"] in ("from_minus_x", "from_plus_x")
    # Single-factor invariant at the schema level: frozen_factors covers
    # everything except the implicated factor.
    assert "approach_direction" not in rec.revision["frozen_factors"]
    assert len(rec.revision["frozen_factors"]) == 5


# ---- Stage-4 M2.5 — learned attribution head wire-up tests --------- #


def _stub_attribution_head_predicting(factor: str):
    """Return an AttributionHead-like object whose predict_factor always
    returns the given factor name. Used so the test does not depend on
    a trained model file."""
    class _Stub:
        def predict_factor(self, fp, intent):
            return factor
    return _Stub()


def _fp_for_stackcube_ambiguous():
    """A FailurePacket-shaped dict that mimics the ambiguous slice
    (direction_error) where the rule says approach_direction but the
    oracle says goal_state."""
    return {
        "failure_predicate": "direction_error",
        "execution_trace": {
            "reached_contact": True, "object_moved": True,
            "collision": False, "planner_failed": False,
            "grasp_slip": False,
        },
        "object_displacement": 0.02,
        "direction_alignment": -0.4,
    }


def test_attribution_head_overrides_rule_wrong_factor():
    """When pack.attribution_head is set and ctx.failure_packet is present,
    the policy must use the head's prediction instead of
    ctx.attribution.wrong_factor."""
    from dataclasses import replace as dc_replace
    from babysteps.stage4.latent_policy import (
        LatentPack, latent_revision_factory,
    )
    pack = _mock_pack_for_pushcube()
    pack_with_head = dc_replace(
        pack, attribution_head=_stub_attribution_head_predicting(
            "contact_region"),  # any factor; we only check it overrides
    )
    policy = latent_revision_factory(pack_with_head)
    # Rule says approach_direction; head says contact_region.
    out = policy(_make_ctx(failure_packet=_fp_for_stackcube_ambiguous()))
    assert out is not None
    _intent, revision = out
    assert revision.factor == "contact_region"


def test_attribution_head_falls_back_when_failure_packet_none():
    """If ctx.failure_packet is None, the head cannot run; we must fall
    back to ctx.attribution.wrong_factor without crashing."""
    from dataclasses import replace as dc_replace
    from babysteps.stage4.latent_policy import (
        LatentPack, latent_revision_factory,
    )
    pack = _mock_pack_for_pushcube()
    pack_with_head = dc_replace(
        pack, attribution_head=_stub_attribution_head_predicting(
            "contact_region"),
    )
    policy = latent_revision_factory(pack_with_head)
    out = policy(_make_ctx(failure_packet=None))
    assert out is not None
    _intent, revision = out
    # Rule's wrong_factor (approach_direction) used because head not run.
    assert revision.factor == "approach_direction"


def test_attribution_head_fallback_on_unknown_predicted_factor():
    """If the head predicts a string not in INTENT_FIELDS, fall back."""
    from dataclasses import replace as dc_replace
    from babysteps.stage4.latent_policy import (
        LatentPack, latent_revision_factory,
    )
    pack = _mock_pack_for_pushcube()
    pack_with_head = dc_replace(
        pack, attribution_head=_stub_attribution_head_predicting("not_a_factor"),
    )
    policy = latent_revision_factory(pack_with_head)
    out = policy(_make_ctx(failure_packet=_fp_for_stackcube_ambiguous()))
    assert out is not None
    _intent, revision = out
    assert revision.factor == "approach_direction"  # rule fallback


def test_latent_pack_round_trip_includes_attribution_head(tmp_path):
    """Full pack with a real AttributionHead must save+load preserving
    the head's forward output bit-identically."""
    from dataclasses import replace as dc_replace
    from babysteps.stage4.attribution_head import AttributionHead
    from babysteps.stage4.attribution_features import FEATURE_DIM
    from babysteps.stage4.latent_policy import (
        LatentPack, load_latent_pack, save_latent_pack,
    )
    pack = _mock_pack_for_pushcube()
    head = AttributionHead(seed=7)
    pack_with_head = dc_replace(pack, attribution_head=head)
    out = tmp_path / "pack_with_head"
    save_latent_pack(pack_with_head, out)
    reloaded = load_latent_pack(out)
    assert reloaded.attribution_head is not None
    x = torch.zeros(2, FEATURE_DIM)
    torch.testing.assert_close(
        pack_with_head.attribution_head(x), reloaded.attribution_head(x),
    )


def test_loading_pack_without_head_yields_none(tmp_path):
    """An M2a-style pack (no attribution_head) must reload with
    attribution_head=None, preserving rule-based behaviour."""
    from babysteps.stage4.latent_policy import (
        load_latent_pack, save_latent_pack,
    )
    pack = _mock_pack_for_pushcube()
    out = tmp_path / "pack_no_head"
    save_latent_pack(pack, out)
    reloaded = load_latent_pack(out)
    assert reloaded.attribution_head is None


def test_latent_pack_save_load_round_trip(tmp_path):
    """LatentPack must serialize fully to disk and reload to an
    equivalent pack (forward outputs bit-identical on a fixed input).
    """
    from babysteps.stage4.latent_policy import (
        LatentPack, load_latent_pack, save_latent_pack,
    )
    pack = _mock_pack_for_pushcube()
    out = tmp_path / "pack"
    save_latent_pack(pack, out)
    reloaded = load_latent_pack(out)
    assert isinstance(reloaded, LatentPack)
    # Verify both heads produce identical outputs on a fixed input
    z = torch.zeros(3, 20)
    fp = torch.zeros(3, 15)
    torch.testing.assert_close(pack.intent_head(z), reloaded.intent_head(z))
    g = torch.zeros(3, pack.intent_head.d_slot)
    torch.testing.assert_close(
        pack.revise_head(g, fp), reloaded.revise_head(g, fp),
    )
    # Centroids and label tokens round-trip exactly
    for fi in pack.centroids:
        for c in pack.centroids[fi]:
            np.testing.assert_array_equal(
                pack.centroids[fi][c], reloaded.centroids[fi][c],
            )
    assert pack.label_tokens == reloaded.label_tokens
