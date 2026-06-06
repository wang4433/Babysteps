"""Sim-free guards for Stage-5 B.2 — the LEARNED residual ReviseHead in the
latent loop.

B.1 (test_stage5_residual_revise_head) proved a residual-conditioned head learns
the corrective direction in the abstract. Here we cover the B.2 INTEGRATION: a
4-way LatentPack + a residual ReviseHead are saved to disk, loaded by
`make_latent_learned_reviser`, and the reviser must decode the goal-relative
residual to the SAME corrected face the hand rule (`direction_to_face`) picks —
for every demo/correct mismatch, in the vision-grounded centroid space. Plus the
residual-head save/load round-trip and the --dump-tuples collection contract.

CPU/torch only; no sim, no GPU, no mani_skill.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import torch

from babysteps.envs.pushcube_adapter import PushCubeAdapter
from babysteps.envs.scene import direction_to_face, face_to_push_unit
from babysteps.schemas import INTENT_FIELDS
from babysteps.stage4.latent_policy import LatentPack, save_latent_pack
from babysteps.stage4.intent_head import IntentHead
from babysteps.stage4.revise_head import (
    FP_VECTOR_DIM, FP_VECTOR_DIM_RESIDUAL, ReviseHead, load_revise_head,
    save_revise_head, train_revise_head_l2, vectorize_failure_packet_residual,
)
from babysteps.stage4.slot_decode import decode_slot
from scripts.stage5_natural_loop_eval import (
    make_latent_learned_reviser, run_natural_episode,
)
from tests.conftest import FakeEnvRunner

_CONTACT_IDX = INTENT_FIELDS.index("contact_region")
_D_SLOT = 32
# Class i -> face token i. The 4 cardinal contact faces (4-way latent space).
_FACES = ("minus_x_face", "plus_x_face", "minus_y_face", "plus_y_face")


def _centroids():
    C = np.zeros((4, _D_SLOT), dtype=np.float32)
    for i in range(4):
        C[i, i] = 3.0
    return {_CONTACT_IDX: {i: C[i] for i in range(4)}}


def _save_4way_pack(out_dir):
    """A synthetic but well-formed 4-way pack (4 separable contact centroids)."""
    centroids = _centroids()
    pack = LatentPack(
        intent_head=IntentHead(z_dim=8, n_factors=len(INTENT_FIELDS),
                               d_slot=_D_SLOT, hidden=16, seed=0),
        revise_head=ReviseHead(d_slot=_D_SLOT, fp_dim=FP_VECTOR_DIM,
                               hidden=16, seed=0),
        centroids=centroids,
        label_tokens={_CONTACT_IDX: _FACES},
        attribution_head=None,
    )
    save_latent_pack(pack, out_dir)
    return centroids[_CONTACT_IDX]


def _train_residual_head(centroids, out_path, *, seed=0):
    """Train a residual head on (centroid[demo], residual->correct) tuples, where
    the residual points along the correct face's push direction."""
    rng = np.random.default_rng(seed)
    g_pre, fp, g_tgt = [], [], []
    rec = {"revision": {"factor": "contact_region"},
           "failure_packet": {"failure_predicate": "direction_error"}}
    for di in range(4):
        for ei in range(4):
            if di == ei:
                continue
            for _ in range(30):
                g_pre.append(centroids[di] + rng.normal(0, 0.2, _D_SLOT).astype(np.float32))
                # residual ~ where the cube still needs to go = push dir of correct face
                res = face_to_push_unit(_FACES[ei]) + rng.normal(0, 0.05, 2)
                fp.append(vectorize_failure_packet_residual(rec, res))
                g_tgt.append(centroids[ei])
    head = ReviseHead(d_slot=_D_SLOT, fp_dim=FP_VECTOR_DIM_RESIDUAL,
                      hidden=64, seed=seed)
    train_revise_head_l2(head, np.stack(g_pre), np.stack(fp), np.stack(g_tgt),
                         n_epochs=500, lr=1e-2, seed=seed)
    save_revise_head(head, out_path)
    return head


def test_latent_learned_reviser_matches_hand_rule(tmp_path):
    """The learned residual head, loaded via make_latent_learned_reviser, decodes
    every demo/correct mismatch to the SAME face the goal-relative hand rule
    picks — in the vision-grounded centroid space, no direction_to_face call."""
    centroids = _save_4way_pack(tmp_path)
    _train_residual_head(centroids, tmp_path / "revise_head_residual.pt")
    reviser = make_latent_learned_reviser(
        tmp_path, tmp_path / "revise_head_residual.pt")

    adapter = PushCubeAdapter()
    base = adapter.oracle_correct_intent(FakeEnvRunner().reset(0))

    n_ok = 0
    n_total = 0
    for demo_face in _FACES:
        for correct_face in _FACES:
            if demo_face == correct_face:
                continue
            n_total += 1
            # Residual points toward the correct face's push direction; cube did
            # not move (disp=0) so residual = goal = push_unit(correct)*0.1.
            goal = tuple(0.1 * face_to_push_unit(correct_face))
            scene = SimpleNamespace(cube_xy=(0.0, 0.0), goal_xy=goal)
            fp = SimpleNamespace(object_displacement_vec=(0.0, 0.0),
                                 failure_predicate="direction_error")
            initial = replace(base, contact_region=demo_face)
            out = reviser(initial, fp, scene, adapter)
            # The hand rule's answer for this residual:
            hand = direction_to_face(np.asarray(goal))
            assert out.contact_region == hand, (demo_face, correct_face,
                                                out.contact_region, hand)
            n_ok += int(out.contact_region == correct_face)
    # The hand rule == correct_face for an axis-aligned residual, so the learned
    # head should hit the correct face on every pair.
    assert n_ok == n_total, f"{n_ok}/{n_total} pairs recovered"


def test_latent_learned_single_factor_invariant(tmp_path):
    """The reviser edits ONLY contact_region; all other factors are preserved."""
    centroids = _save_4way_pack(tmp_path)
    _train_residual_head(centroids, tmp_path / "revise_head_residual.pt")
    reviser = make_latent_learned_reviser(
        tmp_path, tmp_path / "revise_head_residual.pt")
    adapter = PushCubeAdapter()
    base = adapter.oracle_correct_intent(FakeEnvRunner().reset(0))
    initial = replace(base, contact_region="plus_x_face")
    scene = SimpleNamespace(cube_xy=(0.0, 0.0),
                            goal_xy=tuple(0.1 * face_to_push_unit("minus_y_face")))
    fp = SimpleNamespace(object_displacement_vec=(0.0, 0.0),
                         failure_predicate="direction_error")
    out = reviser(initial, fp, scene, adapter)
    for f in INTENT_FIELDS:
        if f == "contact_region":
            continue
        assert getattr(out, f) == getattr(initial, f), f


def test_residual_head_save_load_roundtrip(tmp_path):
    centroids = _centroids()[_CONTACT_IDX]
    head = _train_residual_head(centroids, tmp_path / "h.pt", seed=1)
    loaded = load_revise_head(tmp_path / "h.pt")
    assert loaded.fp_dim == FP_VECTOR_DIM_RESIDUAL
    assert loaded.d_slot == _D_SLOT
    g = torch.randn(3, _D_SLOT)
    fp = torch.randn(3, FP_VECTOR_DIM_RESIDUAL)
    with torch.no_grad():
        a = head(g, fp).numpy()
        b = loaded(g, fp).numpy()
    assert np.allclose(a, b, atol=1e-6)


def test_dump_tuples_wellformed_on_mismatch():
    """run_natural_episode with a tuple_sink collects a well-formed residual-head
    tuple on a failed mismatched episode (demo +x, exec -x)."""
    adapter = PushCubeAdapter()
    runner = FakeEnvRunner()
    sink: list = []
    r = run_natural_episode(
        adapter, runner, demo_seed=400, exec_seed=2,  # +x demo, -x exec
        demo_motion=None, exec_motion=None,
        revisers=["same_intent"], tuple_sink=sink)
    assert r["direction_mismatch"] is True
    assert r["initial_success"] is False
    assert len(sink) == 1
    t = sink[0]
    assert t["demo_face"] == "minus_x_face"      # contact to push +x
    assert t["correct_face"] == "plus_x_face"    # contact to push -x
    assert len(t["residual_xy"]) == 2
    assert t["residual_xy"][0] < -1e-3           # residual points -x (toward goal)
    assert t["failure_predicate"] in ("direction_error", "goal_not_satisfied")


def test_dump_tuples_empty_on_match():
    """No tuple is collected when the initial attempt succeeds (matched dirs)."""
    adapter = PushCubeAdapter()
    runner = FakeEnvRunner()
    sink: list = []
    run_natural_episode(
        adapter, runner, demo_seed=400, exec_seed=8,  # both +x -> success
        demo_motion=None, exec_motion=None,
        revisers=["same_intent"], tuple_sink=sink)
    assert sink == []
