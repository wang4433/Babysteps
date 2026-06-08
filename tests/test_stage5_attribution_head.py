"""Sim-free tests for the distilled multimodal attribution head (step 4).

Guards: dataset geometry (clean residual rule + hard-negative identity), the
head's mask semantics + single code path, the DistilledAttributor Protocol
conformance + menu restriction, additive AttributionObs back-compat, save/load,
and the HEADLINE proof-of-concept — a residual-only mask cannot separate a
Class-A hard negative from a clean contact_region failure with an identical
residual, but the symbolic context modality can. NO GPU/Vulkan, no gitignored
models/ or datasets/stage5/.
"""
from __future__ import annotations

import numpy as np
import torch

from babysteps.envs.scene import face_to_push_unit
from babysteps.schemas import INTENT_FIELDS, Intent
from babysteps.stage5.attribution_dataset import (
    PUSH_DISTANCE,
    make_contactregion_examples,
    make_dataset,
    make_hardneg_objmotion_examples,
)
from babysteps.stage5.attribution_head import (
    D_RES,
    MASK_PRESETS,
    AttributionHead,
    DistilledAttributor,
    build_residual_feat,
    evaluate_attribution,
    features_from_example,
    load_attribution_head,
    resolve_mask,
    save_attribution_head,
    train_attribution_head,
)
from babysteps.stage5.revision_policy import (
    Attributor,
    AttributionObs,
    OracleAttributor,
    RandomAttributor,
    RevisionDecision,
    candidates_for,
    compile_single_slot_edit,
)
from babysteps.stage5.shared_revision_policy import FACTOR_ORDER


# --------------------------------------------------------------------------- #
# dataset geometry
# --------------------------------------------------------------------------- #

def test_clean_residual_follows_push_rule():
    ex = make_contactregion_examples()
    assert len(ex) == 6  # 3 directions x 2 wrong faces
    for e in ex:
        assert e.true_factor == "contact_region"
        d, w = e.meta["direction"], e.meta["wrong_face"]
        correct = {"+x": "minus_x_face", "+y": "minus_y_face",
                   "-y": "plus_y_face"}[d]
        expect = PUSH_DISTANCE * (face_to_push_unit(correct)
                                  - face_to_push_unit(w))
        assert np.allclose(e.residual_xy, expect, atol=1e-9)


def test_six_distinct_residual_classes():
    ex = make_contactregion_examples()
    res = {tuple(np.round(e.residual_xy, 6)) for e in ex}
    assert len(res) == 6


def test_hardneg_identity_same_residual_different_factor():
    """A Class-A hard negative is byte-identical to a clean example in residual,
    trajectory, AND contacted face — differing ONLY in object_motion."""
    clean = {tuple(np.round(e.residual_xy, 9)): e
             for e in make_contactregion_examples()}
    hard = make_hardneg_objmotion_examples()
    assert len(hard) == 6
    matched = 0
    for h in hard:
        assert h.true_factor == "object_motion"
        key = tuple(np.round(h.residual_xy, 9))
        if key in clean:
            c = clean[key]
            matched += 1
            # identical residual + face + trajectory; only motion differs.
            assert h.initial_intent.contact_region == c.initial_intent.contact_region
            assert h.trajectory_xy == c.trajectory_xy
            assert h.initial_intent.object_motion != c.initial_intent.object_motion
            assert c.true_factor == "contact_region"
    assert matched == 6  # every hard negative shadows a clean example


def test_residual_modality_is_direction_only():
    """build_residual_feat delegates to the deployed unit-normalised e_fail, so
    two residuals on the same ray collapse to the same residual feature."""
    a = build_residual_feat("direction_error", (0.1, -0.1))
    b = build_residual_feat("direction_error", (0.2, -0.2))
    assert a.shape == (D_RES,)
    assert np.allclose(a, b, atol=1e-6)


# --------------------------------------------------------------------------- #
# head mechanics
# --------------------------------------------------------------------------- #

def _toy_feats(b=3):
    ex = make_dataset()[:b]
    feats = [features_from_example(e) for e in ex]
    return {k: torch.tensor(np.stack([f[k] for f in feats]))
            for k in ("res", "traj", "ctx")}


def test_forward_shape():
    head = AttributionHead()
    feats = _toy_feats(4)
    mask = torch.ones(4, 4)
    out = head(feats, mask)
    assert out.shape == (4, len(FACTOR_ORDER))


def test_mask_zeroes_modality_and_changes_output():
    head = AttributionHead()
    feats = _toy_feats(2)
    full = head(feats, torch.ones(2, 4))
    none_ctx = head(feats, torch.tensor([[1., 1., 0., 0.], [1., 1., 0., 0.]]))
    # dropping a modality changes the logits (mask bits + zeroed encoder).
    assert not torch.allclose(full, none_ctx)


def test_masked_modality_gets_zero_gradient():
    """A masked-off modality's encoder receives zero gradient (true isolation,
    not just a zeroed forward contribution)."""
    head = AttributionHead()
    feats = _toy_feats(2)
    mask = torch.tensor([[1., 0., 0., 1.], [1., 0., 0., 1.]])  # traj off
    out = head(feats, mask)
    out.sum().backward()
    assert head.enc_traj.weight.grad is not None
    assert torch.count_nonzero(head.enc_traj.weight.grad) == 0
    assert torch.count_nonzero(head.enc_res.weight.grad) > 0  # res is on


def test_distilled_requires_aligned_factor_order():
    head = AttributionHead(n_factors=len(FACTOR_ORDER))
    try:
        DistilledAttributor(head, factor_order=FACTOR_ORDER[:-1])
        assert False, "expected misalignment error"
    except ValueError:
        pass


def test_ablation_presets_share_one_codepath():
    head = AttributionHead()
    feats = _toy_feats(2)
    for name in ("residual_only", "traj_only", "ctx_only", "multimodal"):
        m = torch.tensor(np.tile(resolve_mask(name), (2, 1)), dtype=torch.float32)
        out = head(feats, m)
        assert out.shape == (2, len(FACTOR_ORDER))


def test_resolve_mask():
    assert resolve_mask("multimodal") == (1, 1, 0, 1)
    assert resolve_mask((1, 0, 0, 1)) == (1, 0, 0, 1)
    for bad in ("nope", (1, 0, 0)):
        try:
            resolve_mask(bad)
            assert False, "expected error"
        except (KeyError, ValueError):
            pass


# --------------------------------------------------------------------------- #
# DistilledAttributor
# --------------------------------------------------------------------------- #

def _obs_from_example(e):
    return AttributionObs(
        task="PokeCube-v1", factor_menu=INTENT_FIELDS,
        failure_predicate=e.predicate, initial_intent=e.initial_intent,
        residual_xy=e.residual_xy, trajectory_xy=e.trajectory_xy, key=0)


def test_distilled_attributor_satisfies_protocol():
    head = AttributionHead()
    d = DistilledAttributor(head)
    assert isinstance(d, Attributor)
    assert d.name == "distilled"
    res = d.attribute(_obs_from_example(make_dataset()[0]))
    assert res.factor in INTENT_FIELDS
    assert res.cost["n_calls"] == 0 and res.latency_s >= 0.0


def test_distilled_argmax_restricted_to_menu():
    head = AttributionHead()
    d = DistilledAttributor(head)
    # a menu without direction_grounding must never produce it
    obs = _obs_from_example(make_dataset()[0])
    for _ in range(20):
        assert d.attribute(obs).factor != "direction_grounding"
    # a 1-factor menu forces that factor
    obs1 = AttributionObs(task="PokeCube-v1", factor_menu=("contact_region",),
                          failure_predicate="direction_error",
                          initial_intent=make_dataset()[0].initial_intent,
                          residual_xy=(0.1, -0.1))
    assert d.attribute(obs1).factor == "contact_region"


def test_save_load_roundtrip(tmp_path):
    ex = make_dataset(n_per_case=8, noise=0.01)
    head = train_attribution_head(ex, epochs=50)
    p = tmp_path / "head.pt"
    save_attribution_head(head, p, default_mask="multimodal")
    head2, cfg = load_attribution_head(p)
    assert cfg["default_mask"] == (1, 1, 0, 1)
    feats = _toy_feats(3)
    m = torch.ones(3, 4)
    assert torch.allclose(head(feats, m), head2(feats, m), atol=1e-6)


# --------------------------------------------------------------------------- #
# additive back-compat
# --------------------------------------------------------------------------- #

def test_attributionobs_additive_backcompat():
    intent = make_dataset()[0].initial_intent
    # old-style construction (no new fields) still works
    obs = AttributionObs(task="PokeCube-v1", factor_menu=INTENT_FIELDS,
                         failure_predicate="direction_error",
                         initial_intent=intent)
    assert obs.residual_xy is None and obs.trajectory_xy == ()
    # oracle/random ignore the new fields, behave identically
    assert OracleAttributor("contact_region").attribute(obs).factor == "contact_region"
    r1 = RandomAttributor(0).attribute(obs).factor
    obs2 = AttributionObs(task="PokeCube-v1", factor_menu=INTENT_FIELDS,
                          failure_predicate="direction_error",
                          initial_intent=intent, residual_xy=(0.1, 0.0))
    assert RandomAttributor(0).attribute(obs2).factor == r1  # key-determined


def test_distilled_decision_is_single_slot():
    """A distilled attribution flows into a single-slot edit (headline invariant
    enforced downstream by the compiler)."""
    head = AttributionHead()
    d = DistilledAttributor(head)
    e = make_dataset()[0]
    obs = _obs_from_example(e)
    factor = d.attribute(obs).factor
    cands = candidates_for("PokeCube-v1", factor) if factor == "contact_region" \
        else candidates_for("PokeCube-v1", factor)
    new_val = next(c for c in cands if c != getattr(e.initial_intent, factor))
    revised = compile_single_slot_edit(
        e.initial_intent, RevisionDecision(factor, new_val))
    changed = [f for f in INTENT_FIELDS
               if getattr(e.initial_intent, f) != getattr(revised, f)]
    assert changed == [factor]


# --------------------------------------------------------------------------- #
# HEADLINE: a residual/positional shortcut is INSUFFICIENT for factor
# attribution; the symbolic intent-context modality is NECESSARY. (NOT a fusion
# claim — ctx alone suffices BY CONSTRUCTION; see the module honesty notes.)
# --------------------------------------------------------------------------- #

def test_matched_pairs_byte_identical_under_noise():
    """The headline cannot be a small-noise artifact: a matched clean/hardneg
    pair has a byte-identical residual AND trajectory even with noise>0 (noise is
    keyed by (direction, wrong_face, replicate), shared across the pair)."""
    ex = make_dataset(n_per_case=4, noise=0.02, seed=0)
    clean = {}
    for e in ex:
        if e.true_factor == "contact_region":
            clean[(e.meta["direction"], e.meta["wrong_face"],
                   tuple(np.round(e.residual_xy, 12)))] = e
    matched = 0
    for h in ex:
        if h.true_factor != "object_motion":
            continue
        key = (h.meta["direction"], h.meta["wrong_face"],
               tuple(np.round(h.residual_xy, 12)))
        assert key in clean, "hard negative residual not byte-identical to a clean one"
        c = clean[key]
        assert h.trajectory_xy == c.trajectory_xy
        assert h.initial_intent.object_motion != c.initial_intent.object_motion
        matched += 1
    assert matched == 6 * 4


def test_residual_shortcut_insufficient_context_necessary():
    train = make_dataset(n_per_case=64, noise=0.01, seed=0)
    test = make_dataset(n_per_case=16, noise=0.01, seed=100)
    head = train_attribution_head(train, modality_dropout=0.5, epochs=600, seed=0)

    res_only = evaluate_attribution(head, test, "residual_only")
    res_traj = evaluate_attribution(head, test, "res_traj")
    res_ctx = evaluate_attribution(head, test, "res_ctx")
    multi = evaluate_attribution(head, test, "multimodal")

    # Residual-only: clean contact_region is solvable (the positional cue) but
    # the hard negatives (byte-identical residual) collapse -> overall ~chance.
    assert res_only["by_kind"]["clean"] > 0.7
    assert res_only["by_kind"]["hardneg_objmotion"] < 0.35
    # Trajectory does NOT help (the matched pair has an identical path).
    assert res_traj["by_kind"]["hardneg_objmotion"] < 0.45
    # Context (inferred object_motion token) is NECESSARY and here SUFFICIENT
    # (ctx alone separates BOTH kinds by construction) -> this is the
    # insufficiency-of-residual result, not a fusion result.
    assert res_ctx["by_kind"]["hardneg_objmotion"] > 0.9
    assert multi["accuracy"] > 0.9
    assert multi["by_kind"]["clean"] > 0.9
    assert multi["by_kind"]["hardneg_objmotion"] > 0.9
    # and any context-bearing mask beats residual-only by a wide margin.
    assert multi["accuracy"] - res_only["accuracy"] > 0.3


def test_mask_presets_cover_design_arms():
    for name in ("residual_only", "traj_only", "res_ctx", "multimodal",
                 "multimodal_gpu"):
        assert name in MASK_PRESETS
