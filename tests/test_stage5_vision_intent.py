"""Sim-free guards for Stage-5 full-vision: VisionIntentExtractor.

The extractor decodes a Stage-0 Intent straight from a frozen-encoder feature
(replacing scripted_demo_to_intent). These cover: (1) it decodes the grounded
factor from a trained pack and keeps the task-constant factors from the template
(single-factor change), (2) the persisted StandardScaler is applied at inference
so a pack trained on standardized features decodes the RAW feature correctly
(the V-JEPA fix), (3) the (seed, motion) cache path round-trips. CPU/torch only.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from babysteps.envs.pushcube_adapter import PushCubeAdapter
from babysteps.schemas import INTENT_FIELDS
from babysteps.stage4.intent_head import IntentHead, train_intent_head_joint
from babysteps.stage4.latent_policy import LatentPack, save_latent_pack
from babysteps.stage4.revise_head import ReviseHead
from babysteps.stage4.slot_decode import build_factor_centroids
from babysteps.stage4.vision_intent import (
    MOTION_TAG, TAG_MOTION, DualViewIntentExtractor, VisionIntentExtractor,
    demo_feature_path,
)
from tests.conftest import FakeEnvRunner

_CONTACT_IDX = INTENT_FIELDS.index("contact_region")
_MOTION_IDX = INTENT_FIELDS.index("object_motion")
_FACES = ("minus_x_face", "plus_x_face", "minus_y_face", "plus_y_face")
_MOTIONS = ("translate_+x", "translate_-x", "translate_+y", "translate_-y")
_D_Z = 8
_D_SLOT = 16


def _class_feature(c: int, rng, noise=0.0):
    v = np.zeros(_D_Z, dtype=np.float32)
    v[c] = 5.0
    if noise:
        v = v + rng.normal(0, noise, _D_Z).astype(np.float32)
    return v


def _build_trained_pack(tmp_path, *, standardize=False):
    """Train a tiny IntentHead so contact_region (factor 2) separates 4 classes,
    save it as a pack (+ scaler.npz if standardize)."""
    rng = np.random.default_rng(0)
    Z, yc = [], []
    for c in range(4):
        for _ in range(40):
            Z.append(_class_feature(c, rng, noise=0.3))
            yc.append(c)
    Z = np.stack(Z).astype(np.float32)
    yc = np.asarray(yc)

    mean = np.zeros(_D_Z, dtype=np.float32)
    scale = np.ones(_D_Z, dtype=np.float32)
    Ztr = Z
    if standardize:
        mean = Z.mean(0).astype(np.float32)
        scale = (Z.std(0) + 1e-6).astype(np.float32)
        Ztr = ((Z - mean) / scale).astype(np.float32)

    head = IntentHead(z_dim=_D_Z, n_factors=len(INTENT_FIELDS),
                      d_slot=_D_SLOT, hidden=32, seed=0)
    train_intent_head_joint(head, Ztr, {_CONTACT_IDX: (yc, 4)},
                            n_epochs=300, lr=1e-2, seed=0)
    head.eval()
    with torch.no_grad():
        G = head(torch.from_numpy(Ztr)).numpy()
    centroids = build_factor_centroids(G, {_CONTACT_IDX: yc})
    pack = LatentPack(
        intent_head=head,
        revise_head=ReviseHead(d_slot=_D_SLOT, hidden=32, seed=0),
        centroids=centroids, label_tokens={_CONTACT_IDX: _FACES},
        attribution_head=None,
    )
    save_latent_pack(pack, tmp_path)
    if standardize:
        np.savez(tmp_path / "scaler.npz", mean=mean, scale=scale)
    return mean, scale


def _template():
    adapter = PushCubeAdapter()
    return adapter.oracle_correct_intent(FakeEnvRunner().reset(0))


def test_decode_grounded_factor_keeps_constants(tmp_path):
    _build_trained_pack(tmp_path, standardize=False)
    template = _template()
    ex = VisionIntentExtractor.from_pack(tmp_path, template)
    rng = np.random.default_rng(7)
    for c, face in enumerate(_FACES):
        feat = _class_feature(c, rng, noise=0.05)
        out = ex.decode(feat)
        assert out.contact_region == face, (c, face, out.contact_region)
        # every NON-grounded factor stays at the template value
        for f in INTENT_FIELDS:
            if f == "contact_region":
                continue
            assert getattr(out, f) == getattr(template, f), f


def test_standardized_pack_decodes_raw_feature(tmp_path):
    """Pack trained on standardized features + persisted scaler -> the extractor
    must apply the scaler so a RAW feature still decodes correctly (V-JEPA fix)."""
    mean, scale = _build_trained_pack(tmp_path, standardize=True)
    assert (tmp_path / "scaler.npz").exists()
    ex = VisionIntentExtractor.from_pack(tmp_path, _template())
    assert np.allclose(ex.mean, mean) and np.allclose(ex.scale, scale)
    rng = np.random.default_rng(3)
    for c, face in enumerate(_FACES):
        out = ex.decode(_class_feature(c, rng, noise=0.05))  # RAW feature
        assert out.contact_region == face, (c, face, out.contact_region)


def test_identity_scaler_when_no_scaler_file(tmp_path):
    _build_trained_pack(tmp_path, standardize=False)
    ex = VisionIntentExtractor.from_pack(tmp_path, _template())
    assert np.allclose(ex.mean, 0.0) and np.allclose(ex.scale, 1.0)


def test_decode_from_cache_path_roundtrip(tmp_path):
    _build_trained_pack(tmp_path, standardize=False)
    ex = VisionIntentExtractor.from_pack(tmp_path, _template())
    feats = tmp_path / "feats"
    feats.mkdir()
    rng = np.random.default_rng(1)
    # class 2 -> minus_y_face; store under (seed=1042, motion=+y) -> tag 'py'
    motion = "translate_+y"
    assert _FACES[2] == "minus_y_face"
    np.save(demo_feature_path(feats, 1042, motion, "dinov2"),
            _class_feature(2, rng, noise=0.02))
    out = ex.decode_from_cache(feats, 1042, motion, "dinov2")
    assert out.contact_region == "minus_y_face"
    # path uses the short tag, not the raw '+y' token
    assert demo_feature_path(feats, 1042, motion, "dinov2").name == "seed_1042_py_dinov2.npy"


def test_motion_tag_roundtrip():
    for m, t in MOTION_TAG.items():
        assert TAG_MOTION[t] == m


def _build_pack_for_factor(pack_dir, factor_idx, tokens, *, standardize=False, seed=0):
    """Train a tiny IntentHead so `factor_idx` separates len(tokens) classes,
    save as a pack at `pack_dir` (generalizes _build_trained_pack to any factor).
    Class c's feature is a one-hot-ish spike at dim c — so a per-view feature's
    class fully determines the decoded token, independent of the other view."""
    pack_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n_cls = len(tokens)
    Z, yc = [], []
    for c in range(n_cls):
        for _ in range(40):
            Z.append(_class_feature(c, rng, noise=0.3))
            yc.append(c)
    Z = np.stack(Z).astype(np.float32)
    yc = np.asarray(yc)

    mean = np.zeros(_D_Z, dtype=np.float32)
    scale = np.ones(_D_Z, dtype=np.float32)
    Ztr = Z
    if standardize:
        mean = Z.mean(0).astype(np.float32)
        scale = (Z.std(0) + 1e-6).astype(np.float32)
        Ztr = ((Z - mean) / scale).astype(np.float32)

    head = IntentHead(z_dim=_D_Z, n_factors=len(INTENT_FIELDS),
                      d_slot=_D_SLOT, hidden=32, seed=0)
    train_intent_head_joint(head, Ztr, {factor_idx: (yc, n_cls)},
                            n_epochs=300, lr=1e-2, seed=0)
    head.eval()
    with torch.no_grad():
        G = head(torch.from_numpy(Ztr)).numpy()
    centroids = build_factor_centroids(G, {factor_idx: yc})
    pack = LatentPack(
        intent_head=head,
        revise_head=ReviseHead(d_slot=_D_SLOT, hidden=32, seed=0),
        centroids=centroids, label_tokens={factor_idx: tuple(tokens)},
        attribution_head=None,
    )
    save_latent_pack(pack, pack_dir)
    if standardize:
        np.savez(pack_dir / "scaler.npz", mean=mean, scale=scale)


def test_decode_factor_single_and_guard(tmp_path):
    """decode_factor returns one grounded factor's token; KeyError if ungrounded."""
    _build_trained_pack(tmp_path, standardize=False)  # grounds contact_region only
    ex = VisionIntentExtractor.from_pack(tmp_path, _template())
    rng = np.random.default_rng(11)
    for c, face in enumerate(_FACES):
        assert ex.decode_factor(_class_feature(c, rng, noise=0.05), _CONTACT_IDX) == face
    # object_motion is NOT grounded by this pack -> explicit error, not silent.
    import pytest
    with pytest.raises(KeyError):
        ex.decode_factor(_class_feature(0, rng), _MOTION_IDX)


def test_dual_view_routes_each_factor_to_its_view(tmp_path):
    """Two per-view packs: global grounds object_motion, contact grounds
    contact_region. The router must take each factor from ITS view, independently."""
    gdir, cdir = tmp_path / "global", tmp_path / "contact"
    _build_pack_for_factor(gdir, _MOTION_IDX, _MOTIONS)
    _build_pack_for_factor(cdir, _CONTACT_IDX, _FACES)
    template = _template()
    ex = DualViewIntentExtractor.from_packs(
        {"global": gdir, "contact": cdir},
        {"object_motion": "global", "contact_region": "contact"},
        template,
    )
    rng = np.random.default_rng(2)
    # Cross every motion class with every face class; each must decode from its
    # own view's feature and not leak from the other.
    for mc, motion in enumerate(_MOTIONS):
        for fc, face in enumerate(_FACES):
            out = ex.decode({
                "global": _class_feature(mc, rng, noise=0.05),
                "contact": _class_feature(fc, rng, noise=0.05),
            })
            assert out.object_motion == motion, (mc, motion, out.object_motion)
            assert out.contact_region == face, (fc, face, out.contact_region)
            # non-routed factors stay at the template.
            for f in INTENT_FIELDS:
                if f in ("object_motion", "contact_region"):
                    continue
                assert getattr(out, f) == getattr(template, f), f


def test_dual_view_validation_errors(tmp_path):
    """__post_init__ guards: a view must ground the factor routed to it, and
    routing can't reference unknown views/factors."""
    import pytest

    gdir, cdir = tmp_path / "global", tmp_path / "contact"
    _build_pack_for_factor(gdir, _MOTION_IDX, _MOTIONS)
    _build_pack_for_factor(cdir, _CONTACT_IDX, _FACES)
    template = _template()
    # contact_region routed to the GLOBAL view, which only grounds object_motion.
    with pytest.raises(KeyError):
        DualViewIntentExtractor.from_packs(
            {"global": gdir, "contact": cdir},
            {"contact_region": "global"}, template)
    # routing references a view that wasn't provided.
    with pytest.raises(KeyError):
        DualViewIntentExtractor.from_packs(
            {"global": gdir}, {"object_motion": "wrist"}, template)


def test_dual_view_decode_from_cache(tmp_path):
    """Per-view feature dirs round-trip through demo_feature_path."""
    gdir, cdir = tmp_path / "global", tmp_path / "contact"
    _build_pack_for_factor(gdir, _MOTION_IDX, _MOTIONS)
    _build_pack_for_factor(cdir, _CONTACT_IDX, _FACES)
    ex = DualViewIntentExtractor.from_packs(
        {"global": gdir, "contact": cdir},
        {"object_motion": "global", "contact_region": "contact"}, _template())
    gfeat, cfeat = tmp_path / "gf", tmp_path / "cf"
    gfeat.mkdir(); cfeat.mkdir()
    rng = np.random.default_rng(4)
    motion = "translate_-y"  # class 3 in _MOTIONS -> token translate_-y
    assert _MOTIONS[3] == "translate_-y" and _FACES[1] == "plus_x_face"
    np.save(demo_feature_path(gfeat, 700, motion, "dinov2"), _class_feature(3, rng, noise=0.02))
    np.save(demo_feature_path(cfeat, 700, motion, "dinov2"), _class_feature(1, rng, noise=0.02))
    out = ex.decode_from_cache(
        {"global": gfeat, "contact": cfeat}, 700, motion, "dinov2")
    assert out.object_motion == "translate_-y"
    assert out.contact_region == "plus_x_face"


def test_run_natural_episode_vision_branch(tmp_path):
    """Integration: run_natural_episode in vision mode decodes the initial intent
    from the cached demo feature (not scripted) and populates the vision fields."""
    from scripts.stage5_natural_loop_eval import run_natural_episode

    _build_trained_pack(tmp_path, standardize=False)
    ex = VisionIntentExtractor.from_pack(tmp_path, _template())
    feats = tmp_path / "feats"
    feats.mkdir()
    rng = np.random.default_rng(5)
    # demo +x -> contact minus_x_face (class 0). Store the class-0 feature so the
    # vision decode is correct for (demo_seed=1000, motion=+x).
    np.save(demo_feature_path(feats, 1000, "translate_+x", "dinov2"),
            _class_feature(0, rng, noise=0.02))

    adapter = PushCubeAdapter()
    runner = FakeEnvRunner()  # exec direction = exec_seed % 4 (seed 2 -> -x)
    row = run_natural_episode(
        adapter, runner, demo_seed=1000, exec_seed=2,
        demo_motion="translate_+x", exec_motion="translate_-x",
        revisers=["same_intent"],
        vision_extractor=ex, vision_features_dir=feats, vision_suffix="dinov2")

    assert row["vision_decoded_contact"] == "minus_x_face"
    assert row["vision_decode_correct"] is True
    assert row["initial_intent_contact"] == "minus_x_face"  # vision, not scripted
    assert row["direction_mismatch"] is True   # demo +x vs exec -x
    assert row["initial_success"] is False     # wrong push direction
