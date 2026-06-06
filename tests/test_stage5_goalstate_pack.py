"""Sim-free tests for the StackCube goal_state pack pipeline.

Covers (1) the probe's --dump-features helper (correct per-(seed,class) file +
label layout, order-asserted) and (2) the goal_state pack trainer end-to-end on
synthetic separable features: it must train an IntentHead + centroids that ground
goal_state and round-trip through load_latent_pack / VisionIntentExtractor.
CPU/torch only; no GPU/Vulkan, no dataset files.
"""
from __future__ import annotations

import importlib
import json

import numpy as np


def _probe():
    return importlib.import_module("scripts.stage5_goal_state_probe")


def test_dump_goalstate_features_layout(tmp_path):
    probe = _probe()
    near, stack = probe._NEAR_TOKEN, probe._STACK_TOKEN
    seeds = [0, 1, 2]
    d = 6
    # Row order per _collect_clip_multipool: (near, stack) per seed.
    Z = np.arange(2 * len(seeds) * d, dtype=np.float32).reshape(2 * len(seeds), d)
    y_str = [near, stack, near, stack, near, stack]
    Z_by_pool = {"first_last": Z, "final_frame": Z * 2}

    probe._dump_goalstate_features(
        tmp_path, seeds, Z_by_pool, y_str,
        pool="first_last", suffix="dinov2_fl", task="StackCube-v1")

    payload = json.loads((tmp_path / "labels.json").read_text())
    assert payload["factor"] == "goal_state" and payload["pool"] == "first_last"
    labels = payload["labels"]
    # 2 classes x 3 seeds = 6 files + labels.
    assert len(labels) == 6
    assert labels["seed_0000_near"] == near and labels["seed_0000_stack"] == stack
    # The dumped feature for (seed0, stack) is row 1 of the first_last pool.
    got = np.load(tmp_path / "seed_0000_stack_dinov2_fl.npy")
    np.testing.assert_array_equal(got, Z[1])
    got_near = np.load(tmp_path / "seed_0002_near_dinov2_fl.npy")
    np.testing.assert_array_equal(got_near, Z[4])  # row 2*2+0


def test_dump_rejects_wrong_order(tmp_path):
    probe = _probe()
    near, stack = probe._NEAR_TOKEN, probe._STACK_TOKEN
    Z = np.zeros((2, 4), dtype=np.float32)
    # Wrong order (stack before near) must raise, not silently mislabel.
    import pytest
    with pytest.raises(RuntimeError):
        probe._dump_goalstate_features(
            tmp_path, [0], {"first_last": Z}, [stack, near],
            pool="first_last", suffix="x", task="StackCube-v1")


def _make_dump(tmp_path, suffix="dinov2_fl", n_per=30, d=8, noise=0.3, seed=0):
    """Write a synthetic goal_state dump: class near=spike@0, stack=spike@1."""
    rng = np.random.default_rng(seed)
    near = "cube_at_target"
    stack = "cubeA_on_cubeB"
    labels = {}
    for cls_i, (tok, tag) in enumerate(((near, "near"), (stack, "stack"))):
        for k in range(n_per):
            v = np.zeros(d, dtype=np.float32)
            v[cls_i] = 5.0
            v = v + rng.normal(0, noise, d).astype(np.float32)
            stem = f"seed_{k:04d}_{tag}"
            np.save(tmp_path / f"{stem}_{suffix}.npy", v)
            labels[stem] = tok
    (tmp_path / "labels.json").write_text(json.dumps({
        "task": "StackCube-v1", "factor": "goal_state", "pool": "first_last",
        "feature_suffix": suffix, "labels": labels,
    }))
    return near, stack


def test_goalstate_trainer_grounds_goal_state(tmp_path):
    from scripts.stage5_train_goalstate_pack import main as train_main
    from babysteps.envs.stackcube_adapter import StackCubeAdapter
    from babysteps.schemas import INTENT_FIELDS
    from babysteps.stage4.vision_intent import VisionIntentExtractor
    from tests.conftest import FakeEnvRunner

    feats = tmp_path / "feats"
    feats.mkdir()
    near, stack = _make_dump(feats)
    pack_dir = tmp_path / "pack"

    rc = train_main(["--features-dir", str(feats), "--feature-suffix", "dinov2_fl",
                     "--out-dir", str(pack_dir), "--cv"])
    assert rc == 0

    # The pack must ground goal_state and round-trip through the extractor.
    gidx = INTENT_FIELDS.index("goal_state")
    template = StackCubeAdapter().oracle_correct_intent(FakeEnvRunner().reset(0))
    ex = VisionIntentExtractor.from_pack(pack_dir, template)
    assert gidx in ex.centroids
    rng = np.random.default_rng(99)
    d = 8
    for cls_i, tok in ((0, near), (1, stack)):
        v = np.zeros(d, dtype=np.float32)
        v[cls_i] = 5.0
        v = v + rng.normal(0, 0.05, d).astype(np.float32)
        out = ex.decode(v)
        assert out.goal_state == tok, (cls_i, tok, out.goal_state)
        # decode_factor agrees and only goal_state moves off the template.
        assert ex.decode_factor(v, gidx) == tok
