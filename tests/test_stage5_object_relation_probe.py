"""Sim-free tests for the Stage-5 Step-2 object-centric relation probe.

Pins the pure logic of ``babysteps/stage5/object_blobs.py`` and
``scripts/stage5_object_relation_probe.py`` (blob localization, pixel→patch
mapping, window pooling, feature construction, the no-coordinate-leakage
contract, relation-beats-single-object, verdict thresholds) and the
importability of ``scripts/stage5_extract_object_patch_tokens.py`` — all on
synthetic data, no dataset files, no GPU/Vulkan/ManiSkill.
"""
from __future__ import annotations

import importlib
import inspect

import numpy as np


def _probe_mod():
    return importlib.import_module("scripts.stage5_object_relation_probe")


def _blobs():
    return importlib.import_module("babysteps.stage5.object_blobs")


# --------------------------------------------------------------------------- #
# object_blobs: localization + pooling
# --------------------------------------------------------------------------- #


def _synthetic_frame() -> np.ndarray:
    f = np.full((512, 512, 3), 120, np.uint8)
    f[100:140, 80:120] = [220, 30, 30]      # red cubeA, centroid ~ (99.5, 119.5)
    f[360:400, 400:440] = [30, 210, 30]     # green cubeB, centroid ~ (419.5, 379.5)
    return f


def test_detect_cube_centroids_synthetic():
    ob = _blobs()
    c = ob.detect_cube_centroids(_synthetic_frame())
    assert c["cubeA"] is not None and c["cubeB"] is not None
    np.testing.assert_allclose(c["cubeA"], (99.5, 119.5), atol=1.0)
    np.testing.assert_allclose(c["cubeB"], (419.5, 379.5), atol=1.0)
    # cubeA (red) is up-left of cubeB (green): smaller u and smaller v.
    assert c["cubeA"][0] < c["cubeB"][0] and c["cubeA"][1] < c["cubeB"][1]


def test_detect_returns_none_when_absent():
    ob = _blobs()
    out = ob.detect_cube_centroids(np.full((64, 64, 3), 120, np.uint8))
    assert out["cubeA"] is None and out["cubeB"] is None


def test_pixel_to_patch_rc_mapping_and_clamp():
    ob = _blobs()
    # (u, v) = (col, row). Top-left -> (0,0); bottom-right -> (15,15).
    assert ob.pixel_to_patch_rc((0, 0), img_size=512, grid=16) == (0, 0)
    assert ob.pixel_to_patch_rc((511, 511), img_size=512, grid=16) == (15, 15)
    # u maps to col, v maps to row (order matters).
    prow, pcol = ob.pixel_to_patch_rc((480, 32), img_size=512, grid=16)
    assert (prow, pcol) == (1, 15)
    # out-of-range clamps, never raises.
    assert ob.pixel_to_patch_rc((9999, -5), img_size=512, grid=16) == (0, 15)


def test_pool_patch_window_shapes_and_radius():
    ob = _blobs()
    grid, d = 16, 8
    pt = np.arange(grid * grid * d, dtype=np.float32).reshape(grid * grid, d)
    r0 = ob.pool_patch_window(pt, (5, 5), grid=grid, radius=0)
    assert r0.shape == (d,)
    # radius 0 == the single center patch (row-major flat index).
    np.testing.assert_array_equal(r0, pt[ob.patch_rc_to_flat(5, 5, grid=grid)])
    # radius 1 == mean over the 3x3 window.
    r1 = ob.pool_patch_window(pt, (5, 5), grid=grid, radius=1)
    idx = [ob.patch_rc_to_flat(rr, cc, grid=grid)
           for rr in range(4, 7) for cc in range(4, 7)]
    np.testing.assert_allclose(r1, pt[idx].mean(axis=0))
    # corner clamps the window (no wrap / no out-of-bounds).
    rc = ob.pool_patch_window(pt, (0, 0), grid=grid, radius=1)
    assert np.isfinite(rc).all()


def test_object_local_tokens_drops_missing():
    ob = _blobs()
    grid, d = 16, 4
    pt = np.random.RandomState(0).randn(grid * grid, d).astype(np.float32)
    toks = ob.object_local_tokens(pt, {"cubeA": (100.0, 100.0), "cubeB": None},
                                   grid=grid, radius=1)
    assert toks["cubeA"] is not None and toks["cubeA"].shape == (d,)
    assert toks["cubeB"] is None


# --------------------------------------------------------------------------- #
# probe: feature construction + leakage contract
# --------------------------------------------------------------------------- #


def test_build_appearance_features_shapes():
    mod = _probe_mod()
    n, d = 6, 8
    A = np.random.RandomState(1).randn(n, d).astype(np.float32)
    B = np.random.RandomState(2).randn(n, d).astype(np.float32)
    feats = mod.build_appearance_features(A, B)
    assert set(feats) == {
        "A_tok (cubeA local)", "B_tok (cubeB local)",
        "[A_tok;B_tok]", "B_tok-A_tok (HEADLINE)",
    }
    assert feats["A_tok (cubeA local)"].shape == (n, d)
    assert feats["[A_tok;B_tok]"].shape == (n, 2 * d)
    assert feats["B_tok-A_tok (HEADLINE)"].shape == (n, d)
    assert all(v.dtype == np.float32 for v in feats.values())


def test_no_coordinate_leakage_into_headline_features():
    """The appearance rungs must be a pure function of token VALUES.

    Structural guard: build_appearance_features takes ONLY (A_tok, B_tok) — no
    centroid/box/coordinate parameter — and its outputs reproduce exactly from
    the tokens. So no image-plane coordinate can leak into the headline probe.
    The lone coordinate feature is build_uv_upper_bound (dim 2, flagged).
    """
    mod = _probe_mod()
    params = list(inspect.signature(mod.build_appearance_features).parameters)
    assert params == ["A_tok", "B_tok"], params

    n, d = 5, 8
    A = np.random.RandomState(3).randn(n, d).astype(np.float32)
    B = np.random.RandomState(4).randn(n, d).astype(np.float32)
    feats = mod.build_appearance_features(A, B)
    # Exact reconstruction from tokens (no extra columns appended).
    np.testing.assert_array_equal(feats["A_tok (cubeA local)"], A)
    np.testing.assert_array_equal(feats["B_tok-A_tok (HEADLINE)"], (B - A).astype(np.float32))
    # None of the appearance rungs has the +2 width that a uv concat would add.
    for v in feats.values():
        assert v.shape[1] in (d, 2 * d)

    # The uv upper-bound rung is the ONLY coordinate feature, and it is 2-d.
    cA = np.random.RandomState(5).randn(n, 2).astype(np.float32)
    cB = np.random.RandomState(6).randn(n, 2).astype(np.float32)
    uv = mod.build_uv_upper_bound(cA, cB)
    assert uv.shape == (n, 2)
    np.testing.assert_array_equal(uv, (cB - cA).astype(np.float32))


def test_relation_beats_single_object_synthetic():
    """Tokens linear in object position: B_tok-A_tok recovers the label while
    A_tok alone is ~chance.

    This is a WIRING / protocol check (does the probe machinery recover a
    relation that is present?), NOT a validation of the object-centric
    hypothesis on real data. The real guards are (a) the signature constraint +
    width assertion on the appearance rungs, and (b) the random-location control
    over actual DINOv2 tokens in the committed run.
    """
    mod = _probe_mod()
    from babysteps.envs.scene import goal_direction_to_motion

    rng = np.random.default_rng(0)
    n, d = 64, 16
    M = rng.normal(size=(d, 2))                      # linear position embedding
    posA = rng.normal(0, 0.1, size=(n, 2))
    dirs = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], float)
    posB = posA + dirs[np.arange(n) % 4] * 0.2
    A_tok = (posA @ M.T + rng.normal(0, 1e-3, (n, d))).astype(np.float32)
    B_tok = (posB @ M.T + rng.normal(0, 1e-3, (n, d))).astype(np.float32)
    y_str = [goal_direction_to_motion(posB[i] - posA[i]) for i in range(n)]
    classes = sorted(set(y_str))
    y = np.array([classes.index(v) for v in y_str], dtype=np.int64)

    feats = mod.build_appearance_features(A_tok, B_tok)
    rel = mod._direct_lr_probe(feats["B_tok-A_tok (HEADLINE)"], y, seed=0)
    a0 = mod._direct_lr_probe(feats["A_tok (cubeA local)"], y, seed=0)
    assert rel["probe_acc_mean"] > 0.90
    assert a0["probe_acc_mean"] < 0.55
    assert rel["probe_acc_mean"] > a0["probe_acc_mean"] + 0.3


def test_pool_at_centroids_nan_row_for_missing_blob():
    mod = _probe_mod()
    grid, d, n = 16, 4, 3
    grids = np.random.RandomState(0).randn(n, grid * grid, d).astype(np.float32)
    cent = np.array([[100.0, 100.0], [np.nan, np.nan], [200.0, 50.0]], np.float32)
    toks = mod._pool_at_centroids(grids, cent, grid=grid, img_size=512, radius=1)
    assert toks.shape == (n, d)
    assert not np.isnan(toks[0]).any()
    assert np.isnan(toks[1]).all()          # missing blob -> NaN row (dropped later)
    assert not np.isnan(toks[2]).any()


def test_assert_no_coord_leak():
    """Width guard passes for token-width rungs, raises on a uv-concat leak."""
    import pytest

    mod = _probe_mod()
    d, n = 8, 4
    A = np.zeros((n, d), np.float32)
    ok = {
        "tok": A, "concat": np.zeros((n, 2 * d), np.float32),
        "delta": A.copy(),
    }
    mod._assert_no_coord_leak(ok, d)                       # no raise
    leaked = {"tok+uv": np.zeros((n, d + 2), np.float32)}  # 768+2 == 770 pattern
    with pytest.raises(AssertionError):
        mod._assert_no_coord_leak(leaked, d)


def test_extractor_layout_calibration_helpers():
    """The row-major layout self-check helpers are internally consistent."""
    ex = importlib.import_module("scripts.stage5_extract_object_patch_tokens")
    ob = _blobs()
    grid, patch = 16, 14
    img = ex._calibration_image(6, 10, grid=grid, patch=patch)
    assert img.shape == (grid * patch, grid * patch, 3)
    # exactly the (6,10) patch cell is bright; a neighbour cell is not.
    assert (img[6 * patch:7 * patch, 10 * patch:11 * patch] == 255).all()
    assert (img[0:patch, 0:patch] == 120).all()
    # flat index matches the probe-side row-major mapping.
    assert ex._expected_flat(6, 10, grid=grid) == ob.patch_rc_to_flat(6, 10, grid=grid)
    assert ex._expected_flat(6, 10, grid=grid) == 6 * grid + 10


def test_step2_verdict_thresholds():
    mod = _probe_mod()
    assert mod._step2_verdict(0.85).startswith("STRONG")
    assert mod._step2_verdict(0.70).startswith("USEFUL")
    assert mod._step2_verdict(0.50).startswith("WEAK")
    assert mod._step2_verdict(0.42).startswith("NO LIFT")


def test_extractor_imports_sim_free():
    """The GPU extractor must import on the login node with no ManiSkill."""
    import sys

    mod = importlib.import_module("scripts.stage5_extract_object_patch_tokens")
    assert hasattr(mod, "main")
    assert "mani_skill" not in sys.modules, (
        "stage5_extract_object_patch_tokens imported mani_skill at module load; "
        "it reads cached frames and must never touch Vulkan."
    )
