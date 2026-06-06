"""Sim-free tests for the Stage-5 StackCube goal_state separability probe.

Pins the pure logic of ``scripts/stage5_goal_state_probe.py`` — the canonical
goal-config geometry (``build_goal_configs``) and the probe ladder
(``run_probe`` + gate) — on synthetic data, no dataset files and no GPU/Vulkan.
The sim-touching render loop (``_collect_features``) is GPU-only and exercised
by the SLURM job, not here; this test asserts it imports without pulling in
mani_skill at module load.
"""
from __future__ import annotations

import importlib

import numpy as np


def _mod():
    return importlib.import_module("scripts.stage5_goal_state_probe")


def test_imports_sim_free():
    """The GPU probe must import on the login node (lazy sim import)."""
    import sys

    mod = _mod()
    assert hasattr(mod, "main")
    assert "mani_skill" not in sys.modules, (
        "stage5_goal_state_probe imported mani_skill at module load; the sim "
        "import must stay lazy (inside _make_env / _collect_features)."
    )


def test_build_goal_configs_geometry():
    mod = _mod()
    cubeB_p = (0.10, -0.05, 0.02)
    cubeA_q = (1.0, 0.0, 0.0, 0.0)
    cfg = mod.build_goal_configs(cubeB_p, cubeA_q, offset=0.06, direction_idx=0)

    assert set(cfg) == {"stack_on", "place_near"}
    (sx, sy, sz), sq = cfg["stack_on"]
    (nx, ny, nz), nq = cfg["place_near"]

    # Stack: directly above cubeB (same xy), raised by exactly one cube edge.
    assert (sx, sy) == (0.10, -0.05)
    np.testing.assert_allclose(sz, 0.02 + mod._CUBE_TOP_DZ)
    # Near: cubeB resting height, displaced by exactly `offset` (dir 0 = +x).
    np.testing.assert_allclose(nz, 0.02)
    np.testing.assert_allclose((nx, ny), (0.10 + 0.06, -0.05))
    # cubeA quaternion preserved in both configs.
    assert sq == cubeA_q and nq == cubeA_q


def test_build_goal_configs_directions_are_distinct_and_offset():
    mod = _mod()
    cubeB_p = (0.0, 0.0, 0.02)
    q = (1.0, 0.0, 0.0, 0.0)
    near_xy = [
        mod.build_goal_configs(cubeB_p, q, offset=0.06, direction_idx=i)["place_near"][0][:2]
        for i in range(4)
    ]
    # All four cardinal placements distinct.
    assert len({tuple(round(c, 6) for c in xy) for xy in near_xy}) == 4
    # Each is exactly 0.06 from cubeB in the xy plane.
    for xy in near_xy:
        np.testing.assert_allclose(np.hypot(xy[0], xy[1]), 0.06, atol=1e-9)
    # direction_idx wraps mod 4 (so it can take the loop counter directly).
    assert mod.build_goal_configs(cubeB_p, q, direction_idx=4)["place_near"] == \
        mod.build_goal_configs(cubeB_p, q, direction_idx=0)["place_near"]


def test_clip_drop_xy_geometry():
    """The clip-mode drop target: cubeB.xy for stack, cubeB.xy+offset for near."""
    mod = _mod()
    cubeB_xy = (0.10, -0.05)
    # stack-on drops onto the real cubeB (no displacement).
    assert mod.clip_drop_xy(cubeB_xy, "cubeA_on_cubeB") == (0.10, -0.05)
    # place-near is displaced by exactly `offset` along the chosen cardinal.
    near = mod.clip_drop_xy(cubeB_xy, "cube_at_target", direction_idx=0, offset=0.08)
    np.testing.assert_allclose(near, (0.10 + 0.08, -0.05))
    # all four cardinals distinct + each exactly `offset` from cubeB.
    near_pts = [
        mod.clip_drop_xy((0.0, 0.0), "cube_at_target", direction_idx=i, offset=0.08)
        for i in range(4)
    ]
    assert len({tuple(round(c, 6) for c in p) for p in near_pts}) == 4
    for p in near_pts:
        np.testing.assert_allclose(np.hypot(p[0], p[1]), 0.08, atol=1e-9)


def test_clip_pool_frame_indices():
    """The temporal-pooling frame subsets for the clip-pool diagnostic."""
    mod = _mod()
    idx = mod.clip_pool_frame_indices(10)
    assert idx["spatial_mean (all)"] == list(range(10))
    assert idx["final_frame"] == [9]
    assert idx["first_last"] == [0, 9]
    assert idx["last5_mean"] == [5, 6, 7, 8, 9]
    # Single-frame clip: every pooling collapses onto frame 0 (no crash).
    one = mod.clip_pool_frame_indices(1)
    assert one["final_frame"] == [0] and one["first_last"] == [0, 0]
    assert one["last5_mean"] == [0] and one["spatial_mean (all)"] == [0]
    # Short clip: last5 clamps to the available frames.
    assert mod.clip_pool_frame_indices(3)["last5_mean"] == [0, 1, 2]
    import pytest
    with pytest.raises(ValueError):
        mod.clip_pool_frame_indices(0)


def test_run_probe_separates_clean_clusters():
    """On linearly-separable 2-class features the probe clears the 0.90 gate."""
    mod = _mod()
    rng = np.random.default_rng(0)
    n_per, d = 24, 16
    a = rng.normal(-4.0, 0.3, size=(n_per, d))
    b = rng.normal(+4.0, 0.3, size=(n_per, d))
    Z = np.concatenate([a, b], axis=0).astype(np.float32)
    y = np.array([0] * n_per + [1] * n_per, dtype=np.int64)

    res = mod.run_probe(Z, y, factor_idx=mod._GOAL_STATE_IDX, d_slot=8,
                        n_epochs=40, seed=0)
    assert res["dim"] == d
    for col in ("direct", "intent"):
        for k in ("probe_acc_mean", "majority_class_acc", "shuffled_features_acc"):
            assert k in res[col]
    # Direct LR on well-separated clusters is essentially perfect; shuffled ~chance.
    assert res["direct"]["probe_acc_mean"] >= 0.90
    assert res["direct"]["shuffled_features_acc"] <= 0.75
    assert mod._gate(res["direct"]["probe_acc_mean"],
                     res["direct"]["majority_class_acc"],
                     res["direct"]["shuffled_features_acc"]) == "PASS"


def test_run_probe_fails_on_noise():
    """Pure noise features must NOT clear the gate (guards against a leaky probe)."""
    mod = _mod()
    rng = np.random.default_rng(1)
    n_per, d = 24, 16
    Z = rng.normal(0.0, 1.0, size=(2 * n_per, d)).astype(np.float32)
    y = np.array([0] * n_per + [1] * n_per, dtype=np.int64)
    res = mod.run_probe(Z, y, factor_idx=mod._GOAL_STATE_IDX, d_slot=8,
                        n_epochs=40, seed=0)
    assert mod._gate(res["intent"]["probe_acc_mean"],
                     res["intent"]["majority_class_acc"],
                     res["intent"]["shuffled_features_acc"]) == "FAIL"


def test_verdict_messaging():
    mod = _mod()
    assert mod._verdict(0.97, 0.5, 0.5).startswith("PASS")
    assert mod._verdict(0.55, 0.5, 0.5).startswith("FAIL")


def test_camera_elevation_geometry():
    """Elevation angle: 90deg straight-down, 0deg horizontal (pure geometry)."""
    mod = _mod()
    # Straight down (nadir): horiz=0 -> 90deg.
    np.testing.assert_allclose(mod._camera_elevation_deg((0, 0, 1.0), (0, 0, 0)), 90.0)
    # 45deg: equal horizontal and vertical offset.
    np.testing.assert_allclose(
        mod._camera_elevation_deg((1.0, 0.0, 1.0), (0.0, 0.0, 0.0)), 45.0)
    # Horizontal-ish: large horiz, tiny dz -> near 0.
    assert mod._camera_elevation_deg((1.0, 0.0, 0.01), (0.0, 0.0, 0.0)) < 1.0


def test_camera_presets_are_high_oblique_not_nadir():
    """Guard the dual-camera contract: every oblique preset must look OVER the
    gripper (clearly elevated) WITHOUT being nadir (pure top-down destroys the
    height cue that defines stacking — the reviewer tautology trap)."""
    mod = _mod()
    assert mod._CAMERA_PRESETS["default"] is None
    # Default ManiSkill render_camera is a LOW oblique (~15deg) — the presets
    # must be meaningfully higher than that but never straight down.
    for name, preset in mod._CAMERA_PRESETS.items():
        if preset is None:
            continue
        eye, target = preset
        elev = mod._camera_elevation_deg(eye, target)
        assert 40.0 < elev < 85.0, f"{name} elevation {elev:.1f} not high-oblique"
        # Camera is above the table, looking down at the cube region.
        assert eye[2] > target[2], f"{name} eye must be above target"
