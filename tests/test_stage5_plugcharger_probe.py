"""Sim-free tests for the Stage-5 PlugCharger latent-groundability scout.

Pins the pure logic of ``scripts/stage5_plugcharger_probe.py`` — the
charger-relative label geometry (``quat_yaw`` / ``Rz`` / ``charger_rel_xy`` /
``charger_rel_yaw`` / ``median_split_labels``) and the probe ladder
(``run_probe`` + gate + verdict) — on synthetic data, no dataset files and no
GPU/Vulkan. The sim-touching render loop (``_collect_frames``) is GPU-only and
exercised by the SLURM job; this test asserts the module imports without pulling
in mani_skill at load.
"""
from __future__ import annotations

import importlib

import numpy as np


def _mod():
    return importlib.import_module("scripts.stage5_plugcharger_probe")


def _zrot_quat(theta: float) -> tuple[float, float, float, float]:
    """A pure +z rotation by ``theta`` as a [qw, qx, qy, qz] quaternion."""
    return (float(np.cos(theta / 2)), 0.0, 0.0, float(np.sin(theta / 2)))


def test_imports_sim_free():
    """The GPU probe must import on the login node (lazy sim import)."""
    import sys

    mod = _mod()
    assert hasattr(mod, "main")
    assert "mani_skill" not in sys.modules, (
        "stage5_plugcharger_probe imported mani_skill at module load; the sim "
        "import must stay lazy (inside _make_env / _collect_frames / main)."
    )


def test_quat_yaw_recovers_z_rotation():
    mod = _mod()
    for deg in (-60.0, -30.0, 0.0, 30.0, 60.0):
        theta = np.deg2rad(deg)
        np.testing.assert_allclose(mod.quat_yaw(_zrot_quat(theta)), theta, atol=1e-9)
    # +30 deg has qz > 0; -30 deg has qz < 0 (sign carries the yaw direction).
    assert _zrot_quat(np.deg2rad(30))[3] > 0
    assert _zrot_quat(np.deg2rad(-30))[3] < 0


def test_Rz_and_charger_rel_xy_derotate():
    mod = _mod()
    # receptacle yaw = pi de-rotates a known world offset by 180 deg.
    rx, ry = mod.charger_rel_xy((0.1, 0.2), (0.0, 0.0), np.pi)
    np.testing.assert_allclose((rx, ry), (-0.1, -0.2), atol=1e-9)
    # receptacle yaw = pi/2: Rz(-pi/2) @ (1, 0) = (0, -1).
    rx, ry = mod.charger_rel_xy((1.0, 0.0), (0.0, 0.0), np.pi / 2)
    np.testing.assert_allclose((rx, ry), (0.0, -1.0), atol=1e-9)
    # Rz is a proper rotation: orthonormal, det 1.
    R = mod.Rz(0.7)
    np.testing.assert_allclose(R @ R.T, np.eye(2), atol=1e-9)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-9)


def test_charger_rel_yaw_is_camera_frame_difference():
    mod = _mod()
    # charger world yaw 30 deg, receptacle world yaw 180 deg -> apparent -150 deg.
    rel = mod.charger_rel_yaw(_zrot_quat(np.deg2rad(30)), _zrot_quat(np.pi))
    np.testing.assert_allclose(rel, np.deg2rad(30) - np.pi, atol=1e-9)


def test_median_split_labels_balanced():
    mod = _mod()
    labels = mod.median_split_labels([1.0, 2.0, 3.0, 4.0], lo="cw", hi="ccw")
    assert labels == ["cw", "cw", "ccw", "ccw"]
    # Even count -> exactly balanced; only two classes present.
    rng = np.random.default_rng(0)
    vals = rng.normal(size=40)
    labs = mod.median_split_labels(vals, lo="left", hi="right")
    n_hi = sum(v == "right" for v in labs)
    assert 18 <= n_hi <= 22 and set(labs) == {"left", "right"}


def test_factor_specs_value_functions():
    mod = _mod()
    assert set(mod.FACTOR_SPECS) == {"charger_yaw", "charger_xy", "receptacle_yaw"}
    # raw_pose = [x, y, z, qw, qx, qy, qz]
    charger = np.array([0.1, 0.2, 0.012, *_zrot_quat(np.deg2rad(30))])
    recep = np.array([0.0, 0.0, 0.1, *_zrot_quat(np.pi)])
    # charger_yaw cell = apparent (camera-frame) yaw.
    np.testing.assert_allclose(
        mod.FACTOR_SPECS["charger_yaw"]["value"](charger, recep),
        np.deg2rad(30) - np.pi, atol=1e-9)
    # charger_xy cell = lateral coord in receptacle frame (Rz(-pi)@(0.1,0.2) -> -0.2).
    np.testing.assert_allclose(
        mod.FACTOR_SPECS["charger_xy"]["value"](charger, recep), -0.2, atol=1e-9)
    # receptacle_yaw cell = absolute receptacle yaw (the camera-cancelled control).
    np.testing.assert_allclose(
        mod.FACTOR_SPECS["receptacle_yaw"]["value"](charger, recep), np.pi, atol=1e-9)
    # Each cell maps to a real intent factor index.
    for spec in mod.FACTOR_SPECS.values():
        assert spec["intent"] in mod.INTENT_FIELDS


def test_run_probe_separates_clean_clusters():
    """On linearly-separable 2-class features the probe clears the 0.90 gate."""
    mod = _mod()
    rng = np.random.default_rng(0)
    n_per, d = 24, 16
    a = rng.normal(-4.0, 0.3, size=(n_per, d))
    b = rng.normal(+4.0, 0.3, size=(n_per, d))
    Z = np.concatenate([a, b], axis=0).astype(np.float32)
    y = np.array([0] * n_per + [1] * n_per, dtype=np.int64)

    idx = mod.INTENT_FIELDS.index("object_motion")
    res = mod.run_probe(Z, y, factor_idx=idx, d_slot=8, n_epochs=40, seed=0)
    assert res["dim"] == d
    for col in ("direct", "intent"):
        for k in ("probe_acc_mean", "majority_class_acc", "shuffled_features_acc"):
            assert k in res[col]
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
    idx = mod.INTENT_FIELDS.index("constraint_region")
    res = mod.run_probe(Z, y, factor_idx=idx, d_slot=8, n_epochs=40, seed=0)
    assert mod._gate(res["intent"]["probe_acc_mean"],
                     res["intent"]["majority_class_acc"],
                     res["intent"]["shuffled_features_acc"]) == "FAIL"


def test_verdict_messaging():
    mod = _mod()
    # Candidate factor: PASS vs FAIL.
    assert mod._verdict(0.97, 0.5, 0.5, role="primary candidate").startswith("PASS")
    assert mod._verdict(0.55, 0.5, 0.5, role="primary candidate").startswith("FAIL")
    # Negative control: a FAIL is the REQUIRED outcome; a PASS is flagged suspect.
    assert "as required" in mod._verdict(0.55, 0.5, 0.5, role="negative control")
    assert mod._verdict(0.97, 0.5, 0.5, role="negative control").startswith("UNEXPECTED")
