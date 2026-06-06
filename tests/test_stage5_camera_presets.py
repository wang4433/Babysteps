"""Sim-free guards for the shared Stage-5 dual-camera presets.

The high-oblique presets must look OVER the gripper (clearly elevated) WITHOUT
being nadir — pure top-down collapses the height that defines stacking. The
pose-tuple math is sim-free; only look_at_pose_list / oblique_camera_configs
touch mani_skill (not exercised here).
"""
from __future__ import annotations

import importlib
import sys

import numpy as np


def _mod():
    return importlib.import_module("babysteps.render.camera_presets")


def test_camera_presets_import_sim_free():
    _mod()
    assert "mani_skill" not in sys.modules, (
        "camera_presets imported mani_skill at module load; the sim import must "
        "stay lazy (inside look_at_pose_list)."
    )


def test_camera_elevation_geometry():
    mod = _mod()
    np.testing.assert_allclose(mod.camera_elevation_deg((0, 0, 1.0), (0, 0, 0)), 90.0)
    np.testing.assert_allclose(
        mod.camera_elevation_deg((1.0, 0.0, 1.0), (0.0, 0.0, 0.0)), 45.0)
    assert mod.camera_elevation_deg((1.0, 0.0, 0.01), (0.0, 0.0, 0.0)) < 1.0


def test_presets_are_high_oblique_not_nadir():
    mod = _mod()
    assert mod.CAMERA_PRESETS["default"] is None
    for name, preset in mod.CAMERA_PRESETS.items():
        if preset is None:
            continue
        eye, target = preset
        elev = mod.camera_elevation_deg(eye, target)
        assert 40.0 < elev < 85.0, f"{name} elevation {elev:.1f} not high-oblique"
        assert eye[2] > target[2], f"{name} eye must be above target"


def test_oblique_camera_configs_default_is_none():
    mod = _mod()
    assert mod.oblique_camera_configs("default") is None
    import pytest
    with pytest.raises(ValueError):
        mod.oblique_camera_configs("does_not_exist")


def test_presets_match_probe_inline_values():
    """The probe carries an inline copy (until it is DRY-refactored to import
    this module); the values MUST match so committed/future runs are comparable."""
    mod = _mod()
    probe = importlib.import_module("scripts.stage5_goal_state_probe")
    for name, preset in mod.CAMERA_PRESETS.items():
        assert probe._CAMERA_PRESETS[name] == preset, name
