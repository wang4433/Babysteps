"""Tests for babysteps.demo — the surviving trajectory_to_motion helper.

The `demo_to_intent` tests have moved into tests/test_pushcube_adapter.py
because the extractor itself moved into PushCubeAdapter.scripted_demo_to_intent
in Sub-project A."""
from __future__ import annotations

import pytest

from babysteps.demo import trajectory_to_motion


def test_trajectory_to_motion_plus_x():
    assert trajectory_to_motion([(0.0, 0.0), (0.05, 0.0), (0.10, 0.0)]) == "translate_+x"


def test_trajectory_to_motion_minus_y():
    assert trajectory_to_motion([(0.0, 0.0), (0.0, -0.1)]) == "translate_-y"


def test_trajectory_to_motion_dominant_axis():
    """Mixed motion snaps on the dominant axis."""
    assert trajectory_to_motion([(0.0, 0.0), (0.2, 0.05)]) == "translate_+x"


def test_trajectory_to_motion_empty_raises():
    with pytest.raises(ValueError, match="at least"):
        trajectory_to_motion([])


def test_trajectory_to_motion_single_point_raises():
    with pytest.raises(ValueError, match="at least"):
        trajectory_to_motion([(0.0, 0.0)])
