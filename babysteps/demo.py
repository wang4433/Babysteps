"""Scripted demo-evidence utilities.

After Sub-project A, per-task scripted extractors live in their respective
TaskAdapter (e.g., PushCubeAdapter.scripted_demo_to_intent). What stays here
is the small, task-agnostic helper used by those extractors.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from babysteps.envs.scene import goal_direction_to_motion


def trajectory_to_motion(traj: Iterable[tuple[float, float]]) -> str:
    """Snap a (≥2-point) xy trajectory to one of OBJECT_MOTIONS.

    Uses net displacement (final − initial) along the dominant axis.
    Raises ValueError on trajectories shorter than 2 points.
    """
    pts = list(traj)
    if len(pts) < 2:
        raise ValueError(
            f"trajectory_to_motion needs at least 2 points, got {len(pts)}"
        )
    initial = np.asarray(pts[0], dtype=np.float64)
    final = np.asarray(pts[-1], dtype=np.float64)
    return goal_direction_to_motion(final - initial)
