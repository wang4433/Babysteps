"""Stage-4 firewall-strict feature extraction.

This module is allowed to read ONLY DemoEvidence-shaped fields:
object_trajectory, contact_region_label, and final_state. See
babysteps/stage4/__init__.py for the firewall rationale. Pulling any
label-side intent field or any privileged SceneState field into this file
would leak the answer into the features and invalidate the recoverability
number; the static firewall test in tests/test_stage4_features.py guards
against exactly that.
"""
from __future__ import annotations

import numpy as np

from babysteps.schemas import CONTACT_REGIONS, GOAL_STATES

_CONTACT_ORDER: tuple[str, ...] = tuple(sorted(CONTACT_REGIONS))
_GOAL_ORDER: tuple[str, ...] = tuple(sorted(GOAL_STATES))

FEATURE_DIM: int = 9 + len(_CONTACT_ORDER) + len(_GOAL_ORDER)


def extract_episode_features(record: dict) -> np.ndarray:
    """Return a deterministic-order feature vector built from demo evidence."""
    demo = record["demo"]
    traj = np.asarray(demo["object_trajectory"], dtype=np.float64)
    if traj.ndim != 2 or traj.shape[1] != 2 or traj.shape[0] < 1:
        raise ValueError(f"object_trajectory must be (T, 2); got {traj.shape}")

    start = traj[0]
    end = traj[-1]
    disp = end - start
    disp_norm = float(np.linalg.norm(disp))
    angle = float(np.arctan2(disp[1], disp[0]))
    path_len = float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))) \
        if traj.shape[0] >= 2 else 0.0

    contact_oh = np.zeros(len(_CONTACT_ORDER), dtype=np.float64)
    contact_oh[_CONTACT_ORDER.index(demo["contact_region_label"])] = 1.0

    goal_oh = np.zeros(len(_GOAL_ORDER), dtype=np.float64)
    goal_oh[_GOAL_ORDER.index(demo["final_state"])] = 1.0

    return np.concatenate([
        start.astype(np.float64),
        end.astype(np.float64),
        disp.astype(np.float64),
        np.array([disp_norm, angle, path_len], dtype=np.float64),
        contact_oh,
        goal_oh,
    ])
