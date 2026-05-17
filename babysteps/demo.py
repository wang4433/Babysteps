"""Demo proxy → Intent.

Stage-0 scripted intent extraction. Replaces what DINO/VLM will do in later
stages. The critical invariant is enforced by the function signature alone:
`demo_to_intent` takes only a `DemoEvidence`, never a `SceneState`. This is
the privileged-firewall (goal.md §5: "Keep simulator privileged state out of
the demo-to-intent input path").
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from babysteps.envs.scene import face_to_approach, goal_direction_to_motion
from babysteps.schemas import CONTACT_REGIONS, DemoEvidence, Intent


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


def demo_to_intent(evidence: DemoEvidence) -> Intent:
    """Scripted demo-evidence → structured Intent.

    Reads only `evidence` fields — never any SceneState, never any
    privileged ground truth. The contact_region_label is allowed because it
    is a label *on the demo* (what was shown), not a simulator state read.
    """
    contact_region = evidence.contact_region_label
    if contact_region not in CONTACT_REGIONS:
        raise ValueError(
            f"DemoEvidence.contact_region_label must be one of "
            f"{sorted(CONTACT_REGIONS)}, got {contact_region!r}"
        )
    motion = trajectory_to_motion(evidence.object_trajectory)
    approach = face_to_approach(contact_region)
    return Intent(
        goal_state="cube_at_target",
        object_motion=motion,
        contact_region=contact_region,
        approach_direction=approach,
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_push",
    )
