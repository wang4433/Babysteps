"""Tests for babysteps.demo — scripted demo evidence → Intent.

The most important test here is the privileged-firewall regression:
demo_to_intent must take only a DemoEvidence (NOT a SceneState). If anyone
in the future widens the signature, this test breaks.
"""
from __future__ import annotations

import inspect

import pytest

from babysteps.demo import demo_to_intent, trajectory_to_motion
from babysteps.schemas import DemoEvidence, Intent


def _evidence(traj, face) -> DemoEvidence:
    return DemoEvidence(
        camera="third_person",
        demonstrator_type="proxy_oracle",
        object_trajectory=tuple(tuple(p) for p in traj),
        contact_region_label=face,
        final_state="cube_at_target",
        rgbd_video_path=None,
    )


# ---------- trajectory_to_motion ---------------------------------------- #


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


# ---------- demo_to_intent ---------------------------------------------- #


def test_demo_to_intent_signature_takes_only_demo_evidence():
    """Privileged-firewall regression guard. If anyone adds a SceneState
    parameter to demo_to_intent, this test fails — by design."""
    sig = inspect.signature(demo_to_intent)
    params = list(sig.parameters.values())
    assert len(params) == 1, (
        f"demo_to_intent must take ONLY DemoEvidence (privileged firewall). "
        f"Got params: {[p.name for p in params]}"
    )
    annot = params[0].annotation
    # `from __future__ import annotations` makes this a string at runtime; handle both.
    annot_name = annot if isinstance(annot, str) else getattr(annot, "__name__", str(annot))
    assert annot is DemoEvidence or annot_name == "DemoEvidence", (
        f"demo_to_intent parameter must be annotated as DemoEvidence, got {annot!r}"
    )


def test_demo_to_intent_plus_x_trajectory():
    ev = _evidence([(0.0, 0.0), (0.10, 0.0)], "minus_x_face")
    intent = demo_to_intent(ev)
    assert isinstance(intent, Intent)
    assert intent.goal_state == "cube_at_target"
    assert intent.object_motion == "translate_+x"
    assert intent.contact_region == "minus_x_face"
    assert intent.approach_direction == "from_minus_x"
    assert intent.constraint_region == "none"
    assert intent.embodiment_mapping == "proxy_contact_to_franka_push"


def test_demo_to_intent_minus_y_trajectory():
    ev = _evidence([(0.0, 0.0), (0.0, -0.1)], "plus_y_face")
    intent = demo_to_intent(ev)
    assert intent.object_motion == "translate_-y"
    assert intent.contact_region == "plus_y_face"
    assert intent.approach_direction == "from_plus_y"


def test_demo_to_intent_rejects_unknown_contact_region():
    ev = _evidence([(0.0, 0.0), (0.1, 0.0)], "not_a_face")
    # Intent.__post_init__ should reject this — error bubbles up.
    with pytest.raises(ValueError, match="contact_region"):
        demo_to_intent(ev)
