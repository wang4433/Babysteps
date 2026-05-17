"""Tests for babysteps.revision.

Critical property: `revise_intent` changes EXACTLY ONE factor. Every other
factor is byte-identical to the input. The summarizer's non-regression score
audits this on every revised episode; this test makes the property a
construction-time guarantee.
"""
from __future__ import annotations

import pytest

from babysteps.failure import Attribution
from babysteps.revision import revise_intent
from babysteps.schemas import INTENT_FIELDS, Intent, Revision, SceneState


def _intent(approach="from_minus_x") -> Intent:
    return Intent(
        goal_state="cube_at_target",
        object_motion="translate_+x",
        contact_region="minus_x_face",
        approach_direction=approach,
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_push",
    )


def _scene(blocked=("from_minus_x",)) -> SceneState:
    return SceneState(
        cube_xy=(0.0, 0.0),
        cube_z=0.02,
        goal_xy=(0.2, 0.0),
        tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0),
        blocked_sides=tuple(blocked),
    )


def _attribution_approach() -> Attribution:
    return Attribution(
        semantic_failure=True,
        wrong_factor="approach_direction",
        freeze=tuple(f for f in INTENT_FIELDS if f != "approach_direction"),
        revise=("approach_direction",),
    )


# ---------- happy path ------------------------------------------------- #


def test_revise_intent_returns_revised_intent_and_revision():
    revised, rev = revise_intent(_intent(), _attribution_approach(), _scene())
    assert isinstance(revised, Intent)
    assert isinstance(rev, Revision)


def test_revise_intent_changes_exactly_one_factor():
    initial = _intent()
    revised, _ = revise_intent(initial, _attribution_approach(), _scene())
    for f in INTENT_FIELDS:
        if f == "approach_direction":
            assert getattr(revised, f) != getattr(initial, f), (
                "approach_direction must be changed by approach_substitution"
            )
        else:
            assert getattr(revised, f) == getattr(initial, f), (
                f"factor {f!r} must be frozen by approach_substitution, "
                f"but changed from {getattr(initial, f)!r} to {getattr(revised, f)!r}"
            )


def test_revise_intent_picks_unblocked_alternative():
    """The new approach_direction must not be in scene.blocked_sides."""
    revised, _ = revise_intent(_intent(), _attribution_approach(),
                                _scene(blocked=("from_minus_x",)))
    assert revised.approach_direction not in {"from_minus_x"}


def test_revision_record_fields():
    rev_record = revise_intent(_intent(), _attribution_approach(), _scene())[1]
    assert rev_record.operator == "approach_substitution"
    assert rev_record.factor == "approach_direction"
    assert rev_record.old_value == "from_minus_x"
    assert rev_record.new_value != "from_minus_x"
    assert rev_record.new_value not in {"from_minus_x"}  # not blocked
    # frozen_factors enumerates every field except approach_direction.
    assert set(rev_record.frozen_factors) == set(
        f for f in INTENT_FIELDS if f != "approach_direction"
    )


# ---------- edge: every standard alternative blocked -------------------- #


def test_revise_intent_falls_back_to_from_above_when_all_cardinals_blocked():
    blocked_cardinals = ("from_minus_x", "from_plus_x", "from_minus_y", "from_plus_y")
    revised, _ = revise_intent(_intent(), _attribution_approach(), _scene(blocked=blocked_cardinals))
    assert revised.approach_direction == "from_above"


# ---------- unhandled factors raise ------------------------------------ #


@pytest.mark.parametrize("factor", ["contact_region", "goal_state", "object_motion"])
def test_revise_intent_unhandled_factor_raises(factor: str):
    attr = Attribution(
        semantic_failure=True,
        wrong_factor=factor,
        freeze=(),
        revise=(factor,),
    )
    with pytest.raises(NotImplementedError, match=factor):
        revise_intent(_intent(), attr, _scene())


def test_revise_intent_no_semantic_failure_is_ambiguous():
    """If semantic_failure is False, revise_intent shouldn't be called.
    The function is allowed to raise — Stage 0 makes this defensive."""
    attr = Attribution(
        semantic_failure=False, wrong_factor=None, freeze=INTENT_FIELDS, revise=(),
    )
    with pytest.raises((ValueError, NotImplementedError, KeyError)):
        revise_intent(_intent(), attr, _scene())
