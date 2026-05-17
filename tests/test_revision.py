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


@pytest.mark.parametrize("factor", ["object_motion"])
def test_revise_intent_unhandled_factor_raises(factor: str):
    """Stage-0 supports approach_direction, contact_region, goal_state, and
    constraint_region; everything else raises NotImplementedError."""
    attr = Attribution(
        semantic_failure=True,
        wrong_factor=factor,
        freeze=(),
        revise=(factor,),
    )
    with pytest.raises(NotImplementedError, match=factor):
        revise_intent(_intent(), attr, _scene())


# ---------- contact_substitution (Sub-project B) ----------------------- #


def _pick_intent(contact="minus_x_face") -> Intent:
    return Intent(
        goal_state="cube_lifted_at_target",
        object_motion="lift_up",
        contact_region=contact,
        approach_direction="from_above",
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_grasp",
    )


def _attribution_contact() -> Attribution:
    return Attribution(
        semantic_failure=True,
        wrong_factor="contact_region",
        freeze=tuple(f for f in INTENT_FIELDS if f != "contact_region"),
        revise=("contact_region", "embodiment_mapping"),
    )


def test_contact_substitution_returns_revised_intent_and_revision():
    revised, rev = revise_intent(
        _pick_intent(),
        _attribution_contact(),
        _scene(blocked=("minus_x_face",)),
    )
    assert isinstance(revised, Intent)
    assert isinstance(rev, Revision)


def test_contact_substitution_changes_exactly_one_factor():
    initial = _pick_intent("minus_x_face")
    revised, _ = revise_intent(
        initial, _attribution_contact(),
        _scene(blocked=("minus_x_face",)),
    )
    for f in INTENT_FIELDS:
        if f == "contact_region":
            assert getattr(revised, f) != getattr(initial, f)
        else:
            assert getattr(revised, f) == getattr(initial, f), (
                f"factor {f!r} must be frozen by contact_substitution"
            )


def test_contact_substitution_prefers_orthogonal_face():
    """Stage-0 contact_substitution: minus_x_face → minus_y_face (90°
    around z rotation of the gripper)."""
    initial = _pick_intent("minus_x_face")
    revised, _ = revise_intent(
        initial, _attribution_contact(),
        _scene(blocked=("minus_x_face",)),
    )
    assert revised.contact_region == "minus_y_face"


def test_contact_substitution_falls_back_when_orthogonal_blocked():
    """If both current and orthogonal are blocked, pick any remaining
    cardinal in fallback order."""
    initial = _pick_intent("minus_x_face")
    revised, _ = revise_intent(
        initial, _attribution_contact(),
        _scene(blocked=("minus_x_face", "minus_y_face")),
    )
    assert revised.contact_region in {"plus_x_face", "plus_y_face"}
    assert revised.contact_region != "minus_x_face"
    assert revised.contact_region != "minus_y_face"


def test_contact_substitution_record_fields():
    initial = _pick_intent("minus_x_face")
    _, rev = revise_intent(
        initial, _attribution_contact(),
        _scene(blocked=("minus_x_face",)),
    )
    assert rev.operator == "contact_substitution"
    assert rev.factor == "contact_region"
    assert rev.old_value == "minus_x_face"
    assert rev.new_value == "minus_y_face"
    assert set(rev.frozen_factors) == set(
        f for f in INTENT_FIELDS if f != "contact_region"
    )


def test_contact_substitution_raises_when_all_faces_blocked():
    """No fallback exists if every cardinal face is blocked — Stage-0 is
    honest about this."""
    initial = _pick_intent("minus_x_face")
    all_blocked = ("minus_x_face", "plus_x_face", "minus_y_face", "plus_y_face")
    with pytest.raises(RuntimeError, match="no unblocked contact_region"):
        revise_intent(initial, _attribution_contact(),
                      _scene(blocked=all_blocked))


def test_revise_intent_no_semantic_failure_is_ambiguous():
    """If semantic_failure is False, revise_intent shouldn't be called.
    The function is allowed to raise — Stage 0 makes this defensive."""
    attr = Attribution(
        semantic_failure=False, wrong_factor=None, freeze=INTENT_FIELDS, revise=(),
    )
    with pytest.raises((ValueError, NotImplementedError, KeyError)):
        revise_intent(_intent(), attr, _scene())


# ---------- Sub-project C: goal_refinement -------------------------- #


def test_goal_refinement_happy_path():
    """cube_at_target → cubeA_on_cubeB."""
    from babysteps.failure import Attribution
    from babysteps.revision import revise_intent
    from babysteps.schemas import Intent, SceneState

    intent = Intent(
        goal_state="cube_at_target",
        object_motion="translate_+x",
        contact_region="minus_x_face",
        approach_direction="from_above",
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_pick_and_place",
    )
    scene = SceneState(
        cube_xy=(0.0, 0.0), cube_z=0.02, goal_xy=(0.1, 0.0),
        tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0),
        blocked_sides=(),
    )
    attribution = Attribution(
        semantic_failure=True,
        wrong_factor="goal_state",
        freeze=("object_motion", "contact_region", "approach_direction",
                "constraint_region", "embodiment_mapping"),
        revise=("goal_state",),
    )
    revised, record = revise_intent(intent, attribution, scene)
    assert revised.goal_state == "cubeA_on_cubeB"
    # All other factors carry over unchanged (factor-local invariant).
    assert revised.object_motion == "translate_+x"
    assert revised.contact_region == "minus_x_face"
    assert revised.approach_direction == "from_above"
    assert revised.constraint_region == "none"
    assert revised.embodiment_mapping == "proxy_contact_to_franka_pick_and_place"
    # Revision record shape.
    assert record.operator == "goal_refinement"
    assert record.factor == "goal_state"
    assert record.old_value == "cube_at_target"
    assert record.new_value == "cubeA_on_cubeB"
    assert "goal_state" not in record.frozen_factors


def test_goal_refinement_unknown_source_raises():
    """Stage-0 supports only cube_at_target → cubeA_on_cubeB; other goal_state
    sources must raise NotImplementedError to be honest about coverage."""
    import pytest
    from babysteps.failure import Attribution
    from babysteps.revision import revise_intent
    from babysteps.schemas import Intent, SceneState

    intent = Intent(
        goal_state="cube_lifted_at_target",   # PickCube goal — not C's source
        object_motion="lift_up",
        contact_region="minus_x_face",
        approach_direction="from_above",
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_grasp",
    )
    scene = SceneState(
        cube_xy=(0.0, 0.0), cube_z=0.02, goal_xy=(0.1, 0.0),
        tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0),
        blocked_sides=(),
    )
    attribution = Attribution(
        semantic_failure=True,
        wrong_factor="goal_state",
        freeze=("object_motion", "contact_region", "approach_direction",
                "constraint_region", "embodiment_mapping"),
        revise=("goal_state",),
    )
    with pytest.raises(NotImplementedError) as exc:
        revise_intent(intent, attribution, scene)
    msg = str(exc.value)
    assert "cube_lifted_at_target" in msg


def test_goal_refinement_preserves_frozen_factors():
    """The Revision record must list every factor except goal_state as frozen."""
    from babysteps.failure import Attribution
    from babysteps.revision import revise_intent
    from babysteps.schemas import INTENT_FIELDS, Intent, SceneState

    intent = Intent(
        goal_state="cube_at_target",
        object_motion="translate_-y",
        contact_region="plus_x_face",
        approach_direction="from_plus_y",
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_pick_and_place",
    )
    scene = SceneState(
        cube_xy=(0.0, 0.0), cube_z=0.02, goal_xy=(0.0, -0.1),
        tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0),
        blocked_sides=(),
    )
    attribution = Attribution(
        semantic_failure=True, wrong_factor="goal_state",
        freeze=tuple(f for f in INTENT_FIELDS if f != "goal_state"),
        revise=("goal_state",),
    )
    _, record = revise_intent(intent, attribution, scene)
    expected_frozen = tuple(f for f in INTENT_FIELDS if f != "goal_state")
    assert set(record.frozen_factors) == set(expected_frozen)


# ---------- Sub-project D: constraint_introduction ------------------ #


def test_constraint_introduction_happy_path():
    """(constraint_region=none, contact_region=faucet_base) →
    (faucet_base_static, handle_grip). Two-factor revision."""
    from babysteps.failure import Attribution
    from babysteps.revision import revise_intent
    from babysteps.schemas import Intent, SceneState

    intent = Intent(
        goal_state="faucet_turned",
        object_motion="turn",
        contact_region="faucet_base",
        approach_direction="from_above",
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_turn",
    )
    scene = SceneState(
        cube_xy=(0.1, 0.0), cube_z=0.1, goal_xy=(0.1, 0.0),
        tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0),
        blocked_sides=(),
    )
    attribution = Attribution(
        semantic_failure=True,
        wrong_factor="constraint_region",
        freeze=("goal_state", "object_motion", "approach_direction",
                "embodiment_mapping"),
        revise=("constraint_region", "contact_region"),
    )
    revised, record = revise_intent(intent, attribution, scene)
    # BOTH revised:
    assert revised.constraint_region == "faucet_base_static"
    assert revised.contact_region == "handle_grip"
    # Other factors carry over unchanged:
    assert revised.goal_state == "faucet_turned"
    assert revised.object_motion == "turn"
    assert revised.approach_direction == "from_above"
    assert revised.embodiment_mapping == "proxy_contact_to_franka_turn"
    # Revision record names constraint_region as the primary factor.
    assert record.operator == "constraint_introduction"
    assert record.factor == "constraint_region"
    assert record.old_value == "none"
    assert record.new_value == "faucet_base_static"
    # Both revised factors must NOT appear in frozen_factors.
    assert "constraint_region" not in record.frozen_factors
    assert "contact_region" not in record.frozen_factors


def test_constraint_introduction_unknown_constraint_source_raises():
    """If constraint_region is already set, the operator must raise."""
    import pytest
    from babysteps.failure import Attribution
    from babysteps.revision import revise_intent
    from babysteps.schemas import Intent, SceneState

    intent = Intent(
        goal_state="faucet_turned", object_motion="turn",
        contact_region="faucet_base", approach_direction="from_above",
        constraint_region="faucet_base_static",   # already constrained
        embodiment_mapping="proxy_contact_to_franka_turn",
    )
    scene = SceneState(
        cube_xy=(0.1, 0.0), cube_z=0.1, goal_xy=(0.1, 0.0),
        tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0),
        blocked_sides=(),
    )
    attribution = Attribution(
        semantic_failure=True, wrong_factor="constraint_region",
        freeze=("goal_state", "object_motion", "approach_direction",
                "embodiment_mapping"),
        revise=("constraint_region", "contact_region"),
    )
    with pytest.raises(NotImplementedError) as exc:
        revise_intent(intent, attribution, scene)
    msg = str(exc.value)
    assert "faucet_base_static" in msg or "constraint_region" in msg


def test_constraint_introduction_wrong_contact_source_raises():
    """If contact_region isn't faucet_base, the operator must raise."""
    import pytest
    from babysteps.failure import Attribution
    from babysteps.revision import revise_intent
    from babysteps.schemas import Intent, SceneState

    intent = Intent(
        goal_state="faucet_turned", object_motion="turn",
        contact_region="handle_grip",   # already correct contact
        approach_direction="from_above",
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_turn",
    )
    scene = SceneState(
        cube_xy=(0.1, 0.0), cube_z=0.1, goal_xy=(0.1, 0.0),
        tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0),
        blocked_sides=(),
    )
    attribution = Attribution(
        semantic_failure=True, wrong_factor="constraint_region",
        freeze=("goal_state", "object_motion", "approach_direction",
                "embodiment_mapping"),
        revise=("constraint_region", "contact_region"),
    )
    with pytest.raises(NotImplementedError):
        revise_intent(intent, attribution, scene)


def test_constraint_introduction_frozen_factors_audit():
    """The Revision record's frozen_factors lists exactly the 4 factors
    NOT revised."""
    from babysteps.failure import Attribution
    from babysteps.revision import revise_intent
    from babysteps.schemas import Intent, SceneState

    intent = Intent(
        goal_state="faucet_turned", object_motion="turn",
        contact_region="faucet_base", approach_direction="from_above",
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_turn",
    )
    scene = SceneState(
        cube_xy=(0.1, 0.0), cube_z=0.1, goal_xy=(0.1, 0.0),
        tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0),
        blocked_sides=(),
    )
    attribution = Attribution(
        semantic_failure=True, wrong_factor="constraint_region",
        freeze=("goal_state", "object_motion", "approach_direction",
                "embodiment_mapping"),
        revise=("constraint_region", "contact_region"),
    )
    _, record = revise_intent(intent, attribution, scene)
    expected = {"goal_state", "object_motion", "approach_direction",
                "embodiment_mapping"}
    assert set(record.frozen_factors) == expected
