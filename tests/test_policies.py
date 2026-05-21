import random
from babysteps.failure import Attribution
from babysteps.policies import RetryContext, one_shot, resample_factor, same_intent_retry
from babysteps.schemas import INTENT_FIELDS, Intent, SceneState

_BASE = Intent(
    goal_state="cube_at_target",
    object_motion="translate_+x",
    contact_region="minus_x_face",
    approach_direction="from_minus_x",
    constraint_region="none",
    embodiment_mapping="proxy_contact_to_franka_push",
)
_TOKS = ("minus_x_face", "plus_x_face", "minus_y_face", "plus_y_face")


def test_resample_excludes_current_value():
    rng = random.Random(0)
    for _ in range(50):
        new = resample_factor(_BASE, "contact_region", _TOKS, rng)
        assert new != _BASE.contact_region
        assert new in _TOKS


def test_resample_single_alternative_is_deterministic():
    rng = random.Random(1)
    new = resample_factor(_BASE, "goal_state", ("cube_at_target", "cubeA_on_cubeB"), rng)
    assert new == "cubeA_on_cubeB"


def test_resample_no_alternative_raises():
    import pytest
    rng = random.Random(2)
    with pytest.raises(ValueError):
        resample_factor(_BASE, "contact_region", ("minus_x_face",), rng)


_SCENE = SceneState(
    cube_xy=(0.0, 0.0), cube_z=0.02, goal_xy=(0.1, 0.0),
    tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0), blocked_sides=(),
)


def _ctx(**kw):
    defaults = dict(
        initial_intent=_BASE,
        attribution=Attribution(True, "contact_region", (), ("contact_region",)),
        scene=_SCENE,
        oracle_correct_intent=replace_intent_contact(_BASE, "plus_x_face"),
        oracle_wrong_factor="contact_region",
        task_valid_tokens={"contact_region": _TOKS},
        rng=random.Random(0),
        revise_fn=lambda i, a, s: (i, None),
    )
    defaults.update(kw)
    return RetryContext(**defaults)


def replace_intent_contact(intent, value):
    from dataclasses import replace
    return replace(intent, contact_region=value)


def test_one_shot_returns_none():
    assert one_shot(_ctx()) is None


def test_same_intent_retry_keeps_intent_unchanged():
    out = same_intent_retry(_ctx())
    assert out is not None
    revised, rev = out
    assert revised == _BASE
    assert rev.operator == "same_intent_retry"
    assert set(rev.frozen_factors) == set(INTENT_FIELDS)


from babysteps.policies import babysteps_selective, oracle_factor_revision


def _real_revise_ctx(**kw):
    # revise_fn delegates to the real shared reviser so selective/oracle
    # produce genuine single-factor edits.
    from babysteps import revision as revision_mod
    return _ctx(revise_fn=revision_mod.revise_intent, **kw)


def test_selective_revises_attributed_factor():
    # contact_failure attribution → contact_region revised, others frozen.
    attr = Attribution(True, "contact_region", tuple(
        f for f in INTENT_FIELDS if f != "contact_region"), ("contact_region",))
    out = babysteps_selective(_real_revise_ctx(attribution=attr))
    assert out is not None
    revised, rev = out
    assert rev.factor == "contact_region"
    assert revised.contact_region != _BASE.contact_region


from babysteps.policies import (
    random_factor_revision, text_feedback_replan, full_replan_analogue,
)


def test_random_factor_revision_changes_exactly_one_editable_factor():
    out = random_factor_revision(_ctx(
        task_valid_tokens={
            "approach_direction": ("from_minus_x", "from_plus_x"),
            "contact_region": _TOKS,
        }))
    assert out is not None
    revised, rev = out
    changed = [f for f in INTENT_FIELDS if getattr(revised, f) != getattr(_BASE, f)]
    assert len(changed) == 1
    assert changed[0] in ("approach_direction", "contact_region")
    assert rev.operator == "random_factor_revision"


def test_full_replan_fixes_implicated_and_perturbs_all_other_editables():
    # approach_blocked: implicated=approach_direction; editables include
    # contact_region too → full_replan changes approach (fix) + contact (extra).
    attr = Attribution(
        True, "approach_direction",
        tuple(f for f in INTENT_FIELDS if f != "approach_direction"),
        ("approach_direction", "contact_region"))
    from babysteps import revision as revision_mod
    out = full_replan_analogue(_ctx(
        attribution=attr,
        oracle_wrong_factor="approach_direction",
        task_valid_tokens={
            "approach_direction": ("from_minus_x", "from_plus_x", "from_plus_y"),
            "contact_region": _TOKS,
        },
        revise_fn=revision_mod.revise_intent))
    assert out is not None
    revised, rev = out
    changed = {f for f in INTENT_FIELDS if getattr(revised, f) != getattr(_BASE, f)}
    assert "approach_direction" in changed   # implicated fixed
    assert "contact_region" in changed       # extra editable perturbed
    assert rev.operator == "full_replan_analogue"


def test_text_feedback_perturbs_only_sibling_editables():
    # approach_blocked revise-set siblings = {contact_region}; full would also
    # touch any other editables, text_feedback touches only siblings.
    attr = Attribution(
        True, "approach_direction",
        tuple(f for f in INTENT_FIELDS if f != "approach_direction"),
        ("approach_direction", "contact_region"))
    from babysteps import revision as revision_mod
    out = text_feedback_replan(_ctx(
        attribution=attr,
        oracle_wrong_factor="approach_direction",
        task_valid_tokens={
            "approach_direction": ("from_minus_x", "from_plus_x"),
            "contact_region": _TOKS,
        },
        revise_fn=revision_mod.revise_intent))
    assert out is not None
    revised, rev = out
    changed = {f for f in INTENT_FIELDS if getattr(revised, f) != getattr(_BASE, f)}
    assert "approach_direction" in changed
    assert "contact_region" in changed
    assert rev.operator == "text_feedback_replan"


def test_oracle_revises_ground_truth_factor():
    # Even if attribution is wrong, oracle uses oracle_wrong_factor.
    wrong_attr = Attribution(True, "approach_direction", (), ("approach_direction",))
    out = oracle_factor_revision(_real_revise_ctx(
        attribution=wrong_attr, oracle_wrong_factor="contact_region"))
    assert out is not None
    revised, rev = out
    assert rev.factor == "contact_region"
