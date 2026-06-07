"""Sim-free tests for Stage-5 P2 headline-experiment plumbing.

Covers:
  * babysteps.stage5.selectivity.selectivity_metrics — slot-local C1 edit,
    free-replan C2 that flips a correct non-implicated factor, revised=None,
    and both the 6-factor and 7-factor (CrossView) menus.
  * babysteps.stage5.vlm_attribute.parse_free_form_output NEUTRAL normalization
    (case / quotes / whitespace forgiven; out-of-vocab prose → parse-fail).
  * The diagnose_free_form repair path via MockVLMClient (prose first, then
    valid JSON).

No env / GPU / Vulkan import — runs on the login node.
"""
from __future__ import annotations

from dataclasses import replace

from babysteps.schemas import INTENT_FIELDS, Intent
from babysteps.stage5.selectivity import selectivity_metrics
from babysteps.stage5.vlm_attribute import (
    MockVLMClient,
    get_factor_menu,
    parse_free_form_output,
)

# 6-factor and 7-factor menus.
MENU_6 = get_factor_menu("PushCube-v1")
MENU_7 = get_factor_menu("CrossViewPush-v1")

# Ground-truth correct intent (the scene's oracle). The implicated (wrong)
# factor in the initial intent is `contact_region`.
GT = Intent(
    goal_state="cube_at_target",
    object_motion="translate_+x",
    contact_region="plus_x_face",
    approach_direction="from_plus_x",
    constraint_region="none",
    embodiment_mapping="proxy_contact_to_franka_push",
)

# Initial: same as GT except contact_region is wrong (the implicated factor).
INITIAL = replace(GT, contact_region="minus_x_face")

IMPLICATED = "contact_region"


# ---------- selectivity_metrics ----------------------------------------- #


def test_selectivity_slot_local_c1_edit_matching_gt():
    """A slot-local C1 edit fixes ONLY the implicated factor and matches GT →
    preservation 1.0, no unnecessary changes, no harmful changes."""
    revised = replace(INITIAL, contact_region="plus_x_face")  # == GT now
    m = selectivity_metrics(INITIAL, revised, GT, IMPLICATED, MENU_6)
    assert m["preservation"] == 1.0
    assert m["unnecessary_changes_count"] == 0
    assert m["unnecessary_changes_rate"] == 0.0
    assert m["harmful_changes_count"] == 0
    assert m["harmful_changes_rate"] == 0.0


def test_selectivity_free_replan_flips_correct_non_implicated_factor():
    """A C2 free-replan that ALSO flips a non-implicated factor that was
    already correct → unnecessary > 0 AND harmful > 0."""
    revised = replace(
        INITIAL,
        contact_region="plus_x_face",          # fixes the implicated factor
        approach_direction="from_minus_x",     # was correct (from_plus_x) → harmful
    )
    m = selectivity_metrics(INITIAL, revised, GT, IMPLICATED, MENU_6)
    # approach_direction changed but is not implicated → unnecessary.
    assert m["unnecessary_changes_count"] == 1
    assert m["unnecessary_changes_rate"] == 1 / (len(MENU_6) - 1)
    # approach_direction was == GT in initial, now != GT → harmful.
    assert m["harmful_changes_count"] == 1
    assert m["harmful_changes_rate"] == 1 / len(MENU_6)
    # preservation drops: one of the 5 non-implicated factors changed.
    assert m["preservation"] == 4 / 5


def test_selectivity_revised_none_is_fully_preserving():
    """revised=None (C2 parse-fail or C1 revise exception) → nothing was changed,
    so preservation 1.0 and zero change/harm counts. Whether the repair
    *succeeded* is tracked separately (success / parse_failure), so the
    selectivity axis must not double-penalize a non-repair as if it rewrote all."""
    m = selectivity_metrics(INITIAL, None, GT, IMPLICATED, MENU_6)
    assert m["preservation"] == 1.0
    assert m["unnecessary_changes_count"] == 0
    assert m["harmful_changes_count"] == 0
    assert m["harmful_changes_rate"] == 0.0


def test_selectivity_no_op_revision_preserves_but_does_not_fix():
    """revised == initial (no edit): preservation 1.0 (nothing changed) and no
    unnecessary changes; contact_region stays wrong so it is NOT counted harmful
    (it was already wrong in initial, not moved away from a correct value)."""
    m = selectivity_metrics(INITIAL, INITIAL, GT, IMPLICATED, MENU_6)
    assert m["preservation"] == 1.0
    assert m["unnecessary_changes_count"] == 0
    assert m["harmful_changes_count"] == 0


def test_selectivity_crossview_seven_factor_menu():
    """7-factor CrossView menu: implicated = direction_grounding. A slot-local
    fix preserves the other 6 and is non-harmful; an extra flip of a correct
    factor registers as unnecessary + harmful over the 7-factor menu."""
    gt7 = Intent(
        goal_state="cube_at_target",
        object_motion="translate_+x",
        contact_region="plus_x_face",
        approach_direction="from_plus_x",
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_push",
        direction_grounding="observer_frame",
    )
    initial7 = replace(gt7, direction_grounding="actor_frame")  # the bug
    implicated7 = "direction_grounding"

    # Slot-local fix.
    revised_good = replace(initial7, direction_grounding="observer_frame")
    m = selectivity_metrics(initial7, revised_good, gt7, implicated7, MENU_7)
    assert m["preservation"] == 1.0
    assert m["unnecessary_changes_count"] == 0
    assert m["harmful_changes_count"] == 0
    assert m["unnecessary_changes_rate"] == 0.0

    # Over-broad replan: also flips a correct factor.
    revised_bad = replace(
        initial7,
        direction_grounding="observer_frame",
        object_motion="translate_-x",  # was correct → harmful + unnecessary
    )
    m2 = selectivity_metrics(initial7, revised_bad, gt7, implicated7, MENU_7)
    assert m2["unnecessary_changes_count"] == 1
    assert m2["unnecessary_changes_rate"] == 1 / (len(MENU_7) - 1)
    assert m2["harmful_changes_count"] == 1
    assert m2["harmful_changes_rate"] == 1 / len(MENU_7)
    assert m2["preservation"] == 5 / 6  # 6 non-implicated; 1 changed


def test_selectivity_keys_present_for_both_branches():
    """Both the revised-None and revised-present branches return the same key
    set (so the eval aggregator never KeyErrors)."""
    keys = {
        "preservation",
        "unnecessary_changes_count", "unnecessary_changes_rate",
        "harmful_changes_count", "harmful_changes_rate",
        "edit_cardinality",
    }
    a = selectivity_metrics(INITIAL, None, GT, IMPLICATED, MENU_6)
    b = selectivity_metrics(INITIAL, GT, GT, IMPLICATED, MENU_6)
    assert set(a) == keys
    assert set(b) == keys


# ---------- parse_free_form_output neutral normalization ---------------- #


def _json6(contact="plus_x_face", approach="from_plus_x"):
    return (
        '{"goal_state":"cube_at_target","object_motion":"translate_+x",'
        f'"contact_region":"{contact}","approach_direction":"{approach}",'
        '"constraint_region":"none",'
        '"embodiment_mapping":"proxy_contact_to_franka_push"}'
    )


def test_parse_free_form_correct_cased_tokens():
    intent = parse_free_form_output(_json6())
    assert intent is not None
    assert intent.contact_region == "plus_x_face"


def test_parse_free_form_mixed_case_normalizes():
    raw = _json6(contact="PLUS_X_FACE")
    intent = parse_free_form_output(raw)
    assert intent is not None
    # Canonical lowercase token is recovered.
    assert intent.contact_region == "plus_x_face"


def test_parse_free_form_whitespace_and_inner_quotes_normalize():
    # Inner whitespace around a value (JSON string contains leading/trailing
    # spaces) must normalize to the canonical token.
    raw = _json6(approach="  from_plus_x  ")
    intent = parse_free_form_output(raw)
    assert intent is not None
    assert intent.approach_direction == "from_plus_x"


def test_parse_free_form_out_of_vocab_prose_fails():
    raw = _json6(contact="the left side of the cube near the gripper")
    assert parse_free_form_output(raw) is None


def test_parse_free_form_no_synonym_invention():
    """Neutral normalization must NOT invent synonyms: a plausible paraphrase
    that is not an exact (case-insensitive) token must parse-fail."""
    raw = _json6(approach="approach from the positive x direction")
    assert parse_free_form_output(raw) is None


# ---------- diagnose_free_form repair path ------------------------------ #


def test_diagnose_free_form_repair_prose_then_json():
    """MockVLMClient emits un-parseable prose first, valid JSON on repair →
    the ONE format-repair retry recovers a valid Intent and the verbose API
    returns the repair raw text."""
    mock = MockVLMClient(
        free_form_response="I think you should approach from the other side.",
        free_form_repair_response=_json6(contact="plus_x_face"),
    )
    intent, raw = mock.diagnose_free_form_verbose(
        task="PushCube-v1", image_path="x.png", initial_intent=INITIAL,
        failure_predicate="approach_blocked",
    )
    assert intent is not None
    assert intent.contact_region == "plus_x_face"
    # Raw text persisted is the repair reply (the one that parsed).
    assert raw == _json6(contact="plus_x_face")
    # The thin wrapper returns the same Intent.
    assert mock.diagnose_free_form(
        task="PushCube-v1", image_path="x.png", initial_intent=INITIAL,
        failure_predicate="approach_blocked",
    ) is not None


def test_diagnose_free_form_repair_still_fails_returns_none():
    """If BOTH the first reply and the repair reply are un-parseable prose,
    diagnose_free_form_verbose returns (None, repair_raw)."""
    mock = MockVLMClient(
        free_form_response="prose, no json",
        free_form_repair_response="still prose, still no json",
    )
    intent, raw = mock.diagnose_free_form_verbose(
        task="PushCube-v1", image_path="x.png", initial_intent=INITIAL,
        failure_predicate="approach_blocked",
    )
    assert intent is None
    assert raw == "still prose, still no json"


def test_diagnose_free_form_first_try_parses_no_repair_needed():
    """When the first reply parses, the raw text is the FIRST reply (no repair
    consumed)."""
    mock = MockVLMClient(
        free_form_response=_json6(),
        free_form_repair_response=_json6(contact="minus_x_face"),
    )
    intent, raw = mock.diagnose_free_form_verbose(
        task="PushCube-v1", image_path="x.png", initial_intent=INITIAL,
        failure_predicate="approach_blocked",
    )
    assert intent is not None
    assert raw == _json6()  # first reply, repair untouched
    assert intent.contact_region == "plus_x_face"


def test_default_factor_menu_is_six():
    assert tuple(INTENT_FIELDS) == MENU_6


# ---------- edit_cardinality (Stage-5 unified main table) ---------------- #


def test_edit_cardinality_counts_changed_factors():
    # Slot-local single edit → cardinality 1.
    one = selectivity_metrics(
        INITIAL, replace(INITIAL, contact_region="plus_x_face"),
        GT, IMPLICATED, MENU_6)
    assert one["edit_cardinality"] == 1
    # Two-factor rewrite → cardinality 2.
    two = selectivity_metrics(
        INITIAL, replace(INITIAL, contact_region="plus_x_face",
                         approach_direction="from_minus_x"),
        GT, IMPLICATED, MENU_6)
    assert two["edit_cardinality"] == 2
    # No-op and parse-fail → cardinality 0.
    assert selectivity_metrics(
        INITIAL, INITIAL, GT, IMPLICATED, MENU_6)["edit_cardinality"] == 0
    assert selectivity_metrics(
        INITIAL, None, GT, IMPLICATED, MENU_6)["edit_cardinality"] == 0
