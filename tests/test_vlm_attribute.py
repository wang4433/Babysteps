"""Sim-free tests for babysteps.stage5.vlm_attribute.

Tests parsers and the mock VLM client. The real InternVL3.5 path is GPU-only
and is exercised by scripts/stage5_p2_vlm_eval.py against a Slurm A100 job.
"""
from __future__ import annotations

import pytest

from babysteps.schemas import INTENT_FIELDS, Intent
from babysteps.stage5.vlm_attribute import (
    INTENT_FACTOR_NAMES,
    TASK_PROMPT_INFO,
    MockVLMClient,
    build_constrained_prompt,
    build_free_form_prompt,
    get_factor_menu,
    parse_constrained_output,
    parse_free_form_output,
)

ALL_TASKS = (
    "PushCube-v1", "PickCube-v1", "StackCube-v1",
    "TurnFaucet-v1", "CrossViewPush-v1",
)


SAMPLE_INTENT = Intent(
    goal_state="cube_at_target",
    object_motion="translate_+x",
    contact_region="minus_x_face",
    approach_direction="from_minus_x",
    constraint_region="none",
    embodiment_mapping="proxy_contact_to_franka_push",
)


def test_constrained_prompt_lists_all_six_factors():
    prompt = build_constrained_prompt(
        task="PushCube-v1",
        initial_intent=SAMPLE_INTENT, failure_predicate="approach_blocked",
    )
    for f in INTENT_FIELDS:
        assert f in prompt, f"factor {f!r} missing from constrained prompt"
    assert "approach_blocked" in prompt
    # The exact intent JSON should appear so the VLM can read it.
    assert "from_minus_x" in prompt


def test_free_form_prompt_lists_all_six_keys():
    prompt = build_free_form_prompt(
        task="PushCube-v1",
        initial_intent=SAMPLE_INTENT, failure_predicate="approach_blocked",
    )
    for f in INTENT_FIELDS:
        assert f in prompt
    assert "JSON" in prompt or "json" in prompt


@pytest.mark.parametrize("task", ALL_TASKS)
def test_constrained_prompt_includes_task_context(task: str):
    prompt = build_constrained_prompt(
        task=task, initial_intent=SAMPLE_INTENT,
        failure_predicate="approach_blocked",
    )
    info = TASK_PROMPT_INFO[task]
    assert info["name"] in prompt
    # Every expected token for every factor with hints must appear in the prompt.
    for factor, valid in info["expected_tokens"].items():
        for tok in valid:
            assert tok in prompt, (
                f"expected token {tok!r} for {factor} missing from {task} prompt"
            )


@pytest.mark.parametrize("task", ALL_TASKS)
def test_free_form_prompt_includes_task_context(task: str):
    prompt = build_free_form_prompt(
        task=task, initial_intent=SAMPLE_INTENT,
        failure_predicate="approach_blocked",
    )
    info = TASK_PROMPT_INFO[task]
    assert info["name"] in prompt
    for factor, valid in info["expected_tokens"].items():
        for tok in valid:
            assert tok in prompt


def test_constrained_prompt_unknown_task_raises():
    with pytest.raises(KeyError):
        build_constrained_prompt(
            task="NoSuchTask-v1", initial_intent=SAMPLE_INTENT,
            failure_predicate="approach_blocked",
        )


def test_stackcube_prompt_surfaces_cubeA_on_cubeB():
    """The StackCube fix: the prompt MUST surface cubeA_on_cubeB so the VLM
    can notice the symbolic mismatch with the intent's cube_at_target."""
    prompt = build_constrained_prompt(
        task="StackCube-v1", initial_intent=SAMPLE_INTENT,
        failure_predicate="goal_not_satisfied",
    )
    assert "cubeA_on_cubeB" in prompt


def test_parse_constrained_clean_factor():
    assert parse_constrained_output("approach_direction") == "approach_direction"


def test_parse_constrained_with_whitespace_and_quotes():
    assert parse_constrained_output('  "approach_direction"  \n') == "approach_direction"


def test_parse_constrained_with_extra_prose():
    """The VLM may add a leading sentence; we extract the first factor mentioned."""
    raw = "The wrong factor is approach_direction because the route is blocked."
    assert parse_constrained_output(raw) == "approach_direction"


def test_parse_constrained_invalid_returns_none():
    assert parse_constrained_output("banana") is None
    assert parse_constrained_output("") is None


def test_parse_free_form_clean_json():
    raw = (
        '{"goal_state":"cube_at_target","object_motion":"translate_+x",'
        '"contact_region":"plus_x_face","approach_direction":"from_plus_x",'
        '"constraint_region":"none","embodiment_mapping":"proxy_contact_to_franka_push"}'
    )
    intent = parse_free_form_output(raw)
    assert intent is not None
    assert intent.approach_direction == "from_plus_x"


def test_parse_free_form_with_code_fence():
    raw = (
        "```json\n"
        '{"goal_state":"cube_at_target","object_motion":"translate_+x",'
        '"contact_region":"plus_x_face","approach_direction":"from_plus_x",'
        '"constraint_region":"none","embodiment_mapping":"proxy_contact_to_franka_push"}\n'
        "```"
    )
    intent = parse_free_form_output(raw)
    assert intent is not None
    assert intent.approach_direction == "from_plus_x"


def test_parse_free_form_missing_key_returns_none():
    raw = '{"goal_state":"cube_at_target","object_motion":"translate_+x"}'
    assert parse_free_form_output(raw) is None


def test_parse_free_form_invalid_token_returns_none():
    raw = (
        '{"goal_state":"cube_at_target","object_motion":"translate_+x",'
        '"contact_region":"BANANA","approach_direction":"from_plus_x",'
        '"constraint_region":"none","embodiment_mapping":"proxy_contact_to_franka_push"}'
    )
    assert parse_free_form_output(raw) is None


def test_mock_vlm_returns_canned_response():
    mock = MockVLMClient(constrained_response="approach_direction",
                         free_form_response='{"goal_state":"cube_at_target",'
                         '"object_motion":"translate_+x",'
                         '"contact_region":"plus_x_face",'
                         '"approach_direction":"from_plus_x",'
                         '"constraint_region":"none",'
                         '"embodiment_mapping":"proxy_contact_to_franka_push"}')
    assert mock.diagnose_constrained(
        task="PushCube-v1",
        image_path="x.png", initial_intent=SAMPLE_INTENT,
        failure_predicate="approach_blocked",
    ) == "approach_direction"
    intent = mock.diagnose_free_form(
        task="PushCube-v1",
        image_path="x.png", initial_intent=SAMPLE_INTENT,
        failure_predicate="approach_blocked",
    )
    assert intent.approach_direction == "from_plus_x"


def test_factor_names_constant_matches_intent_fields():
    """INTENT_FACTOR_NAMES must equal INTENT_FIELDS — the VLM's menu IS the schema."""
    assert tuple(INTENT_FACTOR_NAMES) == INTENT_FIELDS


def test_task_prompt_info_covers_p2_tasks():
    """All five P2 eval tasks must each have a prompt-info entry."""
    for task in ALL_TASKS:
        assert task in TASK_PROMPT_INFO
        info = TASK_PROMPT_INFO[task]
        assert info["name"]
        assert info["success_description"]
        assert info["expected_tokens"]
        assert "goal_state" in info["expected_tokens"]
        assert info["factor_menu"]


def test_factor_menu_six_vs_seven():
    """CrossViewPush is the only task with a 7-factor menu (adds
    direction_grounding). The other four are 6-factor."""
    for task in ("PushCube-v1", "PickCube-v1", "StackCube-v1", "TurnFaucet-v1"):
        menu = get_factor_menu(task)
        assert len(menu) == 6
        assert "direction_grounding" not in menu
    menu = get_factor_menu("CrossViewPush-v1")
    assert len(menu) == 7
    assert "direction_grounding" in menu


def test_crossview_prompt_lists_direction_grounding_in_menu():
    """CrossViewPush C1 menu must include direction_grounding as a choice."""
    prompt = build_constrained_prompt(
        task="CrossViewPush-v1", initial_intent=SAMPLE_INTENT,
        failure_predicate="direction_error",
    )
    # The choice list — not the success description — must include the token.
    # We assert the substring "direction_grounding]" so we catch the menu line,
    # not just the success-description mention.
    assert "direction_grounding" in prompt
    # The expected fix value should also be surfaced.
    assert "observer_frame" in prompt


def test_pushcube_prompt_does_not_offer_direction_grounding():
    """Non-cross-view tasks must NOT include direction_grounding in the menu."""
    prompt = build_constrained_prompt(
        task="PushCube-v1", initial_intent=SAMPLE_INTENT,
        failure_predicate="approach_blocked",
    )
    assert "direction_grounding" not in prompt


def test_parse_constrained_respects_factor_menu():
    """direction_grounding should ONLY parse under the 7-factor menu."""
    assert parse_constrained_output("direction_grounding") is None
    assert parse_constrained_output(
        "direction_grounding",
        factor_menu=get_factor_menu("CrossViewPush-v1"),
    ) == "direction_grounding"


# ---------- multi-image (wrist + third-person) prompts ------------------ #


def test_constrained_prompt_single_view_has_one_image_placeholder():
    """Default (single third-person view) keeps exactly one <image>."""
    prompt = build_constrained_prompt(
        task="PushCube-v1", initial_intent=SAMPLE_INTENT,
        failure_predicate="approach_blocked",
    )
    assert prompt.count("<image>") == 1


def test_constrained_prompt_wrist_view_has_two_labeled_placeholders():
    """wrist_view=True emits two <image> placeholders, labeled third-person
    then wrist — matching the [third_person, wrist] image order passed to
    _chat_multi."""
    prompt = build_constrained_prompt(
        task="PushCube-v1", initial_intent=SAMPLE_INTENT,
        failure_predicate="approach_blocked", wrist_view=True,
    )
    assert prompt.count("<image>") == 2
    assert "third-person" in prompt
    assert "wrist" in prompt
    # Third-person label must precede the wrist label (image order matters).
    assert prompt.index("third-person") < prompt.index("wrist")
    # Still a valid C1 prompt: all factors + the failure observation present.
    for f in INTENT_FIELDS:
        assert f in prompt
    assert "approach_blocked" in prompt


def test_free_form_prompt_wrist_view_has_two_placeholders():
    prompt = build_free_form_prompt(
        task="PushCube-v1", initial_intent=SAMPLE_INTENT,
        failure_predicate="approach_blocked", wrist_view=True,
    )
    assert prompt.count("<image>") == 2
    assert "wrist" in prompt


def test_mock_vlm_accepts_wrist_image_path():
    """The mock client must accept wrist_image_path (uniform call site in the
    eval driver) and still return its canned response."""
    mock = MockVLMClient(constrained_response="contact_region")
    factor = mock.diagnose_constrained(
        task="PushCube-v1", image_path="tp.png",
        initial_intent=SAMPLE_INTENT, failure_predicate="approach_blocked",
        wrist_image_path="wrist.png",
    )
    assert factor == "contact_region"
    intent = mock.diagnose_free_form(
        task="PushCube-v1", image_path="tp.png",
        initial_intent=SAMPLE_INTENT, failure_predicate="approach_blocked",
        wrist_image_path="wrist.png",
    )
    assert intent is not None


def test_parse_free_form_crossview_seven_field_json():
    """7-field JSON is required for CrossViewPush — 6-field must fail."""
    six_field = (
        '{"goal_state":"cube_at_target","object_motion":"translate_+x",'
        '"contact_region":"plus_x_face","approach_direction":"from_plus_x",'
        '"constraint_region":"none","embodiment_mapping":"proxy_contact_to_franka_push"}'
    )
    # Without direction_grounding key, parser must return None under 7-menu.
    out = parse_free_form_output(
        six_field, factor_menu=get_factor_menu("CrossViewPush-v1"),
    )
    assert out is None
    # With direction_grounding key, parser must return a valid Intent.
    seven_field = (
        '{"goal_state":"cube_at_target","object_motion":"translate_+x",'
        '"contact_region":"plus_x_face","approach_direction":"from_plus_x",'
        '"constraint_region":"none","embodiment_mapping":"proxy_contact_to_franka_push",'
        '"direction_grounding":"observer_frame"}'
    )
    out = parse_free_form_output(
        seven_field, factor_menu=get_factor_menu("CrossViewPush-v1"),
    )
    assert out is not None
    assert out.direction_grounding == "observer_frame"
