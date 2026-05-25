"""Sim-free tests for babysteps.stage5.vlm_attribute.

Tests parsers and the mock VLM client. The real InternVL3.5 path is GPU-only
and is exercised by scripts/stage5_p2_vlm_eval.py against a Slurm A100 job.
"""
from __future__ import annotations

import pytest

from babysteps.schemas import INTENT_FIELDS, Intent
from babysteps.stage5.vlm_attribute import (
    INTENT_FACTOR_NAMES,
    MockVLMClient,
    build_constrained_prompt,
    build_free_form_prompt,
    parse_constrained_output,
    parse_free_form_output,
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
        initial_intent=SAMPLE_INTENT, failure_predicate="approach_blocked",
    )
    for f in INTENT_FIELDS:
        assert f in prompt, f"factor {f!r} missing from constrained prompt"
    assert "approach_blocked" in prompt
    # The exact intent JSON should appear so the VLM can read it.
    assert "from_minus_x" in prompt


def test_free_form_prompt_lists_all_six_keys():
    prompt = build_free_form_prompt(
        initial_intent=SAMPLE_INTENT, failure_predicate="approach_blocked",
    )
    for f in INTENT_FIELDS:
        assert f in prompt
    assert "JSON" in prompt or "json" in prompt


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
        image_path="x.png", initial_intent=SAMPLE_INTENT,
        failure_predicate="approach_blocked",
    ) == "approach_direction"
    intent = mock.diagnose_free_form(
        image_path="x.png", initial_intent=SAMPLE_INTENT,
        failure_predicate="approach_blocked",
    )
    assert intent.approach_direction == "from_plus_x"


def test_factor_names_constant_matches_intent_fields():
    """INTENT_FACTOR_NAMES must equal INTENT_FIELDS — the VLM's menu IS the schema."""
    assert tuple(INTENT_FACTOR_NAMES) == INTENT_FIELDS
