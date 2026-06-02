"""Sim-free tests for the Stage-5 VLM demo intent-READ probe.

Covers the additive surface in ``babysteps.stage5.vlm_attribute``:
``build_intent_read_prompt``, ``parse_object_motion_output``,
``object_motion_question``, ``OBJECT_MOTION_MENU_4``, and
``MockVLMClient.read_object_motion``. No torch / imageio / GPU.
"""
from __future__ import annotations

import pytest

from babysteps import schemas
from babysteps.stage5.vlm_attribute import (
    MOTION_DIRECTION_MENU, MockVLMClient, OBJECT_MOTION_MENU_4,
    before_after_focus, build_before_after_prompt, build_intent_read_prompt,
    object_motion_question, parse_motion_direction, parse_object_motion_output,
)


def test_menu_is_subset_of_schema_object_motions():
    assert set(OBJECT_MOTION_MENU_4) <= set(schemas.OBJECT_MOTIONS)
    assert len(OBJECT_MOTION_MENU_4) == 4


@pytest.mark.parametrize("task", ["PushCube-v1", "StackCube-v1"])
def test_prompt_has_image_context_convention_and_menu(task):
    p = build_intent_read_prompt(task=task)
    assert p.startswith("<image>")
    assert "START" in p and "END" in p          # temporal panels described
    assert "LOWER-LEFT" in p and "UPPER-RIGHT" in p  # axis convention stated
    for tok in OBJECT_MOTION_MENU_4:            # full value menu offered
        assert tok in p
    assert object_motion_question(task) in p   # per-task object evidence


def test_prompt_describes_object_evidence_not_motor_program():
    # Demo captions/prompts describe object motion, never a Franka program.
    push = build_intent_read_prompt(task="PushCube-v1")
    stack = build_intent_read_prompt(task="StackCube-v1")
    assert "blue cube" in push
    assert "red cube" in stack and "green cube" in stack


def test_object_motion_question_unknown_task_raises():
    with pytest.raises(KeyError):
        object_motion_question("TurnFaucet-v1")


@pytest.mark.parametrize("tok", list(OBJECT_MOTION_MENU_4))
def test_parse_clean_token_roundtrips(tok):
    assert parse_object_motion_output(tok) == tok
    assert parse_object_motion_output(f'  "{tok}"  ') == tok


def test_parse_tolerates_prose_preamble():
    raw = "The cube moved in direction translate_-y based on the panels."
    assert parse_object_motion_output(raw) == "translate_-y"


def test_parse_unknown_and_empty_return_none():
    assert parse_object_motion_output("translate_+z") is None
    assert parse_object_motion_output("lift_up") is None
    assert parse_object_motion_output("") is None
    assert parse_object_motion_output("   ") is None


def test_parse_does_not_confuse_x_and_y_tokens():
    assert parse_object_motion_output("translate_+y") == "translate_+y"
    assert parse_object_motion_output("translate_+x") == "translate_+x"


def test_mock_client_reads_configured_token():
    c = MockVLMClient(object_motion_response="translate_-x")
    assert c.read_object_motion(task="StackCube-v1", image_path="ignored.png") == (
        "translate_-x"
    )


def test_mock_client_default_returns_valid_menu_token():
    c = MockVLMClient()
    out = c.read_object_motion(task="PushCube-v1", image_path="ignored.png")
    assert out in OBJECT_MOTION_MENU_4


# ---- before/after multi-image motion read (the working representation) ---- #


@pytest.mark.parametrize("task", ["PushCube-v1", "StackCube-v1"])
def test_before_after_prompt_has_two_image_slots_and_word_menu(task):
    p = build_before_after_prompt(task=task)
    assert p.count("<image>") == 2          # START and END as separate images
    assert "Image-1" in p and "Image-2" in p
    assert before_after_focus(task) in p
    for w in ("left", "right", "up", "down", "none"):
        assert w in p


def test_before_after_focus_unknown_task_raises():
    with pytest.raises(KeyError):
        before_after_focus("TurnFaucet-v1")


@pytest.mark.parametrize("w", list(MOTION_DIRECTION_MENU))
def test_parse_motion_direction_clean_and_punctuated(w):
    assert parse_motion_direction(w) == w
    assert parse_motion_direction(f"{w.capitalize()}.") == w
    assert parse_motion_direction(f"The cube moved {w} across the table.") == w


def test_parse_motion_direction_unknown_returns_none():
    assert parse_motion_direction("diagonally") is None
    assert parse_motion_direction("") is None


def test_mock_read_motion_direction():
    c = MockVLMClient(motion_direction_response="right")
    out = c.read_motion_direction(
        task="PushCube-v1", start_path="a.png", end_path="b.png",
    )
    assert out == "right"
