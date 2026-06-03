"""Stage-5 P2 — VLM-based failure attribution + free-form replan baseline.

Loads InternVL3.5-8B (BF16) on a single A100-40GB and exposes two diagnosis
modes against a first-person failure-attempt frame:

* :meth:`InternVLClient.diagnose_constrained` (C1) — picks ONE intent factor
  name from the 6-factor menu (matches :data:`babysteps.schemas.INTENT_FIELDS`).
  Output downstream feeds the existing discrete revision pipeline.
* :meth:`InternVLClient.diagnose_free_form` (C2) — emits the full revised
  intent JSON. Used verbatim as the retry intent (baseline only).

The module is ``import``-safe on CPU/login nodes: the heavy ``transformers``
import + GPU allocation happen inside :meth:`InternVLClient.load`, not at
import time. Tests use :class:`MockVLMClient` to avoid both.

Reference: HuggingFace model card for ``OpenGVLab/InternVL3_5-8B``
(``load_image`` dynamic-tile preprocessing copied from the model card).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from babysteps.schemas import (
    APPROACH_DIRECTIONS, CONSTRAINT_REGIONS, CONTACT_REGIONS,
    DIRECTION_GROUNDINGS, EMBODIMENT_MAPPINGS, GOAL_STATES, INTENT_FIELDS,
    Intent, OBJECT_MOTIONS,
)

# The 6 core factors — what most tasks use as their C1 menu. CrossViewPush
# extends this to 7 (see TASK_PROMPT_INFO[...]["factor_menu"]).
INTENT_FACTOR_NAMES: tuple[str, ...] = INTENT_FIELDS

# Per-factor token whitelist used for C2 JSON validation. Keys cover all 7
# possible Intent fields; per-task `factor_menu` decides which subset of
# keys must be present in a valid C2 JSON.
_FACTOR_TOKENS = {
    "goal_state": GOAL_STATES,
    "object_motion": OBJECT_MOTIONS,
    "contact_region": CONTACT_REGIONS,
    "approach_direction": APPROACH_DIRECTIONS,
    "constraint_region": CONSTRAINT_REGIONS,
    "embodiment_mapping": EMBODIMENT_MAPPINGS,
    "direction_grounding": DIRECTION_GROUNDINGS,
}

_MODEL_ID = "OpenGVLab/InternVL3_5-8B"


# ---------- per-task prompt context ------------------------------------- #

# Each entry supplies the VLM with (a) what task it is looking at,
# (b) a one-line description of what success looks like, (c) per-factor
# valid-token hints (`expected_tokens`) for the factors that matter for
# this task, and (d) the `factor_menu` the VLM picks from in C1 and emits
# as JSON keys in C2.
#
# expected_tokens drives the "valid X tokens for this task" prompt line.
# It exists because some Stage-0 failure modes are symbolic — the wrong
# factor is the *token name* in the intent, not anything visible in the
# image. StackCube (goal_state=cube_at_target → cubeA_on_cubeB) and
# CrossViewPush (direction_grounding=actor_frame → observer_frame) are
# the canonical cases.
#
# factor_menu is the C1 ranked-choice. PushCube/PickCube/StackCube/
# TurnFaucet stay at the 6 core factors. CrossViewPush adds
# direction_grounding (the 7th additive factor from Sub-project E).
_FACTOR_MENU_6: tuple[str, ...] = INTENT_FIELDS
_FACTOR_MENU_7: tuple[str, ...] = INTENT_FIELDS + ("direction_grounding",)

TASK_PROMPT_INFO: dict[str, dict[str, object]] = {
    "PushCube-v1": {
        "name": "PushCube",
        "success_description": (
            "the cube is pushed sideways across the table to a marked "
            "target position"
        ),
        "expected_tokens": {"goal_state": ("cube_at_target",)},
        "factor_menu": _FACTOR_MENU_6,
    },
    "PickCube-v1": {
        "name": "PickCube",
        "success_description": (
            "the cube is grasped, lifted above the table, and held at a "
            "target xyz position"
        ),
        "expected_tokens": {"goal_state": ("cube_lifted_at_target",)},
        "factor_menu": _FACTOR_MENU_6,
    },
    "StackCube-v1": {
        "name": "StackCube",
        "success_description": (
            "cubeA (red) is picked up and placed on top of cubeB (green), "
            "with cubeA resting stably on cubeB"
        ),
        "expected_tokens": {"goal_state": ("cubeA_on_cubeB",)},
        "factor_menu": _FACTOR_MENU_6,
    },
    "TurnFaucet-v1": {
        "name": "TurnFaucet",
        "success_description": (
            "the faucet handle is rotated past its target angle. The Franka "
            "parallel-jaw gripper cannot enclose the handle, so the only "
            "mechanically feasible interaction is to poke the handle "
            "sideways rather than grasping it"
        ),
        "expected_tokens": {
            "goal_state": ("faucet_turned",),
            "embodiment_mapping": (
                "proxy_contact_to_franka_poke_turn",
            ),
        },
        "factor_menu": _FACTOR_MENU_6,
    },
    "CrossViewPush-v1": {
        "name": "CrossViewPush",
        "success_description": (
            "the cube is pushed to a target position. The demo was recorded "
            "from an observer camera offset from the actor's egocentric "
            "frame, so the demo trajectory direction must be interpreted in "
            "the observer's frame rather than the actor's (egocentric) frame"
        ),
        "expected_tokens": {
            "goal_state": ("cube_at_target",),
            "direction_grounding": ("observer_frame",),
        },
        "factor_menu": _FACTOR_MENU_7,
    },
}


def get_factor_menu(task: str) -> tuple[str, ...]:
    """Return the C1 factor menu (and C2 required JSON keys) for `task`."""
    if task not in TASK_PROMPT_INFO:
        known = sorted(TASK_PROMPT_INFO)
        raise KeyError(f"unknown task {task!r}; known: {known}")
    return TASK_PROMPT_INFO[task]["factor_menu"]  # type: ignore[return-value]


def _format_task_context(task: str) -> str:
    if task not in TASK_PROMPT_INFO:
        known = sorted(TASK_PROMPT_INFO)
        raise KeyError(f"unknown task {task!r}; known: {known}")
    info = TASK_PROMPT_INFO[task]
    lines = [
        f"Task: {info['name']}.",
        f"On a successful attempt, {info['success_description']}.",
    ]
    expected = info["expected_tokens"]  # type: ignore[index]
    for factor, valid in expected.items():  # type: ignore[union-attr]
        valid_str = ", ".join(valid)
        lines.append(
            f"For this task the {factor} factor should be one of: "
            f"[{valid_str}]."
        )
    return " ".join(lines)


# ---------- prompt builders --------------------------------------------- #


# Two-view image header. When the caller supplies a first-person wrist frame
# alongside the third-person frame, the prompt opens with two labeled <image>
# placeholders (one per image, in [third_person, wrist] order — matching the
# image_paths list passed to _chat_multi). The wrist frame is the robot's
# egocentric execution view; the third-person frame is the overhead scene view.
_MULTI_VIEW_HEADER = (
    "Image-1 (overhead third-person view of the scene): <image>\n"
    "Image-2 (robot wrist camera, first-person execution view): <image>\n"
)
_SINGLE_VIEW_HEADER = "<image>\n"


def _image_header(wrist_view: bool) -> str:
    return _MULTI_VIEW_HEADER if wrist_view else _SINGLE_VIEW_HEADER


def build_constrained_prompt(
    *, task: str, initial_intent: Intent, failure_predicate: str,
    wrist_view: bool = False,
) -> str:
    """C1 prompt: pick ONE factor name from the per-task factor menu.

    ``wrist_view`` switches the image header to two labeled placeholders
    (third-person + first-person wrist); the caller routes both frames through
    :meth:`InternVLClient._chat_multi`. Default (single view) is byte-for-byte
    the original prompt.
    """
    factor_list = ", ".join(get_factor_menu(task))
    intent_json = json.dumps(initial_intent.to_dict(), sort_keys=True)
    return (
        f"{_image_header(wrist_view)}"
        "You are diagnosing a robot manipulation failure.\n"
        f"{_format_task_context(task)}\n"
        f"The robot attempted: {intent_json}\n"
        f"Failure observation: {failure_predicate}\n"
        "Which ONE intent factor was wrong? Choose exactly one from:\n"
        f"[{factor_list}]\n"
        "Output ONLY the factor name, nothing else."
    )


def _format_allowed_values_menu(task: str) -> str:
    """Per-factor allowed-VALUE menu for the C2 free-form prompt.

    C2 is told the same value vocabulary that C1 effectively gets via the
    constrained menu + the per-task expected_tokens hints. Without this, C2 is
    a parser straw-man (it is asked for JSON keys but never told the legal
    values, so it invents prose values that the strict parser rejects). Pulls
    directly from :data:`_FACTOR_TOKENS` for the per-task factor menu — no
    task-specific synonyms, just the schema whitelist. Values are sorted for a
    deterministic, snapshot-stable prompt.
    """
    lines = []
    for field in get_factor_menu(task):
        vals = ", ".join(sorted(_FACTOR_TOKENS[field]))
        lines.append(f"- {field}: one of [{vals}]")
    return "\n".join(lines)


def build_free_form_prompt(
    *, task: str, initial_intent: Intent, failure_predicate: str,
    wrist_view: bool = False,
) -> str:
    """C2 prompt: emit the full corrected intent as JSON with the per-task
    factor menu as required keys. ``wrist_view`` adds the first-person wrist
    placeholder (see :func:`build_constrained_prompt`).

    Fair-baseline design (validated on real InternVL, 2026-06-03). TWO halves,
    both required:
      1. Enumerate the allowed VALUES per factor, so C2 emits executable, in-
         schema intents. Without it StackCube C2 invents physically-sensible but
         out-of-vocab tokens (e.g. object_motion='translate_+z' for stacking) and
         parse-fails (0.30); the format-repair retry does NOT rescue them because
         the model anchors on its own first out-of-vocab guess.
      2. State the attempt FAILED and must be changed, so the model does not just
         echo the already-valid current intent when the failure is not visible in
         the frame. Without it PushCube C2 echoes and success collapses 0.96->0.12.
    Menu-alone anchors PushCube into echoing; corrective-alone lets StackCube
    anchor on its out-of-vocab first guess — only the combination is strong on
    both. The format-repair re-prompt (:func:`build_free_form_repair_prompt`) is a
    third safety net. The instruction is neutral on selectivity: C2 is free to
    change any/all factors, it is only told the attempt failed and must change.
    """
    factor_list = ", ".join(get_factor_menu(task))
    intent_json = json.dumps(initial_intent.to_dict(), sort_keys=True)
    return (
        f"{_image_header(wrist_view)}"
        "You are a robot manipulation planner.\n"
        f"{_format_task_context(task)}\n"
        f"The robot attempted: {intent_json}\n"
        f"Failure observation: {failure_predicate}\n"
        "This attempt FAILED. Output a corrected full intent that fixes the\n"
        "failure; it must not be identical to the attempted intent above.\n"
        "Use these exact keys:\n"
        f"{factor_list}.\n"
        "Each value MUST be chosen ONLY from the allowed values for that key:\n"
        f"{_format_allowed_values_menu(task)}\n"
        "Output ONLY the JSON object, nothing else."
    )


def build_free_form_repair_prompt(
    *, task: str, initial_intent: Intent, failure_predicate: str,
    prior_output: str, wrist_view: bool = False,
) -> str:
    """C2 format-repair re-prompt (one retry after a parse failure).

    Re-states the request with the explicit per-factor allowed-value menu
    inlined and shows the model its own un-parseable prior output, so the
    repair is a FORMAT fix (valid JSON / in-vocab tokens), not a second
    free-form guess. Stays neutral: it does not tell the model which value to
    pick, only which values are legal.
    """
    factor_list = ", ".join(get_factor_menu(task))
    intent_json = json.dumps(initial_intent.to_dict(), sort_keys=True)
    return (
        f"{_image_header(wrist_view)}"
        "You are a robot manipulation planner.\n"
        f"{_format_task_context(task)}\n"
        f"The robot attempted: {intent_json}\n"
        f"Failure observation: {failure_predicate}\n"
        "Your previous answer could not be parsed:\n"
        f"{prior_output.strip()}\n"
        "Re-output the corrected full intent as a single JSON object with "
        f"these exact keys: {factor_list}.\n"
        "Choose each value ONLY from:\n"
        f"{_format_allowed_values_menu(task)}\n"
        "Output ONLY the JSON object, nothing else."
    )


# ---------- demo intent-READ (Stage-5 option-3 de-risking probe) -------- #
#
# Distinct from C1/C2 (which diagnose a *failure*). Here the VLM READS the
# `object_motion` factor straight off a third-person DEMO strip (start /
# middle / end panels), to measure VLM-vs-oracle agreement as a candidate
# distillation supervision signal. The varied-intent cuts only vary
# object_motion, so it is the single decisive factor to read.

# The four lateral directions PushCube / StackCube object_motion ranges over.
OBJECT_MOTION_MENU_4: tuple[str, ...] = (
    "translate_+x", "translate_-x", "translate_+y", "translate_-y",
)

# Camera-calibrated world-axis -> image-direction convention. Derived from a
# blob-displacement calibration over the labeled varied-intent seeds against
# the FIXED `render_camera` shared by the cube tasks (PushCube +x demos move
# the cube lower-left; StackCube +y demos move red lower-right, etc.). The
# VLM cannot guess a world-frame convention from pixels, so we state it — the
# same mapping the DINOv2 linear probe gets to *learn* from its folds.
_AXIS_CONVENTION = (
    "In this fixed camera view the world axes project onto the image as: "
    "+x points toward the LOWER-LEFT, -x toward the UPPER-RIGHT, "
    "+y toward the LOWER-RIGHT, -y toward the UPPER-LEFT."
)

# What "object_motion" denotes per task (object evidence, not a motor program).
_OBJECT_MOTION_QUESTION: dict[str, str] = {
    "PushCube-v1": (
        "the direction the blue cube is pushed across the table"
    ),
    "StackCube-v1": (
        "the direction the red cube (cubeA) must travel to be placed on top "
        "of the green cube (cubeB)"
    ),
}


def object_motion_question(task: str) -> str:
    if task not in _OBJECT_MOTION_QUESTION:
        known = sorted(_OBJECT_MOTION_QUESTION)
        raise KeyError(f"no object_motion question for {task!r}; known: {known}")
    return _OBJECT_MOTION_QUESTION[task]


def build_intent_read_prompt(*, task: str, n_panels: int = 3) -> str:
    """Demo-read prompt: name the `object_motion` token from a demo strip."""
    menu = ", ".join(OBJECT_MOTION_MENU_4)
    what = object_motion_question(task)
    return (
        "<image>\n"
        f"The image shows {n_panels} panels left-to-right: the START, "
        "MIDDLE, and END of a robot demonstration filmed third-person.\n"
        f"{_format_task_context(task)}\n"
        f"{_AXIS_CONVENTION}\n"
        f"Identify {what}.\n"
        f"Output ONLY one token from: [{menu}], nothing else."
    )


def parse_object_motion_output(
    raw: str, *, menu: tuple[str, ...] = OBJECT_MOTION_MENU_4,
) -> Optional[str]:
    """Return the first `menu` token in `raw`, else None.

    Tolerant of whitespace, quotes, and a prose preamble. Tokens contain
    ``+``/``-`` (not ``\\b``-friendly), so this scans by substring; the four
    tokens are mutually non-substring so order is unambiguous.
    """
    if not raw:
        return None
    stripped = raw.strip().strip('"').strip("'").strip()
    if stripped in menu:
        return stripped
    for tok in menu:
        if tok in raw:
            return tok
    return None


# ---------- before/after MULTI-IMAGE motion read ----------------------- #
#
# Diagnostic run #1 showed a single 3-panel strip makes the VLM report
# "moves left-to-right across the panels" (reads layout, not physics).
# Feeding START and END as SEPARATE images (a before/after comparison) and
# asking for an IMAGE-RELATIVE direction fixes it — the caller maps that
# image direction to the world `object_motion` token via a fixed,
# camera-calibrated lookup (perception by the VLM; bookkeeping in code).

MOTION_DIRECTION_MENU: tuple[str, ...] = ("left", "right", "up", "down", "none")

_BEFORE_AFTER_FOCUS: dict[str, str] = {
    "PushCube-v1": "the small BLUE cube on the table",
    "StackCube-v1": "the RED cube",
}


def before_after_focus(task: str) -> str:
    if task not in _BEFORE_AFTER_FOCUS:
        known = sorted(_BEFORE_AFTER_FOCUS)
        raise KeyError(f"no before/after focus for {task!r}; known: {known}")
    return _BEFORE_AFTER_FOCUS[task]


def build_before_after_prompt(*, task: str) -> str:
    """Two-image prompt: image-relative direction the focus object moved."""
    focus = before_after_focus(task)
    return (
        "Image-1: <image>\nImage-2: <image>\n"
        "Image-1 is the START and Image-2 is the END of a robot demonstration, "
        "filmed from the SAME fixed third-person camera.\n"
        f"Focus ONLY on {focus}. Comparing its position in Image-1 versus "
        "Image-2, which way did it move across the table surface?\n"
        "Answer with exactly one word: left, right, up, down, or none."
    )


def parse_motion_direction(
    raw: str, *, menu: tuple[str, ...] = MOTION_DIRECTION_MENU,
) -> Optional[str]:
    """Return the image-relative direction word in `raw`, else None."""
    if not raw:
        return None
    s = raw.strip().strip('".').strip("'").strip().lower()
    if s in menu:
        return s
    low = raw.lower()
    for w in menu:
        if re.search(rf"\b{w}\b", low):
            return w
    return None


# ---------- output parsers ---------------------------------------------- #


def parse_constrained_output(
    raw: str, *, factor_menu: tuple[str, ...] = INTENT_FACTOR_NAMES,
) -> Optional[str]:
    """Return the first ``factor_menu`` token in ``raw``, else None.

    Tolerant of leading/trailing whitespace, surrounding quotes, and a
    one-sentence prose preamble (e.g. 'The wrong factor is X because ...').

    ``factor_menu`` defaults to the 6-factor schema. CrossViewPush passes
    the 7-factor menu (with `direction_grounding`).
    """
    if not raw:
        return None
    # Try literal match first (fast path for clean outputs).
    stripped = raw.strip().strip('"').strip("'").strip()
    if stripped in factor_menu:
        return stripped
    # Fallback: scan for first factor name anywhere in the string.
    # ``\b`` is fine here — factor names are alphanumeric + underscore.
    for name in factor_menu:
        if re.search(rf"\b{re.escape(name)}\b", raw):
            return name
    return None


def _normalize_token_value(raw_value: object, allowed: frozenset[str]) -> Optional[str]:
    """NEUTRAL normalization of a single C2 factor value against ``allowed``.

    Strips surrounding whitespace/quotes and does a case-insensitive exact
    match against the schema whitelist for that factor. Returns the CANONICAL
    token (the one in ``allowed``) on match, else None.

    Deliberately NOT a synonym dictionary: it never decides what the VLM
    "meant" — it only forgives casing/quoting/whitespace differences on an
    otherwise-exact token. Anything genuinely out-of-vocab (prose, a paraphrase)
    returns None so the caller can signal parse-fail and retry.
    """
    if not isinstance(raw_value, str):
        return None
    cleaned = raw_value.strip().strip('"').strip("'").strip()
    if cleaned in allowed:
        return cleaned
    lowered = cleaned.lower()
    for tok in allowed:
        if tok.lower() == lowered:
            return tok
    return None


def parse_free_form_output(
    raw: str, *, factor_menu: tuple[str, ...] = INTENT_FACTOR_NAMES,
) -> Optional[Intent]:
    """Parse a JSON intent. Returns None on any failure (missing keys, bad
    tokens, malformed JSON). Strips a single `````json …````` fence if present.

    ``factor_menu`` lists the JSON keys that MUST be present in the parsed
    object. Defaults to the 6-factor schema.

    Values pass through NEUTRAL normalization (case-insensitive + whitespace/
    quote-stripped exact match against the schema whitelist) so a fair C2
    baseline isn't penalised for casing/quoting — see
    :func:`_normalize_token_value`. No task-specific synonyms.
    """
    if not raw:
        return None
    s = raw.strip()
    # Strip code fence.
    fence = re.match(r"```(?:json)?\s*(.*?)\s*```", s, flags=re.DOTALL)
    if fence:
        s = fence.group(1).strip()
    # Some models include explanatory text before/after — try to find the
    # outermost {...} block.
    match = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if match:
        s = match.group(0)
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    # Strict key check + NEUTRAL value normalization over the per-task menu.
    normalized: dict[str, str] = {}
    for field in factor_menu:
        if field not in obj:
            return None
        canonical = _normalize_token_value(obj[field], _FACTOR_TOKENS[field])
        if canonical is None:
            return None
        normalized[field] = canonical
    try:
        return Intent.from_dict(normalized)
    except (TypeError, ValueError, KeyError):
        return None


# ---------- clients ----------------------------------------------------- #


@dataclass
class MockVLMClient:
    """Sim-free stand-in. Returns canned responses; ignores the image.

    ``free_form_repair_response`` is the canned reply to the ONE format-repair
    re-prompt. It defaults to ``None`` meaning "repeat ``free_form_response``";
    set it to a valid-JSON string (with ``free_form_response`` set to prose) to
    exercise the prose-then-repair path in tests.
    """
    constrained_response: str = "approach_direction"
    free_form_response: str = (
        '{"goal_state":"cube_at_target","object_motion":"translate_+x",'
        '"contact_region":"plus_x_face","approach_direction":"from_plus_x",'
        '"constraint_region":"none","embodiment_mapping":"proxy_contact_to_franka_push"}'
    )
    free_form_repair_response: Optional[str] = None
    object_motion_response: str = "translate_+x"
    motion_direction_response: str = "left"

    def read_object_motion(
        self, *, task: str, image_path: str | Path,
    ) -> Optional[str]:
        # Build exercised for realism even in mock mode.
        _ = build_intent_read_prompt(task=task)
        return parse_object_motion_output(self.object_motion_response)

    def read_motion_direction(
        self, *, task: str, start_path: str | Path, end_path: str | Path,
    ) -> Optional[str]:
        _ = build_before_after_prompt(task=task)
        return parse_motion_direction(self.motion_direction_response)

    def diagnose_constrained(
        self, *, task: str, image_path: str | Path, initial_intent: Intent,
        failure_predicate: str, wrist_image_path: str | Path | None = None,
    ) -> Optional[str]:
        # Build/parse exercised for realism even in mock mode (including the
        # two-view header when a wrist frame is supplied). Image is ignored.
        _ = build_constrained_prompt(
            task=task, initial_intent=initial_intent,
            failure_predicate=failure_predicate,
            wrist_view=wrist_image_path is not None,
        )
        return parse_constrained_output(
            self.constrained_response, factor_menu=get_factor_menu(task),
        )

    def diagnose_free_form(
        self, *, task: str, image_path: str | Path, initial_intent: Intent,
        failure_predicate: str, wrist_image_path: str | Path | None = None,
    ) -> Optional[Intent]:
        intent, _raw = self.diagnose_free_form_verbose(
            task=task, image_path=image_path, initial_intent=initial_intent,
            failure_predicate=failure_predicate,
            wrist_image_path=wrist_image_path,
        )
        return intent

    def diagnose_free_form_verbose(
        self, *, task: str, image_path: str | Path, initial_intent: Intent,
        failure_predicate: str, wrist_image_path: str | Path | None = None,
    ) -> tuple[Optional[Intent], str]:
        """Like :meth:`diagnose_free_form` but also returns the raw VLM text
        (the repair reply if a repair happened, else the first reply) so the
        eval loop can persist ``raw_vlm_text``. Exercises the ONE format-repair
        retry: if ``free_form_response`` fails to parse, retry with
        ``free_form_repair_response`` (defaults to ``free_form_response``)."""
        menu = get_factor_menu(task)
        _ = build_free_form_prompt(
            task=task, initial_intent=initial_intent,
            failure_predicate=failure_predicate,
            wrist_view=wrist_image_path is not None,
        )
        raw = self.free_form_response
        intent = parse_free_form_output(raw, factor_menu=menu)
        if intent is not None:
            return intent, raw
        # ONE format-repair retry.
        repair_raw = (
            self.free_form_repair_response
            if self.free_form_repair_response is not None
            else self.free_form_response
        )
        _ = build_free_form_repair_prompt(
            task=task, initial_intent=initial_intent,
            failure_predicate=failure_predicate, prior_output=raw,
            wrist_view=wrist_image_path is not None,
        )
        intent = parse_free_form_output(repair_raw, factor_menu=menu)
        return intent, repair_raw


class InternVLClient:
    """Real InternVL3.5-8B client. Heavy imports happen in :meth:`load`.

    ``max_new_tokens_*`` are split per method because the constrained C1
    answer is one factor name (~6 tokens) while the free-form C2 answer
    is a 6-field JSON object (~80 tokens). A single shared budget at 64
    truncated every C2 output during the first run (100% parse failure
    on all three tasks); 256 is comfortably above the JSON length.
    """

    def __init__(self, model_id: str = _MODEL_ID, max_num_tiles: int = 12,
                 max_new_tokens_constrained: int = 32,
                 max_new_tokens_free_form: int = 256) -> None:
        self.model_id = model_id
        self.max_num_tiles = max_num_tiles
        self.max_new_tokens_constrained = max_new_tokens_constrained
        self.max_new_tokens_free_form = max_new_tokens_free_form
        self._model = None
        self._tokenizer = None
        self._load_image = None  # populated from the model card helper

    def load(self) -> None:
        """Materialize the model on cuda:0 in BF16. Idempotent."""
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        # The HF model card ships a ``load_image`` dynamic-tile preprocessor.
        # We re-implement the same logic here so we don't depend on
        # ``trust_remote_code`` helper modules being available.
        self._load_image = _build_load_image()

        self._model = AutoModel.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
        ).eval().cuda()
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True, use_fast=False,
        )

    def _chat(self, *, image_path: str | Path, question: str,
              max_new_tokens: int) -> str:
        import torch
        if self._model is None:
            self.load()
        pixel_values = self._load_image(
            str(image_path), max_num=self.max_num_tiles,
        ).to(torch.bfloat16).cuda()
        gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False)
        response = self._model.chat(
            self._tokenizer, pixel_values, question, gen_kwargs,
        )
        return response

    def diagnose_constrained(
        self, *, task: str, image_path: str | Path, initial_intent: Intent,
        failure_predicate: str, wrist_image_path: str | Path | None = None,
    ) -> Optional[str]:
        prompt = build_constrained_prompt(
            task=task, initial_intent=initial_intent,
            failure_predicate=failure_predicate,
            wrist_view=wrist_image_path is not None,
        )
        if wrist_image_path is not None:
            raw = self._chat_multi(
                image_paths=[image_path, wrist_image_path], question=prompt,
                max_new_tokens=self.max_new_tokens_constrained,
            )
        else:
            raw = self._chat(
                image_path=image_path, question=prompt,
                max_new_tokens=self.max_new_tokens_constrained,
            )
        return parse_constrained_output(raw, factor_menu=get_factor_menu(task))

    def diagnose_free_form(
        self, *, task: str, image_path: str | Path, initial_intent: Intent,
        failure_predicate: str, wrist_image_path: str | Path | None = None,
    ) -> Optional[Intent]:
        intent, _raw = self.diagnose_free_form_verbose(
            task=task, image_path=image_path, initial_intent=initial_intent,
            failure_predicate=failure_predicate,
            wrist_image_path=wrist_image_path,
        )
        return intent

    def _chat_one(self, *, image_path, wrist_image_path, question: str) -> str:
        """Route a single C2 turn through the single- or multi-image path."""
        if wrist_image_path is not None:
            return self._chat_multi(
                image_paths=[image_path, wrist_image_path], question=question,
                max_new_tokens=self.max_new_tokens_free_form,
            )
        return self._chat(
            image_path=image_path, question=question,
            max_new_tokens=self.max_new_tokens_free_form,
        )

    def diagnose_free_form_verbose(
        self, *, task: str, image_path: str | Path, initial_intent: Intent,
        failure_predicate: str, wrist_image_path: str | Path | None = None,
    ) -> tuple[Optional[Intent], str]:
        """C2 free-form replan with ONE format-repair retry.

        Returns ``(intent_or_None, raw_vlm_text)``. If the first reply fails to
        parse, re-prompt ONCE with the explicit per-factor allowed-value menu
        inlined (:func:`build_free_form_repair_prompt`) and parse again. If the
        repair reply also fails, returns ``(None, repair_raw)``. The returned
        raw text is whichever reply was parsed last (repair reply if a repair
        happened), so the caller can persist it for parse debugging.
        """
        menu = get_factor_menu(task)
        prompt = build_free_form_prompt(
            task=task, initial_intent=initial_intent,
            failure_predicate=failure_predicate,
            wrist_view=wrist_image_path is not None,
        )
        raw = self._chat_one(
            image_path=image_path, wrist_image_path=wrist_image_path,
            question=prompt,
        )
        intent = parse_free_form_output(raw, factor_menu=menu)
        if intent is not None:
            return intent, raw
        # ONE format-repair retry with the explicit allowed-value menu inlined.
        repair_prompt = build_free_form_repair_prompt(
            task=task, initial_intent=initial_intent,
            failure_predicate=failure_predicate, prior_output=raw,
            wrist_view=wrist_image_path is not None,
        )
        repair_raw = self._chat_one(
            image_path=image_path, wrist_image_path=wrist_image_path,
            question=repair_prompt,
        )
        intent = parse_free_form_output(repair_raw, factor_menu=menu)
        return intent, repair_raw

    def read_object_motion(
        self, *, task: str, image_path: str | Path,
    ) -> Optional[str]:
        prompt = build_intent_read_prompt(task=task)
        raw = self._chat(
            image_path=image_path, question=prompt,
            max_new_tokens=self.max_new_tokens_constrained,
        )
        return parse_object_motion_output(raw)

    def _chat_multi(self, *, image_paths, question: str,
                    max_new_tokens: int) -> str:
        """InternVL multi-image chat (one <image> placeholder per path)."""
        import torch
        if self._model is None:
            self.load()
        pvs = [
            self._load_image(str(p), max_num=self.max_num_tiles)
            .to(torch.bfloat16).cuda()
            for p in image_paths
        ]
        num_patches_list = [pv.size(0) for pv in pvs]
        pixel_values = torch.cat(pvs, dim=0)
        gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False)
        return self._model.chat(
            self._tokenizer, pixel_values, question, gen_kwargs,
            num_patches_list=num_patches_list,
        )

    def read_motion_direction(
        self, *, task: str, start_path: str | Path, end_path: str | Path,
    ) -> Optional[str]:
        prompt = build_before_after_prompt(task=task)
        raw = self._chat_multi(
            image_paths=[start_path, end_path], question=prompt,
            max_new_tokens=self.max_new_tokens_constrained,
        )
        return parse_motion_direction(raw)


# ---------- model-card load_image helper -------------------------------- #


def _build_load_image():
    """Build the dynamic-tile preprocessor from the InternVL3.5 model card.

    Returns a callable ``load_image(path, max_num=12) -> torch.Tensor``
    of shape ``(n_tiles, 3, 448, 448)`` BF16-castable.
    """
    import torch
    import torchvision.transforms as T
    from PIL import Image
    from torchvision.transforms.functional import InterpolationMode

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    IMAGE_SIZE = 448

    transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((IMAGE_SIZE, IMAGE_SIZE),
                 interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height,
                                   image_size):
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif (ratio_diff == best_ratio_diff
                  and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]):
                best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(image, min_num=1, max_num=12, image_size=IMAGE_SIZE,
                            use_thumbnail=True):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1)
            for i in range(1, n + 1) for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        target_aspect_ratio = find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size,
        )
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size,
            )
            processed_images.append(resized_img.crop(box))
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def load_image(image_path, max_num=12):
        image = Image.open(image_path).convert("RGB")
        images = dynamic_preprocess(
            image, image_size=IMAGE_SIZE, use_thumbnail=True, max_num=max_num,
        )
        pixel_values = torch.stack([transform(img) for img in images])
        return pixel_values

    return load_image
