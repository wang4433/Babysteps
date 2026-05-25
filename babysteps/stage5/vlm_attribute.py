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
    EMBODIMENT_MAPPINGS, GOAL_STATES, INTENT_FIELDS, Intent, OBJECT_MOTIONS,
)

# The VLM's menu of factor names — IS the Stage-0 6-factor schema.
INTENT_FACTOR_NAMES: tuple[str, ...] = INTENT_FIELDS

# Per-factor whitelist used for C2 JSON validation.
_FACTOR_TOKENS = {
    "goal_state": GOAL_STATES,
    "object_motion": OBJECT_MOTIONS,
    "contact_region": CONTACT_REGIONS,
    "approach_direction": APPROACH_DIRECTIONS,
    "constraint_region": CONSTRAINT_REGIONS,
    "embodiment_mapping": EMBODIMENT_MAPPINGS,
}

_MODEL_ID = "OpenGVLab/InternVL3_5-8B"


# ---------- prompt builders --------------------------------------------- #


def build_constrained_prompt(
    *, initial_intent: Intent, failure_predicate: str,
) -> str:
    """C1 prompt: pick ONE factor name from the fixed 6-set."""
    factor_list = ", ".join(INTENT_FACTOR_NAMES)
    intent_json = json.dumps(initial_intent.to_dict(), sort_keys=True)
    return (
        "<image>\n"
        "You are diagnosing a robot manipulation failure.\n"
        f"The robot attempted: {intent_json}\n"
        f"Failure observation: {failure_predicate}\n"
        "Which ONE intent factor was wrong? Choose exactly one from:\n"
        f"[{factor_list}]\n"
        "Output ONLY the factor name, nothing else."
    )


def build_free_form_prompt(
    *, initial_intent: Intent, failure_predicate: str,
) -> str:
    """C2 prompt: emit the full corrected intent as JSON."""
    factor_list = ", ".join(INTENT_FACTOR_NAMES)
    intent_json = json.dumps(initial_intent.to_dict(), sort_keys=True)
    return (
        "<image>\n"
        "You are a robot manipulation planner.\n"
        f"The robot attempted: {intent_json}\n"
        f"Failure observation: {failure_predicate}\n"
        "Output the corrected full intent as JSON with these exact keys:\n"
        f"{factor_list}.\n"
        "Output ONLY the JSON object, nothing else."
    )


# ---------- output parsers ---------------------------------------------- #


def parse_constrained_output(raw: str) -> Optional[str]:
    """Return the first INTENT_FACTOR_NAMES token in ``raw``, else None.

    Tolerant of leading/trailing whitespace, surrounding quotes, and a
    one-sentence prose preamble (e.g. 'The wrong factor is X because ...').
    """
    if not raw:
        return None
    # Try literal match first (fast path for clean outputs).
    stripped = raw.strip().strip('"').strip("'").strip()
    if stripped in INTENT_FACTOR_NAMES:
        return stripped
    # Fallback: scan for first factor name anywhere in the string.
    # ``\b`` is fine here — factor names are alphanumeric + underscore.
    for name in INTENT_FACTOR_NAMES:
        if re.search(rf"\b{re.escape(name)}\b", raw):
            return name
    return None


def parse_free_form_output(raw: str) -> Optional[Intent]:
    """Parse a JSON intent. Returns None on any failure (missing keys, bad
    tokens, malformed JSON). Strips a single `````json …````` fence if present."""
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
    # Strict key/token check.
    for field in INTENT_FIELDS:
        if field not in obj:
            return None
        if obj[field] not in _FACTOR_TOKENS[field]:
            return None
    try:
        return Intent(
            goal_state=obj["goal_state"],
            object_motion=obj["object_motion"],
            contact_region=obj["contact_region"],
            approach_direction=obj["approach_direction"],
            constraint_region=obj["constraint_region"],
            embodiment_mapping=obj["embodiment_mapping"],
        )
    except (TypeError, ValueError):
        return None


# ---------- clients ----------------------------------------------------- #


@dataclass
class MockVLMClient:
    """Sim-free stand-in. Returns canned responses; ignores the image."""
    constrained_response: str = "approach_direction"
    free_form_response: str = (
        '{"goal_state":"cube_at_target","object_motion":"translate_+x",'
        '"contact_region":"plus_x_face","approach_direction":"from_plus_x",'
        '"constraint_region":"none","embodiment_mapping":"proxy_contact_to_franka_push"}'
    )

    def diagnose_constrained(
        self, *, image_path: str | Path, initial_intent: Intent,
        failure_predicate: str,
    ) -> Optional[str]:
        # Build/parse exercised for realism even in mock mode.
        _ = build_constrained_prompt(
            initial_intent=initial_intent, failure_predicate=failure_predicate,
        )
        return parse_constrained_output(self.constrained_response)

    def diagnose_free_form(
        self, *, image_path: str | Path, initial_intent: Intent,
        failure_predicate: str,
    ) -> Optional[Intent]:
        _ = build_free_form_prompt(
            initial_intent=initial_intent, failure_predicate=failure_predicate,
        )
        return parse_free_form_output(self.free_form_response)


class InternVLClient:
    """Real InternVL3.5-8B client. Heavy imports happen in :meth:`load`."""

    def __init__(self, model_id: str = _MODEL_ID, max_num_tiles: int = 12,
                 max_new_tokens: int = 64) -> None:
        self.model_id = model_id
        self.max_num_tiles = max_num_tiles
        self.max_new_tokens = max_new_tokens
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

    def _chat(self, *, image_path: str | Path, question: str) -> str:
        import torch
        if self._model is None:
            self.load()
        pixel_values = self._load_image(
            str(image_path), max_num=self.max_num_tiles,
        ).to(torch.bfloat16).cuda()
        gen_kwargs = dict(max_new_tokens=self.max_new_tokens, do_sample=False)
        response = self._model.chat(
            self._tokenizer, pixel_values, question, gen_kwargs,
        )
        return response

    def diagnose_constrained(
        self, *, image_path: str | Path, initial_intent: Intent,
        failure_predicate: str,
    ) -> Optional[str]:
        prompt = build_constrained_prompt(
            initial_intent=initial_intent, failure_predicate=failure_predicate,
        )
        raw = self._chat(image_path=image_path, question=prompt)
        return parse_constrained_output(raw)

    def diagnose_free_form(
        self, *, image_path: str | Path, initial_intent: Intent,
        failure_predicate: str,
    ) -> Optional[Intent]:
        prompt = build_free_form_prompt(
            initial_intent=initial_intent, failure_predicate=failure_predicate,
        )
        raw = self._chat(image_path=image_path, question=prompt)
        return parse_free_form_output(raw)


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
