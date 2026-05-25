# Stage-5 P2 VLM Attribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Use a frozen vision-language model (InternVL3.5-8B) to diagnose which Stage-0 intent factor caused a failure, and compare **C1 — VLM-as-constrained-diagnoser + slot-local revision** against **C2 — VLM-as-free-form-replanner**. The paper-facing claim: VLMs are good at *diagnosis* but wasteful at *repair*; constraining them to "pick one factor" preserves the rest of the intent while matching success.

**Architecture:** Three layers. (a) A sim-free `babysteps.stage5.vlm_attribute` module that loads InternVL3.5, formats two prompt templates, parses outputs (factor-name for C1, JSON for C2) and degrades to a mock when no GPU. (b) A GPU script that re-renders failure-attempt frames for held-out seeds (deterministic). (c) An eval driver that, for each held-out failure episode, runs both conditions through `episode.run_episode` with a VLM-driven policy, computes the four metrics from the user spec (`attribution_accuracy`, `final_success_rate`, `frozen_factor_preservation`, `unnecessary_factor_change_rate`), and writes per-task + cross-task reports under `reports/stage5/p2_vlm_attribution/`.

**Tech Stack:** Python 3, `transformers>=4.52.1`, `torch` + CUDA, `flash-attn` (optional), HF model `OpenGVLab/InternVL3_5-8B` (BF16, ~17 GB → A100-40GB), the existing `babysteps` sim-free package + ManiSkill `env_runner.run` (for retry rollouts), Slurm on Gilbreth (a100-40gb partition, `--qos=standby` per project standing memory).

---

## Architectural Decisions (defaults chosen, surfaced for review)

These were open in the user's spec; the plan locks them in. Override before execution if any is wrong.

| Question | Decision | Why |
|---|---|---|
| Package location | `babysteps/stage5/__init__.py` + `babysteps/stage5/vlm_attribute.py` | User spec named this path; Stage-5 P1 code lives in `stage4/` for historical reasons but a clean `stage5/` is correct going forward. |
| Episode set | Three tasks: **PushCube-v1, PickCube-v1, StackCube-v1** | These have existing `babysteps_selective` baselines + per-task Stage-4 M2.5 packs. TurnFaucet has no M2.5 pack. CrossViewPush's wrong factor is `direction_grounding` — the 7th additive token, outside the spec's 6-factor menu. Restricting to 3 keeps the VLM menu fixed at 6. |
| Sample size | **50 held-out seeds per task → ~150 episodes total**, filtered to `failure_predicate != "none"` | Matches Stage-5 P1 held-out range (seeds 100-149). Each task's failure rate is ~100% in current data, so ~150 failure episodes. At ~5 s per VLM call × 2 conditions × 150 = ~25 min on A100. |
| Frame source | **Re-render attempt frames post-hoc, deterministic seed.** Single end-of-attempt RGB frame per episode. Saved as PNG. | Simpler than extending P1's frame-saving (which targets demo frames). Single frame is the minimum honest test of the VLM's diagnostic power. Multi-frame is a follow-up if C1 ≪ C2. |
| C1 revision mechanism | **Discrete `babysteps.revision.revise_intent`**, fed the VLM's factor name as the attribution | Avoids conflating VLM-attribution quality with the latent ReviseHead's quality. Tests exactly the claim: "does the VLM's factor choice succeed when plugged into the existing slot-local edit?" |
| C2 revision mechanism | **Apply the VLM's full intent JSON directly** as the retry intent, after schema validation. Operator: new `vlm_free_form_replan`. | Pure baseline — no learned edit, no rule table. If JSON is malformed or violates whitelists, count as `c2_parse_failed=True` (retry never runs, final_success=False). |
| Multi-frame? | **Single frame.** | Spec is open; simpler. If C1 underperforms, a follow-up plan adds multi-frame. |
| Rule-table baseline | Computed inline from `babysteps.failure.attribute_failure` over the same held-out cut | Costs nothing — no extra GPU job. |

---

## File Structure

**New files:**
```text
babysteps/stage5/__init__.py                                # tiny package marker
babysteps/stage5/vlm_attribute.py                           # VLM loader + 2 prompts + parsers (+ mock)
tests/test_vlm_attribute.py                                 # sim-free unit tests (mock VLM)
scripts/stage5_p2_render_failure_frames.py                  # GPU: capture end-of-attempt frame per seed
scripts/stage5_p2_vlm_eval.py                               # GPU: VLM × C1/C2 eval driver
slurm/stage5_p2_render_failure_frames.sbatch                # A100-40gb job for frame render
slurm/stage5_p2_vlm.sbatch                                  # A100-40gb job for VLM eval
docs/superpowers/specs/2026-05-25-stage5-p2-vlm-attribution-design.md   # spec snapshot (one-pager)
```

**Modified files:**
```text
babysteps/schemas.py                                        # add 2 revision operators to whitelist
slurm/CLAUDE.md                                             # record P2 gate results when run
goal.md                                                     # tick P2 done note (after gate passes)
```

**Output paths (created by run):**
```text
datasets/stage5/p2_vlm/<task>/frames/seed_NNNN_attempt.png         # one PNG per failure seed
datasets/stage5/p2_vlm/<task>/episodes.jsonl                       # re-collected held-out failure episodes
reports/stage5/p2_vlm_attribution/<task>/c1_results.json
reports/stage5/p2_vlm_attribution/<task>/c2_results.json
reports/stage5/p2_vlm_attribution/<task>/report.md
reports/stage5/p2_vlm_attribution/summary.md                       # cross-task gate dashboard
```

---

## Task 1: New `stage5` package marker

**Files:**
- Create: `babysteps/stage5/__init__.py`

- [ ] **Step 1: Create empty package init**

`babysteps/stage5/__init__.py`:
```python
"""Stage-5 ICLR track package.

P1 (vision encoder swap) lives historically in babysteps/stage4/vision_features.py.
P2 (VLM attribution) lives here. New Stage-5 code should land in this package.
"""
```

- [ ] **Step 2: Verify import works**

Run: `python -c "import babysteps.stage5; print(babysteps.stage5.__doc__.splitlines()[0])"`
Expected: `Stage-5 ICLR track package.`

- [ ] **Step 3: Commit**

```bash
git add babysteps/stage5/__init__.py
git commit -m "feat(stage5): create babysteps.stage5 package for P2 work

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Whitelist the two new revision operators

**Files:**
- Modify: `babysteps/schemas.py` (REVISION_OPERATORS frozenset)
- Test: `tests/test_schemas.py` (snapshot guard already covers new tokens via additive contract)

- [ ] **Step 1: Add operators to the whitelist**

In `babysteps/schemas.py`, locate the `REVISION_OPERATORS: frozenset[str]` definition. Add two new entries at the bottom of the set (preserve existing entries verbatim):

```python
    "latent_revision",
    # Stage-5 P2 — VLM-driven retry operators.
    # vlm_constrained_revision: VLM picks ONE factor name from the 6-factor
    #   menu; the existing discrete revision.revise_intent then performs the
    #   slot-local edit. Single-factor invariant preserved.
    # vlm_free_form_replan: VLM emits the full 6-factor revised intent JSON;
    #   we use it verbatim as the retry intent (after schema validation).
    #   May change any subset of factors — used as the C2 baseline only.
    "vlm_constrained_revision",
    "vlm_free_form_replan",
})
```

(Replace the closing `})` line of the existing set, not the whole set.)

- [ ] **Step 2: Run schema tests**

Run: `pytest tests/test_schemas.py -v`
Expected: PASS (additive change; existing snapshots unaffected).

- [ ] **Step 3: Add a positive test for the new operators**

Append to `tests/test_schemas.py`:
```python
def test_vlm_revision_operators_in_whitelist() -> None:
    """Stage-5 P2 operators are whitelisted (additive schema change)."""
    from babysteps.schemas import REVISION_OPERATORS, Revision
    assert "vlm_constrained_revision" in REVISION_OPERATORS
    assert "vlm_free_form_replan" in REVISION_OPERATORS
    # Round-trip: a Revision using each operator constructs cleanly.
    for op in ("vlm_constrained_revision", "vlm_free_form_replan"):
        r = Revision(
            operator=op, factor="approach_direction",
            old_value="from_minus_x", new_value="from_plus_x",
            frozen_factors=("goal_state", "object_motion"),
        )
        assert r.operator == op
```

- [ ] **Step 4: Run the new test**

Run: `pytest tests/test_schemas.py::test_vlm_revision_operators_in_whitelist -v`
Expected: PASS.

- [ ] **Step 5: Run full sim-free suite to confirm no regression**

Run: `pytest tests/ -x -q 2>&1 | tail -20`
Expected: all tests PASS (count should match the prior ~343 + 1 = ~344).

- [ ] **Step 6: Commit**

```bash
git add babysteps/schemas.py tests/test_schemas.py
git commit -m "feat(schemas): whitelist VLM revision operators for Stage-5 P2

Adds vlm_constrained_revision and vlm_free_form_replan to
REVISION_OPERATORS. Additive change — existing snapshots unaffected.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `babysteps/stage5/vlm_attribute.py` — the VLM module

**Files:**
- Create: `babysteps/stage5/vlm_attribute.py`
- Test: `tests/test_vlm_attribute.py`

This module is the only place that imports `transformers` and `torch.cuda`. Everything else is plain Python. It must run in two modes: **real** (loads InternVL3.5 on GPU) and **mock** (returns canned responses; used by tests and dry-runs).

- [ ] **Step 1: Write the failing test file (mock mode + parsing)**

`tests/test_vlm_attribute.py`:
```python
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
```

- [ ] **Step 2: Run test, verify it fails on import**

Run: `pytest tests/test_vlm_attribute.py -x -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'babysteps.stage5.vlm_attribute'`.

- [ ] **Step 3: Implement the module**

`babysteps/stage5/vlm_attribute.py`:
```python
"""Stage-5 P2 — VLM-based failure attribution + free-form replan baseline.

Loads InternVL3.5-8B (BF16) on a single A100-40GB and exposes two diagnosis
modes against a first-person failure-attempt frame:

* :meth:`InternVLClient.diagnose_constrained` (C1) — picks ONE intent factor
  name from the 6-factor menu (matches `babysteps.schemas.INTENT_FIELDS`).
  Output downstream feeds the existing discrete revision pipeline.
* :meth:`InternVLClient.diagnose_free_form` (C2) — emits the full revised
  intent JSON. Used verbatim as the retry intent (baseline only).

The module is `import`-safe on CPU/login nodes: the heavy `transformers`
import + GPU allocation happen inside :meth:`InternVLClient.load`, not at
import time. Tests use :class:`MockVLMClient` to avoid both.

Reference: HuggingFace model card for OpenGVLab/InternVL3_5-8B
(load_image dynamic tile preprocessing copied from the model card).
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
    """Return the first INTENT_FACTOR_NAMES token in raw, else None.

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
    # `\b` is fine here — factor names are alphanumeric + underscore.
    for name in INTENT_FACTOR_NAMES:
        if re.search(rf"\b{re.escape(name)}\b", raw):
            return name
    return None


def parse_free_form_output(raw: str) -> Optional[Intent]:
    """Parse a JSON intent. Returns None on any failure (missing keys, bad
    tokens, malformed JSON). Strips a single ```json …``` fence if present."""
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

        # The HF model card ships a `load_image` dynamic-tile preprocessor.
        # We re-implement the same logic here so we don't depend on
        # trust_remote_code helper modules being available.
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

    Returns a callable load_image(path, max_num=12) -> torch.Tensor
    (n_tiles, 3, 448, 448) BF16-castable.
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
```

- [ ] **Step 4: Run unit tests, verify they pass**

Run: `pytest tests/test_vlm_attribute.py -v`
Expected: All 11 tests PASS. (No GPU, no `transformers` import is triggered because we only exercise mock + parsers.)

- [ ] **Step 5: Run full sim-free suite to confirm no regression**

Run: `pytest tests/ -x -q 2>&1 | tail -10`
Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add babysteps/stage5/vlm_attribute.py tests/test_vlm_attribute.py
git commit -m "feat(stage5 p2): InternVL3.5 attribution module + mock client

Implements babysteps.stage5.vlm_attribute with two prompt templates
(C1 constrained factor-name, C2 free-form JSON), tolerant parsers,
and a sim-free MockVLMClient. Heavy transformers/torch imports happen
in InternVLClient.load(), keeping the module importable on the login
node. 11 sim-free unit tests cover prompts, parsers, and the mock.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Failure-frame renderer script

**Files:**
- Create: `scripts/stage5_p2_render_failure_frames.py`
- Create: `slurm/stage5_p2_render_failure_frames.sbatch`

We need ONE RGB PNG per held-out failure seed showing the first-person view at the end of the failed attempt (or initial scene, if `planner_failed=True` aborted immediately).

The pattern mirrors `scripts/stage5_render_demo_frames.py` but captures the **execute phase**, not the demo. We run the same scripted oracle skill compiler in the same `env_runner` to get the actual attempt rollout (or its abort point), then call `render_frame(env)` once at the end.

- [ ] **Step 1: Write the script**

`scripts/stage5_p2_render_failure_frames.py`:
```python
"""Stage-5 P2 — render one first-person failure-attempt frame per held-out seed.

For each seed in --seeds, re-run the standard Stage-0 episode (oracle demo →
initial_intent → executor scene with blocked_sides → run attempt) and save a
single PNG capturing the end-of-attempt RGB frame from the wrist camera. For
seeds where the planner aborts before rollout (planner_failed=True), the
saved frame is the post-reset initial scene (which still shows the obstacle).

Output: <out-dir>/seed_NNNN_attempt.png  (one per seed)

Also writes <out-dir>/episodes.jsonl mirroring the per-seed EpisodeRecord
(failure_packet + initial_intent + oracle_wrong_factor) so the eval script
has a single self-contained dataset directory.

Example::

    python scripts/stage5_p2_render_failure_frames.py \\
        --task PushCube-v1 --seeds 100-149 \\
        --out-dir datasets/stage5/p2_vlm/PushCube-v1/

GPU/Vulkan required.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from babysteps.episode import generate_proxy_demo, run_episode  # noqa: E402
from babysteps.envs.task_registry import get_task_entry  # noqa: E402
from babysteps.render.common import render_frame  # noqa: E402


def _parse_seed_range(s: str) -> list[int]:
    if "-" in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(s)]


def _save_png(path: Path, frame: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(frame.astype(np.uint8))
    img.save(path)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True,
                   choices=["PushCube-v1", "PickCube-v1", "StackCube-v1"])
    p.add_argument("--seeds", required=True,
                   help="Seed range A-B (inclusive) or single int.")
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args(argv)

    seeds = _parse_seed_range(args.seeds)
    entry = get_task_entry(args.task)
    adapter = entry.adapter_factory()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = args.out_dir / "episodes.jsonl"
    n_saved = 0
    n_failure = 0
    with episodes_path.open("w") as ef:
        for seed in seeds:
            # Reset, demo, intent, executor scene — same path as run_episode.
            env_runner = adapter.env_runner()
            scene_initial = env_runner.reset(seed)
            demo_evidence = generate_proxy_demo(env_runner, scene_initial, adapter)
            initial_intent = adapter.scripted_demo_to_intent(demo_evidence)
            scene_executor = replace(
                scene_initial,
                blocked_sides=adapter.default_blocked_factory(initial_intent),
            )
            # Re-reset to the executor scene so the attempt frame is rendered
            # from the same physical state as the real attempt.
            env_runner.reset(seed)
            # Run the attempt — env_runner.run drives the env internally;
            # after it returns, the env state is at the end-of-attempt (or
            # the abort point).
            attempt = env_runner.run(initial_intent, scene_executor)
            try:
                frame = render_frame(env_runner.env)
            except Exception as exc:
                print(f"WARN: render_frame failed for seed {seed}: {exc}",
                      file=sys.stderr)
                continue
            _save_png(args.out_dir / "frames" / f"seed_{seed:04d}_attempt.png",
                      frame)
            n_saved += 1

            # Build episode record (failure_packet + oracle wrong factor).
            from babysteps.failure import attribute_failure
            failure_packet = adapter.build_failure_packet(
                initial_intent, attempt, scene_executor,
            )
            oracle = adapter.oracle_wrong_factor(initial_intent, scene_executor)
            attribution = attribute_failure(failure_packet)
            is_failure = failure_packet.failure_predicate != "none"
            if is_failure:
                n_failure += 1
            row = {
                "seed": seed,
                "task": args.task,
                "frame_path": str(
                    args.out_dir / "frames" / f"seed_{seed:04d}_attempt.png"
                ),
                "initial_intent": initial_intent.to_dict(),
                "failure_predicate": failure_packet.failure_predicate,
                "oracle_wrong_factor": oracle,
                "rule_table_wrong_factor": attribution.wrong_factor,
                "is_failure": is_failure,
                "initial_success": bool(attempt.success),
            }
            ef.write(json.dumps(row, sort_keys=True) + "\n")
    adapter.close()
    print(f"saved {n_saved} frames ({n_failure} failures) → {args.out_dir}")
    print(f"wrote {episodes_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Write the sbatch wrapper**

`slurm/stage5_p2_render_failure_frames.sbatch`:
```bash
#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --job-name=stage5-p2-failure-frames
#SBATCH --output=slurm/logs/stage5-p2-failure-frames-%j.out
#SBATCH --error=slurm/logs/stage5-p2-failure-frames-%j.err
# Stage-5 P2 — render one first-person attempt frame per held-out seed,
# for each of the three eval tasks. Seeds 100-149 (50 per task → 150 total).

set -euo pipefail
cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

for TASK in PushCube-v1 PickCube-v1 StackCube-v1; do
    echo "=== $TASK ==="
    python scripts/stage5_p2_render_failure_frames.py \
        --task "$TASK" --seeds 100-149 \
        --out-dir "datasets/stage5/p2_vlm/$TASK"
done
echo
echo "=== JOB DONE === ($(date))"
```

- [ ] **Step 3: Sim-free smoke — dry-import the script**

Run: `python -c "import scripts.stage5_p2_render_failure_frames as m; print(m.__doc__.splitlines()[0])"`
Expected: `Stage-5 P2 — render one first-person failure-attempt frame per held-out seed.` (no GPU is initialized — imports are deferred).

- [ ] **Step 4: Commit (script + sbatch only — no run yet)**

```bash
git add scripts/stage5_p2_render_failure_frames.py slurm/stage5_p2_render_failure_frames.sbatch
git commit -m "feat(stage5 p2): failure-frame renderer for held-out seeds

Re-runs Stage-0 oracle pipeline on held-out seeds, captures end-of-attempt
RGB frame from wrist camera, saves PNG + per-seed JSONL with the failure
packet and oracle wrong-factor labels. A100-40gb sbatch covers PushCube /
PickCube / StackCube seeds 100-149.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 5: Submit the render job and wait for completion**

Run: `sbatch slurm/stage5_p2_render_failure_frames.sbatch`
Then poll `squeue -u $USER` until done (or use the harness's task notification).
Expected: 3 directories under `datasets/stage5/p2_vlm/` each containing `frames/seed_*.png` and `episodes.jsonl`.

- [ ] **Step 6: Sanity-check the output**

Run:
```bash
for T in PushCube-v1 PickCube-v1 StackCube-v1; do
  echo "=== $T ==="
  ls "datasets/stage5/p2_vlm/$T/frames/" | wc -l
  python -c "
import json
rows = [json.loads(l) for l in open('datasets/stage5/p2_vlm/$T/episodes.jsonl')]
n_fail = sum(r['is_failure'] for r in rows)
print(f'total={len(rows)} failures={n_fail}')
print('predicate distribution:', {})
from collections import Counter
print(Counter(r['failure_predicate'] for r in rows))
"
done
```
Expected: ~50 frames per task; ~45-50 failure rows per task (high failure rate by design).

---

## Task 5: VLM eval driver

**Files:**
- Create: `scripts/stage5_p2_vlm_eval.py`
- Create: `slurm/stage5_p2_vlm.sbatch`

This script runs C1 and C2 for one task. For each held-out failure episode, it:

1. Reads the cached failure frame + initial intent + oracle wrong factor from `datasets/stage5/p2_vlm/<task>/episodes.jsonl`.
2. Calls the VLM in C1 mode → factor name → applies discrete revision via `adapter.revise_intent(...)` → reruns `env_runner.run(revised, scene)` to get retry success.
3. Calls the VLM in C2 mode → full intent JSON → uses verbatim as retry intent → reruns `env_runner.run(revised, scene)` to get retry success.
4. Computes per-episode and per-condition metrics; writes `c1_results.json`, `c2_results.json`, `report.md`.

Also computes the rule-table baseline accuracy inline (no VLM needed) for the C1 ≥ rule-table gate.

- [ ] **Step 1: Write the eval driver**

`scripts/stage5_p2_vlm_eval.py`:
```python
"""Stage-5 P2 — VLM attribution + retry eval. Compares C1 (constrained
diagnosis + slot-local revision) against C2 (VLM free-form replan).

For each held-out failure episode (cached frame + failure_packet from
scripts/stage5_p2_render_failure_frames.py), runs both conditions through
the real env_runner retry mechanism and computes:

* attribution_accuracy (C1 only; C2 doesn't pick a factor)
* final_success_rate            (both)
* frozen_factor_preservation    (both; for C2, frozen = factors other than
                                  the oracle wrong factor)
* unnecessary_factor_change_rate (both)
* parse_failure_rate            (both; C1: factor name not in menu;
                                  C2: malformed JSON / invalid token)

Also computes the rule-table attribution accuracy on the same set for the
G_P2_acc gate (C1 acc >= rule-table acc).

Example::

    python scripts/stage5_p2_vlm_eval.py \\
        --task PushCube-v1 \\
        --episodes datasets/stage5/p2_vlm/PushCube-v1/episodes.jsonl \\
        --out-dir reports/stage5/p2_vlm_attribution/PushCube-v1/

Pass --mock for sim-free smoke (no GPU, no VLM call); --max-episodes N
to subset for debugging.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from babysteps.envs.task_registry import get_task_entry  # noqa: E402
from babysteps.failure import Attribution, attribute_failure  # noqa: E402
from babysteps.schemas import (  # noqa: E402
    INTENT_FIELDS, Intent, Revision,
)
from babysteps.stage5.vlm_attribute import (  # noqa: E402
    InternVLClient, MockVLMClient,
)


def _read_episodes(path: Path) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return [r for r in rows if r.get("is_failure", False)]


def _make_vlm_attribution(factor: str) -> Attribution:
    """Build an Attribution where VLM's factor is the wrong_factor."""
    return Attribution(
        semantic_failure=True,
        wrong_factor=factor,
        freeze=tuple(f for f in INTENT_FIELDS if f != factor),
        revise=(factor,),
    )


def _factors_changed(a: Intent, b: Intent) -> tuple[str, ...]:
    return tuple(f for f in INTENT_FIELDS if getattr(a, f) != getattr(b, f))


def _per_episode_c1(
    *, vlm_factor: Optional[str], oracle_factor: str,
    initial_intent: Intent, revised_intent: Optional[Intent],
    retry_success: Optional[bool], initial_success: bool,
) -> dict:
    """Compute C1 metrics for one episode."""
    if vlm_factor is None:
        return {
            "vlm_factor": None,
            "parse_failed": True,
            "attribution_correct": False,
            "factors_changed": [],
            "frozen_factor_preserved": None,
            "unnecessary_change": None,
            "final_success": bool(initial_success),
            "retry_success": None,
        }
    factors_changed = _factors_changed(initial_intent, revised_intent) \
        if revised_intent is not None else ()
    frozen_preserved = all(
        f == vlm_factor or f not in factors_changed for f in INTENT_FIELDS
    )
    # Unnecessary: any non-(VLM-factor) change is unnecessary by C1 contract.
    unnecessary = any(f != vlm_factor for f in factors_changed)
    final = bool(retry_success) if retry_success is not None else bool(initial_success)
    return {
        "vlm_factor": vlm_factor,
        "parse_failed": False,
        "attribution_correct": vlm_factor == oracle_factor,
        "factors_changed": list(factors_changed),
        "frozen_factor_preserved": frozen_preserved,
        "unnecessary_change": unnecessary,
        "final_success": final,
        "retry_success": retry_success,
    }


def _per_episode_c2(
    *, revised_intent: Optional[Intent], oracle_factor: str,
    initial_intent: Intent, retry_success: Optional[bool],
    initial_success: bool,
) -> dict:
    """Compute C2 metrics. For C2 there is no 'predicted factor' — instead
    we measure which factors changed vs the oracle-frozen set (all but the
    true wrong factor)."""
    if revised_intent is None:
        return {
            "parse_failed": True,
            "factors_changed": [],
            "frozen_factor_preserved": None,
            "unnecessary_change": None,
            "fixed_oracle_factor": None,
            "final_success": bool(initial_success),
            "retry_success": None,
        }
    factors_changed = _factors_changed(initial_intent, revised_intent)
    # Frozen-preserved (C2 sense): no factor OTHER than oracle_factor changed.
    frozen_preserved = all(
        f == oracle_factor or f not in factors_changed for f in INTENT_FIELDS
    )
    # Unnecessary: any factor change other than the oracle's wrong factor.
    unnecessary = any(f != oracle_factor for f in factors_changed)
    fixed_oracle = oracle_factor in factors_changed
    final = bool(retry_success) if retry_success is not None else bool(initial_success)
    return {
        "parse_failed": False,
        "factors_changed": list(factors_changed),
        "frozen_factor_preserved": frozen_preserved,
        "unnecessary_change": unnecessary,
        "fixed_oracle_factor": fixed_oracle,
        "final_success": final,
        "retry_success": retry_success,
    }


def _aggregate(rows: list[dict], keys: list[str]) -> dict:
    """Mean of each key, ignoring None entries."""
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None]
        out[k + "_rate"] = sum(bool(v) for v in vals) / len(vals) if vals else None
        out["n_" + k] = len(vals)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True,
                   choices=["PushCube-v1", "PickCube-v1", "StackCube-v1"])
    p.add_argument("--episodes", type=Path, required=True,
                   help="Path to episodes.jsonl from stage5_p2_render_failure_frames.py")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--mock", action="store_true",
                   help="Use MockVLMClient (no GPU, no transformers).")
    p.add_argument("--max-episodes", type=int, default=None,
                   help="Subset for debugging.")
    p.add_argument("--conditions", default="c1,c2",
                   help="Comma list: c1,c2 or just one.")
    args = p.parse_args(argv)

    episodes = _read_episodes(args.episodes)
    if args.max_episodes:
        episodes = episodes[: args.max_episodes]
    print(f"loaded {len(episodes)} failure episodes for {args.task}")

    # Adapter for retry rollouts.
    entry = get_task_entry(args.task)
    adapter = entry.adapter_factory()

    # VLM client.
    vlm = MockVLMClient() if args.mock else InternVLClient()
    if not args.mock:
        print("loading InternVL3.5-8B ...")
        vlm.load()
        print("loaded.")

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    c1_rows: list[dict] = []
    c2_rows: list[dict] = []
    rule_correct = 0
    rule_total = 0

    for ep in episodes:
        seed = ep["seed"]
        initial = Intent.from_dict(ep["initial_intent"])
        oracle_factor = ep["oracle_wrong_factor"]
        rule_factor = ep["rule_table_wrong_factor"]
        if rule_factor is not None:
            rule_total += 1
            if rule_factor == oracle_factor:
                rule_correct += 1
        # Rebuild executor scene for the retry rollout (deterministic seed).
        env_runner = adapter.env_runner()
        scene_initial = env_runner.reset(seed)
        scene_executor = replace(
            scene_initial,
            blocked_sides=adapter.default_blocked_factory(initial),
        )

        # ---------- C1: VLM constrained → discrete revision ---------- #
        if "c1" in conditions:
            vlm_factor = vlm.diagnose_constrained(
                image_path=ep["frame_path"],
                initial_intent=initial,
                failure_predicate=ep["failure_predicate"],
            )
            revised: Optional[Intent] = None
            retry_success: Optional[bool] = None
            if vlm_factor is not None:
                try:
                    attribution = _make_vlm_attribution(vlm_factor)
                    revised, _rev = adapter.revise_intent(
                        initial, attribution, scene_executor,
                    )
                    env_runner.reset(seed)
                    attempt = env_runner.run(revised, scene_executor)
                    retry_success = bool(attempt.success)
                except Exception as exc:
                    print(f"WARN: C1 retry exception seed {seed}: {exc}",
                          file=sys.stderr)
                    revised, retry_success = None, None
            row = _per_episode_c1(
                vlm_factor=vlm_factor, oracle_factor=oracle_factor,
                initial_intent=initial, revised_intent=revised,
                retry_success=retry_success, initial_success=ep["initial_success"],
            )
            row.update({"seed": seed, "oracle_wrong_factor": oracle_factor})
            c1_rows.append(row)
            print(f"  C1 seed={seed} vlm={vlm_factor!r:>22} "
                  f"oracle={oracle_factor!r:>22} "
                  f"retry_success={retry_success}")

        # ---------- C2: VLM free-form → verbatim retry ---------- #
        if "c2" in conditions:
            revised2 = vlm.diagnose_free_form(
                image_path=ep["frame_path"],
                initial_intent=initial,
                failure_predicate=ep["failure_predicate"],
            )
            retry_success2: Optional[bool] = None
            if revised2 is not None:
                try:
                    env_runner.reset(seed)
                    attempt2 = env_runner.run(revised2, scene_executor)
                    retry_success2 = bool(attempt2.success)
                except Exception as exc:
                    print(f"WARN: C2 retry exception seed {seed}: {exc}",
                          file=sys.stderr)
                    retry_success2 = None
            row2 = _per_episode_c2(
                revised_intent=revised2, oracle_factor=oracle_factor,
                initial_intent=initial, retry_success=retry_success2,
                initial_success=ep["initial_success"],
            )
            row2.update({"seed": seed, "oracle_wrong_factor": oracle_factor})
            c2_rows.append(row2)
            print(f"  C2 seed={seed} revised={revised2 is not None} "
                  f"retry_success={retry_success2}")

    adapter.close()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    rule_acc = rule_correct / rule_total if rule_total else None
    if "c1" in conditions:
        c1_summary = _aggregate(c1_rows, [
            "attribution_correct", "frozen_factor_preserved",
            "unnecessary_change", "final_success", "parse_failed",
        ])
        (args.out_dir / "c1_results.json").write_text(json.dumps({
            "task": args.task,
            "rule_table_accuracy": rule_acc,
            "n_episodes": len(c1_rows),
            "summary": c1_summary,
            "per_episode": c1_rows,
        }, indent=2, sort_keys=True) + "\n")
        print(f"\nC1 summary: {c1_summary}")

    if "c2" in conditions:
        c2_summary = _aggregate(c2_rows, [
            "frozen_factor_preserved", "unnecessary_change",
            "fixed_oracle_factor", "final_success", "parse_failed",
        ])
        (args.out_dir / "c2_results.json").write_text(json.dumps({
            "task": args.task,
            "n_episodes": len(c2_rows),
            "summary": c2_summary,
            "per_episode": c2_rows,
        }, indent=2, sort_keys=True) + "\n")
        print(f"C2 summary: {c2_summary}")

    _write_report_md(args.out_dir / "report.md", args.task, rule_acc,
                     c1_rows if "c1" in conditions else None,
                     c2_rows if "c2" in conditions else None)
    print(f"\nwrote {args.out_dir}/")
    return 0


def _write_report_md(out_path: Path, task: str, rule_acc: Optional[float],
                     c1_rows: Optional[list[dict]], c2_rows: Optional[list[dict]]) -> None:
    def rate(rows, key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return sum(bool(v) for v in vals) / len(vals) if vals else float("nan")
    lines = [f"# Stage-5 P2 VLM attribution — {task}", ""]
    if rule_acc is not None:
        lines.append(f"- Rule-table attribution accuracy (baseline): **{rule_acc:.3f}**")
        lines.append("")
    if c1_rows is not None:
        lines.extend([
            "## C1 — VLM-constrained diagnosis + slot-local revision (ours)",
            "",
            f"- n_episodes: {len(c1_rows)}",
            f"- attribution_accuracy: **{rate(c1_rows, 'attribution_correct'):.3f}**",
            f"- frozen_factor_preservation: **{rate(c1_rows, 'frozen_factor_preserved'):.3f}**",
            f"- unnecessary_factor_change: **{rate(c1_rows, 'unnecessary_change'):.3f}**",
            f"- final_success_rate: **{rate(c1_rows, 'final_success'):.3f}**",
            f"- parse_failure_rate: {rate(c1_rows, 'parse_failed'):.3f}",
            "",
        ])
    if c2_rows is not None:
        lines.extend([
            "## C2 — VLM free-form replan (baseline)",
            "",
            f"- n_episodes: {len(c2_rows)}",
            f"- frozen_factor_preservation: **{rate(c2_rows, 'frozen_factor_preserved'):.3f}**",
            f"- unnecessary_factor_change: **{rate(c2_rows, 'unnecessary_change'):.3f}**",
            f"- fixed_oracle_factor_rate: {rate(c2_rows, 'fixed_oracle_factor'):.3f}",
            f"- final_success_rate: **{rate(c2_rows, 'final_success'):.3f}**",
            f"- parse_failure_rate: {rate(c2_rows, 'parse_failed'):.3f}",
            "",
        ])
    if c1_rows is not None and c2_rows is not None:
        d_pres = (rate(c1_rows, "frozen_factor_preserved")
                   - rate(c2_rows, "frozen_factor_preserved")) * 100
        d_succ = (rate(c1_rows, "final_success")
                   - rate(c2_rows, "final_success")) * 100
        lines.extend([
            "## Gates",
            "",
            f"- **C1 attribution ≥ rule-table**: "
            f"{rate(c1_rows, 'attribution_correct'):.3f} vs {rule_acc:.3f} → "
            f"{'PASS' if rate(c1_rows, 'attribution_correct') >= rule_acc else 'FAIL'}",
            f"- **C1 preservation ≫ C2 preservation** "
            f"(Δ = {d_pres:+.1f}pp; PASS if Δ > 0)",
            f"- **C1 success ≥ C2 success within 5pp** "
            f"(Δ = {d_succ:+.1f}pp; PASS if Δ ≥ -5)",
            "",
        ])
    out_path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Write the sbatch**

`slurm/stage5_p2_vlm.sbatch`:
```bash
#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=01:30:00
#SBATCH --job-name=stage5-p2-vlm
#SBATCH --output=slurm/logs/stage5-p2-vlm-%j.out
#SBATCH --error=slurm/logs/stage5-p2-vlm-%j.err
# Stage-5 P2 — VLM attribution + retry, C1 vs C2, three tasks.
#
# Loads InternVL3.5-8B (BF16, ~17 GB) once per task invocation. ~150 episodes
# x 2 conditions x ~5s/call x 3 tasks ≈ ~75 min wall. Reuses cached failure
# frames from datasets/stage5/p2_vlm/<task>/.

set -euo pipefail
cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${HF_HOME:-/scratch/gilbreth/wang4433/hf_cache}"

for TASK in PushCube-v1 PickCube-v1 StackCube-v1; do
    echo "=== $TASK ==="
    python scripts/stage5_p2_vlm_eval.py \
        --task "$TASK" \
        --episodes "datasets/stage5/p2_vlm/$TASK/episodes.jsonl" \
        --out-dir "reports/stage5/p2_vlm_attribution/$TASK"
done

echo
echo "=== JOB DONE === ($(date))"
```

- [ ] **Step 3: Sim-free smoke — mock mode, 2 episodes**

We can't run the full mock-mode pipeline without GPU/Vulkan (the retry rollout calls `env_runner.run`). The script-level smoke is just an import check:

Run: `python -c "import scripts.stage5_p2_vlm_eval as m; print(m.__doc__.splitlines()[0])"`
Expected: `Stage-5 P2 — VLM attribution + retry eval. Compares C1 (constrained` (no GPU init).

- [ ] **Step 4: Commit (script + sbatch only)**

```bash
git add scripts/stage5_p2_vlm_eval.py slurm/stage5_p2_vlm.sbatch
git commit -m "feat(stage5 p2): VLM eval driver — C1 vs C2 on held-out failures

scripts/stage5_p2_vlm_eval.py iterates cached failure episodes, calls
InternVL3.5-8B in both constrained (factor-name) and free-form (JSON)
modes, runs the retry rollout through env_runner, and reports
attribution accuracy, frozen-factor preservation, unnecessary-change,
final-success, and three pass/fail gates per task.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 5: Submit the VLM eval job**

(Only after Task 4 completed and frames exist on disk.)

Run: `sbatch slurm/stage5_p2_vlm.sbatch`
Then poll `squeue -u $USER`.
Expected: 3 directories under `reports/stage5/p2_vlm_attribution/` each containing `c1_results.json`, `c2_results.json`, `report.md`.

- [ ] **Step 6: Inspect per-task reports**

Run:
```bash
for T in PushCube-v1 PickCube-v1 StackCube-v1; do
  echo "============ $T ============"
  cat "reports/stage5/p2_vlm_attribution/$T/report.md"
done
```

- [ ] **Step 7: Commit the result artifacts**

```bash
git add reports/stage5/p2_vlm_attribution/ datasets/stage5/p2_vlm/*/episodes.jsonl
# Frames are LARGE PNGs — do NOT commit unless small (<5 MB each).
git commit -m "data(stage5 p2): VLM attribution eval results (3 tasks)

Per-task report cards under reports/stage5/p2_vlm_attribution/.
Frames excluded from VCS; episodes.jsonl is the reproducibility entry.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Cross-task summary report

**Files:**
- Create: `scripts/stage5_p2_summary.py`

Aggregates the three per-task reports into a single dashboard for the paper.

- [ ] **Step 1: Write the summary script**

`scripts/stage5_p2_summary.py`:
```python
"""Stage-5 P2 — cross-task summary of VLM attribution results.

Reads reports/stage5/p2_vlm_attribution/<task>/{c1_results.json,c2_results.json}
and emits reports/stage5/p2_vlm_attribution/summary.{json,md}.

Usage::
    python scripts/stage5_p2_summary.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
REPORT_ROOT = _ROOT / "reports" / "stage5" / "p2_vlm_attribution"
TASKS = ("PushCube-v1", "PickCube-v1", "StackCube-v1")


def main(argv: list[str] | None = None) -> int:
    rows = []
    for task in TASKS:
        td = REPORT_ROOT / task
        if not (td / "c1_results.json").exists():
            print(f"SKIP {task}: no c1_results.json", file=sys.stderr)
            continue
        c1 = json.loads((td / "c1_results.json").read_text())
        c2 = json.loads((td / "c2_results.json").read_text())
        rows.append({"task": task, "c1": c1, "c2": c2})
    summary = {"per_task": []}
    for r in rows:
        c1s, c2s = r["c1"]["summary"], r["c2"]["summary"]
        c1_acc = c1s.get("attribution_correct_rate")
        rule_acc = r["c1"].get("rule_table_accuracy")
        c1_pres = c1s.get("frozen_factor_preserved_rate")
        c2_pres = c2s.get("frozen_factor_preserved_rate")
        c1_succ = c1s.get("final_success_rate")
        c2_succ = c2s.get("final_success_rate")
        summary["per_task"].append({
            "task": r["task"],
            "c1_attribution_acc": c1_acc,
            "rule_table_acc": rule_acc,
            "c1_frozen_preservation": c1_pres,
            "c2_frozen_preservation": c2_pres,
            "c1_final_success": c1_succ,
            "c2_final_success": c2_succ,
            "delta_pres_pp": (c1_pres - c2_pres) * 100
                if (c1_pres is not None and c2_pres is not None) else None,
            "delta_success_pp": (c1_succ - c2_succ) * 100
                if (c1_succ is not None and c2_succ is not None) else None,
        })
    (REPORT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # Markdown
    lines = ["# Stage-5 P2 VLM Attribution — Cross-task Summary", ""]
    lines.append("| task | C1 attr acc | rule-table | C1 pres | C2 pres | Δpres pp | C1 succ | C2 succ | Δsucc pp |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in summary["per_task"]:
        def fmt(v):
            return f"{v:.3f}" if isinstance(v, float) else "—"
        def fpp(v):
            return f"{v:+.1f}" if isinstance(v, float) else "—"
        lines.append("| " + " | ".join([
            r["task"], fmt(r["c1_attribution_acc"]), fmt(r["rule_table_acc"]),
            fmt(r["c1_frozen_preservation"]), fmt(r["c2_frozen_preservation"]),
            fpp(r["delta_pres_pp"]),
            fmt(r["c1_final_success"]), fmt(r["c2_final_success"]),
            fpp(r["delta_success_pp"]),
        ]) + " |")
    lines.append("")
    lines.append("## Gates")
    lines.append("")
    for r in summary["per_task"]:
        gate_acc = (r["c1_attribution_acc"] is not None
                    and r["rule_table_acc"] is not None
                    and r["c1_attribution_acc"] >= r["rule_table_acc"])
        gate_pres = (r["delta_pres_pp"] is not None and r["delta_pres_pp"] > 0)
        gate_succ = (r["delta_success_pp"] is not None
                     and r["delta_success_pp"] >= -5)
        lines.append(f"- **{r['task']}**: "
                     f"attr {'PASS' if gate_acc else 'FAIL'} · "
                     f"pres {'PASS' if gate_pres else 'FAIL'} · "
                     f"succ {'PASS' if gate_succ else 'FAIL'}")
    (REPORT_ROOT / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {REPORT_ROOT}/summary.{{json,md}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run the summary**

Run: `python scripts/stage5_p2_summary.py`
Expected: `reports/stage5/p2_vlm_attribution/summary.{json,md}` written.

- [ ] **Step 3: Inspect**

Run: `cat reports/stage5/p2_vlm_attribution/summary.md`

- [ ] **Step 4: Commit**

```bash
git add scripts/stage5_p2_summary.py reports/stage5/p2_vlm_attribution/summary.*
git commit -m "feat(stage5 p2): cross-task summary report

Aggregates per-task C1/C2 results into reports/stage5/p2_vlm_attribution/
summary.{json,md} with the three pass/fail gates per task.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Record results in goal.md and slurm/CLAUDE.md

**Files:**
- Modify: `slurm/CLAUDE.md`  (add a "Recorded gate results" subsection for P2)
- Modify: `goal.md`  (tick P2 done note IF gates pass; otherwise add a "P2 status: partial — gate X failed because Y" note)

- [ ] **Step 1: Update slurm/CLAUDE.md with the gate numbers**

Append a new section after the existing CrossViewPush entry. Use exact numbers from `reports/stage5/p2_vlm_attribution/summary.md`. Template:

```markdown
### Stage-5 P2 — VLM attribution (job <JOBID>, 2026-05-25)

InternVL3.5-8B, ~150 held-out failure episodes across 3 tasks.

| task | C1 attr acc | rule-table | C1 pres | C2 pres | Δpres pp | C1 succ | C2 succ | Δsucc pp |
|---|---|---|---|---|---|---|---|---|
| PushCube-v1 | ... | ... | ... | ... | ... | ... | ... | ... |
| PickCube-v1 | ... | ... | ... | ... | ... | ... | ... | ... |
| StackCube-v1 | ... | ... | ... | ... | ... | ... | ... | ... |

Gates: <pass/fail per task>.
```

- [ ] **Step 2: Update goal.md Stage-5 P2 status**

Locate the Stage-5 P2 section in `goal.md` and append a status note matching the template used by P1:

```markdown
> **Status (2026-05-25):** P2 done on PushCube/PickCube/StackCube with
> InternVL3.5-8B. Gates: <pass/fail summary>. See
> `reports/stage5/p2_vlm_attribution/summary.md`.
```

- [ ] **Step 3: Commit**

```bash
git add slurm/CLAUDE.md goal.md
git commit -m "docs(stage5 p2): record VLM attribution gate results

Updates slurm/CLAUDE.md with the recorded numbers and goal.md with the
P2 status line. Mirrors the P1 / CrossViewPush recording convention.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Optional — design spec snapshot

Project convention: every sub-project has a paired design spec under `docs/superpowers/specs/`. This step captures the decisions made above so a future agent can recover context.

**Files:**
- Create: `docs/superpowers/specs/2026-05-25-stage5-p2-vlm-attribution-design.md`

- [ ] **Step 1: Write the spec**

Mirror the structure of an existing spec (e.g. find one with `ls docs/superpowers/specs/`). Cover: goal, two conditions, two prompt templates, the 4 metrics, the 3 gates, the architectural decisions table from this plan, the file inventory, and the dataset/model artifacts.

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-05-25-stage5-p2-vlm-attribution-design.md
git commit -m "docs(stage5 p2): design spec snapshot

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Self-review (run after writing)

**Spec coverage:**
- ✅ Two prompt templates (C1 + C2) → Task 3 build_*_prompt functions
- ✅ Output parsers tolerant of formatting drift → Task 3 parse_*_output + unit tests
- ✅ Mock mode for sim-free unit tests → Task 3 MockVLMClient
- ✅ Failure-frame source → Task 4 render_failure_frames script
- ✅ Per-episode C1 + C2 + rule-table → Task 5 eval driver
- ✅ Four metrics (attribution acc, final success, frozen preservation, unnecessary change) → Task 5 _per_episode_c1/c2 + _aggregate
- ✅ Three gates → Task 5 _write_report_md + Task 6 summary
- ✅ A100-40gb sbatch with standby QoS → Tasks 4 & 5
- ✅ `reports/stage5/p2_vlm_attribution/` output → Task 5 out-dir
- ✅ `babysteps/stage5/vlm_attribute.py` location → Task 1 + 3
- ✅ Two new revision operators → Task 2

**Placeholder scan:** No TBD, no "implement later", no "similar to Task N", all test code complete.

**Type consistency:** `MockVLMClient.diagnose_constrained` / `diagnose_free_form` signatures match `InternVLClient`'s. `_make_vlm_attribution` returns `babysteps.failure.Attribution` (verified in source). `adapter.revise_intent(initial, attribution, scene)` signature matches the policies module's usage of `ctx.revise_fn`.

**Non-goals preserved:**
- No VLM fine-tuning (frozen weights, BF16 inference only).
- VLM never produces a revised value in C1 (only the factor name; ReviseHead-equivalent discrete revision performs the edit).
- No multi-turn VLM; single-shot per episode.
- Stage-0 schema additive (operators only, no removals).

---

**Plan complete and saved to `docs/superpowers/plans/2026-05-25-stage5-p2-vlm-attribution.md`.**
