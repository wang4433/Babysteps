# Stage-0 Procedural Baseline Table — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 7-method × 3-task procedural baseline comparison table that isolates *selective factor revision* as the variable behind failure recovery.

**Architecture:** Inject a `RetryContext`-based retry policy into `run_episode` (default = selective, snapshot-stable). Seven sim-free policy functions vary only the change-set. Two new label-based metrics (`harmful_revision_rate`, `correct_factor_fixed`) and a comparison-table aggregator are added to `eval.py`. A sweep runner drives the real-sim sweep on GPU; all policy/metric logic is unit-tested against the fake env.

**Tech Stack:** Python 3, dataclasses, pytest, NumPy. No new third-party deps.

**Spec:** `docs/superpowers/specs/2026-05-20-stage0-baselines-design.md`.

**Conventions (read once):**
- Run the suite with `python -m pytest tests/ -q` (sim-free, ~1.3s, no GPU).
- TDD: failing test first, minimal impl, green, commit.
- Commit message trailer: `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`.
- Snapshot guard: `tests/test_*_adapter.py` and `tests/test_crossview.py` compare `run_episode(...).to_jsonl_line()` byte-for-byte against `tests/snapshots/*.jsonl`. Default `run_episode` behavior MUST stay byte-identical.

---

### Task 1: Add baseline revision-operator tokens to the schema whitelist

**Files:**
- Modify: `babysteps/schemas.py:79-86` (`REVISION_OPERATORS`)
- Test: `tests/test_schemas.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_schemas.py`:

```python
def test_baseline_revision_operators_whitelisted():
    from babysteps.schemas import REVISION_OPERATORS
    for op in (
        "same_intent_retry",
        "random_factor_revision",
        "text_feedback_replan",
        "full_replan_analogue",
    ):
        assert op in REVISION_OPERATORS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_schemas.py::test_baseline_revision_operators_whitelisted -v`
Expected: FAIL (`assert 'same_intent_retry' in REVISION_OPERATORS`).

- [ ] **Step 3: Add the tokens (additive)**

In `babysteps/schemas.py`, extend `REVISION_OPERATORS`:

```python
REVISION_OPERATORS: frozenset[str] = frozenset({
    "approach_substitution",
    "contact_substitution",            # B: PickCube — rotate gripper axis
    "goal_refinement",                 # C: StackCube — sharpen under-specified goal
    "constraint_introduction",         # D: deprecated, kept in whitelist
    "embodiment_substitution",         # D: NEW — swap grasp_turn → poke_turn
    "grounding_substitution",          # E: cross-view — swap actor_frame → observer_frame
    # M3 baselines — procedural retry policies (not single-factor operators):
    "same_intent_retry",
    "random_factor_revision",
    "text_feedback_replan",
    "full_replan_analogue",
})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_schemas.py -q`
Expected: PASS (existing snapshots unaffected — adding to a frozenset changes no serialized bytes).

- [ ] **Step 5: Commit**

```bash
git add babysteps/schemas.py tests/test_schemas.py
git commit -m "feat(baselines): whitelist 4 procedural retry-policy operators

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `task_valid_tokens` adapter hook + per-task token sets

Per the spec, resampling draws from **task-valid** alternative tokens (not global whitelists). The base class returns `{}` (no editable factors); the three main-table adapters override.

**Files:**
- Modify: `babysteps/envs/task_adapter.py` (add method to `BaseTaskAdapter`)
- Modify: `babysteps/envs/pushcube_adapter.py`, `pickcube_adapter.py`, `stackcube_adapter.py`
- Test: `tests/test_task_adapter.py`, `tests/test_pushcube_adapter.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_task_adapter.py`:

```python
def test_base_task_valid_tokens_defaults_empty():
    from babysteps.envs.task_adapter import BaseTaskAdapter

    class _Stub(BaseTaskAdapter):
        task_id = "PushCube-v1"
        def make_env_runner(self): raise NotImplementedError
        def oracle_correct_intent(self, scene): raise NotImplementedError
        def default_blocked_factory(self, intent): return ()
        def oracle_wrong_factor(self, i, s): return "none"
        def scripted_demo_to_intent(self, e): raise NotImplementedError
        def compile_skill(self, i, s): return None

    assert _Stub().task_valid_tokens() == {}
```

Append to `tests/test_pushcube_adapter.py`:

```python
def test_pushcube_task_valid_tokens():
    from babysteps.envs.pushcube_adapter import PushCubeAdapter
    toks = PushCubeAdapter().task_valid_tokens()
    # PushCube edits approach_direction and contact_region only.
    assert set(toks) == {"approach_direction", "contact_region"}
    assert set(toks["contact_region"]) == {
        "minus_x_face", "plus_x_face", "minus_y_face", "plus_y_face",
    }
    assert set(toks["approach_direction"]) == {
        "from_minus_x", "from_plus_x", "from_minus_y", "from_plus_y",
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_task_adapter.py::test_base_task_valid_tokens_defaults_empty tests/test_pushcube_adapter.py::test_pushcube_task_valid_tokens -v`
Expected: FAIL (`AttributeError: 'PushCubeAdapter' object has no attribute 'task_valid_tokens'`).

- [ ] **Step 3: Add the base hook**

In `babysteps/envs/task_adapter.py`, add to `BaseTaskAdapter` (after `observe_demo`):

```python
    def task_valid_tokens(self) -> dict[str, tuple[str, ...]]:
        """Per-factor task-valid alternative tokens for baseline resampling.

        Keys are the *task-editable* factors (those with >1 plausible token
        for this task). Values are the plausible tokens. Default: no editable
        factors. The three main-table adapters override. Used only by the M3
        baseline policies; the selective loop never calls this."""
        return {}
```

- [ ] **Step 4: Override in the three adapters**

In `babysteps/envs/pushcube_adapter.py`, add to `PushCubeAdapter`:

```python
    def task_valid_tokens(self) -> dict[str, tuple[str, ...]]:
        return {
            "approach_direction": (
                "from_minus_x", "from_plus_x", "from_minus_y", "from_plus_y",
            ),
            "contact_region": (
                "minus_x_face", "plus_x_face", "minus_y_face", "plus_y_face",
            ),
        }
```

In `babysteps/envs/pickcube_adapter.py`, add to `PickCubeAdapter`:

```python
    def task_valid_tokens(self) -> dict[str, tuple[str, ...]]:
        # PickCube's controlled fault is a slip-prone contact face.
        return {
            "contact_region": (
                "minus_x_face", "plus_x_face", "minus_y_face", "plus_y_face",
            ),
        }
```

In `babysteps/envs/stackcube_adapter.py`, add to `StackCubeAdapter`:

```python
    def task_valid_tokens(self) -> dict[str, tuple[str, ...]]:
        return {
            "goal_state": ("cube_at_target", "cubeA_on_cubeB"),
            "contact_region": (
                "minus_x_face", "plus_x_face", "minus_y_face", "plus_y_face",
            ),
        }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_task_adapter.py tests/test_pushcube_adapter.py tests/test_pickcube_adapter.py tests/test_stackcube_adapter.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add babysteps/envs/task_adapter.py babysteps/envs/pushcube_adapter.py babysteps/envs/pickcube_adapter.py babysteps/envs/stackcube_adapter.py tests/test_task_adapter.py tests/test_pushcube_adapter.py
git commit -m "feat(baselines): task_valid_tokens hook + 3 main-table token sets

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `RetryContext` + `resample_factor` in a new sim-free module

**Files:**
- Create: `babysteps/policies.py`
- Test: `tests/test_policies.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_policies.py`:

```python
import random
from babysteps.policies import resample_factor
from babysteps.schemas import Intent

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_policies.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'babysteps.policies'`).

- [ ] **Step 3: Create the module with `RetryContext` + `resample_factor`**

Create `babysteps/policies.py`:

```python
"""Stage-0 procedural retry policies for the M3 baseline table.

Each policy is a pure function `(RetryContext) -> Optional[(Intent, Revision)]`.
Returning None means "no retry" (the one_shot baseline). All policies are
sim-free and import no simulator. See
docs/superpowers/specs/2026-05-20-stage0-baselines-design.md.

These are DETERMINISTIC PROCEDURAL ANALOGUES of replanning, not LLM/VLM
planners. full_replan_analogue / text_feedback_replan must always be reported
as procedural analogues.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import Callable, Mapping, Optional

from babysteps.failure import Attribution
from babysteps.schemas import INTENT_FIELDS, Intent, Revision, SceneState


@dataclass(frozen=True)
class RetryContext:
    """Everything a retry policy needs. Built once per episode by run_episode."""

    initial_intent: Intent
    attribution: Attribution
    scene: SceneState
    oracle_correct_intent: Intent
    oracle_wrong_factor: str
    task_valid_tokens: Mapping[str, tuple[str, ...]]
    rng: random.Random
    # adapter.revise_intent, bound — used by selective/oracle policies so this
    # module never imports the adapter (avoids an import cycle).
    revise_fn: Callable[[Intent, Attribution, SceneState], tuple[Intent, Revision]]


def resample_factor(
    intent: Intent,
    factor: str,
    valid_tokens: tuple[str, ...],
    rng: random.Random,
) -> str:
    """Return a task-valid token for `factor` other than its current value.

    Excludes the CURRENT value only (not the oracle value) — see spec §2:
    this lets random_factor_revision occasionally land on the correct value,
    while extra perturbations of already-correct factors are necessarily wrong.
    """
    current = getattr(intent, factor)
    alternatives = [t for t in valid_tokens if t != current]
    if not alternatives:
        raise ValueError(
            f"resample_factor: no task-valid alternative for {factor!r} "
            f"(current={current!r}, tokens={valid_tokens!r})"
        )
    return rng.choice(alternatives)


RetryPolicy = Callable[[RetryContext], Optional[tuple[Intent, Revision]]]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_policies.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add babysteps/policies.py tests/test_policies.py
git commit -m "feat(baselines): policies module with RetryContext + resample_factor

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: The two non-perturbing policies — `one_shot`, `same_intent_retry`

**Files:**
- Modify: `babysteps/policies.py`
- Test: `tests/test_policies.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_policies.py`:

```python
from babysteps.failure import Attribution
from babysteps.policies import RetryContext, one_shot, same_intent_retry
from babysteps.schemas import SceneState

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
```

(Note: `from dataclasses import replace` and `from babysteps.schemas import INTENT_FIELDS` are already imported at top of the test file from Task 3; add them if missing.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_policies.py::test_one_shot_returns_none tests/test_policies.py::test_same_intent_retry_keeps_intent_unchanged -v`
Expected: FAIL (`ImportError: cannot import name 'one_shot'`).

- [ ] **Step 3: Implement the two policies**

Append to `babysteps/policies.py`:

```python
def one_shot(ctx: RetryContext) -> Optional[tuple[Intent, Revision]]:
    """No retry — the lower-bound baseline."""
    return None


def same_intent_retry(ctx: RetryContext) -> Optional[tuple[Intent, Revision]]:
    """Retry the identical intent (a fresh rollout may recover by luck)."""
    rev = Revision(
        operator="same_intent_retry",
        factor="none",
        old_value="",
        new_value="",
        frozen_factors=INTENT_FIELDS,
    )
    return ctx.initial_intent, rev
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_policies.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add babysteps/policies.py tests/test_policies.py
git commit -m "feat(baselines): one_shot + same_intent_retry policies

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Attribution-using policies — `babysteps_selective`, `oracle_factor_revision`

**Files:**
- Modify: `babysteps/policies.py`
- Test: `tests/test_policies.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_policies.py`:

```python
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


def test_oracle_revises_ground_truth_factor():
    # Even if attribution is wrong, oracle uses oracle_wrong_factor.
    wrong_attr = Attribution(True, "approach_direction", (), ("approach_direction",))
    out = oracle_factor_revision(_real_revise_ctx(
        attribution=wrong_attr, oracle_wrong_factor="contact_region"))
    assert out is not None
    revised, rev = out
    assert rev.factor == "contact_region"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_policies.py::test_selective_revises_attributed_factor tests/test_policies.py::test_oracle_revises_ground_truth_factor -v`
Expected: FAIL (`ImportError: cannot import name 'babysteps_selective'`).

- [ ] **Step 3: Implement the two policies**

Append to `babysteps/policies.py`:

```python
def babysteps_selective(ctx: RetryContext) -> Optional[tuple[Intent, Revision]]:
    """Ours: revise only the attributed implicated factor."""
    return ctx.revise_fn(ctx.initial_intent, ctx.attribution, ctx.scene)


def oracle_factor_revision(ctx: RetryContext) -> Optional[tuple[Intent, Revision]]:
    """Upper bound: revise the ground-truth wrong factor."""
    factor = ctx.oracle_wrong_factor
    oracle_attr = Attribution(
        semantic_failure=True,
        wrong_factor=factor,
        freeze=tuple(f for f in INTENT_FIELDS if f != factor),
        revise=(factor,),
    )
    return ctx.revise_fn(ctx.initial_intent, oracle_attr, ctx.scene)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_policies.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add babysteps/policies.py tests/test_policies.py
git commit -m "feat(baselines): babysteps_selective + oracle_factor_revision policies

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Perturbing policies — `random_factor_revision`, `text_feedback_replan`, `full_replan_analogue`

These share a helper that applies the correct fix then resamples extra factors.

**Files:**
- Modify: `babysteps/policies.py`
- Test: `tests/test_policies.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_policies.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_policies.py -k "random_factor or full_replan or text_feedback" -v`
Expected: FAIL (`ImportError: cannot import name 'random_factor_revision'`).

- [ ] **Step 3: Implement the three policies + shared helper**

Append to `babysteps/policies.py`:

```python
def _editable_factors(ctx: RetryContext) -> tuple[str, ...]:
    """Task-editable factors (those with a task-valid token set)."""
    return tuple(f for f in INTENT_FIELDS if f in ctx.task_valid_tokens)


def _perturb(
    intent: Intent, factors: tuple[str, ...], ctx: RetryContext,
) -> Intent:
    """Resample each factor in `factors` to a task-valid non-current token."""
    out = intent
    for f in factors:
        new = resample_factor(out, f, tuple(ctx.task_valid_tokens[f]), ctx.rng)
        out = replace(out, **{f: new})
    return out


def _frozen_against_ground_truth(ctx: RetryContext) -> tuple[str, ...]:
    """The factors that SHOULD be preserved = all but the true wrong factor."""
    return tuple(f for f in INTENT_FIELDS if f != ctx.oracle_wrong_factor)


def random_factor_revision(ctx: RetryContext) -> Optional[tuple[Intent, Revision]]:
    """Ignore attribution: resample one random editable factor."""
    editable = _editable_factors(ctx)
    factor = ctx.rng.choice(editable)
    old = getattr(ctx.initial_intent, factor)
    revised = _perturb(ctx.initial_intent, (factor,), ctx)
    rev = Revision(
        operator="random_factor_revision",
        factor=factor,
        old_value=old,
        new_value=getattr(revised, factor),
        frozen_factors=_frozen_against_ground_truth(ctx),
    )
    return revised, rev


def text_feedback_replan(ctx: RetryContext) -> Optional[tuple[Intent, Revision]]:
    """Fix implicated correctly, then perturb its sibling editable factors
    (attribution.revise minus the implicated factor)."""
    fixed, _ = ctx.revise_fn(ctx.initial_intent, ctx.attribution, ctx.scene)
    siblings = tuple(
        f for f in ctx.attribution.revise
        if f != ctx.attribution.wrong_factor and f in ctx.task_valid_tokens
    )
    revised = _perturb(fixed, siblings, ctx)
    rev = Revision(
        operator="text_feedback_replan",
        factor=ctx.attribution.wrong_factor or "none",
        old_value=getattr(ctx.initial_intent, ctx.attribution.wrong_factor),
        new_value=getattr(revised, ctx.attribution.wrong_factor),
        frozen_factors=_frozen_against_ground_truth(ctx),
    )
    return revised, rev


def full_replan_analogue(ctx: RetryContext) -> Optional[tuple[Intent, Revision]]:
    """Fix implicated correctly, then perturb ALL other editable factors."""
    fixed, _ = ctx.revise_fn(ctx.initial_intent, ctx.attribution, ctx.scene)
    others = tuple(
        f for f in _editable_factors(ctx) if f != ctx.attribution.wrong_factor
    )
    revised = _perturb(fixed, others, ctx)
    rev = Revision(
        operator="full_replan_analogue",
        factor=ctx.attribution.wrong_factor or "none",
        old_value=getattr(ctx.initial_intent, ctx.attribution.wrong_factor),
        new_value=getattr(revised, ctx.attribution.wrong_factor),
        frozen_factors=_frozen_against_ground_truth(ctx),
    )
    return revised, rev


POLICIES: dict[str, RetryPolicy] = {
    "one_shot": one_shot,
    "same_intent_retry": same_intent_retry,
    "random_factor_revision": random_factor_revision,
    "babysteps_selective": babysteps_selective,
    "text_feedback_replan": text_feedback_replan,
    "full_replan_analogue": full_replan_analogue,
    "oracle_factor_revision": oracle_factor_revision,
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_policies.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add babysteps/policies.py tests/test_policies.py
git commit -m "feat(baselines): random/text_feedback/full_replan policies + POLICIES registry

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Inject the policy into `run_episode` (default selective, snapshot-stable)

**Files:**
- Modify: `babysteps/episode.py`
- Test: `tests/test_episode.py`, then re-run snapshot tests

- [ ] **Step 1: Write the failing test**

Append to `tests/test_episode.py`:

```python
def test_run_episode_one_shot_policy_has_no_retry(fake_env_runner):
    from babysteps.policies import one_shot
    from tests.test_episode import _StubAdapter  # reuse this file's stub
    adapter = _StubAdapter(fake_env_runner)
    rec = run_episode(
        episode_id="t", seed=1, adapter=adapter, policy=one_shot)
    assert rec.retry is None
    assert rec.revision is None


def test_run_episode_default_policy_is_selective(fake_env_runner):
    import inspect
    params = inspect.signature(run_episode).parameters
    assert "policy" in params
```

(If `_StubAdapter` is named differently in `tests/test_episode.py`, use the existing stub adapter from this file — see `test_run_episode_blocked_then_retry_success`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_episode.py::test_run_episode_default_policy_is_selective -v`
Expected: FAIL (`assert 'policy' in params`).

- [ ] **Step 3: Refactor `run_episode` to call a policy**

In `babysteps/episode.py`:

(a) Add imports near the top:

```python
from babysteps.policies import RetryContext, RetryPolicy, babysteps_selective
import random
```

(b) Change the signature (line ~135):

```python
def run_episode(
    *,
    episode_id: str,
    seed: int,
    adapter: BaseTaskAdapter,
    policy: RetryPolicy = babysteps_selective,
    record_baseline_metrics: bool = False,
) -> EpisodeRecord:
```

(c) Replace the attribution+revision block (current lines ~215-242) so it builds a `RetryContext` and calls `policy`. The selective default must reproduce today's records exactly, so when `policy is babysteps_selective` the attribution path is identical:

```python
    attribution = adapter.attribute_failure(failure_packet)
    oracle_correct_intent = adapter.oracle_correct_intent(scene_executor)
    ctx = RetryContext(
        initial_intent=initial_intent,
        attribution=attribution,
        scene=scene_executor,
        oracle_correct_intent=oracle_correct_intent,
        oracle_wrong_factor=oracle_wrong_factor,
        task_valid_tokens=adapter.task_valid_tokens(),
        rng=random.Random(_stable_hash(seed, "policy")),
        revise_fn=adapter.revise_intent,
    )

    try:
        proposal = policy(ctx)
    except NotImplementedError as exc:
        # selective/oracle may raise for unsupported factor transitions.
        fp_dict = {
            "failure_predicate": failure_packet.failure_predicate,
            "wrong_factor": attribution.wrong_factor,
            "oracle_wrong_factor": oracle_wrong_factor,
            "execution_trace": dict(failure_packet.execution_trace),
            "revision_error": str(exc),
        }
        metrics = _compute_metrics(
            initial_success=False, retry_success=False,
            failure_predicate=failure_packet.failure_predicate,
            wrong_factor_predicted=attribution.wrong_factor,
            oracle_wrong_factor=oracle_wrong_factor,
            factors_changed=(),
        )
        return EpisodeRecord(
            episode_id=episode_id, stage="stage_0", task=adapter.task_id,
            claim_boundary=CLAIM_BOUNDARY,
            demo=demo_dict, execution=execution_dict, failure_packet=fp_dict,
            revision=None, retry=None, metrics=metrics,
        )

    if proposal is None:
        # one_shot: no retry. Record the failed initial attempt only.
        fp_dict = {
            "failure_predicate": failure_packet.failure_predicate,
            "wrong_factor": attribution.wrong_factor,
            "oracle_wrong_factor": oracle_wrong_factor,
            "execution_trace": dict(failure_packet.execution_trace),
        }
        metrics = _compute_metrics(
            initial_success=bool(attempt_1.success), retry_success=None,
            failure_predicate=failure_packet.failure_predicate,
            wrong_factor_predicted=attribution.wrong_factor,
            oracle_wrong_factor=oracle_wrong_factor,
            factors_changed=(),
        )
        if record_baseline_metrics:
            metrics.update(_baseline_metrics(
                initial_intent, initial_intent, oracle_correct_intent,
                oracle_wrong_factor, adapter.task_valid_tokens()))
        return EpisodeRecord(
            episode_id=episode_id, stage="stage_0", task=adapter.task_id,
            claim_boundary=CLAIM_BOUNDARY,
            demo=demo_dict, execution=execution_dict, failure_packet=fp_dict,
            revision=None, retry=None, metrics=metrics,
        )

    revised_intent, revision_record = proposal
    attempt_2 = env_runner.run(revised_intent, scene_executor)
    factors_changed = _diff_intents(initial_intent, revised_intent)
    # ... (existing fp_dict / revision_dict / retry_dict / metrics block,
    #      unchanged, then the record return) ...
```

Keep the existing tail (`fp_dict` with `freeze`/`revise`, `revision_dict`,
`retry_dict`, `_compute_metrics(...)`, and the final `EpisodeRecord(...)`)
exactly as it is today. After computing `metrics` in that tail, add:

```python
    if record_baseline_metrics:
        metrics.update(_baseline_metrics(
            initial_intent, revised_intent, oracle_correct_intent,
            oracle_wrong_factor, adapter.task_valid_tokens()))
```

(d) Add the helpers near the top of the module (after `_diff_intents`):

```python
import hashlib


def _stable_hash(seed: int, salt: str) -> int:
    """Deterministic 32-bit seed from (episode seed, salt)."""
    h = hashlib.sha256(f"{seed}:{salt}".encode()).hexdigest()
    return int(h[:8], 16)


def _baseline_metrics(
    initial: Intent,
    revised: Intent,
    oracle_correct: Intent,
    oracle_wrong_factor: str,
    task_valid_tokens: dict,
) -> dict:
    """Label-based baseline metrics (spec §5). Computed only for baseline runs.

    - correct_factor_fixed: retry set the true wrong factor to its correct value.
    - should_preserve: editable factors other than the true wrong factor.
    - harmful_revision: any should-preserve factor changed (was correct → wrong).
    """
    editable = tuple(f for f in INTENT_FIELDS if f in task_valid_tokens)
    should_preserve = tuple(
        f for f in editable if f != oracle_wrong_factor
    )
    changed = set(_diff_intents(initial, revised))
    correct_factor_fixed = (
        getattr(revised, oracle_wrong_factor)
        == getattr(oracle_correct, oracle_wrong_factor)
    )
    n_preserved = sum(1 for f in should_preserve if f not in changed)
    harmful = any(f in changed for f in should_preserve)
    return {
        "correct_factor_fixed": bool(correct_factor_fixed),
        "harmful_revision": bool(harmful),
        "n_should_preserve": int(len(should_preserve)),
        "n_preserved": int(n_preserved),
    }
```

- [ ] **Step 4: Run unit tests AND the snapshot guards**

Run: `python -m pytest tests/test_episode.py tests/test_pushcube_adapter.py tests/test_pickcube_adapter.py tests/test_stackcube_adapter.py tests/test_crossview.py tests/test_turnfaucet_adapter.py -q`
Expected: PASS. **The snapshot tests must stay green** — default `run_episode` (no `policy`, `record_baseline_metrics=False`) produces byte-identical records. If a snapshot test fails, the default path drifted; fix the refactor (do not re-capture snapshots).

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest tests/ -q`
Expected: PASS (≥302 prior tests + new ones).

- [ ] **Step 6: Commit**

```bash
git add babysteps/episode.py tests/test_episode.py
git commit -m "refactor(baselines): inject RetryPolicy into run_episode (default selective)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Fresh execution seed per attempt

Threads a `rollout_seed` through the `EnvRunner` contract so a retry re-rolls
stochastic rollout components while holding scene layout fixed. The fake env is
deterministic, so this changes no fake-env outcomes (honest caveat: spec §4).

**Files:**
- Modify: `babysteps/envs/task_adapter.py` (`EnvRunner` Protocol)
- Modify: `tests/conftest.py` (fake runners accept the kwarg)
- Modify: `babysteps/episode.py` (pass `rollout_seed` for attempt 2)
- Test: `tests/test_episode.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_episode.py`:

```python
def test_env_runner_run_accepts_rollout_seed(fake_env_runner):
    # The fake runner must accept the optional kwarg (deterministic: ignored).
    from babysteps.schemas import Intent, SceneState
    scene = fake_env_runner.reset(0)
    intent = Intent(
        goal_state="cube_at_target", object_motion="translate_+x",
        contact_region="plus_x_face", approach_direction="from_minus_x",
        constraint_region="none", embodiment_mapping="proxy_contact_to_franka_push")
    r1 = fake_env_runner.run(intent, scene, rollout_seed=1)
    r2 = fake_env_runner.run(intent, scene, rollout_seed=2)
    assert r1.success == r2.success  # deterministic fake env
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_episode.py::test_env_runner_run_accepts_rollout_seed -v`
Expected: FAIL (`TypeError: run() got an unexpected keyword argument 'rollout_seed'`).

- [ ] **Step 3: Add `rollout_seed` to the contract and the fakes**

In `babysteps/envs/task_adapter.py`, update the Protocol method:

```python
    def run(
        self, intent: Intent, scene: SceneState, *, rollout_seed: int | None = None
    ) -> AttemptResult: ...
```

In `tests/conftest.py`, update **every** fake runner's `run` signature
(`FakeEnvRunner`, `FakePickEnvRunner`, and any other fake runner classes) to
accept and ignore the kwarg:

```python
    def run(
        self, intent: Intent, scene: SceneState, *, rollout_seed: int | None = None
    ) -> AttemptResult:
        # Deterministic fake: rollout_seed is accepted for contract parity and
        # intentionally ignored (no per-rollout stochasticity).
        _ = rollout_seed
        ...  # existing body unchanged
```

In `babysteps/episode.py`, pass a fresh seed on the **retry** call only:

```python
    attempt_2 = env_runner.run(
        revised_intent, scene_executor,
        rollout_seed=_stable_hash(seed, "attempt_2"),
    )
```

Leave attempt 1 (`env_runner.run(initial_intent, scene_executor)`) unchanged so
the selective snapshots remain byte-identical (no kwarg passed → default None).

- [ ] **Step 4: Run tests + snapshot guards**

Run: `python -m pytest tests/ -q`
Expected: PASS, snapshots green (the retry call passing a kwarg does not change
the fake env's deterministic output; the record bytes are unchanged).

- [ ] **Step 5: Commit**

```bash
git add babysteps/envs/task_adapter.py tests/conftest.py babysteps/episode.py tests/test_episode.py
git commit -m "feat(baselines): rollout_seed contract for fresh-seed-per-attempt

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

> **Note for the GPU sweep (Task 11):** the real runners (`pushcube_runner.py`,
> `pickcube_runner.py`, `stackcube_runner.py`) should accept `rollout_seed` and,
> where the env exposes per-rollout randomness independent of layout, use it to
> re-seed. If a runner is deterministic, it accepts-and-ignores the kwarg and
> `same_intent_retry` reads 0% — a valid, honestly-reported row (spec §4).

---

### Task 9: Aggregate the two new metrics in `eval.py`

**Files:**
- Modify: `babysteps/eval.py` (`compute_metrics`)
- Test: `tests/test_eval.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_eval.py`. Build minimal records carrying the baseline
metric keys:

```python
def _baseline_record(*, correct_fixed, harmful, n_should, n_preserved,
                     initial_success=False, retry_success=True, revised=True):
    from babysteps.schemas import EpisodeRecord
    metrics = {
        "initial_success": initial_success,
        "retry_success": retry_success,
        "num_attempts_to_success": 2 if retry_success else 2,
        "wrong_factor_predicted": "contact_region",
        "oracle_wrong_factor": "contact_region",
        "factor_attribution_correct": True,
        "factors_changed": ["contact_region"],
        "frozen_factors_preserved": True,
        "correct_factor_fixed": correct_fixed,
        "harmful_revision": harmful,
        "n_should_preserve": n_should,
        "n_preserved": n_preserved,
    }
    return EpisodeRecord(
        episode_id="x", stage="stage_0", task="PushCube-v1",
        claim_boundary="third_person_demo_proxy_not_human_demo",
        demo={}, execution={}, failure_packet={},
        revision={} if revised else None,
        retry={"success": retry_success} if revised else None,
        metrics=metrics)


def test_compute_metrics_baseline_columns():
    recs = [
        _baseline_record(correct_fixed=True, harmful=False, n_should=1, n_preserved=1),
        _baseline_record(correct_fixed=True, harmful=True, n_should=2, n_preserved=0),
    ]
    m = compute_metrics(recs)
    assert m["correct_factor_fixed_rate"] == 1.0
    assert m["harmful_revision_rate"] == 0.5
    # preserved/should = (1 + 0) / (1 + 2)
    assert abs(m["frozen_preservation_rate_gt"] - (1 / 3)) < 1e-9


def test_compute_metrics_baseline_columns_absent_when_no_keys():
    # Records without baseline keys → rates default to 0.0, evaluated count 0.
    from babysteps.schemas import EpisodeRecord
    rec = EpisodeRecord(
        episode_id="x", stage="stage_0", task="PushCube-v1",
        claim_boundary="third_person_demo_proxy_not_human_demo",
        demo={}, execution={}, failure_packet={}, revision={}, retry={},
        metrics={"initial_success": False, "retry_success": True,
                 "factors_changed": ["contact_region"]})
    m = compute_metrics([rec])
    assert m["n_baseline_evaluated"] == 0
    assert m["correct_factor_fixed_rate"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_eval.py -k baseline -v`
Expected: FAIL (`KeyError: 'correct_factor_fixed_rate'`).

- [ ] **Step 3: Extend `compute_metrics`**

In `babysteps/eval.py`, inside `compute_metrics`, after the existing
per-revision diagnostics loop (around line 73), add aggregation over records
that carry baseline keys, then add the keys to the returned dict:

```python
    # M3 baseline metrics (present only on baseline runs; absent → 0 / count 0).
    n_baseline_evaluated = 0
    n_correct_factor_fixed = 0
    n_harmful = 0
    total_should_preserve = 0
    total_preserved = 0
    for r in records_list:
        m = r.metrics
        if "correct_factor_fixed" not in m:
            continue
        n_baseline_evaluated += 1
        if m.get("correct_factor_fixed") is True:
            n_correct_factor_fixed += 1
        if m.get("harmful_revision") is True:
            n_harmful += 1
        total_should_preserve += int(m.get("n_should_preserve", 0))
        total_preserved += int(m.get("n_preserved", 0))

    correct_factor_fixed_rate = _safe_div(n_correct_factor_fixed, n_baseline_evaluated)
    harmful_revision_rate = _safe_div(n_harmful, n_baseline_evaluated)
    frozen_preservation_rate_gt = _safe_div(total_preserved, total_should_preserve)
```

Add to the returned dict (alongside the existing keys):

```python
        "n_baseline_evaluated": n_baseline_evaluated,
        "correct_factor_fixed_rate": correct_factor_fixed_rate,
        "harmful_revision_rate": harmful_revision_rate,
        "frozen_preservation_rate_gt": frozen_preservation_rate_gt,
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_eval.py -q`
Expected: PASS (existing eval tests unaffected — new keys are additive).

- [ ] **Step 5: Commit**

```bash
git add babysteps/eval.py tests/test_eval.py
git commit -m "feat(baselines): aggregate correct_factor_fixed + harmful_revision in eval

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: Comparison-table builder + Markdown writer

**Files:**
- Modify: `babysteps/eval.py`
- Test: `tests/test_eval.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_eval.py`:

```python
def test_compute_comparison_table_shape_and_order(tmp_path):
    from babysteps.eval import compute_comparison_table, write_comparison_table
    # metrics_by_method_task[method][task] = a compute_metrics() dict.
    fake_metrics = {"final_success_rate": 0.9, "retry_success_rate": 0.9,
                    "correct_factor_fixed_rate": 1.0,
                    "frozen_preservation_rate_gt": 1.0,
                    "harmful_revision_rate": 0.0,
                    "num_attempts_to_success_mean": 1.8}
    methods = ["one_shot", "babysteps_selective", "full_replan_analogue"]
    tasks = ["PushCube-v1", "PickCube-v1", "StackCube-v1"]
    by = {mth: {t: dict(fake_metrics) for t in tasks} for mth in methods}
    table = compute_comparison_table(by, methods=methods, tasks=tasks)
    assert [row["method"] for row in table["rows"]] == methods
    # each row has a per-task value + a mean for each column
    assert "mean" in table["rows"][0]["final_success_rate"]
    out = tmp_path / "table.md"
    write_comparison_table(table, out)
    assert out.exists() and "babysteps_selective" in out.read_text()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_eval.py::test_compute_comparison_table_shape_and_order -v`
Expected: FAIL (`ImportError: cannot import name 'compute_comparison_table'`).

- [ ] **Step 3: Implement the table builder + writer**

Append to `babysteps/eval.py`:

```python
# M3 comparison table — columns in reporting order (dir: ↑ better / ↓ better).
COMPARISON_COLUMNS: tuple[tuple[str, str], ...] = (
    ("final_success_rate", "↑"),
    ("retry_success_rate", "↑"),
    ("correct_factor_fixed_rate", "↑"),
    ("frozen_preservation_rate_gt", "↑"),
    ("harmful_revision_rate", "↓"),
    ("num_attempts_to_success_mean", "↓"),
)


def compute_comparison_table(
    metrics_by_method_task: dict[str, dict[str, dict]],
    *,
    methods: list[str],
    tasks: list[str],
) -> dict:
    """Assemble the 7-method × 3-task table. Each cell pulls a column value
    from the per-(method, task) compute_metrics() dict; 'mean' is the simple
    mean across tasks."""
    rows = []
    for method in methods:
        row: dict = {"method": method}
        for col, _dir in COMPARISON_COLUMNS:
            per_task = {
                t: float(metrics_by_method_task[method][t].get(col, 0.0))
                for t in tasks
            }
            per_task["mean"] = sum(per_task.values()) / len(tasks)
            row[col] = per_task
        rows.append(row)
    return {"methods": methods, "tasks": tasks, "rows": rows}


def write_comparison_table(table: dict, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tasks = table["tasks"]
    header_cols = []
    for col, direction in COMPARISON_COLUMNS:
        for t in tasks:
            header_cols.append(f"{col} {direction} ({t.rsplit('-v', 1)[0]})")
        header_cols.append(f"{col} {direction} (mean)")
    header = "| method | " + " | ".join(header_cols) + " |"
    sep = "|" + "---|" * (len(header_cols) + 1)
    lines = [header, sep]
    for row in table["rows"]:
        cells = [row["method"]]
        for col, _dir in COMPARISON_COLUMNS:
            for t in tasks:
                cells.append(f"{row[col][t]:.2f}")
            cells.append(f"{row[col]['mean']:.2f}")
        lines.append("| " + " | ".join(cells) + " |")
    caption = (
        "\n> Stage-0 procedural baseline table. *full_replan_analogue* and "
        "*text_feedback_replan* are deterministic procedural analogues, not "
        "measured LLM/VLM performance. Live replanners are future work.\n"
    )
    out_path.write_text(
        "# BABYSTEPS Stage-0 — Procedural Baseline Comparison\n\n"
        + "\n".join(lines) + "\n" + caption
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_eval.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add babysteps/eval.py tests/test_eval.py
git commit -m "feat(baselines): comparison-table builder + markdown writer

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 11: Sweep runner CLI (`scripts/run_baselines.py`)

Drives `{7 policies} × {3 tasks}` × N seeds through `run_episode` (real runner
via the task registry; `--fake-env` for sim-free CI), writes one report per
(method, task) plus the comparison table. The sweep *logic* is tested with the
fake env.

**Files:**
- Create: `scripts/run_baselines.py`
- Test: `tests/test_run_baselines_cli.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_run_baselines_cli.py`:

```python
import json
from pathlib import Path
import subprocess
import sys


def test_run_baselines_fake_env_smoke(tmp_path):
    # 2 seeds × all methods × PushCube on the fake env → table is produced.
    out = tmp_path / "sweep"
    proc = subprocess.run(
        [sys.executable, "scripts/run_baselines.py",
         "--tasks", "PushCube-v1", "--methods", "all",
         "--n_episodes", "2", "--seed_start", "0",
         "--out_dir", str(out), "--fake-env"],
        capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    table_md = out / "comparison_table.md"
    table_json = out / "comparison_table.json"
    assert table_md.exists()
    assert table_json.exists()
    data = json.loads(table_json.read_text())
    methods = [r["method"] for r in data["rows"]]
    assert "babysteps_selective" in methods
    assert "full_replan_analogue" in methods
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_run_baselines_cli.py -v`
Expected: FAIL (`scripts/run_baselines.py` does not exist → returncode != 0).

- [ ] **Step 3: Implement the sweep runner**

Create `scripts/run_baselines.py`:

```python
#!/usr/bin/env python
"""Run the Stage-0 procedural baseline sweep: methods × tasks × seeds.

Sim-free CI uses --fake-env. The real sweep runs on GPU via the task registry.
See docs/superpowers/specs/2026-05-20-stage0-baselines-design.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from babysteps.episode import run_episode
from babysteps.eval import (
    compute_comparison_table, compute_metrics, write_comparison_table,
)
from babysteps.policies import POLICIES

MAIN_TABLE_METHODS = [
    "one_shot", "same_intent_retry", "random_factor_revision",
    "babysteps_selective", "text_feedback_replan", "full_replan_analogue",
    "oracle_factor_revision",
]


def _make_adapter(task: str, fake_env: bool):
    """Return a task adapter. --fake-env swaps the runner for the conftest fake
    via the registry's fake-env hook."""
    from babysteps.envs.task_registry import make_adapter
    return make_adapter(task, fake_env=fake_env)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--methods", nargs="+", default=["all"])
    ap.add_argument("--n_episodes", type=int, default=20)
    ap.add_argument("--seed_start", type=int, default=0)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--fake-env", action="store_true")
    args = ap.parse_args()

    methods = MAIN_TABLE_METHODS if args.methods == ["all"] else args.methods
    args.out_dir.mkdir(parents=True, exist_ok=True)

    by: dict[str, dict[str, dict]] = {m: {} for m in methods}
    for task in args.tasks:
        for method in methods:
            adapter = _make_adapter(task, args.fake_env)
            records = []
            try:
                for i in range(args.n_episodes):
                    seed = args.seed_start + i
                    records.append(run_episode(
                        episode_id=f"{task}_{method}_seed_{seed:04d}",
                        seed=seed, adapter=adapter,
                        policy=POLICIES[method],
                        record_baseline_metrics=True,
                    ))
            finally:
                adapter.close()
            metrics = compute_metrics(records)
            run_dir = args.out_dir / method / task
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "samples.jsonl").write_text(
                "\n".join(r.to_jsonl_line() for r in records) + "\n")
            (run_dir / "report.json").write_text(
                json.dumps(metrics, indent=2, sort_keys=True) + "\n")
            by[method][task] = metrics

    table = compute_comparison_table(by, methods=methods, tasks=list(args.tasks))
    write_comparison_table(table, args.out_dir / "comparison_table.md")
    (args.out_dir / "comparison_table.json").write_text(
        json.dumps(table, indent=2, sort_keys=True) + "\n")
    print(f"wrote comparison table to {args.out_dir}/comparison_table.md")


if __name__ == "__main__":
    main()
```

> **Registry dependency:** this assumes `babysteps/envs/task_registry.py`
> exposes `make_adapter(task, fake_env=False)`. **Verify the real signature
> first** (`grep -n "def make_adapter\|def " babysteps/envs/task_registry.py`).
> If the registry has no `fake_env` path, add one in this task: a thin branch
> that, when `fake_env=True`, builds the adapter but injects the conftest fake
> runner. Keep that addition minimal and sim-free; mirror how
> `tests/test_stage0_collect_cli.py` obtains a fake-env adapter (read it first).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_run_baselines_cli.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest tests/ -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/run_baselines.py tests/test_run_baselines_cli.py
git commit -m "feat(baselines): methods×tasks×seeds sweep runner + fake-env smoke test

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 12: GPU sbatch script for the real sweep

**Files:**
- Create: `slurm/run_baselines.sbatch`

- [ ] **Step 1: Write the sbatch script**

Create `slurm/run_baselines.sbatch` (mirror an existing render sbatch for the
partition/module lines — read `slurm/` for the exact header used by the
CrossView job 10737370 before editing):

```bash
#!/bin/bash
#SBATCH --job-name=baselines
#SBATCH --output=slurm/logs/baselines-%j.out
#SBATCH --error=slurm/logs/baselines-%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# (Match the env activation used by the CrossView sbatch — copy those lines.)
python scripts/run_baselines.py \
  --tasks PushCube-v1 PickCube-v1 StackCube-v1 \
  --methods all \
  --n_episodes 24 --seed_start 0 \
  --out_dir datasets/stage0_baselines
echo "baseline sweep complete"
```

- [ ] **Step 2: Verify it parses (no submission)**

Run: `bash -n slurm/run_baselines.sbatch`
Expected: no output (syntax OK). Do NOT `sbatch` it here — that's a user-run GPU step.

- [ ] **Step 3: Commit**

```bash
git add slurm/run_baselines.sbatch
git commit -m "feat(baselines): sbatch script for the GPU 7x3 baseline sweep

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 13: Reconcile the locked claim doc + RUNBOOK

**Files:**
- Modify: `docs/milestone1_locked_claim.md` (§4)
- Modify: `RUNBOOK.md`

- [ ] **Step 1: Update `milestone1_locked_claim.md §4`**

Replace the 5-row method list with the 7 rows (`one_shot`,
`same_intent_retry`, `random_factor_revision`, `babysteps_selective`,
`text_feedback_replan`, `full_replan_analogue`, `oracle_factor_revision`),
add the new columns (`correct_factor_fixed ↑`, `harmful_revision_rate ↓`,
`frozen_preservation_rate_gt ↑`), and add the spec §6 procedural-analogue
caption. Note in the section that the 5 original rows are a subset (additive).

- [ ] **Step 2: Add a RUNBOOK entry**

Under the appropriate section in `RUNBOOK.md`, add:

```markdown
## Run the procedural baseline sweep (M3)

Sim-free smoke (login node):
\`\`\`bash
python scripts/run_baselines.py --tasks PushCube-v1 --methods all \
  --n_episodes 2 --seed_start 0 --out_dir /tmp/baselines --fake-env
\`\`\`

Full GPU sweep (7 methods × 3 tasks × 24 seeds):
\`\`\`bash
sbatch slurm/run_baselines.sbatch   # writes datasets/stage0_baselines/comparison_table.md
\`\`\`
```

- [ ] **Step 3: Commit**

```bash
git add docs/milestone1_locked_claim.md RUNBOOK.md
git commit -m "docs(baselines): reconcile locked claim §4 to 7 rows + RUNBOOK sweep entry

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Final Verification

- [ ] Run the full suite: `python -m pytest tests/ -q` → all PASS, no GPU.
- [ ] Confirm snapshot guards green (selective path byte-identical).
- [ ] Confirm the fake-env sweep writes `comparison_table.md` + `.json`.
- [ ] (User, on GPU) `sbatch slurm/run_baselines.sbatch`, then inspect
      `datasets/stage0_baselines/comparison_table.md` against the spec §5
      qualitative pattern and §9 acceptance gate.

---

## Self-Review Notes (author)

- **Spec coverage:** §1 scope → Tasks 3–13; §2 policies → Tasks 3–6; §3
  architecture → Task 7; §4 fresh seed → Task 8; §5 metrics+table → Tasks 9–10;
  §6 caption → Tasks 10, 13; §7 testing → every task (sim-free); §8 doc
  reconciliation → Task 13; §9 acceptance gate → Final Verification.
- **Open verification (do at execution time, not assumed):** the exact
  `task_registry` adapter/fake-env entry point (Task 11 Step 3) and the
  `slurm/` env-activation header (Task 12) — both flagged inline to read the
  real file first rather than guess.
- **Preservation is measured against the ground-truth should-preserve set**
  (all editable factors except `oracle_wrong_factor`), not the policy's
  self-declared `frozen_factors` — otherwise `full_replan_analogue` would
  trivially score 100% preserved.
