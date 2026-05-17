# Sub-project C (StackCube-v1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land StackCube-v1 as the third adapter in the Stage-0 loop, exercising the third intent factor (`goal_state`) via a demo whose 2D-trajectory summarization under-specifies "place cubeA on cubeB" as "translate cubeA to cubeB.xy." Failure → `goal_refinement` revision → retry stacks. Closes the spec's acceptance gate end-to-end via fake-env CLI; the real-sim GPU spot-check is documented for the user to run on a Vulkan node.

**Architecture:** New `StackCubeAdapter` + `StackCubeEnvRunner` + `StackSkill` + `babysteps/render/stackcube.py` slot into the existing Stage-0 dispatch machinery (BaseTaskAdapter, TASK_REGISTRY, RENDER_REGISTRY) with one-row registry additions. The skill compiler dispatches on `intent.goal_state`: 4-waypoint translate-and-drop for the under-specified `cube_at_target`, 5-waypoint pick-and-place for the refined `cubeA_on_cubeB`. New revision operator `goal_refinement` adds one branch to `revise_intent`. The Stage-0 controlled failure is *natural* — the wrong-goal waypoints physically miss the stack — so no `blocked_sides` mechanism is needed (`default_blocked_factory` returns `()`).

**Tech Stack:** Python 3, ManiSkill 3 (gymnasium), pytest, numpy, PIL + imageio for MP4s. All existing Stage-0 infrastructure (adapter ABC, episode loop, failure/revision modules, task/render registries, CLI scripts, render package) is reused without modification.

**Source spec:** `docs/superpowers/specs/2026-05-17-stage0-stackcube-c-design.md` (committed at `a336aeb`).

**Scope guardrails:**
- Sub-project C only. Sub-project D (OpenCabinetDrawer) is a separate plan.
- PushCube + PickCube snapshots MUST stay byte-identical.
- All 184 existing tests must continue to pass at every task boundary.
- Stage-0 one-attempt-then-one-retry per episode (spec §2 non-goal — no multi-attempt loops).
- Privileged-firewall: `scripted_demo_to_intent` stays demo-only; the deliberate under-specification (always returning `goal_state="cube_at_target"`) is the controlled-failure mechanism, NOT a privileged leak.

---

## File Structure

**Create:**
- `babysteps/skills/stack.py` — `StackSkill` dataclass + `compile_intent_to_stack_skill` (pure; dispatches on `intent.goal_state`).
- `babysteps/envs/stackcube_adapter.py` — `StackCubeAdapter(BaseTaskAdapter)`.
- `babysteps/envs/stackcube_runner.py` — real ManiSkill `StackCubeEnvRunner` (mirrors `PickCubeEnvRunner`).
- `babysteps/render/stackcube.py` — `render_episode` for the three-phase MP4 flow.
- `tests/test_stack_skill.py` — 10 sim-free unit tests for the skill compiler.
- `tests/test_stackcube_adapter.py` — 15 sim-free unit tests + snapshot test.
- `tests/snapshots/stackcube_samples_seeds_0_4.jsonl` — captured during Task 8.

**Modify:**
- `babysteps/schemas.py` — 4 new whitelist tokens (Task 1).
- `babysteps/revision.py` — `goal_refinement` branch in `revise_intent` (Task 2).
- `babysteps/envs/task_registry.py` — `_stackcube_entry()` + registry row (Task 6).
- `babysteps/render/__init__.py` — `_stackcube_render()` + registry row (Task 6).
- `babysteps/render/common.py` — `STACKCUBE_MAX_CONTROL_STEPS = 400` (Task 7).
- `tests/conftest.py` — append `FakeStackCubeEnvRunner` + `fake_stack_env_runner` fixture (Task 4).
- `tests/test_schemas.py` — 4 new tests for whitelist additions (Task 1).
- `tests/test_revision.py` — 3 new tests for `goal_refinement` (Task 2).
- `tests/test_task_registry.py` — 1 new `get_task_entry("StackCube-v1")` test (Task 6).
- `tests/test_render_modules.py` — 3 new StackCube render tests (Task 7).
- `tests/test_stage0_collect_cli.py` — extend parametrize with `("StackCube-v1", "stackcube_samples_seeds_0_4.jsonl")` (Task 8).
- `tests/test_pickcube_delta_pp.py` — 1 new StackCube delta_pp test (Task 9).
- `CLAUDE.md` — third srun block + module list update + test count refresh (Task 10).

**Untouched:**
- `babysteps/envs/task_adapter.py` (BaseTaskAdapter is task-agnostic).
- `babysteps/episode.py` (adapter-driven; supports any task that fits the interface).
- `babysteps/failure.py` (no new predicate; `goal_not_satisfied` already maps to `goal_state`).
- `babysteps/eval.py` (task-aware via `records[0].task` since B).
- `scripts/stage0_collect.py`, `scripts/stage0_summarize.py`, `scripts/render_stage0_maniskill.py` (all dispatch through registries via `--task`).
- `tests/snapshots/pushcube_samples_seeds_0_4.jsonl`, `tests/snapshots/pickcube_samples_seeds_0_4.jsonl` (must stay byte-identical).
- All B-era files (PushCube/PickCube adapters, runners, skills, render modules).

---

## Task 0: Baseline — confirm world state matches plan assumptions

**Files:** none — read-only.

- [ ] **Step 1: Run the full test suite**

Run:
```bash
cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover
python -m pytest tests/ -q 2>&1 | tail -5
```

Expected: `184 passed`. If anything fails, STOP and report.

- [ ] **Step 2: Confirm B-era snapshots exist**

Run: `wc -l tests/snapshots/*.jsonl`

Expected:
```
   5 tests/snapshots/pickcube_samples_seeds_0_4.jsonl
   5 tests/snapshots/pushcube_samples_seeds_0_4.jsonl
  10 total
```

If `stackcube_samples_seeds_0_4.jsonl` already exists, this plan needs adjustment — STOP and surface.

- [ ] **Step 3: Confirm B-era registries have exactly 2 entries**

Run:
```bash
python -c "from babysteps.envs.task_registry import TASK_REGISTRY; print(sorted(TASK_REGISTRY))"
python -c "from babysteps.render import RENDER_REGISTRY; print(sorted(RENDER_REGISTRY))"
```

Expected:
```
['PickCube-v1', 'PushCube-v1']
['PickCube-v1', 'PushCube-v1']
```

- [ ] **Step 4: Confirm spec is committed**

Run: `git log --oneline -1 docs/superpowers/specs/2026-05-17-stage0-stackcube-c-design.md`

Expected: shows commit `a336aeb` (or whichever is current HEAD of the spec).

---

## Task 1: Schema deltas

**Files:**
- Modify: `babysteps/schemas.py`
- Modify: `tests/test_schemas.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_schemas.py`:

```python
# ---------- Sub-project C (StackCube) whitelist additions ----------- #


def test_goal_states_contains_cubeA_on_cubeB():
    from babysteps.schemas import GOAL_STATES
    assert "cubeA_on_cubeB" in GOAL_STATES


def test_object_motions_contains_place_on():
    from babysteps.schemas import OBJECT_MOTIONS
    assert "place_on" in OBJECT_MOTIONS


def test_embodiment_mappings_contains_pick_and_place():
    from babysteps.schemas import EMBODIMENT_MAPPINGS
    assert "proxy_contact_to_franka_pick_and_place" in EMBODIMENT_MAPPINGS


def test_revision_operators_contains_goal_refinement():
    from babysteps.schemas import REVISION_OPERATORS
    assert "goal_refinement" in REVISION_OPERATORS
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_schemas.py -v -k "stackcube or cubeA or place_on or pick_and_place or goal_refinement" 2>&1 | tail -20`

Expected: 4 tests FAIL with `AssertionError` (tokens not yet in the whitelists).

- [ ] **Step 3: Add tokens to whitelists**

In `babysteps/schemas.py`, find:

```python
OBJECT_MOTIONS: frozenset[str] = frozenset({
    "translate_+x", "translate_-x", "translate_+y", "translate_-y",
    "lift_up",   # B: PickCube — cube lifted along +z
})
```

Replace with:

```python
OBJECT_MOTIONS: frozenset[str] = frozenset({
    "translate_+x", "translate_-x", "translate_+y", "translate_-y",
    "lift_up",   # B: PickCube — cube lifted along +z
    "place_on",  # C: StackCube — cube placed on top of another cube
})
```

Find:

```python
EMBODIMENT_MAPPINGS: frozenset[str] = frozenset({
    "proxy_contact_to_franka_push",
    "proxy_contact_to_franka_grasp",   # B: PickCube — parallel-jaw grasp
})
```

Replace with:

```python
EMBODIMENT_MAPPINGS: frozenset[str] = frozenset({
    "proxy_contact_to_franka_push",
    "proxy_contact_to_franka_grasp",   # B: PickCube — parallel-jaw grasp
    "proxy_contact_to_franka_pick_and_place",  # C: StackCube — pick + place sequence
})
```

Find:

```python
GOAL_STATES: frozenset[str] = frozenset({
    "cube_at_target",
    "cube_lifted_at_target",           # B: PickCube — cube lifted to goal xyz
})
```

Replace with:

```python
GOAL_STATES: frozenset[str] = frozenset({
    "cube_at_target",
    "cube_lifted_at_target",           # B: PickCube — cube lifted to goal xyz
    "cubeA_on_cubeB",                  # C: StackCube — cubeA resting atop cubeB
})
```

Find:

```python
REVISION_OPERATORS: frozenset[str] = frozenset({
    "approach_substitution",
    "contact_substitution",            # B: PickCube — rotate gripper axis
})
```

Replace with:

```python
REVISION_OPERATORS: frozenset[str] = frozenset({
    "approach_substitution",
    "contact_substitution",            # B: PickCube — rotate gripper axis
    "goal_refinement",                 # C: StackCube — sharpen under-specified goal
})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_schemas.py -v 2>&1 | tail -5`

Expected: all tests pass (was 29, gains 4 → 33).

- [ ] **Step 5: Full suite**

Run: `python -m pytest tests/ -q 2>&1 | tail -5`

Expected: `188 passed` (184 + 4 new). The existing snapshot tests must remain green — `cubeA_on_cubeB` and friends are additive only; no PushCube/PickCube record contains these tokens.

- [ ] **Step 6: Commit**

```bash
git add babysteps/schemas.py tests/test_schemas.py
git commit -m "$(cat <<'EOF'
feat(schemas): Sub-project C whitelist additions (StackCube)

- GOAL_STATES += "cubeA_on_cubeB"
- OBJECT_MOTIONS += "place_on"
- EMBODIMENT_MAPPINGS += "proxy_contact_to_franka_pick_and_place"
- REVISION_OPERATORS += "goal_refinement"

Per spec §4 of docs/superpowers/specs/2026-05-17-stage0-stackcube-c-design.md.
Additive only — PushCube and PickCube records do not contain these tokens
so existing snapshot tests stay byte-identical.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `goal_refinement` revision operator

**Files:**
- Modify: `babysteps/revision.py`
- Modify: `tests/test_revision.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_revision.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_revision.py -v -k "goal_refinement" 2>&1 | tail -10`

Expected: 3 tests FAIL — current `revise_intent` does not handle `wrong_factor="goal_state"` (raises `NotImplementedError` with a generic message, NOT mentioning `cube_lifted_at_target` specifically).

- [ ] **Step 3: Add the `goal_refinement` branch to `revise_intent`**

In `babysteps/revision.py`, find:

```python
    if attribution.wrong_factor == "contact_region":
        old = intent.contact_region
        if old not in CONTACT_REGIONS:
            raise ValueError(
                f"contact_substitution: current contact_region {old!r} not in "
                f"CONTACT_REGIONS"
            )
        new = _pick_unblocked_face(old, scene.blocked_sides)
        revised = replace(intent, contact_region=new)
        frozen = tuple(f for f in INTENT_FIELDS if f != "contact_region")
        rev_record = Revision(
            operator="contact_substitution",
            factor="contact_region",
            old_value=old,
            new_value=new,
            frozen_factors=frozen,
        )
        return revised, rev_record

    raise NotImplementedError(
```

Insert a new branch above the final `raise`:

```python
    if attribution.wrong_factor == "goal_state":
        # Stage-0's goal_refinement is a strict-extension operator:
        # cube_at_target → cubeA_on_cubeB only. Other goal_state transitions
        # are deferred (per spec §6 of
        # docs/superpowers/specs/2026-05-17-stage0-stackcube-c-design.md).
        if intent.goal_state != "cube_at_target":
            raise NotImplementedError(
                f"goal_refinement does not handle transitions from "
                f"goal_state {intent.goal_state!r}. (Stage-0 supports only "
                f"the cube_at_target → cubeA_on_cubeB refinement per "
                f"docs/superpowers/specs/2026-05-17-stage0-stackcube-c-design.md §6)"
            )
        old = intent.goal_state
        new = "cubeA_on_cubeB"
        revised = replace(intent, goal_state=new)
        frozen = tuple(f for f in INTENT_FIELDS if f != "goal_state")
        rev_record = Revision(
            operator="goal_refinement",
            factor="goal_state",
            old_value=old,
            new_value=new,
            frozen_factors=frozen,
        )
        return revised, rev_record

    raise NotImplementedError(
```

Also update the module docstring (line ~9) — find:

```python
Stage 0 implements:
  * `approach_substitution` — for wrong_factor=="approach_direction"
    (Sub-project A / PushCube).
  * `contact_substitution` — for wrong_factor=="contact_region"
    (Sub-project B / PickCube).

Other wrong_factors raise `NotImplementedError` — honest about what is and
isn't validated.
```

Replace with:

```python
Stage 0 implements:
  * `approach_substitution` — for wrong_factor=="approach_direction"
    (Sub-project A / PushCube).
  * `contact_substitution` — for wrong_factor=="contact_region"
    (Sub-project B / PickCube).
  * `goal_refinement` — for wrong_factor=="goal_state"
    (Sub-project C / StackCube; strict-extension: cube_at_target →
    cubeA_on_cubeB only).

Other wrong_factors raise `NotImplementedError` — honest about what is and
isn't validated.
```

Also update the closing `raise NotImplementedError` message at the end of `revise_intent`. Find:

```python
    raise NotImplementedError(
        f"Stage-0 reviser handles 'approach_direction' and 'contact_region'; "
        f"got {attribution.wrong_factor!r}. (Other factors are reserved "
        f"for later sub-projects — see "
        f"docs/superpowers/specs/2026-05-17-stage0-four-scene-roadmap-design.md §6)"
    )
```

Replace with:

```python
    raise NotImplementedError(
        f"Stage-0 reviser handles 'approach_direction', 'contact_region', "
        f"and 'goal_state'; got {attribution.wrong_factor!r}. (Other factors "
        f"are reserved for later sub-projects — see "
        f"docs/superpowers/specs/2026-05-17-stage0-four-scene-roadmap-design.md §6)"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_revision.py -v 2>&1 | tail -10`

Expected: all `test_revision.py` tests pass (was 13, gains 3 → 16).

- [ ] **Step 5: Full suite**

Run: `python -m pytest tests/ -q 2>&1 | tail -5`

Expected: `191 passed` (188 + 3).

- [ ] **Step 6: Commit**

```bash
git add babysteps/revision.py tests/test_revision.py
git commit -m "$(cat <<'EOF'
feat(revision): goal_refinement operator for Sub-project C (StackCube)

Adds the third Stage-0 revision branch. Strict-extension: handles only
the cube_at_target → cubeA_on_cubeB transition (per spec §6). Other
goal_state sources raise NotImplementedError to keep coverage honest.

The operator preserves factor-local invariance — only goal_state
changes; the remaining 5 factors are frozen. Tests cover the happy
path, the unknown-source NotImplementedError, and the frozen-factors
audit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `StackSkill` + `compile_intent_to_stack_skill`

**Files:**
- Create: `babysteps/skills/stack.py`
- Test: `tests/test_stack_skill.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_stack_skill.py`:

```python
"""Tests for babysteps/skills/stack.py — pure waypoint geometry."""
from __future__ import annotations

import numpy as np
import pytest

from babysteps.schemas import Intent, SceneState


def _scene(cubeA_xy=(0.0, 0.0), cubeB_xy=(0.10, 0.0), cubeB_z=0.02):
    return SceneState(
        cube_xy=cubeA_xy,
        cube_z=0.02,
        goal_xy=cubeB_xy,
        tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0),
        blocked_sides=(),
        extra={
            "cubeB_xy": cubeB_xy,
            "cubeB_z": cubeB_z,
            "cubeB_top_z": cubeB_z + 0.04,
        },
    )


def _intent(goal_state="cubeA_on_cubeB"):
    return Intent(
        goal_state=goal_state,
        object_motion="place_on" if goal_state == "cubeA_on_cubeB" else "translate_+x",
        contact_region="minus_x_face",
        approach_direction="from_above",
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_pick_and_place",
    )


def test_compile_returns_stackskill_instance():
    from babysteps.skills.stack import StackSkill, compile_intent_to_stack_skill
    skill = compile_intent_to_stack_skill(_intent(), _scene())
    assert isinstance(skill, StackSkill)


def test_cubeA_on_cubeB_has_five_waypoints():
    from babysteps.skills.stack import compile_intent_to_stack_skill
    skill = compile_intent_to_stack_skill(_intent("cubeA_on_cubeB"), _scene())
    assert skill.waypoints.shape == (5, 7)


def test_cube_at_target_has_four_waypoints():
    from babysteps.skills.stack import compile_intent_to_stack_skill
    skill = compile_intent_to_stack_skill(_intent("cube_at_target"), _scene())
    assert skill.waypoints.shape == (4, 7)


def test_cubeA_on_cubeB_final_waypoint_is_above_cubeB_top():
    """The place_on waypoint puts the TCP at cubeB_top_z + CUBE_HALF_SIZE +
    PLACE_CLEARANCE_M so cubeA settles on top after gripper release."""
    from babysteps.skills.stack import (
        CUBE_HALF_SIZE, PLACE_CLEARANCE_M, compile_intent_to_stack_skill,
    )
    scene = _scene(cubeB_xy=(0.12, 0.05), cubeB_z=0.02)
    skill = compile_intent_to_stack_skill(_intent("cubeA_on_cubeB"), scene)
    final = skill.waypoints[-1]
    expected_z = scene.extra["cubeB_top_z"] + CUBE_HALF_SIZE + PLACE_CLEARANCE_M
    assert final[0] == pytest.approx(0.12)
    assert final[1] == pytest.approx(0.05)
    assert final[2] == pytest.approx(expected_z)


def test_cube_at_target_final_waypoint_is_low_at_cubeB_xy():
    """The translate-release waypoint puts the TCP at cubeB.xy at low z
    (cubeA_z + DESCEND_CLEARANCE_M) — cubeA collides with cubeB and scatters."""
    from babysteps.skills.stack import (
        DESCEND_CLEARANCE_M, compile_intent_to_stack_skill,
    )
    scene = _scene(cubeB_xy=(0.12, 0.05), cubeB_z=0.02)
    skill = compile_intent_to_stack_skill(_intent("cube_at_target"), scene)
    final = skill.waypoints[-1]
    expected_z = scene.cube_z + DESCEND_CLEARANCE_M
    assert final[0] == pytest.approx(0.12)
    assert final[1] == pytest.approx(0.05)
    assert final[2] == pytest.approx(expected_z)


def test_first_waypoint_is_above_cubeA():
    """Both compile paths start with approach above cubeA at travel_z."""
    from babysteps.skills.stack import compile_intent_to_stack_skill
    scene = _scene(cubeA_xy=(-0.05, 0.03))
    for goal in ("cube_at_target", "cubeA_on_cubeB"):
        skill = compile_intent_to_stack_skill(_intent(goal), scene)
        wp0 = skill.waypoints[0]
        assert wp0[0] == pytest.approx(-0.05)
        assert wp0[1] == pytest.approx(0.03)
        assert wp0[2] == pytest.approx(0.25)   # travel_z from tcp_start_pose


def test_grasp_waypoint_is_at_cubeA_z():
    """Waypoint 2 (zero-indexed) is the grasp_close — at cubeA's actual z."""
    from babysteps.skills.stack import compile_intent_to_stack_skill
    scene = _scene()
    skill = compile_intent_to_stack_skill(_intent("cubeA_on_cubeB"), scene)
    grasp = skill.waypoints[2]
    assert grasp[2] == pytest.approx(scene.cube_z)


def test_quaternion_columns_come_from_tcp_start_pose():
    """Columns 3:7 of every waypoint hold the TCP's starting quaternion."""
    from babysteps.skills.stack import compile_intent_to_stack_skill
    scene = _scene()
    skill = compile_intent_to_stack_skill(_intent("cubeA_on_cubeB"), scene)
    tcp_q = np.asarray(scene.tcp_start_pose[3:7])
    for i in range(skill.waypoints.shape[0]):
        assert np.allclose(skill.waypoints[i, 3:7], tcp_q)


def test_compile_raises_on_unknown_goal_state():
    """Goal states outside the C-supported set raise ValueError."""
    from babysteps.skills.stack import compile_intent_to_stack_skill
    scene = _scene()
    # cube_lifted_at_target is a PickCube goal_state; not handled by stack skill.
    bad_intent = Intent(
        goal_state="cube_lifted_at_target",
        object_motion="lift_up",
        contact_region="minus_x_face",
        approach_direction="from_above",
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_grasp",
    )
    with pytest.raises(ValueError) as exc:
        compile_intent_to_stack_skill(bad_intent, scene)
    assert "cube_lifted_at_target" in str(exc.value)


def test_skill_exposes_cubeA_z_and_cubeB_top_z():
    """The compiled skill carries the geometry the env_runner needs."""
    from babysteps.skills.stack import compile_intent_to_stack_skill
    scene = _scene(cubeB_xy=(0.08, 0.0), cubeB_z=0.025)
    skill = compile_intent_to_stack_skill(_intent("cubeA_on_cubeB"), scene)
    assert skill.cubeA_z == pytest.approx(scene.cube_z)
    assert skill.cubeB_top_z == pytest.approx(0.025 + 0.04)
    assert skill.goal_state == "cubeA_on_cubeB"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_stack_skill.py -v 2>&1 | tail -10`

Expected: all 10 tests FAIL with `ModuleNotFoundError: No module named 'babysteps.skills.stack'`.

- [ ] **Step 3: Create `babysteps/skills/stack.py`**

```python
"""Stack skill compiler — turns a StackCube Intent into an executable StackSkill.

This module is pure: no simulator, no I/O. It encodes the Sub-project C
design (docs/superpowers/specs/2026-05-17-stage0-stackcube-c-design.md):

1. **Compile dispatches on intent.goal_state.** Two supported values:
   - `cube_at_target`  → 4-waypoint trajectory (pick + low-z release at
     cubeB.xy). Cube collides with cubeB and scatters — the deliberately
     under-specified Stage-0 demo outcome.
   - `cubeA_on_cubeB`  → 5-waypoint trajectory (pick + lift over cubeB +
     descend onto cubeB top + release). Successful stack.

2. **Geometry is symmetric across the two goal_states for phases 0-2.**
   The difference is in the final phase(s): translate-and-drop-low vs
   lift-and-descend-onto-cubeB. This keeps the geometric tests
   parameterizable and makes the failure narrative read directly off
   the skill's waypoint count.

3. **No slip mechanism.** Unlike PickSkill (which never returns None
   because slip is execution-time), StackSkill also never returns None;
   the failure is purely from the wrong waypoints. ValueError is raised
   only when intent.goal_state is outside the C-supported set.

Geometry constants are calibrated for ManiSkill StackCube-v1
(cube half-size 0.02; pd_ee_delta_pose normalization).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from babysteps.schemas import Intent, SceneState

# StackCube cube half-size — matches ManiSkill's self.cube_half_size.
CUBE_HALF_SIZE: float = 0.02

# Vertical gap above cubeA_z at the pre-close descend waypoint. Gives the
# PD controller a soft landing before grasp_close.
DESCEND_CLEARANCE_M: float = 0.02

# Vertical gap above cubeB's top at the place_on waypoint. Small enough
# that cubeA settles onto cubeB without overshooting.
PLACE_CLEARANCE_M: float = 0.005


@dataclass(frozen=True)
class StackSkill:
    """A compiled stack-and-place trajectory.

    `waypoints` is (N, 7) where N=4 for cube_at_target and N=5 for
    cubeA_on_cubeB. Columns are [x, y, z, qx, qy, qz, qw]. The
    quaternion is the TCP's starting orientation (cubeA is grasped with
    the default gripper rotation; the runner overrides only the gripper
    open/close command, not the orientation).

    `cubeA_z` and `cubeB_top_z` are exposed for the env_runner's success
    checks (e.g., confirming the lift cleared cubeB).
    """

    waypoints: np.ndarray
    cubeA_z: float
    cubeB_top_z: float
    goal_state: str


def _build_translate_waypoints(scene: SceneState) -> np.ndarray:
    """4 waypoints: approach above cubeA, descend, grasp, translate-release
    at cubeB.xy at low z. Used for the under-specified cube_at_target intent;
    cubeA collides with cubeB on release and scatters."""
    cubeA_xy = np.asarray(scene.cube_xy, dtype=np.float64)
    cubeB_xy = np.asarray(scene.extra["cubeB_xy"], dtype=np.float64)
    tcp = np.asarray(scene.tcp_start_pose, dtype=np.float64)
    travel_z = float(tcp[2])
    cubeA_z = float(scene.cube_z)

    wp = np.zeros((4, 7), dtype=np.float64)
    wp[0, 0:2] = cubeA_xy
    wp[0, 2] = travel_z
    wp[1, 0:2] = cubeA_xy
    wp[1, 2] = cubeA_z + DESCEND_CLEARANCE_M
    wp[2, 0:2] = cubeA_xy
    wp[2, 2] = cubeA_z
    wp[3, 0:2] = cubeB_xy
    wp[3, 2] = cubeA_z + DESCEND_CLEARANCE_M
    wp[:, 3:7] = tcp[3:7]
    return wp


def _build_place_on_waypoints(scene: SceneState) -> np.ndarray:
    """5 waypoints: approach above cubeA, descend, grasp, lift over cubeB,
    descend onto cubeB top. Used for the cubeA_on_cubeB intent; cubeA
    settles onto cubeB after the gripper releases."""
    cubeA_xy = np.asarray(scene.cube_xy, dtype=np.float64)
    cubeB_xy = np.asarray(scene.extra["cubeB_xy"], dtype=np.float64)
    cubeB_top_z = float(scene.extra["cubeB_top_z"])
    tcp = np.asarray(scene.tcp_start_pose, dtype=np.float64)
    travel_z = float(tcp[2])
    cubeA_z = float(scene.cube_z)

    wp = np.zeros((5, 7), dtype=np.float64)
    wp[0, 0:2] = cubeA_xy
    wp[0, 2] = travel_z
    wp[1, 0:2] = cubeA_xy
    wp[1, 2] = cubeA_z + DESCEND_CLEARANCE_M
    wp[2, 0:2] = cubeA_xy
    wp[2, 2] = cubeA_z
    wp[3, 0:2] = cubeB_xy
    wp[3, 2] = travel_z
    wp[4, 0:2] = cubeB_xy
    wp[4, 2] = cubeB_top_z + CUBE_HALF_SIZE + PLACE_CLEARANCE_M
    wp[:, 3:7] = tcp[3:7]
    return wp


def compile_intent_to_stack_skill(
    intent: Intent, scene: SceneState,
) -> StackSkill:
    """Returns a StackSkill ready for the env_runner.

    Dispatches on `intent.goal_state`:
      - cube_at_target  → 4-waypoint translate-and-release (under-specified)
      - cubeA_on_cubeB  → 5-waypoint pick-and-place (correct)

    Raises ValueError for any other goal_state (defensive; Intent's
    whitelist already enforces the vocabulary, so this fires only on
    callers passing a goal_state not in the C-supported subset
    e.g. cube_lifted_at_target from PickCube).
    """
    cubeB_top_z = float(scene.extra["cubeB_top_z"])
    if intent.goal_state == "cube_at_target":
        return StackSkill(
            waypoints=_build_translate_waypoints(scene),
            cubeA_z=float(scene.cube_z),
            cubeB_top_z=cubeB_top_z,
            goal_state="cube_at_target",
        )
    if intent.goal_state == "cubeA_on_cubeB":
        return StackSkill(
            waypoints=_build_place_on_waypoints(scene),
            cubeA_z=float(scene.cube_z),
            cubeB_top_z=cubeB_top_z,
            goal_state="cubeA_on_cubeB",
        )
    raise ValueError(
        f"compile_intent_to_stack_skill: goal_state must be one of "
        f"{{'cube_at_target', 'cubeA_on_cubeB'}}, got {intent.goal_state!r}"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_stack_skill.py -v 2>&1 | tail -15`

Expected: all 10 tests PASS.

- [ ] **Step 5: Full suite**

Run: `python -m pytest tests/ -q 2>&1 | tail -5`

Expected: `201 passed` (191 + 10).

- [ ] **Step 6: Commit**

```bash
git add babysteps/skills/stack.py tests/test_stack_skill.py
git commit -m "$(cat <<'EOF'
feat(skills): StackSkill — Sub-project C pick-and-place compiler

Pure geometry compiler that dispatches on intent.goal_state:
  - cube_at_target  → 4-waypoint translate-and-drop at cubeB.xy (low z)
  - cubeA_on_cubeB  → 5-waypoint pick + lift over cubeB + place on top

The 4-vs-5 waypoint split is the Stage-0 controlled-failure mechanism:
wrong-goal waypoints physically miss the stack, no blocked_sides or
slip logic required.

10 sim-free unit tests cover waypoint shape, final-waypoint geometry
per goal_state, defensive ValueError on unknown goal_state, and
quaternion preservation.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `StackCubeAdapter` + `FakeStackCubeEnvRunner` + tests

**Files:**
- Create: `babysteps/envs/stackcube_adapter.py`
- Modify: `tests/conftest.py` (append `FakeStackCubeEnvRunner` + fixture)
- Test: `tests/test_stackcube_adapter.py`

- [ ] **Step 1: Append `FakeStackCubeEnvRunner` to `tests/conftest.py`**

At the end of `tests/conftest.py`, append:

```python
class FakeStackCubeEnvRunner:
    """Deterministic, sim-free env_runner for StackCube unit tests.

    Stage-0 controlled-failure mechanism: outcome is keyed entirely off
    intent.goal_state (no blocked_sides, no slip):
      - cubeA_on_cubeB → success=True, final_obj_xy = cubeB_xy
      - any other       → success=False, final_obj_xy = cubeB_xy + (0.025, 0)
                          (cubeA slid off after collision)
    """

    def __init__(self) -> None:
        self._scenes_by_seed: dict[int, SceneState] = {}

    def reset(self, seed: int) -> SceneState:
        if seed not in self._scenes_by_seed:
            # Deterministic synthetic scene per seed: cubeA at origin,
            # cubeB at (r*cos(θ), r*sin(θ)) with seed-derived r and θ.
            rng = np.random.default_rng(seed)
            r = float(rng.uniform(0.05, 0.12))
            theta = (seed % 4) * (np.pi / 2)
            cubeB_xy = (float(r * np.cos(theta)), float(r * np.sin(theta)))
            cubeB_z = 0.02
            self._scenes_by_seed[seed] = SceneState(
                cube_xy=(0.0, 0.0),
                cube_z=0.02,
                goal_xy=cubeB_xy,
                tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0),
                blocked_sides=(),
                extra={
                    "cubeB_xy": cubeB_xy,
                    "cubeB_z": cubeB_z,
                    "cubeB_top_z": cubeB_z + 0.04,
                },
            )
        return self._scenes_by_seed[seed]

    def run(self, intent: Intent, scene: SceneState) -> AttemptResult:
        # Compile-time sanity (should never raise for valid Intents).
        from babysteps.skills.stack import compile_intent_to_stack_skill
        skill = compile_intent_to_stack_skill(intent, scene)
        assert skill is not None

        cubeA_init = np.asarray(scene.cube_xy, dtype=np.float64)
        cubeB_xy = np.asarray(scene.extra["cubeB_xy"], dtype=np.float64)

        if intent.goal_state == "cubeA_on_cubeB":
            final_xy = (float(cubeB_xy[0]), float(cubeB_xy[1]))
            success = True
        else:
            # Cube slid off after collision — synthesize a small offset.
            final_xy = (float(cubeB_xy[0]) + 0.025, float(cubeB_xy[1]))
            success = False

        synthetic_traj = tuple(
            (float(cubeA_init[0] + (final_xy[0] - cubeA_init[0]) * t),
             float(cubeA_init[1] + (final_xy[1] - cubeA_init[1]) * t))
            for t in np.linspace(0.0, 1.0, 8)
        )
        return AttemptResult(
            initial_obj_xy=tuple(float(v) for v in cubeA_init),     # type: ignore[arg-type]
            final_obj_xy=final_xy,
            goal_xy=scene.goal_xy,
            reached_contact=True,
            object_moved=True,
            planner_failed=False,
            collision=False,
            grasp_slip=False,
            rollout_log_path=None,
            success=success,
            trajectory_xy=synthetic_traj,
        )

    def close(self) -> None:
        pass


@pytest.fixture
def fake_stack_env_runner() -> FakeStackCubeEnvRunner:
    return FakeStackCubeEnvRunner()
```

- [ ] **Step 2: Write the failing adapter tests**

Create `tests/test_stackcube_adapter.py`:

```python
"""Tests for babysteps/envs/stackcube_adapter.py.

Mirrors test_pickcube_adapter.py's shape: parity tests + snapshot test.
The snapshot bootstraps on first run, then enforces byte-equality."""
from __future__ import annotations

from pathlib import Path

import pytest

from babysteps.envs.stackcube_adapter import StackCubeAdapter
from babysteps.envs.task_adapter import BaseTaskAdapter
from babysteps.schemas import (
    CONTACT_REGIONS,
    DemoEvidence,
    Intent,
    SceneState,
)


# ---------- adapter API / parity tests ------------------------------- #


def test_task_id_is_stackcube_v1():
    assert StackCubeAdapter.task_id == "StackCube-v1"


def test_is_subclass_of_basetaskadapter():
    assert issubclass(StackCubeAdapter, BaseTaskAdapter)


def test_oracle_correct_intent_is_cubeA_on_cubeB():
    """The oracle knows the correct goal; uses 'cubeA_on_cubeB' with
    'place_on' motion and the pick-and-place embodiment."""
    scene = SceneState(
        cube_xy=(0.0, 0.0), cube_z=0.02, goal_xy=(0.1, 0.0),
        tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0),
        blocked_sides=(),
        extra={"cubeB_xy": (0.1, 0.0), "cubeB_z": 0.02, "cubeB_top_z": 0.06},
    )
    adapter = StackCubeAdapter()
    intent = adapter.oracle_correct_intent(scene)
    assert intent.goal_state == "cubeA_on_cubeB"
    assert intent.object_motion == "place_on"
    assert intent.embodiment_mapping == "proxy_contact_to_franka_pick_and_place"
    assert intent.approach_direction == "from_above"
    assert intent.constraint_region == "none"
    assert intent.contact_region in CONTACT_REGIONS


def test_default_blocked_factory_is_empty():
    """StackCube's controlled failure is from wrong-goal waypoints, not
    blocking — so default_blocked_factory always returns ()."""
    intent = Intent(
        goal_state="cube_at_target", object_motion="translate_+x",
        contact_region="minus_x_face", approach_direction="from_above",
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_pick_and_place",
    )
    adapter = StackCubeAdapter()
    assert adapter.default_blocked_factory(intent) == ()


def test_oracle_wrong_factor_for_under_specified_intent():
    """When the initial intent has goal_state=cube_at_target (the
    deliberately under-specified value), oracle_wrong_factor returns
    'goal_state'."""
    intent = Intent(
        goal_state="cube_at_target", object_motion="translate_+x",
        contact_region="minus_x_face", approach_direction="from_above",
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_pick_and_place",
    )
    scene = SceneState(
        cube_xy=(0.0, 0.0), cube_z=0.02, goal_xy=(0.1, 0.0),
        tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0),
        blocked_sides=(),
    )
    adapter = StackCubeAdapter()
    assert adapter.oracle_wrong_factor(intent, scene) == "goal_state"


def test_oracle_wrong_factor_for_already_correct_intent():
    """If the initial intent already has goal_state=cubeA_on_cubeB,
    nothing is wrong — return 'none'."""
    intent = Intent(
        goal_state="cubeA_on_cubeB", object_motion="place_on",
        contact_region="minus_x_face", approach_direction="from_above",
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_pick_and_place",
    )
    scene = SceneState(
        cube_xy=(0.0, 0.0), cube_z=0.02, goal_xy=(0.1, 0.0),
        tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0),
        blocked_sides=(),
    )
    adapter = StackCubeAdapter()
    assert adapter.oracle_wrong_factor(intent, scene) == "none"


def test_scripted_demo_to_intent_always_under_specifies_goal():
    """The Stage-0 controlled mechanism: scripted_demo_to_intent always
    returns goal_state='cube_at_target' regardless of the demo's true
    final state."""
    evidence = DemoEvidence(
        camera="third_person",
        demonstrator_type="proxy_oracle",
        object_trajectory=((0.0, 0.0), (0.10, 0.0)),
        contact_region_label="minus_x_face",
        # The demo's TRUE final_state is cubeA_on_cubeB — but the
        # summarizer doesn't see the vertical component.
        final_state="cubeA_on_cubeB",
        rgbd_video_path=None,
    )
    adapter = StackCubeAdapter()
    intent = adapter.scripted_demo_to_intent(evidence)
    assert intent.goal_state == "cube_at_target"   # under-specified
    # object_motion derived from dominant 2D axis (cubeA → cubeB.xy).
    assert intent.object_motion in {
        "translate_+x", "translate_-x", "translate_+y", "translate_-y",
    }


def test_scripted_demo_to_intent_object_motion_matches_trajectory():
    """object_motion reflects the dominant axis of the 2D trajectory."""
    # +x dominant
    evidence_px = DemoEvidence(
        camera="third_person", demonstrator_type="proxy_oracle",
        object_trajectory=((0.0, 0.0), (0.10, 0.01)),
        contact_region_label="minus_x_face", final_state="cubeA_on_cubeB",
        rgbd_video_path=None,
    )
    # -y dominant
    evidence_my = DemoEvidence(
        camera="third_person", demonstrator_type="proxy_oracle",
        object_trajectory=((0.0, 0.0), (0.01, -0.10)),
        contact_region_label="minus_x_face", final_state="cubeA_on_cubeB",
        rgbd_video_path=None,
    )
    adapter = StackCubeAdapter()
    assert adapter.scripted_demo_to_intent(evidence_px).object_motion == "translate_+x"
    assert adapter.scripted_demo_to_intent(evidence_my).object_motion == "translate_-y"


def test_scripted_demo_to_intent_uses_pick_and_place_embodiment():
    evidence = DemoEvidence(
        camera="third_person", demonstrator_type="proxy_oracle",
        object_trajectory=((0.0, 0.0), (0.1, 0.0)),
        contact_region_label="minus_x_face", final_state="cubeA_on_cubeB",
        rgbd_video_path=None,
    )
    adapter = StackCubeAdapter()
    intent = adapter.scripted_demo_to_intent(evidence)
    assert intent.embodiment_mapping == "proxy_contact_to_franka_pick_and_place"


def test_scripted_demo_to_intent_rejects_bad_contact_region():
    """Invalid contact_region_label raises ValueError (defensive
    consistent with Push/Pick adapters)."""
    evidence = DemoEvidence(
        camera="third_person", demonstrator_type="proxy_oracle",
        object_trajectory=((0.0, 0.0), (0.1, 0.0)),
        contact_region_label="bogus_face",
        final_state="cubeA_on_cubeB", rgbd_video_path=None,
    )
    adapter = StackCubeAdapter()
    with pytest.raises(ValueError):
        adapter.scripted_demo_to_intent(evidence)


def test_compile_skill_delegates_to_stack_skill():
    """compile_skill wraps compile_intent_to_stack_skill from the skill module."""
    from babysteps.skills.stack import StackSkill
    intent = Intent(
        goal_state="cubeA_on_cubeB", object_motion="place_on",
        contact_region="minus_x_face", approach_direction="from_above",
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_pick_and_place",
    )
    scene = SceneState(
        cube_xy=(0.0, 0.0), cube_z=0.02, goal_xy=(0.1, 0.0),
        tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0),
        blocked_sides=(),
        extra={"cubeB_xy": (0.1, 0.0), "cubeB_z": 0.02, "cubeB_top_z": 0.06},
    )
    adapter = StackCubeAdapter()
    skill = adapter.compile_skill(intent, scene)
    assert isinstance(skill, StackSkill)


def test_adapter_inherits_default_hooks():
    """StackCubeAdapter does not override the three optional hooks —
    failure attribution and revision use the shared modules unchanged."""
    assert (
        StackCubeAdapter.build_failure_packet
        is BaseTaskAdapter.build_failure_packet
    )
    assert (
        StackCubeAdapter.attribute_failure
        is BaseTaskAdapter.attribute_failure
    )
    assert (
        StackCubeAdapter.revise_intent
        is BaseTaskAdapter.revise_intent
    )


# ---------- end-to-end episode loop test ------------------------------ #


def test_full_episode_via_fake_runner_recovers_via_goal_refinement(
    fake_stack_env_runner,
):
    """One round-trip through run_episode: scripted intent under-specifies →
    goal_not_satisfied → goal_refinement → revised retry succeeds."""
    from babysteps.episode import run_episode

    class _Adapter(StackCubeAdapter):
        def make_env_runner(self):
            return fake_stack_env_runner

    rec = run_episode(
        episode_id="stackcube_underspec_goal_seed_0000",
        seed=0,
        adapter=_Adapter(),
    )
    assert rec.metrics["initial_success"] is False
    assert rec.metrics["retry_success"] is True
    assert rec.metrics["factor_attribution_correct"] is True
    assert rec.metrics["frozen_factors_preserved"] is True
    assert rec.metrics["factors_changed"] == ["goal_state"]
    assert rec.revision is not None
    assert rec.revision["operator"] == "goal_refinement"
    assert rec.revision["factor"] == "goal_state"
    assert rec.revision["old_value"] == "cube_at_target"
    assert rec.revision["new_value"] == "cubeA_on_cubeB"


# ---------- Snapshot acceptance test --------------------------------- #


def test_stackcube_adapter_samples_jsonl_matches_snapshot(fake_stack_env_runner):
    """Generates 5 episodes via fake runner and asserts byte-equality with
    tests/snapshots/stackcube_samples_seeds_0_4.jsonl.

    First-run convenience: if the snapshot does not exist, capture it and
    skip. Subsequent runs verify byte-equality. The same snapshot is
    asserted from the CLI side by tests/test_stage0_collect_cli.py."""
    from babysteps.episode import run_episode

    class _Adapter(StackCubeAdapter):
        def make_env_runner(self):
            return fake_stack_env_runner

    adapter = _Adapter()
    actual_lines = []
    for seed in range(5):
        rec = run_episode(
            episode_id=f"stackcube_underspec_goal_seed_{seed:04d}",
            seed=seed,
            adapter=adapter,
        )
        actual_lines.append(rec.to_jsonl_line())
    actual = "\n".join(actual_lines) + "\n"

    snapshot_path = (
        Path(__file__).parent / "snapshots" / "stackcube_samples_seeds_0_4.jsonl"
    )
    if not snapshot_path.exists():
        # First-run convenience: capture the snapshot.
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(actual)
        pytest.skip(
            f"Captured initial snapshot at {snapshot_path}. Re-run to "
            f"verify byte-equality."
        )
    expected = snapshot_path.read_text()
    assert actual == expected, (
        "StackCubeAdapter samples.jsonl drifted from the snapshot. "
        f"Snapshot at: {snapshot_path}. "
        "If intentional, delete the snapshot file and re-run this test "
        "to re-capture."
    )
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_stackcube_adapter.py -v 2>&1 | tail -20`

Expected: 13 tests FAIL with `ModuleNotFoundError: No module named 'babysteps.envs.stackcube_adapter'`. The snapshot test will fail at adapter import too (before reaching the bootstrap code).

- [ ] **Step 4: Create `babysteps/envs/stackcube_adapter.py`**

```python
"""StackCube-v1 adapter — the third concrete BaseTaskAdapter.

Pulls every StackCube-specific decision behind one class:
  * make_env_runner       → StackCubeEnvRunner (Task 5)
  * oracle_correct_intent → cubeA_on_cubeB / place_on / pick-and-place
  * default_blocked_factory → () — no physical blocking; the Stage-0
                              controlled failure is from wrong-goal
                              waypoints, not from blocked_sides
  * oracle_wrong_factor   → "goal_state" if intent.goal_state ==
                            "cube_at_target", else "none"
  * scripted_demo_to_intent → DELIBERATELY returns goal_state=
                              "cube_at_target". The 2D trajectory
                              summarization can't see vertical motion,
                              so the demo's true stacking is hidden.
  * compile_skill         → wraps skills.stack.compile_intent_to_stack_skill

Hook defaults (build_failure_packet / attribute_failure / revise_intent)
are inherited unchanged from BaseTaskAdapter — the goal_refinement
operator (Task 2) lives in revision.py and dispatches on
attribution.wrong_factor='goal_state' automatically."""
from __future__ import annotations

import numpy as np

from babysteps.envs.task_adapter import BaseTaskAdapter, EnvRunner
from babysteps.schemas import CONTACT_REGIONS, DemoEvidence, Intent, SceneState
from babysteps.skills.stack import compile_intent_to_stack_skill


# Default contact_region for the oracle — pick-and-place doesn't depend
# strongly on which face is grasped (parallel-jaw + lift); pick the same
# canonical value as PickCubeAdapter so the snapshot files have a
# consistent contact_region across tasks.
_DEFAULT_CONTACT_REGION: str = "minus_x_face"


def _dominant_axis_motion(traj: tuple[tuple[float, float], ...]) -> str:
    """Pick translate_<axis> from the (initial, final) trajectory's dominant
    component. Matches the convention used by PushCubeAdapter."""
    if len(traj) < 2:
        return "translate_+x"   # degenerate; arbitrary default
    dx = traj[-1][0] - traj[0][0]
    dy = traj[-1][1] - traj[0][1]
    if abs(dx) >= abs(dy):
        return "translate_+x" if dx >= 0 else "translate_-x"
    return "translate_+y" if dy >= 0 else "translate_-y"


class StackCubeAdapter(BaseTaskAdapter):
    task_id = "StackCube-v1"

    def make_env_runner(self) -> EnvRunner:
        from babysteps.envs.stackcube_runner import StackCubeEnvRunner
        return StackCubeEnvRunner()

    def oracle_correct_intent(self, scene: SceneState) -> Intent:
        return Intent(
            goal_state="cubeA_on_cubeB",
            object_motion="place_on",
            contact_region=_DEFAULT_CONTACT_REGION,
            approach_direction="from_above",
            constraint_region="none",
            embodiment_mapping="proxy_contact_to_franka_pick_and_place",
        )

    def default_blocked_factory(self, intent: Intent) -> tuple[str, ...]:
        # No physical blocking — the Stage-0 controlled failure comes
        # from the under-specified goal_state in scripted_demo_to_intent.
        return ()

    def oracle_wrong_factor(
        self, initial_intent: Intent, scene_executor: SceneState,
    ) -> str:
        if initial_intent.goal_state == "cube_at_target":
            return "goal_state"
        return "none"

    def scripted_demo_to_intent(self, evidence: DemoEvidence) -> Intent:
        contact = evidence.contact_region_label
        if contact not in CONTACT_REGIONS:
            raise ValueError(
                f"DemoEvidence.contact_region_label must be one of "
                f"{sorted(CONTACT_REGIONS)}, got {contact!r}"
            )
        motion = _dominant_axis_motion(evidence.object_trajectory)
        return Intent(
            goal_state="cube_at_target",   # DELIBERATELY under-specified
            object_motion=motion,
            contact_region=contact,
            approach_direction="from_above",
            constraint_region="none",
            embodiment_mapping="proxy_contact_to_franka_pick_and_place",
        )

    def compile_skill(self, intent: Intent, scene: SceneState):
        return compile_intent_to_stack_skill(intent, scene)
```

- [ ] **Step 5: Run the API tests to verify they pass; the snapshot test will bootstrap**

Run: `python -m pytest tests/test_stackcube_adapter.py -v 2>&1 | tail -25`

Expected: 12 of 13 tests PASS; `test_stackcube_adapter_samples_jsonl_matches_snapshot` SKIPs with the bootstrap message. The snapshot file is now created at `tests/snapshots/stackcube_samples_seeds_0_4.jsonl`.

- [ ] **Step 6: Re-run the snapshot test to confirm byte-equality**

Run: `python -m pytest tests/test_stackcube_adapter.py::test_stackcube_adapter_samples_jsonl_matches_snapshot -v`

Expected: PASS (snapshot now exists; assertion matches).

- [ ] **Step 7: Full suite — should pick up adapter + snapshot tests**

Run: `python -m pytest tests/ -q 2>&1 | tail -5`

Expected: `214 passed` (201 + 13 new from test_stackcube_adapter.py). The bootstrap-then-verify ordering above means the snapshot test counts as PASS on this run.

- [ ] **Step 8: Commit (bundle: conftest + adapter + tests + snapshot)**

```bash
git add babysteps/envs/stackcube_adapter.py tests/conftest.py tests/test_stackcube_adapter.py tests/snapshots/stackcube_samples_seeds_0_4.jsonl
git commit -m "$(cat <<'EOF'
feat(c): StackCubeAdapter + FakeStackCubeEnvRunner + snapshot

- babysteps/envs/stackcube_adapter.py — 5-method BaseTaskAdapter
  subclass. scripted_demo_to_intent deliberately under-specifies
  goal_state to cube_at_target (the Stage-0 controlled mechanism);
  default_blocked_factory returns () since failure comes from wrong
  waypoints, not blocking.
- tests/conftest.py — FakeStackCubeEnvRunner deterministic sim-free
  runner. Success iff intent.goal_state == cubeA_on_cubeB. cubeB.xy
  generated by seed (matches the FakeEnv/FakePick pattern).
- tests/test_stackcube_adapter.py — 13 tests: 11 adapter API parity,
  1 full-episode (run_episode + goal_refinement round-trip), 1
  snapshot byte-stability (bootstraps on first run, then enforces).
- tests/snapshots/stackcube_samples_seeds_0_4.jsonl — captured 5-seed
  snapshot.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: `StackCubeEnvRunner` (real ManiSkill)

**Files:**
- Create: `babysteps/envs/stackcube_runner.py`

No unit tests in this task — the runner needs Vulkan/CUDA, which isn't available on the login node. Its correctness is verified by the GPU spot-check (Task 11 manual step). The fake runner from Task 4 covers all sim-free flows.

- [ ] **Step 1: Create `babysteps/envs/stackcube_runner.py`**

```python
"""Real ManiSkill StackCube-v1 env_runner.

Mirrors babysteps/envs/pickcube_runner.py's structure. The key
differences:

  - StackCube obs has cubeA_pose and cubeB_pose (no goal_pos).
  - The waypoint count is goal_state-dependent: 4 phases for
    cube_at_target (translate-and-drop), 5 for cubeA_on_cubeB
    (pick-and-place). The runner reads skill.waypoints.shape[0] and
    builds the gripper schedule accordingly.
  - Per-phase gripper schedule:
      4 phases (cube_at_target):  [OPEN, OPEN, CLOSED, OPEN]
                                   (release at translate-release)
      5 phases (cubeA_on_cubeB):  [OPEN, OPEN, CLOSED, CLOSED, OPEN]
                                   (release only at place_on)
  - No slip mechanism; no blocked_sides logic. Stage-0's controlled
    failure for StackCube is purely from the under-specified
    goal_state (handled by the skill compiler's branch selection).

Note on Gilbreth: requires a GPU/Vulkan node (same as PushCube/PickCube
runners)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from babysteps.schemas import AttemptResult, Intent, SceneState
from babysteps.skills.stack import (
    CUBE_HALF_SIZE,
    compile_intent_to_stack_skill,
)


# Phase-control constants — match PickCubeEnvRunner's PD calibration.
_POS_SCALE: float = 0.1
_PHASE_TOL_M: float = 0.015
_MAX_CONTROL_STEPS: int = 400         # matches PickCubeEnvRunner

_GRIPPER_OPEN: float = 1.0
_GRIPPER_CLOSED: float = -1.0


def _to_np(x):
    arr = x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)
    return arr[0] if arr.ndim == 2 else arr


def _raw_to_xyzw(raw_pose) -> np.ndarray:
    raw = np.asarray(raw_pose, dtype=np.float64)
    return np.concatenate([raw[0:3], raw[4:7], raw[3:4]])


def _read_obs(
    obs,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, float]:
    """(tcp_xyzw, cubeA_xy, cubeA_z, cubeB_xy, cubeB_z) from StackCube obs.

    StackCube-v1's obs.extra has cubeA_pose, cubeB_pose, tcp_pose. No
    goal_pos — the "goal" is implicit (cubeB.xy + cube_height)."""
    tcp = _raw_to_xyzw(_to_np(obs["extra"]["tcp_pose"]))
    cubeA_full = _to_np(obs["extra"]["cubeA_pose"])
    cubeA_xy = cubeA_full[0:2].astype(np.float64)
    cubeA_z = float(cubeA_full[2])
    cubeB_full = _to_np(obs["extra"]["cubeB_pose"])
    cubeB_xy = cubeB_full[0:2].astype(np.float64)
    cubeB_z = float(cubeB_full[2])
    return tcp, cubeA_xy, cubeA_z, cubeB_xy, cubeB_z


def _prop_action(
    tcp_xyzw: np.ndarray, target_xyz: np.ndarray, gripper_cmd: float,
) -> np.ndarray:
    pos_err = target_xyz - tcp_xyzw[0:3]
    action = np.zeros(7, dtype=np.float32)
    action[0:3] = np.clip(pos_err / _POS_SCALE, -1.0, 1.0).astype(np.float32)
    action[6] = np.float32(gripper_cmd)
    return action


def _gripper_at_cubeA(
    tcp: np.ndarray, cubeA_xy: np.ndarray, cubeA_z: float,
    *, threshold: float = 0.04,
) -> bool:
    dxy = float(np.linalg.norm(tcp[0:2] - np.asarray(cubeA_xy, dtype=np.float64)))
    dz = abs(float(tcp[2]) - float(cubeA_z))
    return dxy < threshold and dz < threshold


class StackCubeEnvRunner:
    """Real ManiSkill StackCube-v1 runner.

    Lazy-imports mani_skill on construction. Holds one gym env across
    multiple run(...) calls; each run internally resets to the captured
    seed before executing the compiled stack trajectory."""

    def __init__(self) -> None:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401 — registers StackCube-v1

        self._env = gym.make(
            "StackCube-v1",
            obs_mode="state_dict",
            control_mode="pd_ee_delta_pose",
            sim_backend="cpu",
        )
        self._last_seed: Optional[int] = None

    def reset(self, seed: int) -> SceneState:
        self._last_seed = int(seed)
        obs, _info = self._env.reset(seed=int(seed))
        tcp, cubeA_xy, cubeA_z, cubeB_xy, cubeB_z = _read_obs(obs)
        cubeB_top_z = cubeB_z + 2 * CUBE_HALF_SIZE
        return SceneState(
            cube_xy=(float(cubeA_xy[0]), float(cubeA_xy[1])),
            cube_z=cubeA_z,
            # Convenience: scene.goal_xy = cubeB.xy so existing
            # scene-reading callers (metrics computers, frame-by-frame
            # render utilities) work without StackCube-specific branches.
            goal_xy=(float(cubeB_xy[0]), float(cubeB_xy[1])),
            tcp_start_pose=tuple(float(v) for v in tcp),  # type: ignore[arg-type]
            blocked_sides=(),
            extra={
                "cubeB_xy": (float(cubeB_xy[0]), float(cubeB_xy[1])),
                "cubeB_z": cubeB_z,
                "cubeB_top_z": cubeB_top_z,
            },
        )

    def run(
        self,
        intent: Intent,
        scene: SceneState,
        *,
        rollout_log_path: Optional[Path] = None,
    ) -> AttemptResult:
        skill = compile_intent_to_stack_skill(intent, scene)
        # StackSkill never returns None — defensive only.

        if self._last_seed is None:
            raise RuntimeError("StackCubeEnvRunner.run called before reset()")
        obs, _info = self._env.reset(seed=int(self._last_seed))
        _tcp0, cubeA_xy0, _cubeA_z0, _cubeB_xy0, _cubeB_z0 = _read_obs(obs)
        initial_obj_xy = (float(cubeA_xy0[0]), float(cubeA_xy0[1]))

        targets: list[np.ndarray] = [
            np.asarray(wp[0:3], dtype=np.float64) for wp in skill.waypoints
        ]

        n_phases = len(targets)
        if n_phases == 4:
            # cube_at_target: [approach, descend, grasp, translate-release]
            phase_gripper = (
                _GRIPPER_OPEN, _GRIPPER_OPEN, _GRIPPER_CLOSED, _GRIPPER_OPEN,
            )
        elif n_phases == 5:
            # cubeA_on_cubeB: [approach, descend, grasp, lift, place_on]
            phase_gripper = (
                _GRIPPER_OPEN, _GRIPPER_OPEN, _GRIPPER_CLOSED,
                _GRIPPER_CLOSED, _GRIPPER_OPEN,
            )
        else:
            raise RuntimeError(
                f"StackCubeEnvRunner: unexpected waypoint count {n_phases}; "
                "expected 4 (cube_at_target) or 5 (cubeA_on_cubeB)"
            )

        trajectory: list[tuple[float, float]] = []
        phase_idx = 0
        reached_contact = False
        success = False

        for _step in range(_MAX_CONTROL_STEPS):
            tcp, cubeA_xy, _cubeA_z, _cubeB_xy, _cubeB_z = _read_obs(obs)
            trajectory.append((float(cubeA_xy[0]), float(cubeA_xy[1])))
            target = targets[phase_idx]
            if np.linalg.norm(target - tcp[0:3]) < _PHASE_TOL_M:
                phase_idx += 1
                if phase_idx >= n_phases:
                    break
                target = targets[phase_idx]
            # Contact heuristic: TCP near cubeA, post-approach.
            if phase_idx >= 1:
                reached_contact = reached_contact or _gripper_at_cubeA(
                    tcp, cubeA_xy, skill.cubeA_z,
                )
            action = _prop_action(tcp, target, phase_gripper[phase_idx])
            obs, _r, terminated, truncated, info = self._env.step(action)
            term = bool(_to_np(terminated).item()) if hasattr(terminated, "cpu") else bool(terminated)
            trunc = bool(_to_np(truncated).item()) if hasattr(truncated, "cpu") else bool(truncated)
            succ_field = info.get("success", False) if hasattr(info, "get") else False
            success = bool(_to_np(succ_field).item()) if hasattr(succ_field, "cpu") else bool(succ_field)
            if success or term or trunc:
                break

        _tcp_f, final_cubeA_xy, _, _, _ = _read_obs(obs)
        final_obj_xy = (float(final_cubeA_xy[0]), float(final_cubeA_xy[1]))
        trajectory.append(final_obj_xy)

        object_moved = (
            float(np.linalg.norm(
                np.asarray(final_obj_xy) - np.asarray(initial_obj_xy)
            )) > 0.005
        )

        if rollout_log_path is not None:
            rollout_log_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                rollout_log_path,
                trajectory_xy=np.asarray(trajectory, dtype=np.float64),
                initial_obj_xy=np.asarray(initial_obj_xy, dtype=np.float64),
                final_obj_xy=np.asarray(final_obj_xy, dtype=np.float64),
                goal_xy=np.asarray(scene.goal_xy, dtype=np.float64),
            )

        return AttemptResult(
            initial_obj_xy=initial_obj_xy,
            final_obj_xy=final_obj_xy,
            goal_xy=scene.goal_xy,
            reached_contact=bool(reached_contact),
            object_moved=bool(object_moved),
            planner_failed=False,
            collision=False,
            grasp_slip=False,
            rollout_log_path=str(rollout_log_path) if rollout_log_path else None,
            success=bool(success),
            trajectory_xy=tuple(trajectory),
        )

    def close(self) -> None:
        try:
            self._env.close()
        except Exception:
            pass
```

- [ ] **Step 2: Confirm module imports cleanly without invoking ManiSkill**

Run:
```bash
python -c "from babysteps.envs.stackcube_runner import StackCubeEnvRunner; print('class:', StackCubeEnvRunner.__name__)"
```

Expected: prints `class: StackCubeEnvRunner` without error. (ManiSkill import is deferred to `__init__`, so the class definition itself doesn't pull mani_skill at module load.)

- [ ] **Step 3: Full suite — no new tests, no regressions**

Run: `python -m pytest tests/ -q 2>&1 | tail -5`

Expected: still `214 passed`.

- [ ] **Step 4: Commit**

```bash
git add babysteps/envs/stackcube_runner.py
git commit -m "$(cat <<'EOF'
feat(c): StackCubeEnvRunner — real ManiSkill StackCube-v1 runner

Mirrors PickCubeEnvRunner's structure. Differences:
- Reads cubeA_pose and cubeB_pose from obs.extra (no goal_pos).
- Phase count is goal_state-dependent: 4 phases for cube_at_target
  (translate-and-drop), 5 for cubeA_on_cubeB (pick-and-place). The
  runner reads skill.waypoints.shape[0] to pick the right per-phase
  gripper schedule.
- No slip mechanism; no blocked_sides logic. Stage-0's controlled
  failure for StackCube is purely from the wrong-goal waypoint shape
  (handled by the skill compiler's branch selection).

Correctness verified by the GPU spot-check in CLAUDE.md (Task 11
manual step). The sim-free FakeStackCubeEnvRunner from Task 4
covers all login-node test flows.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Registry entries + parity test

**Files:**
- Modify: `babysteps/envs/task_registry.py`
- Modify: `babysteps/render/__init__.py`
- Modify: `tests/test_task_registry.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_task_registry.py`:

```python
def test_get_task_entry_stackcube():
    from babysteps.envs.stackcube_adapter import StackCubeAdapter
    entry = get_task_entry("StackCube-v1")
    assert isinstance(entry, TaskEntry)
    assert entry.adapter_cls is StackCubeAdapter
    assert entry.episode_id_prefix == "stackcube_underspec_goal"


def test_fake_runner_factory_stackcube():
    entry = get_task_entry("StackCube-v1")
    runner = entry.fake_runner_factory()
    assert hasattr(runner, "reset")
    assert hasattr(runner, "run")
    assert hasattr(runner, "close")
    runner.close()
```

Note: `test_task_registry_matches_render_registry` (added in B's polish) will start failing once we add StackCube-v1 to one registry — it must be added to BOTH in the same commit. Plan keeps both updates together.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_task_registry.py -v 2>&1 | tail -15`

Expected: `test_get_task_entry_stackcube`, `test_fake_runner_factory_stackcube`, and `test_registry_contains_both_stage0_tasks` all FAIL. `test_task_registry_matches_render_registry` will also fail once we add to TASK_REGISTRY but not RENDER_REGISTRY — that's why we update both in the same commit.

Wait — `test_registry_contains_both_stage0_tasks` was written in B's plan to assert `set(TASK_REGISTRY.keys()) == {"PushCube-v1", "PickCube-v1"}`. Adding StackCube-v1 makes this fail. Update it as part of this task too:

In `tests/test_task_registry.py`, find:

```python
def test_registry_contains_both_stage0_tasks():
    """PushCube-v1 (sub-project A) and PickCube-v1 (sub-project B) must be present."""
    assert set(TASK_REGISTRY.keys()) == {"PushCube-v1", "PickCube-v1"}
```

Replace with:

```python
def test_registry_contains_all_stage0_tasks():
    """PushCube-v1 (A), PickCube-v1 (B), and StackCube-v1 (C) must be present."""
    assert set(TASK_REGISTRY.keys()) == {
        "PushCube-v1", "PickCube-v1", "StackCube-v1",
    }
```

Also update `test_get_task_entry_unknown_task_raises` to use a task that's still unknown (the previous one, `StackCube-v1`, is now known). Find:

```python
def test_get_task_entry_unknown_task_raises():
    with pytest.raises(KeyError) as exc:
        get_task_entry("StackCube-v1")
    msg = str(exc.value)
    assert "StackCube-v1" in msg
    assert "PushCube-v1" in msg
    assert "PickCube-v1" in msg
```

Replace with:

```python
def test_get_task_entry_unknown_task_raises():
    with pytest.raises(KeyError) as exc:
        get_task_entry("OpenCabinetDrawer-v1")
    msg = str(exc.value)
    assert "OpenCabinetDrawer-v1" in msg
    assert "PushCube-v1" in msg
    assert "PickCube-v1" in msg
    assert "StackCube-v1" in msg
```

- [ ] **Step 3: Add the registry entry in `babysteps/envs/task_registry.py`**

Find the `_pickcube_entry()` function block. After it, before `TASK_REGISTRY = {...}`, add:

```python
def _stackcube_entry() -> TaskEntry:
    # StackCubeAdapter is safe to import at module load — it does not
    # pull mani_skill (deferred to make_env_runner()). FakeStackCube-
    # EnvRunner is lazy via the _make_fake closure.
    from babysteps.envs.stackcube_adapter import StackCubeAdapter

    def _make_fake() -> EnvRunner:
        from tests.conftest import FakeStackCubeEnvRunner
        return FakeStackCubeEnvRunner()

    return TaskEntry(
        adapter_cls=StackCubeAdapter,
        fake_runner_factory=_make_fake,
        episode_id_prefix="stackcube_underspec_goal",
    )
```

Then find:

```python
TASK_REGISTRY: dict[str, TaskEntry] = {
    "PushCube-v1": _pushcube_entry(),
    "PickCube-v1": _pickcube_entry(),
}
```

Replace with:

```python
TASK_REGISTRY: dict[str, TaskEntry] = {
    "PushCube-v1": _pushcube_entry(),
    "PickCube-v1": _pickcube_entry(),
    "StackCube-v1": _stackcube_entry(),
}
```

- [ ] **Step 4: Add the render-registry entry in `babysteps/render/__init__.py`**

Find the `_pickcube_render()` function block. After it, before `RENDER_REGISTRY = {...}`, add:

```python
def _stackcube_render() -> RenderEpisodeFn:
    from babysteps.render.stackcube import render_episode
    return render_episode
```

Then find:

```python
RENDER_REGISTRY: dict[str, Callable[[], RenderEpisodeFn]] = {
    "PushCube-v1": _pushcube_render,
    "PickCube-v1": _pickcube_render,
}
```

Replace with:

```python
RENDER_REGISTRY: dict[str, Callable[[], RenderEpisodeFn]] = {
    "PushCube-v1": _pushcube_render,
    "PickCube-v1": _pickcube_render,
    "StackCube-v1": _stackcube_render,
}
```

(Note: `babysteps.render.stackcube` is created in Task 7. Until then, `RENDER_REGISTRY["StackCube-v1"]()` would raise ImportError. The lazy factory means the import only fires when `get_render_fn("StackCube-v1")` is called — `RENDER_REGISTRY` itself imports cleanly. Tests that just check `set(RENDER_REGISTRY.keys())` will pass; tests that actually call the factory would fail. We address that in Task 7 by creating the module.)

- [ ] **Step 5: Run task_registry tests to verify they now pass**

Run: `python -m pytest tests/test_task_registry.py -v 2>&1 | tail -15`

Expected: all 10 tests in `test_task_registry.py` PASS (was 8, gains 2 from this task + the two updates above counted as the same tests; net +2 to 10).

- [ ] **Step 6: Full suite**

Run: `python -m pytest tests/ -q 2>&1 | tail -5`

Expected: `216 passed` (214 + 2 new). The render module doesn't exist yet, but no test imports it directly — `RENDER_REGISTRY["StackCube-v1"]` is a lazy factory.

- [ ] **Step 7: Commit**

```bash
git add babysteps/envs/task_registry.py babysteps/render/__init__.py tests/test_task_registry.py
git commit -m "$(cat <<'EOF'
feat(c): wire StackCube-v1 into TASK_REGISTRY and RENDER_REGISTRY

One-row additions to both dispatch tables. The render module
(babysteps/render/stackcube.py) is created in the next task; the
lazy factory means RENDER_REGISTRY itself imports cleanly until
that arrives. Updates the parity-related task_registry tests to
include the new task and changes test_get_task_entry_unknown_task_
raises to use OpenCabinetDrawer-v1 as the unknown (StackCube-v1
is now known).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Render module + stub-env tests

**Files:**
- Modify: `babysteps/render/common.py` (add `STACKCUBE_MAX_CONTROL_STEPS`)
- Create: `babysteps/render/stackcube.py`
- Modify: `tests/test_render_modules.py` (append 3 StackCube tests)

- [ ] **Step 1: Add the StackCube cap to `babysteps/render/common.py`**

Find:

```python
PUSHCUBE_MAX_CONTROL_STEPS: int = 300   # matches PushCubeEnvRunner._MAX_CONTROL_STEPS
PICKCUBE_MAX_CONTROL_STEPS: int = 400   # matches PickCubeEnvRunner._MAX_CONTROL_STEPS
MAX_CONTROL_STEPS: int = 400   # back-compat alias for callers that don't care
```

Replace with:

```python
PUSHCUBE_MAX_CONTROL_STEPS: int = 300   # matches PushCubeEnvRunner._MAX_CONTROL_STEPS
PICKCUBE_MAX_CONTROL_STEPS: int = 400   # matches PickCubeEnvRunner._MAX_CONTROL_STEPS
STACKCUBE_MAX_CONTROL_STEPS: int = 400  # matches StackCubeEnvRunner._MAX_CONTROL_STEPS
MAX_CONTROL_STEPS: int = 400   # back-compat alias for callers that don't care
```

- [ ] **Step 2: Append failing StackCube tests to `tests/test_render_modules.py`**

```python
# ---------- StackCube render tests ------------------------------------ #


class _StubStackEnv:
    """Stand-in for gym.make('StackCube-v1') used in stackcube render tests.

    Like _StubEnv but the obs has cubeA_pose + cubeB_pose (no goal_pos).
    The TCP integrates the action's xyz so phase transitions happen in
    ~10 stub steps."""

    def __init__(self) -> None:
        self.tcp = np.array([0.0, 0.0, 0.25, 0.0, 0.0, 0.0], dtype=np.float64)
        self.cubeA = np.array([0.0, 0.0], dtype=np.float64)
        self.cubeB = np.array([0.10, 0.0], dtype=np.float64)
        self._step_count = 0

    def reset(self, seed: int = 0):
        self.tcp = np.array([0.0, 0.0, 0.25, 0.0, 0.0, 0.0], dtype=np.float64)
        self.cubeA = np.array([0.0, 0.0], dtype=np.float64)
        self.cubeB = np.array([0.10, 0.0], dtype=np.float64)
        self._step_count = 0
        return _StubStackObs(self.tcp, self.cubeA, self.cubeB), {}

    def step(self, action):
        self.tcp[0:3] = self.tcp[0:3] + 0.02 * np.asarray(action[0:3])
        self._step_count += 1
        return (
            _StubStackObs(self.tcp, self.cubeA, self.cubeB),
            0.0, False, False,
            {"success": False},
        )

    def render(self):
        return (np.ones((8, 8, 3), dtype=np.uint8) * (self._step_count % 256))

    def close(self):
        pass


from dataclasses import dataclass as _dc


@_dc
class _StubStackObs:
    tcp: np.ndarray
    cubeA: np.ndarray
    cubeB: np.ndarray

    def __getitem__(self, key: str):
        if key == "extra":
            tcp_raw = np.concatenate([self.tcp[0:3], np.array([1.0]),
                                      self.tcp[3:6]])
            cubeA_full = np.array([self.cubeA[0], self.cubeA[1], 0.02])
            cubeB_full = np.array([self.cubeB[0], self.cubeB[1], 0.02])
            return {
                "tcp_pose": tcp_raw,
                "cubeA_pose": cubeA_full,
                "cubeB_pose": cubeB_full,
            }
        raise KeyError(key)


def test_stackcube_render_episode_emits_three_phase_frames():
    from babysteps.render.stackcube import render_episode
    from babysteps.envs.stackcube_adapter import StackCubeAdapter

    env = _StubStackEnv()
    adapter = StackCubeAdapter()
    frames, titles = render_episode(env, adapter, seed=0, fps=4)

    assert set(frames.keys()) == {"demo", "attempt_blocked", "retry"}
    assert set(titles.keys()) == {"demo", "attempt_blocked", "retry"}
    assert len(frames["demo"]) >= 2
    assert len(frames["attempt_blocked"]) >= 2
    assert len(frames["retry"]) >= 2


def test_stackcube_render_phase2_actually_steps_env():
    """Like PickCube and unlike PushCube — phase 2 steps the env so the
    failed translate-and-drop is visible. Detect by checking the stub
    env's step_count varies the frame intensity."""
    from babysteps.render.stackcube import render_episode
    from babysteps.envs.stackcube_adapter import StackCubeAdapter

    env = _StubStackEnv()
    frames, _ = render_episode(env, StackCubeAdapter(), seed=0, fps=4)
    held = frames["attempt_blocked"]
    assert not all(np.array_equal(held[0], f) for f in held), (
        "StackCube phase 2 should step the env to surface the failed "
        "translate-and-drop; saw all-identical frames."
    )


def test_stackcube_render_titles_mention_goal_state():
    from babysteps.render.stackcube import render_episode
    from babysteps.envs.stackcube_adapter import StackCubeAdapter
    _, titles = render_episode(_StubStackEnv(), StackCubeAdapter(), seed=0, fps=4)
    # Demo subtitle should mention goal_state="cubeA_on_cubeB" (the oracle).
    assert "cubeA_on_cubeB" in titles["demo"][1]
    # Retry subtitle should mention goal_refinement.
    assert "goal_refinement" in titles["retry"][1]
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_render_modules.py -v -k "stackcube" 2>&1 | tail -10`

Expected: 3 tests FAIL with `ModuleNotFoundError: No module named 'babysteps.render.stackcube'`.

- [ ] **Step 4: Create `babysteps/render/stackcube.py`**

```python
"""StackCube-v1 render_episode — three phases for the Stage-0 MP4 set.

Phase 1 (demo): execute the oracle's correct intent (cubeA_on_cubeB).
Successful stack.
Phase 2 (attempt_blocked): execute the scripted-summarizer's under-
specified intent (cube_at_target). Cube drops at cubeB.xy at low z,
collides with cubeB, scatters. The viewer sees the failure happen.
Phase 3 (retry): execute the goal_refinement-revised intent
(cubeA_on_cubeB). Successful stack.

Like PickCube (and unlike PushCube), all three phases actually step
the env — Stage-0's StackCube failure happens at execution time, not
compile time. The phase key 'attempt_blocked' is kept for consistency
with the other render modules; the term is historical (from B/A's
blocked-approach narrative). For StackCube the failure is under-
specification, not blocking.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from babysteps.envs.task_adapter import BaseTaskAdapter
from babysteps.render.common import (
    PHASE_TOL_M,
    STACKCUBE_MAX_CONTROL_STEPS,
    prop_action,
    render_frame,
    to_np,
)
from babysteps.schemas import AttemptResult, DemoEvidence, Intent, SceneState
from babysteps.skills.stack import (
    CUBE_HALF_SIZE,
    compile_intent_to_stack_skill,
)


_GRIPPER_OPEN = 1.0
_GRIPPER_CLOSED = -1.0


def _read_stack_obs(obs):
    """(tcp_xyzw, cubeA_xy, cubeA_z, cubeB_xy, cubeB_z) from a StackCube obs.

    Local to this render module — the StackCube obs has cubeA_pose and
    cubeB_pose, not the cube/goal pair that babysteps.render.common.read_obs
    parses."""
    tcp_raw = to_np(obs["extra"]["tcp_pose"])
    tcp_raw = np.asarray(tcp_raw, dtype=np.float64)
    tcp = np.concatenate([tcp_raw[0:3], tcp_raw[4:7], tcp_raw[3:4]])
    cubeA_full = np.asarray(to_np(obs["extra"]["cubeA_pose"]), dtype=np.float64)
    cubeA_xy = cubeA_full[0:2]
    cubeA_z = float(cubeA_full[2])
    cubeB_full = np.asarray(to_np(obs["extra"]["cubeB_pose"]), dtype=np.float64)
    cubeB_xy = cubeB_full[0:2]
    cubeB_z = float(cubeB_full[2])
    return tcp, cubeA_xy, cubeA_z, cubeB_xy, cubeB_z


def _execute_stack(
    env, intent: Intent, scene: SceneState, frames: list, *, seed: int,
) -> dict:
    """Step the env through StackSkill's waypoints + per-phase gripper
    schedule. Mirrors PickCube's _execute_pick but with cubeA/cubeB obs
    and goal_state-dispatched waypoint count."""
    skill = compile_intent_to_stack_skill(intent, scene)
    obs, _ = env.reset(seed=int(seed))
    targets = [np.asarray(wp[0:3], dtype=np.float64) for wp in skill.waypoints]

    n_phases = len(targets)
    if n_phases == 4:
        phase_gripper = (
            _GRIPPER_OPEN, _GRIPPER_OPEN, _GRIPPER_CLOSED, _GRIPPER_OPEN,
        )
    elif n_phases == 5:
        phase_gripper = (
            _GRIPPER_OPEN, _GRIPPER_OPEN, _GRIPPER_CLOSED,
            _GRIPPER_CLOSED, _GRIPPER_OPEN,
        )
    else:
        raise RuntimeError(
            f"_execute_stack: unexpected waypoint count {n_phases}; "
            "expected 4 or 5"
        )

    phase_idx = 0
    success = False
    frames.append(render_frame(env))
    for _ in range(STACKCUBE_MAX_CONTROL_STEPS):
        tcp, _cubeA_xy, _cubeA_z, _cubeB_xy, _cubeB_z = _read_stack_obs(obs)
        target = targets[phase_idx]
        if np.linalg.norm(target - tcp[0:3]) < PHASE_TOL_M:
            phase_idx += 1
            if phase_idx >= n_phases:
                break
            target = targets[phase_idx]
        action = prop_action(tcp, target, gripper_cmd=phase_gripper[phase_idx])
        obs, _r, term, trunc, info = env.step(action)
        frames.append(render_frame(env))
        term_b = bool(to_np(term).item()) if hasattr(term, "cpu") else bool(term)
        trunc_b = bool(to_np(trunc).item()) if hasattr(trunc, "cpu") else bool(trunc)
        succ = info.get("success", False) if hasattr(info, "get") else False
        success = bool(to_np(succ).item()) if hasattr(succ, "cpu") else bool(succ)
        if success or term_b or trunc_b:
            break

    return {"success": bool(success)}


def render_episode(
    env, adapter: BaseTaskAdapter, seed: int, fps: int,
) -> tuple[dict, dict]:
    """Three-phase BABYSTEPS render for StackCube."""
    short_id = f"seed {seed:04d}"

    # === Phase 1 — DEMO PROXY (oracle's correct intent: cubeA_on_cubeB) ===
    obs, _ = env.reset(seed=seed)
    tcp, cubeA_xy0, cubeA_z, cubeB_xy, cubeB_z = _read_stack_obs(obs)
    cubeB_top_z = cubeB_z + 2 * CUBE_HALF_SIZE
    scene = SceneState(
        cube_xy=(float(cubeA_xy0[0]), float(cubeA_xy0[1])),
        cube_z=cubeA_z,
        goal_xy=(float(cubeB_xy[0]), float(cubeB_xy[1])),
        tcp_start_pose=tuple(float(v) for v in tcp),  # type: ignore[arg-type]
        blocked_sides=(),
        extra={
            "cubeB_xy": (float(cubeB_xy[0]), float(cubeB_xy[1])),
            "cubeB_z": cubeB_z,
            "cubeB_top_z": cubeB_top_z,
        },
    )
    correct_intent = adapter.oracle_correct_intent(scene)
    demo_frames: list = []
    _ = _execute_stack(env, correct_intent, scene, demo_frames, seed=seed)

    # Build the DemoEvidence the loop would build (2D trajectory hides
    # vertical motion — this is the Stage-0 controlled information loss).
    demo_evidence = DemoEvidence(
        camera="third_person",
        demonstrator_type="proxy_oracle",
        object_trajectory=(
            (float(cubeA_xy0[0]), float(cubeA_xy0[1])),
            (float(cubeB_xy[0]), float(cubeB_xy[1])),
        ),
        contact_region_label=correct_intent.contact_region,
        final_state=correct_intent.goal_state,
        rgbd_video_path=None,
    )
    initial_intent = adapter.scripted_demo_to_intent(demo_evidence)
    scene_exec = replace(
        scene, blocked_sides=adapter.default_blocked_factory(initial_intent),
    )

    # === Phase 2 — ATTEMPT 1 (cube_at_target; collides with cubeB) ===
    attempt1_frames: list = []
    _ = _execute_stack(env, initial_intent, scene_exec, attempt1_frames, seed=seed)

    # === Phase 3 — RETRY with goal_refinement-revised intent ===
    # Synthetic AttemptResult: goal_not_satisfied means the cube didn't
    # reach its (sharpened) target. The fp/attribution pipeline only
    # uses the predicate flags to derive wrong_factor.
    fp = adapter.build_failure_packet(
        initial_intent,
        AttemptResult(
            initial_obj_xy=scene.cube_xy, final_obj_xy=scene.goal_xy,
            goal_xy=scene.goal_xy,
            reached_contact=True, object_moved=True,
            planner_failed=False, collision=False, grasp_slip=False,
            rollout_log_path=None, success=False,
        ),
        scene_exec,
    )
    attribution = adapter.attribute_failure(fp)
    revised_intent, _rev = adapter.revise_intent(
        initial_intent, attribution, scene_exec,
    )
    retry_frames: list = []
    out_retry = _execute_stack(
        env, revised_intent, scene_exec, retry_frames, seed=seed,
    )

    demo_title = (
        f"{short_id}  phase 1/3: demo proxy",
        f"goal_state={correct_intent.goal_state}, "
        f"object_motion={correct_intent.object_motion}",
    )
    a1_title = (
        f"{short_id}  phase 2/3: goal_under-specified",
        f"goal_state={initial_intent.goal_state} → cube drops at cubeB.xy "
        f"(collides, scatters)",
    )
    retry_title = (
        f"{short_id}  phase 3/3: retry (success={out_retry['success']})",
        f"goal_refinement: "
        f"{initial_intent.goal_state} → {revised_intent.goal_state}",
    )

    # Tail-pad attempt1 so the scattered cube is on-screen for at least
    # fps frames (mirrors PickCube's slip-visibility padding).
    if attempt1_frames:
        tail = [attempt1_frames[-1]] * fps
        attempt1_frames = attempt1_frames + tail

    return (
        {"demo": demo_frames,
         "attempt_blocked": attempt1_frames,
         "retry": retry_frames},
        {"demo": demo_title,
         "attempt_blocked": a1_title,
         "retry": retry_title},
    )
```

- [ ] **Step 5: Run render tests to verify they pass**

Run: `python -m pytest tests/test_render_modules.py -v 2>&1 | tail -15`

Expected: all 8 tests PASS (was 5 from Sub-project B + 3 new from C).

- [ ] **Step 6: Full suite**

Run: `python -m pytest tests/ -q 2>&1 | tail -5`

Expected: `219 passed` (216 + 3).

- [ ] **Step 7: Commit**

```bash
git add babysteps/render/common.py babysteps/render/stackcube.py tests/test_render_modules.py
git commit -m "$(cat <<'EOF'
feat(render): babysteps.render.stackcube — three-phase StackCube renderer

Like PickCube (and unlike PushCube), all three phases actually step
the env — Stage-0's StackCube failure happens at execution time, not
compile time:
- Phase 1: oracle's cubeA_on_cubeB intent → successful stack.
- Phase 2: scripted summarizer's cube_at_target intent → cube drops
  at cubeB.xy and scatters after collision. Tail-padded by fps frames.
- Phase 3: goal_refinement-revised cubeA_on_cubeB → successful stack.

Adds STACKCUBE_MAX_CONTROL_STEPS=400 to babysteps.render.common.

3 stub-env tests cover the frame contract, phase-2-actually-steps-
env (the key behavioral difference from PushCube), and title
mentions of goal_state / goal_refinement.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: End-to-end CLI snapshot + snapshot file commit

**Files:**
- Modify: `tests/test_stage0_collect_cli.py` (extend parametrize)
- Snapshot: `tests/snapshots/stackcube_samples_seeds_0_4.jsonl` (already created in Task 4 bootstrap; now also enforced from the CLI)

- [ ] **Step 1: Extend the parametrized CLI snapshot test**

In `tests/test_stage0_collect_cli.py`, find:

```python
@pytest.mark.parametrize("task_id,snapshot_name", [
    ("PushCube-v1", "pushcube_samples_seeds_0_4.jsonl"),
    ("PickCube-v1", "pickcube_samples_seeds_0_4.jsonl"),
])
```

Replace with:

```python
@pytest.mark.parametrize("task_id,snapshot_name", [
    ("PushCube-v1", "pushcube_samples_seeds_0_4.jsonl"),
    ("PickCube-v1", "pickcube_samples_seeds_0_4.jsonl"),
    ("StackCube-v1", "stackcube_samples_seeds_0_4.jsonl"),
])
```

- [ ] **Step 2: Run the parametrized snapshot test to verify byte-equality from the CLI side**

Run: `python -m pytest tests/test_stage0_collect_cli.py::test_stage0_collect_cli_matches_snapshot -v 2>&1 | tail -10`

Expected: all 3 parametrized cases PASS. The StackCube case asserts that the script driving the same registry-backed FakeStackCubeEnvRunner produces JSONL byte-equal to the snapshot captured during Task 4.

If the StackCube case FAILS with a diff, the script and the per-adapter test produced different output. The most likely cause is `episode_id_prefix` mismatch (`"stackcube_underspec_goal"` in both task_registry.py and test_stackcube_adapter.py). Verify and re-run.

- [ ] **Step 3: Full suite**

Run: `python -m pytest tests/ -q 2>&1 | tail -5`

Expected: `220 passed` (219 + 1 new parametrized case).

- [ ] **Step 4: Commit**

```bash
git add tests/test_stage0_collect_cli.py
git commit -m "$(cat <<'EOF'
test(c): end-to-end CLI snapshot for StackCube-v1

Extends test_stage0_collect_cli_matches_snapshot's parametrize to
include ("StackCube-v1", "stackcube_samples_seeds_0_4.jsonl"). This
asserts the script (driving via task_registry + the FakeStackCube-
EnvRunner) produces JSONL byte-equal to the snapshot captured by
test_stackcube_adapter_samples_jsonl_matches_snapshot in Task 4.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: `delta_pp >= 10` gate test for StackCube via fake env

**Files:**
- Modify: `tests/test_pickcube_delta_pp.py` (append one StackCube test)

- [ ] **Step 1: Append the test**

```python
def test_stackcube_fake_env_meets_delta_pp_gate(tmp_path: Path, collect_main):
    """Sub-project C acceptance: StackCube fake-env should achieve
    delta_pp >= 10 (Pick4Pass M-BABY-1 bar). With the deterministic
    FakeStackCubeEnvRunner, all 5 seeds follow under-specified-goal →
    goal_refinement → cubeA_on_cubeB → success. Expected: 100.0."""
    out_dir = tmp_path / "out"
    rc = collect_main([
        "--task", "StackCube-v1",
        "--fake-env",
        "--out_dir", str(out_dir),
        "--n_episodes", "5",
        "--seed_start", "0",
    ])
    report = json.loads((out_dir / "report.json").read_text())
    assert report["delta_pp"] >= 10.0, (
        f"StackCube fake-env delta_pp = {report['delta_pp']:.1f} "
        f"(threshold 10.0). Initial rate {report['initial_attempt_success_rate']:.2f}, "
        f"retry rate {report['retry_success_rate']:.2f}, n_with_revision="
        f"{report['n_with_revision']}, n_retry_success={report['n_retry_success']}."
    )
    assert report["passed_acceptance"] is True
    assert rc == 0
```

- [ ] **Step 2: Run the test**

Run: `python -m pytest tests/test_pickcube_delta_pp.py -v 2>&1 | tail -10`

Expected: all 3 tests PASS. StackCube delta_pp should be 100.0 (5 of 5 seeds recover via goal_refinement).

- [ ] **Step 3: Full suite**

Run: `python -m pytest tests/ -q 2>&1 | tail -5`

Expected: `221 passed` (220 + 1).

- [ ] **Step 4: Commit**

```bash
git add tests/test_pickcube_delta_pp.py
git commit -m "$(cat <<'EOF'
test(c): acceptance-gate test — StackCube delta_pp >= 10 via fake env

Codifies Sub-project C's acceptance gate item 5. With the FakeStack-
CubeEnvRunner's deterministic outcome (success iff intent.goal_state
== cubeA_on_cubeB), all 5 seeds follow the under-specified-goal →
goal_refinement → success arc, yielding delta_pp = 100.0.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: CLAUDE.md update

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add a third srun block**

Find the PickCube srun block (`# PickCube (Sub-project B — grasp_slip; closes B's acceptance gate item 4)`). After the closing single-quote of that block, before the closing ``` of the bash code fence, add:

```bash

# StackCube (Sub-project C — goal under-specification; closes C's acceptance gate item 4)
srun --account=rpaleja --partition=a100-40gb --gres=gpu:1 --mem=115G --time=00:20:00 bash -lc '
  cd /scratch/gilbreth/wang4433/babysteps &&
  source /apps/external/conda/2025.09/etc/profile.d/conda.sh &&
  conda activate handover &&
  OUT_DIR=/scratch/gilbreth/wang4433/render_stackcube &&
  LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" \
  python scripts/render_stage0_maniskill.py \
    --task StackCube-v1 \
    --out_dir "$OUT_DIR" \
    --n_episodes 2 \
    --seed_start 0 &&
  ls -lh "$OUT_DIR/videos_maniskill"
'
```

- [ ] **Step 2: Update the "- Code:" line**

Find:

```
- Code:   `babysteps/` (pure modules) + `babysteps/envs/{pushcube,pickcube}_runner.py` (sim adapters),
          `babysteps/envs/task_registry.py` (--task dispatch),
          `babysteps/render/{pushcube,pickcube}.py` (per-task MP4 flows)
```

Replace with:

```
- Code:   `babysteps/` (pure modules) + `babysteps/envs/{pushcube,pickcube,stackcube}_runner.py` (sim adapters),
          `babysteps/envs/task_registry.py` (--task dispatch),
          `babysteps/render/{pushcube,pickcube,stackcube}.py` (per-task MP4 flows)
```

- [ ] **Step 3: Update the "- Scripts:" line**

Find:

```
- Scripts: `scripts/{stage0_collect,render_stage0_maniskill}.py` accept `--task {PushCube-v1,PickCube-v1}`.
```

Replace with:

```
- Scripts: `scripts/{stage0_collect,render_stage0_maniskill}.py` accept `--task {PickCube-v1,PushCube-v1,StackCube-v1}`.
```

- [ ] **Step 4: Update the test-count claim**

Find:

```
- Tests:  184 sim-free unit tests in `tests/` (PushCube + PickCube, snapshot-stable across both)
```

Replace with:

```
- Tests:  221 sim-free unit tests in `tests/` (PushCube + PickCube + StackCube, snapshot-stable across all three)
```

- [ ] **Step 5: Full suite**

Run: `python -m pytest tests/ -q 2>&1 | tail -5`

Expected: still `221 passed` (no test-count drift — this is a docs-only commit).

- [ ] **Step 6: Commit**

```bash
git add CLAUDE.md
git commit -m "$(cat <<'EOF'
docs(claude.md): add StackCube GPU command + module/test refresh

Adds the third srun block (StackCube-v1, closes C's acceptance gate
item 4). Updates the modules / scripts list to include the StackCube
runner, render module, and the third --task choice. Refreshes the
test count from 184 to 221.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Final gate verification

**Files:** none (verification only).

- [ ] **Step 1: Full test suite — must be all green**

Run:
```bash
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover
python -m pytest tests/ -q
```

Expected: 221 passed. Net new tests vs Sub-project B baseline:
- test_schemas.py: +4
- test_revision.py: +3
- test_stack_skill.py: +10
- test_stackcube_adapter.py: +13
- test_task_registry.py: +2
- test_stage0_collect_cli.py: +1 (parametrized)
- test_pickcube_delta_pp.py: +1
- test_render_modules.py: +3
- Total new: 37. Total: 184 + 37 = 221.

If anything fails, STOP and report.

- [ ] **Step 2: PushCube samples.jsonl byte-identical (regression check)**

Run:
```bash
python scripts/stage0_collect.py \
  --out_dir /tmp/pushcube-gate-c \
  --n_episodes 5 --seed_start 0 --fake-env
diff -u tests/snapshots/pushcube_samples_seeds_0_4.jsonl /tmp/pushcube-gate-c/samples.jsonl
echo "---report title---"
head -1 /tmp/pushcube-gate-c/report.md
```

Expected: `diff` produces no output. Report title says `# BABYSTEPS Stage 0 — PushCube Report`.

- [ ] **Step 3: PickCube samples.jsonl byte-identical (regression check)**

Run:
```bash
python scripts/stage0_collect.py \
  --task PickCube-v1 \
  --out_dir /tmp/pickcube-gate-c \
  --n_episodes 5 --seed_start 0 --fake-env
diff -u tests/snapshots/pickcube_samples_seeds_0_4.jsonl /tmp/pickcube-gate-c/samples.jsonl
echo "---report title---"
head -1 /tmp/pickcube-gate-c/report.md
```

Expected: `diff` empty; title says `# BABYSTEPS Stage 0 — PickCube Report`.

- [ ] **Step 4: StackCube samples.jsonl byte-equal to snapshot**

Run:
```bash
python scripts/stage0_collect.py \
  --task StackCube-v1 \
  --out_dir /tmp/stackcube-gate \
  --n_episodes 5 --seed_start 0 --fake-env
diff -u tests/snapshots/stackcube_samples_seeds_0_4.jsonl /tmp/stackcube-gate/samples.jsonl
echo "---report title---"
head -1 /tmp/stackcube-gate/report.md
cat /tmp/stackcube-gate/report.json | python -c "import json,sys; d=json.load(sys.stdin); print('delta_pp:', d['delta_pp'], 'passed:', d['passed_acceptance'])"
```

Expected: `diff` empty; title says `# BABYSTEPS Stage 0 — StackCube Report`; `delta_pp: 100.0 passed: True`.

- [ ] **Step 5: Confirm registry parity holds**

Run:
```bash
python -c "
from babysteps.envs.task_registry import TASK_REGISTRY
from babysteps.render import RENDER_REGISTRY
print('TASK:', sorted(TASK_REGISTRY))
print('RENDER:', sorted(RENDER_REGISTRY))
assert sorted(TASK_REGISTRY) == sorted(RENDER_REGISTRY) == ['PickCube-v1', 'PushCube-v1', 'StackCube-v1']
print('parity OK')
"
```

Expected: prints both registries with all three tasks, then `parity OK`.

- [ ] **Step 6: GPU spot-check (MANUAL — Sub-project C gate item 4)**

This requires a Vulkan compute node. From the login shell, schedule a 20-min A100 job with the StackCube srun command from CLAUDE.md (added in Task 10). Confirm `videos_maniskill/` contains 6 MP4s (2 episodes × 3 phases) named with the `stackcube_underspec_goal_seed_NNNN__{1_demo,2_attempt_blocked,3_retry}.mp4` pattern. Visually verify:

- Phase 1 (`__1_demo`): gripper picks cubeA, lifts above cubeB, descends onto cubeB top, releases. Stack stays.
- Phase 2 (`__2_attempt_blocked`): gripper picks cubeA, carries to cubeB.xy at low z, releases. Cube collides with cubeB and scatters next to it (or onto the table beside cubeB).
- Phase 3 (`__3_retry`): same as phase 1.

This is the only gate item that cannot be automated. The render-module unit tests (Task 7) guard the orchestration; only the human can confirm the MP4 visual.

- [ ] **Step 7: Tag the gate-passing commit (DEFERRED TO USER)**

Once Steps 1-5 are green AND the GPU MP4s look correct, the user may optionally:

```bash
git tag stage0-stackcube-c-gate-pass
git log --oneline -1
```

Do NOT push the tag without the user's explicit OK. The plan stops at "all automated checks pass + manual step documented" — the tag is the user's call.

---

## Self-Review

**Spec coverage:**

- §1 Motivation: framed in plan header.
- §2 Stage-0 controlled failure: documented in StackCubeAdapter docstring (Task 4) and stack.py docstring (Task 3).
- §3 Acceptance gate items 1, 2, 3, 5: Tasks 0, 1-4 and 7 (test counts), 4 + 8 (snapshot byte-stability), 9 (delta_pp >= 10).
- §3 item 4 (GPU MP4 spot-check): Tasks 9 + 10 wire it, Task 11 step 6 documents the manual check.
- §4 Schema deltas: Task 1.
- §5 Failure attribution (no changes): explicitly noted in plan header and Task 2 commit message.
- §6 Revision operator `goal_refinement`: Task 2.
- §7 StackCubeAdapter: Task 4.
- §8 StackSkill compiler: Task 3.
- §9 StackCubeEnvRunner: Task 5.
- §10 ManiSkill StackCube-v1 facts: encoded in Task 5's `_read_obs` (cubeA_pose, cubeB_pose) and Task 4's FakeStackCubeEnvRunner scene constants (cube_z = 0.02, cubeB_top_z = 0.06).
- §11 FakeStackCubeEnvRunner: Task 4.
- §12 Render module: Task 7.
- §13 CLI integration: Task 6.
- §14 Test plan: distributed across Tasks 1-9, summary in Task 11.
- §15 Risks: addressed inline (cube collision physics → tail-pad in Task 7; cube_half_size duplication → constant in Task 3; static-velocity requirement → tail padding in Task 7).
- §16 Plan file: this file. ✓
- §17 Summary: covered by plan header.

**Placeholder scan:** No TBD / TODO / "implement later" lines. Every code step shows full code. Every command shows expected output.

**Type consistency:**
- `StackSkill` dataclass shape: `waypoints, cubeA_z, cubeB_top_z, goal_state` — consistent across Task 3 definition, Task 4 fake runner usage, Task 5 real runner usage, Task 7 render module usage.
- `FakeStackCubeEnvRunner` constants: `cube_z=0.02, cubeB_top_z=0.06` — consistent between Task 4 implementation and stub usage in tests.
- `episode_id_prefix = "stackcube_underspec_goal"` — consistent across Task 4 (test snapshot), Task 6 (task_registry), Task 11 (verification grep).
- Phase names `"demo" / "attempt_blocked" / "retry"` — consistent across the new render module (Task 7) and the dispatcher script's already-committed iteration order.
- `_GRIPPER_OPEN = 1.0`, `_GRIPPER_CLOSED = -1.0` — consistent between Task 5's runner and Task 7's render module.

**Risks / gotchas the engineer should know:**
1. Task 6 introduces a brief window where `RENDER_REGISTRY["StackCube-v1"]` exists but `babysteps.render.stackcube` does not — Task 7 closes this gap. The registry's lazy factory means import-time of `babysteps.render` stays clean; only `get_render_fn("StackCube-v1")()` would fail in the interim. The plan keeps Tasks 6 and 7 close to minimize the gap.
2. Task 4 step 5 is a SKIP, not a PASS, because the snapshot is bootstrapping. Task 4 step 6 explicitly re-runs to verify byte-equality. Both must happen before commit.
3. `test_get_task_entry_unknown_task_raises` is updated to use `OpenCabinetDrawer-v1` (a known-future task name). When Sub-project D lands, that test will need to switch to yet another unknown task name (e.g., `"FakeTask-v1"`). Documented inline in Task 6 step 1.
4. The cube_at_target attempt's real-physics outcome in StackCube is non-deterministic (collision may push cubeA in any direction). The success criterion correctly reports False regardless. The MP4 may look chaotic; visual spot-check should accept "cubeA somewhere other than on top of cubeB" as PASS.
