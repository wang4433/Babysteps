# Sub-project D (TurnFaucet-v1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land TurnFaucet-v1 as the fourth Stage-0 adapter, exercising the `constraint_region` intent factor via a demo whose 2D-trajectory summarization under-specifies the contact (gives `faucet_base` instead of `handle_grip`) and the constraint (gives `none`). Failure → `constraint_introduction` revision (two-factor: adds the constraint AND swaps the contact) → retry rotates the handle. Closes the spec's acceptance gate via fake-env CLI; the real-sim GPU spot-check requires the `partnet_mobility_faucet` asset download.

**Architecture:** New `TurnFaucetAdapter` + `TurnFaucetEnvRunner` + `TurnSkill` + `babysteps/render/turnfaucet.py` slot into the existing Stage-0 dispatch machinery (BaseTaskAdapter, TASK_REGISTRY, RENDER_REGISTRY) with one-row registry additions. The skill compiler dispatches on `intent.contact_region`: `faucet_base` waypoints target a Stage-0-approximated body position; `handle_grip` waypoints target the obs-derived handle position. The new `constraint_violation` failure predicate reuses `AttemptResult.collision` as its Stage-0 proxy signal (additive — all prior snapshots have `collision: false`). The new `constraint_introduction` revision operator is the only Stage-0 operator that revises two factors at once (constraint_region AND contact_region).

**Tech Stack:** Python 3, ManiSkill 3 (gymnasium), pytest, numpy, PIL+imageio. All Stage-0 infrastructure (adapter ABC, episode loop, failure/revision modules, task/render registries, CLI scripts, render package) is reused without modification.

**Source spec:** `docs/superpowers/specs/2026-05-17-stage0-turnfaucet-d-design.md` (committed at `515093c`).

**Scope guardrails:**
- Sub-project D only.
- PushCube + PickCube + StackCube snapshots MUST stay byte-identical.
- All 221 existing tests must continue to pass at every task boundary.
- Stage-0 one-attempt-then-one-retry per episode.
- Privileged-firewall: `scripted_demo_to_intent` stays demo-only; the deliberate under-specification (always returning `contact_region="faucet_base"`, `constraint_region="none"`) is the controlled-failure mechanism, NOT a privileged leak.
- `AttemptResult.collision` is repurposed as the `constraint_violation` proxy signal. All prior snapshots have `collision: false`, so this is additive.

---

## File Structure

**Create:**
- `babysteps/skills/turn.py` — `TurnSkill` dataclass + `compile_intent_to_turn_skill` (pure; dispatches on `intent.contact_region`).
- `babysteps/envs/turnfaucet_adapter.py` — `TurnFaucetAdapter(BaseTaskAdapter)`.
- `babysteps/envs/turnfaucet_runner.py` — real ManiSkill `TurnFaucetEnvRunner`.
- `babysteps/render/turnfaucet.py` — `render_episode` for the three-phase MP4 flow.
- `tests/test_turn_skill.py` — sim-free unit tests for the skill compiler (~6 tests).
- `tests/test_turnfaucet_adapter.py` — sim-free unit tests + snapshot test (~13 tests).
- `tests/snapshots/turnfaucet_samples_seeds_0_4.jsonl` — captured during Task 4.

**Modify:**
- `babysteps/schemas.py` — 7 new whitelist tokens (Task 1).
- `babysteps/failure.py` — `constraint_violation` predicate branch + `FAILURE_TO_FACTOR` entry (Task 2).
- `babysteps/revision.py` — `constraint_introduction` branch in `revise_intent` (Task 3).
- `babysteps/envs/task_registry.py` — `_turnfaucet_entry()` + registry row (Task 7).
- `babysteps/render/__init__.py` — `_turnfaucet_render()` + registry row (Task 7).
- `tests/conftest.py` — append `FakeTurnFaucetEnvRunner` + `fake_turnfaucet_env_runner` fixture (Task 5).
- `tests/test_schemas.py` — 7 new tests (Task 1).
- `tests/test_failure.py` — 2 new tests (Task 2).
- `tests/test_revision.py` — 4 new tests (Task 3).
- `tests/test_task_registry.py` — 2 new tests + rename `test_registry_contains_all_stage0_tasks` to include TurnFaucet, rotate unknown-task sentinel to `"Bogus-v1"` (Task 7).
- `tests/test_render_modules.py` — 3 new TurnFaucet render tests (Task 8).
- `tests/test_stage0_collect_cli.py` — extend parametrize with TurnFaucet (Task 9); rotate unknown-task sentinel to `"Bogus-v1"`.
- `tests/test_pickcube_delta_pp.py` — 1 new TurnFaucet delta_pp test (Task 10).
- `CLAUDE.md` — fourth srun block + asset-download note + module/test refresh (Task 11).

**Untouched:**
- `babysteps/envs/task_adapter.py`, `babysteps/episode.py`, `babysteps/eval.py`.
- `scripts/stage0_collect.py`, `scripts/stage0_summarize.py`, `scripts/render_stage0_maniskill.py`.
- All existing snapshot files (must stay byte-identical).
- All A/B/C-era adapters, runners, skills, render modules.

---

## Task 0: Baseline check

**Files:** none.

- [ ] **Step 1: Full suite**

```bash
cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover
python -m pytest tests/ -q 2>&1 | tail -5
```

Expected: `221 passed`.

- [ ] **Step 2: Three snapshots exist (Push, Pick, Stack); TurnFaucet does not**

Run: `ls tests/snapshots/`

Expected output (alphabetical):
```
pickcube_samples_seeds_0_4.jsonl
pushcube_samples_seeds_0_4.jsonl
stackcube_samples_seeds_0_4.jsonl
```

If `turnfaucet_samples_seeds_0_4.jsonl` already exists, this plan needs adjustment — STOP.

- [ ] **Step 3: Three registries (no TurnFaucet yet)**

```bash
python -c "from babysteps.envs.task_registry import TASK_REGISTRY; print(sorted(TASK_REGISTRY))"
python -c "from babysteps.render import RENDER_REGISTRY; print(sorted(RENDER_REGISTRY))"
```

Expected: `['PickCube-v1', 'PushCube-v1', 'StackCube-v1']` for both.

- [ ] **Step 4: Spec committed**

Run: `git log --oneline -1 docs/superpowers/specs/2026-05-17-stage0-turnfaucet-d-design.md`

Expected: shows commit `515093c` (or current head of the spec).

---

## Task 1: Schema deltas

**Files:**
- Modify: `babysteps/schemas.py`
- Modify: `tests/test_schemas.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_schemas.py`:

```python
# ---------- Sub-project D (TurnFaucet) whitelist additions ---------- #


def test_goal_states_contains_faucet_turned():
    from babysteps.schemas import GOAL_STATES
    assert "faucet_turned" in GOAL_STATES


def test_object_motions_contains_turn():
    from babysteps.schemas import OBJECT_MOTIONS
    assert "turn" in OBJECT_MOTIONS


def test_contact_regions_contains_faucet_base():
    from babysteps.schemas import CONTACT_REGIONS
    assert "faucet_base" in CONTACT_REGIONS


def test_contact_regions_contains_handle_grip():
    from babysteps.schemas import CONTACT_REGIONS
    assert "handle_grip" in CONTACT_REGIONS


def test_constraint_regions_contains_faucet_base_static():
    from babysteps.schemas import CONSTRAINT_REGIONS
    assert "faucet_base_static" in CONSTRAINT_REGIONS


def test_failure_predicates_contains_constraint_violation():
    from babysteps.schemas import FAILURE_PREDICATES
    assert "constraint_violation" in FAILURE_PREDICATES


def test_revision_operators_contains_constraint_introduction():
    from babysteps.schemas import REVISION_OPERATORS
    assert "constraint_introduction" in REVISION_OPERATORS


def test_embodiment_mappings_contains_franka_turn():
    from babysteps.schemas import EMBODIMENT_MAPPINGS
    assert "proxy_contact_to_franka_turn" in EMBODIMENT_MAPPINGS
```

(That's 8 tests total — 7 named in the spec plan plus the embodiment one. Net new on test_schemas.py: +8.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `source /apps/external/conda/2025.09/etc/profile.d/conda.sh && conda activate handover && python -m pytest tests/test_schemas.py -v -k "faucet or handle_grip or constraint_violation or constraint_introduction or franka_turn" 2>&1 | tail -15`

Expected: 8 tests FAIL.

- [ ] **Step 3: Add tokens to whitelists**

In `babysteps/schemas.py`:

Find and replace `GOAL_STATES`:
```python
GOAL_STATES: frozenset[str] = frozenset({
    "cube_at_target",
    "cube_lifted_at_target",           # B: PickCube — cube lifted to goal xyz
    "cubeA_on_cubeB",                  # C: StackCube — cubeA resting atop cubeB
    "faucet_turned",                   # D: TurnFaucet — handle rotated past target
})
```

Find and replace `OBJECT_MOTIONS`:
```python
OBJECT_MOTIONS: frozenset[str] = frozenset({
    "translate_+x", "translate_-x", "translate_+y", "translate_-y",
    "lift_up",   # B: PickCube — cube lifted along +z
    "place_on",  # C: StackCube — cube placed on top of another cube
    "turn",      # D: TurnFaucet — handle rotated around joint axis
})
```

Find and replace `CONTACT_REGIONS`:
```python
CONTACT_REGIONS: frozenset[str] = frozenset({
    "minus_x_face", "plus_x_face", "minus_y_face", "plus_y_face",
    "faucet_base",   # D: TurnFaucet — static body of the faucet (wrong contact)
    "handle_grip",   # D: TurnFaucet — rotating handle (correct contact)
})
```

Find and replace `CONSTRAINT_REGIONS`:
```python
CONSTRAINT_REGIONS: frozenset[str] = frozenset({
    "none",
    "faucet_base_static",   # D: TurnFaucet — body must not be displaced
})
```

Find and replace `FAILURE_PREDICATES`:
```python
FAILURE_PREDICATES: frozenset[str] = frozenset({
    "none",
    "approach_blocked",
    "direction_error",
    "contact_failure",
    "no_motion",
    "goal_not_satisfied",
    "grasp_slip",                      # B: PickCube — grip lost during lift
    "constraint_violation",            # D: TurnFaucet — touched non-articulating link
})
```

Find and replace `REVISION_OPERATORS`:
```python
REVISION_OPERATORS: frozenset[str] = frozenset({
    "approach_substitution",
    "contact_substitution",            # B: PickCube — rotate gripper axis
    "goal_refinement",                 # C: StackCube — sharpen under-specified goal
    "constraint_introduction",         # D: TurnFaucet — add constraint + swap contact
})
```

Find and replace `EMBODIMENT_MAPPINGS`:
```python
EMBODIMENT_MAPPINGS: frozenset[str] = frozenset({
    "proxy_contact_to_franka_push",
    "proxy_contact_to_franka_grasp",   # B: PickCube — parallel-jaw grasp
    "proxy_contact_to_franka_pick_and_place",  # C: StackCube — pick + place sequence
    "proxy_contact_to_franka_turn",    # D: TurnFaucet — grip + tangential pull
})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_schemas.py -v 2>&1 | tail -5`

Expected: all schema tests pass (was 33 from C, gains 8 → 41).

- [ ] **Step 5: Full suite**

Run: `python -m pytest tests/ -q 2>&1 | tail -5`

Expected: `229 passed` (221 + 8).

- [ ] **Step 6: Commit**

```bash
git add babysteps/schemas.py tests/test_schemas.py
git commit -m "$(cat <<'EOF'
feat(schemas): Sub-project D whitelist additions (TurnFaucet)

- GOAL_STATES += "faucet_turned"
- OBJECT_MOTIONS += "turn"
- CONTACT_REGIONS += "faucet_base", "handle_grip"
- CONSTRAINT_REGIONS += "faucet_base_static"
- FAILURE_PREDICATES += "constraint_violation"
- REVISION_OPERATORS += "constraint_introduction"
- EMBODIMENT_MAPPINGS += "proxy_contact_to_franka_turn"

Per spec §4 of docs/superpowers/specs/2026-05-17-stage0-turnfaucet-d-design.md.
Additive only — Push/Pick/Stack records don't contain these tokens so
existing snapshot tests stay byte-identical.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `constraint_violation` failure predicate + attribution

**Files:**
- Modify: `babysteps/failure.py`
- Modify: `tests/test_failure.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_failure.py`:

```python
# ---------- Sub-project D: constraint_violation -------------------- #


def test_constraint_violation_predicate_fires_for_collision_no_motion():
    """When attempt.collision=True AND attempt.object_moved=False (and
    not planner_failed), the failure_predicate is 'constraint_violation'.
    This is more specific than the no_motion predicate which would
    otherwise fire."""
    from babysteps.failure import build_failure_packet
    from babysteps.schemas import AttemptResult, Intent, SceneState

    intent = Intent(
        goal_state="faucet_turned", object_motion="turn",
        contact_region="faucet_base", approach_direction="from_above",
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_turn",
    )
    scene = SceneState(
        cube_xy=(0.1, 0.0), cube_z=0.1, goal_xy=(0.1, 0.0),
        tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0),
        blocked_sides=(),
    )
    attempt = AttemptResult(
        initial_obj_xy=(0.1, 0.0), final_obj_xy=(0.1, 0.0),
        goal_xy=(0.1, 0.0),
        reached_contact=True, object_moved=False,
        planner_failed=False, collision=True, grasp_slip=False,
        rollout_log_path=None, success=False,
    )
    fp = build_failure_packet(intent, attempt, scene)
    assert fp.failure_predicate == "constraint_violation"


def test_attribute_failure_constraint_violation_to_constraint_region():
    """constraint_violation predicate maps to wrong_factor=
    'constraint_region' with revise=(constraint_region, contact_region)
    — the two-factor revision pair."""
    from babysteps.failure import attribute_failure
    from babysteps.schemas import FailurePacket, Intent

    intent = Intent(
        goal_state="faucet_turned", object_motion="turn",
        contact_region="faucet_base", approach_direction="from_above",
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_turn",
    )
    fp = FailurePacket(
        chosen_intent=intent,
        execution_trace={"reached_contact": True, "object_moved": False,
                         "collision": True, "planner_failed": False,
                         "grasp_slip": False},
        failure_predicate="constraint_violation",
        object_displacement=0.0, direction_alignment=None,
    )
    attribution = attribute_failure(fp)
    assert attribution.wrong_factor == "constraint_region"
    assert "constraint_region" in attribution.revise
    assert "contact_region" in attribution.revise
    assert "constraint_region" not in attribution.freeze
    assert "contact_region" not in attribution.freeze
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_failure.py -v -k "constraint_violation" 2>&1 | tail -10`

Expected: 2 tests FAIL — `build_failure_packet` doesn't yet branch on `collision`, and `FAILURE_TO_FACTOR` doesn't have `constraint_violation`.

- [ ] **Step 3: Add the predicate branch in `babysteps/failure.py`**

In `babysteps/failure.py`, find:

```python
    if attempt.success:
        predicate = "none"
    elif attempt.planner_failed:
        predicate = "approach_blocked"
    elif attempt.grasp_slip:
        # Sub-project B: grasp_slip is more specific than the contact /
        # motion / direction predicates below — it carries the strong
        # signal that the gripper DID reach the cube but lost grip. Goes
        # right after planner_failed in the precedence.
        predicate = "grasp_slip"
    elif not attempt.reached_contact:
```

Insert a new branch between `planner_failed` and `grasp_slip`:

```python
    if attempt.success:
        predicate = "none"
    elif attempt.planner_failed:
        predicate = "approach_blocked"
    elif attempt.collision and not attempt.object_moved:
        # Sub-project D: constraint_violation when the gripper contacted a
        # non-articulating link and tried to actuate it. The env_runner
        # marks this case by setting collision=True (Stage-0 proxy for
        # "touched something that didn't move"). More specific than
        # grasp_slip / contact_failure / no_motion — those would also
        # fire here but the constraint signal is the actionable one.
        predicate = "constraint_violation"
    elif attempt.grasp_slip:
        # Sub-project B: grasp_slip is more specific than the contact /
        # motion / direction predicates below — it carries the strong
        # signal that the gripper DID reach the cube but lost grip. Goes
        # right after planner_failed in the precedence.
        predicate = "grasp_slip"
    elif not attempt.reached_contact:
```

Find `FAILURE_TO_FACTOR`:

```python
FAILURE_TO_FACTOR: dict[str, tuple[str, tuple[str, ...]]] = {
    "approach_blocked":   ("approach_direction", ("approach_direction", "contact_region")),
    "direction_error":    ("approach_direction", ("approach_direction",)),
    "contact_failure":    ("contact_region",     ("contact_region",)),
    "no_motion":          ("approach_direction", ("approach_direction", "contact_region")),
    "goal_not_satisfied": ("goal_state",         ("goal_state",)),
    # Sub-project B (PickCube): grasp_slip → contact_region is wrong (the
    # chosen gripper-axis is slip-prone for this cube). embodiment_mapping
    # is also in `revise` to permit future operators that switch grasp
    # primitives; Stage-0's contact_substitution does not touch it.
    "grasp_slip":         ("contact_region",     ("contact_region", "embodiment_mapping")),
}
```

Append the new entry:

```python
FAILURE_TO_FACTOR: dict[str, tuple[str, tuple[str, ...]]] = {
    "approach_blocked":   ("approach_direction", ("approach_direction", "contact_region")),
    "direction_error":    ("approach_direction", ("approach_direction",)),
    "contact_failure":    ("contact_region",     ("contact_region",)),
    "no_motion":          ("approach_direction", ("approach_direction", "contact_region")),
    "goal_not_satisfied": ("goal_state",         ("goal_state",)),
    # Sub-project B (PickCube): grasp_slip → contact_region is wrong (the
    # chosen gripper-axis is slip-prone for this cube). embodiment_mapping
    # is also in `revise` to permit future operators that switch grasp
    # primitives; Stage-0's contact_substitution does not touch it.
    "grasp_slip":         ("contact_region",     ("contact_region", "embodiment_mapping")),
    # Sub-project D (TurnFaucet): constraint_violation → constraint_region
    # is missing AND contact_region was the wrong (faucet_base) target.
    # The constraint_introduction operator revises BOTH factors at once;
    # the two-factor revise tuple is what enables the audit to expect
    # both changes.
    "constraint_violation": ("constraint_region",
                              ("constraint_region", "contact_region")),
}
```

Also update the precedence-comment in `build_failure_packet`'s docstring. Find:

```python
def build_failure_packet(
    intent: Intent, attempt: AttemptResult, scene: SceneState,
) -> FailurePacket:
    """Derive the structured FailurePacket. Predicate precedence: most
    specific first (success → planner_failed → contact → motion → direction
    → goal).
    """
```

Replace with:

```python
def build_failure_packet(
    intent: Intent, attempt: AttemptResult, scene: SceneState,
) -> FailurePacket:
    """Derive the structured FailurePacket. Predicate precedence: most
    specific first (success → planner_failed → constraint_violation
    → grasp_slip → contact_failure → no_motion → direction_error
    → goal_not_satisfied).
    """
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_failure.py -v 2>&1 | tail -10`

Expected: all test_failure.py tests PASS (was 22 from B+C, gains 2 → 24).

- [ ] **Step 5: Full suite — must not regress existing predicate tests**

Run: `python -m pytest tests/ -q 2>&1 | tail -5`

Expected: `231 passed` (229 + 2).

If any pre-existing test in test_failure.py (especially the PickCube grasp_slip precedence tests) fails, the new branch's placement broke ordering. STOP and inspect — the new branch must NOT fire for grasp_slip cases (which have `collision=False`).

- [ ] **Step 6: Commit**

```bash
git add babysteps/failure.py tests/test_failure.py
git commit -m "$(cat <<'EOF'
feat(failure): constraint_violation predicate for Sub-project D

build_failure_packet's precedence chain extends to:
  success → planner_failed → constraint_violation → grasp_slip →
  contact_failure → no_motion → direction_error → goal_not_satisfied

constraint_violation fires when attempt.collision=True AND attempt.
object_moved=False (i.e., the gripper touched something but it
didn't move). Reuses the existing collision flag as a Stage-0 proxy
signal — all prior snapshots have collision=false so this is
additive.

FAILURE_TO_FACTOR["constraint_violation"] = ("constraint_region",
("constraint_region", "contact_region")) — the two-factor revise
tuple that constraint_introduction (next task) operates over.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `constraint_introduction` revision operator

**Files:**
- Modify: `babysteps/revision.py`
- Modify: `tests/test_revision.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_revision.py`:

```python
# ---------- Sub-project D: constraint_introduction ------------------ #


def test_constraint_introduction_happy_path():
    """(constraint_region=none, contact_region=faucet_base) →
    (faucet_base_static, handle_grip). Two-factor revision."""
    from babysteps.failure import Attribution
    from babysteps.revision import revise_intent
    from babysteps.schemas import Intent, SceneState

    intent = Intent(
        goal_state="faucet_turned",
        object_motion="turn",
        contact_region="faucet_base",
        approach_direction="from_above",
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_turn",
    )
    scene = SceneState(
        cube_xy=(0.1, 0.0), cube_z=0.1, goal_xy=(0.1, 0.0),
        tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0),
        blocked_sides=(),
    )
    attribution = Attribution(
        semantic_failure=True,
        wrong_factor="constraint_region",
        freeze=("goal_state", "object_motion", "approach_direction",
                "embodiment_mapping"),
        revise=("constraint_region", "contact_region"),
    )
    revised, record = revise_intent(intent, attribution, scene)
    # BOTH revised:
    assert revised.constraint_region == "faucet_base_static"
    assert revised.contact_region == "handle_grip"
    # Other factors carry over unchanged:
    assert revised.goal_state == "faucet_turned"
    assert revised.object_motion == "turn"
    assert revised.approach_direction == "from_above"
    assert revised.embodiment_mapping == "proxy_contact_to_franka_turn"
    # Revision record names constraint_region as the primary factor.
    assert record.operator == "constraint_introduction"
    assert record.factor == "constraint_region"
    assert record.old_value == "none"
    assert record.new_value == "faucet_base_static"
    # Both revised factors must NOT appear in frozen_factors.
    assert "constraint_region" not in record.frozen_factors
    assert "contact_region" not in record.frozen_factors


def test_constraint_introduction_unknown_source_raises():
    """Stage-0 supports only (none, faucet_base) → (..., handle_grip)."""
    import pytest
    from babysteps.failure import Attribution
    from babysteps.revision import revise_intent
    from babysteps.schemas import Intent, SceneState

    # constraint_region already set — out of scope for Stage-0.
    intent = Intent(
        goal_state="faucet_turned", object_motion="turn",
        contact_region="faucet_base", approach_direction="from_above",
        constraint_region="faucet_base_static",   # already constrained
        embodiment_mapping="proxy_contact_to_franka_turn",
    )
    scene = SceneState(
        cube_xy=(0.1, 0.0), cube_z=0.1, goal_xy=(0.1, 0.0),
        tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0),
        blocked_sides=(),
    )
    attribution = Attribution(
        semantic_failure=True, wrong_factor="constraint_region",
        freeze=("goal_state", "object_motion", "approach_direction",
                "embodiment_mapping"),
        revise=("constraint_region", "contact_region"),
    )
    with pytest.raises(NotImplementedError) as exc:
        revise_intent(intent, attribution, scene)
    msg = str(exc.value)
    assert "faucet_base_static" in msg or "constraint_region" in msg


def test_constraint_introduction_wrong_contact_source_raises():
    """If contact_region isn't faucet_base, the operator must raise."""
    import pytest
    from babysteps.failure import Attribution
    from babysteps.revision import revise_intent
    from babysteps.schemas import Intent, SceneState

    intent = Intent(
        goal_state="faucet_turned", object_motion="turn",
        contact_region="handle_grip",   # already correct contact
        approach_direction="from_above",
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_turn",
    )
    scene = SceneState(
        cube_xy=(0.1, 0.0), cube_z=0.1, goal_xy=(0.1, 0.0),
        tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0),
        blocked_sides=(),
    )
    attribution = Attribution(
        semantic_failure=True, wrong_factor="constraint_region",
        freeze=("goal_state", "object_motion", "approach_direction",
                "embodiment_mapping"),
        revise=("constraint_region", "contact_region"),
    )
    with pytest.raises(NotImplementedError):
        revise_intent(intent, attribution, scene)


def test_constraint_introduction_frozen_factors_audit():
    """The Revision record's frozen_factors lists exactly the 4 factors
    NOT revised (goal_state, object_motion, approach_direction,
    embodiment_mapping)."""
    from babysteps.failure import Attribution
    from babysteps.revision import revise_intent
    from babysteps.schemas import Intent, SceneState

    intent = Intent(
        goal_state="faucet_turned", object_motion="turn",
        contact_region="faucet_base", approach_direction="from_above",
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_turn",
    )
    scene = SceneState(
        cube_xy=(0.1, 0.0), cube_z=0.1, goal_xy=(0.1, 0.0),
        tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0),
        blocked_sides=(),
    )
    attribution = Attribution(
        semantic_failure=True, wrong_factor="constraint_region",
        freeze=("goal_state", "object_motion", "approach_direction",
                "embodiment_mapping"),
        revise=("constraint_region", "contact_region"),
    )
    _, record = revise_intent(intent, attribution, scene)
    expected = {"goal_state", "object_motion", "approach_direction",
                "embodiment_mapping"}
    assert set(record.frozen_factors) == expected
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_revision.py -v -k "constraint_introduction" 2>&1 | tail -15`

Expected: 4 tests FAIL.

- [ ] **Step 3: Add the branch to `revise_intent`**

In `babysteps/revision.py`, find:

```python
    if attribution.wrong_factor == "goal_state":
        # Stage-0's goal_refinement is a strict-extension operator:
```

Insert above (so `constraint_region` is handled before `goal_state` — order doesn't matter for correctness but the predicate-precedence in failure.py puts constraint_violation high):

```python
    if attribution.wrong_factor == "constraint_region":
        # Sub-project D: constraint_introduction is the only Stage-0
        # revision operator that touches TWO factors at once. It is
        # triggered when the demo under-specified BOTH the
        # contact_region (faucet_base) AND the constraint_region (none).
        # The operator adds the constraint AND swaps the contact to
        # handle_grip. Strict-extension: handles only the
        # (none, faucet_base) → (faucet_base_static, handle_grip)
        # transition per spec §6.
        if (intent.constraint_region != "none"
                or intent.contact_region != "faucet_base"):
            raise NotImplementedError(
                f"constraint_introduction does not handle transitions "
                f"from (constraint_region={intent.constraint_region!r}, "
                f"contact_region={intent.contact_region!r}). (Stage-0 "
                f"supports only the (none, faucet_base) → "
                f"(faucet_base_static, handle_grip) refinement per "
                f"docs/superpowers/specs/2026-05-17-stage0-turnfaucet-d-design.md §6)"
            )
        revised = replace(
            intent,
            constraint_region="faucet_base_static",
            contact_region="handle_grip",
        )
        # frozen excludes BOTH revised factors
        frozen = tuple(
            f for f in INTENT_FIELDS
            if f not in ("constraint_region", "contact_region")
        )
        rev_record = Revision(
            operator="constraint_introduction",
            factor="constraint_region",   # primary factor for audit
            old_value="none",
            new_value="faucet_base_static",
            frozen_factors=frozen,
        )
        return revised, rev_record

    if attribution.wrong_factor == "goal_state":
        # Stage-0's goal_refinement is a strict-extension operator:
```

Also update the module docstring (top of file). Find:

```python
Stage 0 implements:
  * `approach_substitution` — for wrong_factor=="approach_direction"
    (Sub-project A / PushCube).
  * `contact_substitution` — for wrong_factor=="contact_region"
    (Sub-project B / PickCube).
  * `goal_refinement` — for wrong_factor=="goal_state"
    (Sub-project C / StackCube; strict-extension: cube_at_target →
    cubeA_on_cubeB only).
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
  * `constraint_introduction` — for wrong_factor=="constraint_region"
    (Sub-project D / TurnFaucet; two-factor revision: adds the
    constraint AND swaps contact_region: (none, faucet_base) →
    (faucet_base_static, handle_grip)).
```

Update `revise_intent`'s function docstring. Find:

```python
    """Return (revised_intent, Revision record). Dispatches on
    `attribution.wrong_factor`. Stage-0 supports approach_direction,
    contact_region, and goal_state; other factors raise NotImplementedError.
    """
```

Replace with:

```python
    """Return (revised_intent, Revision record). Dispatches on
    `attribution.wrong_factor`. Stage-0 supports approach_direction,
    contact_region, goal_state, and constraint_region; other factors
    raise NotImplementedError.
    """
```

Update the final fallback `raise NotImplementedError` message:

```python
    raise NotImplementedError(
        f"Stage-0 reviser handles 'approach_direction', 'contact_region', "
        f"'goal_state', and 'constraint_region'; got {attribution.wrong_factor!r}. "
        f"(Other factors are reserved for later sub-projects — see "
        f"docs/superpowers/specs/2026-05-17-stage0-four-scene-roadmap-design.md §6)"
    )
```

Also update the parametrized `test_revise_intent_unhandled_factor_raises` test in `test_revision.py` — `constraint_region` is now handled. Find the parametrize list and remove `"constraint_region"` if present. Inspect the current list with:

```bash
grep -A6 "test_revise_intent_unhandled_factor_raises" tests/test_revision.py | head -15
```

If the list is `["object_motion", "constraint_region"]`, change it to `["object_motion"]`. If `constraint_region` is not there (Sub-project C may have removed it differently), leave the list alone.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_revision.py -v 2>&1 | tail -10`

Expected: all test_revision.py tests PASS (was 17 from C + new branch handles factor that may have been in unhandled-list).

- [ ] **Step 5: Full suite**

Run: `python -m pytest tests/ -q 2>&1 | tail -5`

Expected: `235 passed` (231 + 4 new tests, modulo any parametrize entry removed in Step 3).

- [ ] **Step 6: Commit**

```bash
git add babysteps/revision.py tests/test_revision.py
git commit -m "$(cat <<'EOF'
feat(revision): constraint_introduction operator for Sub-project D

The fourth Stage-0 revision branch — and the only one that revises
two factors at once. (none, faucet_base) → (faucet_base_static,
handle_grip). Strict-extension: any other source pair raises
NotImplementedError.

Mirrors the Sub-project C pattern: function docstring, module
docstring, final fallback message all refresh to list the new
supported wrong_factor. Removes constraint_region from any
"unhandled factor" parametrize (D handles it now).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `TurnSkill` + `compile_intent_to_turn_skill`

**Files:**
- Create: `babysteps/skills/turn.py`
- Test: `tests/test_turn_skill.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_turn_skill.py`:

```python
"""Tests for babysteps/skills/turn.py — pure TurnSkill geometry."""
from __future__ import annotations

import numpy as np
import pytest

from babysteps.schemas import Intent, SceneState


def _scene(handle_xy=(0.10, 0.0), base_xy=(0.05, 0.0), handle_z=0.10,
           axis_xy=(0.0, 1.0)):
    return SceneState(
        cube_xy=handle_xy,
        cube_z=handle_z,
        goal_xy=handle_xy,
        tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0),
        blocked_sides=(),
        extra={
            "handle_xy": handle_xy,
            "handle_z": handle_z,
            "faucet_base_xy": base_xy,
            "faucet_base_z": 0.0,
            "target_joint_axis_xy": axis_xy,
        },
    )


def _intent(contact_region="handle_grip", constraint_region="faucet_base_static"):
    return Intent(
        goal_state="faucet_turned",
        object_motion="turn",
        contact_region=contact_region,
        approach_direction="from_above",
        constraint_region=constraint_region,
        embodiment_mapping="proxy_contact_to_franka_turn",
    )


def test_compile_returns_turnskill_instance():
    from babysteps.skills.turn import TurnSkill, compile_intent_to_turn_skill
    skill = compile_intent_to_turn_skill(_intent(), _scene())
    assert isinstance(skill, TurnSkill)


def test_handle_grip_waypoints_target_handle_xy():
    from babysteps.skills.turn import compile_intent_to_turn_skill
    scene = _scene(handle_xy=(0.12, 0.04))
    skill = compile_intent_to_turn_skill(_intent("handle_grip"), scene)
    assert skill.waypoints.shape == (4, 7)
    # First three waypoints' xy == handle xy
    for i in range(3):
        assert skill.waypoints[i, 0] == pytest.approx(0.12)
        assert skill.waypoints[i, 1] == pytest.approx(0.04)


def test_faucet_base_waypoints_target_base_xy():
    from babysteps.skills.turn import compile_intent_to_turn_skill
    scene = _scene(base_xy=(0.06, -0.02))
    skill = compile_intent_to_turn_skill(_intent("faucet_base"), scene)
    assert skill.waypoints.shape == (4, 7)
    for i in range(3):
        assert skill.waypoints[i, 0] == pytest.approx(0.06)
        assert skill.waypoints[i, 1] == pytest.approx(-0.02)


def test_pull_waypoint_offset_along_joint_axis():
    """Waypoint 3 is contact_xy + axis_xy * TURN_PULL_DISTANCE_M."""
    from babysteps.skills.turn import (
        TURN_PULL_DISTANCE_M, compile_intent_to_turn_skill,
    )
    scene = _scene(handle_xy=(0.10, 0.05), axis_xy=(1.0, 0.0))
    skill = compile_intent_to_turn_skill(_intent("handle_grip"), scene)
    pull = skill.waypoints[3]
    assert pull[0] == pytest.approx(0.10 + TURN_PULL_DISTANCE_M)
    assert pull[1] == pytest.approx(0.05)


def test_compile_raises_on_unknown_contact_region():
    from babysteps.skills.turn import compile_intent_to_turn_skill
    scene = _scene()
    # minus_x_face is a cube-task contact_region; not handled by TurnSkill.
    bad_intent = Intent(
        goal_state="faucet_turned", object_motion="turn",
        contact_region="minus_x_face", approach_direction="from_above",
        constraint_region="faucet_base_static",
        embodiment_mapping="proxy_contact_to_franka_turn",
    )
    with pytest.raises(ValueError) as exc:
        compile_intent_to_turn_skill(bad_intent, scene)
    assert "minus_x_face" in str(exc.value)


def test_skill_exposes_contact_region_and_axis():
    from babysteps.skills.turn import compile_intent_to_turn_skill
    scene = _scene(axis_xy=(0.6, 0.8))
    skill = compile_intent_to_turn_skill(_intent("handle_grip"), scene)
    assert skill.contact_region == "handle_grip"
    assert skill.target_joint_axis_xy == pytest.approx((0.6, 0.8))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_turn_skill.py -v 2>&1 | tail -10`

Expected: 6 tests FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Create `babysteps/skills/turn.py`**

```python
"""Turn skill compiler — Sub-project D (TurnFaucet) approach-grip-pull.

Pure geometry compiler. Dispatches on intent.contact_region:
  - handle_grip  → waypoints target scene.extra["handle_xy"]
                   (the rotating switch link's centroid)
  - faucet_base  → waypoints target scene.extra["faucet_base_xy"]
                   (the static body's Stage-0 approximation)

The waypoint count is always 4: approach high above contact,
descend with clearance, grip (close gripper at contact_z), then
pull along target_joint_axis_xy for TURN_PULL_DISTANCE_M.

The skill never returns None — failure (when contact_region was
faucet_base) is detected at execution time by the env_runner setting
collision=True. ValueError fires only when intent.contact_region is
outside the D-supported subset (e.g., a cardinal cube face).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from babysteps.schemas import Intent, SceneState

DESCEND_CLEARANCE_M: float = 0.03
TURN_PULL_DISTANCE_M: float = 0.05


@dataclass(frozen=True)
class TurnSkill:
    """A compiled approach-grip-pull trajectory.

    waypoints is (4, 7). Columns are [x, y, z, qx, qy, qz, qw].
    contact_region is one of {"faucet_base", "handle_grip"} and is
    used by the env_runner for failure attribution.
    target_joint_axis_xy is the xy projection of the rotating joint's
    axis, used to direct the pull stroke.
    """
    waypoints: np.ndarray
    contact_region: str
    target_joint_axis_xy: tuple[float, float]


def compile_intent_to_turn_skill(
    intent: Intent, scene: SceneState,
) -> TurnSkill:
    if intent.contact_region == "handle_grip":
        contact_xy = scene.extra["handle_xy"]
        contact_z = scene.extra["handle_z"]
    elif intent.contact_region == "faucet_base":
        contact_xy = scene.extra["faucet_base_xy"]
        contact_z = scene.extra["faucet_base_z"]
    else:
        raise ValueError(
            f"compile_intent_to_turn_skill: contact_region must be one of "
            f"{{'faucet_base', 'handle_grip'}}, got {intent.contact_region!r}"
        )

    contact_xy = np.asarray(contact_xy, dtype=np.float64)
    axis_xy = np.asarray(scene.extra["target_joint_axis_xy"],
                          dtype=np.float64)
    pull_xy = contact_xy + axis_xy * TURN_PULL_DISTANCE_M
    tcp = np.asarray(scene.tcp_start_pose, dtype=np.float64)
    travel_z = float(tcp[2])

    wp = np.zeros((4, 7), dtype=np.float64)
    wp[0, 0:2] = contact_xy
    wp[0, 2] = travel_z
    wp[1, 0:2] = contact_xy
    wp[1, 2] = contact_z + DESCEND_CLEARANCE_M
    wp[2, 0:2] = contact_xy
    wp[2, 2] = contact_z
    wp[3, 0:2] = pull_xy
    wp[3, 2] = contact_z
    wp[:, 3:7] = tcp[3:7]

    return TurnSkill(
        waypoints=wp,
        contact_region=intent.contact_region,
        target_joint_axis_xy=(float(axis_xy[0]), float(axis_xy[1])),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_turn_skill.py -v 2>&1 | tail -10`

Expected: all 6 PASS.

- [ ] **Step 5: Full suite**

Run: `python -m pytest tests/ -q 2>&1 | tail -5`

Expected: `241 passed` (235 + 6).

- [ ] **Step 6: Commit**

```bash
git add babysteps/skills/turn.py tests/test_turn_skill.py
git commit -m "$(cat <<'EOF'
feat(skills): TurnSkill — Sub-project D approach-grip-pull compiler

Pure geometry compiler. 4-waypoint trajectory dispatched on
intent.contact_region (faucet_base vs handle_grip). Pull stroke
direction comes from scene.extra["target_joint_axis_xy"].

6 sim-free unit tests cover waypoint shape, per-contact xy targeting,
axis-derived pull direction, defensive ValueError on unknown contact,
and skill field exposure.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: `TurnFaucetAdapter` + `FakeTurnFaucetEnvRunner` + snapshot

**Files:**
- Create: `babysteps/envs/turnfaucet_adapter.py`
- Modify: `tests/conftest.py` (append FakeTurnFaucetEnvRunner + fixture)
- Test: `tests/test_turnfaucet_adapter.py`

This task BUNDLES adapter + fake runner + 13 tests + snapshot bootstrap.

- [ ] **Step 1: Append `FakeTurnFaucetEnvRunner` to `tests/conftest.py`**

At the end of `tests/conftest.py`, append:

```python
class FakeTurnFaucetEnvRunner:
    """Deterministic, sim-free env_runner for TurnFaucet unit tests.

    Outcome (keyed entirely off intent.contact_region + constraint_region):
      - contact_region == "handle_grip" AND
        constraint_region == "faucet_base_static"
        → success=True, faucet rotated
      - any other combination (typically faucet_base + none)
        → success=False, collision=True (constraint_violation proxy)
    """

    def __init__(self) -> None:
        self._scenes_by_seed: dict[int, SceneState] = {}

    def reset(self, seed: int) -> SceneState:
        if seed not in self._scenes_by_seed:
            rng = np.random.default_rng(seed)
            handle_xy = (
                float(rng.uniform(0.05, 0.12)),
                float(rng.uniform(-0.05, 0.05)),
            )
            handle_z = 0.10
            base_xy = (handle_xy[0] - 0.05, handle_xy[1])
            # Deterministic pull axis: +y.
            axis_xy = (0.0, 1.0)
            self._scenes_by_seed[seed] = SceneState(
                cube_xy=handle_xy,
                cube_z=handle_z,
                goal_xy=handle_xy,
                tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0),
                blocked_sides=(),
                extra={
                    "handle_xy": handle_xy,
                    "handle_z": handle_z,
                    "faucet_base_xy": base_xy,
                    "faucet_base_z": 0.0,
                    "target_joint_axis_xy": axis_xy,
                },
            )
        return self._scenes_by_seed[seed]

    def run(self, intent: Intent, scene: SceneState) -> AttemptResult:
        from babysteps.skills.turn import compile_intent_to_turn_skill
        skill = compile_intent_to_turn_skill(intent, scene)
        assert skill is not None

        handle_xy = tuple(float(v) for v in scene.extra["handle_xy"])
        if (intent.contact_region == "handle_grip"
                and intent.constraint_region == "faucet_base_static"):
            success, collision = True, False
        else:
            success, collision = False, True

        final_xy = handle_xy  # the handle stays where it is; rotation in place
        synthetic_traj = (handle_xy, final_xy)
        return AttemptResult(
            initial_obj_xy=handle_xy,
            final_obj_xy=final_xy,
            goal_xy=scene.goal_xy,
            reached_contact=True,
            object_moved=success,    # only "moves" when the joint rotated
            planner_failed=False,
            collision=collision,
            grasp_slip=False,
            rollout_log_path=None,
            success=success,
            trajectory_xy=synthetic_traj,
        )

    def close(self) -> None:
        pass


@pytest.fixture
def fake_turnfaucet_env_runner() -> FakeTurnFaucetEnvRunner:
    return FakeTurnFaucetEnvRunner()
```

- [ ] **Step 2: Write failing adapter tests**

Create `tests/test_turnfaucet_adapter.py`:

```python
"""Tests for babysteps/envs/turnfaucet_adapter.py.

Mirrors test_stackcube_adapter.py's shape: parity tests + snapshot."""
from __future__ import annotations

from pathlib import Path

import pytest

from babysteps.envs.task_adapter import BaseTaskAdapter
from babysteps.envs.turnfaucet_adapter import TurnFaucetAdapter
from babysteps.schemas import DemoEvidence, Intent, SceneState


def _scene_with_extra():
    return SceneState(
        cube_xy=(0.1, 0.0), cube_z=0.10, goal_xy=(0.1, 0.0),
        tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0),
        blocked_sides=(),
        extra={
            "handle_xy": (0.1, 0.0), "handle_z": 0.10,
            "faucet_base_xy": (0.05, 0.0), "faucet_base_z": 0.0,
            "target_joint_axis_xy": (0.0, 1.0),
        },
    )


def test_task_id_is_turnfaucet_v1():
    assert TurnFaucetAdapter.task_id == "TurnFaucet-v1"


def test_is_subclass_of_basetaskadapter():
    assert issubclass(TurnFaucetAdapter, BaseTaskAdapter)


def test_oracle_correct_intent_is_handle_grip_with_constraint():
    adapter = TurnFaucetAdapter()
    intent = adapter.oracle_correct_intent(_scene_with_extra())
    assert intent.contact_region == "handle_grip"
    assert intent.constraint_region == "faucet_base_static"
    assert intent.goal_state == "faucet_turned"
    assert intent.object_motion == "turn"
    assert intent.approach_direction == "from_above"
    assert intent.embodiment_mapping == "proxy_contact_to_franka_turn"


def test_default_blocked_factory_is_empty():
    intent = Intent(
        goal_state="faucet_turned", object_motion="turn",
        contact_region="faucet_base", approach_direction="from_above",
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_turn",
    )
    adapter = TurnFaucetAdapter()
    assert adapter.default_blocked_factory(intent) == ()


def test_oracle_wrong_factor_for_faucet_base_contact():
    intent = Intent(
        goal_state="faucet_turned", object_motion="turn",
        contact_region="faucet_base", approach_direction="from_above",
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_turn",
    )
    adapter = TurnFaucetAdapter()
    assert adapter.oracle_wrong_factor(intent, _scene_with_extra()) == "constraint_region"


def test_oracle_wrong_factor_for_correct_intent():
    intent = Intent(
        goal_state="faucet_turned", object_motion="turn",
        contact_region="handle_grip", approach_direction="from_above",
        constraint_region="faucet_base_static",
        embodiment_mapping="proxy_contact_to_franka_turn",
    )
    adapter = TurnFaucetAdapter()
    assert adapter.oracle_wrong_factor(intent, _scene_with_extra()) == "none"


def test_scripted_demo_to_intent_always_under_specifies_both():
    """Stage-0 controlled mechanism: scripted_demo_to_intent always
    returns contact_region=faucet_base AND constraint_region=none."""
    evidence = DemoEvidence(
        camera="third_person",
        demonstrator_type="proxy_oracle",
        object_trajectory=((0.0, 0.0), (0.1, 0.0)),
        contact_region_label="handle_grip",   # demo's true label
        final_state="faucet_turned",
        rgbd_video_path=None,
    )
    adapter = TurnFaucetAdapter()
    intent = adapter.scripted_demo_to_intent(evidence)
    assert intent.contact_region == "faucet_base"   # under-specified
    assert intent.constraint_region == "none"        # under-specified
    assert intent.goal_state == "faucet_turned"
    assert intent.object_motion == "turn"
    assert intent.embodiment_mapping == "proxy_contact_to_franka_turn"


def test_scripted_demo_to_intent_ignores_contact_region_label():
    """The label could be anything; the summarizer doesn't use it."""
    evidence_handle = DemoEvidence(
        camera="third_person", demonstrator_type="proxy_oracle",
        object_trajectory=((0.0, 0.0), (0.1, 0.0)),
        contact_region_label="handle_grip",
        final_state="faucet_turned", rgbd_video_path=None,
    )
    evidence_minus_x = DemoEvidence(
        camera="third_person", demonstrator_type="proxy_oracle",
        object_trajectory=((0.0, 0.0), (0.1, 0.0)),
        contact_region_label="minus_x_face",
        final_state="faucet_turned", rgbd_video_path=None,
    )
    adapter = TurnFaucetAdapter()
    i1 = adapter.scripted_demo_to_intent(evidence_handle)
    i2 = adapter.scripted_demo_to_intent(evidence_minus_x)
    assert i1 == i2
    assert i1.contact_region == "faucet_base"


def test_compile_skill_delegates_to_turn_skill():
    from babysteps.skills.turn import TurnSkill
    intent = Intent(
        goal_state="faucet_turned", object_motion="turn",
        contact_region="handle_grip", approach_direction="from_above",
        constraint_region="faucet_base_static",
        embodiment_mapping="proxy_contact_to_franka_turn",
    )
    adapter = TurnFaucetAdapter()
    skill = adapter.compile_skill(intent, _scene_with_extra())
    assert isinstance(skill, TurnSkill)


def test_adapter_inherits_default_hooks():
    assert (
        TurnFaucetAdapter.build_failure_packet
        is BaseTaskAdapter.build_failure_packet
    )
    assert (
        TurnFaucetAdapter.attribute_failure
        is BaseTaskAdapter.attribute_failure
    )
    assert (
        TurnFaucetAdapter.revise_intent
        is BaseTaskAdapter.revise_intent
    )


# ---------- end-to-end episode loop test ------------------------------ #


def test_full_episode_via_fake_runner_recovers_via_constraint_introduction(
    fake_turnfaucet_env_runner,
):
    from babysteps.episode import run_episode

    class _Adapter(TurnFaucetAdapter):
        def make_env_runner(self):
            return fake_turnfaucet_env_runner

    rec = run_episode(
        episode_id="turnfaucet_wrong_contact_seed_0000",
        seed=0,
        adapter=_Adapter(),
    )
    assert rec.metrics["initial_success"] is False
    assert rec.metrics["retry_success"] is True
    assert rec.metrics["factor_attribution_correct"] is True
    assert rec.metrics["frozen_factors_preserved"] is True
    # Two factors changed (constraint_introduction is two-factor).
    assert set(rec.metrics["factors_changed"]) == {"constraint_region", "contact_region"}
    assert rec.revision is not None
    assert rec.revision["operator"] == "constraint_introduction"
    assert rec.revision["factor"] == "constraint_region"
    assert rec.revision["old_value"] == "none"
    assert rec.revision["new_value"] == "faucet_base_static"


# ---------- Snapshot acceptance test --------------------------------- #


def test_turnfaucet_adapter_samples_jsonl_matches_snapshot(
    fake_turnfaucet_env_runner,
):
    from babysteps.episode import run_episode

    class _Adapter(TurnFaucetAdapter):
        def make_env_runner(self):
            return fake_turnfaucet_env_runner

    adapter = _Adapter()
    actual_lines = []
    for seed in range(5):
        rec = run_episode(
            episode_id=f"turnfaucet_wrong_contact_seed_{seed:04d}",
            seed=seed,
            adapter=adapter,
        )
        actual_lines.append(rec.to_jsonl_line())
    actual = "\n".join(actual_lines) + "\n"

    snapshot_path = (
        Path(__file__).parent / "snapshots" / "turnfaucet_samples_seeds_0_4.jsonl"
    )
    if not snapshot_path.exists():
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(actual)
        pytest.skip(
            f"Captured initial snapshot at {snapshot_path}. Re-run to "
            f"verify byte-equality."
        )
    expected = snapshot_path.read_text()
    assert actual == expected, (
        "TurnFaucetAdapter samples.jsonl drifted from the snapshot. "
        f"Snapshot at: {snapshot_path}. "
        "If intentional, delete the snapshot file and re-run this test "
        "to re-capture."
    )
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_turnfaucet_adapter.py -v 2>&1 | tail -20`

Expected: 12 tests FAIL (collection error from missing adapter module).

- [ ] **Step 4: Create `babysteps/envs/turnfaucet_adapter.py`**

```python
"""TurnFaucet-v1 adapter — the fourth concrete BaseTaskAdapter.

Pulls every TurnFaucet-specific decision behind one class:
  * make_env_runner       → TurnFaucetEnvRunner (Task 6)
  * oracle_correct_intent → handle_grip / faucet_base_static / turn
  * default_blocked_factory → () — no physical blocking; the Stage-0
                              controlled failure is the under-specified
                              (contact_region, constraint_region) pair
                              in scripted_demo_to_intent
  * oracle_wrong_factor   → "constraint_region" if intent.contact_region
                            == "faucet_base", else "none"
  * scripted_demo_to_intent → DELIBERATELY returns (contact_region=
                              "faucet_base", constraint_region="none").
                              The 2D summarizer can't distinguish the
                              handle from the body.
  * compile_skill         → wraps skills.turn.compile_intent_to_turn_skill

Hook defaults (build_failure_packet / attribute_failure / revise_intent)
are inherited unchanged from BaseTaskAdapter — the new
constraint_violation predicate (Task 2) and constraint_introduction
operator (Task 3) live in failure.py / revision.py and dispatch
automatically."""
from __future__ import annotations

from babysteps.envs.task_adapter import BaseTaskAdapter, EnvRunner
from babysteps.schemas import DemoEvidence, Intent, SceneState
from babysteps.skills.turn import compile_intent_to_turn_skill


class TurnFaucetAdapter(BaseTaskAdapter):
    task_id = "TurnFaucet-v1"

    def make_env_runner(self) -> EnvRunner:
        from babysteps.envs.turnfaucet_runner import TurnFaucetEnvRunner
        return TurnFaucetEnvRunner()

    def oracle_correct_intent(self, scene: SceneState) -> Intent:
        return Intent(
            goal_state="faucet_turned",
            object_motion="turn",
            contact_region="handle_grip",
            approach_direction="from_above",
            constraint_region="faucet_base_static",
            embodiment_mapping="proxy_contact_to_franka_turn",
        )

    def default_blocked_factory(self, intent: Intent) -> tuple[str, ...]:
        return ()

    def oracle_wrong_factor(
        self, initial_intent: Intent, scene_executor: SceneState,
    ) -> str:
        if initial_intent.contact_region == "faucet_base":
            return "constraint_region"
        return "none"

    def scripted_demo_to_intent(self, evidence: DemoEvidence) -> Intent:
        # Deliberately ignores evidence.contact_region_label —
        # the 2D summarizer can't distinguish the handle from the body.
        return Intent(
            goal_state="faucet_turned",
            object_motion="turn",
            contact_region="faucet_base",      # DELIBERATELY under-specified
            approach_direction="from_above",
            constraint_region="none",           # DELIBERATELY missing
            embodiment_mapping="proxy_contact_to_franka_turn",
        )

    def compile_skill(self, intent: Intent, scene: SceneState):
        return compile_intent_to_turn_skill(intent, scene)
```

- [ ] **Step 5: Run API tests (snapshot will bootstrap)**

Run: `python -m pytest tests/test_turnfaucet_adapter.py -v 2>&1 | tail -25`

Expected: 11 PASS + 1 SKIP (the snapshot test bootstraps).

- [ ] **Step 6: Re-run snapshot test to confirm byte-equality**

Run: `python -m pytest tests/test_turnfaucet_adapter.py::test_turnfaucet_adapter_samples_jsonl_matches_snapshot -v`

Expected: 1 PASS.

- [ ] **Step 7: Full suite**

Run: `python -m pytest tests/ -q 2>&1 | tail -5`

Expected: `253 passed` (241 + 12 new).

- [ ] **Step 8: Commit (bundle: conftest + adapter + tests + snapshot)**

```bash
git add babysteps/envs/turnfaucet_adapter.py tests/conftest.py tests/test_turnfaucet_adapter.py tests/snapshots/turnfaucet_samples_seeds_0_4.jsonl
git commit -m "$(cat <<'EOF'
feat(d): TurnFaucetAdapter + FakeTurnFaucetEnvRunner + snapshot

- babysteps/envs/turnfaucet_adapter.py — 5-method BaseTaskAdapter
  subclass. scripted_demo_to_intent deliberately under-specifies
  BOTH contact_region (=faucet_base) and constraint_region (=none).
  oracle_wrong_factor returns "constraint_region" for that case.
- tests/conftest.py — FakeTurnFaucetEnvRunner deterministic sim-free
  runner. Success iff intent.contact_region==handle_grip AND
  intent.constraint_region==faucet_base_static.
- tests/test_turnfaucet_adapter.py — 12 tests: 10 adapter API parity,
  1 full-episode (run_episode + constraint_introduction two-factor
  round-trip), 1 snapshot byte-stability.
- tests/snapshots/turnfaucet_samples_seeds_0_4.jsonl — captured.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: `TurnFaucetEnvRunner` (real ManiSkill)

**Files:**
- Create: `babysteps/envs/turnfaucet_runner.py`

No unit tests — needs Vulkan + asset download. Smoke import only.

- [ ] **Step 1: Create `babysteps/envs/turnfaucet_runner.py`**

```python
"""Real ManiSkill TurnFaucet-v1 env_runner.

Mirrors babysteps/envs/stackcube_runner.py with these differences:
- Reads target_link_pos (handle xyz) and target_joint_axis (3D axis)
  from obs.extra. Faucet base xy approximated as (handle_xy - (0.05, 0))
  for Stage-0; the real body root has variable per-model geometry.
- 4-phase trajectory (approach, descend, grip, pull). Gripper schedule
  always [OPEN, OPEN, CLOSED, CLOSED].
- Reports collision=True (Stage-0 proxy for constraint_violation)
  when contact_region=faucet_base AND not info["success"]. Otherwise
  collision=False.

Requires partnet_mobility_faucet asset download (see CLAUDE.md)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from babysteps.schemas import AttemptResult, Intent, SceneState
from babysteps.skills.turn import compile_intent_to_turn_skill


_POS_SCALE: float = 0.1
_PHASE_TOL_M: float = 0.015
_MAX_CONTROL_STEPS: int = 400
_GRIPPER_OPEN: float = 1.0
_GRIPPER_CLOSED: float = -1.0


def _to_np(x):
    arr = x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)
    return arr[0] if arr.ndim == 2 else arr


def _raw_to_xyzw(raw_pose) -> np.ndarray:
    raw = np.asarray(raw_pose, dtype=np.float64)
    return np.concatenate([raw[0:3], raw[4:7], raw[3:4]])


def _read_obs(obs):
    """(tcp_xyzw, handle_xyz, joint_axis_xyz) from TurnFaucet obs."""
    tcp = _raw_to_xyzw(_to_np(obs["extra"]["tcp_pose"]))
    handle_xyz = _to_np(obs["extra"]["target_link_pos"]).astype(np.float64)
    axis_xyz = _to_np(obs["extra"]["target_joint_axis"]).astype(np.float64)
    return tcp, handle_xyz, axis_xyz


def _prop_action(tcp_xyzw, target_xyz, gripper_cmd):
    pos_err = target_xyz - tcp_xyzw[0:3]
    action = np.zeros(7, dtype=np.float32)
    action[0:3] = np.clip(pos_err / _POS_SCALE, -1.0, 1.0).astype(np.float32)
    action[6] = np.float32(gripper_cmd)
    return action


class TurnFaucetEnvRunner:
    """Real ManiSkill TurnFaucet-v1 runner."""

    def __init__(self) -> None:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401 — registers TurnFaucet-v1

        self._env = gym.make(
            "TurnFaucet-v1",
            obs_mode="state_dict",
            control_mode="pd_ee_delta_pose",
            sim_backend="cpu",
        )
        self._last_seed: Optional[int] = None

    def reset(self, seed: int) -> SceneState:
        self._last_seed = int(seed)
        obs, _info = self._env.reset(seed=int(seed))
        tcp, handle_xyz, axis_xyz = _read_obs(obs)
        handle_xy = (float(handle_xyz[0]), float(handle_xyz[1]))
        handle_z = float(handle_xyz[2])
        base_xy = (handle_xy[0] - 0.05, handle_xy[1])  # Stage-0 approximation
        base_z = 0.0
        axis_xy = (float(axis_xyz[0]), float(axis_xyz[1]))
        return SceneState(
            cube_xy=handle_xy,
            cube_z=handle_z,
            goal_xy=handle_xy,
            tcp_start_pose=tuple(float(v) for v in tcp),  # type: ignore[arg-type]
            blocked_sides=(),
            extra={
                "handle_xy": handle_xy,
                "handle_z": handle_z,
                "faucet_base_xy": base_xy,
                "faucet_base_z": base_z,
                "target_joint_axis_xy": axis_xy,
            },
        )

    def run(
        self,
        intent: Intent,
        scene: SceneState,
        *,
        rollout_log_path: Optional[Path] = None,
    ) -> AttemptResult:
        skill = compile_intent_to_turn_skill(intent, scene)
        if self._last_seed is None:
            raise RuntimeError("TurnFaucetEnvRunner.run called before reset()")
        obs, _info = self._env.reset(seed=int(self._last_seed))
        tcp0, handle_xyz0, _axis0 = _read_obs(obs)
        initial_obj_xy = (float(handle_xyz0[0]), float(handle_xyz0[1]))

        targets = [np.asarray(wp[0:3], dtype=np.float64) for wp in skill.waypoints]
        phase_gripper = (_GRIPPER_OPEN, _GRIPPER_OPEN, _GRIPPER_CLOSED, _GRIPPER_CLOSED)

        trajectory: list[tuple[float, float]] = []
        phase_idx = 0
        reached_contact = False
        success = False
        for _step in range(_MAX_CONTROL_STEPS):
            tcp, handle_xyz, _axis = _read_obs(obs)
            trajectory.append((float(handle_xyz[0]), float(handle_xyz[1])))
            target = targets[phase_idx]
            if np.linalg.norm(target - tcp[0:3]) < _PHASE_TOL_M:
                phase_idx += 1
                if phase_idx >= len(targets):
                    break
                target = targets[phase_idx]
            # Contact heuristic: TCP near the chosen contact point.
            if phase_idx >= 1:
                cxy = (np.asarray(scene.extra["handle_xy"], dtype=np.float64)
                       if intent.contact_region == "handle_grip"
                       else np.asarray(scene.extra["faucet_base_xy"], dtype=np.float64))
                dxy = float(np.linalg.norm(tcp[0:2] - cxy))
                if dxy < 0.04:
                    reached_contact = True
            action = _prop_action(tcp, target, phase_gripper[phase_idx])
            obs, _r, terminated, truncated, info = self._env.step(action)
            term = bool(_to_np(terminated).item()) if hasattr(terminated, "cpu") else bool(terminated)
            trunc = bool(_to_np(truncated).item()) if hasattr(truncated, "cpu") else bool(truncated)
            succ_field = info.get("success", False) if hasattr(info, "get") else False
            success = bool(_to_np(succ_field).item()) if hasattr(succ_field, "cpu") else bool(succ_field)
            if success or term or trunc:
                break

        _tcp_f, handle_xyz_f, _axis_f = _read_obs(obs)
        final_obj_xy = (float(handle_xyz_f[0]), float(handle_xyz_f[1]))
        trajectory.append(final_obj_xy)

        object_moved = (
            float(np.linalg.norm(np.asarray(final_obj_xy) - np.asarray(initial_obj_xy)))
            > 0.005
        )

        # Stage-0 constraint_violation proxy:
        # if contact_region was faucet_base AND faucet didn't rotate,
        # mark collision=True so build_failure_packet emits the
        # constraint_violation predicate.
        collision = (intent.contact_region == "faucet_base" and not success)

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
            collision=bool(collision),
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

- [ ] **Step 2: Smoke import**

```bash
source /apps/external/conda/2025.09/etc/profile.d/conda.sh && conda activate handover && python -c "from babysteps.envs.turnfaucet_runner import TurnFaucetEnvRunner; print('class:', TurnFaucetEnvRunner.__name__)"
```

Expected: `class: TurnFaucetEnvRunner` without error.

- [ ] **Step 3: Full suite**

Run: `python -m pytest tests/ -q 2>&1 | tail -5`

Expected: still `253 passed`.

- [ ] **Step 4: Commit**

```bash
git add babysteps/envs/turnfaucet_runner.py
git commit -m "$(cat <<'EOF'
feat(d): TurnFaucetEnvRunner — real ManiSkill TurnFaucet-v1 runner

Mirrors StackCubeEnvRunner. Differences:
- Reads target_link_pos (handle) and target_joint_axis from obs.extra.
- Faucet base xy approximated as handle_xy - (0.05, 0); Stage-0 only.
- Always 4 phases: approach, descend, grip, pull.
- collision=True set when contact_region=faucet_base AND not success
  (Stage-0 proxy for constraint_violation predicate).

Real-sim correctness verified by GPU spot-check (Task 11 manual step).
Requires partnet_mobility_faucet asset download.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Registry entries + parity test

**Files:**
- Modify: `babysteps/envs/task_registry.py`
- Modify: `babysteps/render/__init__.py`
- Modify: `tests/test_task_registry.py`
- Modify: `tests/test_stage0_collect_cli.py` (rotate unknown-task sentinel)

- [ ] **Step 1: Update task_registry tests**

In `tests/test_task_registry.py`, find:

```python
def test_registry_contains_all_stage0_tasks():
    """PushCube-v1 (A), PickCube-v1 (B), and StackCube-v1 (C) must be present."""
    assert set(TASK_REGISTRY.keys()) == {
        "PushCube-v1", "PickCube-v1", "StackCube-v1",
    }
```

Replace with:

```python
def test_registry_contains_all_stage0_tasks():
    """PushCube-v1 (A), PickCube-v1 (B), StackCube-v1 (C), and TurnFaucet-v1 (D) must be present."""
    assert set(TASK_REGISTRY.keys()) == {
        "PushCube-v1", "PickCube-v1", "StackCube-v1", "TurnFaucet-v1",
    }
```

Update the unknown-task sentinel — find:

```python
def test_get_task_entry_unknown_task_raises():
    with pytest.raises(KeyError) as exc:
        get_task_entry("OpenCabinetDrawer-v1")
```

Replace `"OpenCabinetDrawer-v1"` with `"Bogus-v1"` (and update the assertions that reference it):

```python
def test_get_task_entry_unknown_task_raises():
    with pytest.raises(KeyError) as exc:
        get_task_entry("Bogus-v1")
    msg = str(exc.value)
    assert "Bogus-v1" in msg
    assert "PushCube-v1" in msg
    assert "PickCube-v1" in msg
    assert "StackCube-v1" in msg
    assert "TurnFaucet-v1" in msg
```

Append 2 new tests:

```python
def test_get_task_entry_turnfaucet():
    from babysteps.envs.turnfaucet_adapter import TurnFaucetAdapter
    entry = get_task_entry("TurnFaucet-v1")
    assert isinstance(entry, TaskEntry)
    assert entry.adapter_cls is TurnFaucetAdapter
    assert entry.episode_id_prefix == "turnfaucet_wrong_contact"


def test_fake_runner_factory_turnfaucet():
    entry = get_task_entry("TurnFaucet-v1")
    runner = entry.fake_runner_factory()
    assert hasattr(runner, "reset")
    assert hasattr(runner, "run")
    assert hasattr(runner, "close")
    runner.close()
```

Also in `tests/test_stage0_collect_cli.py`, find the unknown-task test and update `"OpenCabinetDrawer-v1"` → `"Bogus-v1"`:

```python
def test_stage0_collect_cli_unknown_task_errors(...):
    ...
    with pytest.raises(SystemExit) as exc_info:
        collect_main([
            "--task", "Bogus-v1",
            ...
```

- [ ] **Step 2: Verify the updated tests fail (registry/render not updated yet)**

Run: `python -m pytest tests/test_task_registry.py -v 2>&1 | tail -15`

Expected: 4 failures (renamed test + unknown-task rotation + 2 new turnfaucet tests).

- [ ] **Step 3: Add registry entries**

In `babysteps/envs/task_registry.py`, after `_stackcube_entry()`, add:

```python
def _turnfaucet_entry() -> TaskEntry:
    from babysteps.envs.turnfaucet_adapter import TurnFaucetAdapter

    def _make_fake() -> EnvRunner:
        from tests.conftest import FakeTurnFaucetEnvRunner
        return FakeTurnFaucetEnvRunner()

    return TaskEntry(
        adapter_cls=TurnFaucetAdapter,
        fake_runner_factory=_make_fake,
        episode_id_prefix="turnfaucet_wrong_contact",
    )
```

Then update `TASK_REGISTRY`:

```python
TASK_REGISTRY: dict[str, TaskEntry] = {
    "PushCube-v1": _pushcube_entry(),
    "PickCube-v1": _pickcube_entry(),
    "StackCube-v1": _stackcube_entry(),
    "TurnFaucet-v1": _turnfaucet_entry(),
}
```

In `babysteps/render/__init__.py`, after `_stackcube_render()`, add:

```python
def _turnfaucet_render() -> RenderEpisodeFn:
    from babysteps.render.turnfaucet import render_episode
    return render_episode
```

Update `RENDER_REGISTRY`:

```python
RENDER_REGISTRY: dict[str, Callable[[], RenderEpisodeFn]] = {
    "PushCube-v1": _pushcube_render,
    "PickCube-v1": _pickcube_render,
    "StackCube-v1": _stackcube_render,
    "TurnFaucet-v1": _turnfaucet_render,
}
```

(`babysteps.render.turnfaucet` is created in Task 8; lazy factory means the import only fires when `get_render_fn("TurnFaucet-v1")` is called.)

- [ ] **Step 4: Run task_registry + CLI tests**

Run: `python -m pytest tests/test_task_registry.py tests/test_stage0_collect_cli.py -v 2>&1 | tail -15`

Expected: all PASS.

- [ ] **Step 5: Full suite**

Run: `python -m pytest tests/ -q 2>&1 | tail -5`

Expected: `255 passed` (253 + 2 new task_registry tests).

- [ ] **Step 6: Commit**

```bash
git add babysteps/envs/task_registry.py babysteps/render/__init__.py tests/test_task_registry.py tests/test_stage0_collect_cli.py
git commit -m "$(cat <<'EOF'
feat(d): wire TurnFaucet-v1 into TASK_REGISTRY + RENDER_REGISTRY

One-row additions to both dispatch tables. The render module
(babysteps/render/turnfaucet.py) lands in the next task; the lazy
factory means RENDER_REGISTRY imports cleanly until then. Updates
the parity test to include TurnFaucet and rotates the unknown-task
sentinel to "Bogus-v1" in both test_task_registry and test_stage0_
collect_cli.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Render module + tests

**Files:**
- Create: `babysteps/render/turnfaucet.py`
- Modify: `tests/test_render_modules.py` (append 3 tests + stub env)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_render_modules.py`:

```python
# ---------- TurnFaucet render tests ---------------------------------- #


class _StubTurnEnv:
    """Stand-in for gym.make('TurnFaucet-v1').

    Obs has tcp_pose, target_link_pos (handle xyz), and
    target_joint_axis (3D)."""

    def __init__(self) -> None:
        self.tcp = np.array([0.0, 0.0, 0.25, 0.0, 0.0, 0.0], dtype=np.float64)
        self.handle = np.array([0.10, 0.0, 0.10], dtype=np.float64)
        self.axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        self._step_count = 0

    def reset(self, seed: int = 0):
        self.tcp = np.array([0.0, 0.0, 0.25, 0.0, 0.0, 0.0], dtype=np.float64)
        self.handle = np.array([0.10, 0.0, 0.10], dtype=np.float64)
        self.axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        self._step_count = 0
        return _StubTurnObs(self.tcp, self.handle, self.axis), {}

    def step(self, action):
        self.tcp[0:3] = self.tcp[0:3] + 0.02 * np.asarray(action[0:3])
        self._step_count += 1
        return (
            _StubTurnObs(self.tcp, self.handle, self.axis),
            0.0, False, False,
            {"success": False},
        )

    def render(self):
        return (np.ones((8, 8, 3), dtype=np.uint8) * (self._step_count % 256))

    def close(self):
        pass


@_dc
class _StubTurnObs:
    tcp: np.ndarray
    handle: np.ndarray
    axis: np.ndarray

    def __getitem__(self, key: str):
        if key == "extra":
            tcp_raw = np.concatenate([self.tcp[0:3], np.array([1.0]),
                                      self.tcp[3:6]])
            return {
                "tcp_pose": tcp_raw,
                "target_link_pos": self.handle,
                "target_joint_axis": self.axis,
            }
        raise KeyError(key)


def test_turnfaucet_render_episode_emits_three_phase_frames():
    from babysteps.render.turnfaucet import render_episode
    from babysteps.envs.turnfaucet_adapter import TurnFaucetAdapter

    env = _StubTurnEnv()
    adapter = TurnFaucetAdapter()
    frames, titles = render_episode(env, adapter, seed=0, fps=4)

    assert set(frames.keys()) == {"demo", "attempt_blocked", "retry"}
    assert set(titles.keys()) == {"demo", "attempt_blocked", "retry"}
    assert len(frames["demo"]) >= 2
    assert len(frames["attempt_blocked"]) >= 2
    assert len(frames["retry"]) >= 2


def test_turnfaucet_render_phase2_actually_steps_env():
    from babysteps.render.turnfaucet import render_episode
    from babysteps.envs.turnfaucet_adapter import TurnFaucetAdapter

    env = _StubTurnEnv()
    frames, _ = render_episode(env, TurnFaucetAdapter(), seed=0, fps=4)
    held = frames["attempt_blocked"]
    assert not all(np.array_equal(held[0], f) for f in held), (
        "TurnFaucet phase 2 should step the env."
    )


def test_turnfaucet_render_titles_mention_constraint_region():
    from babysteps.render.turnfaucet import render_episode
    from babysteps.envs.turnfaucet_adapter import TurnFaucetAdapter
    _, titles = render_episode(_StubTurnEnv(), TurnFaucetAdapter(), seed=0, fps=4)
    # Demo subtitle mentions the oracle's constraint_region.
    assert "faucet_base_static" in titles["demo"][1]
    # Retry subtitle mentions constraint_introduction.
    assert "constraint_introduction" in titles["retry"][1]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_render_modules.py -v -k "turnfaucet" 2>&1 | tail -10`

Expected: 3 FAIL.

- [ ] **Step 3: Create `babysteps/render/turnfaucet.py`**

```python
"""TurnFaucet-v1 render_episode — three phases for the Stage-0 MP4 set.

Phase 1 (demo): oracle intent (handle_grip + faucet_base_static).
Faucet rotates.
Phase 2 (attempt_blocked): scripted intent (faucet_base + none).
Gripper touches the static body, no rotation. Tail-padded.
Phase 3 (retry): constraint_introduction-revised intent (handle_grip
+ faucet_base_static). Faucet rotates.

Like PickCube and StackCube (unlike PushCube), all three phases step
the env."""
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
from babysteps.skills.turn import compile_intent_to_turn_skill


_GRIPPER_OPEN = 1.0
_GRIPPER_CLOSED = -1.0


def _read_turn_obs(obs):
    """(tcp_xyzw, handle_xyz, axis_xyz) from TurnFaucet obs."""
    tcp_raw = np.asarray(to_np(obs["extra"]["tcp_pose"]), dtype=np.float64)
    tcp = np.concatenate([tcp_raw[0:3], tcp_raw[4:7], tcp_raw[3:4]])
    handle_xyz = np.asarray(to_np(obs["extra"]["target_link_pos"]), dtype=np.float64)
    axis_xyz = np.asarray(to_np(obs["extra"]["target_joint_axis"]), dtype=np.float64)
    return tcp, handle_xyz, axis_xyz


def _execute_turn(env, intent, scene, frames, *, seed):
    skill = compile_intent_to_turn_skill(intent, scene)
    obs, _ = env.reset(seed=int(seed))
    targets = [np.asarray(wp[0:3], dtype=np.float64) for wp in skill.waypoints]
    phase_gripper = (_GRIPPER_OPEN, _GRIPPER_OPEN, _GRIPPER_CLOSED, _GRIPPER_CLOSED)

    phase_idx = 0
    success = False
    frames.append(render_frame(env))
    for _ in range(STACKCUBE_MAX_CONTROL_STEPS):
        tcp, _h, _a = _read_turn_obs(obs)
        target = targets[phase_idx]
        if np.linalg.norm(target - tcp[0:3]) < PHASE_TOL_M:
            phase_idx += 1
            if phase_idx >= len(targets):
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


def render_episode(env, adapter, seed, fps):
    short_id = f"seed {seed:04d}"

    # === Phase 1 — DEMO (oracle's handle_grip + faucet_base_static) ===
    obs, _ = env.reset(seed=seed)
    tcp, handle_xyz, axis_xyz = _read_turn_obs(obs)
    handle_xy = (float(handle_xyz[0]), float(handle_xyz[1]))
    handle_z = float(handle_xyz[2])
    base_xy = (handle_xy[0] - 0.05, handle_xy[1])
    axis_xy = (float(axis_xyz[0]), float(axis_xyz[1]))
    scene = SceneState(
        cube_xy=handle_xy, cube_z=handle_z, goal_xy=handle_xy,
        tcp_start_pose=tuple(float(v) for v in tcp),  # type: ignore[arg-type]
        blocked_sides=(),
        extra={
            "handle_xy": handle_xy, "handle_z": handle_z,
            "faucet_base_xy": base_xy, "faucet_base_z": 0.0,
            "target_joint_axis_xy": axis_xy,
        },
    )
    correct_intent = adapter.oracle_correct_intent(scene)
    demo_frames: list = []
    _ = _execute_turn(env, correct_intent, scene, demo_frames, seed=seed)

    demo_evidence = DemoEvidence(
        camera="third_person",
        demonstrator_type="proxy_oracle",
        object_trajectory=(handle_xy, handle_xy),
        contact_region_label="handle_grip",
        final_state="faucet_turned",
        rgbd_video_path=None,
    )
    initial_intent = adapter.scripted_demo_to_intent(demo_evidence)
    scene_exec = replace(
        scene, blocked_sides=adapter.default_blocked_factory(initial_intent),
    )

    # === Phase 2 — ATTEMPT 1 (faucet_base + none; collision, no rotation) ===
    attempt1_frames: list = []
    _ = _execute_turn(env, initial_intent, scene_exec, attempt1_frames, seed=seed)

    # === Phase 3 — RETRY (constraint_introduction-revised) ===
    fp = adapter.build_failure_packet(
        initial_intent,
        AttemptResult(
            initial_obj_xy=scene.cube_xy, final_obj_xy=scene.cube_xy,
            goal_xy=scene.goal_xy,
            reached_contact=True, object_moved=False,
            planner_failed=False, collision=True, grasp_slip=False,
            rollout_log_path=None, success=False,
        ),
        scene_exec,
    )
    attribution = adapter.attribute_failure(fp)
    revised_intent, _rev = adapter.revise_intent(initial_intent, attribution, scene_exec)
    retry_frames: list = []
    out_retry = _execute_turn(env, revised_intent, scene_exec, retry_frames, seed=seed)

    demo_title = (
        f"{short_id}  phase 1/3: demo proxy",
        f"contact_region={correct_intent.contact_region}, "
        f"constraint_region={correct_intent.constraint_region}",
    )
    a1_title = (
        f"{short_id}  phase 2/3: constraint_violation",
        f"contact_region={initial_intent.contact_region} (faucet body) → no rotation",
    )
    retry_title = (
        f"{short_id}  phase 3/3: retry (success={out_retry['success']})",
        f"constraint_introduction: "
        f"({initial_intent.constraint_region}, {initial_intent.contact_region}) → "
        f"({revised_intent.constraint_region}, {revised_intent.contact_region})",
    )

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

- [ ] **Step 4: Run render tests**

Run: `python -m pytest tests/test_render_modules.py -v 2>&1 | tail -15`

Expected: all 11 PASS (was 8 from C, + 3 new D).

- [ ] **Step 5: Full suite**

Run: `python -m pytest tests/ -q 2>&1 | tail -5`

Expected: `258 passed` (255 + 3).

- [ ] **Step 6: Commit**

```bash
git add babysteps/render/turnfaucet.py tests/test_render_modules.py
git commit -m "$(cat <<'EOF'
feat(render): babysteps.render.turnfaucet — three-phase TurnFaucet renderer

All three phases step the env. Phase 2 shows the gripper touching the
faucet body with no rotation (constraint_violation visual). Phase 3
shows the gripper at the handle, rotating the faucet (after the
constraint_introduction two-factor revision).

3 stub-env tests cover frame contract, phase-2-actually-steps-env,
and title mentions of constraint_region + constraint_introduction.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: CLI snapshot extension

**Files:**
- Modify: `tests/test_stage0_collect_cli.py` (extend parametrize)

- [ ] **Step 1: Extend parametrize**

In `tests/test_stage0_collect_cli.py`, find:

```python
@pytest.mark.parametrize("task_id,snapshot_name", [
    ("PushCube-v1", "pushcube_samples_seeds_0_4.jsonl"),
    ("PickCube-v1", "pickcube_samples_seeds_0_4.jsonl"),
    ("StackCube-v1", "stackcube_samples_seeds_0_4.jsonl"),
])
```

Replace with:

```python
@pytest.mark.parametrize("task_id,snapshot_name", [
    ("PushCube-v1", "pushcube_samples_seeds_0_4.jsonl"),
    ("PickCube-v1", "pickcube_samples_seeds_0_4.jsonl"),
    ("StackCube-v1", "stackcube_samples_seeds_0_4.jsonl"),
    ("TurnFaucet-v1", "turnfaucet_samples_seeds_0_4.jsonl"),
])
```

- [ ] **Step 2: Run parametrized snapshot test**

Run: `python -m pytest tests/test_stage0_collect_cli.py::test_stage0_collect_cli_matches_snapshot -v 2>&1 | tail -10`

Expected: all 4 parametrized PASS.

- [ ] **Step 3: Full suite**

Run: `python -m pytest tests/ -q 2>&1 | tail -5`

Expected: `259 passed` (258 + 1).

- [ ] **Step 4: Commit**

```bash
git add tests/test_stage0_collect_cli.py
git commit -m "$(cat <<'EOF'
test(d): end-to-end CLI snapshot for TurnFaucet-v1

Extends test_stage0_collect_cli_matches_snapshot's parametrize to
include the TurnFaucet snapshot captured in Task 5.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: delta_pp gate test

**Files:**
- Modify: `tests/test_pickcube_delta_pp.py`

- [ ] **Step 1: Append test**

```python
def test_turnfaucet_fake_env_meets_delta_pp_gate(tmp_path: Path, collect_main):
    """Sub-project D acceptance: TurnFaucet fake-env should achieve
    delta_pp >= 10. With FakeTurnFaucetEnvRunner's deterministic
    outcome (success iff handle_grip + faucet_base_static), all 5
    seeds follow under-specified → constraint_introduction → success
    arc, yielding delta_pp = 100.0."""
    out_dir = tmp_path / "out"
    rc = collect_main([
        "--task", "TurnFaucet-v1",
        "--fake-env",
        "--out_dir", str(out_dir),
        "--n_episodes", "5",
        "--seed_start", "0",
    ])
    report = json.loads((out_dir / "report.json").read_text())
    assert report["delta_pp"] >= 10.0, (
        f"TurnFaucet fake-env delta_pp = {report['delta_pp']:.1f}. "
        f"Initial {report['initial_attempt_success_rate']:.2f}, "
        f"retry {report['retry_success_rate']:.2f}, n_with_revision="
        f"{report['n_with_revision']}, n_retry_success={report['n_retry_success']}."
    )
    assert report["passed_acceptance"] is True
    assert rc == 0
```

- [ ] **Step 2: Run + full suite + commit**

```bash
python -m pytest tests/test_pickcube_delta_pp.py -v 2>&1 | tail -10
python -m pytest tests/ -q 2>&1 | tail -5
```

Expected: 4 delta_pp tests pass; full suite `260 passed`.

```bash
git add tests/test_pickcube_delta_pp.py
git commit -m "$(cat <<'EOF'
test(d): acceptance-gate test — TurnFaucet delta_pp >= 10 via fake env

Codifies Sub-project D's acceptance gate item 5. Deterministic fake
returns delta_pp = 100.0 for the under-specified → constraint_
introduction → success arc.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: CLAUDE.md + final gate verification

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add fourth srun block + asset download note**

In CLAUDE.md, find the StackCube srun block (added in Sub-project C Task 10). After its closing `'`, before the closing ``` of the bash fence, add:

```bash

# TurnFaucet (Sub-project D — constraint_violation; closes D's acceptance gate item 4)
# Requires partnet_mobility_faucet asset (one-time):
#   python -m mani_skill.utils.download_asset partnet_mobility_faucet
srun --account=rpaleja --partition=a100-40gb --gres=gpu:1 --mem=115G --time=00:20:00 bash -lc '
  cd /scratch/gilbreth/wang4433/babysteps &&
  source /apps/external/conda/2025.09/etc/profile.d/conda.sh &&
  conda activate handover &&
  OUT_DIR=/scratch/gilbreth/wang4433/render_turnfaucet &&
  LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" \
  python scripts/render_stage0_maniskill.py \
    --task TurnFaucet-v1 \
    --out_dir "$OUT_DIR" \
    --n_episodes 2 \
    --seed_start 0 &&
  ls -lh "$OUT_DIR/videos_maniskill"
'
```

- [ ] **Step 2: Update "- Code:" line**

Find:

```
- Code:   `babysteps/` (pure modules) + `babysteps/envs/{pushcube,pickcube,stackcube}_runner.py` (sim adapters),
          `babysteps/envs/task_registry.py` (--task dispatch),
          `babysteps/render/{pushcube,pickcube,stackcube}.py` (per-task MP4 flows)
```

Replace with:

```
- Code:   `babysteps/` (pure modules) + `babysteps/envs/{pushcube,pickcube,stackcube,turnfaucet}_runner.py` (sim adapters),
          `babysteps/envs/task_registry.py` (--task dispatch),
          `babysteps/render/{pushcube,pickcube,stackcube,turnfaucet}.py` (per-task MP4 flows)
```

- [ ] **Step 3: Update "- Scripts:" line**

Find:

```
- Scripts: `scripts/{stage0_collect,render_stage0_maniskill}.py` accept `--task {PickCube-v1,PushCube-v1,StackCube-v1}`.
```

Replace with:

```
- Scripts: `scripts/{stage0_collect,render_stage0_maniskill}.py` accept `--task {PickCube-v1,PushCube-v1,StackCube-v1,TurnFaucet-v1}`.
```

- [ ] **Step 4: Update "- Tests:" line**

Find:

```
- Tests:  221 sim-free unit tests in `tests/` (PushCube + PickCube + StackCube, snapshot-stable across all three)
```

Replace with:

```
- Tests:  260 sim-free unit tests in `tests/` (PushCube + PickCube + StackCube + TurnFaucet, snapshot-stable across all four)
```

- [ ] **Step 5: Run final gate verification**

```bash
source /apps/external/conda/2025.09/etc/profile.d/conda.sh && conda activate handover
python -m pytest tests/ -q 2>&1 | tail -3
# All 4 byte-equality checks:
for task_pair in "PushCube-v1 pushcube" "PickCube-v1 pickcube" "StackCube-v1 stackcube" "TurnFaucet-v1 turnfaucet"; do
  task=$(echo $task_pair | awk '{print $1}')
  prefix=$(echo $task_pair | awk '{print $2}')
  echo "--- $task ---"
  python scripts/stage0_collect.py --task $task --out_dir /tmp/d-gate-$prefix --n_episodes 5 --seed_start 0 --fake-env > /dev/null
  diff -q tests/snapshots/${prefix}_samples_seeds_0_4.jsonl /tmp/d-gate-$prefix/samples.jsonl
  cat /tmp/d-gate-$prefix/report.json | python -c "import json,sys; d=json.load(sys.stdin); print('  delta_pp:', d['delta_pp'], 'passed:', d['passed_acceptance'])"
done
# Registry parity:
python -c "
from babysteps.envs.task_registry import TASK_REGISTRY
from babysteps.render import RENDER_REGISTRY
assert sorted(TASK_REGISTRY) == sorted(RENDER_REGISTRY) == ['PickCube-v1', 'PushCube-v1', 'StackCube-v1', 'TurnFaucet-v1']
print('parity OK')
"
```

Expected:
- 260 passed.
- 4 diffs all silent (byte-identical).
- 4 `delta_pp: 100.0 passed: True`.
- `parity OK`.

- [ ] **Step 6: Commit (CLAUDE.md only)**

```bash
git add CLAUDE.md
git commit -m "$(cat <<'EOF'
docs(claude.md): add TurnFaucet GPU command + module/test refresh

Adds the fourth srun block (TurnFaucet-v1, with asset-download
prerequisite). Refreshes the Code/Scripts/Tests lines to include
TurnFaucet. Test count: 260.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 7: GPU spot-check (MANUAL)**

User schedules the TurnFaucet `srun` command from CLAUDE.md after running `python -m mani_skill.utils.download_asset partnet_mobility_faucet` once. Confirm `videos_maniskill/` contains 6 MP4s. Visually verify phase 2 shows no faucet rotation and phase 3 shows rotation.

- [ ] **Step 8: Tag (DEFERRED TO USER)**

```bash
git tag stage0-turnfaucet-d-gate-pass
```

Do NOT push.

---

## Self-Review

**Spec coverage:**
- §1 Motivation: framed in plan header.
- §2 Stage-0 controlled failure: documented in TurnFaucetAdapter and TurnSkill docstrings.
- §3.1-3 + 5: Tasks 0, 1-5, 9, 10.
- §3.4 (GPU spot-check): Task 11 step 7.
- §4 Schema deltas: Task 1.
- §5 Failure attribution: Task 2.
- §6 Revision operator: Task 3.
- §7 Adapter: Task 5.
- §8 Skill: Task 4.
- §9 Runner: Task 6.
- §10 ManiSkill facts: encoded in Task 6's _read_obs (target_link_pos / target_joint_axis).
- §11 FakeTurnFaucetEnvRunner: Task 5.
- §12 Render module: Task 8.
- §13 CLI integration: Task 7.
- §14 Test plan: Tasks 1-10.
- §15 Risks: addressed inline.

**Type consistency:**
- `TurnSkill` shape: waypoints + contact_region + target_joint_axis_xy — consistent.
- `episode_id_prefix = "turnfaucet_wrong_contact"` — consistent across Tasks 5, 7, 11.
- collision flag repurposing — consistent between failure.py predicate logic and turnfaucet_runner.py setting.
- Phase names: standard.

**Risks:**
1. The `collision` flag repurposing could break a future task that wants "real" collision semantics. Documented in spec §15.
2. `partnet_mobility_faucet` download is a new user prerequisite — explicitly noted in CLAUDE.md.
3. The two-factor revision is a precedent. Future operators may want to revise 3+ factors; the `Revision.factor` / `old_value` / `new_value` fields will need extension (currently store only the primary). Not a Stage-0 concern.
