# Stage-0 TaskAdapter Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the Stage-0 PushCube-only orchestration into a `TaskAdapter` interface so Sub-projects B (PickCube) and C (StackCube) can slot in without touching the loop. PushCube behaviour stays byte-identical.

**Architecture:** Introduce `BaseTaskAdapter` (ABC) at `babysteps/envs/task_adapter.py` with 6 abstract methods + 3 overridable hooks (defaulting to the shared `failure` / `revision` modules). Reimplement PushCube as `PushCubeAdapter`. Rename `babysteps/execution.py` → `babysteps/skills/push.py`. Move the push-specific `demo_to_intent` into the adapter; keep `trajectory_to_motion` as a shared utility. `episode.run_episode` becomes pure orchestration, parameterized by `adapter:`.

**Tech Stack:** Python 3.10+, `dataclasses`, `abc.ABC`, `pytest`. No new third-party deps. ManiSkill (`mani_skill`) only matters for the optional `PushCubeEnvRunner` path; fake-env runs all CI tests.

**Reference docs (read first):**
- `docs/superpowers/specs/2026-05-16-stage0-task-adapter-refactor-design.md` — design spec, the source of truth for this plan.
- `goal.md` — Stage-0 boundary (object-centric intent factors).
- `docs/superpowers/specs/2026-05-15-stage0-pushcube-blocked-design.md` — the predecessor (what's being refactored).

**Pre-flight (one-time, manual):** `/scratch/gilbreth/wang4433/babysteps` is **not currently a git repository**. The plan's `git add` / `git commit` steps assume `git init` has been run from the repo root. If you want to skip git, omit the commit steps — the rest of the plan still works.

```bash
cd /scratch/gilbreth/wang4433/babysteps
git init
git add CLAUDE.md README.md goal.md technical_def.md pyproject.toml \
        babysteps/ scripts/ tests/ docs/
git commit -m "chore: initialize repo before Sub-project A refactor"
```

---

## File Structure (after this plan)

```
babysteps/
  __init__.py                       (unchanged)
  schemas.py                        EDIT: SceneState gains `extra: dict`
  demo.py                           EDIT: only trajectory_to_motion remains
  episode.py                        EDIT: takes adapter:, all push helpers gone
  failure.py                        (unchanged)
  revision.py                       (unchanged)
  eval.py, viz.py                   (unchanged)
  execution.py                      DELETED
  skills/
    __init__.py                     NEW (1 line)
    push.py                         NEW = old execution.py verbatim
  envs/
    __init__.py                     (unchanged)
    scene.py                        (unchanged)
    pushcube_runner.py              EDIT: import path only
    task_adapter.py                 NEW — EnvRunner Protocol + BaseTaskAdapter ABC
    pushcube_adapter.py             NEW — PushCubeAdapter(BaseTaskAdapter)

scripts/
  stage0_collect.py                 EDIT: PushCubeAdapter wiring
  stage0_summarize.py               (unchanged)
  smoke_pushcube.py                 (unchanged)
  render_stage0_maniskill.py        EDIT: PushCubeAdapter + skills.push imports
  render_stage0_topdown.py          (check, edit if it imports execution.py)

tests/
  conftest.py                       EDIT: import path only
  test_schemas.py                   EDIT: 1 new SceneState test
  test_demo.py                      EDIT: 4 demo_to_intent tests moved out
  test_execution.py                 RENAMED → test_push_skill.py (imports updated)
  test_episode.py                   EDIT: signature update + spy rewrite
  test_failure.py, test_revision.py, test_eval.py  (unchanged)
  test_task_adapter.py              NEW
  test_pushcube_adapter.py          NEW (includes the snapshot test)
  snapshots/
    pushcube_samples_seeds_0_4.jsonl  NEW (baseline fixture)
```

---

## Tasks

### Task 0: Verify clean baseline

Before any changes, confirm everything currently passes so deviations are attributable to the refactor.

**Files:** none

- [ ] **Step 1: Activate env and run the full test suite**

```bash
cd /scratch/gilbreth/wang4433/babysteps
conda activate handover
python -m pytest tests/ -q
```

Expected: all 85 tests pass.

- [ ] **Step 2: Verify the fake-env data pipeline works end-to-end**

```bash
rm -rf /tmp/babysteps_baseline
python scripts/stage0_collect.py \
    --out_dir /tmp/babysteps_baseline \
    --n_episodes 5 --seed_start 0 --fake-env
```

Expected: exit 0; `/tmp/babysteps_baseline/samples.jsonl` has exactly 5 lines; `report.md` reports `passed_acceptance: True`.

- [ ] **Step 3: Record line count + first line for visual sanity**

```bash
wc -l /tmp/babysteps_baseline/samples.jsonl
head -c 400 /tmp/babysteps_baseline/samples.jsonl
```

If anything fails, **stop and fix the baseline first** — the refactor cannot start from a broken tree.

---

### Task 1: Capture the snapshot fixture

The byte-equality regression bar for the whole refactor lives in `tests/snapshots/`.

**Files:**
- Create: `tests/snapshots/pushcube_samples_seeds_0_4.jsonl`

- [ ] **Step 1: Generate the baseline JSONL**

```bash
cd /scratch/gilbreth/wang4433/babysteps
python scripts/stage0_collect.py \
    --out_dir /tmp/babysteps_baseline \
    --n_episodes 5 --seed_start 0 --fake-env
```

Expected: exit 0, 5 lines in `/tmp/babysteps_baseline/samples.jsonl`.

- [ ] **Step 2: Copy into the test fixtures directory**

```bash
mkdir -p tests/snapshots
cp /tmp/babysteps_baseline/samples.jsonl tests/snapshots/pushcube_samples_seeds_0_4.jsonl
wc -l tests/snapshots/pushcube_samples_seeds_0_4.jsonl
```

Expected: 5 lines.

- [ ] **Step 3: Commit the fixture**

```bash
git add tests/snapshots/pushcube_samples_seeds_0_4.jsonl
git commit -m "test: snapshot pushcube samples.jsonl baseline (pre-A)"
```

---

### Task 2: Extend `SceneState` with `extra: dict`

Add the forward-compatibility field for B/C. Preserve PushCube JSON byte-equality (empty-skip serialization).

**Files:**
- Modify: `babysteps/schemas.py:143-181` (SceneState block)
- Modify: `tests/test_schemas.py` (add one new test)

- [ ] **Step 1: Add failing test in `tests/test_schemas.py`**

Append at the end of the SceneState section (after `test_scene_roundtrip_tuple_blocked_sides`):

```python
def test_scene_roundtrip_with_extra():
    s = SceneState(
        cube_xy=(0.0, 0.0),
        cube_z=0.02,
        goal_xy=(0.2, 0.05),
        tcp_start_pose=(0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0),
        blocked_sides=(),
        extra={"gripper_width": 0.08, "base_cube_xy": [0.1, 0.0]},
    )
    rt = SceneState.from_dict(s.to_dict())
    assert rt.extra == {"gripper_width": 0.08, "base_cube_xy": [0.1, 0.0]}


def test_scene_empty_extra_omitted_from_json():
    """Empty extra must NOT appear as a key in to_dict — this is what
    preserves byte-for-byte JSON equality for pre-A PushCube records."""
    s = SceneState(
        cube_xy=(0.0, 0.0), cube_z=0.02, goal_xy=(0.2, 0.05),
        tcp_start_pose=(0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0),
        blocked_sides=(),
    )
    d = s.to_dict()
    assert "extra" not in d
    # And the default is an empty dict, round-trippable.
    rt = SceneState.from_dict(d)
    assert rt.extra == {}


def test_scene_default_extra_is_empty_dict():
    s = SceneState(
        cube_xy=(0.0, 0.0), cube_z=0.02, goal_xy=(0.2, 0.05),
        tcp_start_pose=(0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0),
        blocked_sides=(),
    )
    assert s.extra == {}
```

- [ ] **Step 2: Run the new tests, confirm they fail**

```bash
python -m pytest tests/test_schemas.py -k extra -v
```

Expected: 3 failures (extra is not a known argument to SceneState).

- [ ] **Step 3: Edit `babysteps/schemas.py` — extend SceneState**

Replace the existing SceneState block (lines ~143–181) with:

```python
@dataclass(frozen=True)
class SceneState:
    """Simulator-side ground truth + feasibility flags.

    Every field is privileged — must not flow into demo_to_intent. Consumed
    only by the skill compiler (waypoint geometry + blocked_sides feasibility
    check) and by metric computation (oracle labels).

    `extra` is an adapter-owned payload for forward compatibility with non-
    push tasks (PickCube populates gripper_width etc.; StackCube populates a
    second cube's pose). It is serialized only when non-empty so PushCube
    records remain byte-identical to pre-A snapshots."""

    cube_xy: tuple[float, float]
    cube_z: float
    goal_xy: tuple[float, float]
    tcp_start_pose: tuple[float, float, float, float, float, float, float]
    blocked_sides: tuple[str, ...]
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "cube_xy": list(self.cube_xy),
            "cube_z": float(self.cube_z),
            "goal_xy": list(self.goal_xy),
            "tcp_start_pose": list(self.tcp_start_pose),
            "blocked_sides": list(self.blocked_sides),
        }
        if self.extra:
            d["extra"] = dict(self.extra)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SceneState":
        cube_xy = tuple(float(v) for v in d["cube_xy"])
        goal_xy = tuple(float(v) for v in d["goal_xy"])
        tcp = tuple(float(v) for v in d["tcp_start_pose"])
        if len(cube_xy) != 2 or len(goal_xy) != 2:
            raise ValueError("cube_xy and goal_xy must have length 2")
        if len(tcp) != 7:
            raise ValueError(f"tcp_start_pose must have length 7, got {len(tcp)}")
        return cls(
            cube_xy=cube_xy,          # type: ignore[arg-type]
            cube_z=float(d["cube_z"]),
            goal_xy=goal_xy,          # type: ignore[arg-type]
            tcp_start_pose=tcp,        # type: ignore[arg-type]
            blocked_sides=tuple(d["blocked_sides"]),
            extra=dict(d.get("extra", {})),
        )
```

- [ ] **Step 4: Run the new tests, confirm they pass**

```bash
python -m pytest tests/test_schemas.py -k extra -v
```

Expected: 3 passes.

- [ ] **Step 5: Run the full suite — confirm no regressions**

```bash
python -m pytest tests/ -q
```

Expected: all pre-existing tests still pass; 3 new SceneState tests pass; zero failures.

- [ ] **Step 6: Commit**

```bash
git add babysteps/schemas.py tests/test_schemas.py
git commit -m "feat(schemas): SceneState.extra dict for per-task payloads (empty-skip)"
```

---

### Task 3: Move `execution.py` to `babysteps/skills/push.py`

Pure rename + import-path updates. No behaviour change.

**Files:**
- Create: `babysteps/skills/__init__.py`
- Create: `babysteps/skills/push.py` (verbatim copy of `babysteps/execution.py`)
- Delete: `babysteps/execution.py`
- Modify: `babysteps/envs/pushcube_runner.py` (1 import line)
- Modify: `tests/conftest.py` (1 import line)
- Modify: `scripts/render_stage0_maniskill.py` (1 import block)
- Rename: `tests/test_execution.py` → `tests/test_push_skill.py` (1 import block)

- [ ] **Step 1: Create the new package and module**

```bash
mkdir -p babysteps/skills
```

Create `babysteps/skills/__init__.py`:

```python
"""Per-task skill compilers. One module per ManiSkill task family."""
```

Copy verbatim:

```bash
cp babysteps/execution.py babysteps/skills/push.py
```

- [ ] **Step 2: Update `babysteps/envs/pushcube_runner.py` import**

Find this line:

```python
from babysteps.execution import compile_intent_to_push_skill
```

Replace with:

```python
from babysteps.skills.push import compile_intent_to_push_skill
```

- [ ] **Step 3: Update `tests/conftest.py` import**

Find:

```python
from babysteps.execution import compile_intent_to_push_skill  # noqa: E402
```

Replace with:

```python
from babysteps.skills.push import compile_intent_to_push_skill  # noqa: E402
```

- [ ] **Step 4: Update `scripts/render_stage0_maniskill.py` imports**

Find:

```python
from babysteps.execution import (
    CUBE_HALF_SIZE, PRE_CONTACT_STANDOFF,
    PUSH_TRAVEL_SCALE, PUSH_TRAVEL_MAX_M,
)
```

Replace with:

```python
from babysteps.skills.push import (
    CUBE_HALF_SIZE, PRE_CONTACT_STANDOFF,
    PUSH_TRAVEL_SCALE, PUSH_TRAVEL_MAX_M,
)
```

- [ ] **Step 5: Rename and update `tests/test_execution.py`**

```bash
git mv tests/test_execution.py tests/test_push_skill.py
```
(If git isn't initialized, plain `mv`.)

In `tests/test_push_skill.py`, find:

```python
from babysteps.execution import (
    CUBE_HALF_SIZE,
    PRE_CONTACT_STANDOFF,
    PUSH_TRAVEL_MAX_M,
    PUSH_TRAVEL_SCALE,
    PushSkill,
    build_push_waypoints,
    compile_intent_to_push_skill,
)
```

Replace with:

```python
from babysteps.skills.push import (
    CUBE_HALF_SIZE,
    PRE_CONTACT_STANDOFF,
    PUSH_TRAVEL_MAX_M,
    PUSH_TRAVEL_SCALE,
    PushSkill,
    build_push_waypoints,
    compile_intent_to_push_skill,
)
```

Also update the module docstring (top of file) — change `babysteps.execution` to `babysteps.skills.push` if it appears.

- [ ] **Step 6: Repo-wide grep for stragglers**

```bash
grep -rn "babysteps.execution\|babysteps\.execution\|from babysteps import execution" \
    babysteps/ scripts/ tests/ docs/ 2>/dev/null
```

Expected: zero matches. If any appear (e.g., `scripts/render_stage0_topdown.py`), update them analogously.

- [ ] **Step 7: Delete the old module**

```bash
rm babysteps/execution.py
```

- [ ] **Step 8: Run the full suite**

```bash
python -m pytest tests/ -q
```

Expected: same pass count as Task 2 step 5 (no behaviour change, just import paths).

- [ ] **Step 9: Smoke-test the fake-env CLI**

```bash
rm -rf /tmp/babysteps_taskN3
python scripts/stage0_collect.py \
    --out_dir /tmp/babysteps_taskN3 \
    --n_episodes 5 --seed_start 0 --fake-env
diff tests/snapshots/pushcube_samples_seeds_0_4.jsonl \
     /tmp/babysteps_taskN3/samples.jsonl
```

Expected: empty diff. (No byte changes from a pure rename.)

- [ ] **Step 10: Commit**

```bash
git add babysteps/skills/ babysteps/envs/pushcube_runner.py \
        tests/conftest.py tests/test_push_skill.py \
        scripts/render_stage0_maniskill.py
git rm babysteps/execution.py tests/test_execution.py
git commit -m "refactor: move execution.py to babysteps/skills/push.py"
```

---

### Task 4: Create `BaseTaskAdapter` ABC and `EnvRunner` Protocol

The interface every adapter implements. PushCubeAdapter (Task 5) will be the first concrete implementation.

**Files:**
- Create: `babysteps/envs/task_adapter.py`
- Create: `tests/test_task_adapter.py`

- [ ] **Step 1: Write the failing test file `tests/test_task_adapter.py`**

```python
"""Tests for babysteps.envs.task_adapter — the BaseTaskAdapter ABC and the
EnvRunner Protocol that every concrete adapter implements."""
from __future__ import annotations

import pytest

from babysteps.envs.task_adapter import BaseTaskAdapter, EnvRunner
from babysteps.failure import Attribution
from babysteps.schemas import (
    AttemptResult, DemoEvidence, FailurePacket, Intent, Revision, SceneState,
)


# ---------- BaseTaskAdapter is abstract --------------------------------- #


def test_base_adapter_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseTaskAdapter()  # type: ignore[abstract]


def test_partial_subclass_cannot_be_instantiated():
    class HalfAdapter(BaseTaskAdapter):
        task_id = "TestTask-v0"
        def make_env_runner(self):  # noqa: D401
            raise NotImplementedError
        # Missing: oracle_correct_intent, default_blocked_factory,
        # oracle_wrong_factor, scripted_demo_to_intent, compile_skill.
    with pytest.raises(TypeError):
        HalfAdapter()  # type: ignore[abstract]


# ---------- env_runner caching + close ---------------------------------- #


class _CountingAdapter(BaseTaskAdapter):
    """Subclass that tracks how many times make_env_runner is invoked."""
    task_id = "CountTask-v0"

    def __init__(self):
        super().__init__()
        self.make_calls = 0

    def make_env_runner(self):
        self.make_calls += 1
        class _Runner:
            def reset(self, seed): return _ok_scene()
            def run(self, intent, scene): return _ok_attempt()
            def close(self): pass
        return _Runner()

    def oracle_correct_intent(self, scene): return _ok_intent()
    def default_blocked_factory(self, intent): return ()
    def oracle_wrong_factor(self, intent, scene): return "none"
    def scripted_demo_to_intent(self, evidence): return _ok_intent()
    def compile_skill(self, intent, scene): return None


def test_env_runner_caches_after_first_call():
    a = _CountingAdapter()
    assert a.make_calls == 0
    r1 = a.env_runner()
    r2 = a.env_runner()
    r3 = a.env_runner()
    assert a.make_calls == 1
    assert r1 is r2 is r3


def test_close_idempotent_and_releases_runner():
    a = _CountingAdapter()
    a.env_runner()       # construct
    assert a._env_runner is not None
    a.close()
    assert a._env_runner is None
    a.close()            # second call is a no-op
    assert a._env_runner is None


def test_close_then_env_runner_reallocates():
    a = _CountingAdapter()
    a.env_runner()
    a.close()
    a.env_runner()
    assert a.make_calls == 2


# ---------- Concrete stub used to exercise the hook defaults ------------ #


class _StubAdapter(BaseTaskAdapter):
    task_id = "StubTask-v0"

    def make_env_runner(self):
        raise NotImplementedError

    def oracle_correct_intent(self, scene):
        return _ok_intent()

    def default_blocked_factory(self, intent):
        return ()

    def oracle_wrong_factor(self, intent, scene):
        return "none"

    def scripted_demo_to_intent(self, evidence):
        return _ok_intent()

    def compile_skill(self, intent, scene):
        return None


def _ok_intent() -> Intent:
    return Intent(
        goal_state="cube_at_target",
        object_motion="translate_+x",
        contact_region="minus_x_face",
        approach_direction="from_minus_x",
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_push",
    )


def _ok_scene() -> SceneState:
    return SceneState(
        cube_xy=(0.0, 0.0), cube_z=0.02, goal_xy=(0.2, 0.0),
        tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0),
        blocked_sides=("from_minus_x",),
    )


def _ok_attempt(success: bool = False) -> AttemptResult:
    return AttemptResult(
        initial_obj_xy=(0.0, 0.0), final_obj_xy=(0.0, 0.0), goal_xy=(0.2, 0.0),
        reached_contact=False, object_moved=False,
        planner_failed=True, collision=False, grasp_slip=False,
        rollout_log_path=None, success=success,
    )


# ---------- Hook defaults delegate to shared modules -------------------- #


def test_build_failure_packet_default_delegates(monkeypatch):
    """The default hook must call babysteps.failure.build_failure_packet."""
    import babysteps.failure as failure_mod
    calls = []
    original = failure_mod.build_failure_packet

    def spy(intent, attempt, scene):
        calls.append((intent, attempt, scene))
        return original(intent, attempt, scene)

    monkeypatch.setattr(failure_mod, "build_failure_packet", spy)

    fp = _StubAdapter().build_failure_packet(_ok_intent(), _ok_attempt(), _ok_scene())
    assert isinstance(fp, FailurePacket)
    assert len(calls) == 1
    assert calls[0][0] == _ok_intent()


def test_attribute_failure_default_delegates(monkeypatch):
    import babysteps.failure as failure_mod
    calls = []
    original = failure_mod.attribute_failure

    def spy(fp):
        calls.append(fp)
        return original(fp)

    monkeypatch.setattr(failure_mod, "attribute_failure", spy)

    fp = _StubAdapter().build_failure_packet(_ok_intent(), _ok_attempt(), _ok_scene())
    attr = _StubAdapter().attribute_failure(fp)
    assert isinstance(attr, Attribution)
    assert len(calls) == 1


def test_revise_intent_default_delegates(monkeypatch):
    import babysteps.revision as revision_mod
    calls = []
    original = revision_mod.revise_intent

    def spy(intent, attribution, scene):
        calls.append((intent, attribution, scene))
        return original(intent, attribution, scene)

    monkeypatch.setattr(revision_mod, "revise_intent", spy)

    fp = _StubAdapter().build_failure_packet(_ok_intent(), _ok_attempt(), _ok_scene())
    attr = _StubAdapter().attribute_failure(fp)
    revised, rev = _StubAdapter().revise_intent(_ok_intent(), attr, _ok_scene())
    assert isinstance(revised, Intent)
    assert isinstance(rev, Revision)
    assert len(calls) == 1


# ---------- EnvRunner Protocol -------------------------------------------- #


def test_env_runner_protocol_has_required_methods():
    """Structural typing — anything with reset/run/close satisfies it."""
    class _StubRunner:
        def reset(self, seed): return _ok_scene()
        def run(self, intent, scene): return _ok_attempt()
        def close(self): pass

    runner: EnvRunner = _StubRunner()   # type-check only; no runtime check
    assert runner.reset(0) == _ok_scene()
```

- [ ] **Step 2: Run the test, confirm it fails (module doesn't exist)**

```bash
python -m pytest tests/test_task_adapter.py -v
```

Expected: import error / `ModuleNotFoundError: babysteps.envs.task_adapter`.

- [ ] **Step 3: Create `babysteps/envs/task_adapter.py`**

```python
"""TaskAdapter interface — the boundary that the episode loop and CLI scripts
sit behind, so they don't depend on any concrete task.

Sub-projects B (PickCube) and C (StackCube) each add one concrete adapter
without touching the loop or the orchestration code.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol

from babysteps.failure import Attribution
from babysteps.schemas import (
    AttemptResult,
    DemoEvidence,
    FailurePacket,
    Intent,
    Revision,
    SceneState,
)


class EnvRunner(Protocol):
    """Minimal env_runner contract. Implementations: the fake in
    tests/conftest.py and the real ManiSkill PushCubeEnvRunner. Future tasks
    add their own runners; the adapter constructs them in make_env_runner.

    Relocated from babysteps.episode — its natural home is alongside the
    adapter that consumes it. Purely a logical-grouping move; no circular
    import is created or removed."""

    def reset(self, seed: int) -> SceneState: ...
    def run(self, intent: Intent, scene: SceneState) -> AttemptResult: ...
    def close(self) -> None: ...


class BaseTaskAdapter(ABC):
    """One concrete adapter per ManiSkill task. `episode.run_episode` and the
    CLI scripts depend ONLY on this base class.

    Six abstract methods are task-specific (must be implemented). Three
    overridable hooks (`build_failure_packet`, `attribute_failure`,
    `revise_intent`) default to the shared modules; override only when the
    task's failure precedence or revision operators differ.

    Two concrete lifecycle helpers (`env_runner`, `close`) cache the env
    runner so the per-episode allocation cost stays at one — the same as
    pre-A — even though the episode loop calls `env_runner()` every
    iteration."""

    task_id: str  # subclasses set, e.g. "PushCube-v1"

    # ---- env_runner lifecycle (concrete; cached) ------------------------- #

    def __init__(self) -> None:
        self._env_runner: EnvRunner | None = None

    def env_runner(self) -> EnvRunner:
        """Lazily construct (via make_env_runner) and cache the runner. The
        episode loop calls this once per episode; caching preserves the pre-A
        one-runner-across-all-seeds perf for expensive real-sim runners."""
        if self._env_runner is None:
            self._env_runner = self.make_env_runner()
        return self._env_runner

    def close(self) -> None:
        """Release the cached runner if any. Idempotent."""
        if self._env_runner is not None:
            self._env_runner.close()
            self._env_runner = None

    # ---- abstract: must be implemented per task --------------------------- #

    @abstractmethod
    def make_env_runner(self) -> EnvRunner:
        """Construct the concrete env_runner for this task. Called at most
        once per adapter instance — env_runner() caches the result."""

    @abstractmethod
    def oracle_correct_intent(self, scene: SceneState) -> Intent:
        """Read privileged scene state and produce the ground-truth intent
        that would succeed on this scene. Used inside generate_proxy_demo —
        NOT in the privileged-firewalled `scripted_demo_to_intent` path."""

    @abstractmethod
    def default_blocked_factory(self, intent: Intent) -> tuple[str, ...]:
        """Given the initial intent, return the privileged blocked_sides
        tuple that produces a Stage-0 controlled semantic failure."""

    @abstractmethod
    def oracle_wrong_factor(
        self, initial_intent: Intent, scene_executor: SceneState,
    ) -> str:
        """Return the name of the intent factor that is wrong under the
        executor's blocked_sides. 'none' if no factor is wrong."""

    @abstractmethod
    def scripted_demo_to_intent(self, evidence: DemoEvidence) -> Intent:
        """Privileged-firewalled scripted intent extraction. Takes ONLY a
        DemoEvidence — never a SceneState. Future stages replace this with
        DINO/VLM grounding."""

    @abstractmethod
    def compile_skill(self, intent: Intent, scene: SceneState) -> Any:
        """Compile the intent + scene into an executable skill object. The
        env_runner consumes whatever this returns. Returns None when the
        intent is infeasible (e.g., approach_direction in blocked_sides) —
        None propagates as planner_failed=True downstream."""

    # ---- overridable hooks: default delegates to shared modules ----------- #

    def build_failure_packet(
        self, intent: Intent, attempt: AttemptResult, scene: SceneState,
    ) -> FailurePacket:
        from babysteps import failure as failure_mod
        return failure_mod.build_failure_packet(intent, attempt, scene)

    def attribute_failure(self, fp: FailurePacket) -> Attribution:
        from babysteps import failure as failure_mod
        return failure_mod.attribute_failure(fp)

    def revise_intent(
        self, intent: Intent, attribution: Attribution, scene: SceneState,
    ) -> tuple[Intent, Revision]:
        from babysteps import revision as revision_mod
        return revision_mod.revise_intent(intent, attribution, scene)
```

- [ ] **Step 4: Run the test, confirm it passes**

```bash
python -m pytest tests/test_task_adapter.py -v
```

Expected: all task_adapter tests pass (7 abstract/Protocol tests + 3 caching tests = 10).

- [ ] **Step 5: Run the full suite — no regressions**

```bash
python -m pytest tests/ -q
```

Expected: all tests pass (Task 3 count + 10 new task_adapter tests).

- [ ] **Step 6: Commit**

```bash
git add babysteps/envs/task_adapter.py tests/test_task_adapter.py
git commit -m "feat(adapter): BaseTaskAdapter ABC + EnvRunner Protocol"
```

---

### Task 5: Implement `PushCubeAdapter`

Concrete adapter that pulls together everything PushCube-specific. Behaviour mirrors the current `episode._oracle_correct_intent_for_scene`, `episode._default_blocked_sides_factory`, `demo.demo_to_intent`, and `skills.push.compile_intent_to_push_skill`.

**Files:**
- Create: `babysteps/envs/pushcube_adapter.py`
- Create: `tests/test_pushcube_adapter.py`

- [ ] **Step 1: Write the parity test file `tests/test_pushcube_adapter.py`**

```python
"""Tests for babysteps.envs.pushcube_adapter — the first concrete adapter.

Proves byte-equivalent behaviour with the pre-A episode/demo/execution
helpers. The snapshot test is added in Task 9 once the episode refactor
is wired through; the per-method parity tests below land in Task 5."""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from babysteps.envs.pushcube_adapter import PushCubeAdapter
from babysteps.envs.task_adapter import BaseTaskAdapter
from babysteps.schemas import DemoEvidence, Intent, SceneState
from babysteps.skills.push import PushSkill


# ---------- Class-level checks ----------------------------------------- #


def test_pushcube_adapter_subclass_of_base():
    assert issubclass(PushCubeAdapter, BaseTaskAdapter)


def test_pushcube_adapter_task_id():
    assert PushCubeAdapter.task_id == "PushCube-v1"
    assert PushCubeAdapter().task_id == "PushCube-v1"


# ---------- oracle_correct_intent parity ------------------------------- #


def _scene_with_goal(goal_xy: tuple[float, float]) -> SceneState:
    return SceneState(
        cube_xy=(0.0, 0.0),
        cube_z=0.02,
        goal_xy=goal_xy,
        tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0),
        blocked_sides=(),
    )


@pytest.mark.parametrize("goal_xy,expected_face,expected_approach,expected_motion", [
    ((0.15, 0.0),  "minus_x_face", "from_minus_x", "translate_+x"),
    ((-0.15, 0.0), "plus_x_face",  "from_plus_x",  "translate_-x"),
    ((0.0, 0.15),  "minus_y_face", "from_minus_y", "translate_+y"),
    ((0.0, -0.15), "plus_y_face",  "from_plus_y",  "translate_-y"),
])
def test_oracle_correct_intent_per_cardinal(
    goal_xy, expected_face, expected_approach, expected_motion,
):
    scene = _scene_with_goal(goal_xy)
    intent = PushCubeAdapter().oracle_correct_intent(scene)
    assert intent.contact_region == expected_face
    assert intent.approach_direction == expected_approach
    assert intent.object_motion == expected_motion
    assert intent.goal_state == "cube_at_target"
    assert intent.constraint_region == "none"
    assert intent.embodiment_mapping == "proxy_contact_to_franka_push"


# ---------- default_blocked_factory parity ----------------------------- #


@pytest.mark.parametrize("approach", [
    "from_minus_x", "from_plus_x", "from_minus_y", "from_plus_y",
])
def test_default_blocked_factory_blocks_the_intent_approach(approach):
    intent = Intent(
        goal_state="cube_at_target",
        object_motion="translate_+x",
        contact_region="minus_x_face",
        approach_direction=approach,
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_push",
    )
    assert PushCubeAdapter().default_blocked_factory(intent) == (approach,)


# ---------- oracle_wrong_factor parity --------------------------------- #


def test_oracle_wrong_factor_when_intent_approach_in_blocked():
    scene = SceneState(
        cube_xy=(0.0, 0.0), cube_z=0.02, goal_xy=(0.15, 0.0),
        tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0),
        blocked_sides=("from_minus_x",),
    )
    intent = PushCubeAdapter().oracle_correct_intent(_scene_with_goal((0.15, 0.0)))
    assert PushCubeAdapter().oracle_wrong_factor(intent, scene) == "approach_direction"


def test_oracle_wrong_factor_when_intent_approach_unblocked():
    scene = SceneState(
        cube_xy=(0.0, 0.0), cube_z=0.02, goal_xy=(0.15, 0.0),
        tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0),
        blocked_sides=("from_plus_y",),  # unrelated to intent's approach
    )
    intent = PushCubeAdapter().oracle_correct_intent(_scene_with_goal((0.15, 0.0)))
    assert PushCubeAdapter().oracle_wrong_factor(intent, scene) == "none"


# ---------- scripted_demo_to_intent parity ----------------------------- #


def _evidence(traj, face) -> DemoEvidence:
    return DemoEvidence(
        camera="third_person",
        demonstrator_type="proxy_oracle",
        object_trajectory=tuple(tuple(p) for p in traj),
        contact_region_label=face,
        final_state="cube_at_target",
        rgbd_video_path=None,
    )


def test_scripted_demo_to_intent_signature_takes_only_demo_evidence():
    """Privileged-firewall regression guard, moved from test_demo.py."""
    sig = inspect.signature(PushCubeAdapter.scripted_demo_to_intent)
    params = list(sig.parameters.values())
    # self + evidence = 2.
    assert len(params) == 2, (
        f"scripted_demo_to_intent must take ONLY (self, DemoEvidence); "
        f"got params: {[p.name for p in params]}"
    )
    ev_param = params[1]
    annot = ev_param.annotation
    annot_name = annot if isinstance(annot, str) else getattr(annot, "__name__", str(annot))
    assert annot is DemoEvidence or annot_name == "DemoEvidence", (
        f"second parameter must be annotated DemoEvidence, got {annot!r}"
    )


def test_scripted_demo_to_intent_plus_x():
    ev = _evidence([(0.0, 0.0), (0.10, 0.0)], "minus_x_face")
    intent = PushCubeAdapter().scripted_demo_to_intent(ev)
    assert intent.object_motion == "translate_+x"
    assert intent.contact_region == "minus_x_face"
    assert intent.approach_direction == "from_minus_x"
    assert intent.goal_state == "cube_at_target"
    assert intent.constraint_region == "none"
    assert intent.embodiment_mapping == "proxy_contact_to_franka_push"


def test_scripted_demo_to_intent_minus_y():
    ev = _evidence([(0.0, 0.0), (0.0, -0.1)], "plus_y_face")
    intent = PushCubeAdapter().scripted_demo_to_intent(ev)
    assert intent.object_motion == "translate_-y"
    assert intent.contact_region == "plus_y_face"
    assert intent.approach_direction == "from_plus_y"


def test_scripted_demo_to_intent_rejects_unknown_contact_region():
    ev = _evidence([(0.0, 0.0), (0.1, 0.0)], "not_a_face")
    with pytest.raises(ValueError, match="contact_region"):
        PushCubeAdapter().scripted_demo_to_intent(ev)


# ---------- compile_skill parity --------------------------------------- #


def _correct_push_intent() -> Intent:
    return Intent(
        goal_state="cube_at_target",
        object_motion="translate_+x",
        contact_region="minus_x_face",
        approach_direction="from_minus_x",
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_push",
    )


def test_compile_skill_unblocked_returns_pushskill():
    intent = _correct_push_intent()
    scene = SceneState(
        cube_xy=(0.0, 0.0), cube_z=0.02, goal_xy=(0.2, 0.0),
        tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0),
        blocked_sides=(),
    )
    skill = PushCubeAdapter().compile_skill(intent, scene)
    assert isinstance(skill, PushSkill)
    assert skill.contact_region == "minus_x_face"


def test_compile_skill_blocked_returns_none():
    intent = _correct_push_intent()
    scene = SceneState(
        cube_xy=(0.0, 0.0), cube_z=0.02, goal_xy=(0.2, 0.0),
        tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0),
        blocked_sides=("from_minus_x",),
    )
    assert PushCubeAdapter().compile_skill(intent, scene) is None


# ---------- Hook inheritance check ------------------------------------- #


def test_hooks_inherited_from_base():
    """PushCubeAdapter does not override the three hooks."""
    assert PushCubeAdapter.build_failure_packet is BaseTaskAdapter.build_failure_packet
    assert PushCubeAdapter.attribute_failure is BaseTaskAdapter.attribute_failure
    assert PushCubeAdapter.revise_intent is BaseTaskAdapter.revise_intent
```

- [ ] **Step 2: Run, confirm it fails (PushCubeAdapter doesn't exist)**

```bash
python -m pytest tests/test_pushcube_adapter.py -v
```

Expected: `ModuleNotFoundError: babysteps.envs.pushcube_adapter`.

- [ ] **Step 3: Create `babysteps/envs/pushcube_adapter.py`**

```python
"""PushCube-v1 adapter — the first concrete BaseTaskAdapter.

Pulls every PushCube-specific decision behind one class:
  * make_env_runner       → PushCubeEnvRunner
  * oracle_correct_intent → uses goal direction to derive face/approach/motion
  * default_blocked_factory → (intent.approach_direction,)
  * oracle_wrong_factor   → "approach_direction" if intent's approach is in
                            blocked_sides, else "none"
  * scripted_demo_to_intent → contact_region_label + trajectory → Intent
  * compile_skill         → wraps skills.push.compile_intent_to_push_skill

Hook defaults (build_failure_packet / attribute_failure / revise_intent) are
inherited unchanged from BaseTaskAdapter."""
from __future__ import annotations

import numpy as np

from babysteps.demo import trajectory_to_motion
from babysteps.envs.scene import direction_to_face, face_to_approach
from babysteps.envs.task_adapter import BaseTaskAdapter, EnvRunner
from babysteps.schemas import CONTACT_REGIONS, DemoEvidence, Intent, SceneState
from babysteps.skills.push import compile_intent_to_push_skill


class PushCubeAdapter(BaseTaskAdapter):
    task_id = "PushCube-v1"

    def make_env_runner(self) -> EnvRunner:
        from babysteps.envs.pushcube_runner import PushCubeEnvRunner
        return PushCubeEnvRunner()

    def oracle_correct_intent(self, scene: SceneState) -> Intent:
        goal_vec = np.array(scene.goal_xy) - np.array(scene.cube_xy)
        face = direction_to_face(goal_vec)
        approach = face_to_approach(face)
        if abs(goal_vec[0]) >= abs(goal_vec[1]):
            motion = "translate_+x" if goal_vec[0] >= 0 else "translate_-x"
        else:
            motion = "translate_+y" if goal_vec[1] >= 0 else "translate_-y"
        return Intent(
            goal_state="cube_at_target",
            object_motion=motion,
            contact_region=face,
            approach_direction=approach,
            constraint_region="none",
            embodiment_mapping="proxy_contact_to_franka_push",
        )

    def default_blocked_factory(self, intent: Intent) -> tuple[str, ...]:
        return (intent.approach_direction,)

    def oracle_wrong_factor(
        self, initial_intent: Intent, scene_executor: SceneState,
    ) -> str:
        if initial_intent.approach_direction in scene_executor.blocked_sides:
            return "approach_direction"
        return "none"

    def scripted_demo_to_intent(self, evidence: DemoEvidence) -> Intent:
        contact = evidence.contact_region_label
        if contact not in CONTACT_REGIONS:
            raise ValueError(
                f"DemoEvidence.contact_region_label must be one of "
                f"{sorted(CONTACT_REGIONS)}, got {contact!r}"
            )
        motion = trajectory_to_motion(evidence.object_trajectory)
        return Intent(
            goal_state="cube_at_target",
            object_motion=motion,
            contact_region=contact,
            approach_direction=face_to_approach(contact),
            constraint_region="none",
            embodiment_mapping="proxy_contact_to_franka_push",
        )

    def compile_skill(self, intent: Intent, scene: SceneState):
        return compile_intent_to_push_skill(intent, scene)
```

- [ ] **Step 4: Run, confirm passes**

```bash
python -m pytest tests/test_pushcube_adapter.py -v
```

Expected: all parity tests pass (~14 tests).

- [ ] **Step 5: Run the full suite — no regressions**

```bash
python -m pytest tests/ -q
```

Expected: all tests pass (Task 4 count + ~14 new pushcube_adapter tests).

- [ ] **Step 6: Commit**

```bash
git add babysteps/envs/pushcube_adapter.py tests/test_pushcube_adapter.py
git commit -m "feat(adapter): PushCubeAdapter with parity tests"
```

---

### Task 6: Refactor `episode.run_episode` to use the adapter

Switch the loop and `generate_proxy_demo` to call adapter methods. Delete the now-dead helpers from `episode.py`. Update `test_episode.py`.

**Files:**
- Modify: `babysteps/episode.py` (most of the file)
- Modify: `tests/test_episode.py` (5 call sites + 1 spy rewrite)

- [ ] **Step 1: Update `tests/test_episode.py` first (these tests will define the new signature)**

Replace the file's contents with:

```python
"""Integration test for the Stage-0 episode loop with a fake env_runner.

After Sub-project A, run_episode takes an adapter:. This file uses a stub
adapter wired around the deterministic FakeEnvRunner (conftest fixture)."""
from __future__ import annotations

import inspect

import pytest

from babysteps.envs.task_adapter import BaseTaskAdapter
from babysteps.episode import run_episode
from babysteps.schemas import (
    CLAIM_BOUNDARY, DemoEvidence, EpisodeRecord, Intent, SceneState,
)


# Stub adapter that uses the conftest FakeEnvRunner via injection. We avoid
# importing PushCubeAdapter here so this test stays decoupled from any one
# concrete adapter.
def _make_stub_adapter(fake_runner, *, blocked_factory=None):
    from babysteps.envs.pushcube_adapter import PushCubeAdapter

    class _StubAdapter(PushCubeAdapter):
        def make_env_runner(self):
            return fake_runner
        if blocked_factory is not None:
            def default_blocked_factory(self, intent):
                return blocked_factory(intent)
    return _StubAdapter()


def test_run_episode_blocked_then_retry_success(fake_env_runner):
    adapter = _make_stub_adapter(fake_env_runner)
    rec = run_episode(
        episode_id="pushcube_blocked_approach_seed_0000",
        seed=0,
        adapter=adapter,
    )
    assert isinstance(rec, EpisodeRecord)
    assert rec.task == "PushCube-v1"
    assert rec.claim_boundary == CLAIM_BOUNDARY
    assert rec.demo["demonstrator_type"] == "proxy_oracle"
    assert rec.execution["success"] is False
    assert rec.failure_packet["failure_predicate"] == "approach_blocked"
    assert rec.failure_packet["wrong_factor"] == "approach_direction"
    assert rec.revision is not None
    assert rec.revision["operator"] == "approach_substitution"
    assert rec.revision["factor"] == "approach_direction"
    assert rec.retry is not None
    assert rec.retry["success"] is True
    m = rec.metrics
    assert m["initial_success"] is False
    assert m["retry_success"] is True
    assert m["num_attempts_to_success"] == 2
    assert m["factor_attribution_correct"] is True
    assert m["frozen_factors_preserved"] is True
    assert m["factors_changed"] == ["approach_direction"]


def test_run_episode_top_level_keys_match_goal_md(fake_env_runner):
    rec = run_episode(
        episode_id="pushcube_blocked_approach_seed_0000",
        seed=0,
        adapter=_make_stub_adapter(fake_env_runner),
    )
    d = rec.to_dict()
    expected = {
        "episode_id", "stage", "task", "claim_boundary",
        "demo", "execution", "failure_packet", "revision", "retry", "metrics",
    }
    assert set(d.keys()) == expected
    assert "goal_xy" not in rec.demo
    assert "blocked_sides" not in rec.demo


def test_run_episode_jsonl_roundtrip(fake_env_runner):
    rec = run_episode(
        episode_id="pushcube_blocked_approach_seed_0000",
        seed=0,
        adapter=_make_stub_adapter(fake_env_runner),
    )
    line = rec.to_jsonl_line()
    rt = EpisodeRecord.from_jsonl_line(line)
    assert rt.episode_id == rec.episode_id
    assert rt.failure_packet["failure_predicate"] == "approach_blocked"


def test_run_episode_scripted_demo_to_intent_called_with_only_demo_evidence(
    fake_env_runner,
):
    """Privileged-firewall enforcement: adapter.scripted_demo_to_intent
    must be called with a DemoEvidence and nothing else."""
    from babysteps.envs.pushcube_adapter import PushCubeAdapter

    call_args = []

    class _SpyAdapter(PushCubeAdapter):
        def make_env_runner(self):
            return fake_env_runner
        def scripted_demo_to_intent(self, evidence):
            call_args.append(evidence)
            return super().scripted_demo_to_intent(evidence)

    run_episode(
        episode_id="pushcube_blocked_approach_seed_0000",
        seed=0,
        adapter=_SpyAdapter(),
    )
    assert len(call_args) == 1
    assert isinstance(call_args[0], DemoEvidence)


def test_run_episode_already_succeeds_no_revision(fake_env_runner):
    """If blocked_sides is empty, the initial intent succeeds and no
    revision/retry is recorded."""
    rec = run_episode(
        episode_id="pushcube_unblocked_seed_0000",
        seed=0,
        adapter=_make_stub_adapter(
            fake_env_runner,
            blocked_factory=lambda intent: (),   # never blocks
        ),
    )
    assert rec.execution["success"] is True
    assert rec.failure_packet["failure_predicate"] == "none"
    assert rec.revision is None
    assert rec.retry is None
    assert rec.metrics["initial_success"] is True
    assert rec.metrics["num_attempts_to_success"] == 1


def test_run_episode_multiple_seeds_all_succeed(fake_env_runner):
    adapter = _make_stub_adapter(fake_env_runner)
    for seed in range(4):
        rec = run_episode(
            episode_id=f"pushcube_blocked_approach_seed_{seed:04d}",
            seed=seed,
            adapter=adapter,
        )
        assert rec.execution["success"] is False
        assert rec.retry["success"] is True, (
            f"seed {seed} did not recover; revised approach was "
            f"{rec.revision['new_value']!r}"
        )


def test_run_episode_signature_takes_adapter_keyword():
    """Guard against accidentally restoring the old env_runner= kwarg."""
    sig = inspect.signature(run_episode)
    assert "adapter" in sig.parameters
    assert "env_runner" not in sig.parameters
    assert "blocked_sides_factory" not in sig.parameters
```

- [ ] **Step 2: Run the updated test, confirm it fails**

```bash
python -m pytest tests/test_episode.py -v
```

Expected: failures — `run_episode` does not accept `adapter=` yet.

- [ ] **Step 3: Rewrite `babysteps/episode.py`**

Replace the file's contents with:

```python
"""Stage-0 episode loop: orchestrates demo → intent → execute → fail → revise → retry.

Pure orchestration: no simulator import, no I/O, no PushCube assumptions. The
adapter is injected; every task-specific decision (skill compilation, scripted
demo→intent, blocked-sides factory, oracle wrong-factor labelling, failure
attribution, intent revision) is dispatched through the adapter.

A single `run_episode(...)` call produces one `EpisodeRecord` matching the
shape mandated by `goal.md` §"Episode Data Format" — see `test_episode.py`
for the snapshot guard.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Optional

from babysteps.envs.task_adapter import BaseTaskAdapter, EnvRunner
from babysteps.schemas import (
    CLAIM_BOUNDARY,
    INTENT_FIELDS,
    AttemptResult,
    DemoEvidence,
    EpisodeRecord,
    Intent,
    SceneState,
)


# ---------- demo proxy generation -------------------------------------- #


def generate_proxy_demo(
    env_runner: EnvRunner, scene: SceneState, adapter: BaseTaskAdapter,
) -> DemoEvidence:
    """Run the adapter's oracle scripted skill on `scene` (with blocked_sides=())
    and pack the result into DemoEvidence. The DemoEvidence carries only
    demo-visible quantities — no goal_xy, no blocked_sides."""
    correct = adapter.oracle_correct_intent(scene)
    unblocked = replace(scene, blocked_sides=())
    demo_attempt = env_runner.run(correct, unblocked)
    traj = demo_attempt.trajectory_xy
    if not traj:
        traj = (demo_attempt.initial_obj_xy, demo_attempt.final_obj_xy)
    return DemoEvidence(
        camera="third_person",
        demonstrator_type="proxy_oracle",
        object_trajectory=traj,
        contact_region_label=correct.contact_region,
        final_state="cube_at_target",
        rgbd_video_path=None,
    )


# ---------- per-episode metrics ---------------------------------------- #


def _compute_metrics(
    *,
    initial_success: bool,
    retry_success: Optional[bool],
    failure_predicate: str,
    wrong_factor_predicted: Optional[str],
    oracle_wrong_factor: str,
    factors_changed: tuple[str, ...],
) -> dict:
    """Per-episode metrics for the eval/summarize step."""
    if not initial_success and retry_success is True:
        num_attempts = 2
    elif initial_success:
        num_attempts = 1
    else:
        num_attempts = 2

    attribution_correct: Optional[bool] = (
        None if wrong_factor_predicted is None
        else wrong_factor_predicted == oracle_wrong_factor
    )

    frozen_preserved: Optional[bool]
    if wrong_factor_predicted is None:
        frozen_preserved = None
    else:
        frozen_preserved = (
            tuple(factors_changed) == (wrong_factor_predicted,)
            or len(factors_changed) == 0
        )

    return {
        "initial_success":           bool(initial_success),
        "retry_success":             retry_success,
        "num_attempts_to_success":   int(num_attempts),
        "failure_type":              failure_predicate,
        "wrong_factor_predicted":    wrong_factor_predicted,
        "oracle_wrong_factor":       oracle_wrong_factor,
        "factor_attribution_correct": attribution_correct,
        "factors_changed":           list(factors_changed),
        "frozen_factors_preserved":  frozen_preserved,
    }


def _diff_intents(a: Intent, b: Intent) -> tuple[str, ...]:
    return tuple(f for f in INTENT_FIELDS if getattr(a, f) != getattr(b, f))


# ---------- the loop --------------------------------------------------- #


def run_episode(
    *,
    episode_id: str,
    seed: int,
    adapter: BaseTaskAdapter,
) -> EpisodeRecord:
    """One Stage-0 blocked-approach episode for the adapter's task.

    Steps:
      1. Construct env_runner via adapter. Reset → SceneState (blocked_sides=()).
      2. Generate proxy demo via adapter's oracle.
      3. Derive initial intent via adapter.scripted_demo_to_intent.
      4. Build executor scene with adapter.default_blocked_factory(initial_intent).
      5. Attempt 1: env_runner.run(initial_intent, scene_executor).
      6. Build failure packet via adapter, attribute, revise.
      7. Attempt 2: env_runner.run(revised_intent, scene_executor).
      8. Pack EpisodeRecord with metrics. task = adapter.task_id.
    """
    env_runner = adapter.env_runner()      # cached on the adapter
    scene_initial = env_runner.reset(seed)
    demo_evidence = generate_proxy_demo(env_runner, scene_initial, adapter)
    initial_intent = adapter.scripted_demo_to_intent(demo_evidence)
    scene_executor = replace(
        scene_initial,
        blocked_sides=adapter.default_blocked_factory(initial_intent),
    )
    oracle_wrong_factor = adapter.oracle_wrong_factor(
        initial_intent, scene_executor,
    )
    attempt_1 = env_runner.run(initial_intent, scene_executor)
    failure_packet = adapter.build_failure_packet(
        initial_intent, attempt_1, scene_executor,
    )

        demo_dict = {
            "camera": demo_evidence.camera,
            "rgbd_video": demo_evidence.rgbd_video_path,
            "object_trajectory": [list(p) for p in demo_evidence.object_trajectory],
            "contact_region_label": demo_evidence.contact_region_label,
            "final_state": demo_evidence.final_state,
            "demonstrator_type": demo_evidence.demonstrator_type,
        }
        execution_dict = {
            "camera": "robot_first_person",
            "robot": "Franka",
            "initial_intent": initial_intent.to_dict(),
            "success": bool(attempt_1.success),
        }

        if failure_packet.failure_predicate == "none":
            fp_dict = {
                "failure_predicate": "none",
                "wrong_factor": None,
                "oracle_wrong_factor": oracle_wrong_factor,
                "execution_trace": dict(failure_packet.execution_trace),
            }
            metrics = _compute_metrics(
                initial_success=True,
                retry_success=None,
                failure_predicate="none",
                wrong_factor_predicted=None,
                oracle_wrong_factor=oracle_wrong_factor,
                factors_changed=(),
            )
            return EpisodeRecord(
                episode_id=episode_id,
                stage="stage_0",
                task=adapter.task_id,
                claim_boundary=CLAIM_BOUNDARY,
                demo=demo_dict,
                execution=execution_dict,
                failure_packet=fp_dict,
                revision=None,
                retry=None,
                metrics=metrics,
            )

        attribution = adapter.attribute_failure(failure_packet)
        try:
            revised_intent, revision_record = adapter.revise_intent(
                initial_intent, attribution, scene_executor,
            )
        except NotImplementedError as exc:
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

        attempt_2 = env_runner.run(revised_intent, scene_executor)
        factors_changed = _diff_intents(initial_intent, revised_intent)

        fp_dict = {
            "failure_predicate": failure_packet.failure_predicate,
            "wrong_factor": attribution.wrong_factor,
            "oracle_wrong_factor": oracle_wrong_factor,
            "execution_trace": dict(failure_packet.execution_trace),
            "object_displacement": failure_packet.object_displacement,
            "direction_alignment": failure_packet.direction_alignment,
            "freeze": list(attribution.freeze),
            "revise": list(attribution.revise),
        }
        revision_dict = revision_record.to_dict()
        retry_dict = {
            "success": bool(attempt_2.success),
            "num_retries": 1,
            "final_intent": revised_intent.to_dict(),
        }
        metrics = _compute_metrics(
            initial_success=bool(attempt_1.success),
            retry_success=bool(attempt_2.success),
            failure_predicate=failure_packet.failure_predicate,
            wrong_factor_predicted=attribution.wrong_factor,
            oracle_wrong_factor=oracle_wrong_factor,
            factors_changed=factors_changed,
        )

        return EpisodeRecord(
            episode_id=episode_id,
            stage="stage_0",
            task=adapter.task_id,
            claim_boundary=CLAIM_BOUNDARY,
            demo=demo_dict,
            execution=execution_dict,
            failure_packet=fp_dict,
            revision=revision_dict,
            retry=retry_dict,
            metrics=metrics,
        )
```

Notes on what got removed:
- `EnvRunner` Protocol — now in `task_adapter.py`.
- `BlockedSidesFactory`, `_default_blocked_sides_factory` — moved into `PushCubeAdapter.default_blocked_factory`.
- `_oracle_correct_intent_for_scene` — moved into `PushCubeAdapter.oracle_correct_intent`.
- `from babysteps.demo import demo_to_intent`, `from babysteps.envs.scene import direction_to_face, face_to_approach` — no longer used by episode.py.
- Direct import of `failure` / `revision` modules — adapter now dispatches.
- The inline oracle-wrong-factor branch — now `adapter.oracle_wrong_factor(...)`.

`run_episode` does NOT own the env_runner lifecycle. It calls
`adapter.env_runner()` (cached) once per episode and reads the runner. The
caller is responsible for calling `adapter.close()` at the end of its
sequence of `run_episode` calls — see Task 8 for the CLI pattern. Tests
that use the FakeEnvRunner can skip `adapter.close()` (the fake's close is
a no-op).

- [ ] **Step 4: Run `tests/test_episode.py` — confirm passes**

```bash
python -m pytest tests/test_episode.py -v
```

Expected: all 7 tests pass.

- [ ] **Step 5: Run the full suite**

```bash
python -m pytest tests/ -q
```

Expected: all pass. (test_episode.py grew by 1 — the signature guard test.)

- [ ] **Step 6: Commit**

```bash
git add babysteps/episode.py tests/test_episode.py
git commit -m "refactor(episode): run_episode takes adapter, drops push helpers"
```

---

### Task 7: Move `demo_to_intent` into the adapter; trim `demo.py`

The push-specific extractor has already been re-implemented in
`PushCubeAdapter.scripted_demo_to_intent` (Task 5). Delete it from `demo.py`
and migrate the corresponding tests.

**Files:**
- Modify: `babysteps/demo.py` (delete `demo_to_intent`, keep `trajectory_to_motion`)
- Modify: `tests/test_demo.py` (delete the 4 `demo_to_intent` tests)

- [ ] **Step 1: Verify nothing else imports `demo_to_intent`**

```bash
grep -rn "from babysteps.demo import" babysteps/ scripts/ tests/ 2>/dev/null
grep -rn "babysteps\.demo\.demo_to_intent" babysteps/ scripts/ tests/ 2>/dev/null
```

Expected: the only imports of `demo_to_intent` are in `tests/test_demo.py`
(about to be removed) and possibly `scripts/render_stage0_maniskill.py`
(handled in Task 8). If any other file still imports it, fix those first.

- [ ] **Step 2: Edit `babysteps/demo.py` — delete `demo_to_intent`**

Replace the file's contents with:

```python
"""Scripted demo-evidence utilities.

After Sub-project A, per-task scripted extractors live in their respective
TaskAdapter (e.g., PushCubeAdapter.scripted_demo_to_intent). What stays here
is the small, task-agnostic helper used by those extractors.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from babysteps.envs.scene import goal_direction_to_motion


def trajectory_to_motion(traj: Iterable[tuple[float, float]]) -> str:
    """Snap a (≥2-point) xy trajectory to one of OBJECT_MOTIONS.

    Uses net displacement (final − initial) along the dominant axis.
    Raises ValueError on trajectories shorter than 2 points.
    """
    pts = list(traj)
    if len(pts) < 2:
        raise ValueError(
            f"trajectory_to_motion needs at least 2 points, got {len(pts)}"
        )
    initial = np.asarray(pts[0], dtype=np.float64)
    final = np.asarray(pts[-1], dtype=np.float64)
    return goal_direction_to_motion(final - initial)
```

- [ ] **Step 3: Edit `tests/test_demo.py` — delete the 4 `demo_to_intent` tests**

Replace the file's contents with:

```python
"""Tests for babysteps.demo — the surviving trajectory_to_motion helper.

The `demo_to_intent` tests have moved into tests/test_pushcube_adapter.py
because the extractor itself moved into PushCubeAdapter.scripted_demo_to_intent
in Sub-project A."""
from __future__ import annotations

import pytest

from babysteps.demo import trajectory_to_motion


def test_trajectory_to_motion_plus_x():
    assert trajectory_to_motion([(0.0, 0.0), (0.05, 0.0), (0.10, 0.0)]) == "translate_+x"


def test_trajectory_to_motion_minus_y():
    assert trajectory_to_motion([(0.0, 0.0), (0.0, -0.1)]) == "translate_-y"


def test_trajectory_to_motion_dominant_axis():
    """Mixed motion snaps on the dominant axis."""
    assert trajectory_to_motion([(0.0, 0.0), (0.2, 0.05)]) == "translate_+x"


def test_trajectory_to_motion_empty_raises():
    with pytest.raises(ValueError, match="at least"):
        trajectory_to_motion([])


def test_trajectory_to_motion_single_point_raises():
    with pytest.raises(ValueError, match="at least"):
        trajectory_to_motion([(0.0, 0.0)])
```

- [ ] **Step 4: Run the full suite**

```bash
python -m pytest tests/ -q
```

Expected: all pass. (5 demo_to_intent tests are gone — their replacements live in `tests/test_pushcube_adapter.py`.)

- [ ] **Step 5: Smoke-test the fake-env CLI vs. snapshot**

```bash
rm -rf /tmp/babysteps_task7
python scripts/stage0_collect.py \
    --out_dir /tmp/babysteps_task7 \
    --n_episodes 5 --seed_start 0 --fake-env
```

This will currently fail because `stage0_collect.py` still passes `env_runner=runner` to `run_episode`. That CLI gets fixed in Task 8.

You can do a partial verification by writing a tiny driver inline:

```bash
python - <<'PY'
from babysteps.envs.pushcube_adapter import PushCubeAdapter
from babysteps.episode import run_episode
adapter = PushCubeAdapter()
# Use the fake runner instead of the real one by subclassing inline:
from tests.conftest import FakeEnvRunner
fake = FakeEnvRunner()
class _StubAdapter(PushCubeAdapter):
    def make_env_runner(self):
        return fake
records = []
for seed in range(5):
    rec = run_episode(
        episode_id=f"pushcube_blocked_approach_seed_{seed:04d}",
        seed=seed,
        adapter=_StubAdapter(),
    )
    print(rec.to_jsonl_line())
PY
```

Diff this output against the snapshot fixture manually if you want — the byte-equality test in Task 9 makes this rigorous.

- [ ] **Step 6: Commit**

```bash
git add babysteps/demo.py tests/test_demo.py
git commit -m "refactor(demo): demo_to_intent moved to PushCubeAdapter, trajectory_to_motion stays"
```

---

### Task 8: Update CLI scripts

`stage0_collect.py` and `render_stage0_maniskill.py` now wire `PushCubeAdapter`.

**Files:**
- Modify: `scripts/stage0_collect.py` (~10 lines)
- Modify: `scripts/render_stage0_maniskill.py` (~30 lines — the helpers it used vanished)
- (Check) `scripts/render_stage0_topdown.py` — fix if it imports `babysteps.execution` or `babysteps.demo.demo_to_intent`

- [ ] **Step 1: Update `scripts/stage0_collect.py`**

Replace the `_make_runner` helper and the `run_episode` call site:

Find:

```python
def _make_runner(use_fake: bool):
    if use_fake:
        # Tests conftest carries the deterministic fake.
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from tests.conftest import FakeEnvRunner   # noqa: WPS433
        return FakeEnvRunner()
    from babysteps.envs.pushcube_runner import PushCubeEnvRunner  # noqa: WPS433
    return PushCubeEnvRunner()
```

Replace with:

```python
def _make_adapter(use_fake: bool):
    """Build a PushCubeAdapter wired to the right env runner."""
    from babysteps.envs.pushcube_adapter import PushCubeAdapter  # noqa: WPS433
    if use_fake:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from tests.conftest import FakeEnvRunner   # noqa: WPS433
        fake = FakeEnvRunner()

        class _FakePushCubeAdapter(PushCubeAdapter):
            def make_env_runner(self):
                return fake
        return _FakePushCubeAdapter()
    return PushCubeAdapter()
```

Find:

```python
    runner = _make_runner(args.fake_env)
    records: list[EpisodeRecord] = []
    try:
        for i in range(args.n_episodes):
            seed = args.seed_start + i
            episode_id = f"pushcube_blocked_approach_seed_{seed:04d}"
            rec = run_episode(
                episode_id=episode_id,
                seed=seed,
                env_runner=runner,
            )
            records.append(rec)
            ...
    finally:
        runner.close()
```

Replace with:

```python
    adapter = _make_adapter(args.fake_env)
    records: list[EpisodeRecord] = []
    try:
        for i in range(args.n_episodes):
            seed = args.seed_start + i
            episode_id = f"pushcube_blocked_approach_seed_{seed:04d}"
            rec = run_episode(
                episode_id=episode_id,
                seed=seed,
                adapter=adapter,
            )
            records.append(rec)
            with samples_path.open("a") as f:
                f.write(rec.to_jsonl_line() + "\n")
            print(
                f"[{i + 1}/{args.n_episodes}] seed={seed} "
                f"initial_success={rec.metrics['initial_success']} "
                f"retry_success={rec.metrics['retry_success']} "
                f"failure_type={rec.metrics['failure_type']}",
                flush=True,
            )
    finally:
        adapter.close()
```

(`runner.close()` is now `adapter.close()` — the adapter caches the env_runner
internally; `close()` releases it. Idempotent.)

- [ ] **Step 2: Smoke-test stage0_collect.py with fake env**

```bash
rm -rf /tmp/babysteps_task8
python scripts/stage0_collect.py \
    --out_dir /tmp/babysteps_task8 \
    --n_episodes 5 --seed_start 0 --fake-env
```

Expected: exit 0; samples.jsonl has 5 lines.

- [ ] **Step 3: Verify byte equality vs. snapshot**

```bash
diff tests/snapshots/pushcube_samples_seeds_0_4.jsonl \
     /tmp/babysteps_task8/samples.jsonl
echo "exit: $?"
```

Expected: empty diff, exit 0. If diff is non-empty, **STOP** — the refactor has changed output. Investigate before continuing.

- [ ] **Step 4: Update `scripts/render_stage0_maniskill.py`**

Find the import block (around lines 184–197):

```python
from babysteps.demo import demo_to_intent
from babysteps.envs.scene import direction_to_face, face_to_approach, face_to_push_unit
from babysteps.execution import (
    CUBE_HALF_SIZE, PRE_CONTACT_STANDOFF,
    PUSH_TRAVEL_SCALE, PUSH_TRAVEL_MAX_M,
)
from babysteps.failure import attribute_failure, build_failure_packet
from babysteps.revision import revise_intent
from babysteps.schemas import AttemptResult, DemoEvidence, Intent, SceneState
from babysteps.episode import (
    _default_blocked_sides_factory,
    _oracle_correct_intent_for_scene,
    generate_proxy_demo,
)
```

Replace with:

```python
from babysteps.envs.pushcube_adapter import PushCubeAdapter
from babysteps.envs.scene import face_to_push_unit
from babysteps.failure import attribute_failure, build_failure_packet
from babysteps.revision import revise_intent
from babysteps.schemas import AttemptResult, DemoEvidence, SceneState
from babysteps.skills.push import (
    CUBE_HALF_SIZE, PRE_CONTACT_STANDOFF,
    PUSH_TRAVEL_SCALE, PUSH_TRAVEL_MAX_M,
)
```

Then, in the body of `main(...)` where it currently calls
`_oracle_correct_intent_for_scene(scene)`, `demo_to_intent(demo_evidence)`,
and `_default_blocked_sides_factory(initial_intent)`, change them to method
calls on a single `adapter = PushCubeAdapter()` you create near the top of
`main`:

Find:

```python
            correct_intent = _oracle_correct_intent_for_scene(scene)
```

Replace with:

```python
            correct_intent = adapter.oracle_correct_intent(scene)
```

Find:

```python
            initial_intent = demo_to_intent(demo_evidence)
            scene_exec = SceneState(
                cube_xy=scene.cube_xy, cube_z=scene.cube_z,
                goal_xy=scene.goal_xy, tcp_start_pose=scene.tcp_start_pose,
                blocked_sides=_default_blocked_sides_factory(initial_intent),
            )
```

Replace with:

```python
            initial_intent = adapter.scripted_demo_to_intent(demo_evidence)
            from dataclasses import replace
            scene_exec = replace(
                scene,
                blocked_sides=adapter.default_blocked_factory(initial_intent),
            )
```

Just under the `try:` block (right after `env = gym.make(...)`), add:

```python
    adapter = PushCubeAdapter()
```

(`adapter` is then in scope inside the for-loop.)

- [ ] **Step 5: Static-check the renderer with `python -c`**

This script needs a GPU node to fully run, but a syntax/import check is
cheap on the login node:

```bash
python -c "import ast; ast.parse(open('scripts/render_stage0_maniskill.py').read())"
python -c "import sys; sys.path.insert(0, '.'); \
           import scripts.render_stage0_maniskill" 2>&1 | head -5
```

The second command will likely error on `import mani_skill.envs` (no Vulkan
on login node) — that's expected. What matters is that the error is
`mani_skill`-related, not `ImportError: cannot import name '_default_blocked_sides_factory'`.

- [ ] **Step 6: Check `scripts/render_stage0_topdown.py`**

```bash
grep -E "from babysteps\.(execution|demo)" scripts/render_stage0_topdown.py
```

If it imports either, repeat the same pattern: switch to `babysteps.skills.push`
and/or `PushCubeAdapter.scripted_demo_to_intent`.

- [ ] **Step 7: Run the full test suite as a final check**

```bash
python -m pytest tests/ -q
```

Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add scripts/stage0_collect.py scripts/render_stage0_maniskill.py
# include render_stage0_topdown.py only if it was edited
git commit -m "refactor(scripts): wire PushCubeAdapter through CLI scripts"
```

---

### Task 9: Snapshot acceptance test

Locks the byte-equality acceptance gate in CI.

**Files:**
- Modify: `tests/test_pushcube_adapter.py` (append the snapshot test)

- [ ] **Step 1: Append the snapshot acceptance test to `tests/test_pushcube_adapter.py`**

Append at the end of the file:

```python
# ---------- Snapshot acceptance test ----------------------------------- #


def test_pushcube_adapter_samples_jsonl_matches_pre_a_snapshot(fake_env_runner):
    """The byte-equality regression bar for Sub-project A.

    Captures the same five episodes the pre-A code produced and asserts
    the JSONL stream is byte-for-byte identical. If this test diffs, the
    refactor has changed observable output and Sub-project A is not done.
    """
    from pathlib import Path
    from babysteps.envs.pushcube_adapter import PushCubeAdapter
    from babysteps.episode import run_episode

    class _FakeAdapter(PushCubeAdapter):
        def make_env_runner(self):
            return fake_env_runner

    adapter = _FakeAdapter()
    actual_lines = []
    for seed in range(5):
        rec = run_episode(
            episode_id=f"pushcube_blocked_approach_seed_{seed:04d}",
            seed=seed,
            adapter=adapter,
        )
        actual_lines.append(rec.to_jsonl_line())
    actual = "\n".join(actual_lines) + "\n"

    snapshot_path = (
        Path(__file__).parent / "snapshots" / "pushcube_samples_seeds_0_4.jsonl"
    )
    expected = snapshot_path.read_text()
    assert actual == expected, (
        "PushCubeAdapter samples.jsonl drifted from the pre-A snapshot. "
        f"Snapshot at: {snapshot_path}. "
        "If this drift is intentional, re-capture with "
        "`python scripts/stage0_collect.py --out_dir /tmp/baseline "
        "--n_episodes 5 --seed_start 0 --fake-env` and copy the "
        "samples.jsonl into the snapshots/ dir."
    )
```

- [ ] **Step 2: Run the snapshot test**

```bash
python -m pytest tests/test_pushcube_adapter.py::test_pushcube_adapter_samples_jsonl_matches_pre_a_snapshot -v
```

Expected: PASS.

If FAIL: the assertion message tells you what to do. Investigate the diff
manually with:

```bash
python - <<'PY'
from pathlib import Path
from babysteps.envs.pushcube_adapter import PushCubeAdapter
from babysteps.episode import run_episode
from tests.conftest import FakeEnvRunner
fake = FakeEnvRunner()
class A(PushCubeAdapter):
    def make_env_runner(self): return fake
out = []
for seed in range(5):
    out.append(run_episode(
        episode_id=f"pushcube_blocked_approach_seed_{seed:04d}",
        seed=seed, adapter=A(),
    ).to_jsonl_line())
Path("/tmp/post_a.jsonl").write_text("\n".join(out) + "\n")
PY
diff tests/snapshots/pushcube_samples_seeds_0_4.jsonl /tmp/post_a.jsonl | head -40
```

- [ ] **Step 3: Run the full suite as the final acceptance gate**

```bash
python -m pytest tests/ -q
```

Expected: all pass. This is the spec's acceptance gate (1).

- [ ] **Step 4: Commit**

```bash
git add tests/test_pushcube_adapter.py
git commit -m "test(adapter): snapshot acceptance for PushCubeAdapter == pre-A"
```

---

### Task 10: Manual ManiSkill spot-check (optional, GPU node)

This is the spec's acceptance gate (3) — visual verification that real-sim
MP4s still look right. Skipped if no GPU node is available; the snapshot
test (Task 9) is the load-bearing gate.

**Files:** none

- [ ] **Step 1: Allocate a GPU node on Gilbreth**

```bash
salloc --gres=gpu:1 --time=00:30:00
# Once allocated, on the compute node:
cd /scratch/gilbreth/wang4433/babysteps
conda activate handover
```

- [ ] **Step 2: Render 2 episodes post-A**

```bash
mkdir -p /tmp/render_post_a
LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" \
python scripts/render_stage0_maniskill.py \
    --out_dir /tmp/render_post_a \
    --n_episodes 2 --seed_start 0
ls -la /tmp/render_post_a/videos_maniskill/
```

Expected: 6 MP4 files (3 phases × 2 episodes), each > 50 KB.

- [ ] **Step 3: Visual sanity check (manual)**

Open the MP4s locally (e.g., scp them or use VSCode's remote preview).
Confirm:
- Phase 1 (demo): the gripper pushes the cube toward the goal.
- Phase 2 (attempt_blocked): static frame with "approach_blocked" overlay.
- Phase 3 (retry): the gripper approaches from the opposite side and pushes
  the cube to goal.

If anything looks broken, file the issue and investigate before declaring
Sub-project A done.

---

## Self-Review

This section is for the plan author — quickly verify nothing dangled.

**Spec coverage check:** every spec section maps to one or more tasks.

| Spec section | Plan task(s) |
|--------------|--------------|
| §3 acceptance gate (1) full pytest | Task 9 step 3 |
| §3 acceptance gate (2) snapshot byte-equality | Task 1 (capture) + Task 9 (assert) |
| §3 acceptance gate (3) manual ManiSkill spot-check | Task 10 |
| §4.1 BaseTaskAdapter ABC | Task 4 |
| §4.2 SceneState.extra | Task 2 |
| §4.3 PushCubeAdapter | Task 5 |
| §4.4 episode.run_episode refactor | Task 6 |
| §4.5 execution.py → skills/push.py | Task 3 |
| §4.6 demo.py trim | Task 7 |
| §5 file manifest | Tasks 2–8 cover all entries |
| §6.1 test_task_adapter.py | Task 4 step 1 |
| §6.2 test_pushcube_adapter.py | Task 5 step 1 + Task 9 step 1 |
| §6.3 existing test edits | Task 3 step 5 (test_push_skill rename) + Task 6 step 1 (test_episode) + Task 7 step 3 (test_demo) |
| §6.4 snapshot fixture creation | Task 1 |
| §7 risks | Mitigations in code: Task 2 step 3 (extra empty-skip), Task 3 step 6 (repo-wide grep), Task 6 step 1 (signature guard test) |

**Type-consistency check:** the `BaseTaskAdapter` signatures in Task 4 match every consumer in Tasks 5, 6, and 8. `EnvRunner` Protocol signature (`reset`, `run`, `close`) matches `FakeEnvRunner` and `PushCubeEnvRunner`.

**Placeholder scan:** no TBDs, no "add error handling", no "similar to Task N", all code blocks are complete.
