# Stage-0 TaskAdapter Refactor — Design Spec

**Sub-project A** of the Stage-0 multi-task extension (PushCube → +PickCube → +StackCube).
Sub-projects B (PickCube full BABYSTEPS loop) and C (StackCube full BABYSTEPS loop) get
their own spec + plan files.

- Date: 2026-05-16
- Predecessor spec: `2026-05-15-stage0-pushcube-blocked-design.md`
- Goal-of-record: `goal.md` (Stage-0 boundary; object-centric intent schema)

## 1. Motivation

The current Stage-0 implementation is hard-wired to `PushCube-v1` in 11 places:
`schemas.py` whitelists, `demo.demo_to_intent`, `execution.compile_intent_to_push_skill`,
`envs/pushcube_runner.py`, `episode.run_episode` (3 sites: task literal, oracle intent
helper, blocked-sides factory), and the CLI scripts. Extending Stage-0 to PickCube and
StackCube without a refactor would duplicate the orchestration, the privileged-firewall
discipline, and the failure/revision plumbing once per task. Sub-project A introduces a
single `TaskAdapter` interface that pulls every task-specific decision behind one
boundary, keeps PushCube behaviour byte-identical, and unblocks B/C.

## 2. Non-Goals

- No new failure predicates, attribution rules, or revision operators.
- No `EMBODIMENT_MAPPINGS` / `REVISION_OPERATORS` whitelist additions.
- No PickCube or StackCube adapter — those land in B and C.
- No generalization of `render_stage0_maniskill.py` (still PushCube-only after A).
- No `--task` CLI flag (dead weight with only one adapter; B adds it).
- No new MP4s, no new dataset shapes. PushCube MP4s remain identical.

## 3. Acceptance Gate

A is done when **all three** hold:

1. `python -m pytest tests/ -q` — all 85 existing tests pass without their assertions
   changing (signatures may update; assertions about `EpisodeRecord` content do not).
2. **Snapshot byte-equality:** a freshly-run
   `scripts/stage0_collect.py --fake-env --n_episodes 5 --seed_start 0` produces a
   `samples.jsonl` that is byte-for-byte equal to `tests/snapshots/pushcube_samples_seeds_0_4.jsonl`
   (captured from the pre-refactor code).
3. Manual visual spot-check on a GPU compute node:
   `render_stage0_maniskill.py --n_episodes 2 --seed_start 0` produces three MP4s per
   episode whose frame count and rough trajectory match the pre-refactor MP4s
   (MP4 container metadata is non-deterministic, so checksum equality is not required).

## 4. Architecture

### 4.1 `BaseTaskAdapter` ABC

New file `babysteps/envs/task_adapter.py`:

```python
from abc import ABC, abstractmethod
from typing import Any

from babysteps.failure import Attribution
from babysteps.schemas import (
    AttemptResult, DemoEvidence, FailurePacket, Intent, Revision, SceneState,
)


class EnvRunner(Protocol):
    """Relocated from babysteps.episode — its natural home is alongside the
    adapter that consumes it. No circular import is created or removed; this
    is purely a logical-grouping move."""
    def reset(self, seed: int) -> SceneState: ...
    def run(self, intent: Intent, scene: SceneState) -> AttemptResult: ...
    def close(self) -> None: ...


class BaseTaskAdapter(ABC):
    """One concrete adapter per ManiSkill task. The episode loop and the CLI
    scripts depend ONLY on this base class — never on a concrete adapter."""

    task_id: str  # subclass must set, e.g. "PushCube-v1"

    # ---- env_runner lifecycle (concrete; cached) ----------------------- #
    def __init__(self) -> None:
        self._env_runner: EnvRunner | None = None

    def env_runner(self) -> EnvRunner:
        """Lazily construct (via make_env_runner) and cache the runner. The
        episode loop calls this every iteration; caching preserves the pre-A
        one-runner-across-all-seeds perf characteristic for real-sim runs."""
        if self._env_runner is None:
            self._env_runner = self.make_env_runner()
        return self._env_runner

    def close(self) -> None:
        """Release the cached runner if any. Idempotent."""
        if self._env_runner is not None:
            self._env_runner.close()
            self._env_runner = None

    # ---- abstract (must be implemented per task) ----------------------- #
    @abstractmethod
    def make_env_runner(self) -> EnvRunner: ...

    @abstractmethod
    def oracle_correct_intent(self, scene: SceneState) -> Intent: ...

    @abstractmethod
    def default_blocked_factory(self, intent: Intent) -> tuple[str, ...]: ...

    @abstractmethod
    def oracle_wrong_factor(self, intent: Intent, scene: SceneState) -> str: ...

    @abstractmethod
    def scripted_demo_to_intent(self, evidence: DemoEvidence) -> Intent: ...

    @abstractmethod
    def compile_skill(self, intent: Intent, scene: SceneState) -> Any: ...

    # ---- overridable hooks (default = shared rule-based modules) ------- #
    def build_failure_packet(
        self, intent: Intent, attempt: AttemptResult, scene: SceneState,
    ) -> FailurePacket:
        from babysteps.failure import build_failure_packet
        return build_failure_packet(intent, attempt, scene)

    def attribute_failure(self, fp: FailurePacket) -> Attribution:
        from babysteps.failure import attribute_failure
        return attribute_failure(fp)

    def revise_intent(
        self, intent: Intent, attribution: Attribution, scene: SceneState,
    ) -> tuple[Intent, Revision]:
        from babysteps.revision import revise_intent
        return revise_intent(intent, attribution, scene)
```

`PushCubeAdapter` (Section 4.3) implements the 6 abstract methods and inherits the 3
hook defaults — net behaviour identical to the current code. Sub-projects B and C will
override `revise_intent` (Pick: `contact_region_substitution`) and possibly
`build_failure_packet` (Stack: predicate precedence may differ for misalignment).

### 4.2 `SceneState.extra: dict`

`schemas.SceneState` gains one field:

```python
@dataclass(frozen=True)
class SceneState:
    cube_xy: tuple[float, float]
    cube_z: float
    goal_xy: tuple[float, float]
    tcp_start_pose: tuple[float, float, float, float, float, float, float]
    blocked_sides: tuple[str, ...]
    extra: dict = field(default_factory=dict)   # NEW — adapter-owned payload
```

`to_dict` serializes `extra` **only when non-empty**, and emits it as the last key in
the dict. `from_dict` defaults missing `extra` to `{}`. This empty-skip rule is what
preserves byte-for-byte JSON equality for PushCube records (which never populate
`extra`):

```python
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
    ...
    extra = dict(d.get("extra", {}))
    return cls(..., extra=extra)
```

PushCube readers (the push skill compiler, the reviser, the failure packet builder)
ignore `extra`. PickCube will populate `extra["gripper_width"]` etc.; StackCube
`extra["base_cube_xy"]`.

The weak typing inside `extra` is acceptable because the only producer/consumer pair
for any given key is one adapter and its sibling skill compiler.

### 4.3 `PushCubeAdapter`

New file `babysteps/envs/pushcube_adapter.py`:

```python
import numpy as np

from babysteps.envs.scene import direction_to_face, face_to_approach
from babysteps.envs.task_adapter import BaseTaskAdapter, EnvRunner
from babysteps.envs.pushcube_runner import PushCubeEnvRunner
from babysteps.demo import trajectory_to_motion        # the surviving utility
from babysteps.skills.push import compile_intent_to_push_skill
from babysteps.schemas import CONTACT_REGIONS, DemoEvidence, Intent, SceneState


class PushCubeAdapter(BaseTaskAdapter):
    task_id = "PushCube-v1"

    def make_env_runner(self) -> EnvRunner:
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

    def oracle_wrong_factor(self, intent: Intent, scene: SceneState) -> str:
        return (
            "approach_direction"
            if intent.approach_direction in scene.blocked_sides
            else "none"
        )

    def scripted_demo_to_intent(self, evidence: DemoEvidence) -> Intent:
        contact = evidence.contact_region_label
        if contact not in CONTACT_REGIONS:
            raise ValueError(
                f"DemoEvidence.contact_region_label must be in "
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

Hook defaults (`build_failure_packet`, `attribute_failure`, `revise_intent`) are
inherited unchanged.

### 4.4 `babysteps.episode.run_episode`

The orchestration becomes task-agnostic:

```python
def run_episode(
    *, episode_id: str, seed: int, adapter: BaseTaskAdapter,
) -> EpisodeRecord:
    env = adapter.env_runner()      # cached — see §4.1
    scene_initial = env.reset(seed)
    demo_evidence = generate_proxy_demo(env, scene_initial, adapter)
    initial_intent = adapter.scripted_demo_to_intent(demo_evidence)
    scene_executor = replace(
        scene_initial,
        blocked_sides=adapter.default_blocked_factory(initial_intent),
    )
    oracle_wrong = adapter.oracle_wrong_factor(initial_intent, scene_executor)

    attempt_1 = env.run(initial_intent, scene_executor)
    fp = adapter.build_failure_packet(initial_intent, attempt_1, scene_executor)
    # ... (success / attribute / NotImplementedError / revise / retry paths
    #      unchanged from current episode.py, except every call site uses
    #      `adapter.<hook>` instead of importing the shared module directly)
    return EpisodeRecord(task=adapter.task_id, ...)
```

`generate_proxy_demo` also takes the adapter (so it can call
`adapter.oracle_correct_intent` and `adapter.scripted_demo_to_intent` for the
demo-proxy synthesis path).

The runner's lifecycle (allocate + close) is owned by the adapter, not the
loop. Callers do:

```python
adapter = PushCubeAdapter()
try:
    for seed in range(N):
        run_episode(..., adapter=adapter)
finally:
    adapter.close()
```

This preserves the pre-A one-allocation-across-all-seeds behaviour for
expensive real-sim runners.

### 4.5 `execution.py` → `babysteps/skills/push.py`

The current `babysteps/execution.py` is renamed to `babysteps/skills/push.py` (with a
new `babysteps/skills/__init__.py`). Content is identical — only the module path
changes. This sets up symmetry for B (`babysteps/skills/pick.py`) and C
(`babysteps/skills/stack.py`).

Imports updated:
- `babysteps/envs/pushcube_runner.py` — `from babysteps.execution import …` → `from babysteps.skills.push import …`.
- `babysteps/envs/pushcube_adapter.py` (new) — imports from `babysteps.skills.push`.
- `scripts/render_stage0_maniskill.py` — import path updated.
- `tests/test_execution.py` — renamed to `tests/test_push_skill.py`, import paths updated. Test bodies unchanged.

### 4.6 `babysteps.demo`

The push-specific `demo_to_intent` moves into `PushCubeAdapter.scripted_demo_to_intent`.
The reusable utility `trajectory_to_motion` stays in `babysteps/demo.py`. The module
docstring is updated to reflect its narrower scope: "scripted intent-extraction
utilities; per-task extractors live in the task's adapter."

If after B/C nothing besides `trajectory_to_motion` survives in `babysteps/demo.py`,
the module is folded into `babysteps/envs/scene.py` in a later cleanup. Not in scope
for A.

## 5. File Manifest

| File | Action | Approx LOC |
|------|--------|-----------|
| `babysteps/envs/task_adapter.py` | NEW | ~40 |
| `babysteps/envs/pushcube_adapter.py` | NEW | ~90 |
| `babysteps/skills/__init__.py` | NEW | 1 |
| `babysteps/skills/push.py` | MOVED from `execution.py` | unchanged |
| `babysteps/execution.py` | DELETED | — |
| `babysteps/episode.py` | EDIT | −80 / +30 |
| `babysteps/demo.py` | EDIT | −20 (moved into adapter) |
| `babysteps/schemas.py` | EDIT | +6 (extra field + roundtrip) |
| `babysteps/failure.py`, `revision.py`, `eval.py`, `viz.py` | KEEP | — |
| `scripts/stage0_collect.py` | EDIT | +5 |
| `scripts/render_stage0_maniskill.py` | EDIT | +5 |
| `tests/test_task_adapter.py` | NEW | ~80 |
| `tests/test_pushcube_adapter.py` | NEW | ~60 |
| `tests/test_episode.py` | EDIT | signature update only |
| `tests/test_execution.py` | RENAME to `test_push_skill.py` | imports |
| `tests/conftest.py` | EDIT (if needed) | small |
| `tests/snapshots/pushcube_samples_seeds_0_4.jsonl` | NEW (fixture) | ~5 lines JSONL |

## 6. Test Plan

### 6.1 `tests/test_task_adapter.py`
- `BaseTaskAdapter` is abstract: instantiating it raises `TypeError`.
- A subclass missing any of the 6 abstract methods cannot be instantiated.
- The three hook defaults (`build_failure_packet`, `attribute_failure`,
  `revise_intent`) really delegate to `babysteps.failure` / `babysteps.revision`
  (monkeypatch + assert the delegate was called with the same args).
- A `_StubAdapter` (in the test file) drives a fake env runner through
  `run_episode` to a successful retry, producing an `EpisodeRecord` with
  `task == _StubAdapter.task_id`.

### 6.2 `tests/test_pushcube_adapter.py`
- `PushCubeAdapter().task_id == "PushCube-v1"`.
- `oracle_correct_intent(scene)` produces the same `Intent` as the
  pre-refactor `_oracle_correct_intent_for_scene` for each of 4 canonical
  fixture scenes (one per cardinal goal direction).
- `default_blocked_factory(intent) == (intent.approach_direction,)` for each
  of the 4 cardinal approach directions.
- `scripted_demo_to_intent(evidence)` matches `Intent`-equality with the
  pre-refactor `demo.demo_to_intent(evidence)` across a fixture of 4 evidence
  values.
- `compile_skill(intent, scene)` returns the same `PushSkill` (or `None`
  when blocked) as the pre-refactor `compile_intent_to_push_skill`.
- **Snapshot acceptance:**
  `run_episode(..., adapter=PushCubeAdapter())` over seeds 0..4 with the
  fake env produces a JSONL stream byte-for-byte equal to
  `tests/snapshots/pushcube_samples_seeds_0_4.jsonl`.

### 6.3 Existing tests
All 85 existing tests pass with at most the following mechanical edits:
- `tests/test_episode.py`: replace `run_episode(env_runner=fake, blocked_sides_factory=fac)`
  with `run_episode(adapter=stub_adapter)`. Assertions unchanged.
- `tests/test_execution.py`: rename to `tests/test_push_skill.py`; imports
  switch from `babysteps.execution` to `babysteps.skills.push`.

### 6.4 Snapshot fixture creation (first plan step)
Before any refactor edit:
```bash
python scripts/stage0_collect.py \
    --out_dir /tmp/baseline --n_episodes 5 --seed_start 0 --fake-env
mkdir -p tests/snapshots
cp /tmp/baseline/samples.jsonl tests/snapshots/pushcube_samples_seeds_0_4.jsonl
git add tests/snapshots/pushcube_samples_seeds_0_4.jsonl
git commit -m "Stage-0 snapshot fixture (pre-A baseline)"
```
The snapshot test would fail at this point because the adapter doesn't exist
yet — that's the TDD red bar A starts from.

## 7. Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| `extra: dict` field shifts JSON ordering, breaking byte-equality | `to_dict` emits `extra` last and only if non-empty (`if self.extra:`); PushCube never populates it, so JSON shape is unchanged. |
| Renaming `execution.py` breaks an import in CLI scripts not in tests/ | Grep `from babysteps.execution`, `import babysteps.execution` repo-wide before deletion. |
| `generate_proxy_demo` needs the adapter, but signature change ripples | It's an internal helper — only `episode.run_episode` calls it. Signature change is bounded. |
| Hook delegation pattern subtly changes call semantics | Tests in 6.1 lock in delegation with monkeypatch + call recording. |

## 8. Out-of-Scope, Captured for Follow-On Sub-Projects

- **B (PickCube):** add `EMBODIMENT_MAPPINGS += "proxy_grasp_to_franka_pick"`,
  `GOAL_STATES += "object_grasped"` (or reuse `cube_at_target`),
  `REVISION_OPERATORS += "contact_region_substitution"`. Add
  `babysteps/skills/pick.py`, `babysteps/envs/pickcube_runner.py`,
  `babysteps/envs/pickcube_adapter.py`. Add a fake pick runner in tests.
- **C (StackCube):** add `GOAL_STATES += "cube_stacked_on(base)"`,
  `REVISION_OPERATORS += "goal_refinement"`. Add stack skill + runner +
  adapter. Possibly subclass `SceneState` (or extend `extra`) for two-cube
  geometry.
- **Generalized `--task` CLI:** adds `--task {PushCube-v1, PickCube-v1, StackCube-v1}`
  to `stage0_collect.py` and `render_stage0_maniskill.py`. Adapter registry
  in `babysteps/envs/__init__.py`.
- **Generalized renderer:** `render_stage0_maniskill.py` builds waypoints from
  the adapter's `compile_skill`, not from an inline `_build_waypoints` copy.
