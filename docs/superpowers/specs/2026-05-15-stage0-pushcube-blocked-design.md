# Stage 0 PushCube Blocked-Approach Data Preparation — Design

**Date:** 2026-05-15
**Status:** Approved (failure mechanism = privileged `blocked_sides` flag).
**Supersedes (for Stage 0):** `2026-05-15-pushcube-vertical-slice-design.md`
(that spec uses a `target / goal / interaction.contact_site / reference_frame`
intent schema; Stage 0 uses the object-centric schema mandated by `goal.md`).

---

## 1. Purpose

Build the smallest end-to-end **data-preparation** loop that produces honest
episode records of the form

```text
third-person demonstration proxy
  → structured intent JSON
  → Franka robot-centric execution attempt
  → controlled semantic failure packet
  → factor-local revision record
  → retry record
```

on ManiSkill's `PushCube-v1`, with **no VLM, no DINO, no diffusion, no real
Franka**. This validates the Stage 0 **data contract** (goal.md §"Episode Data
Format") and the **factor-local revision loop** before any learned perception,
attribution, or scoring is bolted on.

## 2. Non-goals (Stage 0)

- No PickCube, no PullCube, no drawer.
- No real human videos (the input is *proxy demos*, not human demos — goal.md
  §"Boundary Line").
- No DINO/DINOv2 feature extraction.
- No VLM intent proposer (intent is derived deterministically from labeled demo
  evidence).
- No diffusion counterfactual scorer.
- No GPU sim, no headless RGB rendering. `obs_mode="state_dict"` defers Vulkan.
- No baselines yet (full replanning, failure-agnostic retry, etc.). The
  metrics schema reserves columns for them; implementation comes later.
- No closed-loop control. Open-loop scripted 3-phase push, identical pattern
  to the Pick4Pass `run_pushcube_loop` reference.

## 3. Boundary line — "proxy demo, not human demo"

Every label, docstring, and report string says **"third-person demonstration
proxy"**. `goal.md` §"Boundary Line" is the canonical phrasing. The paper-facing
claim is *cross-view proxy-demo-to-Franka execution with failure-guided
structured intent revision*. Implementation enforces this with a single
`DEMONSTRATOR_TYPE = "proxy_oracle"` constant and a `claim_boundary` field on
every episode record (`"third_person_demo_proxy_not_human_demo"`).

## 4. Architectural diagram

```text
                  (privileged sim state — labels & success only)
                                     │
                                     ▼
[seed S] ── PushCube-v1 reset ── SceneState (cube_xy, goal_xy, blocked_sides)
                                     │
            ┌────────────────────────┼──────────────────────────┐
            ▼                        │                          ▼
  Demo-proxy generator               │                  Execution attempts
  (oracle scripted push,             │                  (Franka robot-centric)
   external camera = world frame)    │                          │
            │                        │                          │
            ▼                        │                          ▼
  DemoEvidence (object               │             AttemptResult (rollout +
   trajectory, contact label,        │              terminal info, no
   final state)                      │              privileged peek)
            │                        │                          │
            ▼                        │                          ▼
  demo_to_intent (scripted) ──> Intent (initial) ───────► PushSkillCompiler
                                                          (feasibility check
                                                           against blocked_sides
                                                           → planner_failed OR
                                                            scripted waypoints)
                                                                │
                                                                ▼
                                                          run_attempt(env_runner)
                                                                │
                                                                ▼
            ┌──────────────────────────────────────────► detect_failure
            │                                                 │
            │                                                 ▼
            │                                       FailurePacket
            │                                                 │
            │                                                 ▼
            │                                       attribute_failure (rules)
            │                                                 │
            │                                                 ▼
            │                                       {wrong_factor, freeze, revise}
            │                                                 │
            │                                                 ▼
            │                                       revise_intent (operator)
            │                                                 │
            └─────────────────── retry attempt ◄──────────────┘
```

**Boundary rules** (enforced by module layout, not by convention):

1. Only `babysteps/envs/pushcube_runner.py` imports `mani_skill`.
2. `demo_to_intent` consumes a `DemoEvidence` object that carries only
   *demo-visible* quantities (object trajectory, contact-region label, final
   state). It must not read `SceneState.blocked_sides`, `goal_xy`, or any
   field tagged `# privileged`. This is the goal.md §5 constraint: privileged
   state stays out of the demo-to-intent input path.
3. `failure.py` and `revision.py` are pure dataclasses-in / dataclasses-out.
4. `episode.py` takes an `env_runner: Callable` and never imports gymnasium
   or mani_skill. The CLI wires the real one.

## 5. Data contracts

### 5.1 `Intent` (object-centric, per goal.md §"Stage 0 Intent Factors")

```python
@dataclass(frozen=True)
class Intent:
    goal_state: str            # "cube_at_target"
    object_motion: str         # "translate_+x" | "translate_-x" | "translate_+y" | "translate_-y"
    contact_region: str        # "minus_x_face" | "plus_x_face" | "minus_y_face" | "plus_y_face"
    approach_direction: str    # "from_minus_x" | "from_plus_x" | "from_minus_y"
                               # | "from_plus_y" | "from_above"
    constraint_region: str     # "none" for Stage 0
    embodiment_mapping: str    # "proxy_contact_to_franka_push"

    def to_dict(self) -> dict
    @classmethod
    def from_dict(cls, d: dict) -> "Intent"
```

**Decisions:**
- All six factors from `goal.md`. No task-specific fields (e.g. no
  `push_side_correct`, no `drawer_axis_correct`).
- World-frame face/direction strings instead of generic `"left" / "right"` —
  unambiguous under PushCube's random goal placement.
- `approach_direction` and `contact_region` are **decoupled**: contact_region
  is which cube face the EE touches; approach_direction is the path the EE
  takes to reach it. Convention B from brainstorming. This is what makes
  factor-local revision honest — Stage 0 revises `approach_direction`
  alone, while `contact_region` stays fixed.

### 5.2 `DemoEvidence` (what the demo proxy hands forward)

```python
@dataclass(frozen=True)
class DemoEvidence:
    camera: str                       # "third_person" (Stage 0 fixed)
    demonstrator_type: str            # "proxy_oracle"
    object_trajectory: list[tuple[float, float]]   # cube xy over time, world frame
    contact_region_label: str         # face touched in the demo, e.g. "minus_x_face"
    final_state: str                  # "cube_at_target"
    rgbd_video_path: str | None       # None at Stage 0; placeholder for later
```

**Privileged state forbidden:** no `goal_xy`, no `tcp_pose`, no `blocked_sides`.
The contact_region_label is allowed because in the proxy demo the contact face
is observable from a third-person view (it is *what was demonstrated*); it's
*not* the same as reading `SceneState.cube_xy`.

### 5.3 `SceneState` (sim-agnostic, privileged-marked)

```python
@dataclass(frozen=True)
class SceneState:
    cube_xy: tuple[float, float]           # privileged
    cube_z: float                          # privileged
    goal_xy: tuple[float, float]           # privileged
    tcp_start_pose: tuple[float, ...]      # privileged (xyz + quat xyzw)
    blocked_sides: tuple[str, ...]         # privileged (subset of approach_direction values)
```

Every field is privileged. `SceneState` is consumed by the skill compiler
(feasibility check, waypoint geometry) and by `compute_metrics` (oracle
labels). It is **never** passed into `demo_to_intent`.

### 5.4 `AttemptResult` (one execution attempt's evidence)

```python
@dataclass(frozen=True)
class AttemptResult:
    initial_obj_xy: tuple[float, float]
    final_obj_xy:   tuple[float, float]
    goal_xy:        tuple[float, float]
    reached_contact: bool
    object_moved:    bool
    planner_failed:  bool
    collision:       bool
    grasp_slip:      bool       # always False for PushCube; kept for schema stability
    rollout_log_path: str | None    # .npz of per-step poses, or None for planner_failed
    success: bool                # ManiSkill info["success"]
```

### 5.5 `FailurePacket` (the goal.md §"Build Failure Packet" shape)

```python
@dataclass(frozen=True)
class FailurePacket:
    chosen_intent: Intent
    execution_trace: dict       # {reached_contact, object_moved, collision,
                                #  planner_failed, grasp_slip}
    failure_predicate: str      # "approach_blocked" | "direction_error"
                                # | "no_motion" | "goal_not_satisfied" | "none"
    # Diagnostic numerics — for ablations and later diffusion scoring.
    object_displacement: float | None
    direction_alignment: float | None
```

### 5.6 `Revision` (goal.md §"Revise and Retry")

```python
@dataclass(frozen=True)
class Revision:
    operator: str               # "approach_substitution" (Stage 0)
    factor: str                 # "approach_direction"
    old_value: str
    new_value: str
    frozen_factors: tuple[str, ...]
```

### 5.7 `EpisodeRecord` (one JSONL line, matches goal.md "Episode Data Format")

```python
@dataclass(frozen=True)
class EpisodeRecord:
    episode_id: str             # "pushcube_blocked_approach_seed_0001"
    stage: str                  # "stage_0"
    task: str                   # "PushCube-v1"
    claim_boundary: str         # "third_person_demo_proxy_not_human_demo"
    demo: dict                  # DemoEvidence + camera + final_state
    execution: dict             # {camera, robot, initial_intent, success}
    failure_packet: dict        # FailurePacket + attribution (wrong_factor,
                                #   oracle_wrong_factor, freeze, revise)
    revision: dict | None       # Revision; None if no failure
    retry: dict | None          # {success, num_retries, final_intent}; None if no failure
    metrics: dict               # per-episode metrics (see §9)
```

`to_jsonl_line() / from_jsonl_line(s)` produce one JSON object per line —
mirrors Pick4Pass `PushSample.to_json`.

## 6. Control flow

### 6.1 Demo proxy generation

```python
def generate_proxy_demo(env_runner: EnvRunner, scene: SceneState) -> DemoEvidence:
    # 1. Compute correct contact_region from cube → goal direction.
    correct_face = direction_to_face(unit(goal_xy - cube_xy))
    # 2. Build a correct intent (oracle).
    correct_intent = Intent(
        goal_state="cube_at_target",
        object_motion=goal_direction_to_motion(goal_xy - cube_xy),
        contact_region=correct_face,
        approach_direction=face_to_approach(correct_face),  # paired with correct_face
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_push",
    )
    # 3. Execute it via env_runner with blocked_sides=() — the demo proxy
    #    runs in an unblocked variant of the scene.
    demo_attempt = env_runner(correct_intent, scene._replace(blocked_sides=()))
    # 4. Pack the demo-visible evidence.
    return DemoEvidence(
        camera="third_person",
        demonstrator_type="proxy_oracle",
        object_trajectory=demo_attempt.object_trajectory,   # xy per step
        contact_region_label=correct_face,
        final_state="cube_at_target",
        rgbd_video_path=None,
    )
```

The proxy demo is the **oracle** — that's what makes it a proxy. The
contact-region label is derived from privileged ground truth, but it is
exposed *only as demo-visible evidence* (a label on the third-person video).
Later stages replace this with DINO grounding.

### 6.2 Intent extraction (scripted, no VLM)

```python
def demo_to_intent(evidence: DemoEvidence) -> Intent:
    # 1. Read contact_region and final_state from evidence (demo-visible).
    contact_region = evidence.contact_region_label
    # 2. Infer object_motion from the trajectory's net displacement.
    object_motion = trajectory_to_motion(evidence.object_trajectory)
    # 3. Pair approach_direction with contact_region by Stage-0 convention.
    approach_direction = face_to_approach(contact_region)
    return Intent(
        goal_state="cube_at_target",
        object_motion=object_motion,
        contact_region=contact_region,
        approach_direction=approach_direction,
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_push",
    )
```

No `goal_xy` is read here — only `evidence`. That's the privileged-state
firewall.

### 6.3 Skill compilation with feasibility check

```python
def compile_intent_to_push_skill(intent: Intent, scene: SceneState) -> PushSkill | None:
    """Returns a PushSkill ready for env_runner, or None when blocked.
    None propagates as planner_failed=True in the AttemptResult."""
    if intent.approach_direction in scene.blocked_sides:
        return None                                  # ← THE BLOCK
    return PushSkill(
        waypoints=build_push_waypoints(scene, intent),
    )
```

`build_push_waypoints` is pure (3-waypoint geometry, matching Pick4Pass's
`build_push_waypoints` but parameterized by `contact_region` + `goal_xy` —
**not** by `approach_direction`, which is the semantic field). This decouples
the physical push from the semantic feasibility flag.

### 6.4 Episode loop

```python
def run_episode(
    episode_id: str,
    seed: int,
    env_runner: EnvRunner,
    *,
    blocked_sides_factory: Callable[[Intent], tuple[str, ...]],
) -> EpisodeRecord:

    # 1. Reset, observe scene.
    scene_unblocked = env_runner.reset(seed)                 # blocked_sides=()
    # 2. Demo proxy.
    demo_evidence = generate_proxy_demo(env_runner, scene_unblocked)
    # 3. Derive initial intent.
    initial_intent = demo_to_intent(demo_evidence)
    # 4. Configure the executor's scene: block the demo's preferred approach.
    scene_executor = scene_unblocked._replace(
        blocked_sides=blocked_sides_factory(initial_intent),
    )
    # 5. Attempt 1.
    attempt_1 = run_attempt(initial_intent, scene_executor, env_runner)
    failure_packet = build_failure_packet(initial_intent, attempt_1, scene_executor)

    if failure_packet.failure_predicate == "none":
        return EpisodeRecord(... no failure ...)

    # 6. Attribution.
    attribution = attribute_failure(failure_packet)          # rule table

    # 7. Revise (factor-local).
    revised_intent, revision = revise_intent(
        initial_intent, attribution, scene_executor,
    )

    # 8. Attempt 2 (retry).
    attempt_2 = run_attempt(revised_intent, scene_executor, env_runner)
    retry_record = build_retry_record(revised_intent, attempt_2)

    # 9. Pack into EpisodeRecord with per-episode metrics.
    return EpisodeRecord(...)
```

`blocked_sides_factory(initial_intent) = (initial_intent.approach_direction,)`
is the Stage 0 default — it makes attempt 1 deterministically infeasible
regardless of seed. Future stages replace this with a richer feasibility
oracle.

## 7. Failure detection & attribution

### 7.1 Detection — derive predicate from execution trace

```python
def build_failure_packet(intent, attempt, scene) -> FailurePacket:
    et = {
        "reached_contact": attempt.reached_contact,
        "object_moved":    attempt.object_moved,
        "collision":       attempt.collision,
        "planner_failed":  attempt.planner_failed,
        "grasp_slip":      attempt.grasp_slip,
    }
    if attempt.success:           pred = "none"
    elif attempt.planner_failed:  pred = "approach_blocked"
    elif not attempt.reached_contact:                   pred = "contact_failure"
    elif not attempt.object_moved:                      pred = "no_motion"
    elif _direction_alignment(attempt) < 0:             pred = "direction_error"
    else:                                                pred = "goal_not_satisfied"
    return FailurePacket(
        chosen_intent=intent, execution_trace=et,
        failure_predicate=pred,
        object_displacement=_disp(attempt),
        direction_alignment=_direction_alignment(attempt),
    )
```

### 7.2 Attribution — predicate → wrong_factor

```python
FAILURE_TO_FACTOR: dict[str, tuple[str, tuple[str, ...]]] = {
    "approach_blocked":   ("approach_direction", ("approach_direction", "contact_region")),
    "direction_error":    ("approach_direction", ("approach_direction",)),
    "contact_failure":    ("contact_region",     ("contact_region",)),
    "no_motion":          ("approach_direction", ("approach_direction", "contact_region")),
    "goal_not_satisfied": ("goal_state",         ("goal_state",)),
}

def attribute_failure(fp: FailurePacket) -> Attribution:
    wrong_factor, revise = FAILURE_TO_FACTOR[fp.failure_predicate]
    freeze = tuple(f for f in INTENT_FIELDS if f not in revise)
    return Attribution(
        semantic_failure=True,
        wrong_factor=wrong_factor,
        freeze=freeze,
        revise=revise,
    )
```

The mapping is small and intentional. Predicate `approach_blocked` is the only
one Stage 0 actually exercises; the others are stable placeholders for
follow-on stages.

## 8. Revision

```python
OPPOSITE_APPROACH = {
    "from_minus_x": "from_plus_x",
    "from_plus_x":  "from_minus_x",
    "from_minus_y": "from_plus_y",
    "from_plus_y":  "from_minus_y",
}

def revise_intent(intent, attribution, scene) -> tuple[Intent, Revision]:
    if attribution.wrong_factor == "approach_direction":
        old = intent.approach_direction
        # Pick first unblocked alternative — Stage 0 candidate-pool generator.
        candidates = [v for v in OPPOSITE_APPROACH.values() | {"from_above"}
                      if v != old and v not in scene.blocked_sides]
        new = candidates[0] if candidates else "from_above"
        revised = replace(intent, approach_direction=new)
        return revised, Revision(
            operator="approach_substitution",
            factor="approach_direction",
            old_value=old, new_value=new,
            frozen_factors=tuple(f for f in INTENT_FIELDS if f != "approach_direction"),
        )
    raise NotImplementedError(...)
```

Other wrong_factor branches raise `NotImplementedError` in Stage 0 — honest
about what is and isn't validated, matching the vertical-slice spec's pattern.

**Critical property** — the revision changes *exactly one* factor. The
summarizer's non-regression check (next section) audits this on every
revised episode and reports the fraction that satisfy it.

## 9. Per-episode metrics

Every `EpisodeRecord.metrics` carries:

```python
{
  "initial_success":            bool,
  "retry_success":              bool | None,   # None when no failure
  "num_attempts_to_success":    int,           # 1 or 2; cap at MAX_ATTEMPTS
  "failure_type":               str,
  "wrong_factor_predicted":     str | None,
  "oracle_wrong_factor":        str,
  "factor_attribution_correct": bool | None,
  "factors_changed":            tuple[str, ...],   # for non-regression check
  "frozen_factors_preserved":   bool | None,
}
```

The dataset-level summarizer (`scripts/stage0_summarize.py`) aggregates these
into:

```text
final_success_rate
retry_success_rate
num_attempts_to_success (mean)
failure_type_accuracy
intent_factor_attribution_accuracy
unnecessary_factor_change_rate     == 1 - frozen_factor_preservation
frozen_factor_preservation_rate
revision_success_rate
non_regression_score               (revised exactly the predicted factor)
delta_pp                           (retry_success − initial_success), pp
passed_acceptance                  (delta_pp >= 10)
```

These are exactly the goal.md §"Required Metrics" rows. The Pick4Pass acceptance
bar (`delta_pp >= 10 pp`) is kept as the Stage 0 acceptance gate.

## 10. PushCube specifics

### 10.1 Env construction (Pick4Pass-validated configuration)

```python
gym.make(
    "PushCube-v1",
    obs_mode="state_dict",
    control_mode="pd_ee_delta_pose",
    sim_backend="cpu",
)
```

CPU sim is intentional. No RGB. No Vulkan dependency at Stage 0.

### 10.2 Push controller

Open-loop, 3-phase normalized proportional EE control, identical pattern to
Pick4Pass's `run_pushcube_loop.py::_run_one_push`:

1. Approach pre-contact at travel height (no descent).
2. Descend to push height (cube center z).
3. Sweep TCP forward by `_PUSH_TRAVEL_SCALE * intent.push_distance`,
   capped at `_PUSH_TRAVEL_MAX_M`.

`push_distance` is not a separate Stage 0 intent factor — it's derived
internally by the skill compiler from `cube_xy → goal_xy`. (Pick4Pass made
distance a perturbable factor; Stage 0 doesn't, because the goal.md schema
doesn't include distance.)

### 10.3 Geometry constants (copied from Pick4Pass calibration)

```python
CUBE_HALF_SIZE = 0.02
PRE_CONTACT_STANDOFF = 0.005
PUSH_TRAVEL_SCALE = 0.6
PUSH_TRAVEL_MAX_M = 0.15
POS_SCALE = 0.1                # pd_ee_delta_pose normalization
PHASE_TOL_M = 0.015
MAX_CONTROL_STEPS = 300
```

## 11. File layout

```
babysteps/
├── conda.yaml                     # name: handover (reuse), or new minimal spec
├── pyproject.toml                 # editable install: pip install -e .
├── README.md
├── CLAUDE.md (existing)
├── goal.md (existing, authoritative)
├── technical_def.md (existing)
├── docs/superpowers/
│   ├── specs/
│   │   ├── 2026-05-15-pushcube-vertical-slice-design.md   (existing; older schema)
│   │   └── 2026-05-15-stage0-pushcube-blocked-design.md   (THIS FILE)
│   └── plans/
│       └── 2026-05-15-stage0-pushcube-blocked-plan.md     (next: writing-plans)
│
├── babysteps/
│   ├── __init__.py
│   ├── schemas.py
│   ├── demo.py
│   ├── execution.py
│   ├── failure.py
│   ├── revision.py
│   ├── episode.py
│   ├── eval.py
│   └── envs/
│       ├── __init__.py
│       ├── scene.py
│       └── pushcube_runner.py     # only file importing mani_skill
│
├── scripts/
│   ├── smoke_pushcube.py          # ManiSkill loadability + 1 reset
│   ├── stage0_collect.py          # CLI: N episodes → samples.jsonl
│   └── stage0_summarize.py        # samples.jsonl → report.{md,json}
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                # fake env_runner fixture
│   ├── test_schemas.py
│   ├── test_demo.py
│   ├── test_execution.py
│   ├── test_failure.py
│   ├── test_revision.py
│   ├── test_episode.py
│   └── test_eval.py
│
└── datasets/                      # existing; outputs land here
    └── stage0_pushcube_blocked/   # created by stage0_collect.py
        ├── samples.jsonl
        ├── report.md
        ├── report.json
        └── rollouts/<episode_id>/...
```

## 12. Test pyramid

All tests are sim-free except the smoke test. The fake `env_runner` fixture
in `conftest.py` lets `test_episode.py` exercise the full
demo → intent → fail → revise → retry path without ManiSkill.

| Module      | Test file              | Asserts |
|-------------|------------------------|---------|
| `schemas`   | `test_schemas.py`      | JSON round-trips, validation errors, frozen invariants |
| `demo`      | `test_demo.py`         | `demo_to_intent` is deterministic; never reads privileged state (verified by passing only `DemoEvidence`) |
| `execution` | `test_execution.py`    | `build_push_waypoints` geometry; `compile_intent_to_push_skill` returns None ⇔ approach_direction ∈ blocked_sides |
| `failure`   | `test_failure.py`      | every predicate branch + every attribution row |
| `revision`  | `test_revision.py`     | `approach_substitution` flips exactly approach_direction; opposite chosen; frozen_factors enumerates everything else |
| `episode`   | `test_episode.py`      | full loop with fake env_runner: blocked path → revise → unblocked retry succeeds; record JSON shape matches goal.md example |
| `eval`      | `test_eval.py`         | metrics arithmetic, non-regression score, acceptance threshold |

`scripts/smoke_pushcube.py` — runs only if `mani_skill` imports; on Gilbreth's
`handover` env it should succeed. Failure is reported but does not block unit
tests.

## 13. Validation criteria

Stage 0 is complete when **all** of:

1. `pytest tests/` exits 0. (Sim-free, runs anywhere.)
2. `python scripts/smoke_pushcube.py` exits 0 in the `handover` env.
3. `python scripts/stage0_collect.py --n_episodes 5 --seed_start 0 \
    --out_dir datasets/stage0_pushcube_blocked` produces `samples.jsonl` with
   5 episodes, each:
   - `demo.demonstrator_type == "proxy_oracle"`
   - `claim_boundary == "third_person_demo_proxy_not_human_demo"`
   - `failure_packet.failure_predicate == "approach_blocked"`
   - `failure_packet.wrong_factor == "approach_direction"`
   - `revision.operator == "approach_substitution"`
   - `revision.factor == "approach_direction"`
   - `retry.success == True`
4. `python scripts/stage0_summarize.py --samples ... --out_dir ...` produces
   `report.md` showing `delta_pp >= 90.0` (every episode goes 0→1 by
   construction; the gate is at ≥10pp so this is wide of the bar).
5. The episode JSON shape matches goal.md §"Episode Data Format" — verified
   by a snapshot-style test in `test_episode.py`.

Expected total runtime ≤ 90 s on CPU sim for 5 episodes.

## 14. Out-of-scope, captured for follow-on stages

- Actual physical obstacles in the env (would require a custom PushCube
  subclass). Stage 0 fakes the block via `blocked_sides`.
- Multi-camera RGB recording for the demo proxy.
- More than one failure predicate per episode (Stage 0 always fires
  `approach_blocked`).
- Other wrong_factor branches in the reviser. `contact_region` revision is
  the next obvious extension.
- Per-seed perturbation diversity. The block is deterministic given the
  initial intent's approach_direction.
- Baselines (full replanning, failure-agnostic retry). Their hook points
  exist in `episode.py` but no implementation yet.

## 15. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Block-via-flag feels too synthetic for a paper claim. | The episode record is honest: the failure is *labeled* `approach_blocked`, not faked as physical contact. Future stages add a real obstacle in a PushCube subclass; Stage 0 explicitly captures the limitation in §14 and `claim_boundary`. |
| `contact_region` and `approach_direction` look redundant for PushCube. | They diverge in PickCube and drawer tasks. Keeping both at Stage 0 means the schema doesn't change between tasks. |
| Open-loop push fails for some seeds even when correct. | Pick4Pass's calibration (`PUSH_TRAVEL_SCALE = 0.6`) is reused as-is; if a seed misses, the summarizer reports it as `initial_success && retry_success` both False, not as a silent error. |
| Pick4Pass schema diverges from goal.md (push_heading vs object_motion). | Stage 0 deliberately doesn't reuse Pick4Pass's `PushIntent`/`PushFailureTrace` types — only its JSONL/summarizer/non-regression *patterns*. The schema is goal.md's. |
