# BABYSTEPS Vertical Slice on PushCube-v1 — Design

**Date:** 2026-05-15
**Status:** Approved by user (sections 1–7); ready for implementation plan.

---

## 1. Purpose

Build the smallest end-to-end loop that validates the BABYSTEPS *control-flow interfaces*
(intent → skill → failure → revision → retry) on ManiSkill's `PushCube-v1` task, with
**no VLM and no DINO perception yet**. Both are deferred to subsequent milestones.

The slice is considered a success iff, with a deliberately seeded-wrong `interaction.contact_site`
on attempt 1, the loop detects a `direction_error`, attributes it to `contact_site`, flips the
factor, and succeeds on attempt 2.

This proves the *interfaces* between modules are right before the heavy perception/VLM
machinery is plugged in.

## 2. Non-goals

- No VLM intent proposal — intent for attempt 1 is hardcoded.
- No DINO/DINOv2 perception — no RGB processing; pose/state read from sim directly.
- No diffusion counterfactuals — failure attribution is rule-based.
- No multi-task generalization yet — PushCube-v1 only.
- No closed-loop control — open-loop scripted push with three waypoints.
- No headless RGB rendering — `obs_mode="state_dict"` defers the Vulkan/EGL question.

## 3. Architectural pivot from current `CLAUDE.md`

The project `CLAUDE.md` (as of 2026-05-14) describes the VLM as the source of object
localization. The user has clarified the actual architecture is:

```
DINO/DINOv2 — parts, dense correspondences, contact regions, motion cues, cross-view grounding
    ↓ (grounded visual evidence)
VLM         — symbolize grounded evidence into structured intent factors
    ↓ (intent JSON)
Skill compiler → execution → failure detector → reviser
```

This slice does not exercise the DINO or VLM layers. They will appear in subsequent
milestones, and `CLAUDE.md` should be updated to reflect this layering post-slice.

## 4. Module layout

```
babysteps/
├── conda.yaml
├── pyproject.toml
├── CLAUDE.md                              # existing; will be updated post-slice
├── docs/superpowers/specs/                # design docs
│
├── babysteps/
│   ├── __init__.py
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── intent_schema.py               # Intent dataclass + JSON I/O
│   │   └── failure_schema.py              # Failure dataclass + JSON I/O
│   ├── envs/
│   │   ├── __init__.py
│   │   └── maniskill_wrapper.py           # PushCube-v1 wrapper; exposes scene info
│   ├── skills/
│   │   ├── __init__.py
│   │   ├── base_skill.py                  # abstract Skill interface
│   │   ├── push.py                        # PushSkill: contact pose + scripted motion
│   │   └── skill_compiler.py              # compile_intent_to_skill(intent, scene)
│   ├── execution/
│   │   ├── __init__.py
│   │   └── rollout.py                     # Rollout dataclass + run_skill helper
│   ├── failure/
│   │   ├── __init__.py
│   │   ├── failure_detector.py            # detect_failure(rollout, intent, scene)
│   │   └── failure_rules.py               # failure_type → candidate factors table
│   └── revision/
│       ├── __init__.py
│       └── revise_intent.py               # revise_intent(intent, failure)
│
├── scripts/
│   ├── smoke_test.py                      # bare-bones ManiSkill import + reset check
│   └── run_episode.py                     # full loop CLI entry
│
└── tests/
    └── test_slice_pushcube.py             # one integration test
```

**Boundary rule:** `envs/maniskill_wrapper.py` is the only module that imports `mani_skill`.
The rest of the codebase sees a plain `Scene` view (object poses, site poses, table extent,
robot state). Keeps failure/revision logic sim-agnostic.

**Boundary rule:** `failure/failure_rules.py` is separate from `failure_detector.py` even though
small. The rule table is a research artifact — it will grow, be ablated, eventually be replaced
by diffusion scoring. Isolating it makes that transition cleaner.

## 5. Data contracts

### 5.1 Intent

```python
# babysteps/schemas/intent_schema.py
from dataclasses import dataclass, field
from typing import Literal

@dataclass
class Goal:
    predicate: str           # "at_position", "left_of", "inside", ...
    args: list[str]          # e.g. ["cube", "target_site"]

@dataclass
class Interaction:
    mode: Literal["push", "pull", "grasp", "place"]
    contact_site: str        # "minus_x_face" | "plus_x_face" | "minus_y_face"
                             # | "plus_y_face" | "top_face"

@dataclass
class Intent:
    target: str
    goal: Goal
    interaction: Interaction
    reference_frame: Literal["world", "object_centric", "camera"] = "world"
    constraints: list[str] = field(default_factory=list)

    def to_json(self) -> dict: ...
    @classmethod
    def from_json(cls, d: dict) -> "Intent": ...
```

**Decisions:**

- Stdlib `@dataclass`, not pydantic. Add validation later if VLM output requires it.
- `contact_site` is a controlled vocabulary of face names. Strings are friendly for future
  VLM output; the push skill resolves face name → 3D direction vector internally.
- All factors from CLAUDE.md §6 are present, even those the slice does not exercise
  (`reference_frame`, `constraints`). Stable JSON shape across milestones.
- No task-specific fields (per CLAUDE.md §6 risk note). Same Intent type for every task.

**Initial seeded-wrong intent for the slice.** PushCube-v1 randomizes cube and goal
positions per seed, so a hardcoded `contact_site` would be accidentally correct for some
seeds. Instead, the seeded-wrong intent is built dynamically: read the scene, compute
the *correct* face from the goal direction relative to the cube, then flip it. This
guarantees attempt 1 demonstrably fails for any seed and isolates the slice from env
randomization.

```python
def seeded_wrong_intent(scene: Scene) -> Intent:
    cube_p = scene.object_pose("cube").translation
    goal_p = scene.site_pose("target_site").translation
    correct_face = direction_to_face(unit(goal_p - cube_p))   # e.g. "minus_x_face"
    wrong_face   = OPPOSITE_FACE[correct_face]                # e.g. "plus_x_face"
    return Intent(
        target="cube",
        goal=Goal(predicate="at_position", args=["cube", "target_site"]),
        interaction=Interaction(mode="push", contact_site=wrong_face),
        reference_frame="world",
        constraints=[],
    )
```

`direction_to_face` snaps a unit vector to the nearest of the four lateral face names.
For PushCube the goal direction is always roughly in the xy-plane, so the snap is
unambiguous.

**Correct intent (after revision):** identical, with `contact_site = OPPOSITE_FACE[wrong_face]`.

### 5.2 Failure

```python
# babysteps/schemas/failure_schema.py
from dataclasses import dataclass
from typing import Literal

FailureType = Literal[
    "none",
    "contact_failure",
    "direction_error",
    "no_motion",
    "wrong_object_moved",
    "goal_not_satisfied",
    "constraint_violation",
]

@dataclass
class Failure:
    failure_type: FailureType
    goal_satisfied: bool
    contact_established: bool
    object_moved: bool
    wrong_object_moved: bool
    constraint_violated: bool
    # Numerical evidence — kept for ablations and future diffusion scoring
    object_displacement: float | None = None
    direction_alignment: float | None = None   # cos(motion_vec, goal_vec) in [-1, 1]

    def to_json(self) -> dict: ...
    @classmethod
    def from_json(cls, d: dict) -> "Failure": ...
```

**Decisions:**

- Bools drive rule-based attribution. Numerical fields are evidence for later analysis.
- `failure_type` is *derived* from the bools by a small classifier in the detector
  (precedence rules below), not stored independently. Guarantees consistency.
- `wrong_object_moved` and `constraint_violated` are always emitted, even as `False`,
  for stability across tasks.

## 6. Control flow

```python
# scripts/run_episode.py — sketch
def main():
    env = ManiSkillWrapper("PushCube-v1", seed=SEED)
    initial_scene = env.observe_scene()
    intent = seeded_wrong_intent(initial_scene)    # see §5.1

    for attempt in range(MAX_ATTEMPTS):
        scene   = env.observe_scene()
        skill   = compile_intent_to_skill(intent, scene)
        rollout = run_skill(env, skill)
        failure = detect_failure(rollout, intent, scene)

        log_attempt(attempt, intent, failure)

        if failure.failure_type == "none":
            break
        intent = revise_intent(intent, failure)
        env.reset(seed=SEED)         # identical initial state each attempt
```

`MAX_ATTEMPTS = 3`. No business logic in the script — purely orchestration.

## 7. Failure detection

```python
def detect_failure(rollout, intent, scene) -> Failure:
    p0 = rollout.initial_pose(intent.target).translation
    pf = rollout.final_pose(intent.target).translation
    pg = scene.site_pose("target_site").translation

    motion = pf - p0
    goal_vec = pg - p0

    displacement = norm(motion)
    direction_alignment = (
        cos_sim(motion, goal_vec) if displacement > EPS else 0.0
    )

    contact_established = any_contact(rollout, intent.target)
    object_moved        = displacement > MOVE_THRESHOLD          # 5 mm
    goal_satisfied      = bool(rollout.final_info["success"])    # ManiSkill flag

    # Precedence: most specific first
    if goal_satisfied:                       ftype = "none"
    elif not contact_established:            ftype = "contact_failure"
    elif not object_moved:                   ftype = "no_motion"
    elif direction_alignment < 0:            ftype = "direction_error"
    else:                                    ftype = "goal_not_satisfied"

    return Failure(
        failure_type=ftype,
        goal_satisfied=goal_satisfied,
        contact_established=contact_established,
        object_moved=object_moved,
        wrong_object_moved=False,            # N/A for PushCube
        constraint_violated=False,           # N/A for slice
        object_displacement=displacement,
        direction_alignment=direction_alignment,
    )
```

Thresholds (`EPS`, `MOVE_THRESHOLD`) are module-level constants, tunable.

## 8. Revision

```python
# babysteps/failure/failure_rules.py
FAILURE_TO_FACTOR: dict[str, list[str]] = {
    "direction_error":      ["interaction.contact_site", "reference_frame", "goal"],
    "contact_failure":      ["interaction.contact_site", "interaction.mode"],
    "no_motion":            ["interaction.mode", "interaction.contact_site"],
    "wrong_object_moved":   ["target"],
    "goal_not_satisfied":   ["goal"],
    "constraint_violation": ["constraints"],
}

OPPOSITE_FACE = {
    "plus_x_face":  "minus_x_face",
    "minus_x_face": "plus_x_face",
    "plus_y_face":  "minus_y_face",
    "minus_y_face": "plus_y_face",
    # top_face has no slice-level opposite; leave out for now
}
```

```python
# babysteps/revision/revise_intent.py
def revise_intent(intent: Intent, failure: Failure) -> Intent:
    candidates = FAILURE_TO_FACTOR[failure.failure_type]
    factor = candidates[0]                   # slice: take first candidate
    return _apply_flip(intent, factor)

def _apply_flip(intent: Intent, factor: str) -> Intent:
    if factor == "interaction.contact_site":
        return replace(intent,
            interaction=replace(intent.interaction,
                contact_site=OPPOSITE_FACE[intent.interaction.contact_site]),
        )
    raise NotImplementedError(f"flip for {factor} not implemented in slice")
```

Other factors raise `NotImplementedError`. Honest about what is and isn't validated.

## 9. PushCube specifics

### 9.1 Env construction

```python
env = gym.make(
    "PushCube-v1",
    obs_mode="state_dict",                   # no RGB → no Vulkan dependency
    control_mode="pd_ee_delta_pose",
    sim_backend="cpu",                       # CPU sim for first run
)
```

CPU sim is intentional. GPU sim and RGB rendering come in later milestones with DINO.

### 9.2 Push skill

Open-loop scripted, three waypoints, using `pd_ee_delta_pose`:

```python
class PushSkill(Skill):
    def __init__(self, intent: Intent, scene: Scene):
        self.face_dir = face_to_direction(intent.interaction.contact_site)   # e.g. (+1,0,0)
        self.cube_p   = scene.object_pose(intent.target).translation
        self.goal_p   = scene.site_pose("target_site").translation

    def execute(self, env) -> Rollout:
        approach = self.cube_p + self.face_dir * STANDOFF + (0, 0, HOVER_HEIGHT)
        contact  = self.cube_p + self.face_dir * STANDOFF
        push_to  = contact - self.face_dir * PUSH_DISTANCE   # past cube center
        move_to(env, approach)
        move_to(env, contact)
        push_to_pos(env, push_to)
        return Rollout(...)
```

Semantics: `contact_site` is the *cube face* the EE touches. Push direction is *into the
cube* (away from that face). So `plus_x_face` → push in −x. If goal is in +x,
`direction_alignment ≈ −1`.

Constants (tunable): `STANDOFF ≈ 0.05 m`, `HOVER_HEIGHT ≈ 0.05 m`, `PUSH_DISTANCE ≈ 0.20 m`.

## 10. Environment / Gilbreth setup

```yaml
# conda.yaml
name: babysteps
channels: [conda-forge]
dependencies:
  - python=3.11
  - pip
  - pip:
    - mani_skill==3.0.0b22
    - torch==2.4.0
    - "numpy<2"
    - gymnasium
    - pytest
    - pyyaml
```

**Hardware:** A100 40GB on Gilbreth. CPU sim does not use the GPU, but Torch will detect
CUDA 12.x. Pin `torch==2.4.0` (CUDA 12.1 build) and verify with a tiny CUDA check.

**Smoke test before anything else:**

```bash
python scripts/smoke_test.py
```

which does:

```python
import gymnasium as gym
import mani_skill
env = gym.make("PushCube-v1", obs_mode="state_dict",
               control_mode="pd_ee_delta_pose", sim_backend="cpu")
obs, info = env.reset(seed=0)
print("obs keys:", list(obs.keys()))
print("OK")
```

If this fails we stop and fix the env before writing any other code.

## 11. Validation criteria

The slice is complete iff all of these pass:

1. `python scripts/smoke_test.py` exits 0 and prints expected obs keys.
2. `python scripts/run_episode.py` with the seeded-wrong intent (built dynamically
   per §5.1 so attempt 1 is wrong for any seed):
   - Attempt 1 emits `failure_type == "direction_error"`, `direction_alignment < 0`.
   - Reviser flips `contact_site` to `OPPOSITE_FACE[wrong_face]` (the correct face).
   - Attempt 2 emits `failure_type == "none"`, `goal_satisfied == True`.
3. `pytest tests/test_slice_pushcube.py` passes the above as assertions.

Expected runtime: ≤ 60 s end-to-end on CPU sim.

## 12. What this slice deliberately does NOT decide

- The DINO/DINOv2 perception module shape (which features, which views, which time steps).
- The VLM prompt format and JSON-output handling.
- The diffusion counterfactual scorer.
- Multi-camera setup in ManiSkill.
- GPU sim / parallelized rollouts.
- Whether `Intent` will eventually move to pydantic.
- Whether failure detection will become learned rather than rule-based.

Each of the above is a follow-up design conversation, not a blocker for this slice.
