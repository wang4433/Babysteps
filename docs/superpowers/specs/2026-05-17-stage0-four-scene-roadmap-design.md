# Stage-0 Four-Scene Roadmap — Design Spec

**Sub-projects B, C, D** of the Stage-0 multi-task extension.
Sub-project A (TaskAdapter refactor) is closed; this spec scopes the next three
adapters and the schema/operator additions they require. Implementation order:
B (full) → C (full) → D (full). C and D are scoped here so the schema and
operator surface area is decided once.

- Date: 2026-05-17
- Predecessor spec: `2026-05-16-stage0-task-adapter-refactor-design.md`
- Goal-of-record: `goal.md` (Stage-0 boundary; object-centric intent schema)

## 1. Motivation

Sub-project A made the loop sim-agnostic but Stage-0 still proves only **one**
of the six BABYSTEPS intent factors (`approach_direction`). The paper-facing
claim is that BABYSTEPS revises *one* of six factors surgically; with one
scene that claim is half-empty. The four-scene set exercises a distinct
factor per scene, so the headline becomes "across four ManiSkill tasks each
probing a different intent factor, BABYSTEPS revises only the implicated
factor and preserves the other five."

| Scene | Intent factor probed | Stage-0 controlled failure |
|---|---|---|
| PushCube-v1 ✅ (A) | `approach_direction` | the demonstrated approach side is blocked |
| PickCube-v1 (B) | `contact_region` (+ `embodiment_mapping`) | the demonstrated grasp face is blocked |
| StackCube-v1 (C) | `goal_state` | initial goal is under-specified (translate-only) |
| OpenCabinetDrawer-v1 (D) | `constraint_region` (+ `contact_region`) | demonstrated contact is the cabinet body, not the handle |

The scenes were proposed by the user; this spec only commits to how each
slots into the existing adapter abstraction.

## 2. Non-Goals

- **No new revision algorithm.** Stage-0 revisions stay rule-based; learned
  attribution is Stage-3 (per `goal.md`).
- **No re-architecture of `BaseTaskAdapter`.** All four scenes fit the
  six-method interface from A.
- **No multi-attempt loops.** Stage-0 stays one-attempt-then-one-retry per
  episode; multi-revision retries are deferred.
- **No new `claim_boundary` strings.** All four scenes are proxy demos.
- **No CLI multi-task batching.** `--task` flag arrives as part of B; C and D
  reuse it.
- **No render-script generalization beyond what each scene strictly needs.**
  PickCube reuses the three-phase MP4 (demo / blocked / retry); StackCube
  and Drawer scripts are scoped but not built in this roadmap.

## 3. Acceptance Gate

The roadmap is "done" when sub-project B is shipped under the gate below.
C and D each have their own per-sub-project gate (mirroring this).

**B's acceptance gate:**

1. All pre-B tests (118+) still pass byte-identical.
2. New tests: PickCubeAdapter parity + PickSkill geometry + new
   failure/revision branches (~30 tests).
3. Snapshot test: `tests/snapshots/pickcube_samples_seeds_0_4.jsonl` exists
   and is byte-stable under repeated `scripts/stage0_collect.py --fake-env
   --task PickCube-v1 --n_episodes 5 --seed_start 0`.
4. GPU visual spot-check: `render_stage0_maniskill.py --task PickCube-v1
   --n_episodes 2` produces three MP4s per episode where (i) the demo
   top-grasps and lifts, (ii) the blocked attempt fails the top-grasp
   (gripper closes on a blocked face), (iii) the retry succeeds via the
   revised side-grasp.
5. The `report.md` summarizer reports `delta_pp >= 10` between retry
   success rate and initial-attempt success rate on PickCube (matching
   PushCube's M-BABY-1 bar).

## 4. Schema deltas (shared across B/C/D)

All deltas land in `babysteps/schemas.py`. None of them invalidate
PushCube records — PushCube's whitelisted values are a strict subset.
The pre-existing `tests/snapshots/pushcube_samples_seeds_0_4.jsonl`
remains byte-identical (verified by the same snapshot test).

```python
# new whitelist entries (annotated by sub-project)
GOAL_STATES         += {"cube_lifted_at_target"        # B
                       , "cubeA_on_cubeB"              # C
                       , "drawer_open"                 # D
                       }
OBJECT_MOTIONS      += {"lift_up"                      # B
                       , "place_on"                    # C
                       , "pull_open"                   # D
                       }
CONTACT_REGIONS     += {"drawer_handle"                # D
                       }   # B reuses 4 cardinal faces — see §7.1
APPROACH_DIRECTIONS += {}   # "from_above" already in the whitelist
EMBODIMENT_MAPPINGS += {"proxy_contact_to_franka_grasp"          # B
                       , "proxy_contact_to_franka_pick_and_place" # C
                       , "proxy_contact_to_franka_pull"           # D
                       }
CONSTRAINT_REGIONS  += {"cabinet_body_static"          # D
                       }
FAILURE_PREDICATES  += {"grasp_slip"                   # B (already an
                                                      #    AttemptResult
                                                      #    field — promoted
                                                      #    to predicate)
                       , "constraint_violation"        # D
                       }
REVISION_OPERATORS  += {"contact_substitution"         # B
                       , "goal_refinement"             # C
                       , "constraint_introduction"     # D
                       }
```

**Note on PickCube's CONTACT_REGIONS reuse.** A parallel-jaw gripper grasping
from above does not contact the cube's *top* face — the two fingertips touch
two opposite side faces. So PickCube's `contact_region` describes which
gripper-axis principal face is contacted: `"minus_x_face"` means
*x-axis-aligned gripper* (fingertips on ±x faces); `"minus_y_face"` means
*y-axis-aligned gripper* (fingertips on ±y faces). This reuses the existing
four-face vocabulary without inventing `"top_face"`, which would not be a
physically meaningful contact for parallel jaws.

**Privileged-firewall preservation.** None of the new tokens leak privileged
state; they are categorical labels on the same shapes already validated.
The `SceneState.extra` dict (added in A) absorbs per-scene privileged
payloads (StackCube's second cube pose, Drawer's articulation joint id).

## 5. Failure attribution deltas

`babysteps/failure.py::FAILURE_TO_FACTOR` grows from 5 to 7 entries:

```python
FAILURE_TO_FACTOR = {
    # pre-existing (PushCube)
    "approach_blocked":      ("approach_direction", ("approach_direction", "contact_region")),
    "direction_error":       ("approach_direction", ("approach_direction",)),
    "contact_failure":       ("contact_region",     ("contact_region",)),
    "no_motion":             ("approach_direction", ("approach_direction", "contact_region")),
    "goal_not_satisfied":    ("goal_state",         ("goal_state",)),
    # NEW
    "grasp_slip":            ("contact_region",     ("contact_region", "embodiment_mapping")),     # B
    "constraint_violation":  ("constraint_region",  ("constraint_region", "contact_region")),     # D
}
```

`build_failure_packet` predicate-precedence rule extends:
`success → planner_failed → constraint_violation (D) → grasp_slip (B) →
contact → motion → direction → goal`.

The two NEW predicates intentionally beat `contact_failure` because their
evidence is more specific.

## 6. Revision operator deltas

`babysteps/revision.py` grows three operators in addition to the existing
`approach_substitution`. Each new operator preserves the factor-local
invariant: it touches exactly one intent field.

### 6.1 `contact_substitution` (B)
Trigger: `attribution.wrong_factor == "contact_region"`.
- If demo's contact is a side face and that face is in
  `scene.blocked_sides`, swap to `top_face` (and vice versa).
- Embodiment_mapping is also revised when the new contact requires a
  different motor primitive (`grasp` ↔ `push`). This means the operator
  may touch **two** factors when both are flagged by `attribution.revise`.
  This is still factor-local because both belong to the predicate's
  declared `revise` tuple; the audit counts factors in `revise` as
  "expected to change," and the summarizer's `non_regression_score`
  remains the metric.
- Tests: every (contact, blocked) pair has a deterministic new value.

### 6.2 `goal_refinement` (C)
Trigger: `attribution.wrong_factor == "goal_state"`.
- Refinement is a *strict-extension* edit: `"cube_at_target"` →
  `"cubeA_on_cubeB"`. The operator's intent is "the goal was
  under-specified; sharpen it." Not an arbitrary goal swap.
- Stage-0 supports only this single refinement; other goal_state
  transitions raise `NotImplementedError` (honest about coverage,
  matching the pattern from `approach_substitution`).

### 6.3 `constraint_introduction` (D)
Trigger: `attribution.wrong_factor == "constraint_region"`.
- Adds a constraint where the demo specified `"none"`:
  `"none"` → `"cabinet_body_static"`. This narrows the search.
- The downstream skill compiler is expected to honor the new
  constraint by rejecting handles that would violate it (rule-coded
  for Stage-0; learned attribution is Stage-3).

## 7. Per-scene specs

### 7.1 PickCube-v1 (Sub-project B) — full implementation

**Failure narrative.**

```
Demo (third-person proxy, no blockers):
  Top-down grasp with x-axis-aligned gripper. Lift cube to goal.

Executor scene (blocked):
  contact_region "minus_x_face" is flagged slip-prone (the cube has a
  narrow / smooth dimension along x in this seed, so an x-aligned
  pinch loses grip during lift). scene.blocked_sides =
  ("minus_x_face",).

Initial intent (from demo):
  goal_state         = "cube_lifted_at_target"
  object_motion      = "lift_up"
  contact_region     = "minus_x_face"     ← x-axis-aligned gripper
  approach_direction = "from_above"
  constraint_region  = "none"
  embodiment_mapping = "proxy_contact_to_franka_grasp"

Attempt 1:
  Gripper descends, closes on cube. Lift begins. Privileged executor
  flag triggers slip — env_runner sets grasp_slip=True, success=False,
  final cube z back near table z.
  AttemptResult(reached_contact=True, object_moved=False (≈),
                grasp_slip=True, planner_failed=False)
  failure_predicate = "grasp_slip"   (new, beats contact_failure)

Attribution: wrong_factor = "contact_region", revise =
("contact_region", "embodiment_mapping").

Reviser (contact_substitution): rotate gripper axis 90° around z by
swapping contact_region "minus_x_face" → "minus_y_face" (the first
unblocked cardinal face). embodiment_mapping unchanged for Stage-0
(both faces use the same parallel-jaw grasp primitive).

Retry: y-axis-aligned grip holds → lift succeeds → goal reached.
```

**Why this variant probes `contact_region`.**
- The factor that changes is `contact_region` (a NEW factor for
  BABYSTEPS coverage, distinct from PushCube's `approach_direction`).
- `approach_direction` stays `"from_above"` throughout (top-down).
- The semantic is "this gripper axis is slip-prone for this cube
  geometry; try a perpendicular axis."

**oracle_correct_intent.**
- For PickCube the success goal is a lifted cube at the goal xyz. The
  oracle picks the contact_region that is NOT in `scene.blocked_sides`.
  If no contact is blocked (demo scene), default to `"minus_x_face"`.

**default_blocked_factory.**
- Returns `(intent.contact_region,)` — block the contact that the
  demo used. Mirrors PushCube's "block the approach the demo used."

**oracle_wrong_factor.**
- `"contact_region"` if `intent.contact_region in scene.blocked_sides`,
  else `"none"`.

**Skill: PickSkill.**
- Waypoints (4, 7): pre_contact_high (z=travel_z, xy=cube_xy),
  descend (z=cube_z + small clearance, xy=cube_xy), grasp_close
  (z=cube_z, xy=cube_xy — close gripper), lift (z=travel_z,
  xy=goal_xy). Each waypoint also carries the gripper quaternion
  oriented for the chosen `contact_region` (x-axis → identity-quat;
  y-axis → 90°-rotated-around-z quat).
- Always returns a skill (does NOT return None) — slip is detected by
  the env_runner at lift time. This contrasts with PushSkill, which
  returns None for blocked approach. The difference is intentional:
  PushCube's failure is *pre-execution* (the approach is geometrically
  infeasible); PickCube's failure is *during execution* (the gripper
  loses grip).

**env_runner.** `PickCubeEnvRunner` mirrors `PushCubeEnvRunner` but:
1. Steps a 4-phase loop (approach, descend, grasp_close, lift).
2. Sends gripper-open during approach/descend and gripper-close during
   grasp_close/lift.
3. After the grasp_close phase, checks
   `intent.contact_region in scene.blocked_sides`. If True, lift is
   force-failed: the env still steps but at lift completion the runner
   reports `grasp_slip=True, success=False` regardless of the
   simulator's outcome (Stage-0 controlled-failure mechanism, mirrors
   PushCube's `planner_failed` controlled-failure).

### 7.2 StackCube-v1 (Sub-project C) — scoped only

**Failure narrative.**

```
Demo: pick cubeA, place onto cubeB.
Executor scene: same (no physical blocker).

Initial intent (from demo summarization — Stage-0 quirk):
  goal_state         = "cube_at_target"     ← UNDER-SPECIFIED!
  object_motion      = "translate_+x"
  contact_region     = "top_face"
  approach_direction = "from_above"
  constraint_region  = "none"
  embodiment_mapping = "proxy_contact_to_franka_pick_and_place"

Attempt 1: cube translated to (cubeB.xy) but not placed atop → cube on
table next to cubeB. PickCube success predicate flags "not stacked."
  failure_predicate = "goal_not_satisfied"

Attribution: wrong_factor = "goal_state".
Reviser: "cube_at_target" → "cubeA_on_cubeB" (goal_refinement).
Skill compiler responds to the new goal by extending the trajectory
with a "place_on" phase. Retry: stack succeeds.
```

**Why this works.** It's the only one of the four scenes where the
*demo* under-specifies the goal (the third-person proxy sees a place,
but the scripted summarizer only labels it as a translation). This is
the most novel BABYSTEPS demonstration — a failure causes us to
*sharpen* the goal, not just swap a side.

**Schema additions:** `goal_state += "cubeA_on_cubeB"`, `object_motion
+= "place_on"`, `embodiment_mapping +=
"proxy_contact_to_franka_pick_and_place"`.

**SceneState.extra:** `{"cubeB_xy": (x, y), "cubeB_z": z}`.

**Implementation deferred** — surface area is committed by Section 4-6
of this doc.

### 7.3 OpenCabinetDrawer-v1 (Sub-project D) — scoped only

**Failure narrative.**

```
Demo: pull on drawer handle, drawer opens.
Executor scene: ambiguous — the cabinet body and the handle are both
visible contact regions. The scripted summarizer picks the wrong one.

Initial intent:
  goal_state         = "drawer_open"
  object_motion      = "pull_open"
  contact_region     = "cabinet_body"       ← WRONG
  approach_direction = "from_minus_x"
  constraint_region  = "none"               ← MISSING
  embodiment_mapping = "proxy_contact_to_franka_pull"

Attempt 1: pulls on cabinet body → cabinet slides on table (or robot
collides with rigid body) → no_motion + constraint_violation.
  failure_predicate = "constraint_violation"  (new, beats no_motion)

Attribution: wrong_factor = "constraint_region", revise =
("constraint_region", "contact_region").
Reviser: introduces constraint `cabinet_body_static` AND swaps
contact_region cabinet_body → drawer_handle. Skill compiler routes
to the handle and respects the constraint. Retry: drawer opens.
```

**Why this is the right scene for `constraint_region`.** It's the only
scene where the demo's third-person view legitimately ambiguates a
constraint that the executor needs to know. The constraint isn't a
physics flag — it's a semantic one ("don't push the whole cabinet").

**Schema additions:** `contact_region += "drawer_handle"` (and possibly
`cabinet_body` if we choose to whitelist the wrong value rather than
encode it as "drawer_handle" with a flag), `object_motion +=
"pull_open"`, `goal_state += "drawer_open"`, `constraint_region +=
"cabinet_body_static"`, `failure_predicates += "constraint_violation"`,
`embodiment_mapping += "proxy_contact_to_franka_pull"`.

**Implementation deferred** — surface area is committed by Section 4-6
of this doc. New env_runner uses ArticulatedObject; otherwise mirrors
PushCubeEnvRunner.

## 8. CLI / script generalization (per sub-project)

`scripts/stage0_collect.py` gains a `--task` flag:
```bash
--task {PushCube-v1, PickCube-v1, StackCube-v1, OpenCabinetDrawer-v1}
```
The dispatch is a simple `{name: AdapterClass}` map in
`babysteps/envs/task_registry.py` (new file, three lines per scene).
`render_stage0_maniskill.py` receives the same flag in B and is wired
per-scene as each sub-project lands.

## 9. Risks & open questions

1. **B's `grasp_slip` predicate is privileged.** In Stage-0 the env_runner
   knows the cube is slip-prone because the executor scene marks it so;
   in later stages a learned grasp-quality estimator replaces this. The
   privileged-firewall is not crossed — `grasp_slip` is detected in the
   *executor* path, not the demo-to-intent path.
2. **C and D both touch `embodiment_mapping`.** This is correct: each
   scene needs its own motor primitive. The `attribution.revise` tuple
   includes both `contact_region` and `embodiment_mapping` for these
   predicates, so the audit treats both as expected-to-change.
3. **B-app vs B-con (Section 7.1).** B-con is the recommended variant
   because B-app collapses onto PushCube's `approach_direction` path
   and adds no new factor coverage. If the user prefers B-app for
   simulation simplicity, the scope shrinks back to a near-clone of A.
4. **`render_stage0_maniskill.py` per-scene.** Each scene's three-phase
   MP4 has a different visual signature (PushCube: arm enters from
   opposite side; PickCube: grasp face changes; StackCube: trajectory
   gets a stack phase appended; Drawer: arm moves from cabinet body
   to handle). The render script's `_build_waypoints` becomes a
   `dispatch(adapter)` call.

## 10. Plan files

This spec drives three plan files (created at the start of each
sub-project, not all at once):

- `docs/superpowers/plans/2026-05-17-stage0-pickcube-b-plan.md` (next)
- `docs/superpowers/plans/2026-MM-DD-stage0-stackcube-c-plan.md` (after B)
- `docs/superpowers/plans/2026-MM-DD-stage0-drawer-d-plan.md` (after C)

## 11. Summary

The four-scene roadmap commits to a tight set of schema additions, two
new failure predicates, and three new revision operators. Each addition
is justified by a different intent factor. Implementation starts with
PickCube (Sub-project B, variant B-con). C and D are scoped here so the
schema surface area is decided once, but built only after B is green.
