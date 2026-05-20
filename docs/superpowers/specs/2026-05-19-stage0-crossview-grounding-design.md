# Stage-0 Cross-View Grounding — Design Spec

> **Status:** design approved 2026-05-19, pending written-spec review.
> **Claim:** *Failure-guided correction of cross-view imitation grounding.*
> A robot (B) observes another robot (A) succeed from a **different
> viewpoint**, infers a latent imitation target, executes from its **own**
> view, fails because the **direction/view grounding** was misinterpreted,
> and BABYSTEPS revises **only** that factor and retries.
>
> Source of the research framing: `/home/wang4433/scratch/babysteps/update.md`.
> Milestone-1 locked claim: `docs/milestone1_locked_claim.md`.
> This spec is the first concrete realization of the "failure-and-recovery"
> cross-view story chosen during brainstorming.

---

## 1. Scope & Non-Goals

**In scope (this spec, "minimal validation cut"):**

- One task family: **left/right placement**, realized on the existing
  `PushCube-v1` gym env (cube must reach a world goal).
- One new additive intent factor: `direction_grounding`.
- One new revision operator: `grounding_substitution`.
- One new adapter: `CrossViewPushAdapter` (registry key `CrossViewPush-v1`),
  reusing `PushCubeEnvRunner` physics unchanged.
- An **observer camera** at rotated poses for the demo MP4.
- 1 task × 2–3 observer rotations × ~20–30 seeds of collected data.

**Explicitly NOT in scope:**

- Simultaneous two-robot scenes (Robot B does **not** watch Robot A in real
  time — the demo is recorded/temporal; "observer ≠ actor" holds via
  viewpoint, "two robots" holds via separate episodes).
- Real perception (DINO/VLM). Stage-0 intent is **scripted/label-driven**;
  the frame mismatch is **injected** like blocked-approach already is.
- Cross-embodiment. Same arm (Panda) for demo and execution.
- The other two task families from `update.md` (push-to-reveal,
  contact-region) — deferred until this loop is proven.
- Scaling to `update.md`'s 500–2000 episodes — deferred.

---

## 2. The Single-Factor Mechanism (load-bearing)

The entire BABYSTEPS claim requires that the revision change **exactly one**
intent factor (see memory `feedback_single_factor_revision_invariant` and
`goal.md §Core Research Invariant`). The mechanism:

1. The demo is captured from an **observer camera** rotated about the world
   z-axis by yaw `θ_obs ∈ {90°, 180°, 270°}` relative to the actor/robot
   canonical frame. `θ_obs` is privileged data carried in
   `SceneState.extra["observer_yaw_deg"]`.
2. Robot A's **true world** push moves the cube toward the world goal (e.g.,
   `translate_+x`). Seen through the rotated observer camera, that motion
   **appears** as a different observer-relative token (e.g., `θ_obs=180°` →
   appears as `translate_-x`).
3. `scripted_demo_to_intent` stores what B **observed** — the
   observer-relative directional content (`object_motion`, and the matching
   `contact_region` / `approach_direction`). These three stay **frozen**
   across the revision.
4. The new factor `direction_grounding` tells `compile_skill` *which
   transform* to apply when resolving the observer-relative content into a
   **world** push direction:
   - `actor_frame` (the egocentric **bug**): assume observer ≈ self → apply
     **identity** (`θ_applied = 0`) → wrong world direction when `θ_obs ≠ 0`.
   - `observer_frame` (the **fix**): apply the true `θ_obs` → correct world
     direction.
5. **Attempt 1** uses `direction_grounding = actor_frame` → cube pushed away
   from goal → `direction_error`.
6. **Revision** flips **only** `direction_grounding`
   (`actor_frame → observer_frame`). The resolved world direction changes as a
   *downstream compile-time consequence*; **no other stored factor is
   edited**. This is the one-factor edit.
7. **Retry** with the corrected grounding pushes toward the goal → success.

**Token rotation.** Resolving an observer-relative cardinal token through a
yaw multiple of 90° yields another cardinal token (no continuous geometry
needed at the token level):

| yaw applied | +x → | -x → | +y → | -y → |
|---|---|---|---|---|
| 0° (`actor_frame`) | +x | -x | +y | -y |
| 90° (CCW) | +y | -y | -x | +x |
| 180° | -x | +x | -y | +y |
| 270° | -y | +y | +x | -x |

`compile_skill` resolves the **world** push direction, then derives the world
contact face + approach from it via the existing `babysteps.envs.scene`
helpers, and finally builds geometry through
`compile_intent_to_push_skill` using the **resolved** face. The physical
waypoint geometry therefore tracks the resolved world direction, while the
*stored* intent differs between attempts only in `direction_grounding`.

---

## 3. Schema Changes (additive, snapshot-safe)

`babysteps/schemas.py`:

- New whitelist:
  ```python
  DIRECTION_GROUNDINGS: frozenset[str] = frozenset({
      "actor_frame", "observer_frame", "object_frame", "world_frame",
  })
  ```
  Stage-0 uses `actor_frame` (initial/bug) and `observer_frame` (fix);
  `object_frame` / `world_frame` are defined-but-reserved for later cuts.
- `Intent` gains a 7th field `direction_grounding: str = "world_frame"`
  (defaulted). `__post_init__` validates it against `DIRECTION_GROUNDINGS`.
- **Snapshot safety (per approved choice "Defaulted + omit-when-default"):**
  `INTENT_FIELDS` stays the six-tuple it is today. `Intent.to_dict()` emits
  `direction_grounding` **only when it differs from the default
  `"world_frame"`**. `Intent.from_dict()` reads it with the default when
  absent. Result: the four existing tasks (PushCube/PickCube/StackCube/
  TurnFaucet) — which never set `direction_grounding` — serialize
  byte-identically to their locked snapshots. Only `CrossViewPush` records
  carry the key.
  > Rationale: mirrors the existing `SceneState.extra`
  > "serialize-only-when-present" pattern and honors memory
  > `feedback_additive_schema_changes` (add tokens, don't disturb existing).
- New revision operator token: `REVISION_OPERATORS += {"grounding_substitution"}`.
- **No new failure predicate.** The cross-view failure is observably a
  `direction_error` (already in `FAILURE_PREDICATES`); attribution to the
  new factor happens at the adapter level (§5), leaving the shared
  `FAILURE_TO_FACTOR` table untouched.

---

## 4. Direction-Grounding Resolution Helper

New pure helper in `babysteps/envs/scene.py` (sim-agnostic, unit-tested):

```
def resolve_grounded_motion(observed_motion: str,
                            grounding: str,
                            observer_yaw_deg: int) -> str:
    """Map an observer-relative cardinal motion token to the world frame.
    grounding == 'actor_frame'   → apply 0°  (identity; the bug)
    grounding == 'observer_frame'→ apply observer_yaw_deg
    Returns a world-frame OBJECT_MOTIONS token."""
```

Implemented as the 90°-multiple token rotation in §2. Raises on
non-multiple-of-90 yaw (Stage-0 restricts observer rotations to the four
cardinal yaws). World contact face / approach are then obtained from the
existing `direction_to_face` / `face_to_approach` helpers applied to the
resolved world direction.

---

## 5. New Adapter: `CrossViewPushAdapter`

`babysteps/envs/crossview_adapter.py`, subclass of `BaseTaskAdapter`,
`task_id = "PushCube-v1"` (underlying gym env). Registry key
`CrossViewPush-v1`.

| Method | Behavior |
|---|---|
| `make_env_runner` | returns `PushCubeEnvRunner()` (reused unchanged) |
| `oracle_correct_intent(scene)` | the intent with `direction_grounding=observer_frame` and observer-relative directional content consistent with `scene.goal_xy` under `θ_obs` |
| `scripted_demo_to_intent(evidence)` | builds the **initial** intent: observer-relative `object_motion` from the (observer-frame) trajectory, matching `contact_region`/`approach_direction`, and `direction_grounding="actor_frame"` (the egocentric bug) |
| `default_blocked_factory(intent)` | `()` — the controlled failure is the frame bug, not a blocked side |
| `oracle_wrong_factor(initial_intent, scene_executor)` | `"direction_grounding"` when `scene.extra["observer_yaw_deg"] != 0` **and** `initial_intent.direction_grounding == "actor_frame"`; else `"none"` |
| `compile_skill(intent, scene)` | resolve world motion via `resolve_grounded_motion(intent.object_motion, intent.direction_grounding, scene.extra["observer_yaw_deg"])`; derive world face/approach; delegate to `compile_intent_to_push_skill` with the resolved face. Returns `None` only if resolution is impossible (never, for cardinal yaws). |
| `attribute_failure(fp)` **(override)** | if `fp.failure_predicate == "direction_error"` → `Attribution(wrong_factor="direction_grounding", revise=("direction_grounding",), freeze=<other 6>)`; otherwise delegate to the shared `failure_mod.attribute_failure` |
| `revise_intent(intent, attribution, scene)` **(override)** | if `attribution.wrong_factor == "direction_grounding"` → flip `actor_frame → observer_frame`, operator `grounding_substitution`, all other six factors frozen; otherwise delegate to shared `revision_mod.revise_intent` |

`SceneState.extra` carries `{"observer_yaw_deg": int}` (privileged; set at
reset). The firewall holds: `scripted_demo_to_intent` receives only
`DemoEvidence`, never `SceneState` — the observed trajectory in the evidence
is already expressed in the observer frame, so no privileged yaw leaks into
the intent-inference path. The yaw is consumed only by `compile_skill`
(resolution) and `oracle_wrong_factor` (metrics).

---

## 6. Reviser & Failure wiring

- `babysteps/revision.py` gains a `grounding_substitution` branch:
  `direction_grounding: actor_frame → observer_frame` only; other transitions
  raise `NotImplementedError` (honest about what is validated). Single-factor:
  `frozen_factors = INTENT_FIELDS` (the six) since `direction_grounding` is
  not in that tuple — i.e., the revised field is the new factor and every
  schema factor present in `INTENT_FIELDS` is frozen.
  > Note the asymmetry: because `INTENT_FIELDS` deliberately excludes the
  > defaulted 7th factor (§3), the audit of "exactly one changed" is done by
  > the adapter/Revision record, which names `direction_grounding` as the
  > revised factor. The summarizer's `non_regression_score` must be extended
  > to recognize `direction_grounding` as a tracked factor for CrossViewPush
  > records (see §8).
- The shared `FAILURE_TO_FACTOR` table is **not** modified. Cross-view
  attribution is adapter-local (§5), so PushCube's `direction_error →
  approach_direction` mapping is preserved.

---

## 7. Observer Camera & Rendering

`babysteps/render/crossview.py` (new), modeled on `render/pushcube.py`:

- **Phase 1 — demo:** render Robot A's successful (world-correct) push. The
  caption describes the object motion and notes the observer yaw `θ_obs` as
  metadata (per memory `feedback_demo_caption_no_motor_program`: object-evidence
  language, no Franka motor program).
- **Phase 2 — attempt:** render Robot B's first attempt (the `actor_frame`
  grounding); cube visibly moves the **wrong** way (a real failing push, not a
  held-still planner failure).
- **Phase 3 — retry:** render B's corrected (`observer_frame`) attempt; cube
  reaches the goal.
- View configs = the set of `θ_obs` rotations.

> **Implemented first-cut deviation (decided during execution, 2026-05-19):**
> All three phases render from PushCube's **default (world) camera**; the
> observer yaw is applied to the *grounding math* (`observe_demo` −yaw /
> `world_resolved_intent` +yaw), **not** to a physically rotated SAPIEN camera.
> The Phase-1 banner therefore reads "world camera, observer yaw=θ°", not
> "observer view". The cross-view-ness is carried by the data record
> (`observer_yaw_deg` + the `actor_frame → observer_frame` revision), which is
> the claim-bearing artifact; a literal rotated observer camera is a deferred
> visual enhancement. MP4 naming follows the shared harness's fixed phase keys:
> `crossview_grounding_seed_NNNN__{1_demo,2_attempt_blocked,3_retry}.mp4`
> (the `2_attempt_blocked` key is the harness's generic phase-2 label; nothing
> is "blocked" here — the failure is the frame mis-grounding, stated in the
> caption).

---

## 8. Data Collection, Metrics & Baseline

- **Collection:** `scripts/stage0_collect.py --task CrossViewPush-v1
  --n_episodes <N> --seed_start 0` (the `--task` dispatch already routes
  through `task_registry`; add the `CrossViewPush-v1` entry). Each episode
  iterates the configured `θ_obs` values; JSONL records carry the full
  demo/execution/failure_packet/revision/retry/metrics shape from
  `goal.md §Episode Data Format`, plus `direction_grounding` in the intents.
- **Metrics** (extend `scripts/stage0_summarize.py` /
  `babysteps/evaluation` as needed):
  - `initial_success_rate`, `retry_success_rate`, `delta_pp`
    (= retry − initial).
  - `direction_grounding_attribution_accuracy` (= fraction where the
    attributed `wrong_factor` matches `oracle_wrong_factor`).
  - `frozen_factor_preservation_rate` (the six `INTENT_FIELDS` unchanged
    between initial and revised intent; must be 1.0 by construction).
  - `unnecessary_factor_change_rate` (must be 0.0 by construction).
- **Baseline for the comparison story:** `full_replanning` — re-derive the
  whole intent (via `oracle_correct_intent`, all fields free to change) after
  failure. It also recovers success but does **not** preserve the frozen
  factors, so `frozen_factor_preservation_rate < 1.0` /
  `unnecessary_factor_change_rate > 0`. The headline contrast:
  > BABYSTEPS-selective matches full-replanning on retry success while
  > changing strictly fewer already-correct factors.

  Implementing the other baselines from `milestone1_locked_claim.md §4` is
  deferred to Milestone 3; this spec wires only selective + full-replanning so
  the minimal cut produces the contrast.

---

## 9. Testing

Sim-free unit tests (mirror the existing 284-test pattern; all must pass with
no real sim):

- `resolve_grounded_motion`: all 4 cardinal motions × 4 cardinal yaws ×
  {actor_frame, observer_frame}; non-cardinal yaw raises.
- `Intent` 7th-factor: default value, validation, **snapshot byte-stability**
  for the four existing tasks (omit-when-default), round-trip with the key
  present.
- `CrossViewPushAdapter`: `scripted_demo_to_intent` yields
  `actor_frame`; `oracle_wrong_factor` returns `direction_grounding` iff
  rotated+actor_frame; `attribute_failure` override maps `direction_error →
  direction_grounding`; `revise_intent` override flips only the grounding.
- End-to-end loop with a deterministic `FakeCrossViewEnvRunner` (added to
  `tests/conftest.py` + a `task_registry` fake factory): `actor_frame` +
  `θ_obs≠0` → object moved, wrong direction → `direction_error` →
  attribute → revise → `observer_frame` → success. Produces a stable
  `tests/snapshots/crossview_samples_seeds_0_4.jsonl`.
- Single-factor invariant: assert exactly `direction_grounding` differs
  between initial and revised intent on every revised episode.

---

## 10. File-Change Summary

| File | Change |
|---|---|
| `babysteps/schemas.py` | `DIRECTION_GROUNDINGS`; `Intent.direction_grounding` (defaulted); omit-when-default in `to_dict`/`from_dict`; `grounding_substitution` token |
| `babysteps/envs/scene.py` | `resolve_grounded_motion` helper |
| `babysteps/envs/crossview_adapter.py` | **new** `CrossViewPushAdapter` |
| `babysteps/envs/task_registry.py` | one `CrossViewPush-v1` entry (+ fake factory) |
| `babysteps/revision.py` | `grounding_substitution` branch |
| `babysteps/render/crossview.py` | **new** observer-camera 3-phase flow |
| `scripts/render_stage0_maniskill.py` | `CrossViewPush-v1` render dispatch |
| `scripts/stage0_summarize.py` | grounding-attribution + preservation metrics |
| `tests/conftest.py` | `FakeCrossViewEnvRunner` |
| `tests/snapshots/crossview_samples_seeds_0_4.jsonl` | **new** snapshot |
| `tests/test_*.py` | unit tests per §9 |

---

## 11. Acceptance Gate

The minimal cut is accepted when:

1. All existing tests pass and the four existing tasks' snapshots are
   **byte-identical** (additive-schema discipline verified).
2. New sim-free tests (§9) pass, including the single-factor invariant
   assertion and the regenerable CrossViewPush snapshot.
3. A GPU render produces, for ≥2 observer yaws, the 3-phase MP4 triple where
   attempt-2 (post-revision) reaches `info["success"]` and attempt-1 visibly
   pushes the wrong way (real `info["success"]` / cube-position evidence, not
   visual plausibility).
4. Collected JSONL over ~20–30 seeds yields `delta_pp ≥ 10` (retry vs initial
   success), `frozen_factor_preservation_rate == 1.0`, and
   `direction_grounding_attribution_accuracy == 1.0` for the rotated configs.
5. The full-replanning baseline shows recovery with
   `frozen_factor_preservation_rate < 1.0` — establishing the selectivity
   contrast.

> Phrasing note (consistent with prior sub-projects): criterion 3 is a
> **physical-validation** gate on the corrected attempt, not a claim that
> every seed/ yaw succeeds.

---

## 12. Open Questions / Deferred

- Scaling to multiple task families + 500–2000 episodes (`update.md` target)
  — separate spec once this loop is proven.
- Whether to later promote `direction_grounding` into `INTENT_FIELDS`
  (and regenerate all snapshots) once cross-view is a first-class result
  rather than an additive extension.
- The remaining baselines from `milestone1_locked_claim.md §4` — Milestone 3.
