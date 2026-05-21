# Stage-0 Procedural Baseline Table — Design Spec

> **Status:** design approved 2026-05-20, pending written-spec review.
> **Claim under test:** *recovery from execution failure comes from **selective
> factor revision**, not from generic retry or broad replanning.* This spec
> defines a **7-method × 3-task** comparison table of **deterministic
> procedural replanning analogues** (no LLM/VLM in the loop) that isolates
> "how many factors change after failure" as the single variable.
>
> Milestone-1 locked claim: `docs/milestone1_locked_claim.md` (§4 is the table
> layout this spec realizes and supersedes 5→7 rows — see §8).
> Code source-of-truth for the loop: `babysteps/episode.py`,
> `babysteps/revision.py`, `babysteps/failure.py`, `babysteps/eval.py`.

---

## 1. Scope & Non-Goals

**In scope:**

- A `RetryPolicy` abstraction injected into `run_episode`, with **7 policies**
  (§2). The current selective behavior becomes the default policy and stays
  byte-identical (snapshot guard).
- Fresh execution seed per attempt (§4).
- One new dataset-level metric, `harmful_revision_rate` (§5), plus a
  `compute_comparison_table(...)` aggregator + Markdown writer.
- A GPU sweep runner (`scripts/run_baselines.py` + an sbatch script) over
  **{7 policies} × {PushCube-v1, PickCube-v1, StackCube-v1}**.
- Reconciling `milestone1_locked_claim.md §4` from 5 to 7 rows (§8).

**Explicitly NOT in scope:**

- **Live LLM/VLM replanners.** `full_replan_analogue` and
  `text_feedback_replan` are *procedural analogues*, not measured planner
  performance. Live replanners are deferred to Stage 1 / an appendix.
- **TurnFaucet-v1 and CrossViewPush-v1 in the main table.** TurnFaucet's
  physical gate is only partial; CrossViewPush revises the 7th additive factor
  (`direction_grounding`). Both remain separate case studies.
- New failure predicates, new intent factors, or new revision *operators*. The
  baselines reuse the existing attribution + operators; they only vary the
  **change-set** applied after attribution.

---

## 2. The Seven Retry Policies (load-bearing)

A `RetryPolicy` is a pure callable sharing one input bundle and returning the
retry intent (or `None` for "no retry"):

```python
RetryPolicy = Callable[[
    Intent,            # initial_intent
    Attribution,       # from adapter.attribute_failure(failure_packet)
    FailurePacket,
    SceneState,
    Intent,            # oracle_correct_intent  (adapter.oracle_correct_intent)
    str,               # oracle_wrong_factor    (adapter.oracle_wrong_factor)
    random.Random,     # rng seeded per episode (deterministic)
], Optional[tuple[Intent, Revision]]]
```

| Policy | Behavior | Factors changed | Fixes implicated? |
|---|---|---|---|
| `one_shot` | return `None` — no retry | — | no (no retry) |
| `same_intent_retry` | retry `initial_intent` unchanged | none | no |
| `random_factor_revision` | resample **one random task-editable factor** (ignores attribution) | 1 random | only if it picks the implicated factor *and* lands on its correct value (by chance) |
| `babysteps_selective` | `revise_intent` on the attributed factor (today's behavior) | 1 implicated | yes |
| `text_feedback_replan` | fix implicated correctly, then resample its **sibling** factors (`attribution.revise` minus the implicated factor) | implicated + siblings | yes |
| `full_replan_analogue` | fix implicated correctly, then resample **all other task-editable factors** | implicated + all other task-editable | yes |
| `oracle_factor_revision` | `revise_intent` on `oracle_wrong_factor` (ground truth) | 1 true | yes (upper bound) |

**Task-editable factors.** The factors a policy may resample are the
**task-valid editable factors**: those that have more than one task-valid token
for the current task. `object_motion` is frozen-only (never revised — locked
claim §2) and `direction_grounding` stays at its default for these three tasks.
`full_replan_analogue` perturbs *all task-editable factors except the
implicated factor* (after applying the correct fix) — deliberately **not**
"every schema field", to avoid implying it edits frozen/irrelevant fields.

**Resampling rule (task-valid, current-excluded).** `resample_factor(intent,
factor, rng)` draws a new token from the factor's **task-valid alternative
tokens** for the current task (provided by the adapter — see §3), excluding the
**current** value only. Rationale for excluding *only* current (not also the
oracle value):

- The extra factors perturbed by `text_feedback_replan` /
  `full_replan_analogue` were already at their correct (oracle) value, so any
  change away from current is necessarily *wrong* → contributes to
  `harmful_revision_rate`. Excluding current alone already guarantees this.
- `random_factor_revision` must be able to land on the correct value when it
  happens to pick the implicated factor (the "fixes by chance" row). Excluding
  the oracle value would forbid that. So we exclude current only.

Drawing from **task-valid** alternatives (not global `schemas.py` whitelists)
keeps perturbed intents plausible rather than absurd — the baselines must not
be strawmen.

**Honest collapse note (text_feedback vs selective).** For StackCube the
failure's revise-set is `(goal_state,)` — no siblings — so
`text_feedback_replan == babysteps_selective` there. It diverges on PushCube
(`approach_blocked → revise=(approach_direction, contact_region)`) and PickCube
(`grasp_slip → revise=(contact_region, embodiment_mapping)`), which have a
sibling. We report this rather than invent a wider neighborhood.

**selective vs oracle.** When rule-table attribution is correct (Stage-0
mostly is), `babysteps_selective == oracle_factor_revision`. They diverge only
on misattribution; oracle bounds selective from above. Expected and correct.

---

## 3. Architecture (RetryPolicy injection)

`run_episode` currently hardcodes `attribute → revise_intent → one retry`
(`babysteps/episode.py` lines ~215–242). Refactor:

1. Extract that block into a `SelectivePolicy` that wraps
   `adapter.attribute_failure` + `adapter.revise_intent`. Make it the
   **default** so `run_episode(...)` with no policy arg produces byte-identical
   records (snapshot stability — `tests/snapshots/`).
2. `run_episode(*, episode_id, seed, adapter, policy=SelectivePolicy())`.
   The loop computes the failure packet + attribution once, then calls
   `policy(...)`. If the policy returns `None`, the record has `retry=None`
   (the `one_shot` path, same shape as the existing "none"-predicate path).
3. Each policy is a small, independently testable unit. Resampling uses the
   shared `resample_factor` helper.

**New module:** `babysteps/policies.py` (sim-free) holding the `RetryPolicy`
protocol, the 7 policies, and `resample_factor`. The `Revision` record for the
multi-factor policies sets `operator` to the policy name, `factor` to the
primary (implicated, or the random factor), and `frozen_factors` to every
factor not in the change-set — so the existing preservation check
(`_compute_metrics`) works unchanged for them.

**Adapter additions (additive):**

- `BaseTaskAdapter.task_valid_tokens(factor: str) -> tuple[str, ...]` — the
  task-valid alternative tokens used by `resample_factor`. Implemented per
  adapter; factors with ≤1 token are not task-editable.

---

## 4. Fresh Seed Per Attempt

Each attempt re-rolls stochastic rollout components with
`attempt_seed = stable_hash(episode_seed, attempt_idx)` **while holding the
scene layout (object/goal poses) fixed** — i.e., re-roll the rollout, not the
task instance. Requires an additive `EnvRunner.run(intent, scene, *,
rollout_seed=None)` parameter; re-calling `reset(seed)` is **not** acceptable
because it would change the layout and make attempt 2 a different task.

**Determinism caveat (recorded honestly).** `same_intent_retry` only recovers
when the env has per-rollout stochasticity independent of layout. If rollouts
are deterministic given (layout, action sequence), `same_intent_retry` is
provably 0% — still a valid, honestly-reported lower-bound row, not a bug.

---

## 5. Metrics & Comparison Table

**New metric — `harmful_revision_rate` (label-based, ↓).** An episode is
*harmful* if the retry changed any factor whose **initial value already matched
`oracle_correct_intent`** (changing an already-correct factor necessarily makes
it wrong, since the new value ≠ current = correct).
`harmful_revision_rate = |harmful episodes| / |revised episodes|`. It is a
strict subset of `unnecessary_factor_change_rate`. Enabled by persisting
`oracle_correct_intent` (its `to_dict()`) in the `EpisodeRecord` (additive;
snapshots updated deliberately). Computed in `eval.py` from
`initial_intent` + `factors_changed` + `oracle_correct_intent`.

**New metric — `correct_factor_fixed` (↑).** Per episode: the retry set the
**true implicated factor (`oracle_wrong_factor`) to its correct value**. For
`same_intent_retry` → false; `random_factor_revision` → true only on the lucky
case; `selective`/`text_feedback`/`full_replan`/`oracle` → true. This is the
column that explains recovery: *once the wrong factor is fixed, recovery can be
high; BABYSTEPS' edge is preserving the rest.*

**Comparison table.** `compute_comparison_table(datasets_by_method_task)` →
rows = the 7 methods (in reporting order), columns below, reported per task and
as a mean across the three tasks; plus a Markdown writer.

| Column | Dir | Source |
|---|---|---|
| final success rate | ↑ | `final_success_rate` |
| retry success rate | ↑ | `retry_success_rate` |
| correct factor fixed | ↑ | `correct_factor_fixed` (new) |
| frozen-factor preservation | ↑ | `frozen_factor_preservation_rate` |
| harmful revision rate | ↓ | `harmful_revision_rate` (new) |
| attempts to success | ↓ | `num_attempts_to_success_mean` |

`unnecessary_factor_change_rate` (already computed) is retained in the per-run
report for the locked-claim selectivity story, even though it is not a headline
table column here.

**Expected qualitative pattern:**

```text
                         final↑ retry↑ correctFix↑ preserve↑ harmful↓
one_shot                  low     —        —          —         —
same_intent_retry         low    low      0.00       1.00      0.00
random_factor_revision    low    low      ~chance    mid       mid
babysteps_selective       high   high     1.00       HIGH      ~0
text_feedback_replan      high   high     1.00       lower     >0
full_replan_analogue      high*  high*    1.00       LOW       HIGH
oracle_factor_revision    high   high     1.00       HIGH      ~0
```

`*` `full_replan_analogue` recovery may dip below selective when a harmful
collateral edit breaks a previously-correct factor — realistic, and exactly
what `harmful_revision_rate` captures.

---

## 6. Reporting Language (table caption / paper text)

> For Stage 0, *full replanning* and *text-feedback replanning* are
> **deterministic procedural analogues**, not claims about a specific LLM/VLM
> planner's performance. They test whether recovery comes from selective factor
> revision or from generic retry / broad replanning. Live VLM/LLM replanners
> are future work (Stage 1 / appendix).

---

## 7. Testing (sim-free)

All policy logic, `resample_factor`, the harmful/correct-factor metrics, and
table aggregation are unit-tested on the login node against the fake env — no
GPU/Vulkan (project invariant: `tests/` stays sim-free). Tests cover:

- each policy's change-set on a fixed `(intent, attribution, oracle)` fixture;
- `resample_factor` excludes current, draws only task-valid tokens;
- `harmful_revision_rate` and `correct_factor_fixed` on hand-built records;
- snapshot byte-equality of the default `SelectivePolicy` path;
- `compute_comparison_table` shape + a small golden table.

The real-sim sweep (`scripts/run_baselines.py` + sbatch) is the only GPU piece;
it reuses each task's existing runner and writes one dataset + report per
(method, task), then the comparison table.

---

## 8. Doc Reconciliation

Update `milestone1_locked_claim.md §4` row list from 5 to the 7 here
(`one_shot`, `same_intent_retry`, `random_factor_revision`,
`babysteps_selective`, `text_feedback_replan`, `full_replan_analogue`,
`oracle_factor_revision`) — additive (the original 5 are a subset, modulo the
`full_replan → full_replan_analogue` rename) — so the locked doc stays the
single source of truth. Add the new columns (`correct_factor_fixed`,
`harmful_revision_rate`) and the §6 caption.

---

## 9. Acceptance Gate

- All sim-free unit tests pass on the login node (no GPU).
- Default `SelectivePolicy` snapshots unchanged (byte-identical).
- GPU sweep produces a populated 7×3 comparison table (Markdown + JSON) whose
  qualitative pattern matches §5: `babysteps_selective` and `oracle` recover
  with high preservation and ~0 harmful revision; `full_replan_analogue`
  recovers but with markedly lower preservation and higher harmful revision;
  `same_intent_retry` / `random_factor_revision` do **not** reliably recover.
