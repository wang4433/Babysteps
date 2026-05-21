# Milestone 1 — Locked Claim

> **Status:** locked 2026-05-19. This document is the single paper-facing
> source of truth for the BABYSTEPS thesis, the intent-factor list, the
> failure-predicate list, and the main comparison-table design. Where it
> disagrees with scattered phrasing in `CLAUDE.md` (which still carries a
> pre-Stage-0 schema for reference) or `milestones.md`, **this document
> wins.** Code source-of-truth: `babysteps/schemas.py`,
> `babysteps/failure.py`, `babysteps/revision.py`.

---

## 1. Project Thesis (one page)

**Claim.** Robot execution failure is evidence about *what the task meant*,
not only about *how the action went wrong*. BABYSTEPS represents task
interpretation as a small set of structured **intent factors** and uses a
structured **failure packet** to attribute the failure to one implicated
factor, revise **only** that factor, and retry — preserving every factor the
failure did not implicate.

**Core invariant.**

```text
failure → identify implicated intent factor → revise only that factor → preserve the rest
```

If the system regenerates the whole plan after failure, it is no longer
testing this claim. Selective revision — not generic retry, not free
replanning — is the contribution.

**What separates us from neighbors.**

- *Inner Monologue / ReAct / SayCan* convert feedback to text and **replan**;
  they update the planner's context. BABYSTEPS updates the robot's **belief
  about the task** by editing a typed latent factor.
- *Diffusion Policy / VLA* map observation/instruction to **actions**.
  BABYSTEPS does not generate actions from the failure; it diagnoses which
  structured factor to edit. (Diffusion is reserved as an optional
  counterfactual *scorer* for edit ranking — not a controller — in a later
  stage.)
- *Full replanning* can recover success but changes correct factors too. The
  central empirical comparison is **BABYSTEPS-selective vs. full
  replanning**: same-or-better recovery, far higher preservation of correct
  factors.

**Scope honesty (Stage 0).** We do **not** claim human-to-robot transfer
yet. Stage 0 validates the revision mechanism in a controlled simulated
setting where **third-person demonstration proxies** (ManiSkill oracle /
scripted demonstrators, never human video) provide object-centric intent
evidence, while a Franka executes from a robot-centric view. The demo view ≠
execution view condition is the cross-view stressor. The paper-facing
sentence:

> We first validate structured intent revision in controlled third-person
> demonstration proxies, then study whether the same factorization supports
> human-to-robot transfer.

**Reviewer one-liner.**

> Inner Monologue replans after feedback; BABYSTEPS diagnoses and edits the
> single latent task-intent factor that made the previous plan fail, and
> leaves the rest untouched.

---

## 2. Final Intent Factors

Six object-centric factors (frozen schema; `INTENT_FIELDS` in
`babysteps/schemas.py`). No task-specific fields (`drawer_axis_correct`,
`push_side_correct`, … are forbidden — they make the method look
hand-designed per task).

| Factor | Meaning | Stage-0 role |
|---|---|---|
| `goal_state` | desired final object relation / pose | **revision target** (StackCube) |
| `object_motion` | demonstrated / intended object movement | defined + preserved (never revised) |
| `contact_region` | demonstrated / inferred contact site | **revision target** (PickCube) |
| `approach_direction` | route / side used to reach contact | **revision target** (PushCube) |
| `constraint_region` | scene region / object state to preserve | **defined but deferred** — see §2.1 |
| `embodiment_mapping` | how a proxy contact maps to Franka action | **revision target** (TurnFaucet) |

**Four factors are exercised as revision targets** — i.e., Stage-0 shows the
full failure→attribute→revise→retry loop for each:
`approach_direction`, `contact_region`, `goal_state`, `embodiment_mapping`.

`object_motion` is carried in every intent and held **frozen** across
revisions (it is part of the demo evidence and the preservation audit) but is
never itself the wrong factor in Stage-0.

### 2.1 `constraint_region`: defined but deferred

`constraint_region` is part of the locked schema and is preserved on every
revision, but Stage-0 **does not** exercise it as a revision target. The
TurnFaucet sub-project originally drove it (via the `constraint_introduction`
operator) but was reframed to a single-factor `embodiment_mapping` story
(grasp→poke) because the Panda gripper physically cannot grasp partnet
faucet handles. We claim `constraint_region` as **defined and reserved for a
later stage**, not as a Stage-0 result.

The associated code tokens remain in the whitelists as **deprecated /
reserved** and are not removed in Milestone 1 (additive-schema discipline;
removal is a later cleanup pass once `git grep` proves no live references):

- predicate `constraint_violation`
- operator `constraint_introduction`
- contact regions `faucet_base`, constraint region `faucet_base_static`
- embodiment mapping `proxy_contact_to_franka_turn`

---

## 3. Final Failure Predicates

Predicates are derived from the `AttemptResult` by a strict
most-specific-first precedence (`build_failure_packet`), then mapped to the
implicated factor by a rule table (`FAILURE_TO_FACTOR` in
`babysteps/failure.py`). The rule table is the analytic upper bound that a
learned attributor later replaces.

| Predicate | Fires when | Implicated factor | Revise set | Stage-0 status |
|---|---|---|---|---|
| `none` | attempt succeeded | — | — | terminal-success marker |
| `approach_blocked` | planner failed (no feasible approach) | `approach_direction` | `approach_direction`, `contact_region` | **active** (PushCube) |
| `direction_error` | object moved opposite the goal | `approach_direction` | `approach_direction` | active |
| `contact_failure` | never reached contact | `contact_region` | `contact_region` | active |
| `no_motion` | reached contact, object didn't move | `approach_direction` | `approach_direction`, `contact_region` | active |
| `goal_not_satisfied` | moved toward goal but goal predicate unmet | `goal_state` | `goal_state` | **active** (StackCube) |
| `grasp_slip` | gripper reached cube but lost grip | `contact_region` | `contact_region`, `embodiment_mapping` | **active** (PickCube) |
| `grasp_infeasible` | grasp-mode reached handle, jaws can't envelop it | `embodiment_mapping` | `embodiment_mapping` | **active** (TurnFaucet) |
| `constraint_violation` | touched a non-articulating link, no motion | `constraint_region` | `constraint_region`, `contact_region` | **deprecated / reserved** (see §2.1) |

Precedence order (most specific first): `none` → `approach_blocked`
(planner) → `constraint_violation` → `grasp_infeasible` → `grasp_slip` →
`contact_failure` → `no_motion` → `direction_error` → `goal_not_satisfied`.

**Revision operators** (`babysteps/revision.py`), one per active revision
target; each edits exactly one factor (only `constraint_introduction`, now
deprecated, edited two):

| Operator | Wrong factor | Edit | Status |
|---|---|---|---|
| `approach_substitution` | `approach_direction` | pick an unblocked approach (opposite first) | active |
| `contact_substitution` | `contact_region` | rotate to a 90° / unblocked face | active |
| `goal_refinement` | `goal_state` | `cube_at_target` → `cubeA_on_cubeB` | active |
| `embodiment_substitution` | `embodiment_mapping` | `…grasp_turn` → `…poke_turn` | active |
| `constraint_introduction` | `constraint_region` | `(none, faucet_base)` → `(faucet_base_static, handle_grip)` | deprecated / reserved |

---

## 4. Main Comparison-Table Design

Scope (locked): **7 methods × 3 tasks** = PushCube, PickCube, StackCube.
TurnFaucet is excluded from the main table (its physical gate is partial — see
`CLAUDE.md` TurnFaucet section) and is reported separately as a
mechanism-honest case study. Baseline *code* is Milestone 3; this is the
table **layout** the experiments must fill.

The original 5-row design (`one_shot`, `full_replan`, `text_feedback_replan`,
`babysteps_selective`, `oracle_factor_revision`) is a **subset** of the 7 rows
below — the `full_replan` row is now realised as `full_replan_analogue`, and
two intermediate rows (`same_intent_retry`, `random_factor_revision`) have been
added. Design reference:
`docs/superpowers/specs/2026-05-20-stage0-baselines-design.md`; implementation
plan: `docs/superpowers/plans/2026-05-21-stage0-baselines-plan.md`.

**Rows (methods), in reporting order:**

1. **`one_shot`** — execute the initial inferred intent once; no retry.
2. **`same_intent_retry`** — retry the identical intent (fresh rollout may
   recover by luck).
3. **`random_factor_revision`** — resample one random editable factor (ignores
   attribution).
4. **`babysteps_selective` (ours)** — attribute → revise only the implicated
   factor → retry.
5. **`text_feedback_replan`** — fix implicated factor + perturb its sibling
   factors (Inner-Monologue-style analogue).
6. **`full_replan_analogue`** — fix implicated factor + perturb all other
   editable factors (full-replanning analogue).
7. **`oracle_factor_revision`** — revise the ground-truth wrong factor; upper
   bound.

> For Stage 0, *full replanning* and *text-feedback replanning* are **deterministic
> procedural analogues**, not claims about a specific LLM/VLM planner's performance.
> They test whether recovery comes from selective factor revision or from generic
> retry / broad replanning. Live VLM/LLM replanners are future work (Stage 1 / appendix).

**Columns (metrics).** Grouped; ↑ = higher better, ↓ = lower better. Reported
per task and as a mean across the three tasks.

| Group | Metric | Dir | Source key |
|---|---|---|---|
| Outcome | final success rate | ↑ | `final_success_rate` |
| Outcome | retry success rate | ↑ | `retry_success_rate` |
| Outcome | attempts to success | ↓ | `num_attempts_to_success` |
| Diagnosis | correct-factor attribution acc. | ↑ | `intent_factor_attribution_accuracy` |
| Diagnosis | failure-type accuracy | ↑ | `failure_type_accuracy` |
| **Selectivity (headline)** | frozen-factor preservation | ↑ | `frozen_factor_preservation_rate` |
| **Selectivity (headline)** | unnecessary factor-change rate | ↓ | `unnecessary_factor_change_rate` |
| Selectivity | correct factor fixed (the true implicated factor was among the factors the retry edited — recovery to a valid alternative need not equal a single canonical token) | ↑ | `correct_factor_fixed_rate` |
| Selectivity | harmful revision rate | ↓ | `harmful_revision_rate` |
| Selectivity | frozen preservation (GT) | ↑ | `frozen_preservation_rate_gt` |

**Mockup (qualitative pattern the experiments should produce):**

```text
                          final↑  retry↑  attribAcc↑  preserve↑  unnecChange↓  correctFixed↑  harmful↓
one_shot                   low      —         —           —           —             —             —
same_intent_retry          low     low        —           —           —             —             —
random_factor_revision     mid     mid        —           low         high          low           high
babysteps_selective        high    high      high        HIGH        ~0            HIGH          ~0
text_feedback_replan       mid     mid       n/a         low         high          mid           mid
full_replan_analogue       high    high      n/a         LOW         HIGH          mid           HIGH
oracle_factor_revision     high*   high*     1.00        high        ~0            1.00          ~0
```

**The headline result.** Full replanning may match BABYSTEPS on success, but
collapses on *selectivity* (low frozen-factor preservation, high unnecessary
change, high harmful revision rate). BABYSTEPS-selective recovers **while
preserving the correct factors**; oracle bounds it from above. Selectivity
columns (`frozen_factor_preservation_rate`, `unnecessary_factor_change_rate`,
`correct_factor_fixed_rate`, `harmful_revision_rate`,
`frozen_preservation_rate_gt`) are the ones that carry the paper's argument —
they must be in the main table, not an appendix.

---

## 5. Milestone 1 Completion Checklist

- [x] One-page project thesis — §1
- [x] Final list of intent factors — §2 (4 active revision targets;
      `object_motion` frozen-only; `constraint_region` defined-but-deferred)
- [x] Final list of failure predicates — §3 (+ operator map)
- [x] Final comparison-table design — §4 (7 methods × 3 tasks)

Open follow-ons (not Milestone 1): implement the 7 procedural baselines (M3);
deprecated-token removal pass once `git grep` confirms no live references.
