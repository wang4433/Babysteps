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
  structured factor to edit. These methods use different action supervision
  and replace the controller, so they are related work rather than main-table
  recovery baselines.
- *Full replanning* can recover success but changes correct factors too. The
  central empirical comparison is **BABYSTEPS-selective vs. full
  replanning**: same-or-better recovery, far higher preservation of correct
  factors.

**Scope honesty (Stage 0).** We do **not** claim human-to-robot transfer
and we do **not** claim cross-embodiment transfer. Stage 0 is
deliberately **single-Franka cross-view**: one Franka demonstrates the
task on the desk under an oracle / scripted policy and is recorded from
a fixed third-person external camera; the same Franka then attempts the
task and is observed from its own first-person view (wrist /
robot-front camera). The demo camera ≠ execution camera condition is the
cross-view stressor; the demonstrator embodiment = executor embodiment
on purpose, so the embodiment axis cannot confound the result. The
paper-facing sentence:

> We validate structured single-factor intent revision in a controlled
> single-Franka cross-view setup: one Franka demonstrates the task from
> a third-person external camera, and the same Franka then attempts the
> task from its own first-person view. Failures are used to revise one
> structured intent factor at a time. Cross-embodiment and richer
> demonstrators are explicitly out of Stage 0 and are deferred to later
> stages.

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

---

## 6. Stage 5 — Vision-Grounded Latent Intent (ICLR pivot)

> Added 2026-05-24. See `goal.md` §"Stage 5" for the authoritative spec and
> `update.md` §"2026-05-24" for the rationale.

### Paper framing (Framing B)

The paper's contribution is **the representation + the VLM-diagnosis /
learned-repair split**, not just the discrete framework:

> A VLM diagnoses which latent intent factor caused a manipulation
> failure; a learned slot-local editor repairs only that factor in
> continuous visual-intent space, certified with paired counterfactual
> ManiSkill rollouts.

This requires that the slot intents are grounded in raw visual
observations (not handcrafted features). The Stage-0 discrete schema
remains as the supervision signal and certification scaffold.

### Revised comparison-table design (extends §4)

The M3 procedural baselines (7 rows × 3 tasks) remain as the
**Stage-0 baseline table**. Stage 5 adds a second table:

**Rows (methods) — Stage 5 table:**

1. `one_shot` — no retry.
2. `same_intent_retry` — retry identical intent.
3. `babysteps_selective` (rule attr + rule revision) — Stage-0 baseline.
4. `babysteps_latent` (rule attr + learned ReviseHead on handcrafted Z) — Stage-4 baseline.
5. **`babysteps_vision` (rule attr + learned ReviseHead on DINOv2 Z)** — P1 result.
6. **`vlm_diagnosis_slot_edit` (VLM attr + learned ReviseHead on DINOv2 Z)** — P2 result.
7. **`vlm_free_replan`** (VLM regenerates entire intent JSON) — the baseline to beat on selectivity.
8. `oracle_factor_revision` — upper bound.

`vlm_free_replan` is the primary broad-replanning competitor. The oracle row
must be visually separated and labeled as an upper bound. Diffusion Policy,
ACT, and generalist VLAs are excluded from this table because they replace the
action controller and require action-labeled demonstrations; a raw
success/latency comparison would not isolate failure-guided intent revision.
See `docs/related_work_and_baselines.md`.

**Key new columns:**

| Metric | What it shows |
| --- | --- |
| VLM attribution accuracy | Does VLM diagnosis match oracle wrong factor? |
| Vision G1 probe accuracy | Do vision-grounded slots recover discrete factors? |
| G3 counterfactual selectivity | Do paired true-simulator interventions confirm edited-factor improvement without frozen-factor violations? |

**Headline result target:**

```text
vlm_diagnosis_slot_edit matches vlm_free_replan on recovery rate
  but preserves 95%+ of already-correct intent structure
  while vlm_free_replan preserves < 50%.
```

### Tasks — expanded from §4

Promote **TurnFaucet** and **CrossViewPush** to the main table (5 tasks
total). The direction_grounding factor exercised by CrossViewPush is
the most novel cross-view claim; it must be in the main table, not
an appendix. Increase seeds to **50 per task** with error bars.

### Priority order

1. P1 — Vision encoder swap (DINOv2 on demo frames). Gate: G1 ≥ 90%.
2. P2 — VLM attribution baseline (InternVL3.5-8B). Gate: attr acc ≥ rule.
3. P3 — paired ManiSkill counterfactual certification. Gate: edited-factor
   improvement + frozen-factor equivalence to oracle single-slot intervention.
4. P4 — Learned action decoder. Optional / deferrable.

### Groundability scope — the latent claim, written narrow-but-hard

> Added 2026-06-04. Synthesizes the latent-intent pivot + the encoder-swap
> control. Artifacts: `reports/stage5/dinov3_encoder_swap/SUMMARY.md`,
> `reports/stage5/latent_decode_check/`, `reports/stage5/p2_vlm_latent/`,
> `reports/stage5/goal_state_probe*`, `reports/stage5/plugcharger_probe*`.

The latent claim is scoped **narrowly** so it is **hard**: the
vision-grounded latent pipeline (frozen DINOv2 demo frames → trained
IntentHead → per-factor nearest-centroid codebook) is demonstrated
**end-to-end on PushCube** and only PushCube. There, vision-decoded intent
reproduces the oracle JSON **49/50** per factor; the fully-latent loop (latent
input + learned slot-local ReviseHead) matches the discrete fair run
(frozen-preservation **1.00**, success **0.96**) and crushes VLM free-replan
(0.70 / 0.68), with recovery **G4 = +96pp / G5 = +0pp = oracle exactly**. So
replacing the hand-authored JSON method input — and the discrete operator — with
vision costs nothing on the one task where the factors are visible. This answers
the "JSON intent is privileged input" reviewer attack on a real end-to-end task.

**Why only PushCube — the groundability map.** A frozen encoder can ground a
latent factor only when the discriminating evidence is present in the deployed
third-person pixels. We certify this per (task, factor) with the G1 probe
(IntentHead-CV, gate 0.90), and the map is itself a result:

| task · factor | groundable? | why |
| --- | --- | --- |
| PushCube · object_motion / contact_region / approach_direction | **✅ 0.98** | plainly visible in the demo frame |
| PickCube · contact_region | ❌ | symbolic-only: grasp geometry is identical across faces (zero pixel signature) — a *task* limit, encoder-agnostic |
| StackCube · object_motion | ❌ ~0.68 | representation-blocked; object-local pooling gives no lift |
| StackCube · goal_state | ❌ | clean **config 0.99** but real-clip ceiling **~0.82**: the demo hides the vertical stack-vs-place motion (information loss) |
| PlugCharger · charger_yaw (orientation) | ❌ ~0.85 | sub-resolution asymmetric base (~1–3 patches) |
| PlugCharger · charger_xy (position) | ✅-but-trivial | the "where is the object" localization factor, not a manipulation intent |

**The encoder-swap control (the hard part).** A reviewer will ask whether the
negatives are just "DINOv2 wasn't strong enough." We answer with a controlled
swap to **DINOv3** (ViT-L/16 and B/16, incl. res 512) under *identical* labels,
probe, and gate — only the encoder changes (each at its native resize so neither
is handicapped). Result (`SUMMARY.md`):

- the **blocked factors stay blocked**: goal_state real-clip ceiling is
  **identical at ~0.82** across every encoder and resolution; charger
  orientation stays **~0.78–0.86** — both FAIL;
- yet on a factor DINOv2 could *not* ground at deployed resolution — charger
  **position** — DINOv3 crosses the gate (**0.84 → 0.90–0.95**).

DINOv3 helps *exactly* where the information is in the pixels and *not* where it
isn't. This certifies the negative ladder is a **factor-observability** property,
not an encoder-capacity artifact — the sharp, defensible form of the limitation.

**Paper consequence.** The end-to-end *latent* claim is PushCube (clean, proven);
the **5-task discrete framework** carries the structured-breadth claim
(single-factor VLM-diagnosis + slot-local repair vs free-replan); the
groundability map + encoder-swap is an honest **scope/limitation** section, not a
weakness. Next encoder lever is **V-JEPA** (a video/world-model encoder) on the
*temporal* cells — `object_motion`, and `goal_state` whose blocker is precisely
temporal information loss — which a frozen *image* encoder cannot address by
construction.
