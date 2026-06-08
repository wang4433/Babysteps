# Stage-5 — Distilled VLM-free attributor: consolidated result

**Claim.** On a held-out task family, a *distilled, VLM-free* attributor plus the
frozen shared revision policy **removes the online 8B VLM**: it recovers as well
as privileged oracle attribution, ~3000× faster, while preserving the
single-factor-revision invariant. The 8B VLM it replaces is not merely noisy —
it is *systematically wrong* on the held-out family.

This document consolidates the evidence and states the honest scope. It is the
landing point for the "Defend + distill" track; the planned multi-factor
*fusion* extension (build-order step 5) was **abandoned for a principled reason**
(below), not deferred for lack of effort.

---

## 1. Evidence

### 1a. Recovery gate — the headline (job 10982833; 120 episodes / 20 scene-seed clusters; 95% clustered-bootstrap CI)

Held-out **PokeCube-v1** (contact_region family), all episodes initial-failures.
Same first failure / controller / seeds / retry budget across conditions; the
*only* thing that changes between the VLM / distilled / oracle rows is the
attributor (the value policy is held fixed).

| condition | recovery_on_fail | attr_acc | preservation | edit card. | decision latency |
|---|---|---|---|---|---|
| `same_intent_retry` | 0.000 | — | 1.000 | 0.0 | 0 |
| `vlm_free_replan` (fair) | 0.167 | n/a | 0.833 | 1.17 | 3.12 s |
| `vlm_diagnosis_local_edit` (8B VLM) | 0.000 | 0.017 | 0.803 | 0.99 | 0.242 s |
| `shared_revision_policy` @ **VLM** | 0.008 [0.00,0.03] | 0.017 | 0.803 | 1.0 | 0.247 s |
| `shared_revision_policy` @ **distilled** | **0.967 [0.90,1.00]** | **1.000** | **1.000** | 1.0 | **0.001 s** |
| `shared_revision_policy` @ oracle | 0.967 [0.90,1.00] | 1.000 | 1.000 | 1.0 | 0 |
| `oracle_single_slot` (ceiling) | 0.967 [0.90,1.00] | 1.000 | 1.000 | 1.0 | 0 |

Paired clustered diffs:
- `shared@distilled − shared@oracle = +0.000 [0.000, 0.000]` → **distilled attribution == oracle** on the deployed distribution.
- `shared@distilled − shared@vlm = +0.958 [0.875, 1.000]` → swapping the VLM for the distilled head **restores** recovery.
- (prior value-transfer table, job 10979400) `shared@oracle − PushCube-editor@oracle = +0.333` → the one shared scorer *beats* the hand-built per-task editor on the held-out family.

**Read:** under correct attribution the value policy already transfers (== oracle).
The 8B VLM was the bottleneck (0.017 attribution). The distilled head closes that
gap at CPU-millisecond cost, preserving exactly one edited factor.

### 1b. Why the VLM fails — step-1 diagnostic (job 10982774; n=120)

The 8B VLM's constrained factor pick from a single third-person failure frame:

| predicted factor | count |
|---|---|
| **`embodiment_mapping`** | **106 (88%)** |
| `object_motion` | 12 (10%) |
| `contact_region` (correct) | **2 (1.7%)** |

The VLM **systematically** blames `embodiment_mapping` — a plausible-but-wrong
factor — almost never the occluded contact face. The failure is a consistent
bias, which is what makes "remove the VLM" defensible rather than a tuning fluke.

### 1c. Residual-insufficiency diagnostic (sim-free; `reports/stage5/attribution_ablation/`)

A residual/positional shortcut is **insufficient** for factor attribution. On a
deterministic PokeCube geometry with Class-A hard negatives (a misread
`object_motion` whose residual, contacted face, and trajectory are *byte-identical*
to a clean `contact_region` failure):

| modality mask | clean | hard-negative |
|---|---|---|
| residual-only / traj-only / res+traj | 1.000 | **0.000** |
| context-bearing (ctx / res+ctx / multimodal) | 1.000 | 1.000 |

The residual-only arm is pinned at exactly 0.500 overall; only the symbolic
intent-context modality breaks the tie. **Honest caveat:** this is *context
sufficiency by construction*, **not** multimodal *fusion* — see §3.

---

## 2. Method (recap)

- **Decomposition.** Attribution (which factor) is separated from revision (the typed value). A `RevisionPolicy` sees only `(factor, current_value, candidates, e_fail, g_i)` — never task id / gt / scene / full intent. `compile_single_slot_edit` enforces exactly one changed slot.
- **Distilled attributor.** A tiny CPU-torch `AttributionHead` with a *multimodal* interface (per-modality encoders + an explicit `[res, traj, obs, ctx]` mask) wrapped as `DistilledAttributor` (`name="distilled"`), plugged into the evaluator's `attributor_override` exactly like `oracle`/`vlm`. Trained on oracle/geometry labels (the VLM is near-random, so it is a *baseline*, not a teacher).
- **Artifacts.** `babysteps/stage5/attribution_{head,dataset}.py`, `scripts/stage5_{train_attribution_head,attribution_ablation,pokecube_step1_diagnostic}.py`, `slurm/stage5_pokecube_{step1_diagnostic,recovery_gate}.sbatch`. Commits `e087193`, `f38d839`, `4a88042`, `f697085`, `4a8f68a` (branch `stage5-unified-maintable`). 723 sim-free tests green.

---

## 3. Honest scope & limitations

1. **One factor family.** contact_region, PushCube→PokeCube leave-one-task-family-out. Not a broad tabletop-generalization claim.
2. **"Distilled" = oracle/geometry-label trained.** The VLM is near-random (0.017), so this is supervised on oracle/geometry labels with the VLM as the *baseline to beat*, not a teacher. On the deployed distribution the head is effectively a residual→face rule (context is redundant there).
3. **No multimodal *fusion* claim.** The sim-free ablation shows residual-*insufficiency* and context-*sufficiency*; it does **not** show fusion (context alone reaches 1.000 by construction). Genuine fusion needs a regime where no single modality is clean — pixel-*inferred* (imperfect) context, which is GPU/real-pixels work.
4. **Hard-negative robustness is sim-free only.** The deployed PokeCube loop contains clean contact_region failures; the hard negatives are a synthetic probe, not the deployed distribution.

---

## 4. Why the fusion extension was abandoned (principled)

A genuine multi-factor fusion attribution test requires ≥2 factors whose
residuals **overlap** so that a non-positional *inferred* signal is the unique
disambiguator. A survey + code-level analysis (push.py, pushcube_runner.py)
found this is **structurally precluded** by the framework's headline strength:

- **Single-factor decoupling.** The skill compilers deliberately decouple factors so each is independently revisable (the reviewer-facing single-factor invariant). In the push skill only `contact_region` is *causal* for the cube outcome; `approach_direction`, `object_motion`, `goal_state` are **execution-inert** (e.g. `approach_direction` only sets the approach waypoint; a wrong approach with the correct face *succeeds*, and only "fails" via `blocked_sides` → `planner_failed` → the *distinguishable* `approach_blocked` predicate).
- Forcing a second factor to be causal (an obstacle that makes a wrong approach jam reached-but-no-motion) would require new collision physics the runner does not have (`collision=False` is hard-coded) and would defeat the decoupling — a **manufactured benchmark artifact**, not latent-intent evidence.

**Conclusion.** The property that makes single-factor revision honest and visible
(decoupling) is the same property that prevents an in-framework fusion test.
Genuine fusion is **future work** requiring a coupled-factor task *outside* the
decoupled-skill framework (and pixel-inferred, imperfect context).

---

## 5. Future work
- A coupled-factor task (factors whose residuals genuinely overlap) for a real fusion test, with pixel-inferred context.
- A second held-out family for the recovery claim (generalization breadth).
- The single-factor "residual-insufficient *value* revision" case (PickCube occluded grasp-face) as an honest non-manufactured probe of the visual modality.
