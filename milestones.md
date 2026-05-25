For ICLR, I’d organize next work around one question:

> Can execution failure diagnose and selectively revise the wrong structured intent factor better than full replanning or action-level retry methods?

## Milestone 1: Lock The Claim

**Goal:** make the paper claim precise.

Current strong claim:

> BABYSTEPS uses Franka execution failure (observed in the executing
> Franka's first-person view) to revise one implicated intent factor
> while preserving the rest, given a third-person Franka demonstration
> of the same task.

Avoid overclaiming:

> “We solve human demonstration transfer.”
> “We solve cross-embodiment transfer.”

Better wording:

> “We validate structured single-factor intent revision in a controlled
> single-Franka cross-view setup (demo: third-person desk-front camera;
> execution: first-person wrist / robot-front camera). Cross-embodiment
> and richer demonstrators are explicit follow-on stages, not Stage-0
> claims.”

Deliverable:
- One-page project thesis.
- Final list of intent factors.
- Final list of failure predicates.
- Final comparison table design.

## Milestone 2: Finish Clean Multi-Task Stage 0

**Goal:** make the simulated benchmark convincing.

Minimum tasks:

- `PushCube`: blocked approach → revise `approach_direction`.
- `PickCube`: grasp/contact issue → revise `contact_region`.
- `StackCube`: underspecified goal → revise `goal_state`.
- `TurnFaucet` or replacement: constraint/contact violation → revise `constraint_region` or `embodiment_mapping`.

Important: if `TurnFaucet` remains physically awkward, replace it with a cleaner task. ICLR reviewers will care more about clean controlled evidence than forcing a broken environment.

Deliverable:
- 4 tasks.
- 20-50 seeds per task if feasible.
- Saved JSONL logs.
- MP4 examples for each task.
- One summary report per task.

## Milestone 3: Implement Baselines

This is probably the most important next step.

Baselines I would prioritize:

1. **One-shot**  
   Execute initial inferred intent once.

2. **Same-intent retry**  
   Retry the same intent/action after failure.

3. **Random retry**  
   Change low-level parameters randomly, no diagnosis.

4. **Random factor revision**  
   Randomly choose an intent factor to revise.

5. **Full intent replanning**  
   Given demo + failure, regenerate the full intent JSON. All fields may change.

6. **Text-feedback replanning**  
   Inner-Monologue-style: convert failure packet into text and ask for a new plan.

7. **BABYSTEPS no freezing**  
   Uses failure signal but allows non-implicated factors to change.

8. **BABYSTEPS selective**  
   Ours: revise only implicated factor.

9. **Oracle factor revision**  
   Upper bound.

For Diffusion Policy / VLA baselines, I’d include them as either:
- real baselines if we can run them fairly, or
- related-work positioning if we cannot.

Do not put Diffusion Policy in the main table unless we can actually evaluate it or clearly label it as “action-level baseline.”

## Milestone 4: Metrics And Main Table

We need metrics that show our contribution, not just success.

Core metrics:

- Final success rate.
- Retry success rate.
- Correct factor attribution.
- Frozen factor preservation.
- Harmful revision rate.
- Unnecessary factor change rate.
- Attempts to success.

The key result should look like:

> Full replanning may recover success, but changes too many correct factors. BABYSTEPS recovers while preserving correct factors.

That is the paper’s main empirical story.

## Milestone 5: Vision-Grounded Latent Intent (Stage 5 — ICLR target)

> **Updated 2026-05-24.** Replaces the earlier "Richer Cross-View Stress"
> milestone with the Stage 5 vision-grounded latent track. See `goal.md`
> §"Stage 5" for the full spec and `update.md` §"2026-05-24" for rationale.

The paper's contribution is the **representation + VLM-diagnosis /
learned-repair split**:

> A VLM diagnoses which latent intent factor caused a manipulation
> failure; a learned slot-local editor repairs only that factor in
> continuous visual-intent space, verified by counterfactual
> world-model rollouts.

### M5a: Vision Encoder Swap (P1 — critical first step)

**Goal:** Replace handcrafted 20-dim Z with frozen DINOv2 features on
demo RGB frames. Retrain IntentHead. Gate: G1 probe ≥ 90%.

Deliverable:
- `babysteps/stage4/vision_features.py` — DINOv2 extraction + caching.
- Re-rendered varied-intent episodes with saved demo frames.
- G1 probe report on vision-grounded G (pass/fail per cell).
- ReviseHead retrained on vision G; G4/G5 sim rollout eval.

### M5b: VLM Attribution Baseline (P2)

**Goal:** Run GPT-4o / Gemini on failure packets, measure attribution
accuracy. Compare VLM-diag + slot-edit vs. VLM free-form replan.

Deliverable:
- VLM attribution accuracy on 50+ episodes per task.
- Comparison row: `vlm_diagnosis_slot_edit` vs. `vlm_free_replan` on
  recovery rate AND selectivity (frozen-factor preservation).

### M5c: World Model Counterfactual (P3)

**Goal:** Train a latent dynamics model on ManiSkill rollouts. Use it
for G3 selectivity certification (counterfactual slot-drift test).

Deliverable:
- Forward model on rollout data.
- G3 counterfactual selectivity report.
- Revision-ranking ablation (world-model-guided edit selection).

### M5d: Learned Action Decoder (P4 — optional)

**Goal:** Replace skill compiler with a small policy conditioned on G.
Deferrable; the paper is strong with M5a–M5c and existing compilers.

## Milestone 6: Related Work Positioning

Need a clean related-work story:

- **Diffusion Policy / VLA:** maps observation/instruction to action.
- **Third-person imitation:** transfers behavior across viewpoint/embodiment.
- **Affordance / correspondence methods:** infer contact or action possibilities.
- **Inner Monologue / ReAct / SayCan:** replan after feedback.
- **BABYSTEPS:** VLM diagnoses which latent intent factor was wrong; a
  learned slot-local editor revises only that factor in visual-intent
  space.

This distinction needs to appear in the intro, method, and experiments.

## Milestone 7: Paper Draft

ICLR deadlines are usually around September/October.

**Revised schedule (as of 2026-05-24):**

**Late May – mid June:**
M5a — Vision encoder swap. G1 probe on DINOv2 features. This is the
go/no-go gate for Framing B. If G1 fails, diagnose and fix (spatial
pooling, R3M, fine-tuning) before proceeding.

**Mid June – early July:**
M5b — VLM attribution experiments. Build the `vlm_free_replan` baseline.
Generate the Stage 5 comparison table.

**July:**
M5c — World model training + G3 counterfactual. Run 5-task × 50-seed
full evaluation. Ablations (pooling, encoder, slot dim, multi-retry).

**August:**
Write full paper draft, figures, method diagram. Promote TurnFaucet
and CrossViewPush to the main table (5 tasks total).

**Early September:**
Polish, rerun final experiments, write rebuttal-ready limitations.

## Priorities (updated 2026-05-24)

The immediate next step is **M5a** (vision encoder swap):

> Implement `vision_features.py`, re-render episodes with frame capture,
> train IntentHead on DINOv2 features, run G1 probe. This is the
> critical gate — without vision-grounded features, the "latent" claim
> doesn't land.