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

## Milestone 5: Richer Cross-View Stress

For ICLR, we should show that the loop survives more aggressive
versions of the cross-view condition while staying Franka-to-Franka
(the embodiment axis is intentionally held constant — see `goal.md`
"Later Stages").

Possible minimal extension:

- Add multiple third-person camera placements (left / right / oblique
  desk-front) and verify that `direction_grounding` revision still picks
  the correct frame.
- Add a true first-person sensor stream (`panda_wristcam` or
  robot-front RGB-D) for the executing Franka — replacing the single
  default render camera that Stage 0 uses for both phases.
- Add controlled occlusions and lighting / background variation in the
  third-person demo, and show factor-attribution accuracy degrades
  gracefully.
- Run a small hand-labeled object-centric grounding pilot (still on
  Franka demo videos) to test whether DINOv2 / VLM grounding can
  replace the scripted labels Stage 0 uses.

This would support the claim:

> BABYSTEPS does not rely on a single canonical demo viewpoint or on
> privileged labels. It transfers object-centric intent factors across
> realistic cross-view variation, and uses failure to revise the
> implicated factor when the cross-view ambiguity is resolved
> incorrectly.

Cross-embodiment / human-demo extensions are deliberately *not* part of
this milestone — they would reintroduce the embodiment confound that
Stage 0 was designed to remove.

## Milestone 6: Related Work Positioning

Need a clean related-work story:

- **Diffusion Policy / VLA:** maps observation/instruction to action.
- **Third-person imitation:** transfers behavior across viewpoint/embodiment.
- **Affordance / correspondence methods:** infer contact or action possibilities.
- **Inner Monologue / ReAct / SayCan:** replan after feedback.
- **BABYSTEPS:** diagnoses which latent intent factor was wrong and edits only that factor.

This distinction needs to appear in the intro, method, and experiments.

## Milestone 7: Paper Draft Early

ICLR deadlines are usually around September, so we should not wait.

Suggested schedule:

**Now to early June:**  
Lock claim, clean task set, decide whether to keep or replace `TurnFaucet`.

**June:**  
Finish baselines and metrics. Generate first complete main table.

**July:**  
Run larger experiments, ablations, and the richer cross-view pilot
(multiple third-person placements + a real first-person sensor stream).

**August:**  
Write full paper draft, figures, related work, method diagrams.

**Early September:**  
Polish, rerun final experiments, write rebuttal-ready limitations.

## My Recommendation

The next concrete milestone should be:

> By the next professor meeting, show a draft main comparison table with rows for one-shot, full replanning, text-feedback replanning, BABYSTEPS selective, and oracle revision across at least PushCube, PickCube, and StackCube.

That will make the project feel like an ICLR paper, not just a demo.