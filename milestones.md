For ICLR, I’d organize next work around one question:

> Can execution failure diagnose and selectively revise the wrong structured intent factor better than full replanning or action-level retry methods?

## Milestone 1: Lock The Claim

**Goal:** make the paper claim precise.

Current strong claim:

> BABYSTEPS uses robot execution failure to revise one implicated intent factor while preserving the rest.

Avoid overclaiming:

> “We solve human demonstration transfer.”

Better wording:

> “We first validate structured intent revision in controlled third-person demonstration proxies, then study whether the same factorization supports human-to-robot transfer.”

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

## Milestone 5: Human Demo / Correspondence Extension

For ICLR, I think we need at least one bridge toward human demonstrations.

Not necessarily full real-robot deployment, but we should show that the representation is useful for human/proxy transfer.

Possible minimal extension:

- Take a simple human or hand-object video.
- Extract object motion/contact region manually or with VLM/DINO.
- Convert it into the same intent schema.
- Run the BABYSTEPS loop in ManiSkill.
- Show that failure still revises the right factor.

This would support the claim:

> We do not directly imitate human joint motion; we transfer object-centric intent factors.

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
Run larger experiments, ablations, and human-demo/correspondence pilot.

**August:**  
Write full paper draft, figures, related work, method diagrams.

**Early September:**  
Polish, rerun final experiments, write rebuttal-ready limitations.

## My Recommendation

The next concrete milestone should be:

> By the next professor meeting, show a draft main comparison table with rows for one-shot, full replanning, text-feedback replanning, BABYSTEPS selective, and oracle revision across at least PushCube, PickCube, and StackCube.

That will make the project feel like an ICLR paper, not just a demo.