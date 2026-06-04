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

## Milestone 3: Implement Baselines — DONE (2026-05-26)

All seven procedural baselines evaluated on 50 held-out seeds (100-149)
across PushCube, PickCube, StackCube. Job 10826466.

| policy                    | PushCube | PickCube | StackCube |
| ------------------------- | -------- | -------- | --------- |
| `one_shot`                | 0.000    | 0.000    | 0.000     |
| `same_intent_retry`       | 0.000    | 0.000    | 0.000     |
| `random_factor_revision`  | 0.540    | 0.920    | 0.420     |
| **`babysteps_selective`** | **0.980** | **0.920** | **0.700** |
| `text_feedback_replan`    | 0.000    | 0.920    | 0.700     |
| `full_replan_analogue`    | 0.000    | 0.920    | 0.820     |
| `oracle_factor_revision`  | 0.980    | 0.920    | 0.820     |

Full data: `reports/stage5/m3_baselines/main_table.{md,json}`.

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

> **Status (2026-06-03): PARTIAL PASS — PushCube only.** Frozen DINOv2
> ViT-B/14 (spatial_mean) → IntentHead. PushCube end-to-end passes: G1
> `object_motion` 0.95 (n=20, thin — labels collapse to near-binary +x/−x),
> G4 latent vs `same_intent_retry` +96pp, G5 latent vs oracle −2pp (96% vs
> 98%, within the −5pp tolerance). StackCube `object_motion` FAILS G1: 0.42
> (n=40) → 0.68 (n=200), still below the 0.90 gate — a relational
> two-object-direction bottleneck. Object-LOCAL DINO pooling gives NO lift
> over global (falsified at n=200, `reports/stage5/object_relation_probe_n200`);
> the ~0.95 position oracle is tautological by construction. R3M / encoder-swap
> ablations not pursued. **Scope locked to PushCube for the end-to-end latent
> claim.**
>
> **LATENT-INPUT PIVOT (2026-06-03): PushCube end-to-end on latent intent —
> DONE.** The method INPUT is now decoded from third-person demo-view vision
> (DINOv2 → IntentHead → nearest-centroid), severing the hand-authored JSON
> intent; JSON factors are used only for supervision (centroid codebook) +
> oracle eval. The VLM is only the failed-slot diagnoser; it does not extract
> latent intent or write the repair value. Pre-flight
> (GPU-free): vision-decoded initial intent reproduces the JSON intent **49/50
> (0.98)** on the P2 seeds (`reports/stage5/latent_decode_check/PushCube-v1`).
> Full latent P2 run (job 10951957, `--latent`, real InternVL3.5-8B + sim,
> `reports/stage5/p2_vlm_latent/PushCube-v1`): **C1 (latent input + latent
> slot-local ReviseHead) preservation 1.000 / harmful 0.000 / success 0.960
> vs C2 free-replan 0.700 / 0.047 / 0.680 — preservation +30pp, success
> +28pp, all gates PASS** (`n_latent_mismatch=1`). Statistically identical to
> the discrete fair run → the latent input + latent edit cost nothing. This is
> the reviewer-defense: "JSON factors are used for evaluation, not as
> privileged method input." Code: `babysteps/stage5/latent_intent.py`
> (`build_latent_intent` = Sever A, `latent_slot_edit` = Sever B);
> `scripts/stage5_p2_vlm_eval.py --latent`; sim-free tests in
> `tests/test_stage5_latent_intent.py`. **PickCube latent is representation-
> blocked** (its `contact_region` factor has zero pixel signature — the runner
> never sets gripper yaw; collecting data won't help). StackCube descoped
> (goal_state constant in cut). PushCube is the consolidated latent task.
>
> **Fully-latent G4/G5 (job 10954435):** `stage5_p1_run_eval.py
> --latent-initial` decodes attempt-1 from demo-view vision for ALL policies
> (Sever A in the recovery harness via the new `episode.py
> initial_intent_provider` hook)
> + latent revision. latent final 0.960 = oracle 0.960; same_intent_retry
> 0.000 → **G4 +96.0pp PASS, G5 +0.0pp PASS** (latent = oracle exactly).
> Identical to the scripted-attempt-1 run → the whole PushCube loop is now
> latent-input end-to-end (input + revision), at parity with discrete/oracle.
> `reports/stage5/p1_vision_g4_g5_latent/PushCube-v1/`.
>
> **LATENT CONSOLIDATION + FACTOR-GROUNDABILITY MAP (2026-06-03, LOCKED).**
> PushCube is the single CLEAN end-to-end latent task; the latent claim is
> written narrow-but-hard. Accurate statement: *we demonstrate the full
> third-person demo vision → latent intent → first-person BABYSTEPS repair loop
> on PushCube; the 5-task table evaluates selectivity under the audited
> structured schema; the groundability probes explain why the current frozen
> DINOv2 RGB-only interface does not cleanly extend to PickCube/StackCube
> factors.* Which factors a frozen-DINOv2 RGB demo can ground:
>
> | task · factor | latent-groundable? | evidence |
> |---|---|---|
> | PushCube · contact/approach/motion | ✅ clean, proven | faithful 49/50; latent C1 1.00/0.96 ≫ C2; G4 +96 / G5 +0 |
> | PickCube · contact_region | ❌ invisible | symbolic-only; runner never sets gripper yaw |
> | StackCube · object_motion | ❌ representation-blocked | DINOv2 0.68 (<0.90); object-local pooling doesn't help |
> | StackCube · goal_state | ❌ partial | clean config 0.99 but real demo CLIP caps at 0.82 (best pooling) < 0.90 |
>
> goal_state probe ladder (`scripts/stage5_goal_state_probe.py`, 3 modes,
> 8 sim-free tests; `reports/stage5/goal_state_probe{,_clip,_clip_pool}`):
> static pose-injected config **0.992** PASS, but the deployed
> spatial_mean-over-clip pooling falls to **0.650**, and NO temporal pooling on
> real clips clears the gate (last5 0.82 / final_frame 0.79 / first_last 0.71) —
> goal_state's signal is real but final-state-concentrated and washed out by
> arm-clutter + placement noise on execution clips. A genuine 2nd latent task
> would need a NEW task whose revised factor is plainly visible in a
> third-person RGB clip — NOT forcing an invisible/relational factor into DINOv2.

**Goal:** Replace handcrafted 20-dim Z with frozen DINOv2 features on
demo RGB frames. Retrain IntentHead. Gate: G1 probe ≥ 90%.

Deliverable:
- `babysteps/stage4/vision_features.py` — DINOv2 extraction + caching.
- Re-rendered varied-intent episodes with saved demo frames.
- G1 probe report on vision-grounded G (pass/fail per cell).
- ReviseHead retrained on vision G; G4/G5 sim rollout eval.

### M5b: VLM Attribution Baseline (P2)

> **Status (2026-06-03): DONE — validated.** InternVL3.5-8B attribution
> passes the gate on all 5 tasks (≥ rule-table). C1 (VLM-diagnosis +
> slot-local edit) beats C2 (VLM free-replan) on selectivity on every task:
> C1 frozen-factor preservation 1.000 and harmful-change rate 0.000
> everywhere. The fair free-replan baseline was re-run (2026-06-03) with the
> parse-failure straw-man fixed (StackCube C2 parse-fail 0.76 → 0.00); C1
> reuses the prior rollout byte-for-byte, only C2 was re-run. Headlines:
> PushCube C1 98% vs C2 70% success (dual win); PickCube 92% vs 90% but C2
> flips `embodiment_mapping` in 50/50 episodes; StackCube C2 wins raw success
> (82% vs 70%) yet C1 preserves 100% vs 2%; TurnFaucet attr 1.0 vs rule 0.5;
> CrossViewPush attr 1.0 vs rule 0.0. Rule-table still ties C1 on StackCube
> `goal_state` attribution (0.86). Reports: `reports/stage5/p2_vlm_fair`.

**Goal:** Run a VLM (InternVL3.5-8B) on failure packets, measure
attribution accuracy. Compare VLM-diag + slot-edit vs. VLM free-form replan.

Deliverable:
- VLM attribution accuracy on 50+ episodes per task.
- Comparison row: `vlm_diagnosis_slot_edit` vs. `vlm_free_replan` on
  recovery rate AND selectivity (frozen-factor preservation).

### M5c: World Model Counterfactual (P3)

> **Status (2026-06-03): NOT STARTED — de-risking falsified.** No world-model
> code/data/eval exists yet. The P3 de-risking probes were all falsified: VLM
> demo-read (PushCube 0.0, StackCube 0.25) and top-down / oblique viewpoint do
> not recover `object_motion` above the gate. Separately, the TurnFaucet
> (Sub-project D) re-grasp oracle is feasible (28–32% vs 4% poke baseline;
> vertical-axis subset clears the 30% gate) but productionization is pending a
> Stage-4 feature-dim decision. Go/no-go for P3 before the ICLR deadline is
> open — fallback is mechanical G2 bit-identity certification.

**Goal:** Train a latent dynamics model on ManiSkill rollouts. Use it
for G3 selectivity certification (counterfactual slot-drift test).

Deliverable:
- Forward model on rollout data.
- G3 counterfactual selectivity report.
- Revision-ranking ablation (world-model-guided edit selection).

### M5d: Learned Action Decoder (P4 — optional)

> **Status (2026-06-03): NOT STARTED — deferred (optional).**

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

**Revised schedule (as of 2026-06-03):**

**Done (ahead of schedule):**
- M5b (P2 VLM attribution) — complete and validated: all 5 tasks, C1 > C2
  on selectivity, fair free-replan baseline locked.
- M5a (P1 vision encoder) — PushCube end-to-end PASS; StackCube relational
  bottleneck diagnosed (object-local pooling falsified at n=200); scope
  locked to PushCube.

**June (now):**
- **P3 go/no-go decision** (the critical open call): train a latent dynamics
  model for G3 counterfactual selectivity, or fall back to mechanical G2
  bit-identity certification as the paper version.
- Resolve TurnFaucet (Sub-project D) productionization (Stage-4 feature-dim
  decision) — or move it to the appendix as attribution-only.
- Commit the outstanding object-relation diagnostics + n=200 reports.

**July:**
- If P3 greenlit: forward model + G3 counterfactual report. Otherwise
  consolidate the 5-task P2 main table + PushCube end-to-end latent story.
- Run the final 5-task × 50-seed evaluation; ablations (slot dim, multi-retry).

**August:**
Write full paper draft, figures, method diagram. TurnFaucet and
CrossViewPush in the main table (5 tasks total).

**Early September:**
Polish, rerun final experiments, write rebuttal-ready limitations.

## Priorities (updated 2026-06-03)

M5a (P1) and M5b (P2) are done. The immediate next step is the **P3
go/no-go decision**:

> Either train a latent dynamics model for G3 counterfactual selectivity,
> or accept mechanical G2 bit-identity certification as the paper version
> and consolidate the P2 main table + PushCube end-to-end latent story.
> In parallel: resolve TurnFaucet productionization (or appendix it) and
> commit the outstanding object-relation diagnostics.
