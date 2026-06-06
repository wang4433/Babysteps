# Stage-5 — StackCube `goal_state` full-vision loop (the 2nd positive latent task)

**StackCube `goal_state` is now vision-grounded end-to-end and recovered by
slot-local revision** — a second positive latent task beyond PushCube, on a
DIFFERENT factor (`goal_state`, a final-state relation) with a DIFFERENT natural
failure mode (goal under-specification, not direction mismatch).

## How we got here (the levers, with controls)

The StackCube `goal_state` clip-block was diagnosed and removed without touching
the encoder or the camera:

1. **Encoder is not the lever** (prior work): DINOv2 / DINOv3 / V-JEPA 2.1 all cap
   ~0.82 on the deployed clip — encoder-invariant.
2. **Camera viewpoint is not the lever** (`goal_state_camera/FINDINGS.md`, job
   10969709): a high-oblique sweep (~15°→69°) stays ~0.82 — viewpoint-invariant.
   The gripper is co-located with the placed cubes at the final frame and
   occludes the relation from *every* exterior angle. (This is also the
   decisive control that rebuts "your boundary is just where you point the
   camera": the block survives a full camera sweep.)
3. **The levers are the demo RENDER + the POOLING** (`goal_state_retract/FINDINGS.md`,
   jobs 10969792 / 10969827): clearing the gripper (a post-place **retract**) +
   **final-state pooling** (`first_last`) lifts goal_state to IntentHead-CV
   **0.920 ± 0.019** (n=300). goal_state is a final-state factor; whole-clip
   mean-pool dilutes it (~0.77) — the deployed uniform pooling is wrong for it.

## The loop (held-out, seeds 200-249 disjoint from pack-train 0-149)

Pipeline: dump both-class demo features → train a goal_state LatentPack → decode
the initial goal_state from the eval demo clip (VisionIntentExtractor) → execute
on the real StackCube runner → on `goal_not_satisfied`, revise with the
`goal_refinement` operator (`cube_at_target` → `cubeA_on_cubeB`) → retry. Only
goal_state varies off the oracle intent.

**Grounding↔revision spectrum** — the demo render trades grounding for
revision-reliance; the loop closes either way (jobs 10970071 / 10970175):

| demo render | vision stack-recall | mis-grounded | **operator recovery on mis-grounded** | same_intent (open-loop) | all-episode final (same → operator) |
|---|---|---|---|---|---|
| **retract** (grounding-heavy) | **0.98** (49/50) | 1/50 | **1/1 = 1.00** | 0/1 | 0.90 → 0.92 |
| **whole-clip** (revision-heavy) | 0.86 (43/50) | 7/50 | **5/7 = 0.71** | 0/7 | 0.82 → 0.92 |

Balanced 2-class grounding (the gate metric): train-pack first_last CV **0.910 ±
0.013** (n=300). The 0.98 / 0.86 above are *stack-recall* (the eval demos are all
true stacks), the loop-relevant quantity.

## Reading (honest)

- **goal_state grounds end-to-end ≡ oracle** (retract stack-recall 0.98), mirroring
  the PushCube full-vision result ("vision intent ≡ oracle, costs nothing").
- **Revision is load-bearing.** Open-loop `same_intent` recovers **0%** of failures
  in BOTH loops; `goal_refinement` recovers the mis-groundings. The whole-clip
  loop shows it at scale (5/7); the retract loop barely needs it (grounding
  carries it). This is the StackCube Sub-project C thesis — ambiguity →
  failure-as-evidence → slot-local goal_state edit — demonstrated.
- **The revised edit is correct 7/7** on the whole-clip mis-groundings; 5/7 then
  succeed because the **stack EXECUTION** has ~0.9 intrinsic reliability (the 4
  retract / 2 whole-clip "exec-failures" are correct-intent runs the runner
  missed — NOT goal_state errors, and goal_state revision correctly does not
  touch them). The metric is recovery-vs-no-recovery; the ~0.92 final is the
  exec ceiling, not 1.0.

## Caveats (reviewer-facing)

- **The goal_state revision is a single deterministic transition** (`cube_at_target`
  → `cubeA_on_cubeB`, the only valid strict-extension), so `operator` == `oracle`
  here. The contribution is the **vision-grounded goal_state intent + the
  slot-local edit on a 2nd factor/task**, not a learned/feedback-conditioned
  revision policy (goal_state has no continuous feedback signal, unlike PushCube's
  residual head). We deliberately do NOT add a degenerate "learned" head.
- **Final-state (`first_last`) pooling for goal_state** deviates from the deployed
  uniform `spatial_mean`. Defensible as *principled* (goal_state is a final-state
  factor; object_motion/contact_region are trajectory/contact factors), but it is
  a per-factor pooling choice and must be stated as such.
- **The retract demo render is goal-disambiguating.** Defensible — a demonstrator
  naturally finishes the action and the arm clears — and the camera/encoder
  controls show the block was the render+pooling, not capacity. The whole-clip
  loop shows the system still works (via revision) on the *non*-disambiguating
  demo.

## Provenance

- Grounding: `reports/stage5/goal_state_camera/FINDINGS.md` (camera falsified),
  `reports/stage5/goal_state_retract/FINDINGS.md` (retract + n=300 confirm).
- Loop: `reports/stage5/goalstate_loop{,_ambiguous}/StackCube-v1/goalstate_loop_results.json`
  (jobs 10970071 retract, 10970175 whole-clip).
- Code (branch `stage5-dual-camera`): `scripts/stage5_train_goalstate_pack.py`,
  `scripts/stage5_goalstate_loop_eval.py`, `stage5_goal_state_probe.py --retract
  --dump-features`, `stage5_cache_dinov2.py --frame-select`, retract in
  `babysteps/render/stackcube.py`. Pre-GPU adversarial review: 0 confirmed bugs
  (retract proven AutoReset-safe vs ManiSkill 3.0.0b22).
