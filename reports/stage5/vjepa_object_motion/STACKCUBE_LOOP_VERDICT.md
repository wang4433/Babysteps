# Stage-5 — Can V-JEPA 2.1 make a better latent-intent babystep LOOP on StackCube?

**Verdict: No — and the reason is structural, not a V-JEPA weakness.** A frozen
encoder swap cannot give a better StackCube loop, because the factor V-JEPA grounds
well is execution-decoupled, and the factor that drives execution is clip-blocked for
*every* encoder (including V-JEPA). The lever for a 2nd StackCube loop task is the
**demo design**, not the encoder.

This consolidates two probes (object_motion: `FINDINGS.md`; goal_state: this file +
`reports/stage5/goal_state_probe_vjepa21_{clip,config}/`) with the code trace of how
StackCube actually executes.

## The structural decoupling (code-verified)

| | factor | grounding (G1) | drives StackCube execution? |
|---|---|---|---|
| V-JEPA's win | `object_motion` | **0.88** (V-JEPA) vs 0.68 (DINOv2) | **NO** — `object_motion` appears 0× in `stackcube_runner.py`; the skill compiler `skills/stack.py:127` dispatches **only on `goal_state`**. Decorative. |
| Execution driver | `goal_state` | clip **0.78–0.83** (FAIL) for DINOv2 / DINOv3 / **V-JEPA**; config **1.00** | **YES** — branch selection (`cube_at_target` vs `cubeA_on_cubeB`). |

Also: in the loop cut, `goal_state`'s `initial_intent` is **constant** (`cube_at_target`
×40; only the *revision* flips it), so it carries no label variation to ground anyway;
`object_motion` is the only varying vision-groundable factor — and execution ignores it.
`revision.py` has **no `object_motion` operator** (it raises), so `object_motion` is also
not revisable. Net: a "StackCube G4/G5 loop with V-JEPA features" would pass (the
goal_state revision recovers the stack) but **V-JEPA ≡ DINOv2** — the encoder is not
load-bearing anywhere in that loop.

## The goal_state probe (2026-06-05, job 10961878, n=120 seeds 0-59, majority 0.500)

Same script/protocol/seeds as the committed DINO baselines; `--standardize` on (fixes the
un-normalized-Z IntentHead under-read for richer encoders). V-JEPA native 384 crop, clip
token-mean.

**CLIP (deployed temporal representation):**

| pooling | direct LR | IntentHead-CV | gate | DINOv3-L@512 |
|---|---|---|---|---|
| spatial_mean (all) | 0.817 | 0.783 | FAIL | 0.717 |
| last5_mean | 0.833 | **0.833** | FAIL | 0.825 |
| first_last | 0.808 | 0.808 | FAIL | — |
| final_frame | **0.842** | 0.808 | FAIL | 0.817 |

**CONFIG (static ceiling):** V-JEPA **1.000 PASS** (DINOv2 0.99).

## Reading

1. **goal_state is encoder-invariant clip-blocked.** V-JEPA best 0.833 ≈ DINOv3 0.825
   ≈ DINOv2 — a video encoder built for temporal dynamics gains nothing on the clip. The
   final-state poolings (last5/final) edge the full-clip mean for V-JEPA too (0.83 vs
   0.78), exactly like DINO → V-JEPA is *not* exploiting temporal dynamics here either,
   because there are none to exploit (the demo hides the vertical motion).
2. **The block is the deployed DEMO/CLIP REPRESENTATION, not the encoder.** Config ceiling
   = 1.000: every encoder reads the stacked-vs-near relation perfectly when rendered
   cleanly (arm retracted, perfect placement, single frame). The deployed clip caps at
   0.83 for a *pixel-level* reason verified by inspecting the rendered frames
   (`goal_state_probe_vjepa21_{config,clip}/.../frames/`, real demo
   `renders/stackcube/.../seed_0000__1_demo.mp4`):
   - **NOT "the demo hides vertical motion."** The lift IS visible in the clip (mid-frame
     shows cubeA raised in the gripper). The "2D-trajectory info loss" in
     `render/stackcube.py:137-145` refers to the *symbolic* `DemoEvidence.object_trajectory`
     (xy-only, 2 points) consumed by the SCRIPTED intent reader — not the vision/latent path.
   - The only difference between the `cubeA_on_cubeB` and `cube_at_target` clips is the
     **final placement** (on-top vs 6 cm beside); approach/grasp/lift are ~identical.
   - That brief final-state difference is (a) **occluded** by the gripper hovering over the
     cubes at release, (b) **small + oblique** (+0.04 m stack height vs +0.06 m beside, a few
     px at 512), (c) **diluted by whole-clip mean-pool** (hence final_frame/last5 0.83–0.84 >
     spatial_mean 0.78, still < 0.90), and (d) sometimes **absent** because the stack
     execution fails — e.g. seed_0001's stack and near clip last frames are BYTE-IDENTICAL
     (md5 match): no tower formed, cubeA landed beside cubeB, pixel-identical to place-near.
   So the 0.83 cap is a deployed-demo/render artifact (occlusion + small geometry + pooling +
   imperfect execution), falsification-grade and encoder-invariant; the info is recoverable
   (config 1.000) with a better demo render.

## The actual lever for a 2nd StackCube latent loop (not an encoder swap)

Re-render the StackCube demo so the **final placement is cleanly visible** — retract the
arm after release (de-occlude), pool the final frame / last-k (goal_state is a final-state
factor), and ensure the stack execution actually succeeds (so the two classes aren't
pixel-identical). Config says 1.0 is achievable. THEN: vary `goal_state` in `initial_intent`,
train the latent pack, and run G4/G5. That makes the groundable factor == the
execution-driving factor == the revised factor — the alignment PushCube has and StackCube
currently lacks. The bottleneck was never encoder capacity.

## Provenance
- goal_state V-JEPA: `reports/stage5/goal_state_probe_vjepa21_clip/` + `_config/`
  (job 10961878), `slurm/stage5_vjepa_goal_state.sbatch`.
- DINO baselines: `reports/stage5/goal_state_probe_vitl16_clip512/`, `goal_state_clip_pool/`.
- object_motion: `reports/stage5/vjepa_object_motion/FINDINGS.md`.
- Execution trace: `babysteps/skills/stack.py:127`, `babysteps/envs/stackcube_runner.py`
  (object_motion 0×), `babysteps/revision.py` (no object_motion operator).
