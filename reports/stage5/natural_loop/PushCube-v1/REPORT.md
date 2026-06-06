# Stage-5 — Natural-failure, seed-decoupled babystep loop (PushCube +x/−x)

**Claim demonstrated:** under the honest paradigm — the demo encodes one
instance, the Franka executes a *different* instance, and the artificial
`blocked_sides` block is removed — a stale demo intent fails *naturally*, and
**only revisers that consume execution feedback recover; open-loop revisers
recover 0%.** This moves the loop from "open-loop heuristic selection" to
"closed-loop interactive intent alignment."

## Setup

- Real ManiSkill `PushCube-v1` (`scripts/stage5_natural_loop_eval.py`, job
  10966379, n=50 held-out exec seeds 100–149; `demo_seed = exec_seed + 500` —
  a genuinely different instance).
- **No block** (`blocked_sides=()`). Failure is the wrong push direction.
- Binary **+x vs −x** via the goal-injection mechanism, because native PushCube
  pins the goal at +x for every seed and the +x-tuned controller only pushes the
  x-axis reliably (diagnostic job 10966085;
  `reports/stage5/diag_pushcube_seed_geometry/`).
- Four revisers compared (all single-factor edits of `contact_region`):
  - `same_intent` — retry the identical stale intent. *[open-loop]*
  - `rule_orthogonal` — the Stage-0 rule `revision.revise_intent`, which picks
    the 90°-orthogonal face, ignoring the failure direction. *[open-loop]*
  - `feedback_flip` — flip to the opposite face using the observed object
    **displacement vector** (non-privileged execution feedback).
  - `oracle_value` — set `contact_region` to the exec-scene-correct face (the
    executing robot observes its own goal). *[upper bound]*

## Results (final success rate)

| mismatch | initial | same_intent | rule_orthogonal | feedback_flip | oracle_value |
|---|---|---|---|---|---|
| **always** (guaranteed) | **0.00** | 0.00 | 0.00 | **0.98** | 0.96 |
| **random** (honest rate) | **0.48** | 0.48 | 0.48 | **0.98** | 0.98 |
| never (matched control) | 0.98 | 0.98 | 0.98 | 0.98 | 0.98 |

**Recovery on the mismatched episodes only** (random, 26/50 mismatched):

| same_intent | rule_orthogonal | feedback_flip | oracle_value |
|---|---|---|---|
| **0.0%** | **0.0%** | **96.2%** | 96.2% |

## 4-way results (job 10966526; orient_control gripper-yaw fix)

The +x-tuned controller was fixed to push all four cardinal directions (yaw the
gripper 90° via `orient_control`; +y needs −90°, −y +90° — wrist range; job
10966523 = 4×100%, ang_err <7°). With `--axes xy`:

| mismatch | initial | same_intent | rule_orthogonal | feedback_flip | feedback_residual | oracle_value |
|---|---|---|---|---|---|---|
| always (180°) | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| **random** | **0.20** | 0.20 | 0.48 | 0.38 | **0.94** | 0.96 |
| never | 0.98 | 0.98 | 0.98 | 0.98 | 1.00 | 0.98 |

**Recovery on the mismatched episodes only** (random, 40/50 mismatched):

| same_intent | rule_orthogonal | feedback_flip | feedback_residual | oracle_value |
|---|---|---|---|---|
| 0% | 35% | 22.5% | **92.5%** | 95% |

This is the headline ablation: **initial 20% → 94% (feedback_residual) ≈ 96%
(oracle)**, and it isolates the *feedback signal* needed.
- `feedback_flip` (reverse-on-failure, the displacement-vector-only heuristic)
  recovers only the **22.5%** of mismatches that are 180° — it cannot resolve a
  perpendicular +x→+y mismatch (reversing gives −x, still wrong).
- `rule_orthogonal` recovers **35%** — only when the fixed 90° rotation happens
  to point at the goal.
- `feedback_residual` (the observable goal − final-cube vector) recovers
  **92.5%**, matching the privileged oracle (95%).
- **Conclusion for the learned reviser (Step B): a 4-way loop needs a
  goal-RELATIVE feedback signal (the residual), not displacement-vector-only.**
  Displacement-vector-only is sufficient for the binary +x/−x loop above.

## B.2 — the LEARNED latent reviser (no hand rule; job 10966775)

The 4-way table above isolates the *feedback signal* with a hand rule
(`feedback_residual = direction_to_face(goal − final_cube)`). B.2 replaces that
hand rule with a **learned, residual-conditioned ReviseHead operating in the
vision-grounded latent slot space** — the same single-slot ReviseHead family the
rest of the method uses — and shows it hits the same ceiling.

**Pipeline (all additive; smoke + adversarial review + 609 sim-free tests gated the GPU spend):**
1. *4-way vision pack* (`stage5_train_4way_pack.py`, job 10966742): 80 PushCube
   demo clips rendered in all four push directions (the render-path
   `orient_control` gripper-yaw port — 78/80 clean, ang_err <5°), DINOv2 ViT-B/14
   features, IntentHead → 4 `contact_region` centroids. **Held-out IntentHead-CV
   separability: object_motion 0.975, contact_region 1.000, approach_direction
   0.975** (majority 0.25). The committed pack was x-axis-only (2 faces); this is
   the first 4-way vision-grounded PushCube latent.
2. *Real tuples* (`--dump-tuples`, job 10966743): 76
   `{demo_face, correct_face, residual_xy, predicate}` tuples from a **disjoint
   training seed range** (200–299; eval is 100–149). `residual_xy` is the
   non-privileged observed feedback; `correct_face` is the sim-derived training
   label (allowed off the demo→intent path, CLAUDE.md inv #4).
3. *Train residual head* (`stage5_train_residual_revise_head.py`): a ReviseHead
   with `fp_dim = FP_VECTOR_DIM_RESIDUAL` (factor+predicate one-hot + 2D unit
   residual) maps `centroid[demo_face] → centroid[correct_face]`; train acc 1.000.
4. *Eval* (`latent_learned` reviser): `g_slot = centroid[demo_face]`,
   `fp = vectorize_failure_packet_residual(observed residual)`, head → revised
   slot → `decode_slot` (nearest centroid) → corrected face. The **same**
   `_observed_residual` the hand rule uses — so a match is evidence of correctness.

**Held-out 4-way result (seeds 100–149, n=50; random mismatch 40/50):**

| reviser | all | on-mismatch |
|---|---|---|
| same_intent (open-loop) | 0.200 | 0.000 |
| feedback_flip (reverse-only) | 0.380 | 0.225 |
| feedback_residual (hand rule) | 0.940 | 0.925 |
| **latent_learned (LEARNED, no hand rule)** | **0.940** | **0.925** |
| oracle_value (privileged) | 0.960 | 0.950 |

Controls: mismatch=always (180°) → latent_learned 1.000; mismatch=never → initial
0.980, no revision fires. **initial 0.20 → 0.94.**

**Reading.** The learned head matches the hand rule exactly (0.925) and ≈ oracle
(0.950), while open-loop (0.000) and reverse-only (0.225) fail. It recovers the
*perpendicular* mismatches feedback_flip cannot, because it consumes the
goal-relative residual — but it does so as a *learned* slot-local edit decoding
into the vision-grounded centroid space, not a hand-coded geometric rule. The
seed-disjoint train/eval split makes the generalization honest. This is the B
deliverable: **closed-loop, feedback-conditioned, learned slot-local revision in
a vision-grounded latent space.**

**Honest scope.** (a) `latent_learned == feedback_residual` by design — the head
consumes the identical observed residual; the contribution is that the edit is
*learned* (same ReviseHead component as the rest of the method) and lives in the
vision latent space, not that it beats the rule. A skeptic can say it "relearned
direction_to_face"; the rebuttal is that the latent space *supports*
feedback-conditioned slot-local editing at the oracle ceiling — the method's
claim. (b) The eval `g_slot` is the centroid of the (scripted) demo face, not a
per-clip DINOv2 decode; full end-to-end vision (DINOv2 demo decode → IntentHead →
this head) composes the separately-proven P1 demo-decode (held-out 48/50 ≡
oracle) with B.2 and is the remaining stretch. (c) PushCube only; the 4-way
centroid separability (1.000) suggests the approach extends to any task with a
visible cardinal-direction factor.

## Full-vision closed loop — the JSON intent is removed (job 10969371)

B.2 still read the *initial* intent from the scripted sim-state evidence
(`scripted_demo_to_intent`); only the revision was learned-from-vision. The
full-vision loop closes the top: the initial intent is decoded straight from the
demo CLIP feature by `VisionIntentExtractor` (frozen encoder → StandardScaler →
IntentHead → per-factor nearest-centroid), with NO JSON/scripted intent anywhere.
The grounded factors (object_motion, contact_region, approach_direction) are
vision-decoded; the task-constant factors come from a template (they carry no
per-episode pixel signal — `trivially_constant` in the cert).

**Held-out, reviewer-proof seeds.** Exec 100–149; demo seeds 1000–1049 (offset
900) are DISJOINT from the pack-train demo seeds 600–679, so the demo-decode is
genuinely held-out. Both packs separate `contact_region` at 1.000 held-out CV
(DINOv2 1.000; V-JEPA 1.000 — but only *with* the persisted StandardScaler:
V-JEPA feature scale ~0.06, un-standardized the IntentHead underfits).

**Result (random mismatch, n=50, 35 initial failures; on-fail recovery):**

| encoder | vision-decode acc | init | latent_learned | feedback_residual | oracle |
|---|---|---|---|---|---|
| **DINOv2** (deploy default) | **1.000** | 0.300 | **0.971** (34/35) | 0.971 | 0.971 |
| **V-JEPA 2.1** (standardized) | **1.000** | 0.300 | **0.971** | 0.971 | 0.971 |

180° control: latent_learned 1.000 (DINOv2) / 0.980 (V-JEPA). Matched control:
init 0.94 (the 3 failures are controller-intrinsic, not direction errors, so no
reviser can fix them — `on_fail` there is a meaningless 3-sample denominator).

**Reading.**
1. **Vision-decode is perfect (1.000) on held-out demo seeds for BOTH encoders.**
   So the vision-grounded initial intent is behaviorally identical to the oracle
   scripted intent — the initial failures are pure natural geometry mismatch, not
   vision error. The JSON intent is fully removed at no cost.
2. **Full-vision recovery = 0.971 on-fail = oracle**, matching the scripted-intent
   baseline (B.2: 0.925). Swapping scripted→vision intent costs nothing.
3. **DINOv2 ≈ V-JEPA on PushCube (both 1.000 / 0.971).** PushCube's 2D
   cardinal-direction geometry is trivially visible, so the encoder choice is
   immaterial *here* — DINOv2 is the right deploy base. The V-JEPA edge is a
   *relational*-task lever (StackCube object_motion 0.86 vs DINOv2 0.68), not a
   PushCube one; this run confirms it does not hurt, and that the StandardScaler
   fix is what lets it match.

**Honest scope.** `on_mismatch` is geometry-defined (demo vs exec direction); in
vision mode the operative metric is `on_initial_fail` (denominator = actual
initial failures, which would also include any vision-decode error — here zero).
The CV separability number is a per-fold leak-free proxy; the deployed held-out
number is the eval's `vision_decode_acc` (1.000). The encoder forward (frames→z)
is the offline GPU cache step; the loop reads cached features (online in-loop
encoding is a trivial swap, deferred).

## Reading

1. **Failure is natural, not injected.** `initial` tracks the demo/exec
   direction match exactly: 0% (guaranteed mismatch), 48% (random), 98%
   (matched). No `blocked_sides` anywhere — contrast the artificial loop where
   `initial = 0.000` on every seed *regardless of decode*
   (`reports/stage5/p1_vision_g4_g5_latent/PushCube-v1/`).
2. **Open-loop revisers recover 0% of natural failures.** `same_intent` retries
   the same wrong push; `rule_orthogonal` rotates to a y-face (which the
   controller cannot even push) — neither consults the execution outcome to pick
   the corrective value. The Stage-0 block hid this because *any* unblocked
   approach trivially worked.
3. **The displacement vector alone is sufficient.** `feedback_flip` (0.98) is
   **non-privileged** — it reads only where the cube actually went — and matches
   / slightly beats the privileged `oracle_value` (0.96). The 0.98 ceiling is
   the controller's ~2% intrinsic failure (≈1/50), not a revision miss.
4. **Implication for the method.** Slot-local revision must be *conditioned on
   first-person execution feedback*. The current learned ReviseHead
   (`latent_slot_edit`) is not — it sees only demo-G + factor name + a coarse
   predicate, which cannot distinguish +y from −y. Step 3 = learn the
   feedback-conditioned edit (input: demo-G + displacement vector), targeting
   the `feedback_flip`/`oracle_value` ceiling without a hand-coded rule.

## Scope / honesty

- **Binary +x/−x only.** A 4-way loop (the dramatic 20%→88% figure) needs the
  +x-tuned open-loop controller fixed to push y (separate engineering; fix with
  real waypoint geometry, not tolerance widening). Binary is sufficient to
  validate the closed-loop-feedback claim; multi-class is a richness upgrade.
- `feedback_flip` is a hand-coded heuristic that proves signal sufficiency, not
  the learned method — that is Step 3.
- Intent here is the **scripted** demo→intent (faithful). The vision-latent
  decode (DINOv2→IntentHead) on top is a further step once features are cached
  for the binary directions.

## Provenance
- Driver: `scripts/stage5_natural_loop_eval.py`; sbatch
  `slurm/stage5_natural_loop_pushcube.sbatch` (job 10966379).
- Per-condition JSON: `reports/stage5/natural_loop/PushCube-v1/mismatch_{always,random,never}/`.
- Geometry/controller diagnostic: job 10966085,
  `reports/stage5/diag_pushcube_seed_geometry/`.
- Sim-free guards: `tests/test_stage5_natural_loop.py` (7 tests),
  `tests/test_stage5_residual_revise_head.py` (B.1, 2 tests),
  `tests/test_stage5_b2_latent_loop.py` (B.2 integration, 5 tests).
- **B.2 (learned latent reviser):**
  - 4-way vision pack: `scripts/stage5_render_demo_frames.py --four-way-range`
    + `scripts/stage5_cache_dinov2.py` + `scripts/stage5_train_4way_pack.py`
    (`slurm/stage5_b2_render_4way_pack.sbatch`, job 10966742) →
    `models/stage5/p1_vision_4way/PushCube-v1/`.
  - Tuples: `slurm/stage5_b2_collect_tuples.sbatch` (job 10966743) →
    `datasets/stage5/four_way/PushCube-v1/tuples_train.jsonl` (76 tuples).
  - Residual head: `scripts/stage5_train_residual_revise_head.py` →
    `models/stage5/p1_vision_4way/PushCube-v1/revise_head_residual.pt`.
  - Eval: `slurm/stage5_b2_eval_latent.sbatch` (job 10966775) →
    `reports/stage5/natural_loop/PushCube-v1/4way_latent_mismatch_{random,always,never}/`.
  - Smoke (yaw-port validation): job 10966726 (8 clips, ang_err <5°).
- **Full-vision (initial intent decoded from the demo CLIP, no JSON):**
  - `babysteps/stage4/vision_intent.py` (VisionIntentExtractor) +
    `stage5_natural_loop_eval.py --vision-intent`; `stage5_train_4way_pack.py
    --standardize/--feature-suffix`; `stage5_render_demo_frames.py
    --each-seed-all-dirs`. Sim-free guards: `tests/test_stage5_vision_intent.py`
    (6 tests).
  - Eval demos (held-out, 1000-1049 ×4 dirs) + DINOv2/V-JEPA cache:
    `slurm/stage5_fv_render_eval_demos.sbatch` (job 10969285).
  - V-JEPA standardized pack + residual head:
    `slurm/stage5_fv_vjepa_pack.sbatch` (job 10969318) →
    `models/stage5/p1_vision_4way_vjepa/PushCube-v1/`.
  - Full-vision eval (DINOv2 + V-JEPA): `slurm/stage5_fv_eval.sbatch`
    (job 10969371) →
    `reports/stage5/natural_loop/PushCube-v1/fullvision_{dinov2,vjepa21}_mismatch_{random,always,never}/`.
