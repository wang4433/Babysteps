# Redesign failure paradigm — Phase 1 + Phase 3 findings

Status snapshot 2026-05-27.

Source plan: `/home/wang4433/scratch/babysteps/redesign_failure_paradigm.md`.
This report records the *first three* items in the doc's "Recommended
execution order": (1) invariant audit, (2) Phase 1 render code change +
GPU re-render, (3) Phase 3 diagnostic spike.

The Phase 2 decision (PickCube rotation vs Phase 2-lite) is deferred per
the doc until after Phase 3's decision gate.

---

## 1. Single-factor invariant audit — PASS (both new mechanisms)

Files read: `babysteps/schemas.py`, `babysteps/envs/pushcube_adapter.py`,
`babysteps/envs/pickcube_adapter.py`.

### PushCube clutter — PASS

- `pushcube_adapter.py:43` (`oracle_correct_intent`) and `:70`
  (`scripted_demo_to_intent`) both hardcode `constraint_region="none"`.
- The schema's `CONSTRAINT_REGIONS` frozenset (`schemas.py:57-60`) has
  only `{"none", "faucet_base_static"}` — no scene-aware constraint
  tokens exist.
- The only intent factor influenced by the scene-side `blocked_sides`
  is `approach_direction` (`default_blocked_factory`, line 47-48), which
  is the factor revised on retry.
- Adding a clutter object to the scene cannot touch any intent factor
  other than the one already being revised. ✓

### PickCube rotation — PASS

- `pickcube_adapter.py:69-83` (`scripted_demo_to_intent`) extracts only
  `contact_region_label` from the demo. The other five factors are
  hardcoded: `goal_state="cube_lifted_at_target"`, `object_motion=
  "lift_up"`, `approach_direction="from_above"`, `constraint_region=
  "none"`, `embodiment_mapping="proxy_contact_to_franka_grasp"`.
- Cube rotation is an execution-side scene-state change. The demo→intent
  path consumes no pose-dependent execution state. The Intent derived
  in Context A (demo) is unchanged by rotating the cube in Context B
  (execution).
- The revision touches `contact_region` only. ✓

### Out-of-scope flag (Phase 2 implementation detail)

`CONTACT_REGIONS` are world-frame face labels. Rotating the cube around
the z-axis (yaw) keeps all four side faces accessible — just remapped.
Rotating around x or y (tipping onto a side) leaves world-frame
`minus_x_face` labeled on whichever cube face currently has the lowest
x — typically a former top/bottom face, still graspable from above.
**The cube-rotation mechanism, as described in the plan, may not
deterministically block grasp from the demo's world-frame face.**
Prototype the rotation in `pickcube_runner.py` before committing to it:
options include 45–60° tilt (cube unstable), or a wedge under one cube
edge. This is a Phase 2 implementation decision and does not affect the
invariant audit conclusion.

---

## 2. Phase 1 — PushCube clutter render

### Code change (committed in working tree, not staged)

`babysteps/render/pushcube.py`:
- `_OBSTACLE_HALF_W` 0.020 → 0.025
- `_OBSTACLE_HALF_T` 0.075 → 0.025
- `_OBSTACLE_HALF_H` 0.075 → 0.04 (clutter is 5 cm × 5 cm × 8 cm now)
- Color `[0.78, 0.20, 0.20, 1.0]` (red) → `[0.55, 0.45, 0.35, 1.0]`
  (grey-brown)
- z-placement: was `cube_z + _OBSTACLE_HALF_H` (base 2 cm above the
  table — invisible on a 15 cm wall, visible as a "floating box" on a
  small clutter object). Now `cube_z - CUBE_HALF_SIZE + _OBSTACLE_HALF_H`
  so the clutter base sits on the table surface.
- Removed the 90°-around-z rotation in `_move_obstacle_to_block`: the
  prior asymmetric wall needed it to face the EE; the new 5 cm × 5 cm
  cross-section is symmetric.
- Module docstring + Phase 2 comment + `_get_or_build_obstacle`
  docstring updated to describe "grey-brown clutter object" rather
  than "red wall" / "red box".

### Verification

- `python -m pytest tests/test_render_modules.py tests/test_pushcube_adapter.py -q` →
  **41 passed in 0.28s** (all sim-free tests).
- `python -c "from babysteps.render import pushcube; print(...)"` →
  imports cleanly with new constants.

### GPU re-render job — COMPLETED (job 10848944, 54 s wall, exit 0:0)

- Slurm script: `slurm/render_pushcube_clutter.sbatch` (`--qos=standby`,
  a100-40gb, 20 min budget, 3 episodes, seeds 0..2).
- Output directory: `renders/pushcube_clutter/videos_maniskill/`.
- Outputs (3 seeds × 3 phases = 9 MP4s, all written):

| seed | demo (3rd-person) | attempt_blocked (wrist) | retry (wrist) |
| --- | --- | --- | --- |
| 0000 | 312 KB | 1010 KB | 870 KB |
| 0001 | 313 KB |  952 KB | 911 KB |
| 0002 | 310 KB | 1023 KB | 905 KB |

For comparison, the old red-wall renders are 312/1100/868, 313/1100/905,
... — file sizes are within natural variation. Pre-existing harmless
warnings only: Vulkan ICD warning + `panda_wristcam not in supported
robots` (same as on the wall renders).

### Visual verification — PASS

Extracted single frames at ~20–25% through each `_2_attempt_blocked.mp4`
clip via OpenCV (cached at `/tmp/clutter_seed{0,1,2}_attempt_*.png`).

All three seeds show the same expected pattern in the wrist-cam view:

- Wood-grain table fills the background.
- The **grey-brown clutter box** sits on the table directly on the
  gripper's approach side (left/center of the wrist view depending on
  the approach axis for that seed). Visually reads as a small container
  or block on the table — *not* a wall.
- The **blue PushCube target** sits immediately beyond the clutter,
  toward the red bullseye goal.
- The clutter occupies a believable footprint relative to the cube
  (clutter ≈ 5 × 5 cm cross-section vs cube ≈ 4 cm cross-section), and
  sits on the table surface (no floating gap — the z-placement fix
  worked).
- The gripper apparatus enters the frame in the foreground; the
  no-progress break ends the clip at 51 frames (≈ 2.5 s at 20 fps).

No regression observed: phase 1 (`_1_demo.mp4`) shows the same clean
demo as the wall renders (clutter parked below the table is invisible),
and phase 3 (`_3_retry.mp4`) shows the orthogonal retry succeeding.

Conclusion: **Phase 1 ships as-is.** The figure-quality concern in the
plan is resolved.

### What the user should check on the rendered MP4s

For each seed:
1. `_1_demo.mp4` — should show the gripper pushing the cube to goal with
   no obstacle visible. The clutter object should be parked below the
   table (invisible).
2. `_2_attempt_blocked.mp4` — the small grey-brown clutter object should
   sit on the table on the demo's approach side. The gripper should
   approach, stall against the clutter, and the no-progress break should
   end the clip. The clutter should look like a natural object (mug /
   small box) sitting on the table — *not* a wall.
3. `_3_retry.mp4` — gripper takes an orthogonal approach, succeeds. The
   clutter remains parked below the table (out of view).

If the clutter looks floating, mis-placed, or too small to actually
block the arm, return to `_move_obstacle_to_block` and adjust the z
offset or half-extents.

---

## 3. Phase 3 — TurnFaucet poke_turn diagnostic

### What it does

`scripts/stage5_p3_turnfaucet_diagnostic.py` reproduces the production
`TurnFaucetEnvRunner.run` poke_turn dispatch (probe(+1, 80 steps) →
maybe full(+1, 400) → maybe full(-1, 400) → fallback full(+1, 400))
on seeds 100..149. Per seed it records every trial's `(success,
progress, reached_contact, object_moved)`, the final faucet qpos,
the final 9-dim robot qpos, and a categorization:

| category | meaning |
| --- | --- |
| `success` | `info['success']` on the chosen trial |
| `no_contact` | gripper never reached the handle |
| `contact_no_motion` | reached handle, qpos delta < 0.05 rad |
| `partial_rotation` | 0 < \|progress\| < 0.5 |
| `mostly_rotated` | 0.5 ≤ \|progress\| < 0.95 |
| `near_success_no_termination` | \|progress\| ≥ 0.95 but `info['success']=False` |
| `exception` | seed raised |

Also: an `arm_near_limit` flag if any of the 7 Panda arm joints ended
within 0.10 rad of its hard limit — heuristic for "arm collapsed".

### GPU job — COMPLETED (job 10848995, 9:07 wall, exit 0:0)

- Slurm script: `slurm/stage5_p3_turnfaucet_diagnostic.sbatch`
  (`--qos=standby`, a100-40gb, 1 h budget — used 15%).
- Output: `reports/stage5/turnfaucet_diagnostic/{per_seed.jsonl,summary.md}`.
- Job id: **10848995**.

### Results (seeds 100..149, 50 episodes)

| category | count | rate |
| --- | --- | --- |
| `success` | 2 | 4% |
| `contact_no_motion` | **21** | **42%** |
| `no_contact` | 14 | 28% |
| `partial_rotation` | 11 | 22% |
| `mostly_rotated` | 2 | 4% |
| `arm_near_limit` (co-occurring flag) | 18 | 36% |

The 4% success rate confirms the figure cited in `redesign_failure_paradigm.md`
(was based on production data; the diagnostic reproduces it exactly).

### Decision-gate read — PROCEED, with a hard 3-day cutoff

- **Top-1 is `contact_no_motion` (42%)** — the gripper reaches the
  handle, but qpos changes by < 0.05 rad. The empirical reference
  already in the repo (`scripts/_diag_tf_poke5.py`, v5 notes lines
  1-12) calls out exactly this failure mode and the fix: "v4 found
  that multi-step sub-waypoints slow the gripper (per-step action
  drops from ~1.0 to ~0.3) which removes the impulse force that v1
  relied on. Return to v1's single-waypoint LONG sweep." Porting the
  v1 brute-force single-waypoint sweep into the production
  `compile_intent_to_turn_skill` is plausibly 2 days.
- **`no_contact` (28%) + `arm_near_limit` (36%)** — substantial
  overlap. These are waypoint-geometry / IK problems: the arm wedges
  against joint limits before reaching the handle. The diag script's
  `compute_geometry` helper is the reference but does not fully
  resolve these in the v5 run either. Fix: ~1-2 more days, less
  certain ceiling.
- **`partial_rotation` (22%)** — got contact and some rotation, but
  stalled. Same fix as `contact_no_motion` should improve these too
  (longer sweep distance).
- **`mostly_rotated` (4%)** — within striking distance of target.
  Trial-budget / success-threshold tuning, < 1 day.

**Optimistic ceiling estimate**: fixing `contact_no_motion` +
`partial_rotation` + `mostly_rotated` lifts success from 4% to
~72%. Realistic estimate accounting for partial fixes and incomplete
generalization: 30–50%.

**Recommendation**: **proceed with Phase 3 fix**, with the doc's
existing kill-switch enforced: if held-out success < 30% after 3
days of engineering, drop TurnFaucet to appendix and adopt the
4-task narrative (already pre-drafted in `redesign_failure_paradigm.md`
§"Paper narrative" → "Four-task version").

### Concrete next steps (for whoever picks this up)

1. Read `scripts/_diag_tf_poke5.py` lines 1-90 — the
   `build_brute_waypoints` v1-style 3-waypoint single-sweep is the
   target geometry.
2. Compare with `babysteps/skills/turn.py`'s
   `compile_intent_to_turn_skill` for `poke_turn` mode — the
   production waypoints are likely the multi-step version that v5
   notes flagged as force-attenuating.
3. Port the v1 single-sweep into `compile_intent_to_turn_skill` (mode
   = poke). Keep the auto-sign two-trial dispatch in the runner
   unchanged.
4. Re-run `scripts/stage5_p3_turnfaucet_diagnostic.py` on the same
   seeds 100..149 to measure lift.
5. If success ≥ 30% — proceed to Phase 4/5 (re-render + re-eval). If
   < 30% — invoke the appendix path.

### Decision-gate guide (from the redesign doc)

After this job finishes, look at the top-1 category in
`summary.md`:

- If `no_contact` or `contact_no_motion` dominates → fix is on the
  geometry / contact side (waypoint computation + sweep tuning).
  Compare to `scripts/_diag_tf_poke5.py`'s `compute_geometry`/
  `build_brute_waypoints` which is the empirical reference.
  Estimated effort: 2-3 days. **Proceed with Phase 3.**
- If `partial_rotation` dominates → fix is sweep distance / contact
  budget. ~1-2 days. **Proceed.**
- If `arm_near_limit` co-occurs heavily → IK is wedging the arm.
  Likely needs a different waypoint approach (lower contact_z, or
  body-frame joint preferences). 3+ days, possibly more.
  **Borderline — decide on residual time.**
- If `mostly_rotated` or `near_success_no_termination` dominates →
  the existing skill is *almost* working; tighten the trial budget
  or success threshold. ≤1 day. **Proceed.**

If the diagnostic shows mixed failure modes with no clear top-1 fix,
adopt the 4-task narrative and put TurnFaucet in the appendix.

---

## 4. What I did NOT do (deferred per the plan)

- Phase 2 (PickCube rotation mechanism) — gated on Phase 3 outcome per
  the doc.
- Phase 4 / Phase 5 (full re-render and re-eval) — scope determined by
  which mechanisms ultimately change.
- Drafting the `face_inaccessible` predicate (Phase 2 schema work) —
  unnecessary until Phase 2 is committed.
- Paper-narrative edits — final wording waits on Phase 3's decision.

---

## 5. Status summary (end of this session)

| step | status |
| --- | --- |
| Invariant audit | ✓ both mechanisms PASS (§1) |
| Phase 1 code change | ✓ committed to working tree |
| Phase 1 GPU re-render | ✓ job 10848944 COMPLETED (54 s) |
| Phase 1 visual verification | ✓ clutter looks natural across 3 seeds |
| Phase 3 diagnostic code | ✓ `scripts/stage5_p3_turnfaucet_diagnostic.py` |
| Phase 3 GPU diagnostic | ✓ job 10848995 COMPLETED (9 m 7 s) |
| Phase 3 decision read | **PROCEED with 3-day cutoff** (§3) |
| Phase 2 (PickCube rotation) | not started (gated on Phase 3 outcome) |

## 6. Next session

1. Port v1 single-sweep into `babysteps/skills/turn.py`
   `compile_intent_to_turn_skill` (poke mode) — see §3 step 2–3.
2. Re-run `scripts/stage5_p3_turnfaucet_diagnostic.py` to measure lift.
3. If ≥ 30% — start Phase 2 (PickCube rotation, with the
   `face_inaccessible` predicate per the doc's Phase 2 re-examination).
4. If < 30% after 3 days — invoke the appendix path; commit the
   4-task narrative.

## 7. Working-tree state to commit (when ready)

- `babysteps/render/pushcube.py` — Phase 1 clutter object.
- `scripts/stage5_p3_turnfaucet_diagnostic.py` — diagnostic.
- `slurm/render_pushcube_clutter.sbatch` — Phase 1 re-render job.
- `slurm/stage5_p3_turnfaucet_diagnostic.sbatch` — diagnostic job.
- `reports/stage5/redesign_phase1_phase3/findings.md` — this report.
- `reports/stage5/turnfaucet_diagnostic/{per_seed.jsonl,summary.md}` — diag output.
- `redesign_failure_paradigm.md` (in `/home/wang4433/scratch/babysteps/`) — plan doc with audit results + status pointer.

I have NOT committed anything; per the project convention you commit when ready.
