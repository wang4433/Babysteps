# Stage-4 Varied-Intent Schema-Recoverability Notes

## Run metadata

- Date: 2026-05-22
- Cut: the Stage-4 varied-intent collection
  (`docs/superpowers/specs/2026-05-22-stage4-varied-intent-cut-design.md`).
- Primary input JSONLs:
  - `datasets/stage4/varied_intent/PushCube-v1/samples.jsonl` (20 episodes;
    goal-move injection, binary `±x`).
  - `datasets/stage4/varied_intent/StackCube-v1/samples.jsonl` (40 episodes;
    rejection-sampled native, 4 directions balanced 10/10/10/10).
- Commit at run time: `41e9cfc` (HEAD on master).
- Probe: scikit-learn `LogisticRegression` (lbfgs, `max_iter=1000`), 5-fold
  `StratifiedKFold` with `LeaveOneOut` fallback when the smallest class
  has < 5 members; `seed=0`. Same as M1.
- Feature dim: 20 (`babysteps/stage4/features.py`) — was 19 during the
  initial 2026-05-22 run; the angle column was replaced by `[sin, cos]`
  on 2026-05-23 to fix circular wraparound (see "Update" section below).
- Cert: the three-way `cell_class` (`babysteps/stage4/report.py`):
  `trivially_constant` → excluded; `label_identity` → flagged, not counted;
  `geometric` → gated at `probe_acc_mean ≥ 0.90` **AND** clearing both
  `majority_class_acc` and `shuffled_features_acc` by `GATE_MARGIN = 0.10`.

## Headline result (post-2026-05-23 fix)

**12 cells | 2 geometric (2 pass / 0 fail) | 2 label-identity | 8 trivially
constant.** Both geometric cells clear the tightened gate. The constant-
factor loophole is closed. The cut is M2-ready as a meaningful
certification scaffold.

## Update (2026-05-23): circular angle encoding fixes StackCube

The original 2026-05-22 run reported "2 geometric (1 pass / 1 fail)" — see
the "Initial result (2026-05-22)" section below for the original write-up.
That FAIL was traced to a feature-extraction defect in
`babysteps/stage4/features.py`: the displacement direction was emitted as a
single scalar `angle = arctan2(dy, dx)`, which wraps at ±π. StackCube
`translate_-x` samples scattered across +157° and −177°, and the linear
probe could not separate them from `±y`. The diagnostic is reproducible:

```
class            angle_deg range          n
translate_+x   [-39, +36]                10
translate_+y   [+46, +134]               10
translate_-x   [-177, +157]  ← wraps    10
translate_-y   [-135, -76]               10
```

`disp_xy` already separates the four classes cleanly; the raw `angle`
column added noise that swamped that signal for the `-x` class. Encoding
as `[sin(angle), cos(angle)]` is continuous around the circle.

**Three feature variants on the same 40-episode cut** (same probe, same
splitter):

| variant | feat_dim | acc | `-x` correct |
| --- | --- | --- | --- |
| A: current (raw angle) | 19 | 0.725 ± 0.050 | 0 / 10 |
| B: angle → `[sin, cos]` | 20 | **0.950 ± 0.100** | **9 / 10** |
| C: angle removed | 18 | 0.725 ± 0.094 | 4 / 10 |

(Diagnostic script: `diagnostics/diag_angle_variants.py` + `.out`;
reproducible from `datasets/stage4/varied_intent/StackCube-v1/samples.jsonl`.
The script inlines the pre-fix raw-angle definition as variant A so it
keeps reproducing the 0.725 baseline after `features.py` lands the fix.)

The fix was a 1-dim addition to `features.py` (`FEATURE_DIM 19 → 20`)
guarded by a focused test
(`tests/test_stage4_features.py::test_angle_feature_is_continuous_at_pi_wraparound`).
PushCube `object_motion` is unaffected (already PASS at 0.95; binary ±x
does not exercise the wrap). All 361 sim-free tests green.

M1's original hypothesis — "balance + more episodes would clear 0.90 with
the same 19-dim features" — was correct in spirit; the missing step was
the circular-feature fix. With balance AND a continuous angle, the same
linear probe on hand-built features clears 0.95.

## Initial result (2026-05-22, pre-fix)

The varied cut achieved what the spec demanded
(§8.2: "geometric `object_motion` passes in ≥1 task") — PushCube cleared
the tightened gate at 0.95. **StackCube's 4-way `object_motion`
failed at 0.72 despite full class balance.** The contradiction with M1's
"balance lifts to 0.90" prediction was the symptom of the angle-wrap
defect; see the Update section above.

## Per-cell summary (post-fix; mirrors `schema_recoverability.md`)

```
# Stage-4 Schema-Recoverability Probe
Gate: geometric cells must reach probe_acc_mean >= 0.90 AND beat chance & shuffled each by 0.10.
Cells: 12 total | 2 geometric (2 pass / 0 fail) | 2 label-identity | 8 trivially constant.

### PushCube-v1
| factor              | class              | n_unique | n_episodes | majority | shuffled | probe ± std | gate           |
| ------------------- | ------------------ | -------- | ---------- | -------- | -------- | ----------- | -------------- |
| approach_direction  | label_identity     | 2        | 20         | 0.50     | 0.40     | 1.00 ± 0.00 | label_identity |
| constraint_region   | trivially_constant | 1        | 20         | 1.00     | 1.00     | 1.00 ± 0.00 | trivial        |
| contact_region      | label_identity     | 2        | 20         | 0.50     | 0.40     | 1.00 ± 0.00 | label_identity |
| embodiment_mapping  | trivially_constant | 1        | 20         | 1.00     | 1.00     | 1.00 ± 0.00 | trivial        |
| goal_state          | trivially_constant | 1        | 20         | 1.00     | 1.00     | 1.00 ± 0.00 | trivial        |
| object_motion       | geometric          | 3        | 20         | 0.50     | 0.30     | 0.95 ± 0.22 | PASS           |

### StackCube-v1
| factor              | class              | n_unique | n_episodes | majority | shuffled | probe ± std | gate           |
| ------------------- | ------------------ | -------- | ---------- | -------- | -------- | ----------- | -------------- |
| approach_direction  | trivially_constant | 1        | 40         | 1.00     | 1.00     | 1.00 ± 0.00 | trivial        |
| constraint_region   | trivially_constant | 1        | 40         | 1.00     | 1.00     | 1.00 ± 0.00 | trivial        |
| contact_region      | trivially_constant | 1        | 40         | 1.00     | 1.00     | 1.00 ± 0.00 | trivial        |
| embodiment_mapping  | trivially_constant | 1        | 40         | 1.00     | 1.00     | 1.00 ± 0.00 | trivial        |
| goal_state          | trivially_constant | 1        | 40         | 1.00     | 1.00     | 1.00 ± 0.00 | trivial        |
| object_motion       | geometric          | 4        | 40         | 0.25     | 0.38     | 0.95 ± 0.10 | PASS           |
```

Machine-readable form: `schema_recoverability.json` (this directory).

## Two execution-side fixes the GPU run forced

The collection job aborted at the fail-fast injection spike on its first
attempt (one direction succeeded, three failed). Debugging surfaced two real
bugs the sim-free reviews could not have caught:

1. **`PushCubeEnvRunner.run()` re-resets the env and discarded the injection
   done in `reset()`.** The push waypoints were correctly built from the
   injected scene but executed against the (re-set) native layout, so the
   gripper swung through empty space and the cube never moved. `+x` masked it
   because its injected position coincides with native. Fix: extracted
   `_reset_with_injection()`, called after EVERY `env.reset` (idempotent).
2. **Cube-move injection** (the spec's primary mechanism) **hit a reachability
   wall**: even with #1 fixed, the `+x`-tuned open-loop push controller can't
   drive the cube from displaced positions. The closest lateral injection
   positions got a useless graze; the far ones weren't touched at all. Pivoted
   to **goal-move** (the spec's documented fallback: keep cube at native
   reachable pose, move `goal_region` instead). Goal-move robustly recovers
   `+x` AND `-x` (5/5 seeds each in the diagnostic); `±y` lateral pushes
   remain unreliable (the gripper grazes `+x` instead of pushing sideways,
   a controller/contact limitation). So **PushCube varies binary ±x only**,
   relaxing the spec's "≥3 directions per task" to "≥2" for PushCube.
   StackCube (pick-and-place, reaches everywhere) carries the 4-direction
   headline unchanged.

Both fixes landed in commit `41e9cfc`; the diagnostic scripts + their
outputs are archived in `diagnostics/` next to this file
(`diag_injection.{py,sbatch,out,err}` for the cube-move investigation,
`diag_goalmove.{py,sbatch,out,err}` for the pivot).

## Cells that passed

### `PushCube-v1 / object_motion` — probe 0.95 ± 0.22, geometric PASS

20 episodes labelled `{translate_+x: 10, translate_-x: 9, translate_-y: 1}`.
The lone `-y` is a stray: one `-x` demo rollout had a noisy lateral component
that snapped past the y-axis on `goal_direction_to_motion`'s dominant-axis
tiebreak. LOO is triggered (`min class = 1`), and the probe gets 19/20 correct
— the singleton `-y` is the unavoidable miss (LOO cannot predict a class with
one sample). 0.95 clears 0.90 **and** beats chance (0.50) + shuffled (0.30)
by `> 0.10` on both. The signal is the demo trajectory's `dx` sign, which is
exactly what the cert wants: a genuine geometric inference from the rollout,
not a label echo. (Pre-fix this cell also reported 0.95; binary ±x does not
straddle the ±π wrap, so the angle column was harmless on PushCube.)

## Cells that previously did not pass — and the fix

### `StackCube-v1 / object_motion` — was 0.72 FAIL, now 0.95 PASS

40 episodes, 4 classes balanced **exactly 10/10/10/10**, 5-fold
StratifiedKFold (no LOO fallback). The pre-fix 0.72 result was driven
**not** by a representational ceiling of "the 19-dim hand features are too
weak for 4-way" but by **one defective feature column**: the raw `arctan2`
angle. Three candidate contributors were considered (path-length noise,
linear-probe insufficiency, and angle wrap); the diagnostic in the Update
section above isolated the angle wrap as the sole driver. With angle
encoded as `[sin, cos]`, the same linear probe on the same 40 episodes
reaches **0.95 ± 0.10**, with the `translate_-x` class recovered 9/10
(see the Update section's confusion-matrix variant B).

The geometric signal is the demo `disp_xy`, augmented by direction
`(sin θ, cos θ)`. `LogisticRegression` is more than sufficient on the
20-dim feature set.

## Label-identity cells (informational)

PushCube `contact_region` and `approach_direction` each take 2 values across
the cut (one per push direction), and both are recovered at 1.00 — but
recovery is **tautological**: `contact_region_label` is fed in as a one-hot
input feature, and PushCube's `approach_direction = face_to_approach(contact)`
is a deterministic function of the contact one-hot. The cert correctly flags
both as `label_identity` and excludes them from the geometric headline.

## Trivially constant cells (informational)

Eight cells. PushCube's `goal_state`/`constraint_region`/`embodiment_mapping`
and all StackCube intent factors except `object_motion` carry a single
canonical value across their task's episodes. The cert correctly excludes them
from the gate (a constant predictor scores 1.0 but does not certify anything
about the latent).

## Implications for Stage-4 Milestone 2

The varied cut + the circular-angle fix together achieve the spec purpose:
the certification loophole is closed AND both geometric cells clear the
tightened gate. The pipeline produces an **evaluable, non-trivial,
and now-passing** geometric gate using **hand-built 20-dim features**.

This re-frames the M2 target. The pre-fix framing — "M2 must lift
StackCube object_motion above 0.90" — is no longer falsifiable, because
hand features already reach 0.95. M2 must instead show that a learned
latent **adds value the hand features cannot supply**. Candidate
M2 targets, in increasing strength:

1. **Match the hand-feature baseline from richer inputs.** The current
   probe input is the privileged 2-D `object_trajectory`. M2 should run
   the same probe on slot vectors **extracted from `demo.rgbd_video`**
   (no privileged trajectory) and clear `≥ 0.90` on PushCube and
   StackCube `object_motion`. The hand-feature 0.95 becomes the
   reproducibility floor that M2 must match from raw evidence.
2. **Make more factors geometric.** Of the 12 cells, 8 are trivially
   constant and 2 are label-identity (the factor is fed in as a one-hot
   or deterministically derived from one). M2 should expand the
   varied-intent cut so `contact_region`/`approach_direction` (and
   eventually `goal_state`) vary per episode AND become recoverable
   from slot vectors WITHOUT the corresponding one-hot input. This
   closes the **label-identity loophole** the same way the varied cut
   closed the **constant-factor loophole**.
3. **Single-factor isolation.** Change one factor in the input scene
   while holding the rest fixed → exactly one slot should change in the
   learned latent. This is THE BABYSTEPS claim made measurable at the
   representation level (it is currently asserted only at the revision
   level via `tests/test_revision.py`).
4. **Frozen-slot revision in the loop.** Decode a revised slot back
   into an executable intent, run, and clear the Stage-0 acceptance gate
   (`delta_pp ≥ 10`) on the same cut.

M2 can now begin. The spec's "≥1 geometric pass" gate (§8.2) is exceeded
("2 geometric pass / 0 fail"); the cert scaffold works as designed; the
remaining handcrafted-feature ceiling has been honestly raised to 0.95 on
both tasks, which is the floor M2 must match from raw observations.

## Open notes (not blockers)

- PushCube's 1 stray `-y` label is a real artifact of `trajectory_to_motion`'s
  dominant-axis tiebreak under push noise; honest to leave it in. A future
  cleanup could drop the singleton (`n_unique` becomes 2, probe still passes)
  if needed.
- StackCube reaches `passed_acceptance=True` in its per-task `report.md`
  (episode-loop metrics: initial vs retry success). Post-fix the probe
  result agrees with the Stage-0 acceptance signal (both PASS); previously
  they disagreed because of the angle-wrap defect in the probe features.
- The diagnostic scripts (`diag_injection.py`, `diag_goalmove.py`) and their
  outputs are committed alongside this notes file under `diagnostics/`. The
  two execution-side fixes they drove are committed in `41e9cfc`.
- The two earlier reports `reports/stage4/schema_recoverability/` and
  `reports/stage4/schema_recoverability_union/` were generated with the
  pre-fix 19-dim features on the stage0_baselines data (where 17/18 cells
  are trivially constant). They are intentionally left as historical
  evidence; re-running them with the 20-dim features would not change the
  geometric-cell story on that data (only StackCube/object_motion is
  non-trivial there, with n=24 / n=48, and any new number would still be
  on the unbalanced original-pipeline data which is now superseded by
  this varied cut).
