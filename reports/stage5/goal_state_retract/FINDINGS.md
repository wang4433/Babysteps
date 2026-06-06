# Stage-5 goal_state — retract-gripper demo render UNBLOCKS StackCube `goal_state`

**Job 10969792 (2026-06-05). Verdict: retract + final-state pooling clears the
0.90 gate (IntentHead-CV `first_last` 0.908 PASS; direct-LR 0.90-0.93). Borderline
at n=120; needs larger-n confirmation.**

## Setup

`scripts/stage5_goal_state_probe.py --mode clip-pool`, DINOv2 ViT-B/14, default
camera, StackCube-v1 seeds 0-59 (n=120, majority 0.500). Control = no retract;
test = `--retract` (lift the open gripper up-and-back, dwell, so the final frames
show the clean placement without the occluding gripper).

| pooling | control direct-LR | control CV | retract direct-LR | retract CV | gate |
|---|---|---|---|---|---|
| spatial_mean (deployed) | 0.700 | 0.633 | 0.775 | 0.750 | FAIL |
| last5_mean | 0.825 | 0.817 | **0.933** | 0.833 | FAIL |
| first_last | 0.708 | 0.683 | 0.900 | **0.908** | **PASS** |
| final_frame | 0.792 | 0.758 | 0.925 | 0.825 | FAIL |

Armless static-config ceiling (committed): 0.99.

## Reading

- **The retract works.** Saved frames confirm the arm lifts clear and the cubes
  read as a tower (stack) vs side-by-side (near), unoccluded — matching the
  config configs. No scene auto-reset from stepping past the place.
- **goal_state is a FINAL-STATE factor.** The deployed whole-clip `spatial_mean`
  dilutes it even after retract (0.750); final-state poolings recover it
  (`first_last` CV 0.908 PASS; `last5_mean`/`final_frame` direct-LR 0.93/0.925).
- **Underpowered at n=120.** direct-LR (0.90-0.93) sits well above IntentHead-CV
  (0.83-0.91) with high fold std (±0.07-0.18) -> the features separate; the
  IntentHead is the small-n bottleneck. Larger n should firm the CV toward the
  direct-LR (~0.93). Confirmation job: seeds 0-149 (n=300).

## Two levers, both needed (and neither is the camera/encoder)

The StackCube goal_state clip-block decomposes into:
1. **gripper occlusion at the final frame** -> fixed by the retract render;
2. **whole-clip mean-pool diluting a final-state factor** -> fixed by final-state
   pooling.
The camera viewpoint (job 10969709) and the encoder (DINOv2/DINOv3/V-JEPA, prior
work) are NOT the levers. The lever is a goal-disambiguating (retract) demo render
+ factor-matched (final-state) pooling.

## Paper-framing caveat

The deployed P1 encoder pools `spatial_mean` uniformly across tasks/factors. Using
final-state pooling for goal_state is a per-factor change; it is defensible as
*principled* (goal_state is a final-state factor; object_motion/contact_region are
trajectory/contact factors), but must be stated as such, not as per-task tuning.

## n=300 confirmation (job 10969827, seeds 0-149)

The borderline n=120 PASS firms up at n=300 — the `first_last` variance collapses:

| pooling | direct-LR | IntentHead-CV | gate |
|---|---|---|---|
| spatial_mean (deployed) | 0.830 | 0.767 ± 0.136 | FAIL |
| last5_mean | 0.967 | 0.867 ± 0.183 | FAIL |
| **first_last** | 0.937 | **0.920 ± 0.019** | **PASS** |
| final_frame | 0.967 | 0.867 ± 0.183 | FAIL |

`first_last` (spatial_mean over {first, last} frames) is the robust pooling:
0.920 ± **0.019** at n=300 (vs 0.908 ± 0.067 at n=120). last5/final have high
IntentHead-CV variance (±0.18) despite high direct-LR — first_last's start-anchor
stabilizes the IntentHead. **goal_state grounding CONFIRMED: retract render +
first_last pooling = 0.920 solid PASS.** The goal_state pack uses this pooling.

Reports: `reports/stage5/goal_state_retract{,_n300}/StackCube-v1/{,retract/}report.md`.
