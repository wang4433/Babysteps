# Stage-5 dual-camera — does a high-oblique demo camera unblock StackCube `goal_state`?

**Job 10969709 (2026-06-05). Verdict: NO — falsified. The clip-block is
viewpoint-invariant; the cause is gripper occlusion at the final frame, not the
camera angle.**

## Setup

`scripts/stage5_goal_state_probe.py --mode clip-pool`, DINOv2 ViT-B/14
(spatial_mean), StackCube-v1 seeds 0-59 (n=120, majority 0.500), four
`render_camera` viewpoints (`--camera`):

| camera | elevation | deployed `spatial_mean` | best `last5_mean` |
|---|---|---|---|
| `default` (ManiSkill) | ~15° | 0.633 | **0.817** |
| `oblique_high` | ~51° | 0.633 | 0.792 |
| `oblique_higher` | ~69° | 0.575 | 0.792 |
| `oblique_corner` | ~49° | 0.533 | **0.825** |

All cells FAIL the 0.90 gate. The best oblique (0.825) ties the default (0.817);
the steepest view (`oblique_higher`) is the **worst** on the deployed pooling
(0.575). Static armless config ceiling (committed): **0.99**.

## Why (saved frames, seed 0)

At the demo's final frame the gripper has just released and has **not
retracted**, so it sits directly over the placed cubes. A higher/steeper camera
looks *down the same axis the gripper descends*, so it occludes the cubes
**more**, not less (in `oblique_higher` only a sliver of cube is visible). There
is no exterior viewpoint that "looks over" a gripper co-located with the cubes.

So the config(0.99) → clip(~0.82) gap is **not** the viewpoint. It is:
1. the un-retracted gripper occluding the placement at the final frame (from any angle);
2. small cubes (low effective resolution for the A-on-B relation);
3. whole-clip mean-pool dilution (partially recovered by `last5_mean`, ~0.82).

## Consequences

- **Rebuts the "boundary = camera placement" reviewer attack** with the decisive
  control the prior panel said was missing: the goal_state clip-block **survives
  a full oblique camera sweep** (~15°→69°). It is a property of the demo clip's
  final-state content, not where we point the camera.
- **The real lever is a retract-gripper final-state demo render** (let the arm
  move clear, dwell, capture the clean final frames) + final-state pooling — the
  0.99 config ceiling predicts this should clear the gate. This is the concrete
  meaning of the "goal-disambiguating demo render" lever; it is NOT a camera or
  encoder change.
- **Dual-camera plan:** the Camera-1 (high-oblique global) premise for unblocking
  StackCube `goal_state` is falsified. The dual-stream architecture remains
  useful for PushCube → PickCube `contact_region` (occluded grasp), but it is
  decoupled from the goal_state attack.

Reports: `reports/stage5/goal_state_camera/StackCube-v1/{,oblique_*}/report.md`;
frames under each `frames/`.
