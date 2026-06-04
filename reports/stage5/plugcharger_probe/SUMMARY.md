# Stage-5 — PlugCharger-v1 latent-groundability scout — SUMMARY

**Date:** 2026-06-04 · **Question:** is there a *second* clean latent
end-to-end task beyond PushCube — i.e. a PlugCharger factor that the DEPLOYED
frozen encoder (DINOv2 ViT-B/14, `spatial_mean`) can separate from a
third-person demo frame, clearing the 0.90 IntentHead-CV gate *robustly* (not as
a confound artifact, the way StackCube `goal_state` was a clean-config artifact)?

**Verdict: NO.** PlugCharger does not yield a clean latent factor under the
deployed encoder. PushCube remains the single clean end-to-end latent task. This
is a rigorous negative — the candidate that looked best (`charger_yaw` = 0.93)
collapsed to FAIL once a confound was removed.

## The structural gotcha

PlugCharger's third-person human-render camera is `mount=self.receptacle`
(`plug_charger.py:74`): it is rigid in the receptacle body frame and co-rotates
with the receptacle. Consequence we initially mis-read as "receptacle yaw is
camera-cancelled / invisible" — but the camera rotating means the **background
(table + robot) appears to counter-rotate** with receptacle yaw, so receptacle
yaw leaks into the image as a global rotation. Verified by eye: deployed frames
show the table grain / horizon at different tilts across seeds; fix-receptacle
frames are background-identical (only the charger moves).

## Results (n=120 reset frames, 2-class median-split, IntentHead-CV)

| factor (→ intent) | role | deployed 224 | deployed 518 | fix-recep 224 | fix-recep 518 |
|---|---|---|---|---|---|
| `charger_yaw` → object_motion | primary | **0.925 PASS** | **0.908 PASS** | 0.833 FAIL | 0.858 FAIL |
| `charger_xy` → approach_direction | secondary | 0.858 FAIL | 0.875 FAIL | 0.842 FAIL | **0.942 PASS** |
| `receptacle_yaw` → constraint_region | neg. control | **0.958** ⚠ | **0.958** ⚠ | (constant — skipped) | (constant — skipped) |

Deployed = the as-shipped receptacle-mounted camera (background rotates).
Fix-receptacle = receptacle pinned to a canonical pose, so camera + background
are identical every frame and only the charger varies — any separability there
must come from the charger itself.

## Interpretation

1. **Negative control fired (0.958), falsifying the cancellation premise.** The
   `receptacle_yaw` cell is separable purely from the rotating background. A
   factor whose only image evidence is a global background rotation is not a
   manipulable charger intent — and its PASS flags that charger-relative labels
   (which correlate ≈ −0.35 with receptacle yaw) are *confounded* in deployed
   mode.

2. **`charger_yaw` is the headline negative.** Deployed it reads 0.93/0.91
   (PASS), but with the background frozen it drops to **0.83/0.86 (FAIL)** — so
   the deployed PASS was largely the background-rotation cue, not the charger.
   The charger's own ±60° orientation is *not* reliably groundable by frozen
   DINOv2: the asymmetric base (~40×30 mm, ~1–3 patches) is too small/coarse to
   read orientation, and the pegs are sub-patch. This is the PlugCharger version
   of the StackCube `config 0.99 → deployed 0.82` lesson, caught by the control.

3. **`charger_xy` is marginal/bespoke.** It only clears the gate (0.942) with a
   **fixed camera AND 518** resolution; at the deployed 224 (any camera) it is
   0.84–0.88 (FAIL). Reaching it requires a non-default camera plus a
   non-deployed resolution — bespoke task+encoder engineering, and the factor
   itself (which side the object starts on → approach_direction) is the trivial
   "where is the object" signal. Not a clean, deployment-honest latent cell.

## Honesty boundary

- **No sim privilege in the encoded signal:** poses are read only to author
  class labels; the encoder reads pixels, no coordinate is fed to the probe
  (CLAUDE.md invariant #4). Labels are charger-relative (de-rotated into the
  camera frame).
- **Initial-state factors:** all candidates are visible from frame 0, so a
  single reset frame is the deployed signal — there is no StackCube-style
  config→clip discount here (that trap was specific to final-state factors).
  The discount that *did* apply was a different one: a background-rotation
  confound, removed by the fix-receptacle control.
- **What would it take to revive PlugCharger?** A world-fixed third-person
  camera (not the shipped receptacle-mounted one) + 518 resolution would likely
  make `charger_xy` (approach side) a clean ~0.94 cell — but that is a task +
  observation + encoder redesign, not a use of the deployed interface, and only
  buys the weakest factor. Not pursued; logged as a lever, per the "don't force
  invisible/marginal factors into the frozen DINOv2 interface" guidance.

## Contribution

The methodology is itself a clean result for the groundability map: a negative
control + a confound-removing control turned a tempting 0.93 false-positive into
an honest FAIL, and pinned *why* (a camera-mount-induced background rotation).
This strengthens, not weakens, the "PushCube is the single clean latent task"
story — the negative ladder now spans PickCube (invisible), StackCube
object_motion (rep-blocked) / goal_state (final-state clip dilution), and
PlugCharger (background confound + sub-patch object orientation).

Artifacts: `reports/stage5/plugcharger_probe/PlugCharger-v1/<factor>_res<res>/`
(deployed) and `reports/stage5/plugcharger_probe_fixrecep/PlugCharger-v1/<factor>_res<res>/`
(control). Reproduce: `slurm/stage5_plugcharger_probe.sbatch` +
`slurm/stage5_plugcharger_fixrecep.sbatch`.
