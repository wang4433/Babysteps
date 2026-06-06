# Stage-5 — PlugCharger latent scout: `charger_xy` (secondary) — PlugCharger-v1

**Factor:** charger lateral position in the receptacle frame (which side of the slot) → maps to intent factor `approach_direction`.
**Question:** can the DEPLOYED frozen encoder (DINOv2 ViT-B/14, spatial_mean,
resolution 224) read this factor from the natural third-person reset frame
(the demo's opening view)? This is the necessary-condition ceiling for latent-grounding
the factor as a candidate second latent end-to-end task.

- Encoder: `dinov3_vitb16` spatial_mean @ resolution 224 (224 = the deployed P1 encoder exactly, so a PASS is deployment-honest).
- n=120 reset frames (120 seeds), 2-class label {'left': 60, 'right': 60} (majority 0.500).
- Labels are charger-RELATIVE (de-rotated into the receptacle/camera frame) so the probe cannot exploit the constant in-frame receptacle/background.
- Probe: G1 protocol (IntentHead F=6, d_slot=32, n_epochs=300, factor_idx=`approach_direction`) + direct StandardScaler+LR.

## Result

| feature | dim | majority | shuffled | direct LR ± std | IntentHead-CV ± std | gate |
|---|---|---|---|---|---|---|
| `global_dino (spatial_mean)` | 768 | 0.500 | 0.533 | 0.850 ± 0.033 | 0.867 ± 0.061 | FAIL |

## Verdict

- IntentHead-CV **0.867** vs majority **0.500**, shuffled **0.492**, gate **0.90**.
- WEAK LIFT over baselines but below the 0.90 gate. Partially groundable; run the --resolutions 518 scale follow-up before deciding. Not yet a clean latent cell.

## Honesty boundary

- **Camera mounted on the receptacle** (`mount=self.receptacle`) — and the background is NOT cancelled. When the receptacle rotates (yaw ±22.5°) the camera co-rotates, so the table + robot appear to counter-rotate: receptacle yaw leaks into the image as a global background rotation. The `receptacle_yaw` negative control reads ~0.96, FALSIFYING the naive camera-cancellation assumption. Because charger-relative labels share a partial correlation with this cue, the DEPLOYED charger numbers are CONFOUNDED — see the `--fix-receptacle` control, which removes the background and is the honest measure of charger groundability.
- **Initial-state factor:** the charger sits on the table at reset, fully visible from frame 0, so the single reset frame IS the deployed signal — there is NO StackCube-style config→clip discount (that trap was specific to final-state factors like goal_state).
- **No sim privilege in the encoded signal:** poses are read only to author the class labels; the encoder reads pixels and no coordinate is fed to the probe (CLAUDE.md invariant #4).

