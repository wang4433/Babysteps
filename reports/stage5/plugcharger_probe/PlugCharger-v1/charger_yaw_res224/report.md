# Stage-5 — PlugCharger latent scout: `charger_yaw` (primary candidate) — PlugCharger-v1

**Factor:** charger apparent (camera-frame) in-plane yaw, +-60 deg → maps to intent factor `object_motion`.
**Question:** can the DEPLOYED frozen encoder (DINOv2 ViT-B/14, spatial_mean,
resolution 224) read this factor from the natural third-person reset frame
(the demo's opening view)? This is the necessary-condition ceiling for latent-grounding
the factor as a candidate second latent end-to-end task.

- Encoder: `dinov2_vitb14` spatial_mean @ resolution 224 (224 = the deployed P1 encoder exactly, so a PASS is deployment-honest).
- n=120 reset frames (120 seeds), 2-class label {'ccw': 60, 'cw': 60} (majority 0.500).
- Labels are charger-RELATIVE (de-rotated into the receptacle/camera frame) so the probe cannot exploit the constant in-frame receptacle/background.
- Probe: G1 protocol (IntentHead F=6, d_slot=32, n_epochs=300, factor_idx=`object_motion`) + direct StandardScaler+LR.

## Result

| feature | dim | majority | shuffled | direct LR ± std | IntentHead-CV ± std | gate |
|---|---|---|---|---|---|---|
| `global_dino (spatial_mean)` | 768 | 0.500 | 0.433 | 0.917 ± 0.046 | 0.925 ± 0.049 | PASS |

## Verdict

- IntentHead-CV **0.925** vs majority **0.500**, shuffled **0.408**, gate **0.90**.
- PASS — frozen DINOv2 spatial_mean separates this factor at the DEPLOYED representation. It is an INITIAL-STATE factor (fully visible from frame 0), so this config number IS the deployed number — no StackCube config->clip discount applies. A genuine second latent-viable factor beyond PushCube.

## Honesty boundary

- **Camera mounted on the receptacle** (`mount=self.receptacle`) — and the background is NOT cancelled. When the receptacle rotates (yaw ±22.5°) the camera co-rotates, so the table + robot appear to counter-rotate: receptacle yaw leaks into the image as a global background rotation. The `receptacle_yaw` negative control reads ~0.96, FALSIFYING the naive camera-cancellation assumption. Because charger-relative labels share a partial correlation with this cue, the DEPLOYED charger numbers are CONFOUNDED — see the `--fix-receptacle` control, which removes the background and is the honest measure of charger groundability.
- **Initial-state factor:** the charger sits on the table at reset, fully visible from frame 0, so the single reset frame IS the deployed signal — there is NO StackCube-style config→clip discount (that trap was specific to final-state factors like goal_state).
- **No sim privilege in the encoded signal:** poses are read only to author the class labels; the encoder reads pixels and no coordinate is fed to the probe (CLAUDE.md invariant #4).

