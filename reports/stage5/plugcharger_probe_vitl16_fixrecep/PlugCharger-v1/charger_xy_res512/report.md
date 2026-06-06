# Stage-5 — PlugCharger latent scout: `charger_xy` (secondary) — PlugCharger-v1

**Factor:** charger lateral position in the receptacle frame (which side of the slot) → maps to intent factor `approach_direction`.
**Question:** can the DEPLOYED frozen encoder (DINOv2 ViT-B/14, spatial_mean,
resolution 512) read this factor from the natural third-person reset frame
(the demo's opening view)? This is the necessary-condition ceiling for latent-grounding
the factor as a candidate second latent end-to-end task.

- Encoder: `dinov3_vitl16` spatial_mean @ resolution 512 (224 = the deployed P1 encoder exactly, so a PASS is deployment-honest).
- n=120 reset frames (120 seeds), 2-class label {'left': 60, 'right': 60} (majority 0.500).
- Labels are charger-RELATIVE (de-rotated into the receptacle/camera frame) so the probe cannot exploit the constant in-frame receptacle/background.
- Probe: G1 protocol (IntentHead F=6, d_slot=32, n_epochs=300, factor_idx=`approach_direction`) + direct StandardScaler+LR.

## Result

| feature | dim | majority | shuffled | direct LR ± std | IntentHead-CV ± std | gate |
|---|---|---|---|---|---|---|
| `global_dino (spatial_mean)` | 1024 | 0.500 | 0.492 | 0.942 ± 0.042 | 0.950 ± 0.061 | PASS |

## Verdict

- IntentHead-CV **0.950** vs majority **0.500**, shuffled **0.425**, gate **0.90**.
- PASS — frozen DINOv2 spatial_mean separates this factor at the DEPLOYED representation. It is an INITIAL-STATE factor (fully visible from frame 0), so this config number IS the deployed number — no StackCube config->clip discount applies. A genuine second latent-viable factor beyond PushCube.

## Honesty boundary

- **CONTROL mode (`--fix-receptacle`):** the receptacle is pinned to a canonical pose, so the receptacle-mounted camera — and the entire background (table + robot) — is IDENTICAL in every frame; only the charger varies. Any separability here MUST come from the charger itself, not the background-rotation cue. Compare to the deployed run to read off how much of the deployed number was background.
- **Initial-state factor:** the charger sits on the table at reset, fully visible from frame 0, so the single reset frame IS the deployed signal — there is NO StackCube-style config→clip discount (that trap was specific to final-state factors like goal_state).
- **No sim privilege in the encoded signal:** poses are read only to author the class labels; the encoder reads pixels and no coordinate is fed to the probe (CLAUDE.md invariant #4).

