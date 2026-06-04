# Stage-5 — PlugCharger latent scout: `charger_xy` (secondary) — PlugCharger-v1

**Factor:** charger lateral position in the receptacle frame (which side of the slot) → maps to intent factor `approach_direction`.
**Question:** can the DEPLOYED frozen encoder (DINOv2 ViT-B/14, spatial_mean,
resolution 224) read this factor from the natural third-person reset frame
(the demo's opening view)? This is the necessary-condition ceiling for latent-grounding
the factor as a candidate second latent end-to-end task.

- Encoder: `dinov2_vitb14` spatial_mean @ resolution 224 (224 = the deployed P1 encoder exactly, so a PASS is deployment-honest).
- n=120 reset frames (120 seeds), 2-class label {'left': 60, 'right': 60} (majority 0.500).
- Labels are charger-RELATIVE (de-rotated into the receptacle/camera frame) so the probe cannot exploit the constant in-frame receptacle/background.
- Probe: G1 protocol (IntentHead F=6, d_slot=32, n_epochs=300, factor_idx=`approach_direction`) + direct StandardScaler+LR.

## Result

| feature | dim | majority | shuffled | direct LR ± std | IntentHead-CV ± std | gate |
|---|---|---|---|---|---|---|
| `global_dino (spatial_mean)` | 768 | 0.500 | 0.483 | 0.842 ± 0.041 | 0.842 ± 0.067 | FAIL |

## Verdict

- IntentHead-CV **0.842** vs majority **0.500**, shuffled **0.508**, gate **0.90**.
- WEAK LIFT over baselines but below the 0.90 gate. Partially groundable; run the --resolutions 518 scale follow-up before deciding. Not yet a clean latent cell.

## Honesty boundary

- **CONTROL mode (`--fix-receptacle`):** the receptacle is pinned to a canonical pose, so the receptacle-mounted camera — and the entire background (table + robot) — is IDENTICAL in every frame; only the charger varies. Any separability here MUST come from the charger itself, not the background-rotation cue. Compare to the deployed run to read off how much of the deployed number was background.
- **Initial-state factor:** the charger sits on the table at reset, fully visible from frame 0, so the single reset frame IS the deployed signal — there is NO StackCube-style config→clip discount (that trap was specific to final-state factors like goal_state).
- **No sim privilege in the encoded signal:** poses are read only to author the class labels; the encoder reads pixels and no coordinate is fed to the probe (CLAUDE.md invariant #4).

