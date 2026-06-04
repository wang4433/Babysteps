# Stage-5 — StackCube `goal_state` clip temporal-pooling ladder — StackCube-v1

**Question:** the static final config separates at 0.99 but the deployed
spatial_mean-over-clip pooling fell to 0.65. Is that mean-pooling DILUTION
(a final-state-aware pooling recovers it) or a representation block?

- Encoder: `dinov3_vitl16`; same clips, several temporal poolings (frame
  subsets pooled by DINOv2 spatial_mean).
- n=120 (60 seeds x 2 demo CLIPS), label dist {'cubeA_on_cubeB': 60, 'cube_at_target': 60} (majority 0.500); place-near drop = cubeB.xy + 0.06 (balanced).
- Probe: G1 protocol (IntentHead F=6, d_slot=32, n_epochs=300, factor_idx=goal_state) + direct LR.

## Temporal-pooling ladder

| pooling | dim | majority | shuffled | direct LR ± std | IntentHead-CV ± std | gate |
|---|---|---|---|---|---|---|
| `spatial_mean (all)` | 1024 | 0.500 | 0.633 | 0.733 ± 0.068 | 0.717 ± 0.081 | FAIL |
| `last5_mean` | 1024 | 0.500 | 0.725 | 0.825 ± 0.072 | 0.817 ± 0.073 | FAIL |
| `first_last` | 1024 | 0.500 | 0.608 | 0.783 ± 0.072 | 0.800 ± 0.113 | FAIL |
| `final_frame` | 1024 | 0.500 | 0.658 | 0.817 ± 0.114 | 0.817 ± 0.114 | FAIL |

## Verdict

- Deployed `spatial_mean (all)` IntentHead-CV **0.717** (gate 0.90); best final-state pooling **`final_frame` 0.817**.
- PARTIAL — even the best pooling (`final_frame` 0.817) lifts over chance but misses the 0.90 gate. goal_state is only weakly clip-groundable; not a clean latent cell.

## Honesty boundary

- Cube poses / drop targets are injected ONLY to author the two demo clips; the encoder reads pixels, no coordinate is fed to the probe.
- A final-state pooling that 'works' here is NOT free: the deployed P1 encoder pools spatial_mean uniformly. Adopting a goal_state-specific pooling needs a principled justification (final-state vs motion/contact factors), or it reads as per-task tuning.

