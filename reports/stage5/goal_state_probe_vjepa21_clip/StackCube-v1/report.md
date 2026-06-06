# Stage-5 — StackCube `goal_state` clip temporal-pooling ladder — StackCube-v1

**Question:** the static final config separates at 0.99 but the deployed
spatial_mean-over-clip pooling fell to 0.65. Is that mean-pooling DILUTION
(a final-state-aware pooling recovers it) or a representation block?

- Encoder: `vjepa2_1_vitl16`; same clips, several temporal poolings (frame
  subsets pooled by the frozen encoder's spatial_mean).
- n=120 (60 seeds x 2 demo CLIPS), label dist {'cubeA_on_cubeB': 60, 'cube_at_target': 60} (majority 0.500); place-near drop = cubeB.xy + 0.06 (balanced).
- Probe: G1 protocol (IntentHead F=6, d_slot=32, n_epochs=300, factor_idx=goal_state) + direct LR.

## Temporal-pooling ladder

| pooling | dim | majority | shuffled | direct LR ± std | IntentHead-CV ± std | gate |
|---|---|---|---|---|---|---|
| `spatial_mean (all)` | 1024 | 0.500 | 0.525 | 0.817 ± 0.068 | 0.783 ± 0.049 | FAIL |
| `last5_mean` | 1024 | 0.500 | 0.575 | 0.833 ± 0.075 | 0.833 ± 0.075 | FAIL |
| `first_last` | 1024 | 0.500 | 0.525 | 0.808 ± 0.082 | 0.808 ± 0.082 | FAIL |
| `final_frame` | 1024 | 0.500 | 0.508 | 0.842 ± 0.081 | 0.808 ± 0.068 | FAIL |

## Verdict

- Deployed `spatial_mean (all)` IntentHead-CV **0.783** (gate 0.90); best final-state pooling **`last5_mean` 0.833**.
- PARTIAL — even the best pooling (`last5_mean` 0.833) lifts over chance but misses the 0.90 gate. goal_state is only weakly clip-groundable; not a clean latent cell.

## Honesty boundary

- Cube poses / drop targets are injected ONLY to author the two demo clips; the encoder reads pixels, no coordinate is fed to the probe.
- A final-state pooling that 'works' here is NOT free: the deployed P1 encoder pools spatial_mean uniformly. Adopting a goal_state-specific pooling needs a principled justification (final-state vs motion/contact factors), or it reads as per-task tuning.

