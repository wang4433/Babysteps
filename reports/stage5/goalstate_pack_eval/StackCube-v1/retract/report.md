# Stage-5 — StackCube `goal_state` clip temporal-pooling ladder — StackCube-v1

**Question:** the static final config separates at 0.99 but the deployed
spatial_mean-over-clip pooling fell to 0.65. Is that mean-pooling DILUTION
(a final-state-aware pooling recovers it) or a representation block?

- Encoder: `dinov2_vitb14`; same clips, several temporal poolings (frame
  subsets pooled by the frozen encoder's spatial_mean).
- n=100 (50 seeds x 2 demo CLIPS), label dist {'cubeA_on_cubeB': 50, 'cube_at_target': 50} (majority 0.500); place-near drop = cubeB.xy + 0.06 (balanced).
- Probe: G1 protocol (IntentHead F=6, d_slot=32, n_epochs=300, factor_idx=goal_state) + direct LR.

## Temporal-pooling ladder

| pooling | dim | majority | shuffled | direct LR ± std | IntentHead-CV ± std | gate |
|---|---|---|---|---|---|---|
| `spatial_mean (all)` | 768 | 0.500 | 0.540 | 0.730 ± 0.121 | 0.720 ± 0.154 | FAIL |
| `last5_mean` | 768 | 0.500 | 0.350 | 0.920 ± 0.093 | 0.910 ± 0.073 | PASS |
| `first_last` | 768 | 0.500 | 0.360 | 0.870 ± 0.108 | 0.860 ± 0.086 | FAIL |
| `final_frame` | 768 | 0.500 | 0.380 | 0.930 ± 0.098 | 0.910 ± 0.073 | PASS |

## Verdict

- Deployed `spatial_mean (all)` IntentHead-CV **0.720** (gate 0.90); best final-state pooling **`final_frame` 0.910**.
- POOLING DILUTION — goal_state IS pixel-groundable, but the DEPLOYED spatial_mean-over-clip pooling dilutes it (0.720); a final-state-aware pooling (`final_frame` 0.910) recovers it. goal_state is a FINAL-STATE factor; mean-over-trajectory is the wrong pooling for it. Latent-viable ONLY with a per-factor pooling change (caveat: the deployed encoder uses spatial_mean uniformly across tasks / factors -> a StackCube-specific pooling is a reviewer-visible inconsistency unless framed as principled final-state pooling).

## Honesty boundary

- Cube poses / drop targets are injected ONLY to author the two demo clips; the encoder reads pixels, no coordinate is fed to the probe.
- A final-state pooling that 'works' here is NOT free: the deployed P1 encoder pools spatial_mean uniformly. Adopting a goal_state-specific pooling needs a principled justification (final-state vs motion/contact factors), or it reads as per-task tuning.

