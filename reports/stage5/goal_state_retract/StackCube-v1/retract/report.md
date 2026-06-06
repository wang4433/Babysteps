# Stage-5 — StackCube `goal_state` clip temporal-pooling ladder — StackCube-v1

**Question:** the static final config separates at 0.99 but the deployed
spatial_mean-over-clip pooling fell to 0.65. Is that mean-pooling DILUTION
(a final-state-aware pooling recovers it) or a representation block?

- Encoder: `dinov2_vitb14`; same clips, several temporal poolings (frame
  subsets pooled by the frozen encoder's spatial_mean).
- n=120 (60 seeds x 2 demo CLIPS), label dist {'cubeA_on_cubeB': 60, 'cube_at_target': 60} (majority 0.500); place-near drop = cubeB.xy + 0.06 (balanced).
- Probe: G1 protocol (IntentHead F=6, d_slot=32, n_epochs=300, factor_idx=goal_state) + direct LR.

## Temporal-pooling ladder

| pooling | dim | majority | shuffled | direct LR ± std | IntentHead-CV ± std | gate |
|---|---|---|---|---|---|---|
| `spatial_mean (all)` | 768 | 0.500 | 0.633 | 0.775 ± 0.050 | 0.750 ± 0.083 | FAIL |
| `last5_mean` | 768 | 0.500 | 0.592 | 0.933 ± 0.068 | 0.833 ± 0.177 | FAIL |
| `first_last` | 768 | 0.500 | 0.708 | 0.900 ± 0.050 | 0.908 ± 0.067 | PASS |
| `final_frame` | 768 | 0.500 | 0.592 | 0.925 ± 0.067 | 0.825 ± 0.176 | FAIL |

## Verdict

- Deployed `spatial_mean (all)` IntentHead-CV **0.750** (gate 0.90); best final-state pooling **`first_last` 0.908**.
- POOLING DILUTION — goal_state IS pixel-groundable, but the DEPLOYED spatial_mean-over-clip pooling dilutes it (0.750); a final-state-aware pooling (`first_last` 0.908) recovers it. goal_state is a FINAL-STATE factor; mean-over-trajectory is the wrong pooling for it. Latent-viable ONLY with a per-factor pooling change (caveat: the deployed encoder uses spatial_mean uniformly across tasks / factors -> a StackCube-specific pooling is a reviewer-visible inconsistency unless framed as principled final-state pooling).

## Honesty boundary

- Cube poses / drop targets are injected ONLY to author the two demo clips; the encoder reads pixels, no coordinate is fed to the probe.
- A final-state pooling that 'works' here is NOT free: the deployed P1 encoder pools spatial_mean uniformly. Adopting a goal_state-specific pooling needs a principled justification (final-state vs motion/contact factors), or it reads as per-task tuning.

