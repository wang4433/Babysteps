# Stage-5 — StackCube `goal_state` pixel-separability probe (clip) — StackCube-v1

**Question:** can frozen DINOv2 (ViT-B/14, spatial_mean) separate the two
`goal_state` cases — cubeA stacked ON cubeB (`cubeA_on_cubeB`) vs cubeA
placed NEAR/beside cubeB (`cube_at_target`, the place-near reading)?
This is the necessary-condition ceiling for latent-grounding `goal_state`.

- Encoder: `dinov2_vitb14` spatial_mean (the deployed P1 encoder).
- n=120 (60 seeds x 2 demo CLIPS), label dist {'cubeA_on_cubeB': 60, 'cube_at_target': 60} (majority 0.500).
- Demo CLIPS (real Stage-0 executor), DINOv2 spatial_mean pooled over TIME+patches = the DEPLOYED representation: stack = cubeA_on_cubeB onto cubeB; near = cube_at_target dropped at cubeB.xy + 0.08 along a per-seed BALANCED cardinal. Tests whether clip-pooling dilutes the signal the static-config probe saw at 0.99.
- Probe: same protocol as the G1 cells (IntentHead F=6, d_slot=32, n_epochs=300, factor_idx=goal_state) + direct StandardScaler+LR.

## Result

| feature | dim | majority | shuffled | direct LR ± std | IntentHead-CV ± std | gate |
|---|---|---|---|---|---|---|
| `global_dino (spatial_mean)` | 768 | 0.500 | 0.425 | 0.733 ± 0.033 | 0.650 ± 0.077 | FAIL |

## Verdict

- IntentHead-CV **0.650** vs majority **0.500**, shuffled **0.575**, gate **0.90**.
- FAIL — frozen DINOv2 cannot separate the two goal_state configs even at the clean-config ceiling. goal_state is NOT latent-groundable with this encoder → consolidate on PushCube as the single end-to-end latent task.

## Honesty boundary

- **Config ceiling, not the deployed demo.** This encodes the clean canonical goal configs. The deployed StackCube demo is a CLIP that deliberately HIDES vertical motion (2D-trajectory info loss = Sub-project C goal-ambiguity). A PASS means the ambiguity is a demo-DESIGN choice, not a representation limit — a goal-disambiguating demo would make goal_state latent-viable. A FAIL kills it regardless.
- **No sim privilege in the encoded signal.** Cube poses are injected ONLY to author the two configs to render; the encoder reads pixels and no coordinate is ever fed to the probe (CLAUDE.md invariant #4).
- **Two-class, balanced** (majority 0.5) — a clean binary ceiling, not the 4-way object_motion problem.

