# Stage-5 P1 vision-grounded eval — PushCube-v1

- Pack: `models/stage5/p1_vision/PushCube-v1`
- Features: `datasets/stage5/varied_intent/PushCube-v1/features` (spatial_mean DINOv2)
- Eval seeds: `100-149` (n=50)
- Real env: `True`

## Per-policy success rates

| policy | initial | final |
|---|---|---|
| `latent` | 0.000 | 0.960 |
| `babysteps_selective` | 0.000 | 0.980 |
| `same_intent_retry` | 0.000 | 0.000 |
| `oracle_factor_revision` | 0.000 | 0.980 |

## G4 / G5 gates (vs M2a's handcrafted-Z baseline)

- **G4**: Δpp(latent vs same_intent_retry) = +96.00pp (PASS, threshold ≥ 10)
- **G5**: Δpp(latent vs oracle) = -2.00pp (PASS, threshold ≥ -5)
