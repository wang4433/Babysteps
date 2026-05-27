# Stage-5 P2 VLM Attribution — Cross-task Summary

| task | C1 attr acc | rule-table | C1 pres | C2 pres | Δpres pp | C1 succ | C2 succ | Δsucc pp |
|---|---|---|---|---|---|---|---|---|
| PushCube-v1 | 1.000 | 1.000 | 1.000 | 1.000 | +0.0 | 0.980 | 0.960 | +2.0 |
| PickCube-v1 | 1.000 | 1.000 | 1.000 | 1.000 | +0.0 | 0.920 | 0.000 | +92.0 |
| StackCube-v1 | 0.860 | 0.860 | 1.000 | 0.500 | +50.0 | 0.700 | 0.220 | +48.0 |
| TurnFaucet-v1 | 1.000 | 0.500 | 1.000 | 1.000 | +0.0 | 0.040 | 0.020 | +2.0 |
| CrossViewPush-v1 | 1.000 | 0.000 | 1.000 | 1.000 | +0.0 | 1.000 | 1.000 | +0.0 |

## Gates

- **PushCube-v1**: attr PASS · pres PASS · succ PASS
- **PickCube-v1**: attr PASS · pres PASS · succ PASS
- **StackCube-v1**: attr PASS · pres PASS · succ PASS
- **TurnFaucet-v1**: attr PASS · pres PASS · succ PASS
- **CrossViewPush-v1**: attr PASS · pres PASS · succ PASS
