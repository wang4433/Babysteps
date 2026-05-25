# Stage-5 P2 VLM Attribution — Cross-task Summary

| task | C1 attr acc | rule-table | C1 pres | C2 pres | Δpres pp | C1 succ | C2 succ | Δsucc pp |
|---|---|---|---|---|---|---|---|---|
| PushCube-v1 | 1.000 | 1.000 | 1.000 | 1.000 | +0.0 | 0.980 | 0.980 | +0.0 |
| PickCube-v1 | 0.960 | 1.000 | 1.000 | 1.000 | +0.0 | 0.900 | 0.000 | +90.0 |
| StackCube-v1 | 0.000 | 0.860 | 1.000 | 0.380 | +62.0 | 0.000 | 0.000 | +0.0 |

## Gates

- **PushCube-v1**: attr PASS · pres PASS · succ PASS
- **PickCube-v1**: attr FAIL · pres PASS · succ PASS
- **StackCube-v1**: attr FAIL · pres PASS · succ PASS
