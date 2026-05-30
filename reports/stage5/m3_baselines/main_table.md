# Stage-5 M3 — Procedural Baselines Main Table

|  policy | PickCube | PushCube | StackCube |
| ---|---|---|---|
| `one_shot` | 0.000 | 0.000 | 0.000 |
| `same_intent_retry` | 0.000 | 0.000 | 0.000 |
| `random_factor_revision` | 0.920 | 0.540 | 0.420 |
| `babysteps_selective` | 0.920 | 0.980 | 0.700 |
| `text_feedback_replan` | 0.920 | 0.000 | 0.700 |
| `full_replan_analogue` | 0.920 | 0.000 | 0.820 |
| `oracle_factor_revision` | 0.920 | 0.980 | 0.820 |

## Delta-pp vs same_intent_retry

### PickCube-v1

| policy | final | Δpp vs retry |
|---|---|---|
| `one_shot` | 0.000 | +0.0 |
| `same_intent_retry` | 0.000 | +0.0 |
| `random_factor_revision` | 0.920 | +92.0 |
| `babysteps_selective` | 0.920 | +92.0 |
| `text_feedback_replan` | 0.920 | +92.0 |
| `full_replan_analogue` | 0.920 | +92.0 |
| `oracle_factor_revision` | 0.920 | +92.0 |

### PushCube-v1

| policy | final | Δpp vs retry |
|---|---|---|
| `one_shot` | 0.000 | +0.0 |
| `same_intent_retry` | 0.000 | +0.0 |
| `random_factor_revision` | 0.540 | +54.0 |
| `babysteps_selective` | 0.980 | +98.0 |
| `text_feedback_replan` | 0.000 | +0.0 |
| `full_replan_analogue` | 0.000 | +0.0 |
| `oracle_factor_revision` | 0.980 | +98.0 |

### StackCube-v1

| policy | final | Δpp vs retry |
|---|---|---|
| `one_shot` | 0.000 | +0.0 |
| `same_intent_retry` | 0.000 | +0.0 |
| `random_factor_revision` | 0.420 | +42.0 |
| `babysteps_selective` | 0.700 | +70.0 |
| `text_feedback_replan` | 0.700 | +70.0 |
| `full_replan_analogue` | 0.820 | +82.0 |
| `oracle_factor_revision` | 0.820 | +82.0 |

