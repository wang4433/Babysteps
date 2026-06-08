# PokeCube LOTO — clustered-CI + selectivity report

## results_pooled_gi_none.json

- scorer: `models/stage5/shared_policy/pooled_gi_none.pt`
- episodes: 120  | clusters (seeds): 20  | candidates: ['minus_x_face', 'minus_y_face', 'plus_y_face']

### Table 5 — recovery, 95% clustered-bootstrap CI (resampled by scene seed)

| condition | recovery  [95% CI] |
|---|---|
| same_intent / open_loop | 0.000  [0.000, 0.000] |
| random_face | 0.333  [0.250, 0.417] |
| shared_scorer (frozen) | 0.967  [0.900, 1.000] |
| oracle (ceiling) | 0.967  [0.900, 1.000] |

shared-scorer face-pick accuracy: **1.000  [1.000, 1.000]**

### Paired difference CIs (same cluster resample → paired)

- scorer_minus_oracle: **+0.000**  [+0.000, +0.000]  (INCLUDES 0)
- scorer_minus_random: **+0.633**  [+0.550, +0.717]  (excludes 0)
- scorer_minus_open_loop: **+0.967**  [+0.900, +1.000]  (excludes 0)

### Failure attribution

- seed 124: 4 fail, oracle-coincident=True (4/4 also fail oracle)

### Selectivity disclosure (honest)

- direction→face deterministic: **True** (degenerate 1:1 residual-sign→face rule)
- wrong_face changes the choice: **False**
- map: {'+x': ['minus_x_face'], '+y': ['minus_y_face'], '-y': ['plus_y_face']}


