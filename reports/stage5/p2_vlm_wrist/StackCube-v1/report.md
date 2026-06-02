# Stage-5 P2 VLM attribution — StackCube-v1

- Rule-table attribution accuracy (baseline): **0.860**

## C1 — VLM-constrained diagnosis + slot-local revision (ours)

- n_episodes: 50
- attribution_accuracy: **0.880**
- frozen_factor_preservation: **1.000**
- unnecessary_factor_change: **0.000**
- final_success_rate: **0.720**
- parse_failure_rate: 0.000

## C2 — VLM free-form replan (baseline)

- n_episodes: 50
- frozen_factor_preservation: **0.789**
- unnecessary_factor_change: **0.211**
- fixed_oracle_factor_rate: 1.000
- final_success_rate: **0.340**
- parse_failure_rate: 0.620

## Gates

- **C1 attribution ≥ rule-table**: 0.880 vs 0.860 → PASS
- **C1 preservation ≥ C2 preservation** (Δ = +21.1pp; PASS if Δ ≥ 0)
- **C1 success ≥ C2 success within 5pp** (Δ = +38.0pp; PASS if Δ ≥ -5)

