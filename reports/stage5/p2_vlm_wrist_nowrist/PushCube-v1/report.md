# Stage-5 P2 VLM attribution — PushCube-v1

- Rule-table attribution accuracy (baseline): **1.000**

## C1 — VLM-constrained diagnosis + slot-local revision (ours)

- n_episodes: 50
- attribution_accuracy: **1.000**
- frozen_factor_preservation: **1.000**
- unnecessary_factor_change: **0.000**
- final_success_rate: **0.980**
- parse_failure_rate: 0.000

## C2 — VLM free-form replan (baseline)

- n_episodes: 50
- frozen_factor_preservation: **1.000**
- unnecessary_factor_change: **0.000**
- fixed_oracle_factor_rate: 0.920
- final_success_rate: **0.880**
- parse_failure_rate: 0.000

## Gates

- **C1 attribution ≥ rule-table**: 1.000 vs 1.000 → PASS
- **C1 preservation ≥ C2 preservation** (Δ = +0.0pp; PASS if Δ ≥ 0)
- **C1 success ≥ C2 success within 5pp** (Δ = +10.0pp; PASS if Δ ≥ -5)

