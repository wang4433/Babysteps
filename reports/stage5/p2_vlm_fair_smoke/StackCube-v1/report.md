# Stage-5 P2 VLM attribution — StackCube-v1

- Rule-table attribution accuracy (baseline): **0.860**

## C1 — VLM-constrained diagnosis + slot-local revision (ours)

- n_episodes: 50
- attribution_accuracy: **0.860**
- frozen_factor_preservation: **1.000**
- unnecessary_factor_change: **0.000**
- final_success_rate: **0.700**
- parse_failure_rate: 0.000
- preservation (mean over non-implicated factors): **1.000**
- unnecessary_changes_rate (mean): 0.000
- harmful_changes_rate (mean): 0.000

## C2 — VLM free-form replan (baseline)

- n_episodes: 50
- frozen_factor_preservation: **0.020**
- unnecessary_factor_change: **0.980**
- fixed_oracle_factor_rate: 1.000
- final_success_rate: **0.820**
- parse_failure_rate: 0.000
- preservation (mean over non-implicated factors): **0.804**
- unnecessary_changes_rate (mean): 0.196
- harmful_changes_rate (mean): 0.003

## Gates

- **C1 attribution ≥ rule-table**: 0.860 vs 0.860 → PASS
- **C1 preservation ≥ C2 preservation** (Δ = +98.0pp; PASS if Δ ≥ 0)
- **C1 success ≥ C2 success within 5pp** (Δ = -12.0pp; PASS if Δ ≥ -5)
- **C1 selectivity-preservation ≥ C2** (mean preservation Δ = +19.6pp; PASS if Δ ≥ 0)
- **C1 harmful-changes ≤ C2** (mean harmful_changes_rate Δ = -0.3pp; PASS if Δ ≤ 0)

