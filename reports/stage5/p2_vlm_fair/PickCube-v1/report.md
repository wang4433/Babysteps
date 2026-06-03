# Stage-5 P2 VLM attribution — PickCube-v1

- Rule-table attribution accuracy (baseline): **1.000**

## C1 — VLM-constrained diagnosis + slot-local revision (ours)

- n_episodes: 50
- attribution_accuracy: **1.000**
- frozen_factor_preservation: **1.000**
- unnecessary_factor_change: **0.000**
- final_success_rate: **0.920**
- parse_failure_rate: 0.000
- preservation (mean over non-implicated factors): **1.000**
- unnecessary_changes_rate (mean): 0.000
- harmful_changes_rate (mean): 0.000

## C2 — VLM free-form replan (baseline)

- n_episodes: 50
- frozen_factor_preservation: **0.000**
- unnecessary_factor_change: **1.000**
- fixed_oracle_factor_rate: 1.000
- final_success_rate: **0.900**
- parse_failure_rate: 0.000
- preservation (mean over non-implicated factors): **0.800**
- unnecessary_changes_rate (mean): 0.200
- harmful_changes_rate (mean): 0.167

## Gates

- **C1 attribution ≥ rule-table**: 1.000 vs 1.000 → PASS
- **C1 preservation ≥ C2 preservation** (Δ = +100.0pp; PASS if Δ ≥ 0)
- **C1 success ≥ C2 success within 5pp** (Δ = +2.0pp; PASS if Δ ≥ -5)
- **C1 selectivity-preservation ≥ C2** (mean preservation Δ = +20.0pp; PASS if Δ ≥ 0)
- **C1 harmful-changes ≤ C2** (mean harmful_changes_rate Δ = -16.7pp; PASS if Δ ≤ 0)

