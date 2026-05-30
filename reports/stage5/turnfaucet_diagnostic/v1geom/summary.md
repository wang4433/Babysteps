# TurnFaucet poke_turn diagnostic — failure-mode breakdown

Seeds 100..149 (50 total).
Source: `scripts/stage5_p3_turnfaucet_diagnostic.py`.
Dispatch: probe(+1, 80 steps) → if progress ≥ 0.40 full(+1, 400) → else full(-1, 400) → fallback full(+1, 400).

## Category counts

| category | count | rate |
| --- | --- | --- |
| success | 2 | 4% |
| no_contact | 13 | 26% |
| contact_no_motion | 18 | 36% |
| partial_rotation | 16 | 32% |
| mostly_rotated | 1 | 2% |

`arm_near_limit` flag (final robot qpos within 0.10 rad of any hard limit): 20 / 50 (40%).

## How to read this

- `success` — what we want; trial reached `info['success']=True`.
- `no_contact` — gripper never reached the handle. Fix: improve handle-localization / waypoint geometry (the `compute_geometry` helper in `scripts/_diag_tf_poke5.py` is the empirical reference).
- `contact_no_motion` — reached handle, qpos didn't move > 0.05 rad. Fix: contact force / sign / EE pose during sweep.
- `partial_rotation` — pushed the handle but stalled. Fix: longer sweep distance, multi-step contact, lower friction at TCP.
- `mostly_rotated` / `near_success_no_termination` — close to target; may just need a longer trial budget or a tighter success threshold.
- `arm_near_limit` flag — separate signal; co-occurs with any of the above when the IK has wedged the arm against a joint stop. If this co-occurs with `no_contact` it is the most-likely root cause.

## Decision gate (per redesign_failure_paradigm.md §"Phase 3")

If the top-1 category points to a fix that is plausibly ≤3 days of engineering (waypoint geometry, force/direction, trial budget), proceed with Phase 3. Otherwise drop TurnFaucet to appendix and use the 4-task paper narrative.

Per-seed details: see `per_seed.jsonl`.