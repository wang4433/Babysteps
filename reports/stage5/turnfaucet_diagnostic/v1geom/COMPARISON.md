# TurnFaucet poke_turn — v1-geometry port vs perp-axis baseline

Seeds 100..149 (50). Baseline = `../summary.md` (job 10848995, perp(axis_xy)
+ target_link_pos handle). v1geom = this dir (job 10887469, OBB handle centre
+ true circular tangent cross(joint_axis_3d, radius_3d) threaded through
scene.extra). Code: commit-pending poke-geometry port.

## Category counts

| category | baseline | v1geom | Δ |
| --- | --- | --- | --- |
| **success** | 2 (4%) | **2 (4%)** | **0** |
| no_contact | 14 (28%) | 13 (26%) | −1 |
| contact_no_motion | 21 (42%) | 18 (36%) | −3 |
| partial_rotation | 11 (22%) | 16 (32%) | +5 |
| mostly_rotated | 2 (4%) | 1 (2%) | −1 |
| arm_near_limit (flag) | 18 (36%) | 20 (40%) | +2 |

## What the per-seed data shows

- **Success is the noise floor, not a capability.** Baseline succeeds on seeds
  {121, 134}; v1geom succeeds on a *disjoint* pair {107, 109}. The 2/50
  reshuffles run-to-run — there is no stable set of seeds the skill solves.
- **Mechanical improvement is real but marginal and noisy.** mean |progress|
  0.141 → 0.180 (+0.038); median 0.017 → 0.038. 17 seeds improved >0.02,
  12 *worsened* >0.02, 21 unchanged. Mass shifted out of `contact_no_motion`
  into `partial_rotation` (handles turn more), but did not reach the success
  threshold.
- **Bottom-heavy progress:** 30/50 seeds < 0.10 progress; only 6 > 0.50.
- **`arm_near_limit` ≈ 40–46% across every failure category, 0% in both
  successes.** The arm wedging against a Panda joint limit is the common
  denominator of failure and is independent of sweep geometry — an IK /
  reachability problem, not a tangent-direction problem.
- The closest non-success seeds (114: 0.59, 127: 0.45) require ~1.2–1.6 rad
  of rotation; a single 0.22 m lateral sweep cannot complete that arc even
  with good contact.

## Decision-gate read (redesign_failure_paradigm.md §"Phase 3")

The best-motivated, faithfully-verified fix (the v1 geometry — confirmed
numerically equivalent to the empirical reference) **did not move success off
the 4% noise floor.** Remaining blockers are (a) IK reachability/joint-limit
wedging (~40%) and (b) single-sweep cannot achieve the large rotations many
faucets need — both 3+ day efforts with an uncertain ceiling, matching the
doc's "borderline / likely drop" branch for heavy `arm_near_limit` co-occurrence.

**Recommendation: invoke the kill-switch — do not invest the remaining Phase-3
budget in poke-execution.** TurnFaucet's *paper contribution is attribution*
(P2: VLM 1.000 vs rule-table 0.500, the strongest attribution differentiation
of any task), which holds regardless of execution success. Options for the
write-up are a strategic call (see session notes): keep TurnFaucet in the main
table for the attribution claim with the existing execution caveat, or move it
to the appendix and adopt the pre-drafted 4-task narrative.

Job: 10887469 (normal QoS, a100-40gb, 9m13s, exit 0:0).
