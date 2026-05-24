# Stage-4 M2.5 — Real ManiSkill PushCube No-Regression Check

## Run metadata

- Date: 2026-05-23 23:37 EDT.
- Slurm job: **10786302** (a100-40gb partition, standby QoS,
  **54-second wall-clock**).
- Pack: `models/stage4/m25/packs/PushCube-v1/` (M2a IntentHead +
  ReviseHead + centroids on the 20-episode PushCube varied cut, PLUS
  the joint AttributionHead from `models/stage4/m25/joint/attribution_head.pt`).
- Eval seeds: 100-149 (same 50 held-out as M2a A3 PushCube).
- Env: real `PushCubeEnvRunner` via ManiSkill.

## Headline result — NO REGRESSION

| Policy | Initial → Final | Δpp vs latent |
| --- | --- | --- |
| **latent (M2.5)**       | 0/50 → **49/50 (0.980)** | — |
| oracle_factor_revision | 0/50 → 49/50 (0.980) | 0.0 |
| babysteps_selective     | 0/50 → 49/50 (0.980) | 0.0 |
| same_intent_retry      | 0/50 → 0/50 (0.000) | −98.0 |

- **G4 — Δpp(latent vs same_intent_retry) = +98 pp → PASS** (spec ≥ 10).
- **G5 strict — Δpp(latent vs oracle) = 0 pp → PASS** (spec ±5).
- Identical to the M2a A3 PushCube result on the same 50 seeds.

All three revision policies miss the same single seed: **126**
(an env-side noise floor seed that no attribution strategy can fix).

## Why this matters

PushCube was the no-regression check for the learned attribution
head. The rule already correctly maps `approach_blocked →
approach_direction` 44/44 on the Stage-0 baseline data; the joint
head must match that, not break it.

It does, and end-to-end on the real env it matches `oracle_factor_revision`
exactly, just as M2a did. This rules out the failure mode where the
M2.5 head, trained jointly across PushCube and StackCube, somehow
learns to over-predict `goal_state` on PushCube inputs.

## Mechanism (matches M2.5 training data)

From the M2.5 training notes
(`reports/stage4/m25_attribution/joint/notes.md`):

- 188 PushCube training samples, all `approach_blocked → approach_direction`.
- Joint head predictions on the same 188 samples: 188/188 →
  `approach_direction` (matches rule).
- 208 StackCube training samples (3 predicates, all → `goal_state`).
- Joint head predictions on the StackCube ambiguous slice (11
  samples): 11/11 → `goal_state` (rule was 0/11).

The training-time discrimination is binary
(`approach_direction` vs. `goal_state`), conditioned on
`(failure_predicate, embodiment_mapping)`. PushCube is the
`embodiment_mapping == proxy_contact_to_franka_push` side of that
binary; this real-env eval confirms it stays on its side.
