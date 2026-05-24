# Stage-4 M2a A3 — StackCube sim-free smoke (counterfactual-synthesis unblock)

## Run metadata

- Date: 2026-05-23.
- Spec: `docs/superpowers/specs/2026-05-23-stage4-m2-slot-encoder-design.md`.
- Code: `babysteps/stage4/counterfactual.py` +
  `scripts/stage4_m2a_train_pack.py --counterfactual-synthesis`
  (default-on).
- Pack: `models/stage4/m2a/StackCube-v1/` (trained with
  counterfactual synthesis on the 40-episode StackCube varied cut).
- Eval seeds: 200-219 (20 seeds, disjoint from training).
- Env: `FakeStackCubeEnvRunner` (`tests/conftest.py`).

## Headline result

| Policy | Initial success | Final success |
| --- | --- | --- |
| **latent_revision** | 0/20 | **20/20 (1.000)** |
| babysteps_selective | 0/20 | 20/20 (1.000) |
| oracle_factor_revision | 0/20 | 20/20 (1.000) |
| same_intent_retry | 0/20 | 0/20 (0.000) |

- **G4 — Δpp(latent vs same_intent_retry) = +100 pp → PASS** (spec ≥ 10).
- **G5 — Δpp(latent vs oracle) = 0 pp → PASS** (spec ≤ 5 deficit).

## What changed since `reports/stage4/m2a_a2/notes.md`

`m2a_a2/notes.md` §"Data limitation" reported StackCube as **0/40
certable** because the varied cut's `goal_state` is always
`cube_at_target` in initial intents but every revision targets
`cubeA_on_cubeB`, so the encoder had no centroid for the revised
class.

The fix: **counterfactual synthesis for label-identity factors**
(`babysteps/stage4/counterfactual.py`,
`substitute_label_identity_feature(Z, factor, new_value)`). For each
revision whose `new_value` isn't in any initial intent and whose
factor is supported (currently `goal_state`, `contact_region`), the
trainer synthesizes a `Z' = substitute(Z, factor, new_value)` sample
with the corresponding one-hot flipped and labels the synthetic
sample with the revised class. The encoder then has a region for the
previously-unseen class without any new GPU rollouts.

For StackCube this added **34 synthetic samples** (all
`goal_state: cube_at_target → cubeA_on_cubeB`), bringing the
certable revision count from 0/40 to 34/40 (only the 6
`approach_direction` revisions remain uncertable — that factor is
not in any Z one-hot and so cannot be synthesized this way).

## Reproducibility

```bash
# 1. Retrain StackCube pack WITH counterfactual synthesis (default)
python scripts/stage4_m2a_train_pack.py \
    --jsonl datasets/stage4/varied_intent/StackCube-v1/samples.jsonl \
    --out-dir models/stage4/m2a/StackCube-v1/
# Expected: "counterfactual synthesis added 34 sample(s) ({'goal_state': 34})"
# Expected: "trained ReviseHead on 34 certable revision pairs"

# 2. Sim-free smoke (this report)
python scripts/stage4_m2a_run_eval.py \
    --task StackCube-v1 \
    --pack-dir models/stage4/m2a/StackCube-v1 \
    --out-dir reports/stage4/m2a_a3_smoke_fake_stackcube \
    --eval-seeds 200-219 --fake

# 3. Real ManiSkill eval (Slurm, ~30-90 min on a30)
sbatch slurm/stage4_m2a_a3_stackcube.sbatch
```

## Still uncertable (not addressed by counterfactual synthesis)

- **StackCube approach_direction revisions (6/40):** the
  `approach_direction` factor is not in any Z one-hot (it is
  derived from `contact_region` only in PushCube; in StackCube
  there is no one-hot to flip). Path to unblock: extend the
  StackCube varied cut so some initial intents have
  `from_minus_x` instead of always `from_above` (a small GPU
  rerun). Deferred — out of M2a A2/A3 scope.
