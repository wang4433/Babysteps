# Stage-4 M2a A3 — Sim-free smoke (FakeEnvRunner)

## Run metadata

- Date: 2026-05-23.
- Spec: `docs/superpowers/specs/2026-05-23-stage4-m2-slot-encoder-design.md`
  §4 "G4 / G5".
- Code: `babysteps/stage4/latent_policy.py`,
  `scripts/stage4_m2a_train_pack.py`, `scripts/stage4_m2a_run_eval.py`.
- Pack: trained on PushCube varied cut, saved to
  `models/stage4/m2a/PushCube-v1/` (gitignored; regenerate with
  `scripts/stage4_m2a_train_pack.py`).
- Eval seeds: 100–119 (20 seeds, disjoint from training cut 0–19).
- Env: **`FakeEnvRunner`** (`tests/conftest.py`) — deterministic,
  sim-free, no Vulkan. Substitutes for ManiSkill PushCube. Success
  rule: `contact_region == direction_to_face(cube → goal)`.

## Headline result (FakeEnv only — not the production G4/G5)

| Policy | Initial success | Final success | Δpp vs latent |
| --- | --- | --- | --- |
| **latent_revision** | 0/20 | **20/20 (1.000)** | — |
| babysteps_selective | 0/20 | 20/20 (1.000) | 0.0 |
| oracle_factor_revision | 0/20 | 20/20 (1.000) | 0.0 |
| same_intent_retry | 0/20 | 0/20 (0.000) | −100.0 |

**G4 (Δpp vs same_intent_retry):** +100.0 pp → PASS (spec ≥ 10).
**G5 (Δpp vs oracle_factor_revision):**   0.0 pp → PASS (spec ≤ 5 deficit).

## What this PROVES

- **End-to-end wire-up works.** `episode.run_episode(adapter=..,
  policy=latent_revision_factory(pack))` runs through the full Stage-0
  loop (demo → intent → execute → fail → latent revise → retry)
  without exceptions or schema violations.
- **Single-factor invariant preserved at the schema level.** The
  emitted `Revision` has `operator='latent_revision'`,
  `factor='approach_direction'`, `frozen_factors=(5 other factors)` —
  exactly one factor changes (asserted in
  `tests/test_stage4_latent_policy.py::test_latent_policy_changes_exactly_one_factor`).
- **The trained ReviseHead + centroid decoder land on the right token.**
  PushCube has 2 `approach_direction` classes; in every blocked-
  approach episode the latent path produces the OPPOSITE class from
  the initial intent (which is what FakeEnv needs to succeed). 20/20
  retry success matches the rule-based `babysteps_selective` baseline
  exactly.
- **Selectivity sanity at the loop level.** `same_intent_retry`
  (deterministic same intent) gets 0/20 → the FakeEnv really
  requires a revision, not a retry. The latent revision IS what
  unlocks the success.

## What this does NOT prove (yet)

- **Real-env G4/G5.** FakeEnv is a stylized push physics; ManiSkill
  is the actual env in the cert spec. The Slurm sbatch
  `slurm/stage4_m2a_a3_pushcube.sbatch` runs the same eval against
  the real env on seeds 100–149.
- **StackCube G4/G5.** StackCube's varied cut has 0 certable revisions
  (see `reports/stage4/m2a_a2/notes.md` §"Data limitation"), so the
  ReviseHead in `models/stage4/m2a/StackCube-v1/` is at random init
  and would output garbage. Deferred until either (a) the StackCube
  varied cut is extended to balance the revisable factors in initial
  intents, or (b) the counterfactual-synthesis trick is implemented
  for label-identity factors.
- **G3 selectivity (paired counterfactual rollouts).** The latent
  loop's claim about NOT disturbing other factors at the
  *predicted-future-state* level needs paired sim rollouts (same
  seed, with/without the edit) and a paired t-test against natural
  noise. Future A3 work.

## Reproducibility

```bash
# 1. Train the LatentPack (sim-free, ~15s)
python scripts/stage4_m2a_train_pack.py \
    --jsonl datasets/stage4/varied_intent/PushCube-v1/samples.jsonl \
    --out-dir models/stage4/m2a/PushCube-v1/

# 2. Sim-free smoke (this report)
python scripts/stage4_m2a_run_eval.py \
    --task PushCube-v1 \
    --pack-dir models/stage4/m2a/PushCube-v1 \
    --out-dir reports/stage4/m2a_a3_smoke_fake \
    --eval-seeds 100-119 --fake

# 3. Real ManiSkill eval (Slurm, GPU)
sbatch slurm/stage4_m2a_a3_pushcube.sbatch
```

## What's next

- **Submit Slurm `stage4_m2a_a3_pushcube.sbatch`** for the
  ManiSkill G4/G5 cert on real env. Same script, same pack, same
  seeds, just dispatched to the real `PushCubeEnvRunner`. Expected
  wall-clock: 30–60 min (4 policies × 50 seeds × ~10s per episode).
  Output → `reports/stage4/m2a_a3_pushcube/a3_results.json`.
- **StackCube unblock** (deferred): pick the data-extension or
  counterfactual-synthesis path documented in
  `reports/stage4/m2a_a2/notes.md` §"Data limitation".
- **G3 selectivity** (deferred): build the paired-rollout fixture
  + cross-slot semantic-drift metric. Requires either richer
  predicted-future-state representations or the action-decoder side
  of `goal.md` §"Architecture" (M2a doesn't have a learned action
  decoder; revisions go through the existing skill compilers).
