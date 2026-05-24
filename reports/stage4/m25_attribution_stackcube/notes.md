# Stage-4 M2.5 — Real ManiSkill StackCube G5 Strict CLOSED

## Run metadata

- Date: 2026-05-23 23:37 EDT.
- Slurm job: **10786278** (a100-40gb partition, standby QoS,
  **158-second wall-clock**). Submitted via
  `sbatch --partition=a100-40gb --qos=standby slurm/stage4_m25_attribution_stackcube.sbatch`
  after rpaleja a30 queue blocked on `AssocGrpGRES`.
- Pack: `models/stage4/m25/packs/StackCube-v1/` (M2a IntentHead +
  ReviseHead + centroids trained with counterfactual synthesis on the
  40-episode StackCube varied cut, PLUS the joint AttributionHead from
  `models/stage4/m25/joint/attribution_head.pt` trained on
  PushCube + StackCube Stage-0 baseline failures).
- Eval seeds: 200–249 (50 held-out, disjoint from the M2.5 training
  cut which uses seeds 0–84).
- Env: real `StackCubeEnvRunner` via ManiSkill.
- Policies: `latent` (M2.5 = M2a + learned attribution), `babysteps_selective`
  (rule attribution + rule revision = Stage-0 baseline),
  `same_intent_retry`, `oracle_factor_revision`.

## Headline result

| Policy | Initial → Final | Δpp vs latent |
| --- | --- | --- |
| **latent (M2.5)**       | 0/50 → **46/50 (0.920)** | — |
| oracle_factor_revision | 0/50 → 46/50 (0.920) | **0.0** |
| babysteps_selective     | 0/50 → 41/50 (0.820) | −10.0 |
| same_intent_retry      | 0/50 → 0/50 (0.000) | −92.0 |

- **G4 — Δpp(latent vs same_intent_retry) = +92.0 pp → PASS** (spec ≥ 10).
- **G5 strict — Δpp(latent vs oracle_factor_revision) = 0.0 pp → PASS**
  (spec ±5; was −10.0 with M2a's rule-based attribution).
- Δpp(latent vs babysteps_selective) = **+10.0 pp** — the M2.5
  attribution head measurably improves the rule-based Stage-0 baseline
  on this seed set.

## Per-seed diagnosis — the gap closed exactly as predicted

| Set | Seeds | Count |
| --- | --- | --- |
| Latent misses | 202, 212, 235, 243 | 4 |
| Oracle misses | 202, 212, 235, 243 | 4 |
| `babysteps_selective` misses | 201, 202, 209, 212, 220, 225, 233, 235, 243 | 9 |
| **Recovered by M2.5** = baseline_miss − latent_miss | **201, 209, 220, 225, 233** | **5** |
| Latent extras vs oracle | (empty) | 0 |
| Oracle extras vs latent | (empty) | 0 |

**Latent ≡ oracle on a per-seed basis.** The 5 seeds the M2a
rule-based attribution mis-classified (M2a notes
`reports/stage4/m2a_a3_stackcube/notes.md`) — `{201, 209, 220, 225,
233}` — are exactly the 5 seeds the learned attribution head recovers.
The 4 remaining failures `{202, 212, 235, 243}` are env-side genuine
failures even the oracle cannot solve (StackCube skill compiler
ceiling, not an attribution defect).

## What the head does on these 5 seeds (predicate-level mechanism)

From the M2.5 training notes
(`reports/stage4/m25_attribution/joint/notes.md`):

On StackCube, the rule's `direction_error → approach_direction` and
`no_motion → approach_direction` mappings are wrong: with
`chosen_intent.goal_state == "cube_at_target"` (the only initial-intent
goal), the under-specification of `cube_at_target` for `cubeA_on_cubeB`
is the actual implicated factor, regardless of which downstream
predicate fires.

The joint AttributionHead learns this from the 11 ambiguous
training samples (9 `direction_error` + 2 `no_motion`, all labeled
`goal_state`) plus the 53 unambiguous `goal_not_satisfied → goal_state`
samples. The shuffled-label control (kfold acc collapses 1.0 → 0.53)
confirms it learns the discrimination rather than the input identity.

## Constraint check

- **Single-factor revision invariant intact:** the slot-local
  ReviseHead interface still accepts exactly one `(g_slot, fp_vec)`
  pair and emits exactly one revised slot. Every emitted `Revision`
  on the 50 seeds carries `frozen_factors` of length 5 (matching M2a;
  see `tests/test_stage4_latent_policy.py::test_latent_policy_changes_exactly_one_factor`).
- **No Stage-0 baseline semantic changes:** `babysteps.failure.attribute_failure`
  and the M3 procedural baselines are bit-identical to pre-M2.5.
  The M2.5 head is loaded only when explicitly bundled in a
  `LatentPack` and only consulted by the `latent_revision` policy.
- **No render/goal.md/unrelated-WIP touched:** confirmed via
  `git status` before commit.
- **Sim-free local + Slurm long:** training + eval scripting + tests
  all run on the login node; only the real ManiSkill eval runs on GPU.

## What this PROVES

- **End-to-end Stage-4 latent loop now matches oracle on real
  ManiSkill on both tasks.** Demo (rgbd_video + trajectory) →
  IntentHead → G → AttributionHead → ReviseHead → centroid `slot_decode`
  → revised Intent → real Franka pick-and-place rollout → success.
  46/50 held-out seeds match oracle exactly.
- **The 10 pp StackCube G5-strict gap was indeed in the rule-based
  attribution, not the latent revision step.** Replacing the rule
  with a sim-free learned head closes the gap exactly, with no
  changes to ReviseHead, slot_decode, IntentHead, or the slot-local
  interface.
- **goal.md §"Stage 4 / Data Dependencies" is satisfied for the
  attribution component:** the hand-coded `failure.py` rules serve
  as the analytic upper bound (oracle_wrong_factor labels); the
  learned attribution head reaches it on this slice.

## What this does NOT prove

- **Strict G3 selectivity.** Still requires paired counterfactual
  rollouts; out of M2a/M2.5 scope. Deferred.
- **Cross-task transfer of the AttributionHead.** Joint training
  here is over 2 tasks with disjoint label sets; extending to
  PickCube + TurnFaucet + CrossViewPush remains future work.
- **Pixel-encoder substitute for the 20-dim handcrafted Z** (M2b).
  Separate spec; ortho­gonal to attribution.

## Reproducibility

```bash
# 1. Train joint AttributionHead (~5s CPU)
python scripts/stage4_m25_train_attribution.py \
    --task StackCube-v1 --task PushCube-v1 \
    --out-dir models/stage4/m25/joint/ \
    --report-dir reports/stage4/m25_attribution/joint/

# 2. Bundle into StackCube LatentPack (~15s CPU)
python scripts/stage4_m2a_train_pack.py \
    --jsonl datasets/stage4/varied_intent/StackCube-v1/samples.jsonl \
    --out-dir models/stage4/m25/packs/StackCube-v1/ \
    --attribution-head-pt models/stage4/m25/joint/attribution_head.pt

# 3. Real ManiSkill eval (Slurm; a30 rpaleja preferred, fall back to
#    a100-40gb standby when rpaleja queue is full)
sbatch slurm/stage4_m25_attribution_stackcube.sbatch
# Or with QoS override:
sbatch --partition=a100-40gb --qos=standby \
    slurm/stage4_m25_attribution_stackcube.sbatch
```

`a3_results.json` (this directory) holds the full per-seed roll-out,
machine-readable.

## Compared to M2a (2026-05-23 same day, earlier)

| Quantity | M2a (rule attribution) | M2.5 (learned attribution) | Δ |
| --- | --- | --- | --- |
| StackCube latent success | 41/50 (0.820) | **46/50 (0.920)** | **+5 seeds (+10 pp)** |
| StackCube G4 vs SIR | +82 pp | **+92 pp** | +10 |
| StackCube G5 vs oracle | −10 pp | **0 pp** | +10 |
| Seeds correctly attributed | 45/50 | **50/50** | +5 |
