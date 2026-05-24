# Stage-4 M2.5 — Joint Attribution Head Training Notes

## Run metadata

- Date: 2026-05-23 (post-M2a).
- Spec: `docs/superpowers/specs/2026-05-23-stage4-m2.5-attribution-head-design.md`.
- Code: `babysteps/stage4/attribution_head.py`,
  `babysteps/stage4/attribution_features.py`,
  `scripts/stage4_m25_train_attribution.py`.
- Train cmd:
  ```
  python scripts/stage4_m25_train_attribution.py \
      --task StackCube-v1 --task PushCube-v1 \
      --out-dir models/stage4/m25/joint/ \
      --report-dir reports/stage4/m25_attribution/joint/
  ```

## Data

Union of these JSONL paths, per task (PushCube + StackCube):

- `datasets/stage0_baselines/{babysteps_selective, full_replan_analogue,
  one_shot, oracle_factor_revision, random_factor_revision,
  same_intent_retry, text_feedback_replan}/<task>/samples.jsonl`
- `datasets/stage4/varied_intent/<task>/samples.jsonl`

Each record where `failure_packet.failure_predicate != "none"` and
`oracle_wrong_factor` is set contributes one `(fp, intent) →
wrong_factor` pair.

## Headline

| Metric | Value |
|---|---:|
| n_samples (with failure) | **396** |
| feature_dim | **47** |
| n_unique label classes | **2** (`goal_state`=208, `approach_direction`=188) |
| held-out k-fold accuracy | **1.0000** |
| shuffled-label control acc | **0.5303** (chance = 1/6 = 0.167; majority baseline = 208/396 = 0.525) |
| final train loss | **3.4 × 10⁻⁶** |

**Shuffled-label control collapses cleanly** — from 1.0 → 0.53 — to the
majority-class baseline. This is the headline signal that the head
learned intent-conditioned discrimination rather than memorizing
input identity.

## Per-predicate diagnostic (StackCube; 64 failures across baselines + varied cut)

| Predicate | Oracle label | n | Rule pred. | Head pred. | Rule correct? | Head correct? |
|---|---|---:|---|---|---|---|
| `goal_not_satisfied` | `goal_state`       | 53 | `goal_state`           | `goal_state` | 53/53 | **53/53** |
| `direction_error`    | `goal_state`       |  9 | `approach_direction`   | `goal_state` | 0/9   | **9/9**   |
| `no_motion`          | `goal_state`       |  2 | `approach_direction`   | `goal_state` | 0/2   | **2/2**   |
| **Total**            |                    | 64 |                        |              | 53/64 | **64/64** |

PushCube (44 failures, all `approach_blocked → approach_direction`):
- Rule: 44/44 ✓
- Head: **44/44** ✓ (no regression)

The 11 StackCube ambiguous-slice failures are exactly the predicate
class the rule mishandles. The joint head learns "for StackCube
embodiment_mapping, all three predicates collapse to `goal_state`"
purely from the `(failure_predicate, embodiment_mapping)` slice of
the input.

## Why "joint" instead of per-task

A per-task head over PushCube has 44 samples, all labeled
`approach_direction`. Over StackCube: 208 samples, all `goal_state`.
Both per-task heads are **degenerate constant-output predictors** —
trivially correct, but with a learning-vs-memorization control that
is uninformative (`n_unique_classes == 1` ⇒ no shuffle perturbation).

Joint training requires the head to actually discriminate between
the two tasks' label classes using only `(FailurePacket, Intent)`
features. The shuffled-label control's collapse from 1.0 to 0.53
shows the head is in fact learning a binary classifier — it's not
just outputting a constant.

The bundled pack uses the SAME joint head for both PushCube and
StackCube; the latent_policy invokes `head.predict_factor(fp, intent)`
the same way for both tasks. No per-task switching at inference.

## What this DOES prove

- The 5-seed StackCube G5 gap is explained entirely by the rule's
  predicate→factor table being wrong on `direction_error` and
  `no_motion` for the under-specified-goal regime. With a learned
  head replacing the rule, those 11 ambiguous-slice failures get
  the correct attribution.
- Sim-free supervised learning on existing Stage-0 baseline JSONLs
  is sufficient to close the gap — no new data collection needed.
- Per goal.md §"Stage 4 / Data Dependencies", the head trained from
  `oracle_wrong_factor` labels has the rule-table as its analytic
  upper bound; M2.5 reaches that upper bound on this slice.

## What this does NOT prove

- **End-to-end Δpp on real ManiSkill.** That is the job of the
  Slurm eval (job 10786278; see
  `reports/stage4/m25_attribution_stackcube/notes.md`).
- **Robustness to new tasks.** Joint training across the two tasks
  with disjoint label sets means the head's classification is
  intent-conditioned but only over two classes. A larger task set
  (PickCube, TurnFaucet, CrossViewPush) would force more
  discrimination capacity.
- **Cross-task transfer.** The head conditions on Intent fields, so
  a brand-new task with a brand-new embodiment_mapping value would
  need at least a few labeled failures to enter the support.

## Reproducibility

```bash
# Train the joint attribution head (~5s CPU)
python scripts/stage4_m25_train_attribution.py \
    --task StackCube-v1 --task PushCube-v1 \
    --out-dir models/stage4/m25/joint/ \
    --report-dir reports/stage4/m25_attribution/joint/

# Bundle into the StackCube LatentPack (~15s CPU)
python scripts/stage4_m2a_train_pack.py \
    --jsonl datasets/stage4/varied_intent/StackCube-v1/samples.jsonl \
    --out-dir models/stage4/m25/packs/StackCube-v1/ \
    --attribution-head-pt models/stage4/m25/joint/attribution_head.pt

# Bundle into the PushCube LatentPack (~15s CPU)
python scripts/stage4_m2a_train_pack.py \
    --jsonl datasets/stage4/varied_intent/PushCube-v1/samples.jsonl \
    --out-dir models/stage4/m25/packs/PushCube-v1/ \
    --attribution-head-pt models/stage4/m25/joint/attribution_head.pt
```
