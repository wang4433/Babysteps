# Stage-4 Milestone 1 — Schema Recoverability Notes

## Run metadata

- Date: 2026-05-22
- Primary input JSONLs:
  - `datasets/stage0_baselines/babysteps_selective/PushCube-v1/samples.jsonl`
  - `datasets/stage0_baselines/babysteps_selective/PickCube-v1/samples.jsonl`
  - `datasets/stage0_baselines/babysteps_selective/StackCube-v1/samples.jsonl`
- Union input JSONLs (variance check): the three primary files **plus** the
  three matching `oracle_factor_revision/{PushCube,PickCube,StackCube}-v1`
  files. Output in `reports/stage4/schema_recoverability_union/`.
- Commit at run time: `bab90f035a0596a9aa1ea6da8288c391d7054d71`
- Probe: scikit-learn 1.7.2 `LogisticRegression` (multinomial loss, `lbfgs`,
  `max_iter=1000`), 5-fold `StratifiedKFold`, **falling back to `LeaveOneOut`
  when the smallest class has < 5 members**; `seed=0`. The single non-trivial
  cell (StackCube/object_motion, min class = 2) used the LOO branch.
- Feature dim: 19 (`babysteps/stage4/features.py`) = 9 trajectory summary
  stats (start xy, end xy, displacement xy, ‖disp‖, principal angle, path
  length) + 6-way one-hot `contact_region_label` + 4-way one-hot `final_state`.
- Per-cell baselines: `majority_class_acc` (chance) and `shuffled_features_acc`
  (row-permuted `X`, same CV protocol).

## Headline result

**0 of 18 cells pass the 90% gate; 1 fails; 17 are trivially constant.** The
gate is **not meaningfully evaluable on the existing `stage0_baselines` data**:
the initial intent is a single canonical value per task for 17 of 18
`(task, factor)` cells, so there is no within-task label variation for a probe
to recover. The lone factor that does vary — `StackCube/object_motion` — is
recovered at 0.75, below the gate, but only because its two minority direction
classes are starved of samples (see below), not because the demo evidence
lacks the signal.

This is the falsification the milestone was designed to surface (plan intro,
and `goal.md` §"Certification Interface": "If probe recoverability fails,
Stage 4 is not yet a faithful refinement of Stage 0"). It materialises the
plan's Risk #3 ("constant labels within a task") far more strongly than the
plan anticipated: not one or two factors, but seventeen of eighteen.

## Per-cell summary (primary)

```
# Stage-4 Schema-Recoverability Probe
Gate: non-trivial cells must reach probe_acc_mean >= 0.90.
Cells: 18 total | 0 pass | 1 fail | 17 trivially constant.

### PickCube-v1
| factor | n_unique | n_episodes | majority | shuffled | probe ± std | gate |
| --- | --- | --- | --- | --- | --- | --- |
| approach_direction | 1 | 24 | 1.00 | 1.00 | 1.00 ± 0.00 | trivial |
| constraint_region  | 1 | 24 | 1.00 | 1.00 | 1.00 ± 0.00 | trivial |
| contact_region     | 1 | 24 | 1.00 | 1.00 | 1.00 ± 0.00 | trivial |
| embodiment_mapping | 1 | 24 | 1.00 | 1.00 | 1.00 ± 0.00 | trivial |
| goal_state         | 1 | 24 | 1.00 | 1.00 | 1.00 ± 0.00 | trivial |
| object_motion      | 1 | 24 | 1.00 | 1.00 | 1.00 ± 0.00 | trivial |

### PushCube-v1
| factor | n_unique | n_episodes | majority | shuffled | probe ± std | gate |
| --- | --- | --- | --- | --- | --- | --- |
| approach_direction | 1 | 24 | 1.00 | 1.00 | 1.00 ± 0.00 | trivial |
| constraint_region  | 1 | 24 | 1.00 | 1.00 | 1.00 ± 0.00 | trivial |
| contact_region     | 1 | 24 | 1.00 | 1.00 | 1.00 ± 0.00 | trivial |
| embodiment_mapping | 1 | 24 | 1.00 | 1.00 | 1.00 ± 0.00 | trivial |
| goal_state         | 1 | 24 | 1.00 | 1.00 | 1.00 ± 0.00 | trivial |
| object_motion      | 1 | 24 | 1.00 | 1.00 | 1.00 ± 0.00 | trivial |

### StackCube-v1
| factor | n_unique | n_episodes | majority | shuffled | probe ± std | gate |
| --- | --- | --- | --- | --- | --- | --- |
| approach_direction | 1 | 24 | 1.00 | 1.00 | 1.00 ± 0.00 | trivial |
| constraint_region  | 1 | 24 | 1.00 | 1.00 | 1.00 ± 0.00 | trivial |
| contact_region     | 1 | 24 | 1.00 | 1.00 | 1.00 ± 0.00 | trivial |
| embodiment_mapping | 1 | 24 | 1.00 | 1.00 | 1.00 ± 0.00 | trivial |
| goal_state         | 1 | 24 | 1.00 | 1.00 | 1.00 ± 0.00 | trivial |
| object_motion      | 4 | 24 | 0.42 | 0.54 | 0.75 ± 0.43 | FAIL |
```

Machine-readable form: `schema_recoverability.json` (this directory).

## Cells that passed the 90% gate

None. No non-trivial cell reached `probe_acc_mean >= 0.90`.

## Cells that did not pass — and why

### `StackCube-v1 / object_motion` — probe 0.75, majority 0.42, shuffled 0.54, n_unique=4, n=24

This is the only non-trivial cell. Its label takes four values across the 24
episodes — `translate_+y`×10, `translate_-y`×8, `translate_+x`×4,
`translate_-x`×2 — because StackCube randomises cubeB's placement, so the
planar direction cubeA must travel to reach cubeB varies per seed. The probe
reaches **0.75** LOO accuracy, which is above chance (`majority` 0.42) and
above the shuffled-feature baseline (0.54), so the demo trajectory genuinely
carries the directional signal. The shortfall to 0.90 is a **small-sample,
class-imbalance artifact, not a representational gap**. The per-class mean
demo displacement separates all four directions cleanly in `(disp_x, disp_y)`
— `+x ≈ (+0.10, 0)`, `−x ≈ (−0.11, 0)`, `+y ≈ (−0.05, +0.16)`,
`−y ≈ (−0.05, −0.16)` — yet the leave-one-out confusion matrix recovers the
two well-populated y-classes perfectly (`+y` 10/10, `−y` 8/8) and misses
**every** minority x-class episode (`+x` 0/4, `−x` 0/2). All six errors are
exactly the six x-direction episodes: a class with two members cannot be
learned when one is held out under LOO, and the four `+x` episodes are
out-voted by the 18 y-episodes. With balanced classes and more episodes the
same 19-dim features should clear 0.90. (A secondary, smaller effect: the raw
`arctan2` angle feature wraps at ±180°, exactly where `−x` lives, so the angle
column alone is discontinuous for `−x`; the `disp_x`/`disp_y` columns
compensate, so this is not the binding constraint.) The 24-episode,
4-way-imbalanced sample is simply under-powered for a 90% certification.

## Trivially constant cells (informational, not gated)

The other **17** cells are trivially constant (`n_unique_labels == 1`,
short-circuited to `probe_acc_mean = 1.0`, `trivially_constant = True`). The
Stage-0 collection emits one canonical initial intent per task:

- **PushCube-v1**: `{goal cube_at_target, motion translate_+x, contact
  minus_x_face, approach from_minus_x, constraint none, embodiment push}` — all 6 constant.
- **PickCube-v1**: `{cube_lifted_at_target, lift_up, minus_x_face, from_above,
  none, grasp}` — all 6 constant.
- **StackCube-v1**: `{cube_at_target, <object_motion varies>, minus_x_face,
  from_above, none, pick_and_place}` — 5 of 6 constant.

In these datasets the episode-to-episode variation lives in the scene and the
resulting failure/revision, **not** in the initial intent. A constant factor
is "recoverable" only in the degenerate sense that a constant predictor scores
1.0 — which is precisely the certification loophole that makes a 90% gate
non-informative for these cells. (Confirmed identical across the
`babysteps_selective` and `oracle_factor_revision` baselines, since the
initial intent and demo evidence are produced *before* the revision step and
do not depend on the revision policy.)

## Union variance check (primary vs. primary + oracle)

Only the one non-trivial cell can move. `StackCube/object_motion` goes
**0.75 → 0.79** (+4 pp) from n=24 to n=48 — **within the ±5 pp tolerance**, so
the demo-evidence distribution is **not baseline-coupled**. Caveat: because the
initial intent and demo evidence are revision-policy-independent, the
`selective` and `oracle` files largely repeat the same per-seed rows, so the
union is closer to row duplication than to an independent doubling of evidence;
the small uptick (and the shuffled baseline settling from 0.54 to 0.42, i.e.
toward the 0.42 majority rate as the permutation estimate stabilises) should
not be read as new signal. All 17 trivial cells stay trivial.

## Implication for Stage-4 Milestone 2

**Pick (b): revise the data/labelling regime before encoder work — concretely,
fix the data *sampling*, and close the certification loophole; the schema and
the `object_motion` labelling are sound.**

We do **not** proceed to the learned slot encoder (option a). `goal.md`
§"Certification Interface" #1 requires a frozen linear probe to recover each
discrete factor from the learned latent `G_t` at ≥ 90% held-out accuracy. On
the existing `stage0_baselines` data that certification is **non-evaluable**:
17 of 18 `(task, factor)` cells carry a single label, so a constant predictor
satisfies the gate without the latent encoding anything, and the one cell with
genuine variation fails at 0.75 purely for lack of per-class samples (all six
errors are the n=2 and n=4 minority direction classes; the features separate
the directions linearly). Training a SlotEncoder/IntentHead now would let it
pass cert #1 by memorising the task → intent lookup table — not evidence of a
faithful latent intent representation, which is the whole point of Stage 4.

Two cheap fixes are required before M2, both of which sharpen this same probe
into a real test:

1. **Vary the intent factors per episode, with balanced, adequately-sized
   classes.** Extend Stage-0 collection so the *initial* intent actually
   differs across seeds — e.g., randomise the contacted face / push direction
   for PushCube, the grasp face for PickCube, the specified goal for StackCube
   — targeting ≥ ~10 episodes per class so 5-fold CV (not LOO) applies. The
   `StackCube/object_motion` result is the proof of concept that the probe
   *works* once a factor varies; it just needs support.
2. **Tighten the certification so trivially-constant factors do not count as
   "recovered."** Report recoverability only on factors with ≥ 2
   well-populated classes, always against the `majority_class_acc` and
   `shuffled_features_acc` baselines already emitted here (a cell is only
   "recovered" if it clears both chance and the shuffled baseline by a margin,
   not merely the 0.90 absolute).

Option (c) — probe `G_t` conditioned on language rather than `G_t` alone, or
relax the per-factor threshold for genuinely hard factors — is a reasonable
fallback for factors that remain sub-90% *after* richer collection (e.g., if a
factor turns out to depend on a privileged scene field like `blocked_sides`
that is absent from DemoEvidence). It is premature now: until the data
exercises the factors, there is nothing to reframe.
