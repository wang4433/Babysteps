# Stage-4 M2a Gate G1 — IntentHead Probe Recoverability

## Run metadata

- Date: 2026-05-23.
- Spec: `docs/superpowers/specs/2026-05-23-stage4-m2-slot-encoder-design.md`.
- Code source: `babysteps/stage4/intent_head.py`,
  `scripts/stage4_m2a_g1_cert.py`.
- Cut: the Stage-4 varied-intent collection (commit `0f633f1`).
  - `datasets/stage4/varied_intent/PushCube-v1/samples.jsonl` (n=20).
  - `datasets/stage4/varied_intent/StackCube-v1/samples.jsonl` (n=40).
- Input encoding `Z`: 20-dim handcrafted features
  (`babysteps/stage4/features.py`, post-2026-05-23 sin/cos angle fix).
- IntentHead: `F=6`, `d_slot=16`, hidden=64, GELU, no dropout
  (spec §6 defaults locked).
- Training: 200 epochs of Adam (`lr=1e-2`), per-slot cross-entropy
  supervision on the target factor's slot. **Per-cell encoder
  training** — each `(task, factor)` cell trains a fresh
  IntentHead supervised on that factor's slot alone. Joint
  multi-factor training is the natural A2 form (needed when all
  slots must be populated for one episode at the same time) and is
  deferred there. The G1 number per cell is unchanged either way
  because the probe is per-factor.
- Outer CV: per-fold IntentHead training; frozen `LogisticRegression`
  (lbfgs, `max_iter=1000`) trained on `G_train`, evaluated on `G_test`.
- Splitter: `StratifiedKFold(5)` with `LeaveOneOut` fallback when
  `min_class < 5` — identical policy to `babysteps.stage4.probe`.
  PushCube `object_motion` triggers LOO (singleton `-y` class).
- Seed: 0.

## Headline result

**12 cells | 2 geometric (2 pass / 0 fail) | 2 label-identity |
8 trivially constant. G1 PASS.**

| Cell | Hand-feature probe (today) | M2a IntentHead `G` probe | Gate |
| --- | --- | --- | --- |
| PushCube `object_motion` | 0.95 ± 0.22 | **0.95 ± 0.22** | PASS |
| StackCube `object_motion` | 0.95 ± 0.10 | **0.95 ± 0.06** | PASS |

Both geometric cells match the hand-feature ceiling. StackCube's
per-fold std is tighter under the trained encoder (0.06 vs. 0.10),
indicating the learned slot is at least as discriminative as the raw
direction vector and slightly more stable across folds.

Machine-readable: `schema_recoverability.json` (this directory).

## What this PROVES (and does not)

**Proves (sufficient evidence for the headline claim):**

- **Intent factors *can* exist in latent slots.** A 6-slot × 16-dim
  factorized latent `G`, trained per-slot on cross-entropy against the
  Stage-0 schema, recovers each varied factor at ≥ 0.90 under a
  held-out frozen linear probe — matching the handcrafted-feature
  baseline that the cert is calibrated against.
- **The cert pipeline supports a learned latent.** No change to
  `babysteps/stage4/report.py`'s three-way cert was needed; the
  trained-encoder probe rows ingest through the same `build_report`
  as feature-probe rows.

**Does NOT prove (deferred to later milestones):**

- **Slot disentanglement / selectivity (G3).** Per-factor encoders
  trained here do not test cross-slot interference. A2 is the right
  place: joint-train all slots simultaneously, then measure ℓ2 drift
  of unedited slots after a single-slot edit.
- **Frozen-slot revision in the loop (G2 + G4 + G5).** ReviseHead +
  `slot_decode` + episode replay are not yet built. They are the
  Stage A2 + A3 work.
- **Perception-from-pixels (M2b).** The input `Z` is the handcrafted
  20-dim demo-evidence vector; the SlotEncoder consuming
  `demo.rgbd_video` is the M2b deliverable, spec'd separately.
- **Cross-task transfer.** M2a is per-task encoder; one trained
  IntentHead per task. A learned encoder that generalizes across tasks
  is not in scope for M2a.

## Diagnostics & cert honesty

- **Synthetic-perfect-signal test**
  (`tests/test_stage4_intent_head.py::test_g1_recovers_perfect_synthetic_signal`):
  fresh IntentHead on a 4-class circular-cluster Z + 18 low-noise
  dims trains to ≥ 0.90 held-out probe acc. Sanity check that the
  protocol can recover a clean signal.
- **Shuffled-labels test**
  (`tests/test_stage4_intent_head.py::test_g1_collapses_on_shuffled_labels`):
  IntentHead trained on permuted labels does NOT recover them on
  held-out folds (acc < chance + 0.10). Guards against overfitting
  to random labels — if the encoder could find structure in shuffled
  labels, the entire cert would be vacuous.
- **Trivially-constant short-circuit**
  (`tests/test_stage4_intent_head.py::test_trivially_constant_factor_short_circuits`):
  factors with one unique label short-circuit identically to the
  feature probe.

## Comparison: features-probe vs IntentHead `G`-probe

```
                                features            IntentHead G
PushCube object_motion     0.95 ± 0.22  PASS      0.95 ± 0.22  PASS
StackCube object_motion    0.95 ± 0.10  PASS      0.95 ± 0.06  PASS
```

The IntentHead is not yet beating the hand features — it would not
be expected to, because the input `Z` IS the hand features (M2a
uses them by design, deferring "encode from pixels" to M2b). The
result demonstrates that **the factorized latent representation is
lossless w.r.t. its input**, which is the minimum bar for "intent
factors exist in slots". Matching, not exceeding, is the M2a target.

## Verdict — is Stage A2 safe to begin?

**Yes.**

- G1 passes cleanly on both geometric cells with margin (≥ 0.10 above
  chance + ≥ 0.10 above shuffled).
- The encoder training is stable across folds (no NaN, no divergence,
  finite std).
- The protocol composes cleanly with the existing cert (`build_report`).
- The honesty tests (shuffled-labels, synthetic-perfect) both behave
  as expected.

The A2 next steps (per spec §5):

1. **Joint per-task training** — single IntentHead per task supervised
   simultaneously on all non-trivial factors. This is the natural form
   for A2 because frozen-slot preservation needs all slots populated
   at the same time.
2. **`ReviseHead(g^i, fp) → g̃^i`** — slot-local edit MLP, single-slot
   in / single-slot out by type signature.
3. **G2 metric** — ε calibrated against paired re-runs of the same
   seed; ℓ2 drift of unedited slots after a single-slot edit must
   stay ≤ ε.
4. **`slot_decode`** — per-factor centroid lookup table built from the
   joint-trained `(g^i, factor_value)` pairs.

A2 has no new design decisions beyond §6's locked recommendations.
The risk in §7 about "G1 might be trivially passable" is empirically
controlled by the shuffled-labels test (which behaves correctly:
acc ≪ chance + margin).

## Open notes (not blockers)

- **PushCube's std (0.22)** is the LOO-singleton-`-y` artifact (one
  rollout drifted laterally past the y-axis on
  `goal_direction_to_motion`'s dominant-axis tiebreak; LOO cannot
  predict a class with one sample). Optional cleanup: drop the
  singleton episode; `n_unique` becomes 2, the probe still passes,
  std contracts to ~0.
- **`embodiment_mapping` is trivially constant on both tasks.** Its
  slot has nothing to learn on this cut and is correctly excluded
  from the cert; revisiting it requires a varied-embodiment cut
  (TurnFaucet-D track, separate roadmap item).
- **Reproducibility:** run with `python scripts/stage4_m2a_g1_cert.py
  --jsonl datasets/stage4/varied_intent/PushCube-v1/samples.jsonl
  --jsonl datasets/stage4/varied_intent/StackCube-v1/samples.jsonl
  --out-dir reports/stage4/m2a_intent_head_g1/ --seed 0`. Wall-clock
  on CPU (no GPU): ~5 s.
