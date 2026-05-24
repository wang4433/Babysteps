# Stage-4 M2a Stage A2 — Notes

## Run metadata

- Date: 2026-05-23.
- Spec: `docs/superpowers/specs/2026-05-23-stage4-m2-slot-encoder-design.md` §5.
- Code source: `babysteps/stage4/{intent_head,slot_decode,revise_head}.py`,
  `scripts/stage4_m2a_a2_eval.py`.
- Cut: the Stage-4 varied-intent collection (commit `0f633f1`).
  - PushCube-v1: 20 episodes, all 20 with `approach_substitution` revisions
    (10 `from_plus_x → from_minus_x`, 10 reverse).
  - StackCube-v1: 40 episodes; 34 `goal_refinement` revisions
    (`cube_at_target → cubeA_on_cubeB`); 6 `approach_substitution`
    (`from_above → from_minus_x`).

## Headline result

| Cell | G2 (frozen-slot preservation) | Revised-slot decode acc |
| --- | --- | --- |
| PushCube-v1 | **PASS** (max drift = 0.00e+00) | **1.000** on 20/20 certable |
| StackCube-v1 | **PASS** (max drift = 0.00e+00) | n/a — 0/40 certable (see §"Data limitation" below) |

**G2 is the spec-mandated A2 gate; both tasks PASS.** The G2 drift is
0 by construction: `apply_revision(G, factor_idx, fp, head)` writes
only the implicated column of G and copies the rest — there is no
path through the slot-local `ReviseHead` interface to mutate any
other slot. The deterministic encoder ensures the floating-point
equality is bit-exact (test
`tests/test_stage4_revise_head.py::test_apply_revision_only_changes_implicated_slot`
asserts this with `atol=0, rtol=0`).

The **revised-slot decode accuracy** is a bonus diagnostic beyond
the spec's A2 gate. It asks: does ReviseHead actually move the slot
to the right centroid, or does the slot-local interface only work
mechanically? On PushCube the answer is clean — 20/20 held-out
revisions decode correctly.

## What this PROVES

- **The slot-local revision interface is correct** (G2 = 0 by construction
  on both tasks).
- **The whole M2a pipeline runs end-to-end on real data**: joint encoder
  training → centroid lookup → ReviseHead training → applied revision
  → discrete decode round-trip. Sim-free, CPU-only, ~30s wall clock.
- **On PushCube, the revision is learnable from 16 train episodes per
  fold**: the encoder + ReviseHead correctly flips the
  `approach_direction` slot for all 20 held-out test revisions.

## What this does NOT prove

- **End-to-end Δpp (G4 / G5)**. These need sim rollouts of
  `latent_revision=True` against the existing skill compilers. That
  is A3 work, requires GPU (real ManiSkill execution).
- **Cross-slot selectivity (G3)**. A truly disentangled latent must show
  no semantic drift on other slots after an edit, even when the slots
  are co-trained. M2a's G2 is mechanical (other slot vectors are bit-
  identical); G3 is the deeper sim-side selectivity test in goal.md.
- **StackCube decode quality** (see §"Data limitation").

## Data limitation — why StackCube is 0/40 certable

StackCube's varied cut has its `object_motion` initial intents balanced
4-way (the M1.5 cert target — PASSES at 0.95 in M2a A1), but its
**revisable factors are constant in initial intents**:

| Factor | Initial-intent values present | Revision targets |
| --- | --- | --- |
| `goal_state` | only `cube_at_target` (40/40) | `cubeA_on_cubeB` (34) |
| `approach_direction` | only `from_above` (40/40) | `from_minus_x` (6) |

The IntentHead is trained per factor against the initial-intent
distribution. With one class per factor in train, the centroid bank
for that factor has one entry; revisions targeting a class not seen
in initial intents have no centroid to land on, and the revised-slot
decode is **uncertable**. This is correctly accounted for in the
report: 40/40 revisions are uncertable, decode acc reported as n/a.

This is a **dataset property**, not an M2a defect. PushCube does not
have this issue because its varied cut intentionally toggles
`contact_region` (and hence `approach_direction = face_to_approach(contact)`)
across episodes — so both `from_plus_x` and `from_minus_x` are well
represented in initial intents.

### Two paths to unblock StackCube decode acc

1. **Data path (clean, GPU rerun):** extend the StackCube varied cut
   to also vary initial `goal_state` (e.g. ~half the episodes started
   with the underspecified `cube_at_target`, ~half with the
   fully-specified `cubeA_on_cubeB`) and `approach_direction` (~half
   `from_above`, ~half `from_minus_x`). This makes the centroid bank
   complete and the cert numbers comparable across factors. ~40
   extra StackCube episodes, ~1 hour GPU on an a30.
2. **Counterfactual-synthesis path (M2a-internal trick, no GPU):**
   for **label-identity** factors whose value is read from a Z one-hot
   (`goal_state` ← `final_state` one-hot; `contact_region` ←
   `contact_region_label` one-hot), synthesize an additional training
   point `Z' = Z.copy()` with the one-hot flipped to the revised
   class and label set to the revised class. The encoder learns a
   centroid for that class; the centroid bank becomes complete on
   the existing data; the cert covers ~34/40 (the `goal_state`
   revisions). StackCube `approach_direction` (not in any Z one-hot)
   stays uncertable without path 1. Sim-free, ~30 LOC, can be added
   without changing any committed Stage-0 surface.

**Recommendation:** Path 1 is the clean solution and aligns with the
"build the data right" reflex from M1.5. Path 2 is a quick unblock
worth ~30 min if a Stage A3 result depends on a non-trivial
StackCube decode acc. Both are deferred from M2a A2 because the
spec-mandated G2 gate already passes; surfacing the limitation is
the honest result.

## Per-fold detail

PushCube: 5-fold KFold (n_episodes=20, n_train=16 per fold).
StackCube: 5-fold KFold (n_episodes=40, n_train=32 per fold).
Per-fold G2 max drift is 0.00e+00 everywhere; per-fold decode acc on
PushCube is 1.0 in all 5 folds (4 of 4 held-out revisions correct
each fold).

Machine-readable: `a2_results.json` (this directory).

## Verdict — Is A3 safe to begin?

**Yes for the sim-free scaffolding; partial for the gate cert.**

- **A3 sim-free scaffold** (`episode.py` with `latent_revision=True`
  path that calls `ReviseHead + slot_decode` instead of
  `revision.py`'s discrete operator) is safe to build now. It can be
  smoke-tested on the FakeEnv to verify the path runs and the type
  signatures wire up. No GPU needed.
- **A3 gate cert (G3 + G4 + G5)** needs real GPU rollouts. The G2-mechanical
  pass + PushCube decode acc 1.0 means the revision pipeline is at
  least non-broken on real revision data — enough to risk a Slurm job
  for G4/G5 on the PushCube cut first (where decode is certable);
  StackCube G4/G5 wait on the data-cut expansion (path 1 above) or
  the counterfactual-synthesis trick (path 2).

## Open notes (not blockers)

- ReviseHead loss is L2 to the centroid. An alternative is CE on a
  per-slot decoder trained alongside (the same per-slot decoders the
  joint trainer discards). For M2a's small data, L2 was the simpler
  choice and trivially passes on PushCube; both losses would
  converge similarly given the centroids are well-separated by
  IntentHead training.
- The ReviseHead is **shared across factors** within a task — one
  ReviseHead per task, trained on revisions to any factor. This is
  how the slot-local interface works: the `fp_vec` one-hot of the
  implicated factor tells ReviseHead which centroid space to target.
  Per-factor ReviseHeads is a possible alternative but adds N×
  parameters with no expressivity gain for our small data.
- Reproducibility: `python scripts/stage4_m2a_a2_eval.py
  --jsonl datasets/stage4/varied_intent/PushCube-v1/samples.jsonl
  --jsonl datasets/stage4/varied_intent/StackCube-v1/samples.jsonl
  --out-dir reports/stage4/m2a_a2/ --seed 0`. ~30s on CPU.
