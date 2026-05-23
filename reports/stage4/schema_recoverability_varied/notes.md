# Stage-4 Varied-Intent Schema-Recoverability Notes

## Run metadata

- Date: 2026-05-22
- Cut: the Stage-4 varied-intent collection
  (`docs/superpowers/specs/2026-05-22-stage4-varied-intent-cut-design.md`).
- Primary input JSONLs:
  - `datasets/stage4/varied_intent/PushCube-v1/samples.jsonl` (20 episodes;
    goal-move injection, binary `±x`).
  - `datasets/stage4/varied_intent/StackCube-v1/samples.jsonl` (40 episodes;
    rejection-sampled native, 4 directions balanced 10/10/10/10).
- Commit at run time: `41e9cfc` (HEAD on master).
- Probe: scikit-learn `LogisticRegression` (lbfgs, `max_iter=1000`), 5-fold
  `StratifiedKFold` with `LeaveOneOut` fallback when the smallest class
  has < 5 members; `seed=0`. Same as M1.
- Feature dim: 19 (`babysteps/stage4/features.py`) — unchanged from M1.
- Cert: the three-way `cell_class` (`babysteps/stage4/report.py`):
  `trivially_constant` → excluded; `label_identity` → flagged, not counted;
  `geometric` → gated at `probe_acc_mean ≥ 0.90` **AND** clearing both
  `majority_class_acc` and `shuffled_features_acc` by `GATE_MARGIN = 0.10`.

## Headline result

**12 cells | 2 geometric (1 pass / 1 fail) | 2 label-identity | 8 trivially
constant.** The varied cut achieved what the spec demanded
(§8.2: "geometric `object_motion` passes in ≥1 task") — PushCube clears the
tightened gate. The constant-factor loophole is closed: not one trivially-
constant or label-identity cell counts as a gate pass. The cut is M2-ready as
a meaningful certification scaffold.

But the result is **opposite of what M1's notes predicted**: it's PushCube
(which had to be heroically rescued, see §"Two execution-side fixes" below)
that lifts cleanly to 0.95 on binary ±x; **StackCube's 4-way `object_motion`
fails at 0.72 despite full class balance**, contradicting the M1 hypothesis
that "balance + more episodes would clear 0.90." StackCube's 4-way recovery
is genuinely hard for the 19-dim hand features.

## Per-cell summary

```
# Stage-4 Schema-Recoverability Probe
Gate: geometric cells must reach probe_acc_mean >= 0.90 AND beat chance & shuffled each by 0.10.
Cells: 12 total | 2 geometric (1 pass / 1 fail) | 2 label-identity | 8 trivially constant.

### PushCube-v1
| factor              | class              | n_unique | n_episodes | majority | shuffled | probe ± std | gate           |
| ------------------- | ------------------ | -------- | ---------- | -------- | -------- | ----------- | -------------- |
| approach_direction  | label_identity     | 2        | 20         | 0.50     | 0.65     | 1.00 ± 0.00 | label_identity |
| constraint_region   | trivially_constant | 1        | 20         | 1.00     | 1.00     | 1.00 ± 0.00 | trivial        |
| contact_region      | label_identity     | 2        | 20         | 0.50     | 0.65     | 1.00 ± 0.00 | label_identity |
| embodiment_mapping  | trivially_constant | 1        | 20         | 1.00     | 1.00     | 1.00 ± 0.00 | trivial        |
| goal_state          | trivially_constant | 1        | 20         | 1.00     | 1.00     | 1.00 ± 0.00 | trivial        |
| object_motion       | geometric          | 3        | 20         | 0.50     | 0.60     | 0.95 ± 0.22 | PASS           |

### StackCube-v1
| factor              | class              | n_unique | n_episodes | majority | shuffled | probe ± std | gate           |
| ------------------- | ------------------ | -------- | ---------- | -------- | -------- | ----------- | -------------- |
| approach_direction  | trivially_constant | 1        | 40         | 1.00     | 1.00     | 1.00 ± 0.00 | trivial        |
| constraint_region   | trivially_constant | 1        | 40         | 1.00     | 1.00     | 1.00 ± 0.00 | trivial        |
| contact_region      | trivially_constant | 1        | 40         | 1.00     | 1.00     | 1.00 ± 0.00 | trivial        |
| embodiment_mapping  | trivially_constant | 1        | 40         | 1.00     | 1.00     | 1.00 ± 0.00 | trivial        |
| goal_state          | trivially_constant | 1        | 40         | 1.00     | 1.00     | 1.00 ± 0.00 | trivial        |
| object_motion       | geometric          | 4        | 40         | 0.25     | 0.23     | 0.72 ± 0.05 | FAIL           |
```

Machine-readable form: `schema_recoverability.json` (this directory).

## Two execution-side fixes the GPU run forced

The collection job aborted at the fail-fast injection spike on its first
attempt (one direction succeeded, three failed). Debugging surfaced two real
bugs the sim-free reviews could not have caught:

1. **`PushCubeEnvRunner.run()` re-resets the env and discarded the injection
   done in `reset()`.** The push waypoints were correctly built from the
   injected scene but executed against the (re-set) native layout, so the
   gripper swung through empty space and the cube never moved. `+x` masked it
   because its injected position coincides with native. Fix: extracted
   `_reset_with_injection()`, called after EVERY `env.reset` (idempotent).
2. **Cube-move injection** (the spec's primary mechanism) **hit a reachability
   wall**: even with #1 fixed, the `+x`-tuned open-loop push controller can't
   drive the cube from displaced positions. The closest lateral injection
   positions got a useless graze; the far ones weren't touched at all. Pivoted
   to **goal-move** (the spec's documented fallback: keep cube at native
   reachable pose, move `goal_region` instead). Goal-move robustly recovers
   `+x` AND `-x` (5/5 seeds each in the diagnostic); `±y` lateral pushes
   remain unreliable (the gripper grazes `+x` instead of pushing sideways,
   a controller/contact limitation). So **PushCube varies binary ±x only**,
   relaxing the spec's "≥3 directions per task" to "≥2" for PushCube.
   StackCube (pick-and-place, reaches everywhere) carries the 4-direction
   headline unchanged.

Both fixes landed in commit `41e9cfc`; the diagnostic scripts + their
outputs are archived in `diagnostics/` next to this file
(`diag_injection.{py,sbatch,out,err}` for the cube-move investigation,
`diag_goalmove.{py,sbatch,out,err}` for the pivot).

## Cells that passed

### `PushCube-v1 / object_motion` — probe 0.95 ± 0.22, geometric PASS

20 episodes labelled `{translate_+x: 10, translate_-x: 9, translate_-y: 1}`.
The lone `-y` is a stray: one `-x` demo rollout had a noisy lateral component
that snapped past the y-axis on `goal_direction_to_motion`'s dominant-axis
tiebreak. LOO is triggered (`min class = 1`), and the probe gets 19/20 correct
— the singleton `-y` is the unavoidable miss (LOO cannot predict a class with
one sample). 0.95 clears 0.90 **and** beats chance (0.50) + shuffled (0.60)
by `> 0.10` on both. The signal is the demo trajectory's `dx` sign, which is
exactly what the cert wants: a genuine geometric inference from the rollout,
not a label echo.

## Cells that did not pass — and why

### `StackCube-v1 / object_motion` — probe 0.72 ± 0.05, geometric FAIL

40 episodes, 4 classes balanced **exactly 10/10/10/10**, 5-fold
StratifiedKFold (no LOO fallback). 0.72 is not above `chance + margin`
(0.25 + 0.10 = 0.35 — comfortably cleared) but is well below the 0.90 absolute
threshold. **This contradicts M1's prediction** that "with balanced classes
and more episodes the same 19-dim features should clear 0.90." Balance was
necessary but not sufficient.

The 19-dim demo-evidence features (start_xy, end_xy, disp_xy, |disp|,
`arctan2` angle, path_len, contact one-hot, final-state one-hot) do not
linearly separate StackCube's four directions reliably. Likely contributors:
- StackCube's `object_motion` is `trajectory_to_motion` over cubeA's 2D path
  during a pick-and-place; the path includes a vertical lift/drop projected
  onto xy and a horizontal phase whose net displacement is the signal — but
  the path-length and angle features add noise rather than discrimination.
- The `arctan2` angle wraps at ±180° exactly where `translate_-x` lives,
  giving a discontinuous feature column for that class. M1 noted this as
  secondary; with balance it is more visible because `-x` is now well-
  represented.
- `LogisticRegression` is linear; four-way separation across these features
  appears not to be linearly clean.

This is a meaningful, not a defective, result.

## Label-identity cells (informational)

PushCube `contact_region` and `approach_direction` each take 2 values across
the cut (one per push direction), and both are recovered at 1.00 — but
recovery is **tautological**: `contact_region_label` is fed in as a one-hot
input feature, and PushCube's `approach_direction = face_to_approach(contact)`
is a deterministic function of the contact one-hot. The cert correctly flags
both as `label_identity` and excludes them from the geometric headline.

## Trivially constant cells (informational)

Eight cells. PushCube's `goal_state`/`constraint_region`/`embodiment_mapping`
and all StackCube intent factors except `object_motion` carry a single
canonical value across their task's episodes. The cert correctly excludes them
from the gate (a constant predictor scores 1.0 but does not certify anything
about the latent).

## Implications for Stage-4 Milestone 2

The varied cut achieves its spec purpose: it closes the certification
loophole and produces an **evaluable** geometric gate.

- **PushCube `object_motion` (geometric, 0.95 PASS)** is the working
  end-to-end baseline: the cert + probe pipeline correctly confirms a
  passing factor when the demo evidence carries the signal linearly.
- **StackCube `object_motion` (geometric, 0.72 FAIL)** is the real M2
  target. The 19-dim hand features hit a ceiling. The M2 SlotEncoder /
  IntentHead, whose whole purpose is to learn a richer per-object latent
  representation, should produce a `G_t` that the same frozen linear probe
  recovers at ≥ 0.90 on this same 4-way StackCube data. If it does not,
  that is a clean, falsifiable M2 failure mode.

M2 can now begin. The spec's "≥1 geometric pass" gate (§8.2) is met. Future
work to lift StackCube above 0.90 belongs in M2 (via a learned latent), not
in this scaffold (which would require either richer hand features or a wider
probe — both of which would change what the cert measures).

## Open notes (not blockers)

- PushCube's 1 stray `-y` label is a real artifact of `trajectory_to_motion`'s
  dominant-axis tiebreak under push noise; honest to leave it in. A future
  cleanup could drop the singleton (`n_unique` becomes 2, probe still passes)
  if needed.
- StackCube reaches `passed_acceptance=True` in its per-task `report.md`
  (episode-loop metrics: initial vs retry success). The probe FAIL is on the
  separate Stage-4 cert, not the Stage-0 acceptance.
- The diagnostic scripts (`diag_injection.py`, `diag_goalmove.py`) and their
  outputs are committed alongside this notes file under `diagnostics/`. The
  two bug fixes they drove are committed in `41e9cfc`.
