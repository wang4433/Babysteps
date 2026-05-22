# Stage-4 Varied-Intent Data Cut + Tightened Recoverability — Design Spec

> **Status:** design approved 2026-05-22, pending written-spec review.
> **Why this exists:** Stage-4 Milestone-1 (schema recoverability) found the
> 90% gate **non-evaluable** on the existing `stage0_baselines` data: 17 of 18
> `(task, factor)` cells carry a single constant label (a constant predictor
> scores 1.0 — the certification loophole), and the one genuinely-varying cell
> (`StackCube/object_motion`) fails at 0.75 purely for lack of per-class
> samples. See `reports/stage4/schema_recoverability/notes.md`. This spec
> realizes that report's recommendation **(b): fix the data/labelling regime
> before encoder work** — concretely, build a varied + balanced cut and close
> the certification loophole. **It does not start M2 (SlotEncoder/IntentHead);
> M2 stays gated on this cut.**
>
> Authority: `goal.md` §"Stage 4 / Certification Interface" #1. Code
> source-of-truth: `babysteps/stage4/{features,probe,report}.py`,
> `babysteps/envs/{pushcube,stackcube}_runner.py`, `babysteps/envs/*_adapter.py`,
> `scripts/stage0_collect.py`.

---

## 1. Scope & Non-Goals

**In scope:**

- A **varied, balanced** Stage-0 data cut over **PushCube-v1 + StackCube-v1**
  in which `object_motion` varies across ≥3 of the 4 planar directions with
  **~10 episodes per class** in *both* tasks (PushCube additionally varies
  `contact_region` + `approach_direction`, collinearly with push direction).
- A **stratified collection plan** (sim-free, pure) + a **real-GPU** collection
  path: **pose-injection for PushCube** (Approach A) and **rejection sampling
  for StackCube** (Approach B).
- A **tightened certification** in `babysteps/stage4/report.py` that classifies
  every `(task, factor)` cell as **trivially-constant / label-identity /
  geometric** and gates only the geometric cells.
- A **re-run** of schema recoverability on the new cut →
  `reports/stage4/schema_recoverability_varied/`.
- One **MP4** of a PushCube episode where the initial (varied) intent fails on a
  blocked approach and the retry succeeds by revising **exactly**
  `approach_direction`.

**Explicitly NOT in scope (YAGNI):**

- **No M2 work.** No SlotEncoder, IntentHead, ReviseHead, or any learned latent.
  This cut is the *precondition* for M2, not M2.
- **PickCube-v1** and any third task. Its only varying factor under this scheme
  (`contact_region` grasp face) would be **label-identity** recovery anyway
  (the contact label is a feature one-hot), so it adds GPU + scene-control cost
  without adding a *geometric* recoverability result.
- **A constraint/obstacle axis** for within-task independence. Independence here
  comes **across tasks** (push direction in PushCube vs. cubeA→cubeB direction
  in StackCube are different physical DOF). Adding a PushCube obstacle axis is a
  separate, larger task extension.
- **Removing label-fed factors** (`contact_region_label`, `final_state`) from
  the feature vector. They are legitimate demo evidence (the third-person view
  observes contact and final state). We **flag** label-identity recovery, we do
  not remove the features.
- New intent factors, failure predicates, or revision operators. The single-
  factor revision path is reused unchanged.

---

## 2. Background: why only `object_motion` is a genuine recovery

The Stage-4 feature vector (`babysteps/stage4/features.py`, FEATURE_DIM=19) is:

```
[ start_xy(2), end_xy(2), disp_xy(2), |disp|(1), angle(1), path_len(1)   ]  # 9 trajectory stats
[ contact_region_label one-hot(6) ]
[ final_state one-hot(4)          ]
```

Consequences for "recoverability":

| Factor | How a probe recovers it | Class |
|---|---|---|
| `object_motion` | from trajectory **geometry** (`disp_xy`, `angle`) — a real inference | **geometric** |
| `contact_region` | it **is** an input one-hot | label-identity |
| `goal_state` | `final_state` **is** an input one-hot | label-identity |
| `approach_direction` (PushCube) | `= face_to_approach(contact_region)`, a deterministic fn of the contact one-hot | label-identity |
| `constraint_region`, `embodiment_mapping` | constant in this cut | trivially-constant |

So the **headline geometric test is `object_motion`**, which we make vary
(balanced) in both PushCube and StackCube. The label-identity factors are
reported (they genuinely vary and are genuinely recoverable from demo
evidence), but flagged and excluded from the geometric headline so a future
SlotEncoder cannot "pass cert #1" by echoing a label one-hot or memorizing a
constant.

---

## 3. Collection design

### 3.1 Stratified plan (sim-free, pure, new)

A new pure helper (proposed `babysteps/stage4/collection_plan.py`) maps
`(task_id, classes, episodes_per_class, seed_start)` → an ordered list of
`(seed, target_class)` assignments. Deterministic; no simulator. Unit-tested
for determinism and per-class balance. PushCube uses it directly to assign each
seed a target `object_motion` direction; StackCube uses it as the *target
quota* that rejection sampling (3.3) must fill.

Default cut: `classes = {translate_+x, translate_-x, translate_+y,
translate_-y}`, `episodes_per_class = 10` (≥3 of 4 classes must be filled; aim
for all 4). 5-fold StratifiedKFold then applies (no LeaveOneOut fallback), the
condition the M1 note identified as missing.

### 3.2 PushCube goal injection — Approach A

`PushCubeEnvRunner` gets an optional post-`reset` injection hook. After
`reset(seed)`, it sets poses so the **cube→goal** vector points in
`target_class`, then re-reads obs into `SceneState`:

- **Primary mechanism:** reposition the **cube** (a dynamic actor — `set_pose`
  is well-supported) to one of four offsets around the existing goal, realizing
  the four push directions. Cube reposition is lower-risk than moving the
  ManiSkill goal site.
- **Fallback:** move the `goal_region` pose if cube reposition desyncs the
  success check.
- **De-risk first:** a one-off **GPU spike** validates that PushCube-v1's
  success check honors the injected layout (cube reaches goal in the new
  direction) **before** the full collection. This is the first GPU step.

`oracle_correct_intent(scene)` already derives `object_motion` / `contact` /
`approach` from `scene.goal_xy − scene.cube_xy` (pushcube_adapter.py:32), so the
injected geometry flows through unchanged and the **real rollout** produces a
demo trajectory that reflects the varied intent.

### 3.3 StackCube rejection sampling — Approach B

StackCube's `object_motion` already varies natively (cubeA→cubeB direction;
the M1 imbalance was unlucky sampling, not a fixed scene). So **no env-internal
pose-setting** is needed:

1. Scan seeds; on each native `reset(seed)` compute the cubeA→cubeB direction
   (known from reset poses — **no rollout needed to bin**).
2. Accept a seed into its target bin until each class reaches the quota.
3. Cap total seeds scanned; **error out** (don't silently under-fill) if a class
   cannot be filled within the cap.
4. Run full rollouts only for accepted seeds.

### 3.4 Invariants preserved

- **Firewall (goal.md §5).** Injection writes only privileged `SceneState`.
  Demo evidence stays `object_trajectory` + `contact_region_label` +
  `final_state`. `features.py` and its static firewall test
  (`tests/test_stage4_features.py`) are untouched.
- **Single-factor revision.** Unchanged. PushCube still fails on blocked
  approach — `default_blocked_factory` returns `(intent.approach_direction,)`,
  so the block **adapts** to each episode's varied approach → revises
  `approach_direction`. StackCube still fails on the under-specified
  `goal_state` → revises `goal_state`. Varying the *initial* intent adds no
  factors to the revision.

---

## 4. Tightened certification (`babysteps/stage4/report.py`)

Replace the current two-way verdict (`trivially_constant` vs. pass/fail @0.90)
with a **three-way `cell_class`** per `(task, factor)`:

| `cell_class` | Condition | Gate treatment |
|---|---|---|
| `trivially_constant` | `n_unique_labels == 1` | **Excluded** from the gate. Reported, never a pass. |
| `label_identity` | factor ∈ a declared table: `contact_region`, `goal_state`, and `approach_direction` *for PushCube* | "recovered (label-identity)", **flagged, not counted** toward the headline. |
| `geometric` | everything else (today: `object_motion`) | **Gated:** `probe_acc_mean ≥ 0.90` **and** clears `majority_class_acc` and `shuffled_features_acc` by a **margin** (proposed `0.10`, surfaced in the report). |

- The label-identity table is an **explicit, documented constant** in the report
  module (it encodes a judgment — which factors are fed in / deterministic from
  inputs — and so is stated, not inferred).
- `schema_recoverability.json` gains a `cell_class` field per cell and the
  `margin` threshold; the markdown table gains a `class` column.
- The headline summary line becomes "`N geometric cells | M pass`" plus
  informational counts of label-identity and trivially-constant cells.

**Expected result of the re-run:** `object_motion` (geometric) clears 0.90 in
≥1 task — the M1 note's prediction, since the same 19-dim features already hit
0.75 under hostile LOO + imbalance and separate the four directions linearly.

---

## 5. MP4 render

Reuse `babysteps/render/pushcube.py::render_episode` on **one chosen seed from
the varied PushCube cut**, picked so the episode's varied approach is the
blocked side → clean `1_demo → 2_attempt_blocked → 3_retry`. Addition: a
**revision overlay** (or sidecar JSON the render reads) showing
`factor=approach_direction`, `old_value → new_value`, and the explicit
`frozen_factors` list — making the single-factor invariant visible in the
artifact. Captions describe object evidence only (demo-caption invariant).
Output: `renders/stage4_varied/`. GPU/slurm, mirroring
`slurm/render_pushcube.sbatch`.

---

## 6. Testing

**Sim-free (`tests/`, must stay GPU-free — invariant #5):**

- `collection_plan`: determinism, per-class balance, and the "class can't be
  filled → error" path.
- cert classification: synthetic cells exercising all three `cell_class` values
  + the margin gate boundary.
- a synthetic **varied** probe fixture proving the geometric cell clears the
  gate when balanced (locks in the M1 prediction without a GPU run).
- existing firewall + snapshot tests stay green.

**GPU / manual (not in the sim-free suite):** the pose-injection spike, the two
collection runs, the render — new `slurm/` sbatch scripts + `RUNBOOK.md`
entries. Pose-injection code lives in the runner (GPU side); its **plan** and
**cert** logic are sim-free and unit-tested.

---

## 7. Outputs & touch-points

**Outputs:**

- `datasets/stage4/varied_intent/{PushCube,StackCube}-v1/samples.jsonl` (+ per-
  task `report.{json,md}`)
- `reports/stage4/schema_recoverability_varied/` (json + md, tightened)
- `renders/stage4_varied/pushcube_*/` (MP4s)
- new `slurm/` sbatch scripts + `RUNBOOK.md` entries

**Code touch-points:**

- new `babysteps/stage4/collection_plan.py` (sim-free)
- `babysteps/stage4/report.py` (three-way `cell_class` + margin gate)
- `babysteps/envs/pushcube_runner.py` (optional injection hook; GPU side)
- a collection driver in `scripts/` (or a `--vary`/`--stratified-plan` mode on
  `stage0_collect.py`, mirroring its flag conventions + a CLI test)
- `babysteps/render/pushcube.py` (revision overlay)
- new tests under `tests/`; new `slurm/` sbatch; `RUNBOOK.md` + `CODE_MAP.md`
  updates.

Adapters stay sim-free and untouched except where the driver wires injection.

---

## 8. Acceptance gate

1. `object_motion` balanced across ≥3 directions at ~10 ep/class in **both**
   PushCube and StackCube; `samples.jsonl` written for each.
2. Re-run report classifies all `(task, factor)` cells (three-way) and the
   **geometric** `object_motion` cell **passes** (≥0.90, clears chance +
   shuffled by the margin) in ≥1 task.
3. No trivially-constant or label-identity cell is counted as a gate pass.
4. One MP4 shows initial-intent failure → single-factor (`approach_direction`)
   revision → retry success, with frozen factors listed.
5. Sim-free `tests/` suite stays GPU-free and green; new sim-free tests added.
