# Stage 5 P3 — World-Model Counterfactual Verification Design Spec

> **Status:** draft 2026-05-26. Implements Priority 3 of `goal.md` §"Stage 5".
> Builds on P1 (vision-grounded `G_t`, DINOv2 768-d → IntentHead → `G ∈ ℝ^{6×32}`)
> and P2 (VLM-constrained attribution + slot-local ReviseHead).
> Open questions are flagged with **[Q]** — resolve before the plan.

## 1. Problem

Stage 4 G3 (selectivity certification) is currently a **mechanical bit-identity
check**: after a revision, the schema fields other than the edited factor must
be byte-equal to the originals. This is trivially satisfied by the slot-local
operator and proves nothing about the *physics* — a reviewer can reasonably ask:
"if you had edited a different slot, would the executed behavior actually have
preserved the rest?"

For an ICLR claim that single-factor revision is *physically sufficient*, G3
needs a forward prediction: predict the post-execution slot state, and show
that editing slot `i` leaves the post-execution slots `j ≠ i` indistinguishable
from the no-edit baseline (within a simulator noise floor).

## 2. Goal

Train a latent dynamics + reward model over the Stage-5 slot-intent space, and
use it to (a) certify G3 selectivity with forward predictions and (b) rank
candidate revisions in imagination without sim rollouts.

**Gate (G3-WM):** for held-out (episode, edit-slot `i`) pairs, the predicted
post-execution drift on slot `j ≠ i` is statistically indistinguishable from
the same-intent replay noise floor, while the drift on slot `i` is well
outside it.

## 3. Design

### 3.1 What the world model predicts

Two model components:

```
f_φ : (G_demo, a) ──→ G_post              # latent dynamics
r_φ : G_post     ──→ p̂(success) ∈ [0,1]   # reward / success head
```

- `G_demo ∈ ℝ^{6 × 32}` — slot intents from the demo (P1 IntentHead output).
- `a` — skill-primitive parameters (3.2).
- `G_post ∈ ℝ^{6 × 32}` — slot intents extracted from the **post-execution**
  trajectory frames by re-running the same DINOv2 → IntentHead pipeline on the
  attempt video (not the demo).
- `p̂(success)` — predicted task success from `G_post`.

Rationale for choosing **slot-space** dynamics over per-step pixel/latent
dynamics:

1. The selectivity claim is fundamentally about slot drift, not pixel drift.
   Predicting in slot space makes the gate directly interpretable as
   "factor `j` was unaffected by editing factor `i`."
2. Per-step latent dynamics would require per-step DINOv2 caching across full
   trajectories (we currently cache only demo + final-attempt frames). Slot-space
   collapses a full execution to one (G_demo, a) → G_post transition, which is
   the granularity our supervision and revision interface already work at.
3. Per-step is still possible as a P3.5 extension if reviewers ask for it.

**[Q3.1]** Should the model also predict `G_post` *uncertainty* (e.g. an
ensemble or a heteroscedastic head) so the selectivity threshold can be
calibrated per-episode rather than per-task?

### 3.2 Action representation `a`

The skill compiler maps `G` to skill primitive params. The natural `a` is the
**skill-parameter vector** the compiler emits — not low-level Franka deltas.
Concretely (per current skill primitives):

- PushCube: `(approach_side ∈ {+x,-x,+y,-y}, push_distance, push_height)`.
- PickCube: `(grasp_face_id, grasp_offset, lift_height)`.
- StackCube: `(grasp_face_id, place_xy, place_z)`.

We will encode `a` as a per-task fixed-length float vector with a `task_id`
one-hot prepended:

```
a ∈ ℝ^{n_task + k_max}     n_task = 3, k_max = max skill-param dim across tasks
```

This keeps the model task-conditioned without per-task heads. **[Q3.2]** Use
a single shared model with task-conditioning, or one model per task? Shared
is more data-efficient but risks negative transfer if skill semantics diverge.
Default: shared, with a per-task ablation.

### 3.3 Architecture

```python
# babysteps/stage5/world_model.py

class SlotDynamics(nn.Module):
    """f_φ : (G_demo, a) → G_post.

    Inputs:
        G_demo : (B, 6, 32)
        a      : (B, A_dim)
    Output:
        G_post : (B, 6, 32)
    """
    # Flatten G_demo → (B, 192). Concat with a. 2-layer MLP. Reshape.
    # GELU, hidden=256, ~0.1M params. Same scale as IntentHead — keep small.

class SuccessHead(nn.Module):
    """r_φ : G_post → logit(success)."""
    # Flatten (B, 6, 32) → (B, 192). Linear(192, 64) → GELU → Linear(64, 1).
```

No transformer, no attention. The slot intents are already
information-dense and per-task; we don't need sequence modeling at this
level. Keep it as the minimum architecture that can pass the gate.

### 3.4 Training data

The episodes already collected for P1 + P2 give us:

| Slice | What we have | What's missing |
| --- | --- | --- |
| `varied_intent/{task}/` | demo frames + DINOv2 features (50–100 ep/task) | execution-trajectory frames |
| `p2_vlm/{task}/` | demo G, revised G, attempt frames (single PNG), success label | multi-frame attempt video → no `G_post` extractable |

To produce `G_post` we need **multi-frame rollout capture** of the attempt
trajectory, not just the final still. New collection job:

```
scripts/stage5_p3_collect_rollouts.py
  --task PushCube-v1 --n-seeds 200 --frames-per-rollout 16
  → datasets/stage5/p3_rollouts/{task}/{seed}_rollout.npz
                                       /{seed}_G_demo.npy
                                       /{seed}_a.npy
                                       /{seed}_G_post.npy
                                       /{seed}_success.json
```

Per task we want ≥ 200 episodes (3× P1 cut) so the dynamics MLP has enough
samples; this is cheap rendering (~30 min per task on an A100).

**[Q3.4]** Do we need to vary `a` *off-policy* (perturb skill params away from
what `G` would prescribe) so the model sees counterfactual (G, a) pairs at
training time? Otherwise `a` is a deterministic function of `G_demo` and the
model can't disambiguate "intent caused this" from "action caused this." I
think yes — collect with ε-noise on skill params (e.g. random push_distance
± 30%) for ~30% of episodes.

### 3.5 G3 selectivity gate (forward-prediction version)

Procedure for one (task, slot `i`) cell:

1. **Noise-floor distribution.** For each held-out episode `e`, replay the
   same `(G_demo, a)` with `k=10` different sim seeds; extract `G_post^{(k)}`.
   Per-slot drift `δ_j(e) = mean_k ||G_post^{(k)}[j] - G_post_baseline[j]||₂`
   forms the noise distribution `N_j`.
2. **Counterfactual pair.** For the same episode `e`:
   - Baseline: predict `G_post = f_φ(G_demo, a)`.
   - Edit slot `i`: produce `G_demo'` via the ReviseHead (single-factor edit at
     `i`). Predict `G_post' = f_φ(G_demo', a')` where `a'` is the skill-param
     vector the compiler would emit from `G_demo'`.
   - Drift `Δ_j(e) = ||G_post[j] - G_post'[j]||₂` for all slots `j`.
3. **Gate.**
   - **Selectivity:** `Δ_j` for `j ≠ i` falls inside the 95th-percentile of
     `N_j`. Pass rate per cell must be ≥ 90%.
   - **Effectiveness:** `Δ_i` is outside the 95th-percentile of `N_i` (i.e.
     the edit *did* move slot `i`). Pass rate per cell must be ≥ 80%.

The selectivity + effectiveness pair is what distinguishes "the model is just
noisy" from "the model correctly localizes the edit to slot `i`."

**[Q3.5]** Are 95th percentile / 90% pass-rate the right thresholds? Stage 4
G4 uses `Δpp ≥ 10`; this is a different metric (drift vs success delta), so
new thresholds need empirical calibration from the noise distribution on
PushCube before locking the rest in.

### 3.6 Use 2 — revision ranking (bonus)

For an episode where the VLM diagnoses slot `i` with low confidence (or
attribution is ambiguous), enumerate candidate edits at each plausible slot,
predict `r_φ(f_φ(G', a'))` for each, and pick the slot with the highest
predicted success. This is *post-hoc* re-ranking; the VLM and ReviseHead
still produce the candidates.

Pass criterion: on episodes where the VLM attribution is wrong (4% on
PickCube, 100% on StackCube `goal_state`), the ranker recovers the correct
slot ≥ 50% of the time. This is a stretch goal, not a primary P3 gate.

### 3.7 What this does NOT do

- Does not replace the simulator. Sim rollouts remain the final success
  arbiter for Stage-5 end-to-end gates. The world model is for G3
  selectivity certification and imagination ranking.
- Does not predict per-step pixels or low-level state. Slot-space only.
- Does not learn a policy. The skill compiler remains the action source.

## 4. Evaluation

### 4.1 Primary gate (G3-WM)

Per task ∈ {PushCube, PickCube, StackCube}, per slot `i` in the task's
revisable set:

| Metric | Threshold |
| --- | --- |
| Selectivity pass-rate (`Δ_{j≠i}` inside noise) | ≥ 90% |
| Effectiveness pass-rate (`Δ_i` outside noise) | ≥ 80% |

### 4.2 Held-out protocol

- 80/20 train/test split on episodes, stratified by (task, ground-truth
  failed-factor).
- Noise-floor `N_j` is estimated **only from training-set replays** to
  avoid leakage.
- Counterfactual edits use the **frozen** P2 ReviseHead (no retraining).

### 4.3 Sanity baselines

| Baseline | Selectivity expectation |
| --- | --- |
| Identity (predict `G_post = G_demo`) | should pass selectivity trivially but fail effectiveness |
| Shuffled `a` | should fail both |
| Linear regression on `(G_demo, a)` | should be close to MLP; if not, the MLP is overfitting |

### 4.4 Ablations

1. Per-task model vs shared model with task-conditioning (resolves **[Q3.2]**).
2. With vs without action perturbation in training data (resolves **[Q3.4]**).
3. Ensemble of 5 vs single model — for **[Q3.1]** uncertainty calibration.

## 5. Implementation plan (preview — full plan to follow)

| Step | What | Depends on | GPU? |
| --- | --- | --- | --- |
| S1 | `scripts/stage5_p3_collect_rollouts.py` — multi-frame rollout capture + `G_post` extraction | P1 vision_features, P2 skill compiler | Yes (ManiSkill) |
| S2 | Action perturbation in collection (ε-noise on skill params) | S1 | Yes |
| S3 | `babysteps/stage5/world_model.py` — `SlotDynamics` + `SuccessHead`; sim-free unit tests | None | No |
| S4 | `scripts/stage5_p3_train_world_model.py` — train + held-out MSE | S1, S3 | No (small model, CPU torch) |
| S5 | `scripts/stage5_p3_noise_floor.py` — replay-based noise distribution per slot | S1 | Yes (replays) |
| S6 | `scripts/stage5_p3_g3_eval.py` — selectivity + effectiveness gate | S4, S5 | No |
| S7 | Per-task ablations + report (`reports/stage5/p3_world_model/`) | S6 | No |
| S8 | Optional: revision-ranking eval on VLM-misattributed episodes | S6, P2 | No |

PushCube first end-to-end (cleanest signal — P1/P2 at ceiling), then
PickCube, then StackCube.

**Estimated timeline:** S1–S2 in ~3 days (data collection dominant).
S3–S6 in ~1 week. S7–S8 in ~3 days.

## 6. Risks

1. **Slot-space dynamics may be too coarse to capture meaningful drift.**
   If `G_post` is nearly equal to `G_demo` for most episodes (skill execution
   doesn't perturb intent representation), the noise floor will be tiny and
   every edit will look "outside noise." Mitigation: report
   `||G_post - G_demo||` distribution before training; if it's degenerate,
   move to per-step latent dynamics (P3.5).
2. **StackCube is unsolved end-to-end.** Per `reports/stage5/p2_vlm_attribution/`,
   StackCube has 0% success in both C1 and C2. Training the world model on
   StackCube data means the `r_φ` head sees almost no positive examples.
   Mitigation: collect a **success-balanced** rollout cut for StackCube
   (oracle revisions on episodes where we know the correct factor) to give
   `r_φ` positive samples. This is necessary anyway for revision-ranking
   to work on StackCube.
3. **Action perturbation may break sim execution.** If we perturb skill
   params too aggressively, the skill compiler may produce trajectories
   that the controller refuses. Mitigation: validate ε ranges per skill
   primitive on PushCube before extending.
4. **Shared model negative transfer.** If shared task-conditioning hurts,
   the per-task fallback is a 1-day change. Not load-bearing.

## 7. What success looks like

- G3-WM gate passes on PushCube and PickCube (selectivity ≥ 90%,
  effectiveness ≥ 80%) for all revisable slots.
- StackCube either passes the gate or yields a documented negative result
  ("world model cannot certify slot-`goal_state` edits because the skill
  compiler does not execute them"), which is informative for the
  paper's limitations section.
- A one-paragraph result: "We trained a slot-space dynamics model
  `f_φ(G_demo, a) → G_post` on rendered execution trajectories. For every
  task and revisable slot, predicted post-execution drift on non-edited
  slots is statistically indistinguishable from the same-intent simulator
  replay noise floor, while drift on the edited slot is well outside it.
  G3 selectivity is no longer a mechanical bit-identity check — it is a
  forward-prediction certification that single-factor revision is
  physically localized."

## 8. Open questions consolidated

| ID | Question | Default |
| --- | --- | --- |
| Q3.1 | Predict uncertainty (ensemble / heteroscedastic)? | Ensemble of 5 if calibration matters |
| Q3.2 | Shared model with task-conditioning vs per-task? | Shared, with per-task ablation |
| Q3.4 | Off-policy action perturbation in training data? | Yes, ε-noise on ~30% of episodes |
| Q3.5 | Selectivity threshold (95th percentile, 90% pass-rate)? | Calibrate empirically on PushCube |

Resolve these in the plan or in a brief brainstorm before plan-writing.
