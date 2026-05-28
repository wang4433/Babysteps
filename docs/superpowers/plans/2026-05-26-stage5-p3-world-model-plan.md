# Stage-5 P3 World-Model Counterfactual Verification — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** `docs/superpowers/specs/2026-05-26-stage5-p3-world-model-design.md`.

**Goal:** Train a slot-space dynamics + success model and use it to certify G3 selectivity with forward predictions (not bit-identity). Per the spec: `f_φ(G_demo, a) → G_post`, `r_φ(G_post) → p̂(success)`. Gate: predicted post-execution drift on non-edited slots inside the same-intent replay noise floor (≥ 90% pass), drift on the edited slot outside it (≥ 80% pass).

**Scope (this plan, PushCube-first):** Implement everything end-to-end on **PushCube-v1**. The P1 vision IntentHead exists only for PushCube (`models/stage5/p1_vision/PushCube-v1/intent_head.pt`); PickCube has no Stage-5 vision data, StackCube has data but no IntentHead. Extending P1 to PickCube/StackCube is a separate plan and is listed as a non-blocking follow-up here.

**Tech Stack:** Python 3, `torch` + CUDA (small MLP, fits anywhere), ManiSkill (rollout collection + same-seed replays), Stage-4 `IntentHead`/`ReviseHead`, Stage-4 `vision_features.extract_vision_features`, the existing `babysteps` sim-free package + `env_runner.run`, Slurm on Gilbreth (`a100-40gb` partition, `--qos=standby`).

---

## Architectural Decisions (locks the spec's open questions)

| Spec Q | Decision | Why |
|---|---|---|
| Q3.1 (uncertainty) | **Single model, no ensemble** in v1. If the noise floor is too tight to discriminate, swap to a 5-ensemble in a follow-up. | Simplest baseline. Calibration uncertainty is a P3.5 concern. |
| Q3.2 (shared vs per-task) | **Per-task models** for v1. One shared model is a planned ablation, not the primary. | Only PushCube has a P1 IntentHead anyway. Per-task removes the "negative transfer" confound from the first gate run. |
| Q3.4 (action perturbation) | **Yes — ε-noise on 30% of training episodes.** | Without off-policy `a`, the model cannot disambiguate intent-caused from action-caused drift. ε is per-skill (e.g. `push_distance ± 30%`). Documented in `collect_rollouts.py`. |
| Q3.5 (selectivity threshold) | **95th percentile of the noise distribution, 90% per-cell pass-rate.** Re-calibrate after seeing the empirical PushCube noise floor if it is degenerate. | Matches the spec default. Reviewer-readable: "selectivity holds within 95% confidence of simulator noise". |
| Slot-space dynamics or per-step? | **Slot-space** (spec §3.1 framing). | Already debated in the spec. Per-step is a P3.5 fallback if slot-space `G_post ≈ G_demo` for most episodes. |
| Action representation `a` | **Skill-param vector** the compiler emits, with a 3-way `task_id` one-hot prepended. `a_dim = 3 + k_max`. | Spec §3.2. Per-task models technically don't need the one-hot, but keep it so the shared-model ablation is a one-line change. |
| Training-set size per task | **300 episodes** (200 on-policy + 100 with ε-noise). | Stage-4 used 50–100; world models need more samples for the action-conditioning. 300 is ~45 min A100 rendering per task. |
| Noise-floor replays | **k=10 same-seed replays per held-out episode.** | Balances variance estimation vs A100 wall time. Each replay is ~3 s × 50 held-out × 10 = ~25 min per task. |
| What `G_post` is extracted from | **The last 8 frames of the execution trajectory**, pooled with `cls_mean` (same pooling as P1 demo features). | Reuses `extract_vision_features` unmodified. Reviewer-readable: "post-execution intent is the same vision-grounded `G` applied to the executed trajectory." |
| Skipping non-PushCube tasks | **Yes, this plan is PushCube-only.** A follow-up plan extends to PickCube + StackCube after P1 IntentHeads exist there. | See scope note. PushCube is at ceiling in P1/P2, the cleanest signal for a new gate. |

---

## File Structure

**New files:**
```text
babysteps/stage5/world_model.py                                # SlotDynamics + SuccessHead modules
babysteps/stage5/rollout_capture.py                            # helpers: run an episode, save (demo_frames, attempt_frames, a, success)
tests/test_world_model.py                                      # sim-free unit tests (shapes, training step, mock data)
tests/test_rollout_capture.py                                  # sim-free unit tests (mock env runner)
scripts/stage5_p3_collect_rollouts.py                          # GPU: render N episodes (on-policy + ε-perturbed), extract G_demo & G_post
scripts/stage5_p3_train_world_model.py                         # CPU: train SlotDynamics + SuccessHead on collected rollouts
scripts/stage5_p3_noise_floor.py                               # GPU: k=10 same-seed replays per held-out episode, extract per-slot drift
scripts/stage5_p3_g3_eval.py                                   # CPU: counterfactual eval + gate report
slurm/stage5_p3_collect_rollouts.sbatch
slurm/stage5_p3_noise_floor.sbatch
slurm/stage5_p3_train_world_model.sbatch
docs/superpowers/specs/2026-05-26-stage5-p3-world-model-design.md   # already written
```

**Modified files:**
```text
goal.md                                                        # tick P3 done note (after gate passes)
slurm/CLAUDE.md                                                # record P3 gate results when run
```

**Output paths (created by run):**
```text
datasets/stage5/p3_rollouts/PushCube-v1/
    episodes.jsonl                                             # per-episode record: seed, on_policy/perturbed, success, a_vec
    seed_NNNN/
        demo_frames.npz                                        # (T_demo, H, W, 3) uint8
        attempt_frames.npz                                     # (T_attempt, H, W, 3) uint8
        G_demo.npy                                             # (6, 32) float32  (P1 IntentHead applied to demo frames)
        G_post.npy                                             # (6, 32) float32  (P1 IntentHead applied to last-8 attempt frames)
        a.npy                                                  # (a_dim,) float32 skill-param vector
datasets/stage5/p3_noise_floor/PushCube-v1/
    seed_NNNN_replay_KK.npy                                    # G_post for replay KK ∈ {0..9}
    noise_distribution.npz                                     # per-slot ||δ_j|| arrays + 95th percentile cutoffs
models/stage5/p3_world_model/PushCube-v1/
    slot_dynamics.pt
    success_head.pt
    meta.json
reports/stage5/p3_world_model/PushCube-v1/
    dynamics_eval.json                                         # held-out MSE per slot, success-head AUC
    g3_selectivity.json                                        # per-(slot i) drift distributions + gate pass/fail
    report.md
reports/stage5/p3_world_model/summary.md                       # cross-condition (PushCube only in v1) dashboard
```

---

## Task 1: Sanity-check the rollout-capture interface (no new code yet)

Before writing collection code, verify the existing skill compiler exposes a clean `a` and that `env_runner.run` returns multi-frame attempt video. If either is missing, the spec needs a small revision.

**Files:** none modified.

- [ ] **Step 1: Read the relevant existing code paths**

Read (do not modify):
- `babysteps/episode.py` — `run_episode` signature
- `babysteps/envs/task_adapter.py` — `revise_intent`, action compilation entry points
- `babysteps/render/pushcube.py` — what the rendering script captures
- `babysteps/stage4/latent_policy.py` — how `IntentHead` output `G` is consumed

Report:
- Where the skill compiler turns `Intent` (discrete) into action params for PushCube — record the exact field names that should go into `a`.
- Whether `env_runner.run` returns attempt frames or only success/state. (Likely state only; we will need to add a frame-capture wrapper.)
- Whether the existing render scripts re-execute or just replay — relevant for noise-floor.

- [ ] **Step 2: Decision point**

If the skill compiler does NOT expose a clean param vector for PushCube, add a tiny extractor function in `babysteps/stage5/rollout_capture.py` (Task 4) — do not modify `task_adapter.py`. Record the chosen `a` definition here in the plan before moving on.

PushCube `a` candidate (from `babysteps/render/pushcube.py` skill primitives):
```
a = (
    task_id_onehot[3],          # [1, 0, 0] for PushCube
    approach_side_id ∈ {0..3},  # +x, -x, +y, -y (cast to 4-way one-hot)
    push_distance ∈ R,          # meters
    push_height   ∈ R,          # meters
)
# total a_dim = 3 + 4 + 1 + 1 = 9. With k_max=9 for v1 (PushCube only).
```
Confirm field names match the actual skill primitive; if different, update the spec §3.2 inline.

- [ ] **Step 3: Commit**

No code change → no commit. Move to Task 2.

---

## Task 2: `babysteps/stage5/world_model.py` — modules + sim-free unit tests

**Files:**
- Create: `babysteps/stage5/world_model.py`
- Create: `tests/test_world_model.py`

- [ ] **Step 1: Write `SlotDynamics` and `SuccessHead`**

```python
# babysteps/stage5/world_model.py
"""Stage-5 P3 world model: slot-space latent dynamics + success head.

f_φ : (G_demo, a) → G_post     # SlotDynamics
r_φ : G_post     → p̂(success)  # SuccessHead

Both modules operate on the P1 IntentHead output space
G ∈ ℝ^{B, 6, 32}. Action `a` is a per-task skill-param vector with
a task_id one-hot prepended (see plan §"Task 1" for the schema).
"""

import torch
from torch import nn

class SlotDynamics(nn.Module):
    def __init__(self, n_factors: int = 6, d_slot: int = 32,
                 a_dim: int = 9, hidden: int = 256):
        super().__init__()
        self.n_factors = n_factors
        self.d_slot = d_slot
        self.a_dim = a_dim
        z_dim = n_factors * d_slot
        self.net = nn.Sequential(
            nn.Linear(z_dim + a_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, z_dim),
        )

    def forward(self, G_demo: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        B = G_demo.shape[0]
        z = G_demo.view(B, -1)
        h = torch.cat([z, a], dim=-1)
        out = self.net(h)
        return out.view(B, self.n_factors, self.d_slot)


class SuccessHead(nn.Module):
    def __init__(self, n_factors: int = 6, d_slot: int = 32, hidden: int = 64):
        super().__init__()
        z_dim = n_factors * d_slot
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, G_post: torch.Tensor) -> torch.Tensor:
        B = G_post.shape[0]
        return self.net(G_post.view(B, -1)).squeeze(-1)  # logit


def save(model: nn.Module, path: str, init: dict) -> None:
    torch.save({"state_dict": model.state_dict(), "init": init}, path)


def load_slot_dynamics(path: str, map_location: str = "cpu") -> SlotDynamics:
    pkg = torch.load(path, map_location=map_location, weights_only=False)
    m = SlotDynamics(**pkg["init"])
    m.load_state_dict(pkg["state_dict"])
    return m


def load_success_head(path: str, map_location: str = "cpu") -> SuccessHead:
    pkg = torch.load(path, map_location=map_location, weights_only=False)
    m = SuccessHead(**pkg["init"])
    m.load_state_dict(pkg["state_dict"])
    return m
```

- [ ] **Step 2: Write sim-free unit tests**

```python
# tests/test_world_model.py
"""Stage-5 P3 world model — sim-free shape + training-step tests.

These run on the login node (no GPU, no ManiSkill).
"""

import torch
from babysteps.stage5.world_model import (
    SlotDynamics, SuccessHead, save, load_slot_dynamics, load_success_head,
)


def test_slot_dynamics_shape():
    m = SlotDynamics()
    G = torch.randn(4, 6, 32)
    a = torch.randn(4, 9)
    out = m(G, a)
    assert out.shape == (4, 6, 32)


def test_success_head_shape():
    m = SuccessHead()
    G = torch.randn(4, 6, 32)
    logit = m(G)
    assert logit.shape == (4,)


def test_one_training_step_reduces_loss():
    torch.manual_seed(0)
    m = SlotDynamics()
    G = torch.randn(16, 6, 32)
    a = torch.randn(16, 9)
    target = torch.randn(16, 6, 32)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    losses = []
    for _ in range(50):
        out = m(G, a)
        loss = torch.nn.functional.mse_loss(out, target)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.5, f"loss did not drop: {losses[0]} → {losses[-1]}"


def test_round_trip_save_load(tmp_path):
    m = SlotDynamics()
    p = tmp_path / "sd.pt"
    save(m, str(p), {"n_factors": 6, "d_slot": 32, "a_dim": 9, "hidden": 256})
    m2 = load_slot_dynamics(str(p))
    G = torch.randn(2, 6, 32); a = torch.randn(2, 9)
    assert torch.allclose(m(G, a), m2(G, a))
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_world_model.py -v
```

All four tests must pass.

- [ ] **Step 4: Commit**

```bash
git add babysteps/stage5/world_model.py tests/test_world_model.py
git commit -m "feat(stage5 p3): SlotDynamics + SuccessHead modules with sim-free tests

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `babysteps/stage5/rollout_capture.py` — frame-capturing rollout helper

This module wraps `env_runner.run` so we get the multi-frame attempt trajectory. It must remain importable on the login node; the GPU-only ManiSkill calls are gated behind `try/except` (same pattern as `stage5/vlm_attribute.py`'s `_HAS_TORCH` gating).

**Files:**
- Create: `babysteps/stage5/rollout_capture.py`
- Create: `tests/test_rollout_capture.py`

- [ ] **Step 1: Define the rollout-capture API**

```python
# babysteps/stage5/rollout_capture.py
"""Stage-5 P3 rollout capture: run an episode and save demo+attempt frames + skill `a`.

Sim-free at import time. Functions that need ManiSkill raise a clear
RuntimeError on the login node (no Vulkan).
"""

from dataclasses import dataclass
import numpy as np
from typing import Optional

@dataclass
class RolloutRecord:
    seed: int
    on_policy: bool           # False if a was ε-perturbed
    demo_frames: np.ndarray   # (T_d, H, W, 3) uint8
    attempt_frames: np.ndarray  # (T_a, H, W, 3) uint8
    a_vec: np.ndarray         # (a_dim,) float32
    success: bool
    failure_predicate: Optional[str]


def pushcube_action_vec(*, approach_side_id: int, push_distance: float,
                        push_height: float, task_id_onehot: np.ndarray) -> np.ndarray:
    """Pack PushCube skill params into the canonical a_vec shape (9,).

    task_id_onehot: (3,) — [1,0,0] for PushCube.
    """
    side_onehot = np.zeros(4, dtype=np.float32)
    side_onehot[approach_side_id] = 1.0
    return np.concatenate([
        task_id_onehot.astype(np.float32),
        side_onehot,
        np.array([push_distance, push_height], dtype=np.float32),
    ])  # shape (9,)


def perturb_pushcube_action(*, approach_side_id: int, push_distance: float,
                            push_height: float, rng: np.random.Generator,
                            eps: float = 0.30):
    """Apply ε-noise to PushCube skill params. Returns dict of perturbed kwargs.

    Approach side is NOT perturbed (categorical). Distance and height get
    ±eps relative noise.
    """
    return dict(
        approach_side_id=approach_side_id,
        push_distance=push_distance * (1.0 + rng.uniform(-eps, eps)),
        push_height=push_height * (1.0 + rng.uniform(-eps, eps)),
    )


def collect_one(*, task: str, seed: int, on_policy: bool,
                eps: float = 0.30) -> RolloutRecord:
    """Run one episode in ManiSkill, capturing demo + attempt frames.

    Raises RuntimeError if ManiSkill is unavailable (login node).
    """
    # The real implementation:
    # 1. Build the env with frame capture enabled (use render-script pattern;
    #    see memory: 'PushCube render canonical sibling').
    # 2. Run demo phase, save frames.
    # 3. Compute skill params from demo (existing pipeline).
    # 4. If not on_policy, perturb skill params via perturb_pushcube_action.
    # 5. Run attempt phase with the (perhaps perturbed) skill params, save frames.
    # 6. Return the record.
    raise NotImplementedError(
        "GPU-only — implemented in scripts/stage5_p3_collect_rollouts.py "
        "Move helpers here when the script crystallizes."
    )
```

- [ ] **Step 2: Write sim-free tests for the helpers that don't need ManiSkill**

```python
# tests/test_rollout_capture.py
import numpy as np
from babysteps.stage5.rollout_capture import (
    pushcube_action_vec, perturb_pushcube_action, RolloutRecord,
)


def test_pushcube_action_vec_shape_and_dtype():
    a = pushcube_action_vec(
        approach_side_id=2, push_distance=0.12, push_height=0.04,
        task_id_onehot=np.array([1, 0, 0]),
    )
    assert a.shape == (9,)
    assert a.dtype == np.float32
    # task_id one-hot
    assert tuple(a[:3]) == (1.0, 0.0, 0.0)
    # side one-hot, side_id=2
    assert tuple(a[3:7]) == (0.0, 0.0, 1.0, 0.0)
    assert np.isclose(a[7], 0.12) and np.isclose(a[8], 0.04)


def test_perturb_within_eps():
    rng = np.random.default_rng(0)
    out = perturb_pushcube_action(
        approach_side_id=1, push_distance=0.10, push_height=0.05,
        rng=rng, eps=0.30,
    )
    assert out["approach_side_id"] == 1                # not perturbed
    assert 0.07 <= out["push_distance"] <= 0.13        # ±30% of 0.10
    assert 0.035 <= out["push_height"] <= 0.065        # ±30% of 0.05


def test_record_is_picklable():
    import pickle
    r = RolloutRecord(
        seed=0, on_policy=True,
        demo_frames=np.zeros((2, 4, 4, 3), dtype=np.uint8),
        attempt_frames=np.zeros((3, 4, 4, 3), dtype=np.uint8),
        a_vec=np.zeros(9, dtype=np.float32),
        success=True, failure_predicate=None,
    )
    blob = pickle.dumps(r); r2 = pickle.loads(blob)
    assert r2.seed == 0 and r2.success is True
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_rollout_capture.py -v
```

- [ ] **Step 4: Commit**

```bash
git add babysteps/stage5/rollout_capture.py tests/test_rollout_capture.py
git commit -m "feat(stage5 p3): rollout-capture helpers (action packing + ε-perturbation)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: `scripts/stage5_p3_collect_rollouts.py` — GPU collection job

This script does the actual ManiSkill rendering. It is GPU-only and not imported by tests.

**Files:**
- Create: `scripts/stage5_p3_collect_rollouts.py`
- Create: `slurm/stage5_p3_collect_rollouts.sbatch`

- [ ] **Step 1: Read the canonical PushCube render script first**

Per memory `feedback_pushcube_render_canonical_sibling.md`: copy `gym.make` kwargs from `babysteps/render/render_stage0_maniskill.py`, NOT `render_baseline_contrast.py` (which is missing `robot_uids=panda_wristcam` + `sensor_configs`). Record the exact kwargs here in the plan before coding.

- [ ] **Step 2: Implement the collection driver**

Sketch:
```python
# scripts/stage5_p3_collect_rollouts.py
"""Stage-5 P3 rollout collection.

For each (task, seed): render demo frames → run skill primitive → render
attempt frames → extract G_demo and G_post via P1 IntentHead → save record.

A fraction of episodes (default 30%) use ε-perturbed skill params.
"""

import argparse, json, os, numpy as np, torch
from pathlib import Path
from babysteps.stage4.intent_head import IntentHead
from babysteps.stage4.vision_features import extract_vision_features
from babysteps.stage5.rollout_capture import (
    pushcube_action_vec, perturb_pushcube_action,
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--n-seeds", type=int, default=300)
    ap.add_argument("--start-seed", type=int, default=1000)
    ap.add_argument("--perturb-frac", type=float, default=0.30)
    ap.add_argument("--out-dir", type=Path,
                    default=Path("datasets/stage5/p3_rollouts"))
    ap.add_argument("--intent-head", type=Path,
                    default=Path("models/stage5/p1_vision/PushCube-v1/intent_head.pt"))
    args = ap.parse_args()

    # 1. Build env with frame capture (see render_stage0_maniskill.py kwargs).
    # 2. Load IntentHead from args.intent_head.
    # 3. For each seed:
    #    a. Render demo phase → demo_frames.
    #    b. Compute on-policy skill params via existing pipeline.
    #    c. If rng.random() < perturb_frac: perturb skill params.
    #    d. Execute attempt phase → attempt_frames + success/failure_predicate.
    #    e. G_demo = IntentHead(extract_vision_features(demo_frames))
    #    f. G_post = IntentHead(extract_vision_features(attempt_frames[-8:]))
    #    g. a_vec = pushcube_action_vec(...)
    #    h. Save to out-dir/<task>/seed_NNNN/
    # 4. Append row to episodes.jsonl.

if __name__ == "__main__":
    main()
```

Cap attempt-frame capture at ≤ 32 frames (memory).

- [ ] **Step 3: Smoke-test on a single seed locally before launching the full job**

```bash
python scripts/stage5_p3_collect_rollouts.py \
    --task PushCube-v1 --n-seeds 1 --start-seed 1000 \
    --perturb-frac 0.0
```

Check:
- `datasets/stage5/p3_rollouts/PushCube-v1/seed_1000/G_demo.npy` has shape `(6, 32)`.
- `attempt_frames.npz` has at least 8 frames.
- `episodes.jsonl` has one row.

- [ ] **Step 4: Write the sbatch**

```bash
# slurm/stage5_p3_collect_rollouts.sbatch
#SBATCH --job-name=p3_collect
#SBATCH --partition=a100-40gb
#SBATCH --qos=standby
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm/logs/%x-%j.out

# Per project memory: --qos=standby on a100-40gb for ≤1h jobs.

set -euo pipefail
source activate babysteps   # or whatever the project env is
cd "$SLURM_SUBMIT_DIR"

python scripts/stage5_p3_collect_rollouts.py \
    --task PushCube-v1 \
    --n-seeds 300 \
    --start-seed 1000 \
    --perturb-frac 0.30
```

- [ ] **Step 5: Launch + verify**

```bash
sbatch slurm/stage5_p3_collect_rollouts.sbatch
# After it finishes:
ls datasets/stage5/p3_rollouts/PushCube-v1/ | wc -l           # ~301 (300 seed dirs + episodes.jsonl)
wc -l datasets/stage5/p3_rollouts/PushCube-v1/episodes.jsonl  # 300
```

Sanity: success rate should be ≥ 60% on PushCube (skill primitive is robust). If far below, the perturbation ε is too large — drop to 0.15.

- [ ] **Step 6: Commit**

```bash
git add scripts/stage5_p3_collect_rollouts.py slurm/stage5_p3_collect_rollouts.sbatch
git commit -m "feat(stage5 p3): rollout collection with G_demo / G_post extraction

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
# NOTE: data is large; do NOT commit datasets/stage5/p3_rollouts/* unless
# the per-task .npz files are small. Add to .gitignore if needed.
```

---

## Task 5: `scripts/stage5_p3_train_world_model.py` — CPU training

**Files:**
- Create: `scripts/stage5_p3_train_world_model.py`
- Create: `slurm/stage5_p3_train_world_model.sbatch` (small CPU/GPU job, optional)

- [ ] **Step 1: Training script**

```python
# scripts/stage5_p3_train_world_model.py
"""Train SlotDynamics + SuccessHead on collected rollouts.

Held-out split: 80/20 on seed (deterministic). MSE for dynamics,
BCE-with-logits for success head. Reports per-slot held-out MSE and
success-head AUC.
"""

import argparse, json, numpy as np, torch
from pathlib import Path
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from babysteps.stage5.world_model import (
    SlotDynamics, SuccessHead, save,
)

class RolloutDataset(Dataset):
    """Loads (G_demo, a, G_post, success) tuples from p3_rollouts/<task>/."""
    # ... implementation walks seed_NNNN/ dirs.

def train(args):
    # 1. Build train/test datasets (80/20 by seed_id % 5).
    # 2. Train SlotDynamics for ~200 epochs, MSE loss.
    # 3. Train SuccessHead for ~100 epochs, BCE with logits.
    #    NOTE: SuccessHead reads G_post from the GT (not the dynamics model)
    #    during training. At eval time it reads predicted G_post.
    # 4. Save both checkpoints to models/stage5/p3_world_model/<task>/.
    # 5. Write reports/stage5/p3_world_model/<task>/dynamics_eval.json.
    ...

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--rollout-dir", type=Path,
                    default=Path("datasets/stage5/p3_rollouts"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("models/stage5/p3_world_model"))
    args = ap.parse_args()
    train(args)
```

- [ ] **Step 2: Run training**

```bash
python scripts/stage5_p3_train_world_model.py --task PushCube-v1
```

Acceptance:
- Held-out per-slot MSE < `0.5 * Var(G_post_train[:,j,:])` (the model beats the mean predictor on every slot).
- SuccessHead held-out AUC ≥ 0.75. (If lower, the rollouts are mostly all-success or all-fail; need more variance.)

If acceptance fails, **stop and investigate** before going to Task 6 — the gate downstream is meaningless if the model doesn't fit.

- [ ] **Step 3: Commit**

```bash
git add scripts/stage5_p3_train_world_model.py slurm/stage5_p3_train_world_model.sbatch
git commit -m "feat(stage5 p3): train SlotDynamics + SuccessHead on rollouts

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: `scripts/stage5_p3_noise_floor.py` — same-seed replay distribution

**Files:**
- Create: `scripts/stage5_p3_noise_floor.py`
- Create: `slurm/stage5_p3_noise_floor.sbatch`

- [ ] **Step 1: Implement the replay job**

For each held-out seed (the 20% test split from Task 5), replay the same `(G_demo, a)` with `k=10` different sim seeds (e.g. `replay_seed = orig_seed * 1000 + k`). Extract `G_post^{(k)}`. Save per-slot drift `δ_j^{(k)} = ||G_post^{(k)} - mean_k G_post^{(k)}||₂`.

```python
# scripts/stage5_p3_noise_floor.py
"""Estimate per-slot G_post noise floor via same-seed replays.

For each held-out episode, replay (G_demo, a) under k=10 different
sim_seeds. Record per-slot L2 drift from the replay mean. The 95th
percentile of the pooled drift distribution per slot is the gate cutoff.
"""

# Heavy: ~3s × 50 eps × 10 replays = 25 min per task on A100.
```

- [ ] **Step 2: Write the noise distribution to disk**

```text
datasets/stage5/p3_noise_floor/PushCube-v1/
    noise_distribution.npz       # keys: delta_j (50*10, 6), p95_j (6,)
    per_seed_replays.jsonl       # for traceability
```

- [ ] **Step 3: Sanity check the distribution**

Histogram per slot `j`:
- If all `δ_j ≈ 0` → simulator is deterministic; we need to perturb the sim seed in a way that matters (e.g. randomize object spawn pose within the task's spawn distribution). Update the spec §3.5 to clarify what "same-seed replay" means and re-run.
- If `δ_j` is bimodal → there are likely two stable execution outcomes per seed; document and proceed.

- [ ] **Step 4: Commit**

```bash
git add scripts/stage5_p3_noise_floor.py slurm/stage5_p3_noise_floor.sbatch
git commit -m "feat(stage5 p3): same-seed replay noise floor for selectivity gate

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: `scripts/stage5_p3_g3_eval.py` — selectivity + effectiveness gate

This is the headline gate. CPU-only (uses the trained dynamics model and the saved noise floor).

**Files:**
- Create: `scripts/stage5_p3_g3_eval.py`

- [ ] **Step 1: Implement the counterfactual eval**

```python
# scripts/stage5_p3_g3_eval.py
"""Stage-5 P3 G3 forward-prediction gate.

For each held-out episode e and each revisable slot i:
  Baseline:    G_post_pred       = f_φ(G_demo,  a)
  Counter:     G_demo' = ReviseHead(G_demo, slot=i)
                a'     = skill_compiler(G_demo')
                G_post_pred'      = f_φ(G_demo', a')
  Drift:       Δ_j(e) = ||G_post_pred[j] - G_post_pred'[j]||₂   ∀ j ∈ 0..5

Gate per (task, slot i):
  Selectivity:   #{e : Δ_j(e) ≤ p95_j  ∀ j ≠ i} / N ≥ 0.90
  Effectiveness: #{e : Δ_i(e) >  p95_i}          / N ≥ 0.80
"""
```

`ReviseHead` is loaded from `models/stage5/p1_vision/PushCube-v1/revise_head.pt`. The skill compiler signature comes from Task 1's investigation.

- [ ] **Step 2: Write per-task report**

```markdown
# Stage-5 P3 G3 selectivity — PushCube-v1

| slot i | selectivity pass-rate | effectiveness pass-rate | gate |
| --- | --- | --- | --- |
| approach_direction | 0.96 | 0.88 | PASS |
| ... | ... | ... | ... |
```

And `g3_selectivity.json` with the full per-episode breakdown.

- [ ] **Step 3: Run eval**

```bash
python scripts/stage5_p3_g3_eval.py --task PushCube-v1
```

Acceptance per the spec:
- Selectivity ≥ 90% on every revisable slot.
- Effectiveness ≥ 80% on every revisable slot.

If a slot fails, **stop and diagnose** — do not loosen the gate without recording why in `reports/stage5/p3_world_model/PushCube-v1/report.md`. Per project memory `feedback_preservation_gate_ties`, ties at perfect can PASS, but loosening the threshold needs a documented reason.

- [ ] **Step 4: Commit**

```bash
git add scripts/stage5_p3_g3_eval.py
git commit -m "feat(stage5 p3): G3 forward-prediction selectivity + effectiveness gate

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Cross-condition summary + sanity baselines

The spec §4.3 lists three sanity baselines: identity, shuffled `a`, linear regression. These run on the eval script with three additional model variants and the same gate.

**Files:**
- Modify: `scripts/stage5_p3_g3_eval.py` (add `--variant {trained, identity, shuffled, linear}`)
- Create: `reports/stage5/p3_world_model/summary.md`

- [ ] **Step 1: Add the four variants**

| Variant | What `f_φ` is |
|---|---|
| `trained` | the MLP from Task 5 |
| `identity` | `G_post = G_demo` (ignores `a`) |
| `shuffled` | the MLP from Task 5 with `a` randomly permuted at eval |
| `linear` | linear regression `(G_demo, a) → G_post` fit on the same train split |

Expected:
- `identity` passes selectivity trivially, fails effectiveness.
- `shuffled` fails both.
- `linear` is the lower bar that `trained` must beat.

- [ ] **Step 2: Write `summary.md`**

```markdown
# Stage-5 P3 — Cross-condition summary (PushCube-v1)

| variant | sel pass | eff pass | gate |
| --- | --- | --- | --- |
| trained MLP   | 0.96 | 0.88 | PASS |
| identity      | 1.00 | 0.04 | FAIL (effectiveness) |
| shuffled `a`  | 0.42 | 0.30 | FAIL |
| linear        | 0.91 | 0.71 | partial — MLP beats it on effectiveness |
```

- [ ] **Step 3: Commit**

```bash
git add scripts/stage5_p3_g3_eval.py reports/stage5/p3_world_model/summary.md
git commit -m "data(stage5 p3): sanity baselines (identity/shuffled/linear) vs trained MLP

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Record results in goal.md, MEMORY, and slurm log

- [ ] **Step 1: Append P3 status to `goal.md` §"Stage 5"**

Add a "Stage-5 P3 — World model (PushCube-v1)" subsection mirroring how P1 and P2 results were recorded, with the gate table from Task 7.

- [ ] **Step 2: Append to `slurm/CLAUDE.md`**

```markdown
### Stage-5 P3 — World model gate (job <JOBID>, 2026-05-XX)

PushCube-v1 trained MLP:
- Selectivity: 0.96 (all slots)
- Effectiveness: 0.88 (all slots)
- Gate: PASS
```

- [ ] **Step 3: Update auto-memory `project_stage_status_roadmap`**

Replace the P3 line with the concrete outcome (PASS / partial / blocked).

- [ ] **Step 4: Commit**

```bash
git add goal.md slurm/CLAUDE.md
git commit -m "docs(stage5 p3): record world-model G3 gate result for PushCube

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 10 (optional, deferred): revision-ranking eval

Spec §3.6 stretch goal: re-rank ambiguous VLM attributions using `r_φ`. Implementable but not part of the primary G3-WM claim.

Defer until Task 9 passes. When done, create a small follow-on script `scripts/stage5_p3_rerank_eval.py` that loops over P2's misattributed episodes and reports recovery rate.

---

## Out-of-scope follow-ups (write separate plans)

1. **Extend P1 vision IntentHead to PickCube + StackCube.** Currently `models/stage5/p1_vision/` only has PushCube. Without this, P3 cannot run on the other tasks. Likely needs: render PickCube vision data (does not exist yet), train PickCube IntentHead, train StackCube IntentHead from existing 40-episode varied_intent cut.
2. **Per-step latent dynamics (P3.5).** If slot-space `G_post ≈ G_demo` on most episodes (Task 6 noise distribution is degenerate), pivot to per-step `(z_t, a_t) → z_{t+1}` and re-run.
3. **5-ensemble uncertainty.** If a single MLP's predictions are too noisy to discriminate edited vs non-edited slots, train an ensemble and use predicted variance to weight the per-episode threshold.

---

## Self-review (run after writing)

Before declaring this plan done:

- [ ] Each task has a single coherent commit message.
- [ ] All sim-free tests (Tasks 2, 3) pass on the login node.
- [ ] GPU jobs (Tasks 4, 6) use `--qos=standby` on `a100-40gb`.
- [ ] No GPU import is reachable at sim-free test time (use the `vlm_attribute.py` pattern of try/except + `_HAS_X` flags).
- [ ] The gate definitions in Task 7 match the spec §3.5.
- [ ] All output paths under `datasets/stage5/p3_*` and `models/stage5/p3_*` are documented in the file-structure block.
- [ ] `.gitignore` (if needed) excludes the large rollout `.npz` files.
- [ ] The PushCube-only scope is reflected everywhere (no accidental references to PickCube/StackCube in code paths).
