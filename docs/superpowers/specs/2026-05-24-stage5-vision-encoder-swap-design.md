# Stage 5 P1 — Vision Encoder Swap Design Spec

> **Status:** draft 2026-05-24. Implements Priority 1 of `goal.md` §"Stage 5".
> This is the critical first step for the ICLR submission track.

## 1. Problem

Stage 4's IntentHead consumes a 20-dim handcrafted feature vector
(9 trajectory summary stats + 6-way contact_region one-hot + 4-way
final_state one-hot). The continuous slot-intent bottleneck adds nothing
the discrete schema doesn't already provide. For an ICLR "latent intent"
claim, the representation must be grounded in raw visual observations.

## 2. Goal

Replace the handcrafted 20-dim `Z` with features extracted by a **frozen
pretrained vision encoder** applied to the third-person demo RGB frames
rendered by ManiSkill. The IntentHead architecture stays the same; only
`z_dim` changes.

**Gate:** G1 probe recoverability of all six discrete factors from the
vision-grounded `G_t` at ≥ 90% held-out accuracy.

## 3. Design

### 3.1 Encoder choice

Primary: **DINOv2 ViT-B/14** (frozen, torchvision or facebookresearch/dinov2).
- Output dim: 768 (CLS token) per frame.
- Rationale: DINOv2 is the standard frozen visual backbone for
  manipulation representation learning (R3M, SPA, Voltron comparisons
  all benchmark against it). It requires no task-specific fine-tuning
  and provides dense spatial features if we later need them.

Fallback encoders (if DINOv2 G1 fails):
- **R3M** (ResNet-50, 2048-dim) — pre-trained on Ego4D manipulation video.
- **SPA** — spatial-action-aware, designed for manipulation.
- **CLIP ViT-B/32** (512-dim) — language-aligned, useful if P2 VLM
  attribution needs visual-language alignment.

### 3.2 Feature extraction pipeline

```python
# New module: babysteps/stage4/vision_features.py

def extract_demo_features(
    demo_frames: list[np.ndarray],  # T × (H, W, 3) uint8 RGB
    encoder: str = "dinov2_vitb14",
    pool: str = "cls_mean",          # mean-pool CLS over time
    device: str = "cuda",
) -> np.ndarray:
    """(T, H, W, 3) → (d_encoder,) float32 vector."""
```

**Pooling strategies** (ablation):
1. `cls_mean` — mean-pool CLS tokens over T frames. Simplest.
2. `cls_first_last` — concat CLS of first and last frame (2 × 768 = 1536).
   Captures state change without full-sequence cost.
3. `spatial_mean` — mean-pool patch tokens spatially, then over time.
   Retains more spatial structure.

Start with `cls_mean`; ablate in the G1 probe experiment.

### 3.3 Frame source

Demo frames are already rendered by ManiSkill via `env.render()` and
captured by the render modules (`babysteps/render/common.py::render_frame`).
For the vision encoder, we need:

1. **Resolution:** Resize to 224×224 (DINOv2 native) or 518×518 (DINOv2
   with registers). ManiSkill renders at 512×512 by default; a center crop
   + resize is sufficient.
2. **Normalization:** ImageNet mean/std (DINOv2 expects this).
3. **Frame sampling:** Use all frames from the demo phase (typically 13–50
   frames at 20fps). For longer episodes, uniform-sample to max 16 frames.

### 3.4 IntentHead changes

```python
# Existing (Stage 4):
IntentHead(z_dim=20, n_factors=6, d_slot=16, hidden=64)

# Stage 5 P1:
IntentHead(z_dim=768, n_factors=6, d_slot=32, hidden=256)
```

Changes:
- `z_dim`: 20 → 768 (DINOv2 CLS) or 1536 (first+last concat).
- `d_slot`: 16 → 32 (richer slot to accommodate visual information).
- `hidden`: 64 → 256 (proportional scaling).

The IntentHead architecture (Linear → GELU → Linear → reshape to F × d_slot)
stays the same. No attention, no transformer — keep it simple for v1.

### 3.5 Training protocol

Same as Stage 4 M2a: per-slot CE supervision against discrete factor labels,
nested-CV G1 probe on held-out folds.

**Data requirement:** The varied-intent collection (datasets/stage4/varied_intent/)
already has PushCube (20 episodes) and StackCube (40 episodes). We need to:

1. **Re-render with frame capture:** Re-run the varied-intent episodes
   through ManiSkill, saving demo RGB frames (not just trajectory stats).
   This requires a GPU job.
2. **Add PickCube varied cut:** Currently missing from the varied collection.
   Needed for 3-task (or 5-task) evaluation.
3. **Scale to 50+ seeds per task** for statistical power.

### 3.6 Episode data format extension

Add a `demo_frames_path` field to the episode JSON:

```json
{
  "demo": {
    "camera": "third_person",
    "demo_frames_path": "data/.../demo_frames.npz",
    ...
  }
}
```

The `.npz` stores `frames: (T, H, W, 3) uint8` and `timestamps: (T,) float`.
Extracted DINOv2 features are cached alongside as `demo_z_dinov2.npy`.

## 4. Evaluation

### 4.1 G1 probe (primary gate)

Run the same nested-CV protocol as Stage 4 M1/M2a:
1. Per outer fold: fresh IntentHead trained on train-fold (Z_vision, y_factor).
2. Produce G on test fold.
3. Frozen LogisticRegression on G → y, score on test.

**Pass:** all non-trivially-constant (task, factor) cells reach ≥ 90%.

### 4.2 Baselines (within the probe experiment)

| Row | Input Z | IntentHead | Expected |
| --- | --- | --- | --- |
| Handcrafted (Stage 4) | 20-dim trajectory + one-hot | Same arch, small | Known: 17/18 trivial, 1 at 0.75 |
| DINOv2 CLS mean | 768-dim | Scaled arch | Target: ≥ 90% on non-trivial cells |
| DINOv2 first+last | 1536-dim | Scaled arch | Ablation |
| R3M | 2048-dim | Scaled arch | Ablation |
| Raw pixels (flatten) | ~786K-dim | N/A (sanity) | Should fail |

### 4.3 Downstream: ReviseHead + sim rollout

After G1 passes, retrain ReviseHead on vision-grounded G (same L2-to-
centroid protocol as M2a A2). Then run sim rollouts (`stage4_m2a_run_eval.py`)
to verify G4 Δpp ≥ 10 and G5 within 5pp of oracle.

## 5. Implementation plan

| Step | What | Depends on | GPU? |
| --- | --- | --- | --- |
| S1 | `vision_features.py` — DINOv2 extraction + caching | None | Yes (inference) |
| S2 | Re-render varied-intent episodes with frame capture | S1 | Yes (ManiSkill) |
| S3 | IntentHead z_dim=768 training + nested-CV G1 probe | S2 | No (CPU torch) |
| S4 | G1 report: pass/fail per (task, factor) cell | S3 | No |
| S5 | ReviseHead retrain on vision-grounded G | S4 (if G1 passes) | No |
| S6 | Sim rollout eval (G4/G5) | S5 | Yes (ManiSkill) |

**Estimated timeline:** S1–S4 in ~1 week (the GPU rendering is the
bottleneck). S5–S6 in ~3 days.

## 6. Risks

1. **DINOv2 features may not separate all factors.** Mitigation:
   `contact_region` and `approach_direction` are spatial — DINOv2's
   spatial patch features (not just CLS) may be needed. Fall back to
   spatial pooling or R3M.
2. **Demo frames may be too similar across seeds** (same camera, same
   table, small cube). Mitigation: the varied-intent collection
   deliberately varies the push direction / grasp face / goal, so the
   visual difference should be present. If not, this is a genuine
   negative result about what frozen vision features can capture.
3. **Compute cost.** DINOv2 ViT-B/14 inference on 50 episodes × 16
   frames = 800 forward passes. ~2 minutes on an A100. Not a bottleneck.

## 7. What success looks like

- G1 probe ≥ 90% on vision-grounded G for all non-trivial cells.
- The probe result is NOT achievable by a constant predictor or by
  shuffled features (majority-class and shuffled baselines both below 90%).
- ReviseHead retrained on vision G achieves G4 ≥ +10pp, G5 within 5pp.
- A one-paragraph result: "Frozen DINOv2 features on demo video frames,
  passed through a 2-layer IntentHead, produce slot intents from which
  all six Stage-0 discrete factors are linearly recoverable at ≥ 90%.
  The slot-local ReviseHead, retrained on these vision-grounded slots,
  recovers manipulation failures at parity with the oracle revision."
