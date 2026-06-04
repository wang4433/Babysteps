# Stage-5 — DINOv2 → DINOv3 encoder-swap ablation — SUMMARY

**Date:** 2026-06-04 · **Jobs:** 10955091 (ViT-L/16), 10955092 (ViT-B/16),
A100-40GB, standby. **Question:** the Stage-5 limitation is the frozen visual
encoder's ability to recover observation-grounded latent factors — so does a
*stronger image encoder* (DINOv3, Meta's dense-feature successor) cross the SAME
gate (confound-controlled IntentHead-CV ≥ 0.90) on the cells where frozen DINOv2
ViT-B/14 fell short?

**Verdict: NO for the genuinely-blocked factors — but the swap is not a
weak-encoder artifact.** DINOv3 (even ViT-L/16 at 512) does not recover
`goal_state` on real clips or charger orientation. It *does* cleanly improve a
trivial spatial-localization factor (charger position), crossing the gate at
the deployed resolution where DINOv2 failed — which proves the negatives are
real factor-observability limits, not "the encoder wasn't strong enough."

## Setup / fairness

- **Weights:** timm's UNGATED re-host of the official lvd1689m DINOv3 weights
  (`vit_{base,large}_patch16_dinov3.lvd1689m`) — same weights as the gated
  `facebook/dinov3-*` repos, no token. Loaded frozen, `num_classes=0`.
- **Encoder integration (verified):** post-LayerNorm patch tokens sliced at
  `num_prefix_tokens=5` (1 CLS + 4 register), `spatial_mean` over patch+time —
  the exact analog of DINOv2's `x_norm_patchtokens`. Patch size 16 ⇒ resolutions
  224 / 512.
- **Fairness:** identical labels, IntentHead protocol, and gate; identical
  ImageNet normalization and full-frame square FOV. Each encoder uses its OWN
  native resize kernel (DINOv2 bilinear; DINOv3 bicubic+antialias, its timm data
  config) so neither is handicapped by a foreign resize. "Identical probe,
  native preprocessing per encoder" — not "identical pixels".
- **Positive control:** `goal_state config @224` (clean canonical configs). A
  correct integration must reproduce DINOv2's 0.992. It does (L/16 **0.992**,
  B/16 **0.967**) — the timm path is wired correctly end-to-end.

## Results (IntentHead-CV; gate 0.90)

### StackCube `goal_state`

| pooling / mode | DINOv2 @224 | DINOv3-B @224 | DINOv3-L @224 | DINOv3-B @512 | DINOv3-L @512 |
|---|---|---|---|---|---|
| config (control) | **0.992** | **0.967** | **0.992** | — | — |
| clip deployed `spatial_mean` | 0.650 | 0.675 | 0.692 | 0.742 | 0.717 |
| clip best pooling (`last5`/`final`) | 0.817 | 0.808 | 0.800 | 0.817 | 0.817 |

The deployed clip metric nudges up with DINOv3+hi-res (0.65 → 0.72) but the
**ceiling is identical at ~0.82 across every encoder and resolution**, well
below the gate. Root cause is encoder-invariant: the StackCube demo CLIP
deliberately hides the vertical stack-vs-place motion (Sub-project C
goal-ambiguity = 2D-trajectory info loss). Information that is not in the pixels
cannot be recovered by a stronger encoder. **FAIL, confirmed across encoders.**

### PlugCharger (fix-receptacle — background frozen, so any signal is the charger itself)

| factor (→ intent) | DINOv2 @224/512 | DINOv3-B @224/512 | DINOv3-L @224/512 |
|---|---|---|---|
| `charger_yaw` → object_motion (orientation) | 0.833 / 0.858 | 0.817 / 0.858 | 0.783 / 0.850 |
| `charger_xy` → approach_direction (position) | 0.842 / 0.942* | **0.925 / 0.925** | **0.900 / 0.950** |

\* DINOv2 only reached 0.942 on `charger_xy` with a non-default fixed camera + res 518.

- **`charger_yaw` (orientation): FAIL on every encoder (~0.78–0.86).** The
  charger's ±60° in-plane orientation is not reliably groundable: the asymmetric
  base (~40×30 mm, ~1–3 patches) is sub-resolution for reading orientation, and
  the pegs are sub-patch. A stronger/bigger/higher-res encoder does not fix it.
- **`charger_xy` (position): the DINOv3 WIN.** DINOv3 clears the gate at the
  DEPLOYED 224 (0.90–0.925) where DINOv2 needed a bespoke fixed-camera + 518 to
  barely pass. DINOv3 is genuinely a stronger spatial-localization encoder. But
  this factor is the trivial "where is the object on the table" signal, not a
  manipulation-discriminating intent.

### Negative controls (deployed mode — background rotates with the receptacle)

| factor | DINOv2 | DINOv3-B | DINOv3-L |
|---|---|---|---|
| `receptacle_yaw` (neg control) | 0.958 | 0.925–0.942 | 0.908–0.942 |
| `charger_yaw` deployed (background confound) | 0.925 | 0.908–0.925 | 0.900–0.942 |

The controls behave identically across encoders: the receptacle-mounted camera's
background-rotation confound still fires (PASS), and `charger_yaw` deployed PASSes
purely on that confound — vs fix-receptacle FAIL, exactly as for DINOv2. The
encoder swap does not disturb the confound structure.

## Interpretation (the paper claim, sharpened)

The encoder-swap turns a soft limitation statement into a controlled one:

> The limitation is the **factor's pixel-observability under the deployed
> interface**, not the encoder. A stronger frozen image encoder (DINOv3 ViT-L/16,
> even at 512) cleanly improves a localization factor that DINOv2 could not
> ground at deployed resolution (charger position, +6–11pp, crosses the gate),
> yet does **not** recover the genuinely-blocked factors: `goal_state` on real
> clips (the demo hides the discriminating motion → information loss, ceiling
> ~0.82 on every encoder) and charger orientation (sub-resolution asymmetric
> object, ~0.85). The fact that DINOv3 helps exactly where the information IS in
> the pixels, and not where it isn't, certifies the negatives are factor-level,
> not weak-encoder artifacts.

PushCube remains the single clean end-to-end latent task. The groundability
ladder now reads: PushCube ✅ · PickCube ❌ (invisible factor, encoder-agnostic) ·
StackCube `object_motion` ❌ (rep-blocked; deferred V-JEPA cell) · StackCube
`goal_state` ❌ (clip info-loss, encoder-invariant ~0.82) · PlugCharger
`charger_yaw` ❌ (sub-resolution orientation, encoder-invariant ~0.85) ·
PlugCharger `charger_xy` ✅-but-trivial (DINOv3 lifts it over the gate).

## What's next (not run here)

- **`object_motion` (temporal)** is the cell the V-JEPA round targets — a
  video/world-model encoder, not a stronger image encoder. DINOv3 was not run on
  it (it is a between-frame delta, not a single-frame factor).
- A bilinear-controlled DINOv3 re-run is not needed: every blocked cell FAILs at
  DINOv3's *native* (more favorable) bicubic+antialias recipe, so the kernel
  choice cannot be masking a pass.

Artifacts: `reports/stage5/goal_state_probe_{vitl16,vitb16}_{config,clip224,clip512}/`
and `reports/stage5/plugcharger_probe_{vitl16,vitb16}_{fixrecep,deployed}/`.
Reproduce: `sbatch slurm/stage5_dinov3_encoder_swap.sbatch dinov3_vitl16`
(and `dinov3_vitb16`).
