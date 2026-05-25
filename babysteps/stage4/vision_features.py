"""Stage-5 P1 — frozen vision-encoder feature extraction.

Wraps a frozen pretrained vision encoder (default: DINOv2 ViT-B/14)
applied to the third-person demo RGB frames produced by
`babysteps.render.common.render_frame(env)`. Returns a single
(d_encoder,) float32 vector per demo, suitable as drop-in `Z` for
the existing Stage-4 IntentHead.

Stage-4 firewall (carried over): this module consumes only RGB frame
arrays — DemoEvidence-shaped input — and never reads
execution.initial_intent, failure_packet, revision, retry, or any
privileged SceneState field.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F

# ImageNet normalization constants used by DINOv2.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _preprocess_frames(
    frames: list[np.ndarray],
    *,
    resolution: int = 224,
) -> torch.Tensor:
    """List[(H, W, 3) uint8] -> (T, 3, R, R) float32 ImageNet-normalized."""
    # Stack to (T, H, W, 3) uint8, permute to (T, 3, H, W), float in [0, 1].
    arr = np.stack(frames, axis=0)
    if arr.dtype != np.uint8:
        raise ValueError(f"frames must be uint8, got {arr.dtype}")
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"frames must have shape (T, H, W, 3), got {arr.shape}")
    t = torch.from_numpy(arr).permute(0, 3, 1, 2).float().div_(255.0)
    # Resize to (R, R) via bilinear; DINOv2 ViT-B/14 needs the spatial dims
    # divisible by the patch size 14 — 224 = 16*14 is the standard.
    t = F.interpolate(t, size=(resolution, resolution),
                      mode="bilinear", align_corners=False)
    # ImageNet normalize per channel.
    mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
    return (t - mean) / std


def _pool_cls(cls_tokens: torch.Tensor, *, pool: str = "cls_mean") -> torch.Tensor:
    """(T, d) -> (d,) for cls_mean, or (2*d,) for cls_first_last.

    Strategies (per the design spec § 3.2):
      - cls_mean: mean over T (default; simplest baseline).
      - cls_first_last: concat first and last frame CLS — preserves the
        start-vs-end delta that mean-pooling discards. Targets factors
        like object_motion that are naturally between-frame deltas
        (spec § 6 ablation order). For T == 1, duplicates the single CLS
        so the output is still (2*d,).
      spatial_mean is dispatched separately in extract_vision_features
      (it uses model.forward_features patch tokens rather than CLS).
    """
    if pool == "cls_mean":
        return cls_tokens.mean(dim=0)
    if pool == "cls_first_last":
        first = cls_tokens[0]
        last = cls_tokens[-1] if cls_tokens.shape[0] > 1 else cls_tokens[0]
        return torch.cat([first, last], dim=0)
    raise ValueError(f"unknown pool strategy: {pool!r}")


# Module-level cache so successive calls in the cache-features job don't
# reload DINOv2 weights for every seed.
_MODEL_CACHE: dict[tuple[str, str], torch.nn.Module] = {}


def _load_dinov2(encoder: str, device: str) -> torch.nn.Module:
    """Load and freeze a DINOv2 model via torch.hub. Cached per (encoder, device).

    Network/disk hit happens once per process; the model is moved to
    `device`, set to eval mode, and all parameters are frozen.
    """
    key = (encoder, device)
    if key not in _MODEL_CACHE:
        model = torch.hub.load("facebookresearch/dinov2", encoder)
        model.eval()
        model.to(device)
        for p in model.parameters():
            p.requires_grad_(False)
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


def extract_vision_features(
    demo_frames: list[np.ndarray],
    *,
    encoder: str = "dinov2_vitb14",
    pool: str = "cls_mean",
    device: str = "cuda",
    resolution: int = 224,
    _encoder: Optional[torch.nn.Module] = None,  # test/inject hook
) -> np.ndarray:
    """Frozen-encoder feature extraction from demo RGB frames.

    Args:
      demo_frames: list of (H, W, 3) uint8 RGB arrays — exactly what
        babysteps.render.common.render_frame(env) produces.
      encoder: torch.hub model id; default DINOv2 ViT-B/14.
      pool: time-pooling strategy (see `_pool_cls`).
      device: torch device for the encoder.
      resolution: square resize before encoding (DINOv2 wants 224).
      _encoder: optional injected encoder for unit tests (bypasses torch.hub).

    Returns:
      (d_encoder,) float32 numpy vector. d_encoder = 768 for ViT-B/14.

    The Stage-4 firewall applies: this function reads only the frame
    arrays — no labels, no privileged scene state.
    """
    if len(demo_frames) == 0:
        raise ValueError("extract_vision_features needs at least one frame")

    model = _encoder if _encoder is not None else _load_dinov2(encoder, device)
    x = _preprocess_frames(demo_frames, resolution=resolution).to(device)
    with torch.no_grad():
        if pool == "spatial_mean":
            # Patch-token path: DINOv2 forward_features returns a dict with
            # `x_norm_patchtokens` of shape (T, N_patches, d). Mean over both
            # the patch and time axes -> (d,). Stays 768-dim, which keeps the
            # G1 linear probe out of the d >> n overfitting regime that hurt
            # cls_first_last (1536-dim, n=40).
            features = model.forward_features(x)
            patches = features["x_norm_patchtokens"]  # (T, N, d)
            z = patches.mean(dim=(0, 1))
        else:
            cls = model(x)  # (T, d) — DINOv2's default forward returns CLS
            z = _pool_cls(cls, pool=pool)
    return z.detach().cpu().numpy().astype(np.float32)
