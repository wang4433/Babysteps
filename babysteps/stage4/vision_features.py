"""Stage-5 P1 — frozen vision-encoder feature extraction.

Wraps a frozen pretrained vision encoder (default: DINOv2 ViT-B/14)
applied to the third-person demo RGB frames produced by
`babysteps.render.common.render_frame(env)`. Returns a single
(d_encoder,) float32 vector per demo, suitable as drop-in `Z` for
the existing Stage-4 IntentHead.

Encoder-swap ablation (Stage-5 groundability map): the same interface
also serves DINOv3 ViT models via `timm` (``encoder="dinov3_vitl16"``
and friends, or a raw timm model name). We deliberately load DINOv3 from
timm's UNGATED re-host of the official lvd1689m weights (e.g.
``vit_large_patch16_dinov3.lvd1689m``) rather than the license-gated
``facebook/dinov3-*`` HF repos — same weights, no token.

Fairness of the swap: the PROBE (labels, IntentHead, gate) is identical
across encoders, and both use the same ImageNet normalization. Each
encoder gets its OWN recommended resize recipe so neither is handicapped
by the other's: DINOv2 → bilinear (unchanged from the committed runs);
DINOv3 → bicubic + antialias, the recipe in its timm data config. (A
non-native resize could push a frozen encoder off its training
distribution and turn a preprocessing artifact into a false FAIL — so
"identical probe, native preprocessing per encoder" is the controlled
comparison, not "identical pixels".) The other DINOv3 API deltas: patch
size 16 (resolutions must be divisible by 16, e.g. 224 or 512, not 518)
and `num_prefix_tokens` (=5: 1 CLS + 4 register tokens) which are sliced
off; the remaining post-LayerNorm patch tokens are the fair analog of
DINOv2's `x_norm_patchtokens` (verified post-norm). The DINOv2 code path
is left byte-for-byte unchanged so prior numbers reproduce exactly.

Stage-4 firewall (carried over): this module consumes only RGB frame
arrays — DemoEvidence-shaped input — and never reads
execution.initial_intent, failure_packet, revision, retry, or any
privileged SceneState field.
"""
from __future__ import annotations

import os
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
    patch_size: int = 14,
    interpolation: str = "bilinear",
    antialias: bool = False,
) -> torch.Tensor:
    """List[(H, W, 3) uint8] -> (T, 3, R, R) float32 ImageNet-normalized.

    `patch_size` is the encoder's patch grid (14 for DINOv2 ViT-*/14, 16
    for DINOv3 ViT-*/16). The resize target must be divisible by it, or the
    ViT patch embedding silently drops/duplicates a border row — so we reject
    a mismatch loudly rather than encode a subtly wrong grid.

    `interpolation` / `antialias` select the resize recipe. The defaults
    (`bilinear`, no antialias) are the committed DINOv2 setting and are left
    untouched so prior numbers reproduce. The DINOv3 path passes
    `bicubic` + `antialias=True` to match its timm data config (its native
    recipe), so the encoder is not handicapped by a foreign resize.
    """
    if resolution % patch_size != 0:
        raise ValueError(
            f"resolution {resolution} must be divisible by patch_size "
            f"{patch_size} (e.g. 224 or 512 for patch 16; 224 or 518 for "
            f"patch 14)"
        )
    # Stack to (T, H, W, 3) uint8, permute to (T, 3, H, W), float in [0, 1].
    arr = np.stack(frames, axis=0)
    if arr.dtype != np.uint8:
        raise ValueError(f"frames must be uint8, got {arr.dtype}")
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"frames must have shape (T, H, W, 3), got {arr.shape}")
    t = torch.from_numpy(arr).permute(0, 3, 1, 2).float().div_(255.0)
    # Resize to (R, R); the ViT needs the spatial dims divisible by the patch
    # size — 224 = 16*14 (DINOv2) = 14*16 (DINOv3); 512 = 32*16 (DINOv3 hi-res).
    # antialias only affects downsampling (a no-op when upscaling a small render).
    t = F.interpolate(t, size=(resolution, resolution),
                      mode=interpolation, align_corners=False,
                      antialias=antialias)
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

# Short aliases -> UNGATED timm model names for the DINOv3 encoder-swap
# ablation (timm re-hosts the official lvd1689m weights with no license gate;
# the gated facebook/dinov3-* HF repos hold the same weights but need a token).
# A raw timm name (e.g. "vit_large_patch16_dinov3.lvd1689m") is also accepted.
_DINOV3_ALIASES: dict[str, str] = {
    "dinov3_vits16": "vit_small_patch16_dinov3.lvd1689m",
    "dinov3_vits16plus": "vit_small_plus_patch16_dinov3.lvd1689m",
    "dinov3_vitb16": "vit_base_patch16_dinov3.lvd1689m",
    "dinov3_vitl16": "vit_large_patch16_dinov3.lvd1689m",
    "dinov3_vith16plus": "vit_huge_plus_patch16_dinov3.lvd1689m",
    "dinov3_vit7b16": "vit_7b_patch16_dinov3.lvd1689m",
}


def _is_dinov3(encoder: str) -> bool:
    """True if `encoder` names a DINOv3 model (alias or raw repo id)."""
    return "dinov3" in encoder.lower()


def _patch_size_for(encoder: str) -> int:
    """ViT patch grid: DINOv3 is /16, DINOv2 (and unknown) is /14."""
    return 16 if _is_dinov3(encoder) else 14


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


def _load_dinov3(encoder: str, device: str) -> torch.nn.Module:
    """Load and freeze a DINOv3 ViT via timm. Cached per (encoder, device).

    `encoder` is a short alias (see `_DINOV3_ALIASES`) or a raw timm model
    name. Weights download once into HF_HOME from timm's ungated re-host
    (no token). `num_classes=0` drops the classifier head; the model is
    moved to `device`, set to eval, and all params frozen.
    """
    key = (encoder, device)
    if key not in _MODEL_CACHE:
        import timm

        name = _DINOV3_ALIASES.get(encoder, encoder)
        model = timm.create_model(name, pretrained=True, num_classes=0)
        model.eval()
        model.to(device)
        for p in model.parameters():
            p.requires_grad_(False)
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


def _dinov3_features(
    model: torch.nn.Module,
    x: torch.Tensor,
    *,
    pool: str,
    chunk: int = 16,
) -> torch.Tensor:
    """DINOv3 (timm) forward + pooling, mirroring the DINOv2 pooling semantics.

    Frames are processed in chunks of `chunk` (the "slightly bigger" model
    at res 512 is memory-heavy) and recombined. Because every frame has the
    same patch count N, the mean over (T, N) equals the mean over T of each
    frame's mean-over-N — so chunking is numerically equivalent to a single
    batched `patches.mean(dim=(0, 1))`.

    timm `forward_features` returns `(b, num_prefix_tokens + N, d)`
    post-LayerNorm tokens, prefix = [CLS, register×4]. So:
      - spatial_mean: mean of the post-norm patch tokens
        (`[:, num_prefix_tokens:, :]`) — the fair analog of DINOv2
        `x_norm_patchtokens`; CLS + register tokens are dropped.
      - cls_mean / cls_first_last: the CLS token (`[:, 0, :]`), time-pooled
        exactly like the DINOv2 CLS path.
    """
    n_prefix = int(getattr(model, "num_prefix_tokens", 1))
    per_frame: list[torch.Tensor] = []
    for i in range(0, x.shape[0], chunk):
        feats = model.forward_features(x[i : i + chunk])  # (b, n_prefix+N, d)
        if pool == "spatial_mean":
            per_frame.append(feats[:, n_prefix:, :].mean(dim=1))  # (b, d)
        else:
            per_frame.append(feats[:, 0, :])  # (b, d) CLS token
    stacked = torch.cat(per_frame, dim=0)  # (T, d)
    if pool == "spatial_mean":
        return stacked.mean(dim=0)
    return _pool_cls(stacked, pool=pool)


# ---------------------------------------------------------------------------
# V-JEPA 2.1 video-encoder path (Stage-5 temporal-grounding ablation).
#
# Unlike DINOv2/DINOv3 (per-frame image encoders, time-pooled afterwards),
# V-JEPA 2.1 is a *clip* encoder: it consumes one (B, C, T, H, W) video tensor
# and emits patch tokens — so it can, in principle, read the two-object
# RELATIONAL motion direction that frame-mean-pooled DINO destroys. This path
# is fully additive: the DINOv2/DINOv3 branches above are untouched.
#
# Verified against the facebookresearch/vjepa2 source (V-JEPA 2.1 release):
#   - torch.hub entrypoint returns (encoder, predictor); we keep only encoder.
#   - encoder.forward(x) wants x = (B, C, T, 384, 384), T = num_frames (64,
#     tubelet 2); returns ALL patch tokens (B, N, d), no CLS token. We mean
#     over the token axis -> (d,), the temporal analog of DINOv2 spatial_mean.
#   - normalization = ImageNet (same constants as above); eval recipe =
#     resize short-side to round(crop*256/224) + center-crop `crop`.
#   - the repo's hub weight URL is stubbed to http://localhost:8300 ("for
#     testing"), so pretrained=True fails; we build pretrained=False and load
#     the real checkpoint from dl.fbaipublicfiles.com, taking state_dict[key]
#     with the repo's module./backbone. prefix-strip.
# ---------------------------------------------------------------------------

_VJEPA_BASE_URL = "https://dl.fbaipublicfiles.com/vjepa2"

# alias -> (hub entrypoint, checkpoint filename, checkpoint_key, crop, n_frames)
_VJEPA_SPECS: dict[str, tuple[str, str, str, int, int]] = {
    "vjepa2_1_vitl16": (
        "vjepa2_1_vit_large_384", "vjepa2_1_vitl_dist_vitG_384",
        "ema_encoder", 384, 64,
    ),
    # ViT-g uses the xformers arch (needs the xformers package) — optional.
    "vjepa2_1_vitg16": (
        "vjepa2_1_vit_giant_384", "vjepa2_1_vitg_384",
        "target_encoder", 384, 64,
    ),
}


def _is_vjepa(encoder: str) -> bool:
    """True if `encoder` names a V-JEPA model (alias or raw entrypoint)."""
    return "vjepa" in encoder.lower()


def _sample_clip_frames(
    frames: list[np.ndarray], n_frames: int
) -> list[np.ndarray]:
    """Uniformly sample `n_frames` frames from a T-frame clip.

    Indices are `linspace(0, T-1, n_frames)` rounded — a subsample when
    T >= n_frames, an upsample (with repeats) when T < n_frames (most demos
    are 36-69 frames vs the model's 64). Always returns exactly `n_frames`
    frames, monotone non-decreasing in source index.
    """
    t = len(frames)
    if t == 0:
        raise ValueError("_sample_clip_frames needs at least one frame")
    if n_frames <= 0:
        raise ValueError(f"n_frames must be positive, got {n_frames}")
    idx = np.rint(np.linspace(0, t - 1, num=n_frames)).astype(int)
    idx = np.clip(idx, 0, t - 1)
    return [frames[i] for i in idx]


def _preprocess_clip_vjepa(
    frames: list[np.ndarray], *, crop_size: int = 384
) -> torch.Tensor:
    """List[(H, W, 3) uint8] -> (T, 3, crop, crop) float32, V-JEPA eval recipe.

    Mirrors the model's own transform (Resize short-side, CenterCrop, ImageNet
    Normalize) in tensor space, so a frozen V-JEPA encoder sees its native
    preprocessing rather than a foreign square-resize (same fairness argument
    as the DINOv3 path in the module docstring). For the square 512x512 demo
    renders this is a resize to ~438 then a center crop to 384.
    """
    arr = np.stack(frames, axis=0)
    if arr.dtype != np.uint8:
        raise ValueError(f"frames must be uint8, got {arr.dtype}")
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"frames must have shape (T, H, W, 3), got {arr.shape}")
    t = torch.from_numpy(arr).permute(0, 3, 1, 2).float().div_(255.0)
    short = int(round(crop_size * 256 / 224))
    _, _, h, w = t.shape
    if h <= w:
        new_h, new_w = short, max(crop_size, int(round(w * short / h)))
    else:
        new_h, new_w = max(crop_size, int(round(h * short / w))), short
    t = F.interpolate(t, size=(new_h, new_w), mode="bilinear",
                      align_corners=False, antialias=True)
    top = (new_h - crop_size) // 2
    left = (new_w - crop_size) // 2
    t = t[:, :, top:top + crop_size, left:left + crop_size]
    mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
    return (t - mean) / std


def _load_vjepa(encoder: str, device: str) -> torch.nn.Module:
    """Load and freeze a V-JEPA 2.1 encoder. Cached per (encoder, device).

    Builds the architecture via torch.hub (pretrained=False, since the repo's
    hub weight URL is a localhost stub), downloads the real checkpoint once
    into the torch hub cache, and loads `state_dict[checkpoint_key]` with the
    repo's prefix-strip. Returns the encoder only (the entrypoint returns an
    (encoder, predictor) tuple; the predictor is unused here).
    """
    key = (encoder, device)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    if encoder not in _VJEPA_SPECS:
        raise ValueError(
            f"unknown V-JEPA encoder {encoder!r}; known: {list(_VJEPA_SPECS)}"
        )
    entry, fname, ckpt_key, _crop, _nf = _VJEPA_SPECS[encoder]
    built = torch.hub.load(
        "facebookresearch/vjepa2", entry, pretrained=False, trust_repo=True
    )
    enc = built[0] if isinstance(built, (tuple, list)) else built
    ckpt_dir = os.path.join(torch.hub.get_dir(), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    local = os.path.join(ckpt_dir, f"{fname}.pt")
    if not os.path.exists(local):
        torch.hub.download_url_to_file(f"{_VJEPA_BASE_URL}/{fname}.pt", local)
    sd = torch.load(local, map_location="cpu", weights_only=False)
    raw = sd[ckpt_key]
    esd = {
        k.replace("module.", "").replace("backbone.", ""): v
        for k, v in raw.items()
    }
    enc.load_state_dict(esd, strict=True)
    enc.eval()
    enc.to(device)
    for p in enc.parameters():
        p.requires_grad_(False)
    _MODEL_CACHE[key] = enc
    return enc


def _vjepa_features(model: torch.nn.Module, clip: torch.Tensor) -> torch.Tensor:
    """V-JEPA clip forward -> (d,) pooled embedding.

    `clip` is (T, 3, H, W); the encoder wants (B, C, T, H, W). Output is all
    patch tokens (1, N, d) — mean over the token axis is the spatial_mean
    analog (V-JEPA has no CLS token).
    """
    x = clip.permute(1, 0, 2, 3).unsqueeze(0)  # (1, 3, T, H, W)
    out = model(x)
    if isinstance(out, (list, tuple)):
        out = out[-1]
    return out.mean(dim=1).squeeze(0)


def extract_vision_features(
    demo_frames: list[np.ndarray],
    *,
    encoder: str = "dinov2_vitb14",
    pool: str = "cls_mean",
    device: str = "cuda",
    resolution: int = 224,
    vjepa_n_frames: Optional[int] = None,  # V-JEPA clip length (default: spec, 64)
    vjepa_crop: Optional[int] = None,      # V-JEPA crop size (default: spec, 384)
    _encoder: Optional[torch.nn.Module] = None,  # test/inject hook
) -> np.ndarray:
    """Frozen-encoder feature extraction from demo RGB frames.

    Args:
      demo_frames: list of (H, W, 3) uint8 RGB arrays — exactly what
        babysteps.render.common.render_frame(env) produces.
      encoder: model id. DINOv2 torch.hub id (default `dinov2_vitb14`) or a
        DINOv3 alias / gated HF repo id (e.g. `dinov3_vitl16`); dispatch is
        by the substring "dinov3".
      pool: time-pooling strategy (see `_pool_cls`).
      device: torch device for the encoder.
      resolution: square resize before encoding. Must be divisible by the
        encoder's patch size (14 for DINOv2 → 224/518; 16 for DINOv3 → 224/512).
      _encoder: optional injected encoder for unit tests (bypasses the loader).

    Returns:
      (d_encoder,) float32 numpy vector. d_encoder = 768 for DINOv2 ViT-B/14
      or DINOv3 ViT-B/16; 1024 for DINOv3 ViT-L/16.

    The Stage-4 firewall applies: this function reads only the frame
    arrays — no labels, no privileged scene state.
    """
    if len(demo_frames) == 0:
        raise ValueError("extract_vision_features needs at least one frame")

    if _is_vjepa(encoder):
        # Clip-encoder path: sample a fixed-length clip, preprocess with the
        # V-JEPA eval recipe, single forward, token-mean pool. `pool` is
        # ignored (V-JEPA has no CLS; token-mean is the only sensible analog).
        spec = _VJEPA_SPECS.get(encoder)
        n_frames = vjepa_n_frames or (spec[4] if spec else 64)
        crop = vjepa_crop or (spec[3] if spec else 384)
        sel = _sample_clip_frames(demo_frames, n_frames)
        clip = _preprocess_clip_vjepa(sel, crop_size=crop).to(device)
        with torch.no_grad():
            model = _encoder if _encoder is not None else _load_vjepa(encoder, device)
            z = _vjepa_features(model, clip)
        return z.detach().cpu().numpy().astype(np.float32)

    is_v3 = _is_dinov3(encoder)
    # Each encoder gets its native resize recipe (see module docstring):
    # DINOv2 keeps bilinear/no-antialias (committed default); DINOv3 uses
    # bicubic + antialias per its timm data config.
    x = _preprocess_frames(
        demo_frames,
        resolution=resolution,
        patch_size=_patch_size_for(encoder),
        interpolation="bicubic" if is_v3 else "bilinear",
        antialias=is_v3,
    ).to(device)
    with torch.no_grad():
        if is_v3:
            model = _encoder if _encoder is not None else _load_dinov3(encoder, device)
            z = _dinov3_features(model, x, pool=pool)
        else:
            model = _encoder if _encoder is not None else _load_dinov2(encoder, device)
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
