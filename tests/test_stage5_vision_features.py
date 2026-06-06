"""Stage-5 P1 — vision_features module tests.

Sim-free: CPU-only torch. The real DINOv2 model is never loaded in this
suite; instead an injected fake encoder verifies the pre/post pipeline.
A separate GPU smoke test (scripts/stage5_cache_dinov2.py --check) loads
the real encoder.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_preprocess_frames_shape_and_dtype():
    """List[(H,W,3) uint8] -> (T, 3, 224, 224) float32 ImageNet-normalized."""
    from babysteps.stage4.vision_features import _preprocess_frames

    frames = [
        (255 * np.random.rand(512, 512, 3)).astype(np.uint8)
        for _ in range(4)
    ]
    x = _preprocess_frames(frames, resolution=224)
    assert x.shape == (4, 3, 224, 224)
    assert x.dtype == torch.float32
    # ImageNet-normalized: mean roughly in [-2.2, 2.7] for random pixels.
    assert -3.0 < float(x.mean()) < 3.0


def test_pool_cls_mean_collapses_time_dim():
    """(T, d) cls tokens -> (d,) mean. Numerical identity on a hand-built case."""
    from babysteps.stage4.vision_features import _pool_cls

    cls = torch.tensor([
        [1.0, 2.0, 3.0],
        [3.0, 2.0, 1.0],
        [2.0, 2.0, 2.0],
    ])  # (3, 3)
    z = _pool_cls(cls, pool="cls_mean")
    assert z.shape == (3,)
    torch.testing.assert_close(z, torch.tensor([2.0, 2.0, 2.0]))


def test_pool_cls_unknown_strategy_raises():
    from babysteps.stage4.vision_features import _pool_cls

    with pytest.raises(ValueError, match="unknown pool"):
        _pool_cls(torch.zeros(2, 4), pool="not-a-real-strategy")


class _FakeEncoder(torch.nn.Module):
    """Mock DINOv2 — returns a fixed (T, d) CLS embedding so the test
    verifies the pre→encode→pool→numpy pipeline without loading real weights."""

    def __init__(self, d: int = 768):
        super().__init__()
        self.d = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, 3, R, R). Return a deterministic per-frame embedding
        # whose mean across T is exactly arange(d) / d * mean(x) — checkable.
        T = x.shape[0]
        base = torch.arange(self.d, dtype=torch.float32) / self.d
        # Modulate per-frame so the time-mean has a stable identity.
        per_t = x.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(-1)  # (T, 1)
        return base.unsqueeze(0) * per_t  # (T, d)


def test_extract_vision_features_end_to_end_with_fake_encoder():
    """Full path: uint8 frames -> preprocess -> encode -> pool -> numpy.

    With identical input frames and a deterministic _FakeEncoder, the
    output equals (arange(d) / d) * x_normalized.mean() — a numerical
    identity that catches any future regression in the pipeline.
    """
    from babysteps.stage4.vision_features import (
        _preprocess_frames,
        extract_vision_features,
    )

    frames = [
        (128 * np.ones((512, 512, 3), dtype=np.uint8))
        for _ in range(5)
    ]
    z = extract_vision_features(
        frames,
        device="cpu",
        _encoder=_FakeEncoder(d=768),
    )
    assert isinstance(z, np.ndarray)
    assert z.shape == (768,)
    assert z.dtype == np.float32

    # Identity: the fake encoder returns base * per_t where base = arange(d)/d
    # and per_t = x.mean(dim=(1,2,3)). For identical input frames, per_t is the
    # same scalar across T, so the time-mean equals base * scalar.
    x = _preprocess_frames(frames, resolution=224)
    scalar = float(x.mean())
    expected = (np.arange(768, dtype=np.float32) / 768.0) * scalar
    np.testing.assert_allclose(z, expected, rtol=1e-5, atol=1e-6)


def test_extract_vision_features_rejects_empty_frames():
    from babysteps.stage4.vision_features import extract_vision_features

    with pytest.raises(ValueError, match="at least one frame"):
        extract_vision_features([], device="cpu", _encoder=_FakeEncoder())


def test_preprocess_frames_single_frame_T1():
    """T=1 edge case: shape is (1, 3, 224, 224); mean still finite."""
    from babysteps.stage4.vision_features import _preprocess_frames

    frames = [(128 * np.ones((512, 512, 3), dtype=np.uint8))]
    x = _preprocess_frames(frames, resolution=224)
    assert x.shape == (1, 3, 224, 224)
    assert x.dtype == torch.float32
    assert np.isfinite(float(x.mean()))


def test_preprocess_frames_non_square_resizes_correctly():
    """(H!=W) input gets resized to (R, R)."""
    from babysteps.stage4.vision_features import _preprocess_frames

    frames = [(128 * np.ones((480, 640, 3), dtype=np.uint8)) for _ in range(2)]
    x = _preprocess_frames(frames, resolution=224)
    assert x.shape == (2, 3, 224, 224)


def test_preprocess_frames_rejects_non_uint8_dtype():
    """The dtype-check branch must trigger for any non-uint8 input."""
    from babysteps.stage4.vision_features import _preprocess_frames

    frames = [np.ones((512, 512, 3), dtype=np.float32) for _ in range(2)]
    with pytest.raises(ValueError, match="must be uint8"):
        _preprocess_frames(frames, resolution=224)


def test_preprocess_frames_rejects_wrong_shape():
    """The shape-check branch must trigger for any non-(T, H, W, 3) input."""
    from babysteps.stage4.vision_features import _preprocess_frames

    # 4-channel frames trigger shape[-1] != 3.
    frames = [np.zeros((512, 512, 4), dtype=np.uint8) for _ in range(2)]
    with pytest.raises(ValueError, match=r"shape \(T, H, W, 3\)"):
        _preprocess_frames(frames, resolution=224)


def test_pool_cls_first_last_concat_correct():
    """(T, d) -> (2*d,) concatenation of first and last frame CLS."""
    from babysteps.stage4.vision_features import _pool_cls

    cls = torch.tensor([
        [1.0, 2.0, 3.0],
        [3.0, 2.0, 1.0],
        [9.0, 8.0, 7.0],
    ])  # (3, 3)
    z = _pool_cls(cls, pool="cls_first_last")
    assert z.shape == (6,)
    # First frame is [1, 2, 3], last is [9, 8, 7] → concat = [1, 2, 3, 9, 8, 7].
    torch.testing.assert_close(z, torch.tensor([1.0, 2.0, 3.0, 9.0, 8.0, 7.0]))


def test_pool_cls_first_last_T1_duplicates_single_frame():
    """T=1 edge case: output is still (2*d,) with the single CLS duplicated."""
    from babysteps.stage4.vision_features import _pool_cls

    cls = torch.tensor([[5.0, 6.0, 7.0]])  # (1, 3)
    z = _pool_cls(cls, pool="cls_first_last")
    assert z.shape == (6,)
    torch.testing.assert_close(z, torch.tensor([5.0, 6.0, 7.0, 5.0, 6.0, 7.0]))


class _FakePatchEncoder(torch.nn.Module):
    """Mock DINOv2 that exposes forward_features for the spatial_mean path.

    Returns (T, N_patches, d) patch tokens whose spatial-temporal mean is a
    known function of the inputs — checkable.
    """

    def __init__(self, d: int = 768, n_patches: int = 256):
        super().__init__()
        self.d = d
        self.n_patches = n_patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Won't be called in the spatial_mean path, but defining it keeps
        # the duck-typed interface complete.
        raise AssertionError("spatial_mean must use forward_features, not forward")

    def forward_features(self, x: torch.Tensor) -> dict:
        # x: (T, 3, R, R). Return per-frame, per-patch embeddings whose
        # mean over (T, N) equals arange(d)/d * mean(x). Identical structure
        # to _FakeEncoder.forward but expanded to a patch axis.
        T = x.shape[0]
        base = torch.arange(self.d, dtype=torch.float32) / self.d
        per_t = x.mean(dim=(1, 2, 3))  # (T,)
        # Each (t, n) patch token gets base * per_t[t], independent of n.
        patches = base.view(1, 1, self.d) * per_t.view(T, 1, 1)
        patches = patches.expand(T, self.n_patches, self.d).contiguous()
        return {"x_norm_patchtokens": patches}


def test_extract_vision_features_spatial_mean_uses_forward_features():
    """spatial_mean must read patch tokens via forward_features, not CLS."""
    from babysteps.stage4.vision_features import (
        _preprocess_frames,
        extract_vision_features,
    )

    frames = [
        (128 * np.ones((512, 512, 3), dtype=np.uint8))
        for _ in range(5)
    ]
    z = extract_vision_features(
        frames,
        device="cpu",
        pool="spatial_mean",
        _encoder=_FakePatchEncoder(d=768, n_patches=256),
    )
    assert isinstance(z, np.ndarray)
    assert z.shape == (768,)
    assert z.dtype == np.float32

    # Identity: for identical input frames, per_t is the same scalar across T.
    # Mean over (T, N) of base * per_t = base * per_t.mean() = base * scalar.
    x = _preprocess_frames(frames, resolution=224)
    scalar = float(x.mean())
    expected = (np.arange(768, dtype=np.float32) / 768.0) * scalar
    np.testing.assert_allclose(z, expected, rtol=1e-5, atol=1e-6)


# --------------------------------------------------------------------------- #
# DINOv3 encoder-swap path                                                     #
# --------------------------------------------------------------------------- #

def test_encoder_dispatch_helpers():
    """`_is_dinov3` + `_patch_size_for` route DINOv2 vs DINOv3 correctly."""
    from babysteps.stage4.vision_features import (
        _DINOV3_ALIASES,
        _is_dinov3,
        _patch_size_for,
    )

    assert _is_dinov3("dinov3_vitl16") is True
    assert _is_dinov3("facebook/dinov3-vitb16-pretrain-lvd1689m") is True
    assert _is_dinov3("dinov2_vitb14") is False
    assert _patch_size_for("dinov3_vitl16") == 16
    assert _patch_size_for("dinov2_vitb14") == 14
    # A raw timm DINOv3 name routes correctly too.
    assert _is_dinov3("vit_large_patch16_dinov3.lvd1689m") is True
    # Every alias resolves to an ungated timm dinov3 model name.
    for alias, name in _DINOV3_ALIASES.items():
        assert alias.startswith("dinov3_")
        assert "dinov3" in name and name.endswith(".lvd1689m")


def test_preprocess_frames_patch16_rejects_non_divisible_resolution():
    """patch_size 16 must reject 518 (37*14, not a /16 multiple) loudly."""
    from babysteps.stage4.vision_features import _preprocess_frames

    frames = [(128 * np.ones((480, 640, 3), dtype=np.uint8)) for _ in range(2)]
    with pytest.raises(ValueError, match="divisible by patch_size 16"):
        _preprocess_frames(frames, resolution=518, patch_size=16)
    # 512 = 32*16 is valid.
    x = _preprocess_frames(frames, resolution=512, patch_size=16)
    assert x.shape == (2, 3, 512, 512)


class _FakeDinov3(torch.nn.Module):
    """Mock timm DINOv3 ViT with the real `[CLS, register*4, patch]` token
    layout exposed via `forward_features` + `num_prefix_tokens`.

    The CLS and register prefix tokens are filled with large SENTINEL values
    (+999 / -999): any code that fails to slice them off before the patch
    mean produces a wildly wrong result, so the identity check below is a
    sharp test of the `num_prefix_tokens` offset.
    """

    def __init__(self, d: int = 1024, n_patches: int = 196, n_reg: int = 4):
        super().__init__()
        self.d = d
        self.n_patches = n_patches
        self.num_prefix_tokens = 1 + n_reg  # 1 CLS + n_reg register tokens

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        base = torch.arange(self.d, dtype=torch.float32) / self.d
        per_t = x.mean(dim=(1, 2, 3))  # (b,)
        patches = base.view(1, 1, self.d) * per_t.view(b, 1, 1)
        patches = patches.expand(b, self.n_patches, self.d).contiguous()
        cls = torch.full((b, 1, self.d), 999.0)
        reg = torch.full((b, self.num_prefix_tokens - 1, self.d), -999.0)
        return torch.cat([cls, reg, patches], dim=1)  # (b, n_prefix+N, d)


def test_dinov3_spatial_mean_drops_cls_and_register_tokens():
    """spatial_mean on DINOv3 must mean ONLY the patch tokens (num_prefix offset)."""
    from babysteps.stage4.vision_features import (
        _preprocess_frames,
        extract_vision_features,
    )

    frames = [(128 * np.ones((512, 512, 3), dtype=np.uint8)) for _ in range(5)]
    z = extract_vision_features(
        frames,
        encoder="dinov3_vitl16",  # routes through the DINOv3 path
        device="cpu",
        pool="spatial_mean",
        _encoder=_FakeDinov3(d=1024, n_patches=196, n_reg=4),
    )
    assert z.shape == (1024,)
    assert z.dtype == np.float32

    # Patches carry base*scalar; CLS=+999, register=-999. If the slice were
    # wrong the mean would be dominated by the sentinels. Correct slice →
    # exactly base*scalar (and never near ±999). Constant input frames →
    # bicubic/antialias resize gives the same constant, so the identity holds
    # regardless of the DINOv3 resize recipe.
    x = _preprocess_frames(frames, resolution=224, patch_size=16,
                           interpolation="bicubic", antialias=True)
    scalar = float(x.mean())
    expected = (np.arange(1024, dtype=np.float32) / 1024.0) * scalar
    np.testing.assert_allclose(z, expected, rtol=1e-5, atol=1e-6)
    assert np.abs(z).max() < 10.0  # sentinels excluded


def test_dinov3_cls_pool_uses_cls_token():
    """Non-spatial pooling on DINOv3 reads the CLS token (index 0), not patches."""
    from babysteps.stage4.vision_features import extract_vision_features

    frames = [(128 * np.ones((512, 512, 3), dtype=np.uint8)) for _ in range(3)]
    z = extract_vision_features(
        frames,
        encoder="dinov3_vitl16",
        device="cpu",
        pool="cls_mean",
        _encoder=_FakeDinov3(d=64, n_patches=49, n_reg=4),
    )
    assert z.shape == (64,)
    # The CLS token is +999 for every frame → time-mean is all 999.
    np.testing.assert_allclose(z, np.full(64, 999.0, dtype=np.float32))


def test_dinov3_chunking_is_equivalent_to_single_batch():
    """`_dinov3_features` chunking must equal an unchunked pass (mean identity)."""
    from babysteps.stage4.vision_features import _dinov3_features

    torch.manual_seed(0)
    model = _FakeDinov3(d=128, n_patches=64, n_reg=4)
    x = torch.rand(37, 3, 224, 224)  # T not a multiple of chunk
    z_small = _dinov3_features(model, x, pool="spatial_mean", chunk=4)
    z_big = _dinov3_features(model, x, pool="spatial_mean", chunk=1000)
    torch.testing.assert_close(z_small, z_big, rtol=1e-5, atol=1e-6)


# --------------------------------------------------------------------------- #
# V-JEPA 2.1 video-encoder path (Stage-5 temporal-grounding ablation)          #
# --------------------------------------------------------------------------- #

def test_vjepa_dispatch_helpers():
    """`_is_vjepa` routes V-JEPA vs DINOv2/DINOv3; specs are well-formed."""
    from babysteps.stage4.vision_features import _VJEPA_SPECS, _is_vjepa

    assert _is_vjepa("vjepa2_1_vitl16") is True
    assert _is_vjepa("vjepa2_1_vit_large_384") is True
    assert _is_vjepa("dinov2_vitb14") is False
    assert _is_vjepa("dinov3_vitl16") is False
    # Each spec: (hub entrypoint, ckpt filename, ckpt key, crop, n_frames).
    for alias, spec in _VJEPA_SPECS.items():
        assert alias.startswith("vjepa2_1_")
        entry, fname, key, crop, nf = spec
        assert entry.startswith("vjepa2_1_vit")
        assert fname.startswith("vjepa2_1_")
        assert key in ("ema_encoder", "target_encoder")
        assert crop % 16 == 0 and nf % 2 == 0  # patch 16, tubelet 2


def test_sample_clip_frames_subsample_monotone():
    """T >= n: uniform subsample, length n, monotone, spanning [0, T-1]."""
    from babysteps.stage4.vision_features import _sample_clip_frames

    frames = list(range(40))  # sentinels = source indices
    out = _sample_clip_frames(frames, 16)
    assert len(out) == 16
    assert out == sorted(out)        # monotone non-decreasing
    assert out[0] == 0 and out[-1] == 39
    assert all(0 <= i <= 39 for i in out)


def test_sample_clip_frames_upsample_short_clip():
    """T < n: upsample with repeats to exactly n, still monotone in [0, T-1]."""
    from babysteps.stage4.vision_features import _sample_clip_frames

    out = _sample_clip_frames(list(range(5)), 16)
    assert len(out) == 16
    assert out == sorted(out)
    assert out[0] == 0 and out[-1] == 4


def test_sample_clip_frames_edge_cases():
    from babysteps.stage4.vision_features import _sample_clip_frames

    assert _sample_clip_frames([7], 16) == [7] * 16  # T=1 → repeat
    with pytest.raises(ValueError, match="at least one frame"):
        _sample_clip_frames([], 16)
    with pytest.raises(ValueError, match="n_frames must be positive"):
        _sample_clip_frames([1, 2, 3], 0)


def test_preprocess_clip_vjepa_shape_and_norm():
    """List[(H,W,3) uint8] -> (T, 3, crop, crop) float32, ImageNet-normalized."""
    from babysteps.stage4.vision_features import _preprocess_clip_vjepa

    frames = [(128 * np.ones((512, 512, 3), dtype=np.uint8)) for _ in range(4)]
    x = _preprocess_clip_vjepa(frames, crop_size=384)
    assert x.shape == (4, 3, 384, 384)
    assert x.dtype == torch.float32
    assert -3.0 < float(x.mean()) < 3.0
    # uint8 contract enforced (same guard as _preprocess_frames).
    with pytest.raises(ValueError, match="must be uint8"):
        _preprocess_clip_vjepa([np.ones((8, 8, 3), dtype=np.float32)], crop_size=16)


class _FakeVJEPA(torch.nn.Module):
    """Mock V-JEPA clip encoder.

    Asserts it is fed a 5-D (B, C, T, H, W) tensor (i.e. the clip path, never
    the per-frame `_pool_cls` path), and returns (B, N, d) patch tokens (no
    CLS) whose token-mean is a known function of the input — checkable.
    """

    def __init__(self, d: int = 8, n_tokens: int = 6):
        super().__init__()
        self.d = d
        self.n_tokens = n_tokens
        self.embed_dim = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 5, f"V-JEPA expects (B,C,T,H,W), got {tuple(x.shape)}"
        b = x.shape[0]
        base = torch.arange(self.d, dtype=torch.float32) / self.d
        scalar = x.mean()
        return base.view(1, 1, self.d).expand(b, self.n_tokens, self.d) * scalar


def test_extract_vision_features_vjepa_branch_shape_and_identity():
    """Full V-JEPA path: frames -> clip preprocess -> 5-D forward -> token-mean.

    With constant frames and the deterministic _FakeVJEPA, the output equals
    (arange(d)/d) * clip.mean() — a numerical identity that pins the branch
    (5-D layout, token-mean pooling, numpy cast) without real weights.
    """
    from babysteps.stage4.vision_features import (
        _preprocess_clip_vjepa,
        _sample_clip_frames,
        extract_vision_features,
    )

    frames = [(128 * np.ones((512, 512, 3), dtype=np.uint8)) for _ in range(9)]
    z = extract_vision_features(
        frames,
        encoder="vjepa2_1_vitl16",
        device="cpu",
        vjepa_n_frames=4,
        vjepa_crop=32,
        _encoder=_FakeVJEPA(d=8, n_tokens=6),
    )
    assert isinstance(z, np.ndarray)
    assert z.shape == (8,)
    assert z.dtype == np.float32

    sel = _sample_clip_frames(frames, 4)
    clip = _preprocess_clip_vjepa(sel, crop_size=32)
    scalar = float(clip.mean())
    expected = (np.arange(8, dtype=np.float32) / 8.0) * scalar
    np.testing.assert_allclose(z, expected, rtol=1e-5, atol=1e-6)
