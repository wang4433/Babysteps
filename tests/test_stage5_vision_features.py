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
