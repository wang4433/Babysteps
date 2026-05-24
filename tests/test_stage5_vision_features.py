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
    """Full path: uint8 frames -> preprocess -> encode -> pool -> numpy."""
    from babysteps.stage4.vision_features import extract_vision_features

    frames = [
        (128 * np.ones((512, 512, 3), dtype=np.uint8))
        for _ in range(5)
    ]
    z = extract_vision_features(
        frames,
        device="cpu",
        _encoder=_FakeEncoder(d=768),  # injection for test
    )
    assert isinstance(z, np.ndarray)
    assert z.shape == (768,)
    assert z.dtype == np.float32
    # Identical input frames -> identical per-frame embeddings -> mean = embedding.
    # Embedding magnitude > 0 (non-trivial signal).
    assert float(np.abs(z).sum()) > 0.0


def test_extract_vision_features_rejects_empty_frames():
    from babysteps.stage4.vision_features import extract_vision_features

    with pytest.raises(ValueError, match="at least one frame"):
        extract_vision_features([], device="cpu", _encoder=_FakeEncoder())
