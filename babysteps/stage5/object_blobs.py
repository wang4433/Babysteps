"""Sim-free object localization + patch-token pooling for the Stage-5
object-centric relation probe (Step 2).

Pure numpy, no GPU/Vulkan, no ManiSkill — safe on the login node and in the
``tests/`` suite. The job is *localization only*: find where cubeA (red) and
cubeB (green) are in a rendered frame so the GPU extractor can select which
DINOv2 patch tokens to pool. The localization is **pixel-derived** (colour
blobs in the RGB frame), so it is on the deployable path — it does NOT read
privileged sim coordinates (CLAUDE.md invariant #4).

The probe never receives the (u, v) / patch indices as a *feature* — they are
used only to *select* patches. The pooled patch-token VALUES are the feature.
Feeding (u, v) would be near-tautological, because the StackCube
``object_motion`` label IS ``goal_direction_to_motion(cubeB - cubeA)`` (a
cardinal bin of the two cubes' resting positions) — any coordinate just
re-states the label's own input.

Colour convention (ManiSkill StackCube-v1, confirmed in
``babysteps/stage5/vlm_attribute.py``): cubeA is **red**, cubeB is **green**.
"""
from __future__ import annotations

import numpy as np

# cubeA / cubeB colour identities for StackCube-v1.
CUBE_A_COLOR = "red"
CUBE_B_COLOR = "green"


def _channel_dominance(frame_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel ``redness`` and ``greenness`` scores (signed int16).

    ``redness  = R - max(G, B)`` and ``greenness = G - max(R, B)``. Saturated
    cube colours score high; neutral table / white-grey arm score ~0. Computed
    in int16 to avoid uint8 wraparound.
    """
    f = np.asarray(frame_rgb)
    if f.ndim != 3 or f.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) RGB frame, got shape {f.shape}")
    r = f[..., 0].astype(np.int16)
    g = f[..., 1].astype(np.int16)
    b = f[..., 2].astype(np.int16)
    redness = r - np.maximum(g, b)
    greenness = g - np.maximum(r, b)
    return redness, greenness


def _blob_centroid(
    score: np.ndarray, *, thresh: int, min_pixels: int,
) -> tuple[float, float] | None:
    """(u, v) = (col, row) centroid of pixels with ``score > thresh``.

    Returns None if fewer than ``min_pixels`` qualify. ``u`` is horizontal
    (column), ``v`` is vertical (row) — image-plane convention.
    """
    mask = score > thresh
    n = int(mask.sum())
    if n < min_pixels:
        return None
    rows, cols = np.nonzero(mask)
    return float(cols.mean()), float(rows.mean())


def detect_cube_centroids(
    frame_rgb: np.ndarray, *, thresh: int = 40, min_pixels: int = 20,
) -> dict[str, tuple[float, float] | None]:
    """Locate cubeA (red) and cubeB (green) in a rendered RGB frame.

    Pixel-only colour-blob centroids — the deployable-path localizer. Assumes
    one dominant red region (cubeA) and one dominant green region (cubeB), as
    in the StackCube render. Returns ``{"cubeA": (u, v) | None,
    "cubeB": (u, v) | None}`` in pixel coordinates.
    """
    redness, greenness = _channel_dominance(frame_rgb)
    return {
        "cubeA": _blob_centroid(redness, thresh=thresh, min_pixels=min_pixels),
        "cubeB": _blob_centroid(greenness, thresh=thresh, min_pixels=min_pixels),
    }


def pixel_to_patch_rc(
    uv: tuple[float, float], *, img_size: int = 512, grid: int = 16,
) -> tuple[int, int]:
    """Map an image pixel ``(u, v)`` to a DINOv2 patch ``(prow, pcol)``.

    The resize from ``img_size`` to the backbone's 224px input is a uniform
    scale, so the *normalized* fraction (u / img_size) maps straight onto the
    ``grid x grid`` patch lattice. Clamped to ``[0, grid-1]``.
    """
    u, v = float(uv[0]), float(uv[1])
    pcol = int(np.clip(int(u / img_size * grid), 0, grid - 1))
    prow = int(np.clip(int(v / img_size * grid), 0, grid - 1))
    return prow, pcol


def patch_rc_to_flat(prow: int, pcol: int, *, grid: int = 16) -> int:
    """Row-major flat index into a ``(grid*grid, d)`` patch-token array.

    DINOv2 (like every standard ViT) patchifies via a strided Conv2d then
    ``flatten(2).transpose(1, 2)``, so ``x_norm_patchtokens`` are raster-ordered
    top-left → bottom-right: token ``k`` is grid cell ``(k // grid, k % grid)``
    in ``(row=y, col=x)`` image order. The GPU extractor
    (``stage5_extract_object_patch_tokens.py:_assert_row_major_layout``) verifies
    this against the real model before extracting, so this assumption can't drift.
    """
    return prow * grid + pcol


def pool_patch_window(
    patch_tokens: np.ndarray, center_rc: tuple[int, int],
    *, grid: int = 16, radius: int = 1,
) -> np.ndarray:
    """Mean of patch tokens in a ``(2*radius+1)^2`` window around ``center_rc``.

    ``patch_tokens`` is ``(N, d)`` with ``N == grid*grid`` in row-major order
    (``prow*grid + pcol``). The window is clamped to the grid. Returns a ``(d,)``
    pooled appearance vector — the object's local token. ``radius=0`` pools the
    single patch under the centroid.
    """
    pt = np.asarray(patch_tokens)
    if pt.ndim != 2 or pt.shape[0] != grid * grid:
        raise ValueError(
            f"patch_tokens must be (grid*grid, d) = ({grid * grid}, d); "
            f"got {pt.shape}"
        )
    prow, pcol = int(center_rc[0]), int(center_rc[1])
    r0, r1 = max(0, prow - radius), min(grid - 1, prow + radius)
    c0, c1 = max(0, pcol - radius), min(grid - 1, pcol + radius)
    idx = [
        patch_rc_to_flat(rr, cc, grid=grid)
        for rr in range(r0, r1 + 1)
        for cc in range(c0, c1 + 1)
    ]
    return pt[idx].mean(axis=0)


def object_local_tokens(
    patch_tokens: np.ndarray, centroids: dict[str, tuple[float, float] | None],
    *, img_size: int = 512, grid: int = 16, radius: int = 1,
) -> dict[str, np.ndarray | None]:
    """Pool an object-local token for each detected cube.

    Returns ``{"cubeA": (d,) | None, "cubeB": (d,) | None}``. A cube with no
    centroid (detection failed) maps to None — the caller decides how to drop
    or impute it.
    """
    out: dict[str, np.ndarray | None] = {}
    for name, uv in centroids.items():
        if uv is None:
            out[name] = None
            continue
        rc = pixel_to_patch_rc(uv, img_size=img_size, grid=grid)
        out[name] = pool_patch_window(patch_tokens, rc, grid=grid, radius=radius)
    return out
