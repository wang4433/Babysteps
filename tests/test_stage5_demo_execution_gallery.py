"""Sim-free tests for scripts/stage5_build_demo_execution_gallery.py.

The gallery builder reads existing MP4s with OpenCV; these tests pin the
pure, deterministic helpers that decide *what the gallery says* — without a
GPU, a simulator, or any real video file:

  * ``pick_frame_indices`` selects first/mid/late(n-5)/final, clamped and
    monotone even for tiny clips.
  * ``intent_diff_rows`` flags match / MISMATCH / REVISED correctly.
  * ``classify_banner_success`` reads ``success=True/False`` back from a
    burned-in caption banner via the font-exact token ladder — including the
    truncated ``Fals`` case (long policy names run off the frame edge) and the
    demo clip with no flag (must return ``None``). The banner is rendered with
    the SAME font/position as ``babysteps.render.common.annotate_frame``, so
    this is a real round-trip of the caption the renders actually burn in.

All sim-free: pure Pillow + OpenCV + NumPy on the login node.
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest

gal = importlib.import_module("scripts.stage5_build_demo_execution_gallery")


# --------------------------------------------------------------------------
# pick_frame_indices
# --------------------------------------------------------------------------

def test_pick_frame_indices_typical():
    idx = gal.pick_frame_indices(40)
    assert idx == {"first": 0, "mid": 20, "late": 35, "final": 39}


def test_pick_frame_indices_clamped_and_monotone_tiny_clip():
    # n=3: late=n-5 clamps to 0; result must stay within [0,2] and not regress.
    idx = gal.pick_frame_indices(3)
    assert idx["first"] == 0 and idx["final"] == 2
    vals = [idx["first"], idx["mid"], idx["late"], idx["final"]]
    assert all(0 <= v <= 2 for v in vals)
    assert vals == sorted(vals)  # monotone non-decreasing


def test_pick_frame_indices_empty():
    assert gal.pick_frame_indices(0) == {}


# --------------------------------------------------------------------------
# intent_diff_rows
# --------------------------------------------------------------------------

def test_intent_diff_flags_match_mismatch_revised():
    gt = {f: "x" for f in gal.INTENT_FIELDS}
    decoded = dict(gt)
    decoded["contact_region"] = "y"          # decoded disagrees with gt -> MISMATCH
    revised = dict(decoded)
    revised["approach_direction"] = "z"      # revised changes a decoded slot -> REVISED
    rows = {r[0]: r for r in gal.intent_diff_rows(gt, decoded, revised)}
    assert rows["goal_state"][4] == "match"
    assert "MISMATCH" in rows["contact_region"][4]
    assert "REVISED" in rows["approach_direction"][4]
    # exactly one factor revised (single-factor invariant is observable here)
    revised_count = sum("REVISED" in r[4] for r in rows.values())
    assert revised_count == 1


def test_intent_diff_handles_missing_inputs():
    rows = gal.intent_diff_rows(None, None, None)
    assert len(rows) == len(gal.INTENT_FIELDS)
    assert all(flag == "" for *_, flag in rows)


# --------------------------------------------------------------------------
# classify_banner_success — round-trip the burned-in caption
# --------------------------------------------------------------------------

def _banner(title: str, *, width: int = 512, banner_h: int = 60) -> np.ndarray:
    """Replicate annotate_frame's title band: DejaVuSans-Bold 16 at (10,6) on
    a (16,16,16) background, returned as a grayscale crop (what the matcher
    consumes)."""
    import cv2
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (width, banner_h), (16, 16, 16))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(gal._FONT_BOLD, 16)
    except Exception:  # pragma: no cover - font present on the cluster
        pytest.skip("DejaVuSans-Bold not available")
    draw.text((10, 6), title, fill=(255, 255, 255), font=font)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)


def test_classify_reads_success_true():
    b = _banner("seed 0007  phase 3/3: retry (success=True)")
    assert gal.classify_banner_success([b]) is True


def test_classify_reads_success_false():
    b = _banner("seed 0007  phase 3/3: retry (success=False)")
    assert gal.classify_banner_success([b]) is False


def test_classify_reads_truncated_false():
    # Long policy name pushes "(success=False)" partly off the 512px frame, so
    # the rendered word truncates to "Fals" — the ladder must still read False.
    b = _banner("seed 0100  policy: oracle_factor_revision  (success=False)")
    assert gal.classify_banner_success([b]) is False


def test_classify_returns_none_for_demo_without_flag():
    # Demo clips carry no success= token -> must abstain, never guess.
    b = _banner("seed 0007  phase 1/3: demo proxy")
    assert gal.classify_banner_success([b]) is None


def test_classify_returns_none_on_empty():
    assert gal.classify_banner_success([]) is None
