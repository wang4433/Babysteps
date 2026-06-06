"""Sim-free smoke test for scripts/stage5_cache_dinov2.py.

Pins the import-side contract: top-level import must not pull mani_skill /
sapien, and main() must surface a missing/empty --frames-dir with rc != 0.
The GPU-side DINOv2 load is tested by `python scripts/stage5_cache_dinov2.py
--check` and is not part of this sim-free suite.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path


def test_script_module_imports_sim_free():
    """Top-level import must NOT pull mani_skill/sapien — S3 runs on a GPU host
    where the simulator may not be configured."""
    mod = importlib.import_module("scripts.stage5_cache_dinov2")
    assert hasattr(mod, "main")
    assert "mani_skill" not in sys.modules
    assert "sapien" not in sys.modules


def test_main_returns_nonzero_on_missing_frames_dir(tmp_path: Path):
    """main() must surface a non-existent / empty frames-dir with rc != 0
    so a typo in --frames-dir doesn't silently leave an orphan --out-dir."""
    from scripts.stage5_cache_dinov2 import main

    rc = main(["--frames-dir", str(tmp_path / "missing"),
               "--out-dir", str(tmp_path / "out")])
    assert rc != 0


def test_select_frames_final_state_pooling():
    """Frame subsetting for the goal_state final-state pooling (whole-clip mean
    dilutes a final-state factor). Mirrors clip_pool_frame_indices semantics."""
    from scripts.stage5_cache_dinov2 import select_frames

    frames = list(range(10))  # stand-ins; select_frames is index-only
    assert select_frames(frames, "all") == frames
    assert select_frames(frames, "final") == [9]
    assert select_frames(frames, "first_last") == [0, 9]
    assert select_frames(frames, "last5") == [5, 6, 7, 8, 9]
    # short clip: last5 clamps; first_last on a 1-frame clip collapses to [x,x]
    assert select_frames([7], "first_last") == [7, 7]
    assert select_frames([1, 2, 3], "last5") == [1, 2, 3]
    assert select_frames([], "first_last") == []
    import pytest
    with pytest.raises(ValueError):
        select_frames(frames, "nope")
