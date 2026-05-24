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
