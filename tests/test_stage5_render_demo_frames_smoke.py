"""Sim-free smoke tests for scripts/stage5_render_demo_frames.py.

Pins the public surface of the Stage-5 demo-frame re-render script:
  * The module imports without touching mani_skill (it lazy-imports the
    sim only inside ``_make_env`` and the per-task capture helpers).
  * The CLI's seed extraction parses the varied-intent episode_id format
    that ``stage4_collect_varied.py`` emits.
  * The CLI exits non-zero on an empty jsonl, so a malformed input
    surfaces immediately rather than silently writing nothing.

These tests run on the login node — no GPU, no Vulkan.
"""
from __future__ import annotations

import importlib
from pathlib import Path


def test_script_module_imports_sim_free():
    """Top-level import of the script must NOT pull mani_skill or sapien.

    The renderer is GPU-only but the module must remain importable on the
    login node so sim-free tests (and discovery tools) can introspect it
    without a Vulkan device.
    """
    import sys

    mod = importlib.import_module("scripts.stage5_render_demo_frames")
    # _make_env / _pushcube_inject_goal / per-task capture helpers exist
    # so downstream callers (S3) can rely on them being importable.
    assert hasattr(mod, "_make_env")
    assert hasattr(mod, "_capture_pushcube_demo")
    assert hasattr(mod, "_capture_stackcube_demo")
    assert hasattr(mod, "_seed_from_record")
    assert hasattr(mod, "main")
    # Importing the script must not have side-imported the simulator.
    assert "mani_skill" not in sys.modules, (
        "stage5_render_demo_frames imported mani_skill at module load; "
        "the sim import must stay lazy (inside _make_env)."
    )
    assert "sapien" not in sys.modules, (
        "stage5_render_demo_frames imported sapien at module load; "
        "the sim import must stay lazy (inside _pushcube_inject_goal)."
    )


def test_seed_from_record_parses_varied_intent_episode_id():
    """Seed extraction matches the varied-intent episode_id convention."""
    from scripts.stage5_render_demo_frames import _seed_from_record

    assert _seed_from_record({"episode_id": "pushcube_varied_seed_0000"}) == 0
    assert _seed_from_record({"episode_id": "pushcube_varied_seed_0012"}) == 12
    assert _seed_from_record({"episode_id": "stackcube_varied_seed_0084"}) == 84


def test_main_returns_nonzero_on_empty_jsonl(tmp_path: Path):
    """main() must surface an empty jsonl with a non-zero exit, not silently no-op."""
    from scripts.stage5_render_demo_frames import main

    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    out = tmp_path / "frames"
    rc = main(["--jsonl", str(empty), "--out-dir", str(out)])
    assert rc != 0


def test_pushcube_injection_plan_uses_stratified_collection_seed():
    """Regression: script must derive injection from stratified plan, NOT from
    execution.initial_intent.object_motion (the observed motion can drift from
    the injected target when the cube barely moves — see commit c9a5426 bug)."""
    from scripts.stage5_render_demo_frames import _PUSHCUBE_INJECTION_BY_SEED

    # Round-robin: even-indexed in (_PUSHCUBE_DIRS, per_class=10) plan → +x,
    # odd-indexed → -x. The 20-seed PushCube cut is fully covered.
    for seed in range(20):
        assert seed in _PUSHCUBE_INJECTION_BY_SEED, (
            f"seed {seed} not in PushCube injection plan; will fail at render time"
        )
    # Seed 19 (the empirical bug-trigger): must inject -x even though the
    # original episode's observed motion was -y.
    assert _PUSHCUBE_INJECTION_BY_SEED[19] == "translate_-x"
