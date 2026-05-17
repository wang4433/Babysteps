"""End-to-end CLI snapshot tests for scripts/stage0_collect.py.

These tests drive `stage0_collect.main(argv)` with `--fake-env --task ...`
and assert the produced `<out_dir>/samples.jsonl` is byte-equal to the
checked-in snapshot. They guard against script-level regressions
(episode_id naming, missing fake-runner dispatch, etc.) that the
per-adapter snapshot tests in test_pushcube_adapter.py and
test_pickcube_adapter.py do NOT catch — those bypass the script and
call run_episode directly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make scripts/ importable.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))


@pytest.fixture
def collect_main():
    """Lazy-import so each test gets a fresh module if needed."""
    import importlib
    if "stage0_collect" in sys.modules:
        del sys.modules["stage0_collect"]
    mod = importlib.import_module("stage0_collect")
    return mod.main


@pytest.mark.parametrize("task_id,snapshot_name", [
    ("PushCube-v1", "pushcube_samples_seeds_0_4.jsonl"),
    ("PickCube-v1", "pickcube_samples_seeds_0_4.jsonl"),
])
def test_stage0_collect_cli_matches_snapshot(
    tmp_path: Path, collect_main, task_id: str, snapshot_name: str,
):
    """`stage0_collect.py --task X --fake-env --n_episodes 5 --seed_start 0`
    must produce a samples.jsonl byte-equal to the checked-in snapshot."""
    out_dir = tmp_path / "out"
    rc = collect_main([
        "--task", task_id,
        "--fake-env",
        "--out_dir", str(out_dir),
        "--n_episodes", "5",
        "--seed_start", "0",
    ])
    # rc is 0 on PASS, 1 on FAIL (we only care that the script ran).
    assert rc in (0, 1), f"unexpected exit code {rc}"

    actual = (out_dir / "samples.jsonl").read_text()
    snapshot_path = Path(__file__).parent / "snapshots" / snapshot_name
    expected = snapshot_path.read_text()
    assert actual == expected, (
        f"{task_id} CLI output drifted from snapshot {snapshot_path.name}. "
        "If intentional, regenerate via "
        f"`python scripts/stage0_collect.py --task {task_id} --fake-env "
        "--out_dir /tmp/x --n_episodes 5 --seed_start 0` and copy "
        "/tmp/x/samples.jsonl into tests/snapshots/."
    )


def test_stage0_collect_cli_default_task_is_pushcube(tmp_path: Path, collect_main):
    """Backward-compat: invoking without --task must default to PushCube-v1
    and produce the PushCube snapshot byte-equal."""
    out_dir = tmp_path / "out"
    rc = collect_main([
        "--fake-env",
        "--out_dir", str(out_dir),
        "--n_episodes", "5",
        "--seed_start", "0",
    ])
    assert rc in (0, 1)
    actual = (out_dir / "samples.jsonl").read_text()
    expected = (Path(__file__).parent / "snapshots"
                / "pushcube_samples_seeds_0_4.jsonl").read_text()
    assert actual == expected


def test_stage0_collect_cli_unknown_task_errors(tmp_path: Path, collect_main):
    out_dir = tmp_path / "out"
    with pytest.raises((SystemExit, KeyError)) as exc:
        collect_main([
            "--task", "StackCube-v1",
            "--fake-env",
            "--out_dir", str(out_dir),
            "--n_episodes", "1",
        ])
    # argparse choices=… raises SystemExit; if we route through
    # get_task_entry first it raises KeyError. Either is acceptable.
