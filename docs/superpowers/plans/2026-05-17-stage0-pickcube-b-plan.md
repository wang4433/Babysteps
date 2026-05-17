# Sub-project B (PickCube) — Acceptance Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close Sub-project B's acceptance gate by wiring a `--task` flag through the four Stage-0 CLI scripts so `PickCube-v1` can be driven end-to-end (snapshot byte-stability, GPU MP4 spot-check, `delta_pp >= 10`) without regressing the existing `PushCube-v1` flow.

**Architecture:** Introduce `babysteps/envs/task_registry.py` as the single `{task_id → (AdapterClass, FakeRunnerFactory, episode_id_prefix)}` dispatch. `stage0_collect.py`, `stage0_summarize.py` (via `eval.py`), and `render_stage0_maniskill.py` all read from the registry. Render-script per-task differences (PushCube freezes phase 2; PickCube actually executes the grasp-slip lift) are factored into `babysteps/render/{common,pushcube,pickcube}.py` with the script becoming a thin dispatcher.

**Tech Stack:** Python 3, ManiSkill 3 (gymnasium), pytest, numpy, PIL+imageio for MP4s. All adapter / runner / skill / schema / failure / revision code for PickCube already exists and is tested.

**Source spec:** `docs/superpowers/specs/2026-05-17-stage0-four-scene-roadmap-design.md` (Sections 3, 7.1, 8).

**Scope guardrails:**
- Sub-project B only. C (StackCube) and D (Drawer) are deferred (spec §7.2/§7.3 are scoped-only).
- No new schema entries — all of B's deltas (Section 4) already landed during PickCube adapter work (`schemas.py`, `failure.py`, `revision.py` verified 2026-05-17).
- PushCube's `tests/snapshots/pushcube_samples_seeds_0_4.jsonl` MUST stay byte-identical.
- All ~149 explicit `def test_*` (with parametrize expansion ≈163) must continue to pass.
- One-attempt-then-one-retry per episode (spec §2 non-goal — no multi-attempt loops).
- Privileged-firewall: `scripted_demo_to_intent` stays demo-only; `blocked_sides` only consumed by env_runners and oracles.

---

## File Structure

**Create:**
- `babysteps/envs/task_registry.py` — `TASK_REGISTRY` dict and helper `get_task_entry(task_id)`.
- `babysteps/render/__init__.py` — exports `RENDER_REGISTRY: {task_id → render_episode_fn}`.
- `babysteps/render/common.py` — shared frame readers, PD action helper, frame capture, annotation, MP4 writer (moved from `scripts/render_stage0_maniskill.py`).
- `babysteps/render/pushcube.py` — `render_episode(env, adapter, seed, fps)` → `(demo_frames, attempt1_frames, retry_frames, titles)`.
- `babysteps/render/pickcube.py` — same signature; phase 2 actually steps the env so the grasp slip is visible.
- `tests/test_task_registry.py` — registry shape + smoke tests.
- `tests/test_stage0_collect_cli.py` — end-to-end CLI snapshot test (drives `stage0_collect.main`) for both tasks.
- `tests/test_render_modules.py` — sim-free unit tests for the per-task render modules (test that they call the adapter's API and emit the right number of frames, using a stub env).
- `tests/test_pickcube_delta_pp.py` — fake-env end-to-end gate test (`delta_pp >= 10` for PickCube across 5 seeds).

**Modify:**
- `babysteps/eval.py` — `compute_metrics` adds `"task"` field (read from `records[0].task`); `_markdown_report` uses it for the title.
- `scripts/stage0_collect.py` — `--task` flag, registry-based dispatch, task-aware episode_id prefix.
- `scripts/stage0_summarize.py` — no flag change needed (task lives in records); just verify report title reflects the task.
- `scripts/render_stage0_maniskill.py` — `--task` flag, dispatch to `babysteps.render.RENDER_REGISTRY[args.task]`.
- `tests/test_eval.py::test_write_report_creates_md_and_json` — adjust to expect task-name in title.
- `CLAUDE.md` — replace the PushCube-only GPU command with `--task` form; add the PickCube spot-check command.

**Untouched:**
- `babysteps/envs/pickcube_adapter.py`, `babysteps/envs/pickcube_runner.py`, `babysteps/skills/pick.py` — done.
- `babysteps/envs/pushcube_adapter.py`, `babysteps/envs/pushcube_runner.py`, `babysteps/skills/push.py` — done.
- `babysteps/envs/task_adapter.py` (`BaseTaskAdapter`) — done.
- `babysteps/schemas.py`, `babysteps/failure.py`, `babysteps/revision.py` — B's deltas already present.
- `tests/conftest.py` — `FakeEnvRunner` and `FakePickEnvRunner` already defined.
- `tests/snapshots/pushcube_samples_seeds_0_4.jsonl`, `tests/snapshots/pickcube_samples_seeds_0_4.jsonl` — both exist; snapshot tests already in `test_pushcube_adapter.py` and `test_pickcube_adapter.py`.

---

## Task 0: Baseline — confirm the world is in the state this plan assumes

**Files:** none — read-only verification.

- [ ] **Step 1: Run the full test suite to confirm the green baseline**

Run: `cd /scratch/gilbreth/wang4433/babysteps && python -m pytest tests/ -q 2>&1 | tail -20`

Expected: All tests pass (~149+ explicit tests, ~163 with parametrize). If anything fails, STOP and report — this plan presumes a green baseline.

- [ ] **Step 2: Confirm both snapshot fixtures exist and have 5 lines each**

Run: `wc -l tests/snapshots/*.jsonl`

Expected:
```
   5 tests/snapshots/pickcube_samples_seeds_0_4.jsonl
   5 tests/snapshots/pushcube_samples_seeds_0_4.jsonl
  10 total
```

- [ ] **Step 3: Confirm schema/failure/revision deltas are present**

Run:
```bash
grep -E '("grasp_slip"|"contact_substitution"|"cube_lifted_at_target"|"lift_up"|"proxy_contact_to_franka_grasp")' babysteps/schemas.py babysteps/failure.py babysteps/revision.py
```

Expected: hits in all three files. If anything is missing, the spec's Section 4 deltas weren't landed and this plan needs a pre-task to add them. (As of 2026-05-17 they're present.)

- [ ] **Step 4: Commit a marker tag for easy rollback**

Run:
```bash
git status --short
git stash --include-untracked -m "pre-plan-2026-05-17-stage0-pickcube-b stash"
git stash pop
```

Just lists state; do not actually commit a tag. The stash dance is a no-op safety check that working-tree is consistent.

---

## Task 1: `task_registry.py` — single dispatch table

**Files:**
- Create: `babysteps/envs/task_registry.py`
- Test: `tests/test_task_registry.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_task_registry.py`:

```python
"""Tests for babysteps.envs.task_registry — Stage-0 task dispatch."""
from __future__ import annotations

import pytest

from babysteps.envs.task_registry import (
    TASK_REGISTRY,
    TaskEntry,
    get_task_entry,
)


def test_registry_contains_both_stage0_tasks():
    """PushCube-v1 (sub-project A) and PickCube-v1 (sub-project B) must be present."""
    assert set(TASK_REGISTRY.keys()) == {"PushCube-v1", "PickCube-v1"}


def test_get_task_entry_pushcube():
    from babysteps.envs.pushcube_adapter import PushCubeAdapter
    entry = get_task_entry("PushCube-v1")
    assert isinstance(entry, TaskEntry)
    assert entry.adapter_cls is PushCubeAdapter
    assert entry.episode_id_prefix == "pushcube_blocked_approach"


def test_get_task_entry_pickcube():
    from babysteps.envs.pickcube_adapter import PickCubeAdapter
    entry = get_task_entry("PickCube-v1")
    assert isinstance(entry, TaskEntry)
    assert entry.adapter_cls is PickCubeAdapter
    assert entry.episode_id_prefix == "pickcube_grasp_slip"


def test_get_task_entry_unknown_task_raises():
    with pytest.raises(KeyError) as exc:
        get_task_entry("StackCube-v1")
    msg = str(exc.value)
    assert "StackCube-v1" in msg
    assert "PushCube-v1" in msg
    assert "PickCube-v1" in msg


def test_fake_runner_factory_pushcube():
    """Factory returns an env_runner whose .run/.reset/.close work without
    a real simulator (i.e., one of the FakeEnvRunner classes)."""
    entry = get_task_entry("PushCube-v1")
    runner = entry.fake_runner_factory()
    assert hasattr(runner, "reset")
    assert hasattr(runner, "run")
    assert hasattr(runner, "close")
    runner.close()


def test_fake_runner_factory_pickcube():
    entry = get_task_entry("PickCube-v1")
    runner = entry.fake_runner_factory()
    assert hasattr(runner, "reset")
    assert hasattr(runner, "run")
    assert hasattr(runner, "close")
    runner.close()


def test_registry_entries_are_taskentry_instances():
    for task_id, entry in TASK_REGISTRY.items():
        assert isinstance(entry, TaskEntry), f"{task_id} entry is not TaskEntry"
        # adapter_cls's task_id must match the registry key.
        assert entry.adapter_cls.task_id == task_id
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_task_registry.py -v`

Expected: All tests FAIL with `ModuleNotFoundError: No module named 'babysteps.envs.task_registry'`.

- [ ] **Step 3: Write the implementation**

Create `babysteps/envs/task_registry.py`:

```python
"""Stage-0 task dispatch table.

The single source of truth that maps a `--task` CLI arg to:
  * the concrete BaseTaskAdapter subclass to instantiate,
  * a factory for the deterministic sim-free fake env_runner (used by
    `--fake-env` and by the end-to-end snapshot tests),
  * the episode_id prefix that names this task's Stage-0 controlled
    failure (e.g. "pushcube_blocked_approach" vs "pickcube_grasp_slip").

The episode_id_prefix is held here (not on the adapter) because it's a
Stage-0 cosmetic naming convention tied to the controlled-failure
mechanism, not a task-semantic decision the adapter cares about.

Adding a new Stage-0 task (Sub-projects C, D) is a one-entry addition
here plus the corresponding render module entry in babysteps.render.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from babysteps.envs.task_adapter import BaseTaskAdapter, EnvRunner


@dataclass(frozen=True)
class TaskEntry:
    """One row of the Stage-0 task dispatch table."""
    adapter_cls: type[BaseTaskAdapter]
    fake_runner_factory: Callable[[], EnvRunner]
    episode_id_prefix: str


def _pushcube_entry() -> TaskEntry:
    # Imports are lazy so that importing this module does not pull
    # ManiSkill (via the adapter chain) until a task is actually selected.
    from babysteps.envs.pushcube_adapter import PushCubeAdapter

    def _make_fake() -> EnvRunner:
        # FakeEnvRunner lives in tests/conftest.py; the registry deliberately
        # references it so the same code-path that powers `--fake-env` is the
        # one the snapshot tests exercise. The import is local to keep
        # tests/ off the runtime import path.
        from tests.conftest import FakeEnvRunner
        return FakeEnvRunner()

    return TaskEntry(
        adapter_cls=PushCubeAdapter,
        fake_runner_factory=_make_fake,
        episode_id_prefix="pushcube_blocked_approach",
    )


def _pickcube_entry() -> TaskEntry:
    from babysteps.envs.pickcube_adapter import PickCubeAdapter

    def _make_fake() -> EnvRunner:
        from tests.conftest import FakePickEnvRunner
        return FakePickEnvRunner()

    return TaskEntry(
        adapter_cls=PickCubeAdapter,
        fake_runner_factory=_make_fake,
        episode_id_prefix="pickcube_grasp_slip",
    )


TASK_REGISTRY: dict[str, TaskEntry] = {
    "PushCube-v1": _pushcube_entry(),
    "PickCube-v1": _pickcube_entry(),
}


def get_task_entry(task_id: str) -> TaskEntry:
    """Return the TaskEntry for `task_id` or raise KeyError listing the
    known tasks (handy when the user mistypes `--task`)."""
    if task_id not in TASK_REGISTRY:
        known = sorted(TASK_REGISTRY.keys())
        raise KeyError(
            f"unknown task {task_id!r}; known tasks: {known}"
        )
    return TASK_REGISTRY[task_id]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_task_registry.py -v`

Expected: All 7 tests PASS.

- [ ] **Step 5: Verify the full suite still passes**

Run: `python -m pytest tests/ -q 2>&1 | tail -10`

Expected: All tests pass (gains 7 from `test_task_registry.py`).

- [ ] **Step 6: Commit**

```bash
git add babysteps/envs/task_registry.py tests/test_task_registry.py
git commit -m "$(cat <<'EOF'
feat(envs): task_registry.py — single dispatch table for --task

Maps task_id → (AdapterClass, fake_runner_factory, episode_id_prefix)
for Stage-0's PushCube-v1 and PickCube-v1. Subsequent --task wiring
in stage0_collect.py and render_stage0_maniskill.py reads from here.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `--task` flag in `stage0_collect.py`

**Files:**
- Modify: `scripts/stage0_collect.py`
- Test: `tests/test_stage0_collect_cli.py` (new)

- [ ] **Step 1: Write the failing end-to-end CLI snapshot test**

Create `tests/test_stage0_collect_cli.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_stage0_collect_cli.py -v`

Expected: All four parametrized assertions FAIL — current script has no `--task` flag, episode_ids will be wrong for PickCube, and there's no PickCube fake-runner dispatch.

- [ ] **Step 3: Modify `scripts/stage0_collect.py`**

Replace the current contents with:

```python
"""Stage-0 blocked-approach data-collection CLI.

Runs `run_episode` for `--n_episodes` seeded episodes and writes one
EpisodeRecord per line to `<out_dir>/samples.jsonl`. Then computes
metrics and writes `report.{md,json}` next to the samples.

Tasks (dispatched via babysteps.envs.task_registry):
  --task PushCube-v1  (default) — Sub-project A, approach_blocked failure.
  --task PickCube-v1            — Sub-project B, grasp_slip failure.

Backends:
  --fake-env: deterministic sim-free runner from tests/conftest. Each
              task has its own fake (FakeEnvRunner / FakePickEnvRunner)
              wired through the registry.
  (default):  real env_runner from the adapter (needs Vulkan).

If `mani_skill` fails to import and `--fake-env` was NOT requested, this
script aborts with the import error rather than silently falling back —
the user sees the real failure mode.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the project root importable without `pip install -e .`.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from babysteps.envs.task_adapter import BaseTaskAdapter  # noqa: E402
from babysteps.envs.task_registry import (  # noqa: E402
    TASK_REGISTRY,
    get_task_entry,
)
from babysteps.episode import run_episode  # noqa: E402
from babysteps.eval import compute_metrics, write_report  # noqa: E402
from babysteps.schemas import EpisodeRecord  # noqa: E402


def _make_adapter(task_id: str, use_fake: bool) -> BaseTaskAdapter:
    """Build the right adapter for `task_id`, wired to fake or real runner."""
    entry = get_task_entry(task_id)
    if use_fake:
        fake = entry.fake_runner_factory()

        class _FakeAdapter(entry.adapter_cls):  # type: ignore[misc, valid-type]
            def make_env_runner(self):
                return fake

        return _FakeAdapter()
    return entry.adapter_cls()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--task", type=str, default="PushCube-v1",
        choices=sorted(TASK_REGISTRY.keys()),
        help="Which Stage-0 ManiSkill task to drive. Default PushCube-v1 "
             "for backward compatibility.",
    )
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--n_episodes", type=int, default=5)
    p.add_argument("--seed_start", type=int, default=0)
    p.add_argument(
        "--fake-env", action="store_true",
        help="Use the deterministic sim-free fake env_runner from "
             "tests/conftest. Useful for verifying the loop and JSONL "
             "shape on a login node where Vulkan is unavailable.",
    )
    p.add_argument(
        "--rollouts-subdir", type=str, default="rollouts",
        help="Sub-directory of out_dir to hold per-episode rollout .npz "
             "files. Only the real env_runner writes these.",
    )
    args = p.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    samples_path = args.out_dir / "samples.jsonl"
    samples_path.write_text("")   # truncate any prior run

    entry = get_task_entry(args.task)
    adapter = _make_adapter(args.task, args.fake_env)
    records: list[EpisodeRecord] = []
    try:
        for i in range(args.n_episodes):
            seed = args.seed_start + i
            episode_id = f"{entry.episode_id_prefix}_seed_{seed:04d}"
            rec = run_episode(
                episode_id=episode_id,
                seed=seed,
                adapter=adapter,
            )
            records.append(rec)
            with samples_path.open("a") as f:
                f.write(rec.to_jsonl_line() + "\n")
            print(
                f"[{i + 1}/{args.n_episodes}] task={args.task} seed={seed} "
                f"initial_success={rec.metrics['initial_success']} "
                f"retry_success={rec.metrics['retry_success']} "
                f"failure_type={rec.metrics['failure_type']}",
                flush=True,
            )
    finally:
        adapter.close()

    metrics = compute_metrics(records)
    write_report(metrics, args.out_dir)
    print()
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed_acceptance"] else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the new tests to verify they pass**

Run: `python -m pytest tests/test_stage0_collect_cli.py -v`

Expected: all four parametrized assertions PASS.

- [ ] **Step 5: Run the FULL test suite — PushCube snapshots must be byte-identical**

Run: `python -m pytest tests/ -q 2>&1 | tail -10`

Expected: all tests PASS (gain 4 new). If `test_pushcube_adapter_samples_jsonl_matches_pre_a_snapshot` fails, the script's PushCube behavior drifted — STOP and diff.

- [ ] **Step 6: Commit**

```bash
git add scripts/stage0_collect.py tests/test_stage0_collect_cli.py
git commit -m "$(cat <<'EOF'
feat(cli): --task flag in stage0_collect.py (PushCube|PickCube)

Adds --task argparse choice backed by task_registry, with PushCube-v1
as the backward-compat default. Episode_id prefix and fake-runner
factory now come from the registry, so adding a new task is a one-row
addition there. Adds tests/test_stage0_collect_cli.py with snapshot
byte-equality for both tasks via the script entry point.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Task-aware report title in `eval.py`

**Files:**
- Modify: `babysteps/eval.py`
- Modify: `tests/test_eval.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_eval.py` (or modify `test_write_report_creates_md_and_json` in-place):

```python
def test_write_report_title_reflects_task_from_records(tmp_path: Path):
    """The report title should say PickCube when records carry task=PickCube-v1."""
    rec = _record(episode_id="x", initial_success=False, retry_success=True)
    # Override the task field — _record() defaults to PushCube-v1.
    from dataclasses import replace
    rec = replace(rec, task="PickCube-v1")
    metrics = compute_metrics([rec])
    write_report(metrics, tmp_path)
    md = (tmp_path / "report.md").read_text()
    assert "PickCube" in md, f"report.md title missing 'PickCube':\n{md[:200]}"
    assert "PushCube" not in md, (
        f"report.md title leaked PushCube despite PickCube records:\n{md[:200]}"
    )


def test_write_report_title_pushcube_default(tmp_path: Path):
    """Backward-compat: PushCube records still yield a PushCube title."""
    rec = _record(episode_id="x", initial_success=False, retry_success=True)
    metrics = compute_metrics([rec])
    write_report(metrics, tmp_path)
    md = (tmp_path / "report.md").read_text()
    assert "PushCube" in md
```

Also update the existing assertion that hardcodes the title:

```python
def test_write_report_creates_md_and_json(tmp_path: Path):
    records = [_record(episode_id="x", initial_success=False, retry_success=True)]
    metrics = compute_metrics(records)
    write_report(metrics, tmp_path)
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "report.json").exists()
    report_json = json.loads((tmp_path / "report.json").read_text())
    assert report_json["delta_pp"] == pytest.approx(100.0)
    md = (tmp_path / "report.md").read_text()
    assert "BABYSTEPS Stage 0" in md
    assert "PASS" in md or "passed_acceptance" in md
```

(The existing assertions stay; the title check moves to the new tests above.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_eval.py::test_write_report_title_reflects_task_from_records -v`

Expected: FAIL — current `_markdown_report` hardcodes "PushCube" in the title.

- [ ] **Step 3: Modify `babysteps/eval.py`**

In `compute_metrics`, capture the task from the first record (default to "Unknown") and include it in the returned dict:

Find:
```python
    records_list = list(records)
    n_total = len(records_list)
```

Replace with:
```python
    records_list = list(records)
    n_total = len(records_list)
    # All records in one report share a task; if mixed, the first wins
    # and the others are silently included (the summarizer only labels
    # the title, not the per-record diagnostics). Default keeps
    # backward-compat with old fixtures that built records by hand.
    task_id = records_list[0].task if records_list else "PushCube-v1"
```

At the end of the return dict (just before `}`), add:
```python
        "task": task_id,
```

In `_markdown_report`, find:
```python
    return (
        "# BABYSTEPS Stage 0 — PushCube Blocked-Approach Report\n\n"
        f"Acceptance: **{pass_str}** (delta_pp >= "
        f"{metrics['acceptance_threshold_pp']})\n\n"
```

Replace with:
```python
    # Title comes from the task id minus the "-v1" suffix; falls back to
    # "PushCube" so old metrics dicts without a "task" key still render.
    task_label = str(metrics.get("task", "PushCube-v1")).split("-v")[0]
    return (
        f"# BABYSTEPS Stage 0 — {task_label} Report\n\n"
        f"Acceptance: **{pass_str}** (delta_pp >= "
        f"{metrics['acceptance_threshold_pp']})\n\n"
```

- [ ] **Step 4: Run new tests**

Run: `python -m pytest tests/test_eval.py -v`

Expected: all eval tests PASS (gain 2).

- [ ] **Step 5: Run full suite — snapshots must still hold**

Run: `python -m pytest tests/ -q 2>&1 | tail -10`

Expected: all PASS. (Note: samples.jsonl is unaffected by report.md, so snapshot tests stay green.)

- [ ] **Step 6: Commit**

```bash
git add babysteps/eval.py tests/test_eval.py
git commit -m "$(cat <<'EOF'
feat(eval): task-aware report title from records[0].task

compute_metrics now writes a 'task' field into its output dict; the
markdown title reads it (default 'PushCube-v1' so callers that build
metrics dicts by hand still render). Necessary for PickCube reports
to not mis-label as PushCube.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Verify `stage0_summarize.py` already works for PickCube

**Files:**
- Modify (if needed): `scripts/stage0_summarize.py`

`stage0_summarize.py` reads `EpisodeRecord` lines from a jsonl, calls `compute_metrics` and `write_report`. After Task 3, `compute_metrics` reads the task from records, so the title auto-reflects PickCube. No flag changes should be needed.

- [ ] **Step 1: Smoke-run the summarizer against the PickCube snapshot**

```bash
python scripts/stage0_summarize.py \
  --samples tests/snapshots/pickcube_samples_seeds_0_4.jsonl \
  --out_dir /tmp/babysteps-pickcube-summarize-check
```

Expected: prints a metrics dict; `/tmp/.../report.md` exists and its title says "PickCube".

- [ ] **Step 2: Inspect the produced report.md**

Run: `head -3 /tmp/babysteps-pickcube-summarize-check/report.md`

Expected first line: `# BABYSTEPS Stage 0 — PickCube Report`

- [ ] **Step 3: Smoke-run for PushCube too — title must say PushCube**

```bash
python scripts/stage0_summarize.py \
  --samples tests/snapshots/pushcube_samples_seeds_0_4.jsonl \
  --out_dir /tmp/babysteps-pushcube-summarize-check
head -3 /tmp/babysteps-pushcube-summarize-check/report.md
```

Expected first line: `# BABYSTEPS Stage 0 — PushCube Report`

- [ ] **Step 4: No commit yet** — nothing changed in this task. If Steps 2/3 fail, the issue is in `eval.py` from Task 3; fix and re-test.

---

## Task 5: Acceptance gate test — `delta_pp >= 10` for PickCube via fake env

**Files:**
- Test: `tests/test_pickcube_delta_pp.py` (new)

This codifies B's gate item 5 as a fast test that runs the full PickCube loop end-to-end via the fake runner and asserts the report-level acceptance metric.

- [ ] **Step 1: Write the test**

Create `tests/test_pickcube_delta_pp.py`:

```python
"""Acceptance-gate test for Sub-project B (PickCube).

Drives `stage0_collect.main` end-to-end with --fake-env --task PickCube-v1
and asserts the produced report.json shows delta_pp >= 10 (the
BABYSTEPS Stage-0 acceptance bar from Pick4Pass M-BABY-1).

This is the fake-env analogue of the spec's Section 3 acceptance item
5 ('the report.md summarizer reports delta_pp >= 10 between revised-
retry success rate and initial-attempt success rate on PickCube').
The fake runner is deterministic and the FAILURE_TO_FACTOR / revision
pipeline is the same code-path the real runner uses, so this test
proves the orchestration meets the bar; the real-sim version is the
GPU spot-check (manual, in CLAUDE.md)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))


@pytest.fixture
def collect_main():
    import importlib
    if "stage0_collect" in sys.modules:
        del sys.modules["stage0_collect"]
    return importlib.import_module("stage0_collect").main


def test_pickcube_fake_env_meets_delta_pp_gate(tmp_path: Path, collect_main):
    out_dir = tmp_path / "out"
    rc = collect_main([
        "--task", "PickCube-v1",
        "--fake-env",
        "--out_dir", str(out_dir),
        "--n_episodes", "5",
        "--seed_start", "0",
    ])
    # rc==0 means the script's own acceptance check passed; we re-assert
    # independently below to surface the actual numbers in pytest output.
    report = json.loads((out_dir / "report.json").read_text())
    assert report["delta_pp"] >= 10.0, (
        f"PickCube fake-env delta_pp = {report['delta_pp']:.1f} "
        f"(threshold 10.0). Initial rate {report['initial_attempt_success_rate']:.2f}, "
        f"retry rate {report['retry_success_rate']:.2f}, n_with_revision="
        f"{report['n_with_revision']}, n_retry_success={report['n_retry_success']}."
    )
    assert report["passed_acceptance"] is True
    assert rc == 0
```

- [ ] **Step 2: Run the test**

Run: `python -m pytest tests/test_pickcube_delta_pp.py -v`

Expected: PASS. (Initial PickCube intent matches the demo's contact_region, executor blocks that face → grasp_slip → revise to orthogonal face → success. 5 of 5 should recover; delta_pp should be 100.)

If this fails with delta_pp < 10, the controlled-failure / revision pipeline isn't producing the expected lift — STOP and inspect the per-episode metrics rather than tweaking the threshold.

- [ ] **Step 3: Quick sanity — same test for PushCube to confirm the bar still holds**

Add this companion to `tests/test_pickcube_delta_pp.py`:

```python
def test_pushcube_fake_env_meets_delta_pp_gate(tmp_path: Path, collect_main):
    """Regression: PushCube must also pass the gate via the same CLI path."""
    out_dir = tmp_path / "out"
    rc = collect_main([
        "--task", "PushCube-v1",
        "--fake-env",
        "--out_dir", str(out_dir),
        "--n_episodes", "5",
        "--seed_start", "0",
    ])
    report = json.loads((out_dir / "report.json").read_text())
    assert report["delta_pp"] >= 10.0
    assert report["passed_acceptance"] is True
    assert rc == 0
```

Run: `python -m pytest tests/test_pickcube_delta_pp.py -v`

Expected: both tests PASS.

- [ ] **Step 4: Full suite**

Run: `python -m pytest tests/ -q 2>&1 | tail -10`

Expected: all PASS (gain 2).

- [ ] **Step 5: Commit**

```bash
git add tests/test_pickcube_delta_pp.py
git commit -m "$(cat <<'EOF'
test(b): acceptance-gate test — PickCube delta_pp >= 10 via fake env

Codifies Sub-project B's spec §3 item 5 ('report.md summarizer reports
delta_pp >= 10') as a fast deterministic test that drives the full CLI
end-to-end (--task PickCube-v1 --fake-env, 5 seeds). Adds a companion
PushCube test to guard against regressions in the shared pipeline.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: `babysteps/render/common.py` — extract shared render utilities

**Files:**
- Create: `babysteps/render/__init__.py`
- Create: `babysteps/render/common.py`

The current `scripts/render_stage0_maniskill.py` has shared utilities (frame readers, PD action, annotate, MP4 writer) tangled with PushCube-specific phase logic. This task moves the shared bits into `babysteps/render/common.py` so both task render modules can import them, then sets up the empty registry.

- [ ] **Step 1: Create the package**

Create `babysteps/render/__init__.py`:

```python
"""Per-task render dispatch for Stage-0 MP4 generation.

Each task contributes one `render_episode(env, adapter, seed, fps)`
function that returns three lists of RGB frames (one per BABYSTEPS
phase: demo, blocked attempt, retry) plus title metadata for the
on-frame banners. `scripts/render_stage0_maniskill.py` is a thin
dispatcher over RENDER_REGISTRY.

Render modules are sim-free in their tested surface (waypoint
selection + frame counting) and pull in mani_skill only when invoked
end-to-end."""
from __future__ import annotations

from typing import Callable

from babysteps.envs.task_adapter import BaseTaskAdapter


# A render_episode_fn returns:
#   ({phase_name → list[rgb_frame]}, {phase_name → (title, subtitle)})
# where phase_names are "demo", "attempt_blocked", "retry".
RenderEpisodeFn = Callable[..., tuple[dict, dict]]


def _pushcube_render() -> RenderEpisodeFn:
    from babysteps.render.pushcube import render_episode
    return render_episode


def _pickcube_render() -> RenderEpisodeFn:
    from babysteps.render.pickcube import render_episode
    return render_episode


# Lazy: each entry's import happens on first access via RENDER_REGISTRY[task_id]().
# This keeps importing babysteps.render cheap when only one task is needed.
RENDER_REGISTRY: dict[str, Callable[[], RenderEpisodeFn]] = {
    "PushCube-v1": _pushcube_render,
    "PickCube-v1": _pickcube_render,
}


def get_render_fn(task_id: str) -> RenderEpisodeFn:
    if task_id not in RENDER_REGISTRY:
        known = sorted(RENDER_REGISTRY.keys())
        raise KeyError(f"no render module for task {task_id!r}; known: {known}")
    return RENDER_REGISTRY[task_id]()
```

- [ ] **Step 2: Create `babysteps/render/common.py`**

```python
"""Shared utilities for per-task render modules.

Pulled out of scripts/render_stage0_maniskill.py so both pushcube.py
and pickcube.py (and future stackcube.py / drawer.py) can reuse them
without circular dependency on the script."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

# Phase-control constants — match the PD calibration of the env_runners.
POS_SCALE: float = 0.1
PHASE_TOL_M: float = 0.015
MAX_CONTROL_STEPS: int = 400   # superset of both runners' caps


def to_np(x):
    """Convert a possibly-batched torch/cuda tensor to a flat numpy view."""
    arr = x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)
    return arr[0] if arr.ndim == 2 else arr


def raw_to_xyzw(raw_pose) -> np.ndarray:
    """ManiSkill's pose comes as [x, y, z, qw, qx, qy, qz]; we want xyzw."""
    raw = np.asarray(raw_pose, dtype=np.float64)
    return np.concatenate([raw[0:3], raw[4:7], raw[3:4]])


def read_obs(obs) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """(tcp_xyzw, cube_xy, goal_xy, cube_z) from a PushCube/PickCube obs."""
    tcp = raw_to_xyzw(to_np(obs["extra"]["tcp_pose"]))
    cube_full = to_np(obs["extra"]["obj_pose"])
    cube_xy = cube_full[0:2].astype(np.float64)
    cube_z = float(cube_full[2])
    goal_xy = to_np(obs["extra"]["goal_pos"])[0:2].astype(np.float64)
    return tcp, cube_xy, goal_xy, cube_z


def prop_action(
    tcp_xyzw: np.ndarray, target_xyz: np.ndarray, gripper_cmd: float = -1.0,
) -> np.ndarray:
    """Proportional 7-dim action toward target_xyz with explicit gripper cmd.
    Default gripper_cmd=-1 (closed) matches PushSkill's behavior."""
    pos_err = target_xyz - tcp_xyzw[0:3]
    action = np.zeros(7, dtype=np.float32)
    action[0:3] = np.clip(pos_err / POS_SCALE, -1.0, 1.0).astype(np.float32)
    action[6] = np.float32(gripper_cmd)
    return action


def render_frame(env) -> np.ndarray:
    """One (H, W, 3) uint8 RGB frame from env.render()."""
    f = env.render()
    if hasattr(f, "cpu"):
        f = f.cpu().numpy()
    f = np.asarray(f)
    if f.ndim == 4:
        f = f[0]
    if f.dtype != np.uint8:
        f = (255.0 * np.clip(f, 0.0, 1.0)).astype(np.uint8) if f.max() <= 1.0 \
            else f.astype(np.uint8)
    return f


def annotate_frame(
    frame: np.ndarray, title: str, subtitle: str = "",
) -> np.ndarray:
    """Add a black banner with title (and optional subtitle) above frame."""
    from PIL import Image, ImageDraw, ImageFont
    img = Image.fromarray(frame)
    W, H = img.size
    banner_h = 60 if subtitle else 36
    canvas = Image.new("RGB", (W, H + banner_h), (16, 16, 16))
    canvas.paste(img, (0, banner_h))
    draw = ImageDraw.Draw(canvas)
    try:
        font_big = ImageFont.truetype(
            "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf", 16,
        )
        font_sm = ImageFont.truetype(
            "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf", 12,
        )
    except Exception:
        font_big = ImageFont.load_default()
        font_sm = ImageFont.load_default()
    draw.text((10, 6), title, fill=(255, 255, 255), font=font_big)
    if subtitle:
        draw.text((10, 30), subtitle, fill=(200, 200, 200), font=font_sm)
    return np.asarray(canvas)


def save_mp4(frames: Iterable[np.ndarray], out_path: Path, fps: int) -> None:
    """Write `frames` to `out_path` as H.264 MP4."""
    import imageio.v2 as imageio
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(out_path), fps=fps, codec="libx264", quality=8,
        macro_block_size=1,
    )
    for fr in frames:
        writer.append_data(fr)
    writer.close()
```

- [ ] **Step 3: Sanity import test**

Run: `python -c "from babysteps.render import get_render_fn, RENDER_REGISTRY; print(sorted(RENDER_REGISTRY))"`

Expected output:
```
['PickCube-v1', 'PushCube-v1']
```

(The lazy factories are NOT called yet, so missing `babysteps.render.pushcube` / `pickcube` modules don't break this import.)

- [ ] **Step 4: Full test suite — no regression expected**

Run: `python -m pytest tests/ -q 2>&1 | tail -10`

Expected: all PASS (no new tests in this task; the per-task render modules and their tests come next).

- [ ] **Step 5: Commit**

```bash
git add babysteps/render/__init__.py babysteps/render/common.py
git commit -m "$(cat <<'EOF'
feat(render): babysteps.render package — shared common.py + registry

Extracts the shared MP4-rendering utilities (frame reading, PD action,
annotation, mp4 writer) from scripts/render_stage0_maniskill.py into
babysteps/render/common.py and stubs RENDER_REGISTRY in __init__.py.
Per-task render modules (pushcube.py, pickcube.py) land in the next
two tasks; the script becomes a thin dispatcher after that.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: `babysteps/render/pushcube.py` — move PushCube render logic

**Files:**
- Create: `babysteps/render/pushcube.py`
- Test: `tests/test_render_modules.py` (new — covers both pushcube and pickcube render modules)

Move the PushCube-specific phase logic (`_execute_push`, `_build_waypoints`, and the main() phase orchestration) out of the script and into a render-module function. Behavior of the eventual script is unchanged for `--task PushCube-v1`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_render_modules.py`:

```python
"""Tests for per-task render modules (babysteps.render.{pushcube,pickcube}).

These tests use a stub env that mimics gymnasium's reset/step/render API
with deterministic obs and frames, so we exercise the per-task phase
logic (waypoint dispatch, gripper schedule, frame counts) without
needing ManiSkill or a GPU."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest


# ---------- Stub env --------------------------------------------------- #


@dataclass
class _StubObs:
    """A dict-like obs with extra.{tcp_pose, obj_pose, goal_pos}."""
    tcp: np.ndarray
    cube: np.ndarray
    goal: np.ndarray

    def __getitem__(self, key: str):
        if key == "extra":
            # Match ManiSkill's pose convention: [x,y,z,qw,qx,qy,qz].
            tcp_raw = np.concatenate([self.tcp[0:3], np.array([1.0]),
                                      self.tcp[3:6]])
            cube_full = np.array([self.cube[0], self.cube[1], 0.02])
            goal_full = np.array([self.goal[0], self.goal[1], 0.02])
            return {"tcp_pose": tcp_raw, "obj_pose": cube_full,
                    "goal_pos": goal_full}
        raise KeyError(key)


class _StubEnv:
    """Drop-in stand-in for gym.make('PushCube-v1') / 'PickCube-v1'.

    reset(seed) places the cube at the origin and the goal at +x. The TCP
    'tracks' the action target deterministically each step (just integrates
    the action) so phase transitions happen predictably."""

    def __init__(self) -> None:
        self.tcp = np.array([0.0, 0.0, 0.25, 0.0, 0.0, 0.0], dtype=np.float64)
        self.cube = np.array([0.0, 0.0], dtype=np.float64)
        self.goal = np.array([0.12, 0.0], dtype=np.float64)
        self._step_count = 0

    def reset(self, seed: int = 0):
        self.tcp = np.array([0.0, 0.0, 0.25, 0.0, 0.0, 0.0], dtype=np.float64)
        self.cube = np.array([0.0, 0.0], dtype=np.float64)
        self.goal = np.array([0.12, 0.0], dtype=np.float64)
        self._step_count = 0
        return _StubObs(self.tcp, self.cube, self.goal), {}

    def step(self, action):
        # Integrate xyz error directly so target is reached in ~10 steps per phase.
        self.tcp[0:3] = self.tcp[0:3] + 0.02 * np.asarray(action[0:3])
        self._step_count += 1
        return (
            _StubObs(self.tcp, self.cube, self.goal),
            0.0, False, False,
            {"success": False},
        )

    def render(self):
        # Return a tiny deterministic RGB frame (8x8x3 uint8).
        return (np.ones((8, 8, 3), dtype=np.uint8) * (self._step_count % 256))

    def close(self):
        pass


# ---------- PushCube render tests -------------------------------------- #


def test_pushcube_render_episode_emits_three_phase_frames():
    """render_episode returns frames dict with demo/attempt_blocked/retry."""
    from babysteps.render.pushcube import render_episode
    from babysteps.envs.pushcube_adapter import PushCubeAdapter

    env = _StubEnv()
    adapter = PushCubeAdapter()
    frames, titles = render_episode(env, adapter, seed=0, fps=4)

    assert set(frames.keys()) == {"demo", "attempt_blocked", "retry"}
    assert set(titles.keys()) == {"demo", "attempt_blocked", "retry"}
    # All three phases must produce at least one frame.
    assert len(frames["demo"]) >= 1
    assert len(frames["attempt_blocked"]) >= 1  # PushCube: held-still loop
    assert len(frames["retry"]) >= 1
    # PushCube's attempt_blocked is a held-still synthesis (planner_failed).
    # Confirm the held frames don't trigger env.step — same frame N times.
    held = frames["attempt_blocked"]
    assert all(np.array_equal(held[0], f) for f in held)


def test_pushcube_render_titles_contain_phase_label():
    from babysteps.render.pushcube import render_episode
    from babysteps.envs.pushcube_adapter import PushCubeAdapter
    env = _StubEnv()
    _, titles = render_episode(env, PushCubeAdapter(), seed=0, fps=4)
    assert "phase 1/3" in titles["demo"][0]
    assert "phase 2/3" in titles["attempt_blocked"][0]
    assert "phase 3/3" in titles["retry"][0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_render_modules.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'babysteps.render.pushcube'`.

- [ ] **Step 3: Create `babysteps/render/pushcube.py`**

```python
"""PushCube-v1 render_episode — three phases for the Stage-0 MP4 set.

Phase 1 (demo): execute the oracle's correct intent in a fresh seed, capture
all frames.
Phase 2 (attempt_blocked): the demo's approach is blocked; the skill
compiler returns None → planner_failed. The render captures a 'held still'
loop (fps * 2 copies of one initial frame) to convey 'nothing happened'.
Phase 3 (retry): the revised intent (orthogonal approach) succeeds.

Identical semantics to the pre-extraction `_execute_push` /
`_build_waypoints` / main() flow in scripts/render_stage0_maniskill.py."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from babysteps.envs.task_adapter import BaseTaskAdapter
from babysteps.failure import attribute_failure
from babysteps.revision import revise_intent
from babysteps.render.common import (
    MAX_CONTROL_STEPS,
    PHASE_TOL_M,
    prop_action,
    read_obs,
    render_frame,
)
from babysteps.schemas import AttemptResult, DemoEvidence, SceneState


def _build_waypoints(scene: SceneState, intent) -> np.ndarray:
    """4-waypoint PushCube trajectory (approach, pre-contact high, pre-contact
    low, push-end). Identical to the inline _build_waypoints in
    scripts/render_stage0_maniskill.py before this extraction — see
    babysteps.skills.push for the canonical version used by the env_runner."""
    from babysteps.envs.scene import approach_to_unit, face_to_push_unit
    cube_xy = np.asarray(scene.cube_xy, dtype=np.float64)
    goal_xy = np.asarray(scene.goal_xy, dtype=np.float64)
    tcp = np.asarray(scene.tcp_start_pose, dtype=np.float64)
    travel_z = float(tcp[2])
    push_z = float(scene.cube_z)
    push_unit = face_to_push_unit(intent.contact_region)
    approach_unit = approach_to_unit(intent.approach_direction)
    standoff = 0.02 + 0.005
    approach_standoff = 0.10
    pre_contact_xy = cube_xy - push_unit * standoff
    approach_xy = cube_xy + approach_unit * approach_standoff
    cube_to_goal = float(np.linalg.norm(goal_xy - cube_xy))
    push_travel = min(0.6 * cube_to_goal, 0.15)
    push_end_xy = cube_xy + push_unit * push_travel

    wp = np.zeros((4, 7), dtype=np.float64)
    wp[0, 0:2] = approach_xy
    wp[0, 2] = travel_z
    wp[1, 0:2] = pre_contact_xy
    wp[1, 2] = travel_z
    wp[2, 0:2] = pre_contact_xy
    wp[2, 2] = push_z
    wp[3, 0:2] = push_end_xy
    wp[3, 2] = push_z
    wp[:, 3:7] = tcp[3:7]
    return wp


def _execute_push(env, waypoints, frames: list, *, seed: int) -> dict:
    """Step through waypoints capturing one frame per step. Re-resets the env
    at the start so demo / attempt / retry all begin from the same scene."""
    obs, _ = env.reset(seed=int(seed))
    targets = [np.asarray(wp[0:3], dtype=np.float64) for wp in waypoints]
    phase_idx = 0
    success = False

    frames.append(render_frame(env))
    for _ in range(MAX_CONTROL_STEPS):
        tcp, cube_xy, _, _ = read_obs(obs)
        target = targets[phase_idx]
        if np.linalg.norm(target - tcp[0:3]) < PHASE_TOL_M:
            phase_idx += 1
            if phase_idx >= len(targets):
                break
            target = targets[phase_idx]
        action = prop_action(tcp, target, gripper_cmd=-1.0)
        obs, _r, term, trunc, info = env.step(action)
        frames.append(render_frame(env))
        term_b = bool(term) if not hasattr(term, "cpu") \
            else bool(term.cpu().numpy().item())
        trunc_b = bool(trunc) if not hasattr(trunc, "cpu") \
            else bool(trunc.cpu().numpy().item())
        succ = info.get("success", False) if hasattr(info, "get") else False
        success = bool(succ) if not hasattr(succ, "cpu") \
            else bool(succ.cpu().numpy().item())
        if success or term_b or trunc_b:
            break

    tcp, final_cube_xy, _, _ = read_obs(obs)
    return {
        "final_obj_xy": (float(final_cube_xy[0]), float(final_cube_xy[1])),
        "success": bool(success),
    }


def render_episode(
    env, adapter: BaseTaskAdapter, seed: int, fps: int,
) -> tuple[dict, dict]:
    """Run the three-phase BABYSTEPS demo for PushCube and return per-phase
    frame lists and title metadata.

    Returns:
        frames: {"demo": [...], "attempt_blocked": [...], "retry": [...]}
        titles: {"demo": (title, subtitle), ...}
    """
    short_id = f"seed {seed:04d}"

    # === Phase 1 — DEMO PROXY ===
    obs, _ = env.reset(seed=seed)
    tcp_xyzw, cube_xy0, goal_xy, cube_z = read_obs(obs)
    scene = SceneState(
        cube_xy=(float(cube_xy0[0]), float(cube_xy0[1])),
        cube_z=cube_z,
        goal_xy=(float(goal_xy[0]), float(goal_xy[1])),
        tcp_start_pose=tuple(float(v) for v in tcp_xyzw),  # type: ignore[arg-type]
        blocked_sides=(),
    )
    correct_intent = adapter.oracle_correct_intent(scene)
    wp_demo = _build_waypoints(scene, correct_intent)
    demo_frames: list = []
    out_demo = _execute_push(env, wp_demo, demo_frames, seed=seed)

    demo_evidence = DemoEvidence(
        camera="third_person",
        demonstrator_type="proxy_oracle",
        object_trajectory=(
            (float(cube_xy0[0]), float(cube_xy0[1])),
            out_demo["final_obj_xy"],
        ),
        contact_region_label=correct_intent.contact_region,
        final_state=correct_intent.goal_state,
        rgbd_video_path=None,
    )
    initial_intent = adapter.scripted_demo_to_intent(demo_evidence)
    scene_exec = replace(
        scene, blocked_sides=adapter.default_blocked_factory(initial_intent),
    )

    # === Phase 2 — ATTEMPT 1 (planner_failed, held still) ===
    obs, _ = env.reset(seed=seed)
    attempt1_frames = [render_frame(env)] * (fps * 2)

    # === Phase 3 — RETRY with revised approach ===
    fp = adapter.build_failure_packet(
        initial_intent,
        AttemptResult(
            initial_obj_xy=scene.cube_xy, final_obj_xy=scene.cube_xy,
            goal_xy=scene.goal_xy,
            reached_contact=False, object_moved=False,
            planner_failed=True, collision=False, grasp_slip=False,
            rollout_log_path=None, success=False,
        ),
        scene_exec,
    )
    attribution = adapter.attribute_failure(fp)
    revised_intent, _rev = adapter.revise_intent(
        initial_intent, attribution, scene_exec,
    )
    wp_retry = _build_waypoints(scene_exec, revised_intent)
    retry_frames: list = []
    out_retry = _execute_push(env, wp_retry, retry_frames, seed=seed)

    demo_title = (
        f"{short_id}  phase 1/3: demo proxy",
        f"contact_region={correct_intent.contact_region}, "
        f"approach={correct_intent.approach_direction}",
    )
    a1_title = (
        f"{short_id}  phase 2/3: approach_blocked",
        f"approach_direction={initial_intent.approach_direction} "
        f"is blocked → planner_failed",
    )
    retry_title = (
        f"{short_id}  phase 3/3: retry (success={out_retry['success']})",
        f"approach_substitution: "
        f"{initial_intent.approach_direction} → "
        f"{revised_intent.approach_direction}",
    )
    return (
        {"demo": demo_frames,
         "attempt_blocked": attempt1_frames,
         "retry": retry_frames},
        {"demo": demo_title,
         "attempt_blocked": a1_title,
         "retry": retry_title},
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_render_modules.py -v -k pushcube`

Expected: both `test_pushcube_*` tests PASS.

- [ ] **Step 5: Full suite**

Run: `python -m pytest tests/ -q 2>&1 | tail -10`

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add babysteps/render/pushcube.py tests/test_render_modules.py
git commit -m "$(cat <<'EOF'
feat(render): babysteps.render.pushcube — extract PushCube render flow

Lifts _execute_push, _build_waypoints, and the three-phase orchestration
from scripts/render_stage0_maniskill.py into a per-task render module
that conforms to RENDER_REGISTRY's render_episode contract. Adds stub-
env unit tests so the dispatch + frame layout are testable without a
GPU. The script will switch to dispatching through RENDER_REGISTRY in a
later task; behavior is currently unchanged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: `babysteps/render/pickcube.py` — PickCube render flow with real phase-2 stepping

**Files:**
- Create: `babysteps/render/pickcube.py`
- Modify: `tests/test_render_modules.py`

PickCube differs from PushCube in phase 2: the controlled failure is `grasp_slip`, detected at execution time. The render must actually step the env so the viewer sees the gripper close on the (blocked-axis) cube and the lift open up and drop the cube — not a held-still loop.

- [ ] **Step 1: Add the failing PickCube tests to `tests/test_render_modules.py`**

Append:

```python
# ---------- PickCube render tests -------------------------------------- #


def test_pickcube_render_episode_emits_three_phase_frames():
    from babysteps.render.pickcube import render_episode
    from babysteps.envs.pickcube_adapter import PickCubeAdapter

    env = _StubEnv()
    adapter = PickCubeAdapter()
    frames, titles = render_episode(env, adapter, seed=0, fps=4)

    assert set(frames.keys()) == {"demo", "attempt_blocked", "retry"}
    assert set(titles.keys()) == {"demo", "attempt_blocked", "retry"}
    # All three phases must step the env (grasp_slip is execution-time).
    assert len(frames["demo"]) >= 2
    assert len(frames["attempt_blocked"]) >= 2
    assert len(frames["retry"]) >= 2


def test_pickcube_render_phase2_actually_steps_env():
    """Unlike PushCube (held still), PickCube must step the env in phase 2
    so the grasp_slip is visible (gripper closes, lifts, releases). Detect
    by checking the stub env's step_count incremented across phase 2 frames."""
    from babysteps.render.pickcube import render_episode
    from babysteps.envs.pickcube_adapter import PickCubeAdapter

    env = _StubEnv()
    frames, _ = render_episode(env, PickCubeAdapter(), seed=0, fps=4)
    held = frames["attempt_blocked"]
    # In the stub env each step bumps the frame intensity, so consecutive
    # frames differ iff env.step was called. PushCube's phase-2 frames are
    # all identical; PickCube's must differ.
    assert not all(np.array_equal(held[0], f) for f in held), (
        "PickCube phase 2 should step the env to surface grasp_slip; "
        "saw all-identical frames (PushCube-style hold)."
    )


def test_pickcube_render_titles_mention_contact_region():
    from babysteps.render.pickcube import render_episode
    from babysteps.envs.pickcube_adapter import PickCubeAdapter
    _, titles = render_episode(_StubEnv(), PickCubeAdapter(), seed=0, fps=4)
    # Demo subtitle should mention contact_region (which face was grasped).
    assert "contact_region" in titles["demo"][1]
    # Retry subtitle should mention contact_substitution.
    assert "contact_substitution" in titles["retry"][1]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_render_modules.py -v -k pickcube`

Expected: FAIL with `ModuleNotFoundError: No module named 'babysteps.render.pickcube'`.

- [ ] **Step 3: Create `babysteps/render/pickcube.py`**

```python
"""PickCube-v1 render_episode — three phases for the Stage-0 MP4 set.

Phase 1 (demo): execute the oracle's correct intent (top-down grasp with
the demonstrated contact_region). Capture frames.
Phase 2 (attempt_blocked): execute the initial intent in the executor
scene where the demonstrated contact_region is in blocked_sides. The
PickCubeEnvRunner's slip mechanism opens the gripper at lift-time, so
the cube falls back — we render those frames; the grasp_slip is
visually obvious in the MP4.
Phase 3 (retry): execute the revised intent (orthogonal contact_region).

Unlike PushCube, phase 2 here is NOT held-still: the failure happens at
execution time, not compile time, so the viewer needs to see the
attempted-then-failed lift."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from babysteps.envs.task_adapter import BaseTaskAdapter
from babysteps.render.common import (
    MAX_CONTROL_STEPS,
    PHASE_TOL_M,
    prop_action,
    read_obs,
    render_frame,
)
from babysteps.schemas import AttemptResult, DemoEvidence, Intent, SceneState
from babysteps.skills.pick import compile_intent_to_pick_skill


_GRIPPER_OPEN = 1.0
_GRIPPER_CLOSED = -1.0


def _execute_pick(
    env, intent: Intent, scene: SceneState, frames: list, *,
    seed: int,
) -> dict:
    """Step the env through PickSkill's 4 waypoints + gripper schedule.

    The slip behavior (gripper open at lift) is keyed off
    `intent.contact_region in scene.blocked_sides`, mirroring
    PickCubeEnvRunner.run."""
    skill = compile_intent_to_pick_skill(intent, scene)
    obs, _ = env.reset(seed=int(seed))
    targets = [np.asarray(wp[0:3], dtype=np.float64) for wp in skill.waypoints]
    slip = intent.contact_region in scene.blocked_sides
    lift_gripper = _GRIPPER_OPEN if slip else _GRIPPER_CLOSED
    phase_gripper = (
        _GRIPPER_OPEN, _GRIPPER_OPEN, _GRIPPER_CLOSED, lift_gripper,
    )

    phase_idx = 0
    success = False
    frames.append(render_frame(env))
    for _ in range(MAX_CONTROL_STEPS):
        tcp, _cube_xy, _, _ = read_obs(obs)
        target = targets[phase_idx]
        if np.linalg.norm(target - tcp[0:3]) < PHASE_TOL_M:
            phase_idx += 1
            if phase_idx >= len(targets):
                break
            target = targets[phase_idx]
        action = prop_action(tcp, target, gripper_cmd=phase_gripper[phase_idx])
        obs, _r, term, trunc, info = env.step(action)
        frames.append(render_frame(env))
        term_b = bool(term) if not hasattr(term, "cpu") \
            else bool(term.cpu().numpy().item())
        trunc_b = bool(trunc) if not hasattr(trunc, "cpu") \
            else bool(trunc.cpu().numpy().item())
        succ = info.get("success", False) if hasattr(info, "get") else False
        success = bool(succ) if not hasattr(succ, "cpu") \
            else bool(succ.cpu().numpy().item())
        if success or term_b or trunc_b:
            break

    return {"success": bool(success and not slip)}


def render_episode(
    env, adapter: BaseTaskAdapter, seed: int, fps: int,
) -> tuple[dict, dict]:
    """Three-phase BABYSTEPS render for PickCube."""
    short_id = f"seed {seed:04d}"

    # === Phase 1 — DEMO PROXY (oracle's correct intent) ===
    obs, _ = env.reset(seed=seed)
    tcp_xyzw, cube_xy0, goal_xy, cube_z = read_obs(obs)
    scene = SceneState(
        cube_xy=(float(cube_xy0[0]), float(cube_xy0[1])),
        cube_z=cube_z,
        goal_xy=(float(goal_xy[0]), float(goal_xy[1])),
        tcp_start_pose=tuple(float(v) for v in tcp_xyzw),  # type: ignore[arg-type]
        blocked_sides=(),
    )
    correct_intent = adapter.oracle_correct_intent(scene)
    demo_frames: list = []
    _ = _execute_pick(env, correct_intent, scene, demo_frames, seed=seed)

    # Build the DemoEvidence the loop would build.
    demo_evidence = DemoEvidence(
        camera="third_person",
        demonstrator_type="proxy_oracle",
        object_trajectory=(
            (float(cube_xy0[0]), float(cube_xy0[1])),
            (float(scene.goal_xy[0]), float(scene.goal_xy[1])),
        ),
        contact_region_label=correct_intent.contact_region,
        final_state=correct_intent.goal_state,
        rgbd_video_path=None,
    )
    initial_intent = adapter.scripted_demo_to_intent(demo_evidence)
    scene_exec = replace(
        scene, blocked_sides=adapter.default_blocked_factory(initial_intent),
    )

    # === Phase 2 — ATTEMPT 1 (grasp_slip, actually executed) ===
    attempt1_frames: list = []
    _ = _execute_pick(env, initial_intent, scene_exec, attempt1_frames, seed=seed)

    # === Phase 3 — RETRY with revised contact_region ===
    fp = adapter.build_failure_packet(
        initial_intent,
        AttemptResult(
            initial_obj_xy=scene.cube_xy, final_obj_xy=scene.cube_xy,
            goal_xy=scene.goal_xy,
            reached_contact=True, object_moved=False,
            planner_failed=False, collision=False, grasp_slip=True,
            rollout_log_path=None, success=False,
        ),
        scene_exec,
    )
    attribution = adapter.attribute_failure(fp)
    revised_intent, _rev = adapter.revise_intent(
        initial_intent, attribution, scene_exec,
    )
    retry_frames: list = []
    out_retry = _execute_pick(
        env, revised_intent, scene_exec, retry_frames, seed=seed,
    )

    demo_title = (
        f"{short_id}  phase 1/3: demo proxy",
        f"contact_region={correct_intent.contact_region}, "
        f"approach={correct_intent.approach_direction}",
    )
    a1_title = (
        f"{short_id}  phase 2/3: grasp_slip",
        f"contact_region={initial_intent.contact_region} "
        f"is slip-prone → lift opens, cube drops",
    )
    retry_title = (
        f"{short_id}  phase 3/3: retry (success={out_retry['success']})",
        f"contact_substitution: "
        f"{initial_intent.contact_region} → {revised_intent.contact_region}",
    )

    # Pad attempt1 with a tail of the last frame so the slip is on-screen for
    # at least fps*1 frames (otherwise the lift can finish too fast to see).
    if attempt1_frames:
        tail = [attempt1_frames[-1]] * fps
        attempt1_frames = attempt1_frames + tail

    return (
        {"demo": demo_frames,
         "attempt_blocked": attempt1_frames,
         "retry": retry_frames},
        {"demo": demo_title,
         "attempt_blocked": a1_title,
         "retry": retry_title},
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_render_modules.py -v`

Expected: all 5 tests PASS (2 pushcube + 3 pickcube).

- [ ] **Step 5: Full suite**

Run: `python -m pytest tests/ -q 2>&1 | tail -10`

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add babysteps/render/pickcube.py tests/test_render_modules.py
git commit -m "$(cat <<'EOF'
feat(render): babysteps.render.pickcube — three-phase PickCube renderer

Phase 1 demo top-grasps and lifts; phase 2 actually executes the
grasp_slip lift (controlled by blocked_sides) so the failure is
visible in the MP4 — unlike PushCube's held-still phase 2 (which
reflects planner_failed). Phase 3 retries with the revised
contact_region.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Rewrite `scripts/render_stage0_maniskill.py` as a thin dispatcher

**Files:**
- Modify: `scripts/render_stage0_maniskill.py`

- [ ] **Step 1: Replace the script with the dispatcher version**

Replace the entire contents of `scripts/render_stage0_maniskill.py` with:

```python
"""Render Stage-0 episodes as ManiSkill RGB MP4s — multi-task dispatcher.

For each seed, runs the full BABYSTEPS loop (demo proxy → blocked attempt
→ revised retry) for the chosen task, capturing `env.render()` frames per
phase, and writes one MP4 per phase to `<out_dir>/videos_maniskill/`.

Tasks:
  --task PushCube-v1 (default) — phase 2 is held-still (planner_failed).
  --task PickCube-v1            — phase 2 actually executes the failing
                                  lift so the grasp_slip is visible.

This script needs Vulkan. On the Gilbreth login node it works via Mesa's
software Vulkan rasterizer (lavapipe) — slow but real. On a GPU compute
node it uses the NVIDIA Vulkan ICD and runs much faster.

Per-task render flows live in babysteps/render/{pushcube,pickcube}.py;
this script is just the orchestration over a `--task` choice.

Recommended invocation on a GPU node (PickCube):

    cd /scratch/gilbreth/wang4433/babysteps
    conda activate handover
    OUT_DIR=/scratch/gilbreth/wang4433/render_pickcube
    LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" \\
      python scripts/render_stage0_maniskill.py \\
        --task PickCube-v1 --out_dir "$OUT_DIR" \\
        --n_episodes 2 --seed_start 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the project root importable without `pip install -e .`.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from babysteps.envs.task_registry import TASK_REGISTRY, get_task_entry  # noqa: E402
from babysteps.render import get_render_fn  # noqa: E402
from babysteps.render.common import annotate_frame, save_mp4  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--task", type=str, default="PushCube-v1",
        choices=sorted(TASK_REGISTRY.keys()),
    )
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--n_episodes", type=int, default=5)
    p.add_argument("--seed_start", type=int, default=0)
    p.add_argument("--fps", type=int, default=20)
    args = p.parse_args(argv)

    try:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401 — registers PushCube-v1 / PickCube-v1
    except Exception as exc:
        print(
            f"ManiSkill import failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2

    entry = get_task_entry(args.task)
    render_fn = get_render_fn(args.task)
    adapter = entry.adapter_cls()

    videos_dir = args.out_dir / "videos_maniskill"
    videos_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(
        args.task,
        obs_mode="state_dict",
        control_mode="pd_ee_delta_pose",
        sim_backend="cpu",
        render_mode="rgb_array",
    )

    try:
        for i in range(args.n_episodes):
            seed = args.seed_start + i
            episode_id = f"{entry.episode_id_prefix}_seed_{seed:04d}"
            print(f"[{i + 1}/{args.n_episodes}] {episode_id}", flush=True)

            frames, titles = render_fn(env, adapter, seed=seed, fps=args.fps)

            for phase_name, mp4_suffix in [
                ("demo",            "1_demo"),
                ("attempt_blocked", "2_attempt_blocked"),
                ("retry",           "3_retry"),
            ]:
                title, subtitle = titles[phase_name]
                annotated = [
                    annotate_frame(fr, title, subtitle)
                    for fr in frames[phase_name]
                ]
                out_path = videos_dir / f"{episode_id}__{mp4_suffix}.mp4"
                save_mp4(annotated, out_path, args.fps)
                kb = out_path.stat().st_size // 1024
                print(f"   wrote {out_path.name}  ({kb} KB)")
    finally:
        env.close()
        adapter.close()

    print(f"\nDone. MP4s in {videos_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Smoke test — script must --help cleanly with the new flag**

Run: `python scripts/render_stage0_maniskill.py --help 2>&1 | head -30`

Expected: shows `--task {PickCube-v1,PushCube-v1}` in the usage block. No import errors.

- [ ] **Step 3: Smoke test — script imports cleanly without invoking ManiSkill**

Run: `python -c "import importlib, sys; sys.path.insert(0, 'scripts'); m = importlib.import_module('render_stage0_maniskill'); print('main:', m.main.__name__)"`

Expected: `main: main` printed. (Lazy mani_skill import happens inside main(), not at import time.)

- [ ] **Step 4: Full test suite**

Run: `python -m pytest tests/ -q 2>&1 | tail -10`

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/render_stage0_maniskill.py
git commit -m "$(cat <<'EOF'
refactor(scripts): render_stage0_maniskill.py — thin --task dispatcher

Replaces the inline PushCube-only orchestration with a dispatcher that
reads babysteps.render.RENDER_REGISTRY and babysteps.envs.task_registry
to pick the right per-task render flow. PushCube behavior unchanged;
adds PickCube support per Sub-project B acceptance gate.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: GPU spot-check command + CLAUDE.md update

**Files:**
- Modify: `CLAUDE.md`

The remaining gate item (Section 3 item 4 — "GPU visual spot-check: render_stage0_maniskill.py --task PickCube-v1 --n_episodes 2 produces three MP4s per episode") is a manual / sbatch step. Update CLAUDE.md so future cold-start sessions know the exact command.

- [ ] **Step 1: Replace the GPU block in CLAUDE.md**

Find the existing PushCube GPU command block in `CLAUDE.md` (currently starts with `For gpu tasks: we can connect node:`) and update it to include both tasks. Replace from `For gpu tasks:` through the closing single-quote of the bash example with:

````markdown
For GPU tasks (Vulkan + NVIDIA ICD), connect a compute node and render
both tasks' three-phase MP4s. Same script, different --task value:

```bash
# PushCube (Sub-project A — approach_blocked)
srun --account=rpaleja --partition=a100-40gb --gres=gpu:1 --mem=115G --time=00:20:00 bash -lc '
  cd /scratch/gilbreth/wang4433/babysteps &&
  source /apps/external/conda/2025.09/etc/profile.d/conda.sh &&
  conda activate handover &&
  OUT_DIR=/scratch/gilbreth/wang4433/render_pushcube &&
  LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" \
  python scripts/render_stage0_maniskill.py \
    --task PushCube-v1 \
    --out_dir "$OUT_DIR" \
    --n_episodes 2 \
    --seed_start 0 &&
  ls -lh "$OUT_DIR/videos_maniskill"
'

# PickCube (Sub-project B — grasp_slip; closes B's acceptance gate item 4)
srun --account=rpaleja --partition=a100-40gb --gres=gpu:1 --mem=115G --time=00:20:00 bash -lc '
  cd /scratch/gilbreth/wang4433/babysteps &&
  source /apps/external/conda/2025.09/etc/profile.d/conda.sh &&
  conda activate handover &&
  OUT_DIR=/scratch/gilbreth/wang4433/render_pickcube &&
  LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" \
  python scripts/render_stage0_maniskill.py \
    --task PickCube-v1 \
    --out_dir "$OUT_DIR" \
    --n_episodes 2 \
    --seed_start 0 &&
  ls -lh "$OUT_DIR/videos_maniskill"
'
```

Expected output per task: 2 episodes × 3 MP4s = 6 files in
videos_maniskill/, named `<task_prefix>_seed_NNNN__{1_demo,2_attempt_blocked,3_retry}.mp4`.
````

(Replace exactly that block; the rest of CLAUDE.md stays intact. The list of
modules / scripts / tests under it now needs `babysteps/envs/task_registry.py`
and `babysteps/render/` added — update the relevant lines.)

- [ ] **Step 2: Update the "Code:" and "Scripts:" lines**

Find:
```
- Code:   `babysteps/` (pure modules) + `babysteps/envs/pushcube_runner.py` (sim adapter)
- Scripts: `scripts/{smoke_pushcube,stage0_collect,stage0_summarize}.py`
```

Replace with:
```
- Code:   `babysteps/` (pure modules) + `babysteps/envs/{pushcube,pickcube}_runner.py` (sim adapters),
          `babysteps/envs/task_registry.py` (--task dispatch),
          `babysteps/render/{pushcube,pickcube}.py` (per-task MP4 flows)
- Scripts: `scripts/{smoke_pushcube,stage0_collect,stage0_summarize,render_stage0_maniskill}.py`
          — all four scripts accept `--task {PushCube-v1,PickCube-v1}` where applicable
```

- [ ] **Step 3: Update the test-count claim**

Find: `Tests:  85 sim-free unit tests in `tests/``

Replace with: `Tests:  165+ sim-free unit tests in `tests/` (PushCube + PickCube, snapshot-stable across both)`

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "$(cat <<'EOF'
docs(claude.md): refresh GPU command + module list for Sub-project B

Adds PickCube GPU spot-check command (B's acceptance gate item 4),
updates the modules / scripts list for task_registry.py and the
babysteps.render package, and refreshes the test-count claim.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Final gate verification

**Files:** none (verification only).

- [ ] **Step 1: Full test suite — must be all green**

Run: `python -m pytest tests/ -q`

Expected: all tests pass. Net new tests vs baseline:
- `test_task_registry.py`: +7
- `test_stage0_collect_cli.py`: +4 (2 parametrized snapshot + default + unknown-task)
- `test_eval.py`: +2 (PickCube + PushCube title)
- `test_pickcube_delta_pp.py`: +2
- `test_render_modules.py`: +5 (2 pushcube + 3 pickcube)
- Total new: ~20.

If anything fails, do NOT advance to the GPU step.

- [ ] **Step 2: PushCube samples.jsonl byte-identical (Sub-project A regression)**

Run:
```bash
python scripts/stage0_collect.py \
  --out_dir /tmp/pushcube-gate \
  --n_episodes 5 --seed_start 0 --fake-env
diff -u tests/snapshots/pushcube_samples_seeds_0_4.jsonl /tmp/pushcube-gate/samples.jsonl
echo "---"
head -2 /tmp/pushcube-gate/report.md
```

Expected: `diff` produces no output (files identical). Report.md title says PushCube.

- [ ] **Step 3: PickCube samples.jsonl byte-equal to snapshot (Sub-project B gate item 3)**

Run:
```bash
python scripts/stage0_collect.py \
  --task PickCube-v1 \
  --out_dir /tmp/pickcube-gate \
  --n_episodes 5 --seed_start 0 --fake-env
diff -u tests/snapshots/pickcube_samples_seeds_0_4.jsonl /tmp/pickcube-gate/samples.jsonl
echo "---"
head -2 /tmp/pickcube-gate/report.md
cat /tmp/pickcube-gate/report.json | python -c "import json,sys; d=json.load(sys.stdin); print('delta_pp:', d['delta_pp'], 'passed:', d['passed_acceptance'])"
```

Expected: `diff` empty; report title says PickCube; `delta_pp: 100.0 passed: True` (or similar ≥ 10).

- [ ] **Step 4: GPU spot-check (Sub-project B gate item 4) — MANUAL, the only step that needs a compute node**

From the login shell, schedule a 20-min A100 job that runs both render commands from Task 10. Confirm `videos_maniskill/` contains 6 MP4s per task (2 episodes × 3 phases). Visually verify:

- PushCube: phase 1 cube moves; phase 2 frozen frame; phase 3 cube moves from opposite side and reaches goal.
- PickCube: phase 1 gripper closes and lifts to goal; phase 2 gripper closes then opens at lift (cube falls back); phase 3 gripper closes (rotated 90°) and lifts to goal.

This is a manual / sbatch step — there is no automated test for the visual content. The render-module unit tests in Task 7/8 guard the orchestration; only the human can confirm the MP4 looks right.

- [ ] **Step 5: Tag the gate-passing commit**

Once Steps 1-4 are green AND the GPU MP4s look correct:

```bash
git tag stage0-pickcube-b-gate-pass
git log --oneline -1
```

(Do NOT push the tag without the user's explicit OK.)

---

## Self-Review

**Spec coverage:**
- §3 item 1 ("All pre-B tests pass byte-identical"): Tasks 0 + 11 verify.
- §3 item 2 ("New tests: ~30"): already in repo (test_pickcube_adapter etc.); Tasks 1-9 add ~20 more for CLI/registry/render coverage.
- §3 item 3 ("Snapshot test pickcube_samples_seeds_0_4.jsonl byte-stable"): Task 2 + Task 11 step 3.
- §3 item 4 ("GPU visual spot-check"): Tasks 9 + 10 wire it, Task 11 step 4 documents the manual check.
- §3 item 5 ("delta_pp >= 10"): Task 5 codifies as a test; Task 11 step 3 re-verifies via the CLI.
- §4 (schema deltas): no changes — Task 0 step 3 confirms they're already in place.
- §5 (failure attribution): no changes — `grasp_slip` already wired.
- §6 (revision operators): no changes — `contact_substitution` already wired.
- §7.1 (PickCube full implementation): adapter/runner/skill done; CLI wiring + acceptance test are Tasks 1-5; render is Tasks 6-9.
- §8 (CLI / script generalization): Task 1 (registry) + Task 2 (collect) + Task 9 (render) + Task 4 (summarize verified).
- §10 (plan file path): this file at `docs/superpowers/plans/2026-05-17-stage0-pickcube-b-plan.md`. ✓

**Placeholder scan:** No TBD / TODO / "implement later" lines. Every code step shows full code. Every command shows expected output. ✓

**Type consistency:**
- `TaskEntry.adapter_cls` referenced consistently across `task_registry.py` and `stage0_collect.py`. ✓
- `render_episode(env, adapter, seed, fps) -> (frames_dict, titles_dict)` signature consistent across pushcube.py, pickcube.py, the test stub, and the dispatcher script. ✓
- `episode_id_prefix` used in `task_registry.py`, `stage0_collect.py`, and `render_stage0_maniskill.py` — same field name, same string format. ✓
- Phase names `"demo" / "attempt_blocked" / "retry"` consistent in render modules, tests, and the script's per-phase loop. ✓

**Risks / gotchas the engineer should know:**
1. The `_FakeAdapter` class trick in `stage0_collect.py::_make_adapter` uses `entry.adapter_cls` as a dynamic base class. Python supports this but type-checkers will grumble — the `# type: ignore[misc, valid-type]` comment is there for that reason.
2. `tests/test_stage0_collect_cli.py` deletes `stage0_collect` from `sys.modules` before each test to dodge argparse global-state caching. Important if running tests in any other order.
3. PickCube's phase-2 frames are produced by actually stepping the env, which means the per-episode MP4 will be larger than PushCube's frozen-frame phase 2. Expect ~3-5x file size on phase 2 only.
4. The render modules' `_execute_*` helpers re-`env.reset(seed=…)` at the start of each phase to guarantee a clean scene. If you skip this, phase 2's "initial frame" will be wherever phase 1 left the robot — confusing.
