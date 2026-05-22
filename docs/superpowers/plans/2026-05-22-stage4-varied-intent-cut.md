# Stage-4 Varied-Intent Data Cut + Tightened Recoverability — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a varied, balanced Stage-0 cut (PushCube + StackCube, `object_motion` balanced across ≥3 planar directions at ~10 episodes/class), tighten the schema-recoverability certification so constant/label-fed factors no longer get a free pass, re-run the probe, and render one MP4 of a single-factor revision.

**Architecture:** Sim-free pure helpers (injection geometry, stratified/rejection plans, cert classification) are TDD'd on the login node. PushCube varies via a per-episode **cube-pose injection** the runner applies inside `reset()` (Approach A) — threaded through a runner attribute so `episode.run_episode` and its snapshot are untouched. StackCube varies natively, so a **rejection-sampling** driver selects a balanced seed subset (Approach B). Collection + render run on GPU via slurm; the probe re-run and report are sim-free.

**Tech Stack:** Python, NumPy, scikit-learn (`LogisticRegression`, `StratifiedKFold`), ManiSkill/SAPIEN (GPU-only, behind `envs/*_runner.py`), pytest (sim-free suite).

**Spec:** `docs/superpowers/specs/2026-05-22-stage4-varied-intent-cut-design.md`

---

## File Structure

**Create:**
- `babysteps/stage4/collection_plan.py` — pure stratified-assignment + rejection-quota planners (sim-free).
- `scripts/stage4_collect_varied.py` — GPU collection driver (Push: injection; Stack: rejection).
- `slurm/collect_stage4_varied.sbatch` — sbatch for the two collection runs.
- `tests/test_stage4_collection_plan.py` — tests for the planners.
- `tests/test_scene_injection.py` — tests for the injection geometry helpers.

**Modify:**
- `babysteps/envs/scene.py` — add `injected_cube_xy()` + `cubeA_to_cubeB_motion()` (pure).
- `babysteps/envs/pushcube_runner.py` — optional `set_injection()` applied in `reset()` (GPU side).
- `babysteps/stage4/report.py` — three-way `cell_class` + margin gate.
- `scripts/stage4_probe_schema_recoverability.py` — surface `cell_class` counts + markdown column.
- `babysteps/render/pushcube.py` — add `frozen_factors` to the retry caption; optional injection for a varied direction.
- `tests/test_stage4_report.py` — rewrite for the three-way gate.
- `tests/test_render_modules.py` — assert the retry caption lists frozen factors.
- `RUNBOOK.md`, `CODE_MAP.md` — new commands + map entries.

---

## Task 1: Pure injection geometry helpers (`scene.py`)

**Files:**
- Modify: `babysteps/envs/scene.py`
- Test: `tests/test_scene_injection.py` (create)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_scene_injection.py
"""Pure injection geometry — sim-free."""
import numpy as np
import pytest

from babysteps.envs.scene import (
    cubeA_to_cubeB_motion,
    injected_cube_xy,
)


@pytest.mark.parametrize("motion,expected_sign", [
    ("translate_+x", (1.0, 0.0)),
    ("translate_-x", (-1.0, 0.0)),
    ("translate_+y", (0.0, 1.0)),
    ("translate_-y", (0.0, -1.0)),
])
def test_injected_cube_pushes_toward_goal_in_target_motion(motion, expected_sign):
    goal = (0.30, 0.10)
    dist = 0.12
    cube = injected_cube_xy(goal, dist, motion)
    # cube→goal vector points along the target motion, magnitude == dist.
    vec = np.array(goal) - np.array(cube)
    assert np.linalg.norm(vec) == pytest.approx(dist)
    unit = vec / np.linalg.norm(vec)
    assert unit == pytest.approx(np.array(expected_sign), abs=1e-9)


def test_injected_cube_motion_roundtrips_through_goal_direction():
    # The geometry must produce a layout the adapter labels as the same motion.
    from babysteps.envs.scene import goal_direction_to_motion
    goal = (0.25, -0.05)
    for motion in ("translate_+x", "translate_-x", "translate_+y", "translate_-y"):
        cube = injected_cube_xy(goal, 0.1, motion)
        vec = np.array(goal) - np.array(cube)
        assert goal_direction_to_motion(vec) == motion


def test_injected_cube_rejects_non_cardinal_motion():
    with pytest.raises(ValueError):
        injected_cube_xy((0.0, 0.0), 0.1, "lift_up")


def test_cubeA_to_cubeB_motion_matches_displacement_snap():
    # cubeB to the +y of cubeA → cubeA must translate +y to stack.
    assert cubeA_to_cubeB_motion((0.0, 0.0), (0.01, 0.20)) == "translate_+y"
    assert cubeA_to_cubeB_motion((0.0, 0.0), (-0.20, 0.01)) == "translate_-x"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_scene_injection.py -q`
Expected: FAIL — `ImportError: cannot import name 'injected_cube_xy'`.

- [ ] **Step 3: Implement the helpers**

Add to `babysteps/envs/scene.py` (after `goal_direction_to_motion`), and add both names to `__all__`:

```python
def injected_cube_xy(
    goal_xy: tuple[float, float], push_distance: float, target_motion: str,
) -> tuple[float, float]:
    """Cube xy that, pushed toward the FIXED `goal_xy`, travels `target_motion`.

    Used by the Stage-4 varied cut (Approach A): place the cube `push_distance`
    away from the goal on the side opposite the target motion, so the
    cube→goal vector points along `target_motion`. `target_motion` must be one
    of the four cardinal translate tokens."""
    unit = motion_to_unit(target_motion)        # raises on non-cardinal tokens
    gx, gy = float(goal_xy[0]), float(goal_xy[1])
    d = float(push_distance)
    return (gx - d * float(unit[0]), gy - d * float(unit[1]))


def cubeA_to_cubeB_motion(
    cubeA_xy: tuple[float, float], cubeB_xy: tuple[float, float],
) -> str:
    """The cardinal motion cubeA travels to reach cubeB (StackCube binning).

    Uses the same dominant-axis snap as the demo-evidence labeller, so the bin
    a seed lands in equals the `object_motion` label its episode will carry."""
    vec = np.array(cubeB_xy, dtype=np.float64) - np.array(cubeA_xy, dtype=np.float64)
    return goal_direction_to_motion(vec)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_scene_injection.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add babysteps/envs/scene.py tests/test_scene_injection.py
git commit -m "feat(stage4): pure cube-pose injection + stack-motion binning geometry

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Stratified-assignment + rejection-quota planners

**Files:**
- Create: `babysteps/stage4/collection_plan.py`
- Test: `tests/test_stage4_collection_plan.py` (create)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_stage4_collection_plan.py
"""Pure collection planners (sim-free)."""
import pytest

from babysteps.stage4.collection_plan import (
    select_balanced_seeds,
    stratified_seed_plan,
)

_DIRS = ("translate_+x", "translate_-x", "translate_+y", "translate_-y")


def test_stratified_plan_is_balanced_and_deterministic():
    plan = stratified_seed_plan(_DIRS, episodes_per_class=10, seed_start=0)
    assert len(plan) == 40
    # Each class appears exactly episodes_per_class times.
    counts = {c: 0 for c in _DIRS}
    for _seed, cls in plan:
        counts[cls] += 1
    assert all(v == 10 for v in counts.values())
    # Seeds are unique and contiguous from seed_start.
    seeds = [s for s, _ in plan]
    assert seeds == list(range(0, 40))
    # Deterministic.
    assert stratified_seed_plan(_DIRS, 10, 0) == plan


def test_select_balanced_keeps_quota_per_class_in_seed_order():
    # Stream of (seed, observed_class); +x is over-represented, others sparse.
    stream = []
    for s in range(100):
        cls = _DIRS[s % 4] if s < 80 else "translate_+x"
        stream.append((s, cls))
    kept = select_balanced_seeds(stream, _DIRS, episodes_per_class=10)
    assert len(kept) == 40
    # No more than the quota from any class; quota met for all.
    by_cls = {c: 0 for c in _DIRS}
    for s in kept:
        # recover class from the stream
        cls = dict(stream)[s]
        by_cls[cls] += 1
    assert all(v == 10 for v in by_cls.values())
    assert kept == sorted(kept)  # preserves seed order


def test_select_balanced_raises_when_a_class_cannot_be_filled():
    stream = [(s, "translate_+x") for s in range(50)]  # only +x ever appears
    with pytest.raises(ValueError, match="could not fill"):
        select_balanced_seeds(stream, _DIRS, episodes_per_class=10)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_stage4_collection_plan.py -q`
Expected: FAIL — module `babysteps.stage4.collection_plan` does not exist.

- [ ] **Step 3: Implement the planners**

```python
# babysteps/stage4/collection_plan.py
"""Pure collection planners for the Stage-4 varied-intent cut (sim-free).

Two strategies, matching the spec:
  * stratified_seed_plan  — Approach A (PushCube): assign each contiguous seed
    a target class so classes are perfectly balanced. The driver injects the
    cube pose to realise the assigned class.
  * select_balanced_seeds — Approach B (StackCube): given a stream of
    (seed, observed_class) from native resets, keep the first
    `episodes_per_class` seeds of each class (in seed order). Raises if any
    class cannot be filled.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable


def stratified_seed_plan(
    classes: tuple[str, ...], episodes_per_class: int, seed_start: int = 0,
) -> list[tuple[int, str]]:
    """Contiguous (seed, target_class) assignments, balanced by construction.

    Seeds run seed_start .. seed_start + len(classes)*episodes_per_class - 1.
    Class order is round-robin so an interrupted run stays roughly balanced."""
    plan: list[tuple[int, str]] = []
    seed = seed_start
    for _ in range(episodes_per_class):
        for cls in classes:
            plan.append((seed, cls))
            seed += 1
    return plan


def select_balanced_seeds(
    stream: Iterable[tuple[int, str]],
    classes: tuple[str, ...],
    episodes_per_class: int,
) -> list[int]:
    """Keep the first `episodes_per_class` seeds of each class, in seed order.

    `stream` is (seed, observed_class) from native resets. Returns the kept
    seeds sorted ascending. Raises ValueError if any class is underfilled when
    the stream is exhausted."""
    want = set(classes)
    kept: dict[str, list[int]] = {c: [] for c in classes}
    for seed, cls in stream:
        if cls in want and len(kept[cls]) < episodes_per_class:
            kept[cls].append(seed)
        if all(len(kept[c]) >= episodes_per_class for c in classes):
            break
    short = {c: len(v) for c, v in kept.items() if len(v) < episodes_per_class}
    if short:
        raise ValueError(
            f"select_balanced_seeds could not fill {short} "
            f"(wanted {episodes_per_class}/class)"
        )
    return sorted(s for v in kept.values() for s in v)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_stage4_collection_plan.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add babysteps/stage4/collection_plan.py tests/test_stage4_collection_plan.py
git commit -m "feat(stage4): pure stratified + rejection-quota collection planners

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Three-way cert classification + margin gate (`report.py`)

**Files:**
- Modify: `babysteps/stage4/report.py`
- Test: `tests/test_stage4_report.py` (rewrite)

- [ ] **Step 1: Rewrite the failing tests**

Replace the body of `tests/test_stage4_report.py` with:

```python
"""Stage-4 report aggregation + Markdown rendering (three-way gate)."""


def _row(task, factor, acc, *, n_unique=2, trivial=False,
         majority=0.5, shuffled=0.5):
    return {
        "task": task, "factor": factor,
        "probe_acc_mean": acc, "probe_acc_std": 0.0,
        "majority_class_acc": majority, "shuffled_features_acc": shuffled,
        "n_unique_labels": n_unique, "n_episodes": 40,
        "trivially_constant": trivial,
    }


def test_cell_class_is_three_way():
    from babysteps.stage4.report import build_report
    rows = [
        # geometric, clears 0.90 and both baselines by the margin → PASS
        _row("StackCube-v1", "object_motion", 0.95),
        # geometric, below 0.90 → FAIL
        _row("PushCube-v1", "object_motion", 0.80),
        # label-identity (contact_region) → not counted geometric even at 1.0
        _row("PushCube-v1", "contact_region", 1.0),
        # PushCube approach_direction is label-identity too
        _row("PushCube-v1", "approach_direction", 1.0),
        # goal_state is label-identity
        _row("StackCube-v1", "goal_state", 1.0, n_unique=1, trivial=True),
        # trivially constant
        _row("PushCube-v1", "constraint_region", 1.0, n_unique=1, trivial=True),
    ]
    rep = build_report(rows)
    cls = {(t, f): rep["by_task"][t][f]["cell_class"]
           for t in rep["by_task"] for f in rep["by_task"][t]}
    assert cls[("StackCube-v1", "object_motion")] == "geometric"
    assert cls[("PushCube-v1", "object_motion")] == "geometric"
    assert cls[("PushCube-v1", "contact_region")] == "label_identity"
    assert cls[("PushCube-v1", "approach_direction")] == "label_identity"
    assert cls[("StackCube-v1", "goal_state")] == "trivially_constant"
    assert cls[("PushCube-v1", "constraint_region")] == "trivially_constant"

    g = rep["gate"]
    assert g["n_geometric"] == 2
    assert g["n_passing"] == 1          # only StackCube/object_motion
    assert g["n_failing"] == 1          # PushCube/object_motion
    assert ("PushCube-v1", "object_motion") in g["failing_cells"]
    assert g["n_label_identity"] == 2
    assert g["n_trivial"] == 2


def test_geometric_pass_requires_margin_over_baselines():
    from babysteps.stage4.report import build_report
    # 0.92 ≥ 0.90 but only 0.04 over a 0.88 shuffled baseline → FAIL the margin.
    rows = [_row("StackCube-v1", "object_motion", 0.92,
                 majority=0.40, shuffled=0.88)]
    rep = build_report(rows)
    assert rep["by_task"]["StackCube-v1"]["object_motion"]["gate"] == "FAIL"
    assert rep["gate"]["n_passing"] == 0


def test_markdown_table_has_class_column():
    from babysteps.stage4.report import build_report, markdown_table
    rep = build_report([_row("StackCube-v1", "object_motion", 0.95)])
    md = markdown_table(rep)
    assert "StackCube-v1" in md
    assert "object_motion" in md
    assert "geometric" in md
    assert "0.95" in md
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_stage4_report.py -q`
Expected: FAIL — `KeyError: 'cell_class'` / `KeyError: 'n_geometric'`.

- [ ] **Step 3: Implement the three-way classification + margin gate**

Replace `babysteps/stage4/report.py` with:

```python
"""Stage-4 per-task per-factor report aggregation + Markdown rendering.

Each (task, factor) cell is classified three ways (spec §4):

  * trivially_constant — one label for the whole task. Excluded from the gate.
  * label_identity     — recoverable without trajectory geometry because the
                         factor is fed in as a feature one-hot (contact_region
                         ← contact_region_label, goal_state ← final_state) or
                         is a deterministic function of one (PushCube
                         approach_direction = face_to_approach(contact_region)).
                         Reported but NOT counted toward the geometric headline.
  * geometric          — recoverable only from trajectory geometry (today:
                         object_motion). GATED: probe_acc_mean ≥ GATE_THRESHOLD
                         AND clears majority + shuffled baselines by GATE_MARGIN.
"""
from __future__ import annotations

_CELL_KEYS: tuple[str, ...] = (
    "n_episodes",
    "n_unique_labels",
    "majority_class_acc",
    "shuffled_features_acc",
    "probe_acc_mean",
    "probe_acc_std",
    "trivially_constant",
)

GATE_THRESHOLD: float = 0.90
GATE_MARGIN: float = 0.10

# (task, factor) pairs that are label-identity. "*" matches any task.
_LABEL_IDENTITY: frozenset[tuple[str, str]] = frozenset({
    ("*", "contact_region"),
    ("*", "goal_state"),
    ("PushCube-v1", "approach_direction"),
})


def _is_label_identity(task: str, factor: str) -> bool:
    return (("*", factor) in _LABEL_IDENTITY
            or (task, factor) in _LABEL_IDENTITY)


def _cell_class(task: str, factor: str, cell: dict) -> str:
    if cell["trivially_constant"] or cell["n_unique_labels"] <= 1:
        return "trivially_constant"
    if _is_label_identity(task, factor):
        return "label_identity"
    return "geometric"


def _geometric_pass(cell: dict) -> bool:
    acc = cell["probe_acc_mean"]
    return (
        acc >= GATE_THRESHOLD
        and acc >= cell["majority_class_acc"] + GATE_MARGIN
        and acc >= cell["shuffled_features_acc"] + GATE_MARGIN
    )


def _cell_gate(task: str, factor: str, cell: dict) -> str:
    klass = cell["cell_class"]
    if klass == "trivially_constant":
        return "trivial"
    if klass == "label_identity":
        return "label_identity"
    return "PASS" if _geometric_pass(cell) else "FAIL"


def build_report(rows: list[dict]) -> dict:
    """Aggregate annotated probe outputs into a nested report + gate summary.

    Returns::

        {
          "by_task": {task: {factor: {<cell metrics> + cell_class + gate}}},
          "gate": {threshold, margin, n_total, n_trivial, n_label_identity,
                   n_geometric, n_passing, n_failing, failing_cells},
        }

    n_passing / n_failing count GEOMETRIC cells only — label-identity and
    trivially-constant cells never count as a gate pass.
    """
    by_task: dict[str, dict[str, dict]] = {}
    n_total = n_trivial = n_label_identity = n_geometric = 0
    n_passing = n_failing = 0
    failing_cells: list[tuple[str, str]] = []

    for row in rows:
        task = row["task"]
        factor = row["factor"]
        cell = {k: row[k] for k in _CELL_KEYS}
        cell["cell_class"] = _cell_class(task, factor, cell)
        cell["gate"] = _cell_gate(task, factor, cell)
        by_task.setdefault(task, {})[factor] = cell

        n_total += 1
        if cell["cell_class"] == "trivially_constant":
            n_trivial += 1
        elif cell["cell_class"] == "label_identity":
            n_label_identity += 1
        else:
            n_geometric += 1
            if cell["gate"] == "PASS":
                n_passing += 1
            else:
                n_failing += 1
                failing_cells.append((task, factor))

    gate = {
        "threshold": GATE_THRESHOLD,
        "margin": GATE_MARGIN,
        "n_total": n_total,
        "n_trivial": n_trivial,
        "n_label_identity": n_label_identity,
        "n_geometric": n_geometric,
        "n_passing": n_passing,
        "n_failing": n_failing,
        "failing_cells": failing_cells,
    }
    return {"by_task": by_task, "gate": gate}


def markdown_table(report: dict) -> str:
    """Render `report['by_task']` as one Markdown table per task (humans)."""
    header = (
        "| factor | class | n_unique | n_episodes | majority | shuffled "
        "| probe ± std | gate |"
    )
    rule = "| --- | --- | --- | --- | --- | --- | --- | --- |"

    lines: list[str] = []
    by_task = report["by_task"]
    for task in sorted(by_task):
        lines.append(f"### {task}")
        lines.append("")
        lines.append(header)
        lines.append(rule)
        factors = by_task[task]
        for factor in sorted(factors):
            c = factors[factor]
            probe_pm = f"{c['probe_acc_mean']:.2f} ± {c['probe_acc_std']:.2f}"
            lines.append(
                f"| {factor} | {c['cell_class']} | {c['n_unique_labels']} "
                f"| {c['n_episodes']} | {c['majority_class_acc']:.2f} "
                f"| {c['shuffled_features_acc']:.2f} | {probe_pm} | {c['gate']} |"
            )
        lines.append("")
    return "\n".join(lines)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_stage4_report.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add babysteps/stage4/report.py tests/test_stage4_report.py
git commit -m "feat(stage4): three-way cert classification (trivial/label-identity/geometric) + margin gate

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Probe CLI surfaces the new classes

**Files:**
- Modify: `scripts/stage4_probe_schema_recoverability.py:85-99` (`_render_markdown`) and `:128-138` (summary print)
- Test: none new (sim-free CLI exercised by Task 7's re-run; logic covered by Task 3)

- [ ] **Step 1: Update `_render_markdown` summary line**

Replace the `header` list and trailing failing-cells block in `_render_markdown` with:

```python
    g = report["gate"]
    header = [
        "# Stage-4 Schema-Recoverability Probe",
        "",
        (f"Gate: GEOMETRIC cells must reach probe_acc_mean >= "
         f"{GATE_THRESHOLD:.2f} AND clear chance + shuffled by "
         f"{g['margin']:.2f}."),
        (f"Cells: {g['n_total']} total | {g['n_geometric']} geometric "
         f"({g['n_passing']} pass / {g['n_failing']} fail) | "
         f"{g['n_label_identity']} label-identity | {g['n_trivial']} "
         f"trivially constant."),
        "",
    ]
    md = "\n".join(header) + "\n" + markdown_table(report)
    if g["n_failing"]:
        failing = ", ".join(f"{t}/{f}" for t, f in g["failing_cells"])
        md += f"\n**Failing geometric cells (need a notes.md explanation):** {failing}\n"
    return md
```

- [ ] **Step 2: Update the stdout summary in `main()`**

Replace the `print(...)` gate summary (currently `scripts/stage4_probe_schema_recoverability.py:129-131`) with:

```python
    print(f"wrote {args.out_dir}/schema_recoverability.{{json,md}}")
    print(f"geometric gate: {g['n_passing']} pass / {g['n_failing']} fail "
          f"(of {g['n_geometric']} geometric); "
          f"{g['n_label_identity']} label-identity, {g['n_trivial']} trivial")
```

- [ ] **Step 3: Smoke-run the CLI on the existing data (still works)**

Run:
```bash
python scripts/stage4_probe_schema_recoverability.py \
  --out-dir "$CLAUDE_JOB_DIR/probe_smoke"
cat "$CLAUDE_JOB_DIR/probe_smoke/schema_recoverability.md" | head -20
```
Expected: runs without error; markdown now shows a `class` column and the new summary line (object_motion = geometric, others label-identity/trivial on the old constant data).

- [ ] **Step 4: Commit**

```bash
git add scripts/stage4_probe_schema_recoverability.py
git commit -m "feat(stage4): probe CLI reports geometric/label-identity/trivial split

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: PushCube cube-pose injection hook (GPU side)

**Files:**
- Modify: `babysteps/envs/pushcube_runner.py`
- Test: GPU spike (manual) — no sim-free unit test (runner is the Vulkan boundary)

- [ ] **Step 1: Add the injection field + setter + reset application**

In `PushCubeEnvRunner.__init__`, after `self._last_seed = None`, add:
```python
        self._pending_motion: Optional[str] = None
```

Add a method:
```python
    def set_injection(self, target_motion: Optional[str]) -> None:
        """Set (or clear with None) the target object_motion for the NEXT
        reset. The driver calls this per episode before run_episode; reset
        repositions the cube so cube→goal points along target_motion. See
        babysteps.envs.scene.injected_cube_xy."""
        self._pending_motion = target_motion
```

In `reset()`, after the existing `obs, _info = self._env.reset(seed=int(seed))` and the `_read_obs` call, before building `SceneState`, insert:
```python
        if self._pending_motion is not None:
            from babysteps.envs.scene import injected_cube_xy
            push_dist = float(np.linalg.norm(goal_xy - cube_xy))
            new_xy = injected_cube_xy(
                (float(goal_xy[0]), float(goal_xy[1])), push_dist,
                self._pending_motion,
            )
            import sapien
            base = self._env.unwrapped.obj  # PushCube-v1 cube actor
            pose = base.pose.sp if hasattr(base.pose, "sp") else base.pose
            base.set_pose(sapien.Pose(
                p=[new_xy[0], new_xy[1], float(pose.p[2])],
                q=list(pose.q),
            ))
            obs = self._env.unwrapped.get_obs()
            tcp, cube_xy, goal_xy, cube_z = _read_obs(obs)
```

> **Note on the actor handle:** PushCube-v1's cube is `env.unwrapped.obj`. If a
> ManiSkill version exposes it under a different attribute, the spike (Step 2)
> is where you confirm it; fall back to moving `env.unwrapped.goal_region`
> (then re-derive `injected_cube_xy` against the original cube_xy with the goal
> as the moved point) only if cube reposition desyncs the success check.

- [ ] **Step 2: GPU spike — verify the success check honors the injected layout**

On a GPU node (`salloc --gres=gpu:1`), run:
```bash
python - <<'PY'
from babysteps.envs.pushcube_runner import PushCubeEnvRunner
from babysteps.envs.pushcube_adapter import PushCubeAdapter
r = PushCubeEnvRunner(); a = PushCubeAdapter()
for motion in ("translate_+x","translate_-x","translate_+y","translate_-y"):
    r.set_injection(motion)
    scene = r.reset(7)
    intent = a.oracle_correct_intent(scene)
    print(motion, "->", intent.object_motion, "cube", scene.cube_xy, "goal", scene.goal_xy)
    res = r.run(intent, scene)
    print("   success:", res.success, "moved:", res.object_moved)
r.close()
PY
```
Expected: for each `motion`, `intent.object_motion == motion` and the oracle push reports `success: True`. If a direction fails, debug the actor handle / push distance before proceeding (this is the spec's de-risk gate).

- [ ] **Step 3: Commit**

```bash
git add babysteps/envs/pushcube_runner.py
git commit -m "feat(stage4): PushCube reset cube-pose injection for varied push direction

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Varied collection driver + sbatch (GPU)

**Files:**
- Create: `scripts/stage4_collect_varied.py`
- Create: `slurm/collect_stage4_varied.sbatch`
- Test: light sim-free import/arg test added to `tests/test_stage4_collection_plan.py`

- [ ] **Step 1: Write the driver**

```python
# scripts/stage4_collect_varied.py
"""Stage-4 varied-intent collection driver (GPU).

PushCube (Approach A): stratified seed→motion plan; the runner injects the
cube pose per seed so object_motion is balanced by construction.
StackCube (Approach B): native resets binned by cubeA→cubeB direction; keep a
balanced seed subset (rejection sampling), then run those episodes.

Writes one EpisodeRecord per line to <out_dir>/<task>/samples.jsonl plus
report.{json,md} (via babysteps.eval), mirroring scripts/stage0_collect.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from babysteps.envs.scene import cubeA_to_cubeB_motion  # noqa: E402
from babysteps.envs.task_registry import get_task_entry  # noqa: E402
from babysteps.episode import run_episode  # noqa: E402
from babysteps.eval import compute_metrics, write_report  # noqa: E402
from babysteps.stage4.collection_plan import (  # noqa: E402
    select_balanced_seeds,
    stratified_seed_plan,
)

_DIRS = ("translate_+x", "translate_-x", "translate_+y", "translate_-y")


def _collect_pushcube(out_dir: Path, per_class: int, seed_start: int) -> int:
    entry = get_task_entry("PushCube-v1")
    adapter = entry.adapter_cls()
    runner = adapter.env_runner()
    plan = stratified_seed_plan(_DIRS, per_class, seed_start)
    records = []
    try:
        for seed, motion in plan:
            runner.set_injection(motion)
            rec = run_episode(
                episode_id=f"pushcube_varied_seed_{seed:04d}",
                seed=seed, adapter=adapter,
            )
            records.append(rec)
    finally:
        adapter.close()
    return _write(out_dir / "PushCube-v1", records)


def _collect_stackcube(
    out_dir: Path, per_class: int, seed_start: int, max_scan: int,
) -> int:
    entry = get_task_entry("StackCube-v1")
    adapter = entry.adapter_cls()
    runner = adapter.env_runner()
    # Pass 1: bin native resets by cubeA→cubeB direction.
    stream = []
    for seed in range(seed_start, seed_start + max_scan):
        scene = runner.reset(seed)
        cubeB_xy = scene.extra["cubeB_xy"]
        stream.append((seed, cubeA_to_cubeB_motion(scene.cube_xy, cubeB_xy)))
    kept = select_balanced_seeds(stream, _DIRS, per_class)
    # Pass 2: full episodes on the kept seeds.
    records = []
    try:
        for seed in kept:
            rec = run_episode(
                episode_id=f"stackcube_varied_seed_{seed:04d}",
                seed=seed, adapter=adapter,
            )
            records.append(rec)
    finally:
        adapter.close()
    return _write(out_dir / "StackCube-v1", records)


def _write(task_dir: Path, records) -> int:
    task_dir.mkdir(parents=True, exist_ok=True)
    with (task_dir / "samples.jsonl").open("w") as f:
        for rec in records:
            f.write(rec.to_jsonl_line() + "\n")
    metrics = compute_metrics(records)
    write_report(metrics, task_dir)
    print(f"wrote {task_dir}/samples.jsonl ({len(records)} episodes)")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=["PushCube-v1", "StackCube-v1"], required=True)
    p.add_argument("--out_dir", type=Path,
                   default=_ROOT / "datasets/stage4/varied_intent")
    p.add_argument("--per_class", type=int, default=10)
    p.add_argument("--seed_start", type=int, default=0)
    p.add_argument("--max_scan", type=int, default=400,
                   help="StackCube only: max native seeds to scan for binning.")
    args = p.parse_args(argv)
    if args.task == "PushCube-v1":
        return _collect_pushcube(args.out_dir, args.per_class, args.seed_start)
    return _collect_stackcube(
        args.out_dir, args.per_class, args.seed_start, args.max_scan)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Add a sim-free import/arg test**

Append to `tests/test_stage4_collection_plan.py`:

```python
def test_driver_module_imports_and_builds_pushcube_plan():
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "stage4_collect_varied", root / "scripts/stage4_collect_varied.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # must import without a simulator
    assert mod._DIRS == (
        "translate_+x", "translate_-x", "translate_+y", "translate_-y")
    # The PushCube plan it would run is balanced.
    plan = mod.stratified_seed_plan(mod._DIRS, 10, 0)
    assert len(plan) == 40
```

Run: `python -m pytest tests/test_stage4_collection_plan.py -q`
Expected: PASS (4 tests). Confirms the driver module imports with no Vulkan.

- [ ] **Step 3: Write the sbatch script**

```bash
# slurm/collect_stage4_varied.sbatch
#!/bin/bash
#SBATCH --job-name=bs_stage4_varied
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
python scripts/stage4_collect_varied.py --task PushCube-v1
python scripts/stage4_collect_varied.py --task StackCube-v1
```

> Match the resource directives (partition/account) of `slurm/render_pushcube.sbatch`
> if this cluster requires them; copy whatever header that file uses.

- [ ] **Step 4: Commit**

```bash
git add scripts/stage4_collect_varied.py slurm/collect_stage4_varied.sbatch tests/test_stage4_collection_plan.py
git commit -m "feat(stage4): varied-intent collection driver (Push injection + Stack rejection) + sbatch

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Make the single-factor invariant visible in the MP4 caption

**Files:**
- Modify: `babysteps/render/pushcube.py:178-183` (retry title in `render_episode`)
- Test: `tests/test_render_modules.py`

- [ ] **Step 1: Write the failing test**

Find how `test_render_modules.py` invokes the PushCube render (it uses a fake/stub env). Add a test that calls `render_episode` and asserts the retry subtitle names the frozen factors. Append:

```python
def test_pushcube_retry_caption_lists_frozen_factors(pushcube_render_env):
    # pushcube_render_env: the existing fixture/stub used by this module's
    # other PushCube render tests (reuse it; do not create a new sim).
    from babysteps.envs.pushcube_adapter import PushCubeAdapter
    from babysteps.render.pushcube import render_episode
    _frames, titles = render_episode(
        pushcube_render_env, PushCubeAdapter(), seed=0, fps=2)
    _title, subtitle = titles["retry"]
    assert "frozen" in subtitle.lower()
    # The preserved factors must be named so the invariant is visible.
    assert "goal_state" in subtitle
    assert "object_motion" in subtitle
```

> If `test_render_modules.py` uses a different env-construction pattern (e.g. a
> module-level helper rather than a fixture), mirror that exact pattern instead
> of `pushcube_render_env`; the assertion on the subtitle is what matters.

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_render_modules.py -k frozen_factors -q`
Expected: FAIL — subtitle has no "frozen" / factor names yet.

- [ ] **Step 3: Add frozen factors to the retry title**

In `render_episode`, capture the `Revision` (currently discarded as `_rev`) and surface its frozen factors. Change:
```python
    revised_intent, _rev = adapter.revise_intent(
        initial_intent, s["attribution"], scene_exec,
    )
```
to:
```python
    revised_intent, revision = adapter.revise_intent(
        initial_intent, s["attribution"], scene_exec,
    )
```
and change the `retry_title` to:
```python
    retry_title = (
        f"{short_id}  phase 3/3: retry (success={out_retry['success']})",
        f"approach_substitution: "
        f"{initial_intent.approach_direction} → "
        f"{revised_intent.approach_direction}  |  "
        f"frozen (preserved): {', '.join(revision.frozen_factors)}",
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_render_modules.py -k frozen_factors -q`
Expected: PASS. (`revision.frozen_factors` for `approach_substitution` includes `goal_state` and `object_motion` — see `babysteps/revision.py`.)

- [ ] **Step 5: Run the full sim-free suite**

Run: `python -m pytest tests/ -q`
Expected: PASS (all prior tests + the new ones; no GPU).

- [ ] **Step 6: Commit**

```bash
git add babysteps/render/pushcube.py tests/test_render_modules.py
git commit -m "feat(stage4): show frozen (preserved) factors in PushCube retry caption

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Run collection, re-run probe, render, write notes (GPU + docs)

**Files:**
- Create (artifacts): `datasets/stage4/varied_intent/{PushCube,StackCube}-v1/`, `reports/stage4/schema_recoverability_varied/`, `renders/stage4_varied/`, `reports/stage4/schema_recoverability_varied/notes.md`
- Modify: `RUNBOOK.md`, `CODE_MAP.md`

- [ ] **Step 1: Collect the varied cut (GPU)**

Run (GPU node):
```bash
sbatch slurm/collect_stage4_varied.sbatch     # or run the two python commands directly
```
Expected: `datasets/stage4/varied_intent/PushCube-v1/samples.jsonl` (40 lines) and `StackCube-v1/samples.jsonl` (40 lines), each with a `report.{json,md}`.

- [ ] **Step 2: Verify balance**

Run:
```bash
for t in PushCube-v1 StackCube-v1; do
  echo "== $t =="
  python - <<PY
import json, collections
c = collections.Counter()
for line in open("datasets/stage4/varied_intent/$t/samples.jsonl"):
    c[json.loads(line)["execution"]["initial_intent"]["object_motion"]] += 1
print(dict(c))
PY
done
```
Expected: each task shows ≥3 of 4 directions with ~10 episodes/class (the acceptance gate, spec §8.1). If StackCube underfills a class, raise `--max_scan`.

- [ ] **Step 3: Re-run the tightened probe on the varied cut**

Run:
```bash
python scripts/stage4_probe_schema_recoverability.py \
  --jsonl datasets/stage4/varied_intent/PushCube-v1/samples.jsonl \
  --jsonl datasets/stage4/varied_intent/StackCube-v1/samples.jsonl \
  --out-dir reports/stage4/schema_recoverability_varied/
```
Expected: `object_motion` is `geometric` in both tasks; it should **PASS** (≥0.90, clears chance + shuffled by 0.10) in ≥1 task (spec §8.2). Note the result for the notes file.

- [ ] **Step 4: Render the MP4 (GPU)**

Render one PushCube episode with the existing renderer. Pick a seed whose injected direction makes a clean blocked-approach (e.g. one collected seed). Run:
```bash
python scripts/render_stage0_maniskill.py --task PushCube-v1 --seed 0 \
  --out-dir renders/stage4_varied/ --fps 15
```
> If `render_stage0_maniskill.py` does not accept `--out-dir`/`--seed`, use its
> actual flags (see `slurm/render_pushcube.sbatch` for the canonical
> invocation). To render an injected direction, call
> `runner.set_injection(...)` is **not** wired into the render script — for the
> deliverable a native-direction PushCube episode already demonstrates the
> single-factor (`approach_direction`) revision with frozen factors in the
> caption; rendering an injected direction is optional polish, not required by
> the gate (spec §8.4).
Expected: `..__1_demo.mp4`, `..__2_attempt_blocked.mp4`, `..__3_retry.mp4`; the retry caption lists the frozen (preserved) factors.

- [ ] **Step 5: Write the results notes**

Create `reports/stage4/schema_recoverability_varied/notes.md` summarizing: the cut (tasks, per-class counts), the three-way table, the `object_motion` geometric result vs. the M1 0.75, confirmation that no label-identity/trivial cell is counted as a pass, and the MP4 path. Mirror the structure of `reports/stage4/schema_recoverability/notes.md`.

- [ ] **Step 6: Update docs**

- `RUNBOOK.md`: add the collect-varied, probe-rerun, and render commands from Steps 1/3/4.
- `CODE_MAP.md`: note `babysteps/stage4/collection_plan.py`, `scripts/stage4_collect_varied.py`, and the `scene.injected_cube_xy` / `cubeA_to_cubeB_motion` helpers.

- [ ] **Step 7: Commit the artifacts + docs**

```bash
git add datasets/stage4/varied_intent reports/stage4/schema_recoverability_varied \
        renders/stage4_varied RUNBOOK.md CODE_MAP.md
git commit -m "report(stage4): varied-intent cut, tightened recoverability re-run, MP4

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- §1 varied/balanced cut (Push+Stack, object_motion balanced) → Tasks 1, 2, 5, 6, 8.
- §3.1 stratified plan → Task 2. §3.2 PushCube injection → Tasks 1, 5. §3.3 StackCube rejection → Tasks 1, 2, 6.
- §3.4 firewall + single-factor invariants → unchanged loop (Task 6 uses `run_episode` as-is); invariant made visible in Task 7.
- §4 three-way cert + margin gate → Task 3; CLI surfacing → Task 4.
- §5 MP4 → Tasks 7, 8 Step 4.
- §6 sim-free tests → Tasks 1, 2, 3, 6, 7. §7 outputs/touch-points → Tasks 6, 8.
- §8 acceptance gate → Task 8 Steps 2, 3, 4.

**Placeholder scan:** No "TBD/TODO". GPU steps (Tasks 5, 8) are commands with expected output, not code stubs. The two "if your fixture/flags differ" notes (Tasks 7, 8) point at the exact existing pattern to copy rather than inventing one — acceptable, since the sim-free assertion (frozen-factor subtitle) and the artifact targets are concrete.

**Type consistency:** `injected_cube_xy(goal_xy, push_distance, target_motion)`, `cubeA_to_cubeB_motion(cubeA_xy, cubeB_xy)`, `stratified_seed_plan(classes, episodes_per_class, seed_start)`, `select_balanced_seeds(stream, classes, episodes_per_class)`, `set_injection(target_motion)` are used identically across Tasks 1, 2, 5, 6. `build_report` returns `gate` with `n_geometric/n_passing/n_failing/n_label_identity/n_trivial/margin` — consumed by the CLI (Task 4) and tests (Task 3) with matching keys.
