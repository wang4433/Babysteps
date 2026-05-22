# Stage-4 Milestone 1 — Schema-Recoverability Probe Spike (Plan)

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put an honest, falsifiable number on the load-bearing Stage-4 claim
that the Stage-0 discrete intent factors are linearly recoverable from
object-centric demonstration evidence — *before* any encoder, ReviseHead, or
action decoder is trained.

**Why this milestone first:** Stage-4's three certifications are
(1) probe recoverability ≥ 90%, (2) frozen-slot ℓ₂ drift ≤ ε,
(3) selectivity paired-test p > α. Only (1) is reachable without a trained
ReviseHead or action decoder. If (1) fails on the existing Stage-0 data,
Stage-4 is not yet a faithful refinement of Stage-0 by its own contract
(`goal.md` §"Certification Interface"), and the rest of the Stage-4 pipeline
is not worth building until the schema or labelling is fixed. This milestone
is the cheapest possible falsification check on the spec.

**Authority:** `goal.md` §"Stage 4: Object-Centric Latent Slot-Intent
Bottleneck" (lines 449–557). If anything in this plan disagrees with `goal.md`,
`goal.md` wins.

**Architecture:** Three pure, sim-free modules under a new `babysteps/stage4/`
subpackage, driven by one CLI under `scripts/`:

```text
babysteps/stage4/
  __init__.py           docstring + Stage-4 scope + privileged-firewall stance
  features.py           extract_episode_features(record) -> np.ndarray
                        Firewall-strict: reads ONLY DemoEvidence-shaped fields.
  probe.py              train_probe(X, y) -> (model, cv_accuracy)
                        sklearn LogisticRegression, 5-fold stratified CV.
  report.py             build_report(rows) -> dict + markdown_table(report)

scripts/
  stage4_probe_schema_recoverability.py   CLI: jsonl glob -> features -> probe
                                          -> report JSON + Markdown.

tests/
  test_stage4_features.py     fixture-based feature-extraction tests
  test_stage4_probe.py        synthetic-data probe smoke test
  test_stage4_report.py       report aggregation + markdown rendering test

reports/stage4/schema_recoverability/
  schema_recoverability.json  per-task per-factor accuracy table (machine)
  schema_recoverability.md    same table rendered for humans
  notes.md                    written interpretation (gate artifact)
```

**Tech Stack:** Python 3, NumPy, scikit-learn (new optional dep), pytest.
No new third-party deps beyond scikit-learn.

**Spec (authority for the *what*):** `goal.md` §"Stage 4" (lines 449–557).
**Plan (this file, authority for the *how*):**
`docs/superpowers/plans/2026-05-22-stage4-m1-schema-recoverability-probe-plan.md`.

---

## Conventions (read once)

- Run the existing suite with `python -m pytest tests/ -q` (sim-free, no GPU).
  This plan adds three new test modules to that suite; the full suite must
  stay green after every task.
- TDD: failing test first, minimal impl, green, commit. One task = one commit.
- Commit message trailer:
  `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`.
- **Privileged-firewall invariant** (`CLAUDE.md` working invariant 4 + `goal.md`
  §5): The probe must consume **only** observable demonstration evidence —
  i.e., the `DemoEvidence`-shaped fields in `record["demo"]`
  (`object_trajectory`, `contact_region_label`, `final_state`). The probe must
  **not** read `record["execution"]["initial_intent"]` (that is the *label*
  and would be leakage), nor `record["failure_packet"]`, `record["revision"]`,
  `record["retry"]`, nor any privileged `SceneState` field. This rule is what
  makes the recoverability number meaningful; violate it and the milestone is
  wasted.
- Stage-0 code in `babysteps/{schemas,demo,episode,failure,revision,eval,viz}.py`
  and `babysteps/envs/`, `babysteps/render/`, `babysteps/skills/` is **read-only**
  for this milestone. Do not touch it. Stage-4 work lives under
  `babysteps/stage4/` and `scripts/stage4_*.py` only.
- The 302-test snapshot guards under `tests/` must remain byte-identical green.

---

## Datasets to probe (already on disk)

```text
datasets/stage0_baselines/babysteps_selective/{PushCube-v1,PickCube-v1,StackCube-v1}/samples.jsonl   # primary
datasets/stage0_baselines/oracle_factor_revision/{PushCube-v1,PickCube-v1,StackCube-v1}/samples.jsonl  # optional union
```

Each line is one `EpisodeRecord.to_jsonl_line()`. ~24 episodes per
(baseline, task). The CLI takes a list of JSONL paths; default = the six
primary files above. Probes are trained **per task**, never pooled across
tasks (different factor value sets per task).

---

## Acceptance Criteria (the gate)

Milestone 1 is achieved when **all** of the following hold:

1. `reports/stage4/schema_recoverability/schema_recoverability.json` exists
   and contains, for each `(task, factor)` pair, the keys:
   `n_episodes`, `n_unique_labels`, `majority_class_acc`,
   `shuffled_features_acc`, `probe_acc_mean`, `probe_acc_std`.
2. For every `(task, factor)` where `n_unique_labels >= 2`:
   either `probe_acc_mean >= 0.90`, OR `notes.md` contains a one-paragraph
   defensible explanation pointing at the specific (task, factor) cell and
   describing the structural reason recovery is not linear (e.g., "factor is
   determined by `blocked_sides`, which is privileged and not in
   DemoEvidence"; "factor depends on history beyond a single trajectory; a
   sequence model is required").
3. `(task, factor)` cells where `n_unique_labels == 1` are reported with
   `probe_acc_mean = 1.0` and explicitly flagged as `trivially_constant` —
   they do *not* count toward the gate.
4. The full `pytest` suite (existing + 3 new test files) passes on the
   login node with no GPU.
5. `notes.md` ends with a one-paragraph "Implication for Stage-4 Milestone 2"
   section that picks one of:
   (a) proceed to learned slot encoder (recoverability is acceptable),
   (b) revise the failing factor's labelling / schema before encoder work,
   (c) reframe Stage-4 success criteria for the failing factor (e.g., probe
   from `(G_t, language)` not `G_t` alone).

If any non-trivial cell falls below 90% **and** lacks an explanation in
notes.md, the milestone is incomplete.

---

## Task 1: Add `scikit-learn` optional dependency and create `babysteps/stage4/` package skeleton

**Files:**
- Modify: `pyproject.toml` (add `analysis = ["scikit-learn"]` to optional deps).
- Create: `babysteps/stage4/__init__.py`
- Create: `tests/test_stage4_smoke.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_stage4_smoke.py`:

```python
"""Stage-4 package importability smoke test."""


def test_stage4_package_imports():
    import babysteps.stage4  # noqa: F401


def test_stage4_package_docstring_names_firewall():
    import babysteps.stage4
    doc = (babysteps.stage4.__doc__ or "").lower()
    assert "stage 4" in doc
    assert "privileged" in doc or "firewall" in doc
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_stage4_smoke.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'babysteps.stage4'`).

- [ ] **Step 3: Create the package**

Create `babysteps/stage4/__init__.py` with the following module docstring
(verbatim):

```python
"""BABYSTEPS Stage-4 — learned-latent track (see goal.md §"Stage 4").

This subpackage hosts only sim-free analysis code: feature extraction over
the demo-evidence fields of existing Stage-0 episode JSONs, sklearn linear
probes, and the per-task per-factor report builder.

Privileged-firewall invariant: every Stage-4 module here must consume only
DemoEvidence-shaped inputs (object_trajectory, contact_region_label,
final_state). It must never read execution.initial_intent (label leakage),
failure_packet, revision, retry, or any privileged SceneState field. The
firewall is what makes the recoverability number meaningful.
"""
```

- [ ] **Step 4: Add the optional dependency**

In `pyproject.toml`, extend `[project.optional-dependencies]` to add an
`analysis` group, leaving `sim` and `dev` untouched:

```toml
[project.optional-dependencies]
sim = ["mani_skill"]
dev = ["pytest"]
analysis = ["scikit-learn"]
```

- [ ] **Step 5: Install the new optional dep locally**

Run: `pip install -e ".[analysis]"`
Expected: scikit-learn (and its NumPy-compatible deps) installed; no version
conflict with `numpy<2`.

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_stage4_smoke.py -v && python -m pytest tests/ -q`
Expected: new smoke tests PASS; full pre-existing suite stays GREEN.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml babysteps/stage4/__init__.py tests/test_stage4_smoke.py
git commit -m "feat(stage4): add babysteps.stage4 package skeleton + scikit-learn optional dep

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Firewall-strict feature extraction

**Files:**
- Create: `babysteps/stage4/features.py`
- Create: `tests/test_stage4_features.py`
- Reference (read-only): `babysteps/schemas.py:148-184` (`DemoEvidence`).

The feature vector per episode is the concatenation of:

1. **Trajectory summary stats** from `record["demo"]["object_trajectory"]`
   (list of `[x, y]` floats):
   - start xy (2 floats)
   - end xy (2 floats)
   - displacement vector (2 floats)
   - displacement L2 norm (1 float)
   - principal-direction angle in radians via `np.arctan2(dy, dx)` (1 float)
   - path length, summed over consecutive segments (1 float)
2. **One-hot `contact_region_label`** over the `CONTACT_REGIONS` whitelist
   from `schemas.py` (6 floats).
3. **One-hot `final_state`** over the `GOAL_STATES` whitelist (4 floats).

Total feature dim = 9 + 6 + 4 = **19 floats per episode**, deterministic order.

The function must **raise** if asked to read a leakage-prone field (defensive
firewall):

```python
def extract_episode_features(record: dict) -> np.ndarray:
    """Return a 19-dim feature vector built ONLY from demo-evidence fields."""
```

- [ ] **Step 1: Write the failing test**

Create `tests/test_stage4_features.py`:

```python
"""Stage-4 feature-extraction tests — firewall-strict by design."""

import numpy as np
import pytest


def _fake_pushcube_record() -> dict:
    return {
        "demo": {
            "camera": "third_person",
            "demonstrator_type": "proxy_oracle",
            "object_trajectory": [[0.0, 0.0], [0.05, 0.0], [0.10, 0.01]],
            "contact_region_label": "minus_x_face",
            "final_state": "cube_at_target",
        },
        "execution": {"initial_intent": {"goal_state": "cube_at_target"}},
        "failure_packet": {"failure_predicate": "approach_blocked"},
        "revision": None,
        "retry": None,
    }


def test_features_shape_is_19():
    from babysteps.stage4.features import extract_episode_features
    feats = extract_episode_features(_fake_pushcube_record())
    assert feats.shape == (19,)
    assert feats.dtype == np.float64


def test_features_are_deterministic_in_order():
    from babysteps.stage4.features import extract_episode_features
    a = extract_episode_features(_fake_pushcube_record())
    b = extract_episode_features(_fake_pushcube_record())
    np.testing.assert_array_equal(a, b)


def test_one_hot_contact_region_matches_whitelist():
    from babysteps.stage4.features import extract_episode_features
    from babysteps.schemas import CONTACT_REGIONS
    feats = extract_episode_features(_fake_pushcube_record())
    one_hot_slice = feats[9:9 + len(CONTACT_REGIONS)]
    assert one_hot_slice.sum() == pytest.approx(1.0)


def test_displacement_norm_is_positive_for_moving_cube():
    from babysteps.stage4.features import extract_episode_features
    feats = extract_episode_features(_fake_pushcube_record())
    assert feats[6] > 0.0  # displacement norm index


def test_firewall_rejects_missing_demo():
    from babysteps.stage4.features import extract_episode_features
    rec = _fake_pushcube_record()
    rec.pop("demo")
    with pytest.raises(KeyError):
        extract_episode_features(rec)


def test_firewall_extractor_does_not_reference_intent_fields():
    """Static check: extractor source must not mention leakage field names."""
    import inspect
    from babysteps.stage4 import features
    src = inspect.getsource(features)
    for forbidden in (
        "initial_intent",
        "failure_packet",
        "revision",
        "retry",
        "oracle_wrong_factor",
        "wrong_factor",
    ):
        assert forbidden not in src, f"firewall violation: {forbidden!r} in features.py"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_stage4_features.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'babysteps.stage4.features'`).

- [ ] **Step 3: Implement the feature extractor**

Create `babysteps/stage4/features.py`:

```python
"""Stage-4 firewall-strict feature extraction.

This module is allowed to read ONLY DemoEvidence-shaped fields. See
babysteps/stage4/__init__.py for the firewall rationale. Adding a reference
to execution.initial_intent / failure_packet / revision / retry / oracle_*
in this file is a milestone-invalidating bug.
"""
from __future__ import annotations

import numpy as np

from babysteps.schemas import CONTACT_REGIONS, GOAL_STATES

_CONTACT_ORDER: tuple[str, ...] = tuple(sorted(CONTACT_REGIONS))
_GOAL_ORDER: tuple[str, ...] = tuple(sorted(GOAL_STATES))

FEATURE_DIM: int = 9 + len(_CONTACT_ORDER) + len(_GOAL_ORDER)


def extract_episode_features(record: dict) -> np.ndarray:
    """Return a deterministic-order feature vector built from demo evidence."""
    demo = record["demo"]
    traj = np.asarray(demo["object_trajectory"], dtype=np.float64)
    if traj.ndim != 2 or traj.shape[1] != 2 or traj.shape[0] < 1:
        raise ValueError(f"object_trajectory must be (T, 2); got {traj.shape}")

    start = traj[0]
    end = traj[-1]
    disp = end - start
    disp_norm = float(np.linalg.norm(disp))
    angle = float(np.arctan2(disp[1], disp[0]))
    path_len = float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))) \
        if traj.shape[0] >= 2 else 0.0

    contact_oh = np.zeros(len(_CONTACT_ORDER), dtype=np.float64)
    contact_oh[_CONTACT_ORDER.index(demo["contact_region_label"])] = 1.0

    goal_oh = np.zeros(len(_GOAL_ORDER), dtype=np.float64)
    goal_oh[_GOAL_ORDER.index(demo["final_state"])] = 1.0

    return np.concatenate([
        start.astype(np.float64),
        end.astype(np.float64),
        disp.astype(np.float64),
        np.array([disp_norm, angle, path_len], dtype=np.float64),
        contact_oh,
        goal_oh,
    ])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_stage4_features.py -v`
Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add babysteps/stage4/features.py tests/test_stage4_features.py
git commit -m "feat(stage4): firewall-strict feature extractor over demo evidence

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Linear probe with 5-fold CV and two reference baselines

**Files:**
- Create: `babysteps/stage4/probe.py`
- Create: `tests/test_stage4_probe.py`

API:

```python
def train_probe(X: np.ndarray, y: np.ndarray, *, seed: int = 0) -> dict:
    """Return {'probe_acc_mean', 'probe_acc_std', 'majority_class_acc',
    'shuffled_features_acc', 'n_unique_labels', 'n_episodes'}.

    - Logistic regression (multinomial, lbfgs, max_iter=1000).
    - 5-fold StratifiedKFold; falls back to LeaveOneOut if any class has < 5.
    - majority_class_acc: most-common-label fraction (chance baseline).
    - shuffled_features_acc: same CV protocol on row-shuffled X.
    - n_unique_labels == 1 short-circuits to {'probe_acc_mean': 1.0, ...,
      'trivially_constant': True}.
    """
```

- [ ] **Step 1: Write the failing test**

Create `tests/test_stage4_probe.py`:

```python
"""Stage-4 probe smoke tests on synthetic linearly-separable data."""

import numpy as np


def _linearly_separable(n: int = 60, d: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    w = rng.standard_normal(d)
    y = (X @ w > 0).astype(int)
    return X, y


def test_probe_recovers_linear_relationship():
    from babysteps.stage4.probe import train_probe
    X, y = _linearly_separable()
    out = train_probe(X, y, seed=0)
    assert out["n_unique_labels"] == 2
    assert out["probe_acc_mean"] > 0.85
    assert out["shuffled_features_acc"] < out["probe_acc_mean"]


def test_probe_handles_constant_label():
    from babysteps.stage4.probe import train_probe
    X = np.zeros((10, 4))
    y = np.ones(10, dtype=int)
    out = train_probe(X, y)
    assert out["n_unique_labels"] == 1
    assert out["probe_acc_mean"] == 1.0
    assert out["trivially_constant"] is True


def test_probe_majority_class_baseline_is_correct():
    from babysteps.stage4.probe import train_probe
    X = np.zeros((10, 3))
    y = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
    out = train_probe(X, y)
    assert out["majority_class_acc"] == 0.7
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_stage4_probe.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement the probe**

Create `babysteps/stage4/probe.py`:

```python
"""Stage-4 linear probe with chance + shuffled-features baselines."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_score


def _make_splitter(y: np.ndarray) -> Any:
    _, counts = np.unique(y, return_counts=True)
    if counts.min() < 5:
        return LeaveOneOut()
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=0)


def train_probe(X: np.ndarray, y: np.ndarray, *, seed: int = 0) -> dict:
    n_episodes = int(X.shape[0])
    n_unique = int(np.unique(y).size)

    if n_unique <= 1:
        return {
            "n_episodes": n_episodes,
            "n_unique_labels": n_unique,
            "probe_acc_mean": 1.0,
            "probe_acc_std": 0.0,
            "majority_class_acc": 1.0,
            "shuffled_features_acc": 1.0,
            "trivially_constant": True,
        }

    splitter = _make_splitter(y)
    clf = LogisticRegression(max_iter=1000, multi_class="auto", solver="lbfgs")

    probe_scores = cross_val_score(clf, X, y, cv=splitter, scoring="accuracy")

    rng = np.random.default_rng(seed)
    X_shuf = X.copy()
    rng.shuffle(X_shuf)
    shuf_scores = cross_val_score(clf, X_shuf, y, cv=splitter, scoring="accuracy")

    _, counts = np.unique(y, return_counts=True)
    majority = float(counts.max() / counts.sum())

    return {
        "n_episodes": n_episodes,
        "n_unique_labels": n_unique,
        "probe_acc_mean": float(probe_scores.mean()),
        "probe_acc_std": float(probe_scores.std()),
        "majority_class_acc": majority,
        "shuffled_features_acc": float(shuf_scores.mean()),
        "trivially_constant": False,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_stage4_probe.py -v`
Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add babysteps/stage4/probe.py tests/test_stage4_probe.py
git commit -m "feat(stage4): linear probe with CV + chance + shuffled-features baselines

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Per-task per-factor report aggregation + Markdown rendering

**Files:**
- Create: `babysteps/stage4/report.py`
- Create: `tests/test_stage4_report.py`

API:

```python
def build_report(rows: list[dict]) -> dict:
    """rows is a list of probe outputs annotated with 'task' and 'factor'.

    Returns a nested dict keyed by [task][factor] -> probe output, plus a
    top-level 'gate' summary: {'n_total', 'n_trivial', 'n_passing',
    'n_failing', 'failing_cells': [(task, factor), ...]}.
    """


def markdown_table(report: dict) -> str:
    """Render report['by_task'] as a per-task Markdown table for humans."""
```

- [ ] **Step 1: Write the failing test**

Create `tests/test_stage4_report.py`:

```python
"""Stage-4 report aggregation + Markdown rendering."""


def _row(task, factor, acc, n_unique=2, trivial=False):
    return {
        "task": task, "factor": factor,
        "probe_acc_mean": acc, "probe_acc_std": 0.0,
        "majority_class_acc": 0.5, "shuffled_features_acc": 0.5,
        "n_unique_labels": n_unique, "n_episodes": 24,
        "trivially_constant": trivial,
    }


def test_build_report_groups_by_task_then_factor():
    from babysteps.stage4.report import build_report
    rows = [
        _row("PushCube-v1", "approach_direction", 0.95),
        _row("PushCube-v1", "goal_state", 1.0, n_unique=1, trivial=True),
        _row("PickCube-v1", "contact_region", 0.80),
    ]
    rep = build_report(rows)
    assert "PushCube-v1" in rep["by_task"]
    assert "PickCube-v1" in rep["by_task"]
    assert rep["gate"]["n_total"] == 3
    assert rep["gate"]["n_trivial"] == 1
    assert rep["gate"]["n_passing"] == 1   # 0.95 ≥ 0.90
    assert rep["gate"]["n_failing"] == 1   # 0.80 < 0.90
    assert ("PickCube-v1", "contact_region") in rep["gate"]["failing_cells"]


def test_markdown_table_contains_task_headers_and_numbers():
    from babysteps.stage4.report import build_report, markdown_table
    rep = build_report([_row("PushCube-v1", "approach_direction", 0.95)])
    md = markdown_table(rep)
    assert "PushCube-v1" in md
    assert "approach_direction" in md
    assert "0.95" in md
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_stage4_report.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement the report module**

Create `babysteps/stage4/report.py`. Key rules:

- Gate threshold is hard-coded as `0.90` (matches `goal.md` Stage-4 Success
  Criteria). A `(task, factor)` is `failing` iff
  `not trivially_constant and probe_acc_mean < 0.90`.
- `markdown_table` renders one section per task, with columns:
  `factor | n_unique | n_episodes | majority | shuffled | probe ± std | gate`.
  `gate` cell is one of `PASS` / `FAIL` / `trivial`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_stage4_report.py -v`
Expected: all 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add babysteps/stage4/report.py tests/test_stage4_report.py
git commit -m "feat(stage4): per-task per-factor report aggregation + markdown table

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: CLI wiring

**Files:**
- Create: `scripts/stage4_probe_schema_recoverability.py`
- Reference (read-only): `scripts/stage0_summarize.py` for the project's CLI
  style.

CLI contract:

```text
python scripts/stage4_probe_schema_recoverability.py \
    --jsonl datasets/stage0_baselines/babysteps_selective/PushCube-v1/samples.jsonl \
    --jsonl datasets/stage0_baselines/babysteps_selective/PickCube-v1/samples.jsonl \
    --jsonl datasets/stage0_baselines/babysteps_selective/StackCube-v1/samples.jsonl \
    --out-dir reports/stage4/schema_recoverability/ \
    --seed 0
```

- Each `--jsonl` path is read line-by-line into a list of `EpisodeRecord`
  dicts via `EpisodeRecord.from_jsonl_line(line).to_dict()` (round-trips
  through the snapshot guard).
- Episodes are grouped by `record["task"]`.
- For each `task` and each of the six discrete factors
  (`INTENT_FIELDS` from `schemas.py`), the CLI:
  1. Builds `X` by stacking `extract_episode_features(rec)` rows.
  2. Builds `y` by reading `record["execution"]["initial_intent"][factor]`
     (string label).
  3. Encodes `y` via `sklearn.preprocessing.LabelEncoder` (deterministic
     `classes_` ordering).
  4. Calls `train_probe(X, y_encoded)` and annotates with
     `task` and `factor`.
- After all rows, calls `build_report(rows)` and writes:
  - `reports/stage4/schema_recoverability/schema_recoverability.json`
    (the full nested dict)
  - `reports/stage4/schema_recoverability/schema_recoverability.md`
    (the rendered Markdown table)
- Exits 0 if `gate["n_failing"] == 0`, else exits 0 anyway but prints a
  one-line warning to stderr listing the failing cells. The acceptance gate
  is enforced by Task 6's `notes.md`, not by the CLI exit code (because some
  failures are expected and documented as such).

- [ ] **Step 1: Run a dry import check before committing**

Run: `python -c "import importlib.util as u; s=u.spec_from_file_location('m','scripts/stage4_probe_schema_recoverability.py'); m=u.module_from_spec(s); s.loader.exec_module(m)"`
Expected: no `ImportError`.

- [ ] **Step 2: Execute the CLI end-to-end on a single task**

Run:

```bash
mkdir -p reports/stage4/schema_recoverability
python scripts/stage4_probe_schema_recoverability.py \
    --jsonl datasets/stage0_baselines/babysteps_selective/PushCube-v1/samples.jsonl \
    --out-dir reports/stage4/schema_recoverability/
```

Expected: writes `schema_recoverability.json` and `schema_recoverability.md`.
Inspect both for sanity (no NaNs, no negative accuracies, one section per
task in the Markdown).

- [ ] **Step 3: Commit**

```bash
git add scripts/stage4_probe_schema_recoverability.py
git commit -m "feat(stage4): CLI runner for schema-recoverability probe

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Produce the gate artifact (the actual report + interpretation)

**Files:**
- Generated: `reports/stage4/schema_recoverability/schema_recoverability.json`
- Generated: `reports/stage4/schema_recoverability/schema_recoverability.md`
- Authored: `reports/stage4/schema_recoverability/notes.md`

- [ ] **Step 1: Run the full probe across the primary baseline**

```bash
python scripts/stage4_probe_schema_recoverability.py \
    --jsonl datasets/stage0_baselines/babysteps_selective/PushCube-v1/samples.jsonl \
    --jsonl datasets/stage0_baselines/babysteps_selective/PickCube-v1/samples.jsonl \
    --jsonl datasets/stage0_baselines/babysteps_selective/StackCube-v1/samples.jsonl \
    --out-dir reports/stage4/schema_recoverability/ \
    --seed 0
```

- [ ] **Step 2: Run the union probe (primary + oracle baseline) for
  variance-bound sanity check**

```bash
python scripts/stage4_probe_schema_recoverability.py \
    --jsonl datasets/stage0_baselines/babysteps_selective/PushCube-v1/samples.jsonl \
    --jsonl datasets/stage0_baselines/babysteps_selective/PickCube-v1/samples.jsonl \
    --jsonl datasets/stage0_baselines/babysteps_selective/StackCube-v1/samples.jsonl \
    --jsonl datasets/stage0_baselines/oracle_factor_revision/PushCube-v1/samples.jsonl \
    --jsonl datasets/stage0_baselines/oracle_factor_revision/PickCube-v1/samples.jsonl \
    --jsonl datasets/stage0_baselines/oracle_factor_revision/StackCube-v1/samples.jsonl \
    --out-dir reports/stage4/schema_recoverability_union/ \
    --seed 0
```

If the union's per-cell numbers move by more than ±5 pp relative to the
primary, note that in `notes.md` — it indicates the demo-evidence
distribution is baseline-coupled (which it should not be, since DemoEvidence
is supposed to be baseline-independent).

- [ ] **Step 3: Author `notes.md`**

Create `reports/stage4/schema_recoverability/notes.md` with this structure:

```markdown
# Stage-4 Milestone 1 — Schema Recoverability Notes

## Run metadata
- Date: <today>
- Primary input JSONLs: <list>
- Commit at run time: <git rev-parse HEAD>
- Probe: sklearn LogisticRegression, multinomial, lbfgs, 5-fold stratified CV
  (or LOO when min-class count < 5), seed=0.
- Feature dim: 19 (see `babysteps/stage4/features.py`).

## Per-cell summary
<paste the rendered schema_recoverability.md table here, or link to it>

## Cells that passed the 90% gate
<bullet list>

## Cells that did not pass — and why
For each non-trivial cell with probe_acc_mean < 0.90, write one paragraph
naming the (task, factor), citing the actual number, and giving a defensible
structural reason. Acceptable categories:

- "Factor is determined by a privileged scene field (e.g., blocked_sides)
  that is not in DemoEvidence; a learned encoder over RGB will need to
  recover that field too." (→ implication: probe upper bound here requires
  a video-conditioned encoder, not stricter labelling.)
- "Factor depends on multi-step history; single-trajectory summary stats
  drop the relevant information. A sequence model is required." (→
  implication: Stage-4 IntentHead should be sequence-aware.)
- "Factor labelling is ambiguous given DemoEvidence alone; two different
  intents produce the same demo trace. The schema needs disambiguation."
  (→ implication: schema fix before encoder work.)

## Trivially constant cells (informational, not gated)
<bullet list of (task, factor) pairs where n_unique_labels == 1>

## Implication for Stage-4 Milestone 2
Pick ONE:
(a) Proceed to learned slot encoder spike (M2).
(b) Revise <factor> labelling on <task> before M2.
(c) Reframe Stage-4 success criterion for <factor>: probe from
    (G_t, language) instead of G_t alone.

Justify the pick in one paragraph, citing specific cell numbers.
```

- [ ] **Step 4: Commit the gate artifacts**

```bash
git add reports/stage4/schema_recoverability/
git commit -m "report(stage4 m1): schema-recoverability probe results + interpretation notes

- per-task per-factor probe accuracy table (json + md)
- written interpretation in notes.md picking the M2 entry direction

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Final verification

- [ ] **Step 1: Run the full pre-existing test suite**

Run: `python -m pytest tests/ -q`
Expected: all 302+ tests still pass; the three new test files (smoke,
features, probe, report) add their own count.

- [ ] **Step 2: Verify the gate by re-reading `notes.md`**

Check that every non-trivial cell with `probe_acc_mean < 0.90` in
`schema_recoverability.json` has a corresponding paragraph in
`notes.md`. If any are missing, write them and re-commit before declaring
the milestone done.

- [ ] **Step 3: Update `CODE_MAP.md`**

Add a one-line entry under the "`babysteps/`" table for the new
`stage4/` subpackage and under "`scripts/`" for the new CLI, matching the
existing style.

- [ ] **Step 4: Commit the map update**

```bash
git add CODE_MAP.md
git commit -m "docs(stage4): note stage4 probe subpackage + CLI in CODE_MAP

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Out of scope (do NOT do this milestone)

- Do **not** train any encoder (Slot Attention, SAVi, DINOv2, etc.). That is
  Stage-4 Milestone 2.
- Do **not** train any ReviseHead or action decoder. That is Milestones 3+.
- Do **not** modify `babysteps/{schemas,demo,episode,failure,revision,eval}.py`
  or any file under `babysteps/envs/`, `babysteps/render/`, `babysteps/skills/`.
  Stage-0 code is read-only for this milestone.
- Do **not** add the probe CLI to the main test suite as an integration
  test — it reads real data files and is run on demand from `scripts/`.
- Do **not** add `scikit-learn` as a *required* dependency. It is `analysis`-
  optional only.
- Do **not** read `record["execution"]["initial_intent"]` inside
  `features.py` even briefly. That is the *label*. The static firewall test
  in Task 2 enforces this.
- Do **not** add a GPU code path. This entire milestone runs on the login
  node. If `nvidia-smi` is invoked or `torch.cuda.*` is imported anywhere,
  something has gone wrong.
- Do **not** collect new Stage-0 episodes for this milestone. The existing
  `datasets/stage0_baselines/` data is sufficient.

---

## Risks and how to handle them

1. **Small per-task sample size (~24 episodes per baseline).** 5-fold CV may
   collapse to LeaveOneOut when min-class count < 5. The probe module
   handles that branch explicitly; report it in `notes.md` if it triggers
   for any cell.
2. **Class imbalance.** `majority_class_acc` is reported alongside
   `probe_acc_mean`. If a probe's accuracy is high but only matches the
   majority, that is recorded in the JSON and should be flagged in
   `notes.md` as effectively trivial.
3. **Constant labels within a task.** Some factors are constant per task
   (e.g., `goal_state` for PushCube is always `cube_at_target`). The probe
   short-circuits these with `trivially_constant: True` and does *not* count
   them toward the gate.
4. **Label leakage.** The static firewall test in Task 2 prevents most
   leakage. If a new feature is added that depends on a leakage-prone field,
   the test will fail.
5. **Distribution shift between baselines.** Task 6 Step 2 explicitly checks
   for this by running the probe on a union of primary + oracle baselines
   and comparing per-cell numbers.

---

## Verification of milestone completion

The milestone is **achieved** iff all of the following hold:

- All 7 tasks above are checked off and committed.
- `python -m pytest tests/ -q` exits 0.
- `reports/stage4/schema_recoverability/schema_recoverability.json` and
  `schema_recoverability.md` exist and contain at least 3 tasks × 6 factors
  = 18 cells (plus the 7th `direction_grounding` factor when applicable).
- `reports/stage4/schema_recoverability/notes.md` exists and ends with a
  clear pick of (a), (b), or (c) for Milestone 2 entry.
- Every non-trivial sub-90% cell has a written paragraph in `notes.md`.
- `CODE_MAP.md` references the new `babysteps/stage4/` subpackage and the
  new CLI under `scripts/`.

The milestone **is not achieved** if any non-trivial cell fails the 90% gate
without a written explanation, or if any Stage-0 test snapshot breaks.

---

## Estimated effort

One long focused session (≈4–8 hours), assuming the worker is comfortable
with sklearn and the existing `EpisodeRecord` schema. Most of the time is in
Task 6 — writing `notes.md` requires reading the actual probe numbers and
thinking about what they imply for Stage-4 M2.
