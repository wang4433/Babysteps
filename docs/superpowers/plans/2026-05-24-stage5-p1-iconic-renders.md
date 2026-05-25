# Stage-5 P1 Iconic Per-Policy PushCube Renders — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce 20 self-contained "full episode" MP4s — 5 held-out seeds × 4 retry policies — that visually validate the Stage-5 P1 PushCube setup and let a reviewer compare each policy on the same seed.

**Architecture:** Add a `render_policy_episode()` helper to `babysteps/render/pushcube.py` (modeled on the existing `render_baseline_contrast`, but parameterized on a single `RetryPolicy` callable + optional `demo_features` provider). Add a CLI driver that loops over (seed, policy) and writes one concatenated MP4 per pair. Plus a tiny sim-free smoke test and an sbatch wrapper.

**Tech Stack:** Python 3, ManiSkill (GPU/Vulkan), PyTorch (DINOv2 features + LatentPack), numpy, pytest.

**Spec:** `docs/superpowers/specs/2026-05-24-stage5-p1-iconic-renders-design.md`

---

## File Structure

| File | Action | Responsibility |
| --- | --- | --- |
| `babysteps/render/pushcube.py` | Modify (append) | Add `render_policy_episode()` helper (~80 LOC). Reuses `_pushcube_setup`, `_execute_push`, `_get_or_build_obstacle`, `_move_obstacle_to_block`, `_park_obstacle`. |
| `tests/test_render_modules.py` | Modify (append) | Add `test_pushcube_render_policy_episode_smoke()` exercising the helper with the existing `_StubEnv` + `PushCubeAdapter` (sim-free). |
| `scripts/render_stage5_p1_iconic.py` | Create | CLI: loads `PushCubeAdapter`, real ManiSkill PushCube env, `LatentPack`, cached DINOv2 features; loops (seed × policy); writes one MP4 per pair. |
| `slurm/render_stage5_p1_iconic.sbatch` | Create | GPU+Vulkan sbatch wrapper around the CLI. |

No existing files have their behavior changed — `render_baseline_contrast` and all current callers are untouched.

---

## Task 1: Sim-free smoke test for `render_policy_episode` (TDD red)

**Files:**
- Modify: `tests/test_render_modules.py` (append after `test_pushcube_baseline_contrast_perturbs_contact_region`, around line 242)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_render_modules.py` (after the existing PushCube baseline-contrast tests, before the `# ---------- PickCube render tests ---` divider at line ~244):

```python
def test_pushcube_render_policy_episode_smoke_same_intent():
    """render_policy_episode runs the full demo + blocked + retry flow for a
    single policy and returns one flat concatenated frame list plus a single
    (title, subtitle) tuple. Exercised with same_intent_retry because it
    needs no LatentPack / demo_features and the result is deterministic on
    the stub env."""
    from babysteps.render.pushcube import render_policy_episode
    from babysteps.envs.pushcube_adapter import PushCubeAdapter
    from babysteps.policies import same_intent_retry

    frames, title = render_policy_episode(
        _StubEnv(),
        PushCubeAdapter(),
        seed=0,
        policy_name="same_intent_retry",
        policy_callable=same_intent_retry,
        demo_features_provider=None,
        fps=4,
    )
    assert isinstance(frames, list)
    assert len(frames) >= 3  # at least one frame per phase
    assert isinstance(title, tuple) and len(title) == 2
    assert "same_intent_retry" in title[0]
    assert "seed 0000" in title[0]


def test_pushcube_render_policy_episode_smoke_oracle():
    """oracle_factor_revision path: also no demo_features needed. Asserts the
    title encodes the policy name and the frame list is non-empty."""
    from babysteps.render.pushcube import render_policy_episode
    from babysteps.envs.pushcube_adapter import PushCubeAdapter
    from babysteps.policies import oracle_factor_revision

    frames, title = render_policy_episode(
        _StubEnv(),
        PushCubeAdapter(),
        seed=0,
        policy_name="oracle_factor_revision",
        policy_callable=oracle_factor_revision,
        demo_features_provider=None,
        fps=4,
    )
    assert len(frames) >= 3
    assert "oracle_factor_revision" in title[0]


def test_pushcube_render_policy_episode_passes_demo_features_to_policy():
    """When demo_features_provider is supplied, render_policy_episode must
    pass its output through to the policy via RetryContext.demo_features.
    Verified with a stub policy that captures its ctx and returns a no-op
    same-intent revision so the rest of the flow runs cleanly."""
    from babysteps.render.pushcube import render_policy_episode
    from babysteps.envs.pushcube_adapter import PushCubeAdapter
    from babysteps.policies import same_intent_retry
    import numpy as np

    captured = {}
    sentinel = np.array([0.5, -0.25, 1.0], dtype=np.float32)

    def capturing_policy(ctx):
        captured["demo_features"] = ctx.demo_features
        captured["failure_predicate"] = ctx.failure_predicate
        return same_intent_retry(ctx)

    _, _ = render_policy_episode(
        _StubEnv(),
        PushCubeAdapter(),
        seed=0,
        policy_name="latent",
        policy_callable=capturing_policy,
        demo_features_provider=lambda seed: sentinel,
        fps=4,
    )
    assert captured["demo_features"] is sentinel
    # PushCube blocked-approach predicate must be plumbed for latent.
    assert captured["failure_predicate"] is not None
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `python -m pytest tests/test_render_modules.py::test_pushcube_render_policy_episode_smoke_same_intent tests/test_render_modules.py::test_pushcube_render_policy_episode_smoke_oracle tests/test_render_modules.py::test_pushcube_render_policy_episode_passes_demo_features_to_policy -v`

Expected: `ImportError` or `AttributeError` for `render_policy_episode` — function does not exist yet. All three FAIL.

- [ ] **Step 3: Commit the red tests**

```bash
git add tests/test_render_modules.py
git commit -m "test(stage5 p1): red smoke tests for render_policy_episode helper

Three sim-free tests against the existing _StubEnv + PushCubeAdapter:
1. same_intent_retry path (no demo_features) returns frames + title.
2. oracle_factor_revision path (no demo_features) returns frames + title.
3. demo_features_provider sentinel reaches the policy via RetryContext.

Will pass once render_policy_episode() lands in babysteps/render/pushcube.py.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Implement `render_policy_episode()` (TDD green)

**Files:**
- Modify: `babysteps/render/pushcube.py` (append after `render_baseline_contrast`, around line 447)

- [ ] **Step 1: Add the helper**

Append to `babysteps/render/pushcube.py` after the existing `render_baseline_contrast` function (after line 446, after the closing `)` of its return statement):

```python
def render_policy_episode(
    env,
    adapter: BaseTaskAdapter,
    seed: int,
    *,
    policy_name: str,
    policy_callable,
    demo_features_provider=None,
    fps: int = 20,
) -> tuple[list, tuple[str, str]]:
    """Render one continuous PushCube episode under a single retry policy.

    Phases run in order and frames are concatenated into one list:
      1. demo (oracle correct intent, no obstacle)
      2. attempt_blocked (initial intent + wall in place → arm stalls)
      3. retry (policy_callable's revised intent vs the same wall)

    Returns (frames, (title, subtitle)) where the title encodes seed +
    policy_name and the subtitle records the revision (or "no_revision"
    for one_shot-style policies). Designed for the Stage-5 P1 iconic
    contrast renders, where the same (seed, demo, attempt) prefix is
    re-rendered per policy so each MP4 is a self-contained "full episode"
    clip.

    Parameters
    ----------
    policy_name : str
        Short tag burned into the title (e.g. "latent",
        "oracle_factor_revision", "babysteps_selective",
        "same_intent_retry"). Identity only — does not switch behaviour.
    policy_callable : RetryPolicy
        A `(RetryContext) -> Optional[(Intent, Revision)]` function. May
        be the LatentPack closure from
        `babysteps.stage4.latent_policy.latent_revision_factory`.
    demo_features_provider : Optional[Callable[[int], Any]]
        If provided, `demo_features_provider(seed)` is called and the
        result is attached to `RetryContext.demo_features`. Required for
        the latent policy; pass None for all others.

    Notes
    -----
    `fps` is accepted for signature parity with `render_episode` and
    `render_baseline_contrast`; frame capture cadence is governed by
    `_execute_push`, not by this argument.
    """
    short_id = f"seed {seed:04d}"

    # Spawn / park obstacle (no-op if already cached on env).
    obstacle = _get_or_build_obstacle(env)
    _park_obstacle(obstacle)

    s = _pushcube_setup(env, adapter, seed)
    correct_intent = s["correct_intent"]
    initial_intent = s["initial_intent"]
    scene_exec = s["scene_exec"]
    attribution = s["attribution"]
    demo_frames = s["demo_frames"]

    # === Phase 2 — initial intent vs the wall ===
    _move_obstacle_to_block(
        obstacle, s["scene"].cube_xy, s["scene"].cube_z, initial_intent,
    )
    wp_attempt = build_push_waypoints(scene_exec, initial_intent)
    attempt_frames: list = []
    _ = _execute_push(
        env, wp_attempt, attempt_frames, seed=seed,
        capture=render_wrist_frame,
        max_steps=120,
        no_progress_break_steps=20,
        no_progress_eps_m=0.002,
    )
    # Note: leave the wall in place for the retry — every policy must
    # face the same physical obstacle.

    # === Phase 3 — policy retry ===
    fp = adapter.build_failure_packet(
        initial_intent,
        AttemptResult(
            initial_obj_xy=s["scene"].cube_xy,
            final_obj_xy=s["scene"].cube_xy,
            goal_xy=s["scene"].goal_xy,
            reached_contact=False, object_moved=False,
            planner_failed=True, collision=False, grasp_slip=False,
            rollout_log_path=None, success=False,
        ),
        scene_exec,
    )
    demo_features = (
        demo_features_provider(seed) if demo_features_provider is not None else None
    )
    ctx = RetryContext(
        initial_intent=initial_intent,
        attribution=attribution,
        scene=scene_exec,
        oracle_correct_intent=adapter.oracle_correct_intent(scene_exec),
        oracle_wrong_factor=adapter.oracle_wrong_factor(initial_intent, scene_exec),
        task_valid_tokens=adapter.task_valid_tokens(),
        rng=random.Random(seed),
        revise_fn=adapter.revise_intent,
        demo_features=demo_features,
        failure_predicate=fp.failure_predicate,
        failure_packet=fp,
    )
    out = policy_callable(ctx)
    retry_frames: list = []
    if out is None:
        # one_shot-style: no retry. Capture a single still frame so the
        # concatenated clip ends cleanly rather than truncating mid-stream.
        retry_frames.append(render_wrist_frame(env))
        retry_intent = initial_intent
        revision_subtitle = "no_revision (one_shot)"
        retry_success = False
    else:
        retry_intent, revision = out
        out_exec = _execute_push(
            env,
            build_push_waypoints(scene_exec, retry_intent),
            retry_frames,
            seed=seed,
            capture=render_wrist_frame,
        )
        retry_success = bool(out_exec["success"])
        if revision.factor == "none":
            revision_subtitle = "no_revision (same_intent_retry)"
        else:
            revision_subtitle = (
                f"{revision.factor}: {revision.old_value} → {revision.new_value}"
            )

    title = (
        f"{short_id}  policy: {policy_name}  (success={retry_success})",
        f"demo({correct_intent.contact_region}/{correct_intent.approach_direction})"
        f"  →  blocked({initial_intent.approach_direction})"
        f"  →  retry({revision_subtitle})",
    )
    frames = list(demo_frames) + list(attempt_frames) + list(retry_frames)
    return frames, title
```

- [ ] **Step 2: Run the three new tests to confirm green**

Run: `python -m pytest tests/test_render_modules.py::test_pushcube_render_policy_episode_smoke_same_intent tests/test_render_modules.py::test_pushcube_render_policy_episode_smoke_oracle tests/test_render_modules.py::test_pushcube_render_policy_episode_passes_demo_features_to_policy -v`

Expected: 3 PASSED.

- [ ] **Step 3: Run the full sim-free suite to verify no regression**

Run: `python -m pytest tests/ -q`

Expected: All previously-passing tests still pass; total grows by 3.

- [ ] **Step 4: Commit**

```bash
git add babysteps/render/pushcube.py
git commit -m "feat(stage5 p1): render_policy_episode helper for per-policy contrast

Concatenates demo + blocked + this-policy retry into one frame list per
(seed, policy). Parameterised on a single RetryPolicy callable plus an
optional demo_features provider, so the same helper works for latent,
oracle_factor_revision, babysteps_selective, and same_intent_retry.

Reuses _pushcube_setup / _execute_push / obstacle helpers — does not
touch the existing render_episode or render_baseline_contrast paths.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: CLI driver `scripts/render_stage5_p1_iconic.py`

**Files:**
- Create: `scripts/render_stage5_p1_iconic.py`

- [ ] **Step 1: Create the CLI**

Write the file:

```python
#!/usr/bin/env python
"""Stage-5 P1 — iconic per-policy PushCube render contrast.

For each (seed, policy) pair, emits one MP4 containing demo + initial
blocked attempt + that policy's retry, concatenated into a single
"full episode" clip. Designed to populate the empty 'videos' column of
the Stage-5 P1 PushCube held-out report gallery at
reports/stage5/p1_vision_g4_g5/PushCube-v1/report_gallery/.

Output filename: pushcube_seed_NNNN__<policy>_full.mp4

Default scope (5 iconic seeds × 4 policies = 20 MP4s):
  seeds    : 100, 110, 120, 130, 143
             (100/110/120/130: clean wins for all but same_intent_retry;
              143: latent fails, oracle/babysteps succeed)
  policies : latent, oracle_factor_revision,
             babysteps_selective, same_intent_retry

Needs Vulkan. On the Gilbreth login node it falls back to Mesa lavapipe
(slow); on a GPU node it uses the NVIDIA Vulkan ICD.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Make the project root importable without `pip install -e .`.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from babysteps.envs.task_registry import get_task_entry  # noqa: E402
from babysteps.policies import (  # noqa: E402
    babysteps_selective, oracle_factor_revision, same_intent_retry,
)
from babysteps.render.common import annotate_frame, save_mp4  # noqa: E402
from babysteps.render.pushcube import render_policy_episode  # noqa: E402
from babysteps.stage4.latent_policy import (  # noqa: E402
    latent_revision_factory, load_latent_pack,
)


DEFAULT_SEEDS = (100, 110, 120, 130, 143)
DEFAULT_POLICIES = (
    "latent", "oracle_factor_revision",
    "babysteps_selective", "same_intent_retry",
)


def _parse_seeds(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _parse_policies(s: str) -> list[str]:
    out = []
    for name in s.split(","):
        name = name.strip()
        if not name:
            continue
        if name not in DEFAULT_POLICIES:
            raise ValueError(
                f"Unknown policy {name!r}. Valid: {DEFAULT_POLICIES}"
            )
        out.append(name)
    return out


def _make_features_provider(features_dir: Path):
    """Return a callable(seed) -> np.ndarray reading cached DINOv2 features."""
    features_dir = Path(features_dir)

    def _provider(seed: int):
        path = features_dir / f"seed_{seed:04d}_dinov2.npy"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing cached DINOv2 feature for seed {seed}: {path}"
            )
        return np.load(path)

    return _provider


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS),
                   help="Comma-separated seed list (default: 5 iconic seeds).")
    p.add_argument("--policies", default=",".join(DEFAULT_POLICIES),
                   help="Comma-separated policy list (default: all 4).")
    p.add_argument("--pack-dir", type=Path,
                   default=Path("models/stage5/p1_vision/PushCube-v1"),
                   help="LatentPack directory (intent_head.pt / revise_head.pt / centroids.npz / meta.json).")
    p.add_argument("--features-dir", type=Path,
                   default=Path("datasets/stage5/varied_intent/PushCube-v1/features"),
                   help="Directory of cached DINOv2 seed_NNNN_dinov2.npy files.")
    p.add_argument("--out-dir", type=Path,
                   default=Path("renders/stage5_p1_iconic/pushcube"),
                   help="Output directory for MP4s.")
    p.add_argument("--fps", type=int, default=20)
    args = p.parse_args(argv)

    seeds = _parse_seeds(args.seeds)
    policies = _parse_policies(args.policies)

    try:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401 — registers PushCube-v1
    except ImportError as exc:
        print(f"ManiSkill import failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    # Validate feature presence before launching env (cheap fail-fast).
    features_provider = _make_features_provider(args.features_dir)
    if "latent" in policies:
        for s in seeds:
            _ = features_provider(s)  # raises FileNotFoundError on miss

    # Load LatentPack once (only needed if 'latent' is in the policy list,
    # but loading is cheap and keeps the code branchless).
    if "latent" in policies:
        pack = load_latent_pack(args.pack_dir)
        latent_policy = latent_revision_factory(pack)
    else:
        latent_policy = None

    policy_callables = {
        "latent": latent_policy,
        "oracle_factor_revision": oracle_factor_revision,
        "babysteps_selective": babysteps_selective,
        "same_intent_retry": same_intent_retry,
    }
    policy_provider = {
        "latent": features_provider,
        "oracle_factor_revision": None,
        "babysteps_selective": None,
        "same_intent_retry": None,
    }

    entry = get_task_entry("PushCube-v1")
    adapter = entry.adapter_cls()
    # Mirror render_baseline_contrast.py:58–63 exactly — the obstacle helper
    # mutates a static actor's pose, which only works on the CPU sim backend,
    # and the push waypoints assume the pd_ee_delta_pose action shape.
    env = gym.make(
        adapter.gym_env_id,
        obs_mode="state_dict",
        control_mode="pd_ee_delta_pose",
        sim_backend="cpu",
        render_mode="rgb_array",
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    n_total = len(seeds) * len(policies)
    written: list[Path] = []
    for seed in seeds:
        for pol in policies:
            print(f"--- seed {seed:04d} × policy {pol} "
                  f"({len(written) + 1}/{n_total}) ---", flush=True)
            frames, title = render_policy_episode(
                env, adapter, seed,
                policy_name=pol,
                policy_callable=policy_callables[pol],
                demo_features_provider=policy_provider[pol],
                fps=args.fps,
            )
            annotated = [annotate_frame(f, title[0], title[1]) for f in frames]
            out_path = args.out_dir / f"pushcube_seed_{seed:04d}__{pol}_full.mp4"
            save_mp4(annotated, out_path, fps=args.fps)
            written.append(out_path)
            print(f"  wrote {out_path}  ({len(annotated)} frames)", flush=True)

    env.close()
    print(f"\nDone. {len(written)} MP4s under {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Smoke-check `--help` parses (no GPU needed)**

Run: `python scripts/render_stage5_p1_iconic.py --help`

Expected: Argparse help text prints, exit code 0. (ManiSkill import is inside `main()` past arg parsing, so `--help` short-circuits before any GPU touch.)

- [ ] **Step 3: Sim-free unused-import + syntax sanity**

Run: `python -c "import ast; ast.parse(open('scripts/render_stage5_p1_iconic.py').read())"`

Expected: No output (success). Then: `python -m pyflakes scripts/render_stage5_p1_iconic.py || true` — informational only; tolerate `F401` for `gym`/`mani_skill.envs` (the import is required for env registration as a side effect).

- [ ] **Step 4: Commit**

```bash
git add scripts/render_stage5_p1_iconic.py
git commit -m "feat(stage5 p1): CLI for iconic per-policy PushCube renders

Loops 5 iconic seeds (100,110,120,130,143) × 4 policies (latent,
oracle_factor_revision, babysteps_selective, same_intent_retry).
Loads LatentPack from models/stage5/p1_vision/PushCube-v1 and pulls
cached DINOv2 features from datasets/stage5/varied_intent/PushCube-v1/features.
Writes 20 self-contained 'full episode' MP4s under
renders/stage5_p1_iconic/pushcube/.

Validates cached-feature presence per seed before touching the env so
the run fails fast on a missing feature.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Sbatch wrapper `slurm/render_stage5_p1_iconic.sbatch`

**Files:**
- Create: `slurm/render_stage5_p1_iconic.sbatch`

- [ ] **Step 1: Write the sbatch script**

Write the file (modelled on `slurm/render_pushcube.sbatch`, with longer wall time for 20 clips):

```bash
#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --gres=gpu:1
#SBATCH --mem=115G
#SBATCH --time=00:45:00
#SBATCH --job-name=stage5-p1-iconic-pushcube
#SBATCH --output=slurm/logs/stage5-p1-iconic-pushcube-%j.out
#SBATCH --error=slurm/logs/stage5-p1-iconic-pushcube-%j.err
# Stage-5 P1 — iconic per-policy PushCube renders.
# 5 held-out seeds × 4 policies = 20 self-contained "full episode" MP4s.
# Spec:  docs/superpowers/specs/2026-05-24-stage5-p1-iconic-renders-design.md
# Plan:  docs/superpowers/plans/2026-05-24-stage5-p1-iconic-renders.md

set -euo pipefail

cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover

OUT_DIR=renders/stage5_p1_iconic/pushcube
mkdir -p "$OUT_DIR"

LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}" \
python scripts/render_stage5_p1_iconic.py \
    --seeds 100,110,120,130,143 \
    --policies latent,oracle_factor_revision,babysteps_selective,same_intent_retry \
    --pack-dir models/stage5/p1_vision/PushCube-v1 \
    --features-dir datasets/stage5/varied_intent/PushCube-v1/features \
    --out-dir "$OUT_DIR" \
    --fps 20

echo "--- output ---"
ls -lh "$OUT_DIR" | head -25
```

- [ ] **Step 2: Lint shell with `bash -n`**

Run: `bash -n slurm/render_stage5_p1_iconic.sbatch`

Expected: No output (syntactically valid).

- [ ] **Step 3: Commit**

```bash
git add slurm/render_stage5_p1_iconic.sbatch
git commit -m "feat(stage5 p1): sbatch for iconic per-policy PushCube renders

a100-40gb, 45min wall time, drives scripts/render_stage5_p1_iconic.py with
the 5 default seeds × 4 default policies. Output:
renders/stage5_p1_iconic/pushcube/pushcube_seed_NNNN__<policy>_full.mp4 ×20.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Submit the GPU job (manual — user runs this)

**This task is for the user, not the implementing agent.** The implementing agent stops after Task 4 and reports back.

Once tasks 1–4 are committed, the user submits:

```bash
sbatch slurm/render_stage5_p1_iconic.sbatch
```

Then `squeue -u $USER` to watch the job. Expected wall time ~15–20 min; the 45-min cap is a safety margin. Output goes to `slurm/logs/stage5-p1-iconic-pushcube-<jobid>.{out,err}` and 20 MP4s land in `renders/stage5_p1_iconic/pushcube/`.

On success, the user can spot-check one MP4 (e.g. `pushcube_seed_0100__latent_full.mp4`) and the same-seed across-policy comparison (`pushcube_seed_0143__latent_full.mp4` vs `pushcube_seed_0143__oracle_factor_revision_full.mp4`).

---

## Verification checklist (run after Task 4 before handoff)

- [ ] `python -m pytest tests/ -q` — all sim-free tests pass; total count grew by 3.
- [ ] `git log --oneline -5` — shows the 4 new commits (red tests, helper, CLI, sbatch).
- [ ] `python scripts/render_stage5_p1_iconic.py --help` — exits 0.
- [ ] `bash -n slurm/render_stage5_p1_iconic.sbatch` — exits 0.
- [ ] `git status` — clean working tree (only the new file paths committed).
