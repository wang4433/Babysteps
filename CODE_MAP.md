# CODE_MAP

One-screen map of the repository. Each code directory also has its own
`CLAUDE.md` with finer detail. Read `goal.md` and root `CLAUDE.md` first.

## Top-level layout

```text
babysteps/        the Python package — all importable, tested logic
  envs/           sim adapters + runners + scene geometry + task dispatch
  render/         per-task three-phase MP4 render flows
  skills/         intent → executable skill compilers (push/pick/stack/turn)
  stage4/         sim-free Stage-4 M1 schema-recoverability probe (analysis)
scripts/          CLI entry points (collect / summarize / render) + diag scratch
tests/            sim-free pytest suite (343 tests) + JSON snapshots
docs/             design specs, TDD plans, locked claim, archived handover
slurm/            sbatch scripts + logs + the canonical GPU run commands
datasets/         collected Stage-0 episode data (JSONL + videos)
renders/          committed example MP4s, one folder per task
reports/          generated analysis reports (e.g. stage4 schema recoverability)
goal.md           Stage-0 boundary + data contract (AUTHORITY)
CLAUDE.md         high-level project instructions
RUNBOOK.md        copy-paste operational commands
README.md         quickstart
technical_def.md  related-work positioning + factorized-intent math
milestones.md     ICLR milestone roadmap (M1–M7)
goal.md / update.md  Stage-0 goal / cross-view pivot rationale
```

## `babysteps/` — the package

Pure, sim-free modules orchestrating the loop:

| Module | Job |
| --- | --- |
| `schemas.py` | data contracts: `Intent`, failure packet, revision, episode |
| `demo.py` | scripted demo-evidence utilities (proxy demo → factors) |
| `episode.py` | the Stage-0 loop: demo → intent → execute → fail → revise → retry |
| `failure.py` | structured failure detection + factor attribution |
| `revision.py` | factor-local intent revision (single-factor invariant) |
| `eval.py` | dataset-level metrics + Markdown/JSON report writer |
| `viz.py` | sim-free 2D top-down rendering for episodes |

### `babysteps/envs/` — simulation boundary

- `task_adapter.py` — the `TaskAdapter` interface (boundary for episode/CLI).
- `task_registry.py` — `--task` dispatch table.
- `scene.py` — pure, sim-agnostic scene geometry helpers.
- `<task>_adapter.py` — per-task adapters (pushcube, pickcube, stackcube,
  turnfaucet, crossview). Pure logic: intent build, attribution, frozen factors.
- `<task>_runner.py` — real ManiSkill env wrappers (GPU). `crossview_runner`
  is a thin wrapper over `pushcube_runner` (runs on the real PushCube gym env).

### `babysteps/render/` — MP4 flows

`common.py` + one module per task. Each `render_episode` produces the three
Stage-0 phases: `1_demo`, `2_attempt_blocked`, `3_retry`.

### `babysteps/skills/` — skill compilers

`push.py`, `pick.py`, `stack.py`, `turn.py`. Each turns an `Intent` into an
executable skill (waypoints / motion-plan parameters). `turn.py` dispatches the
embodiment_substitution (poke-turn) path for Sub-project D.

### `babysteps/stage4/` — learned-latent track (analysis only)

Sim-free Stage-4 Milestone-1 code: `features.py` (firewall-strict 19-dim
demo-evidence features), `probe.py` (linear probe + chance/shuffled baselines),
`report.py` (per-task per-factor recoverability table). Reads only
DemoEvidence-shaped fields, never `execution.initial_intent` (the label).

## `scripts/`

- `stage0_collect.py` — collect episodes (real or `--fake-env`) → `samples.jsonl`.
- `stage0_summarize.py` — `samples.jsonl` → `report.{json,md}` (derives task).
- `stage4_probe_schema_recoverability.py` — Stage-4 M1: probe schema
  recoverability from demo evidence → `reports/stage4/` (json + md).
- `render_stage0_maniskill.py` — render three-phase MP4s per task (GPU).
- `render_stage0_topdown.py` — sim-free 2D top-down render.
- `smoke_pushcube.py` — PushCube loadability check.
- `_diag_*.py` — **scratch** diagnostics (mostly TurnFaucet poke-turn探索).
  Not part of the tested codebase; `_diag_tf_poke5.py` is the empirical
  reference cited by the TurnFaucet runner.

## `tests/`

343 sim-free unit tests. Per-module (`test_schemas`, `test_failure`,
`test_revision`, `test_episode`, `test_eval`), per-task adapter/skill, CLI
(`test_stage0_collect_cli`), render modules, the Stage-4 probe tests
(`test_stage4_smoke/features/probe/report`), plus a single-factor-revision
invariant test and JSON `snapshots/`. Runs on the login node, no GPU.

## `docs/`

- `superpowers/specs/` — design spec per sub-project (dated).
- `superpowers/plans/` — TDD implementation plan per sub-project (dated).
- `milestone1_locked_claim.md` — the locked paper claim.
- `cold_start_handover_archive.md` — historical long-form handover (drifted schema).

## `slurm/`, `datasets/`, `renders/`

Operational + artifact directories. `slurm/CLAUDE.md` holds the canonical GPU
run commands and recorded gate results. See `RUNBOOK.md` for the full command
catalog.
