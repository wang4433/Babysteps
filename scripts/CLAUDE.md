# scripts/ — CLI entry points

Command-line drivers for the Stage-0 pipeline. For copy-paste invocations see
`RUNBOOK.md`.

## Production scripts

| Script | Job |
| --- | --- |
| `stage0_collect.py` | Collect episodes → `samples.jsonl`. `--task {PushCube,PickCube,StackCube,TurnFaucet,CrossViewPush}-v1`, `--fake-env` for sim-free runs. |
| `stage0_summarize.py` | `samples.jsonl` → `report.{json,md}` with the acceptance gate. Derives the task from the JSONL — **no `--task` flag**. |
| `render_stage0_maniskill.py` | Render the three-phase MP4 set per task (GPU/Vulkan). |
| `render_stage0_topdown.py` | Sim-free 2D top-down render (no GPU). |
| `smoke_pushcube.py` | PushCube loadability smoke check. |

## `_diag_*.py` — scratch diagnostics

**Not part of the tested codebase.** Mostly TurnFaucet poke-turn exploration
(`_diag_tf_*`) plus a few stackcube/compare probes. They are ad-hoc, may rot,
and are not imported by the package or tests.

- `_diag_tf_poke5.py` is the **empirical reference** cited by
  `babysteps/envs/turnfaucet_runner.py` for the auto-sign poke-turn — keep it.
- The rest are safe to prune in a cleanup pass; confirm before deleting since
  they hold hard-won faucet tuning.

## Rules

- New production CLI → mirror the `stage0_*` flag conventions and add a CLI test
  in `tests/` (e.g. `test_stage0_collect_cli.py`).
- Keep GPU-only scripts importable without crashing on the login node where
  feasible (defer Vulkan/sim setup to call time).
