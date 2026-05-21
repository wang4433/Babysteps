# babysteps/ — the package

All importable, **sim-free** logic for the Stage-0 loop lives here. GPU/Vulkan
code is isolated in `envs/*_runner.py`; everything else in this package must
import and run without a simulator (the `tests/` suite depends on this).

## Modules

| Module | Job |
| --- | --- |
| `schemas.py` | Pure data contracts: `Intent` (the six object-centric factors + `direction_grounding`), failure packet, revision operator, episode record. **Single source of truth for the schema** — including the `*_GROUNDINGS` / token whitelists. |
| `demo.py` | Scripted demo-evidence utilities. Turns a proxy demonstration into structured intent factors. Captions describe *object motion*, never a Franka motor program. |
| `episode.py` | The Stage-0 episode loop: demo → intent → execute → fail → revise → retry. Orchestrates the other modules through the `TaskAdapter` interface. |
| `failure.py` | Structured failure detection + attribution. Maps an execution trace to a `failure_predicate` and an implicated wrong factor. |
| `revision.py` | Factor-local intent revision. **Enforces the single-factor invariant**: one factor changes, the rest are frozen. |
| `eval.py` | Dataset-level metrics (delta_pp, preservation rate, attribution accuracy, …) + Markdown/JSON report writer. |
| `viz.py` | Sim-free 2D top-down rendering of an episode (no Vulkan). |

## Subpackages

- `envs/` — the simulation boundary (adapters, runners, scene geometry, dispatch).
- `render/` — per-task three-phase MP4 flows (GPU).
- `skills/` — intent → executable skill compilers.

## Invariants

1. Keep this package sim-free except `envs/*_runner.py`. If a test needs a sim,
   it belongs behind the fake env / a runner, not here.
2. Schema changes are **additive** (add tokens, stop emitting old ones; remove
   later). Edit `schemas.py`, then update snapshots in `tests/snapshots/`.
3. A revision changes exactly one factor. New failure modes get a new
   attribution rule in `failure.py` + a new operator in `revision.py`, never a
   multi-factor edit.
