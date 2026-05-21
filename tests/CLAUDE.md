# tests/ — sim-free unit suite

302 tests, all **sim-free** — they run on the login node with no GPU/Vulkan:

```bash
python -m pytest tests/ -q     # ~1.3s
```

This is a hard constraint. The suite is the fast feedback loop for the whole
project; if a test needs a simulator it belongs behind the fake env, not here.

## Layout

| Area | Files |
| --- | --- |
| Core contracts | `test_schemas.py`, `test_failure.py`, `test_revision.py`, `test_episode.py`, `test_eval.py` |
| Demo evidence | `test_demo.py` |
| Per-task adapters | `test_pushcube_adapter.py`, `test_pickcube_adapter.py`, `test_stackcube_adapter.py`, `test_turnfaucet_adapter.py`, `test_crossview.py`, `test_task_adapter.py`, `test_task_registry.py` |
| Per-task skills | `test_push_skill.py`, `test_pick_skill.py`, `test_stack_skill.py`, `test_turn_skill.py` |
| Metrics gate | `test_pickcube_delta_pp.py` |
| Render modules | `test_render_modules.py` |
| CLI | `test_stage0_collect_cli.py` |
| `conftest.py` | shared fixtures (fake env, sample intents) |
| `snapshots/` | JSON snapshots — stable across all task families |

## Rules

- TDD: write the failing test before the implementation.
- The **single-factor-revision invariant** is asserted in tests
  (`test_revision.py` / `test_crossview.py`) — keep it green; it is the
  headline reviewer-facing property.
- Schema edits → regenerate / update `snapshots/` deliberately; a snapshot
  diff should be a reviewed, intentional change.
