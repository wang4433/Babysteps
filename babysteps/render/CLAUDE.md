# babysteps/render/ — three-phase MP4 flows

Per-task render modules that produce the Stage-0 demonstration MP4 set. Driven
by `scripts/render_stage0_maniskill.py` (GPU). One module per task, plus shared
helpers.

## Files

| File | Job |
| --- | --- |
| `common.py` | Shared utilities (frame capture, caption overlay, MP4 writing, naming). |
| `pushcube.py` | PushCube `render_episode`. |
| `pickcube.py` | PickCube `render_episode`. |
| `stackcube.py` | StackCube `render_episode`. |
| `turnfaucet.py` | TurnFaucet `render_episode` (privileged demo + poke-turn retry). |
| `crossview.py` | CrossViewPush `render_episode` (world camera; observer yaw in grounding math). |

## The three phases

Every `render_episode` emits exactly three clips per seed:

```text
<task_prefix>_seed_NNNN__1_demo.mp4              proxy demonstration
<task_prefix>_seed_NNNN__2_attempt_blocked.mp4   initial (failing) attempt
<task_prefix>_seed_NNNN__3_retry.mp4             revised retry
```

## Rules

- **Demo captions describe object evidence**, never an executable Franka motor
  program — the demo is a third-person proxy, not privileged action data.
- `2_attempt_blocked` must show a *real* failure (e.g. CrossView pushes the
  wrong way; TurnFaucet's grasp physically fails), not a held-still placeholder.
- Keep per-task quirks (camera choice, backend selection) inside the task
  module; shared mechanics go in `common.py`.
