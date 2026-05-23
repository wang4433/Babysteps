# PushCube render — real failure + stable push

**Date:** 2026-05-23
**Author:** Peter (with Claude)
**Status:** approved, ready for implementation plan
**Scope:** render-only (no runner / skill / adapter changes)

## Problem

Two issues in the current PushCube three-phase MP4 render
(`babysteps/render/pushcube.py`):

1. **Phase 2 (`2_attempt_blocked`) is a held-still placeholder.**
   `pushcube.py:155-156` fills the clip with `fps * 2` copies of the
   post-reset frame because the skill compiler short-circuits with
   `planner_failed=True` when the demo's `approach_direction` is in
   `scene.blocked_sides`. The viewer sees the robot do nothing.
   This violates `babysteps/render/CLAUDE.md`:

   > `2_attempt_blocked` must show a *real* failure (e.g. CrossView
   > pushes the wrong way; TurnFaucet's grasp physically fails), not
   > a held-still placeholder.

   The other four render modules (pickcube, stackcube, turnfaucet,
   crossview) already step the env; only PushCube is non-compliant.

2. **The cube "flies away" during the push.** `_execute_push` uses a
   1-step proportional controller with a hard saturation at the
   `pd_ee_delta_pose` limit (`pos_err / 0.1`, clipped to ±1). When the
   TCP descends to push height with a closed gripper and lunges
   horizontally at saturated speed, contact is impulsive — the cube
   gets a hammer-strike and skips rather than rolls. Visible in phases
   1 and 3 (the demo and retry both push), and would be visible in
   phase 2 once it executes for real.

## Goals

- Phase 2 shows the robot visibly attempting the demo's approach,
  physically failing, and the cube unmoved at clip end.
- Phases 1 and 3 push the cube smoothly (no flying / skipping).
- Change is render-only: `babysteps/render/pushcube.py` +
  `babysteps/render/common.py`. No edits to:
  - `babysteps/envs/pushcube_runner.py` — data collection / M1 reports
    stay reproducible.
  - `babysteps/skills/push.py` — single-factor revision math
    unchanged.
  - any adapter — attribution / revision pipeline unchanged.
- Sim-free pytest suite (343 tests) remains green.

## Non-goals

- Replacing the proportional controller globally (e.g. with MPlib).
  Out of scope for this render-only pass; revisit if the runner ever
  needs the same stabilization.
- Adding obstacle-aware semantics to `scene.blocked_sides`. The blocked
  side stays a symbolic flag in the data pipeline; only the render
  materializes it as a physical wall.
- Re-rendering the committed MP4s in `renders/pushcube/`. That is a
  follow-up run, not part of this design.

## Design

### Architecture

Two independent edits in the same module, sharing one new control knob:

1. **Damped per-phase proportional control** — kills the contact
   impulse so the cube no longer flies. Applies to all three phases.
2. **Physical obstacle on the blocked side, parked below the table
   when not in use** — replaces phase 2's held-still placeholder with
   a real, stalled approach.

`_execute_push` becomes a single loop that handles all three phases
identically; the obstacle's pose (parked vs placed) is the only thing
that distinguishes the blocked attempt.

### Component 1 — Damped per-phase proportional controller

**`babysteps/render/common.py`.** Refactor `prop_action` to take an
optional `pos_scale` arg:

```python
def prop_action(
    tcp_xyzw: np.ndarray,
    target_xyz: np.ndarray,
    gripper_cmd: float = -1.0,
    pos_scale: float = POS_SCALE,
) -> np.ndarray:
    pos_err = target_xyz - tcp_xyzw[0:3]
    action = np.zeros(7, dtype=np.float32)
    action[0:3] = np.clip(pos_err / pos_scale, -1.0, 1.0).astype(np.float32)
    action[6] = np.float32(gripper_cmd)
    return action
```

Default keeps current behaviour. Other callers (pickcube, stackcube,
turnfaucet, crossview render modules) are unaffected.

**`babysteps/render/pushcube.py`.** Per-phase scale tuple, indexed by
`phase_idx`:

```python
# (approach, pre_contact_high, descend, push)
_PUSHCUBE_POS_SCALE: tuple[float, ...] = (0.10, 0.10, 0.40, 0.50)
```

- Approach + pre-contact (high travel): 0.10 — same as today, fast
  travel.
- Descend (lowering to push_z): 0.40 — action saturates at ~25% of
  max velocity.
- Push (horizontal contact phase): 0.50 — action saturates at ~20% of
  max velocity. The dropped contact impulse is what stops the cube
  from flying.

Threaded through `_execute_push`:

```python
action = prop_action(
    tcp, target, gripper_cmd=-1.0, pos_scale=_PUSHCUBE_POS_SCALE[phase_idx],
)
```

Values are starting points; will be eyeball-tuned on one seed after
the first render.

### Component 2 — Physical obstacle on the blocked side

**Spawn (once, at the start of `render_episode`).** A static box actor
built via SAPIEN's actor builder. ManiSkill exposes
`env.unwrapped.scene` (a `ManiSkillScene` wrapping a SAPIEN scene),
which has `create_actor_builder()`.

```python
# Dimensions (half-extents)
_OBSTACLE_HALF_W = 0.020   # 0.04 m along the approach axis
_OBSTACLE_HALF_T = 0.0025  # 0.005 m perpendicular to the approach
_OBSTACLE_HALF_H = 0.050   # 0.10 m tall — clears EE travel_z

# Parked pose (below the table)
_OBSTACLE_PARK_Z = -0.50
```

Visual material: red (`[0.78, 0.20, 0.20, 1.0]`). Collision: enabled.
Built static (no rigid dynamics — the wall doesn't move when hit).

Built once per `render_episode` call. SAPIEN does not require
re-adding the actor after `env.reset`; pose is preserved across
resets, so we control visibility purely by moving it.

**Lifecycle.** The obstacle has two poses:

| Phase | Pose |
|---|---|
| 1 (demo) | parked at `(0, 0, -0.50)` |
| 2 (attempt) | `(cube_xy + approach_unit * (CUBE_HALF_SIZE + 0.025), z = cube_z + _OBSTACLE_HALF_H)` |
| 3 (retry) | parked at `(0, 0, -0.50)` |

`approach_unit` is `babysteps.envs.scene.approach_to_unit(initial_intent.approach_direction)`.
`CUBE_HALF_SIZE` is the 0.02 m constant already in `skills/push.py`.
The 0.025 m margin places the wall just outside the cube on the EE's
path.

Pose is set via `actor.set_pose(sapien.Pose(p=[x, y, z], q=[1, 0, 0, 0]))`.

### Component 3 — Phase 2 becomes a real execution

Phase 2 in `render_episode`:

```python
# === Phase 2 — ATTEMPT 1 (approach physically obstructed) ===
_move_obstacle_to_block(obstacle, scene.cube_xy, scene.cube_z, initial_intent)
wp_attempt = build_push_waypoints(scene_exec, initial_intent)
attempt1_frames: list = []
_ = _execute_push(
    env, wp_attempt, attempt1_frames, seed=seed,
    capture=render_wrist_frame,
    max_steps=120,
    no_progress_break_steps=20,
    no_progress_eps_m=0.002,
)
_park_obstacle(obstacle)
```

Note: `build_push_waypoints` does not check `blocked_sides` (that's
`compile_intent_to_push_skill`'s job — `push.py:101`), so the call
returns valid geometry. Phase 3 already uses the same pattern with
`revised_intent`.

New `_execute_push` kwargs:

- `max_steps` (default `PUSHCUBE_MAX_CONTROL_STEPS = 300`): per-call
  step budget. Phase 2 uses 120 so the stalled clip stays around 2–3
  seconds rather than 6.
- `no_progress_break_steps` (default `None`, i.e. disabled): if set,
  break when the TCP hasn't moved at least `no_progress_eps_m` for N
  consecutive steps. Phase 2 uses 20.
- `no_progress_eps_m` (default `0.002`): movement threshold.

Phase 1 and 3 keep the defaults — no behavioural change beyond the
damped `pos_scale`.

### Component 4 — Captions

Phase 2 subtitle changes:

```python
# was: f"approach_direction={...} is blocked → planner_failed"
a1_title = (
    f"{short_id}  phase 2/3: approach_blocked",
    f"approach_direction={initial_intent.approach_direction} "
    f"physically obstructed → arm stalls",
)
```

The pipeline-level synthetic `AttemptResult(planner_failed=True, ...)`
in `_pushcube_setup` is unchanged — that's what makes attribution and
revision still wire up correctly. Only the *visible* wording is
adjusted.

### Data flow

```
render_episode(env, adapter, seed, fps)
  ├─ obstacle = _build_obstacle(env)               # spawned, parked
  ├─ s = _pushcube_setup(env, adapter, seed)       # demo runs here (damped)
  │     └─ _execute_push(... pos_scale=_PUSHCUBE_POS_SCALE[phase_idx])
  ├─ Phase 2:
  │     ├─ _move_obstacle_to_block(obstacle, ..., initial_intent)
  │     ├─ wp_attempt = build_push_waypoints(scene_exec, initial_intent)
  │     ├─ _execute_push(env, wp_attempt, ..., max_steps=120,
  │     │                no_progress_break_steps=20)
  │     └─ _park_obstacle(obstacle)
  └─ Phase 3:
        ├─ wp_retry = build_push_waypoints(scene_exec, revised_intent)
        └─ _execute_push(env, wp_retry, ..., default kwargs)
```

The existing `_pushcube_setup` helper is unchanged in shape; its
internal `_execute_push` call inherits the damped `pos_scale` because
the per-phase tuple is module-level.

### Error handling and edge cases

- **Obstacle blocks phase 1 or 3.** Mitigated by parking at `z = -0.50`,
  well below the table plane. PushCube-v1's table is at `z ≈ 0`; EE
  travel z is ~0.20+; the parked actor cannot interfere.
- **Wall too short — EE flies over.** Eyeball-checked on seed 0. If EE
  visibly arcs over, increase `_OBSTACLE_HALF_H` to 0.075 m (0.15 m
  tall total).
- **Wall too thin — EE clips through.** PhysX handles ~5 mm walls
  reliably; if penetration is visible, bump `_OBSTACLE_HALF_T` to
  0.005 m (0.01 m total thickness).
- **Phase 2 env terminates on heavy contact.** `_execute_push` already
  breaks on `term/trunc`; no extra handling needed. The no-progress
  break is the primary exit condition.
- **`build_push_waypoints` raises on scene_exec.** It does not — the
  geometry is well-defined for any approach/contact pair. The only
  short-circuit (`approach_direction in blocked_sides`) lives in
  `compile_intent_to_push_skill`, which we deliberately bypass.

### Tunable constants (recap)

| Constant | Value | Lives in |
|---|---|---|
| `_PUSHCUBE_POS_SCALE` | `(0.10, 0.10, 0.40, 0.50)` | `render/pushcube.py` |
| `_OBSTACLE_HALF_W` | 0.020 m | `render/pushcube.py` |
| `_OBSTACLE_HALF_T` | 0.0025 m | `render/pushcube.py` |
| `_OBSTACLE_HALF_H` | 0.050 m | `render/pushcube.py` |
| `_OBSTACLE_PARK_Z` | −0.50 m | `render/pushcube.py` |
| `max_steps` (phase 2) | 120 | call-site in `render_episode` |
| `no_progress_break_steps` | 20 | call-site |
| `no_progress_eps_m` | 0.002 m | call-site |

## Testing

- **Sim-free pytest suite (343 tests).** Must remain green.
  `babysteps/render/pushcube.py` is GPU-only and not imported by
  sim-free tests (verified at implementation time: `grep -rl
  "from babysteps.render.pushcube" tests/`). The `prop_action`
  signature change in `render/common.py` is backward-compatible (new
  arg has a default).
- **Render smoke (GPU node).**
  ```
  python scripts/render_stage0_maniskill.py --task pushcube \
    --seeds 0 1 --fps 30 --out renders/pushcube/videos_maniskill/
  ```
  Inspect MP4s for:
  - phase 1: cube rolls to goal, no flying / skipping;
  - phase 2: arm visibly approaches blocked side, hits red wall,
    stalls; cube unmoved; clip length 2–4 s;
  - phase 3: cube rolls to goal cleanly.
- **No new automated test.** Render quality is a visual check; no
  existing render module has a per-task automated test.

## Open follow-ups (out of scope for this design)

- Re-render the committed MP4s in `renders/pushcube/videos_maniskill/`
  after the change lands.
- Decide whether to backport the damped `pos_scale` to
  `pushcube_runner.py`. Defer until next time we collect new data.
- Consider whether the obstacle's red colour and dimensions should
  match a paper-figure aesthetic.
