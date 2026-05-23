# PushCube render — real failure + stable push — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace PushCube's held-still phase-2 placeholder with a real attempt against a physical obstacle, and damp per-phase POS_SCALE so the cube no longer flies under impulsive contact. Render-only scope.

**Architecture:** All edits in `babysteps/render/common.py` (one signature change) and `babysteps/render/pushcube.py` (per-phase POS_SCALE tuple, obstacle build/move/park helpers, `_execute_push` kwargs, `render_episode` + `render_baseline_contrast` phase-2 rewire). Obstacle helpers no-op when the env lacks `scene.create_actor_builder`, keeping sim-free tests green.

**Tech Stack:** Python, NumPy, ManiSkill 3 (`mani_skill`), SAPIEN actor builder, pytest.

**Spec:** `docs/superpowers/specs/2026-05-23-pushcube-render-real-failure-stable-push-design.md`

---

## File Structure

| File | What changes |
|---|---|
| `babysteps/render/common.py` | `prop_action(...)` gains optional `pos_scale: float = POS_SCALE` kwarg. Other callers unaffected. |
| `babysteps/render/pushcube.py` | Per-phase POS_SCALE tuple; `_execute_push` adds `max_steps` / `no_progress_break_steps` / `no_progress_eps_m` kwargs and threads `pos_scale` through `prop_action`; new obstacle helpers (`_get_or_build_obstacle`, `_move_obstacle_to_block`, `_park_obstacle`); `render_episode` and `render_baseline_contrast` phase 2 rewired. |
| `tests/test_render_modules.py` | Stub env wrist camera varies with step count; held-still assertion replaced with "phase 2 stepped the env"; new `prop_action` unit tests. |
| `slurm/render_pushcube.sbatch` | Untouched. Same launch command in Task 9 for GPU smoke. |

No new files. No new packages.

---

## Task 1: Failing tests for the new phase-2 behaviour

**Files:**
- Modify: `tests/test_render_modules.py:43-84` (stub env), `tests/test_render_modules.py:90-108` (held-still test)

The current stub env's `get_sensor_images()` returns a constant 4x4 frame regardless of step count, so we cannot detect "env was stepped" purely from wrist frames. We add step-count variation so the test can prove phase 2 actually drove `env.step`. Then we replace the held-still equality assertion with the new behaviour.

- [ ] **Step 1: Make the stub wrist camera vary with step count**

Edit `tests/test_render_modules.py` — find the `_StubEnv.get_sensor_images` method (around line 76-81) and replace:

```python
    def get_sensor_images(self):
        # First-person wrist camera: a distinctly-shaped 4x4x3 frame, batched
        # (B,H,W,3) like ManiSkill, so tests can tell the wrist view apart from
        # the 8x8 third-person render() view purely by frame shape.
        rgb = np.ones((1, 4, 4, 3), dtype=np.uint8) * 50
        return {"hand_camera": {"rgb": rgb}}
```

with:

```python
    def get_sensor_images(self):
        # First-person wrist camera: a distinctly-shaped 4x4x3 frame, batched
        # (B,H,W,3) like ManiSkill. The fill value tracks _step_count so
        # tests can detect that env.step was actually called (a held-still
        # placeholder produces identical frames; a real execution does not).
        val = np.uint8(self._step_count % 256)
        rgb = np.ones((1, 4, 4, 3), dtype=np.uint8) * val
        return {"hand_camera": {"rgb": rgb}}
```

- [ ] **Step 2: Rewrite the held-still assertion**

Find `test_pushcube_render_episode_emits_three_phase_frames` (around line 90-108) and replace the entire test with:

```python
def test_pushcube_render_episode_emits_three_phase_frames():
    """render_episode returns frames dict with demo/attempt_blocked/retry,
    and phase 2 (attempt_blocked) actually steps the env (no longer a
    held-still placeholder)."""
    from babysteps.render.pushcube import render_episode
    from babysteps.envs.pushcube_adapter import PushCubeAdapter

    env = _StubEnv()
    adapter = PushCubeAdapter()
    frames, titles = render_episode(env, adapter, seed=0, fps=4)

    assert set(frames.keys()) == {"demo", "attempt_blocked", "retry"}
    assert set(titles.keys()) == {"demo", "attempt_blocked", "retry"}
    # All three phases must produce frames.
    assert len(frames["demo"]) >= 1
    assert len(frames["retry"]) >= 1
    # Phase 2 must step the env (no longer a held-still placeholder):
    # multiple frames AND at least one pair differs (proves _StubEnv.step
    # was called, since the stub's wrist frame fill value tracks
    # _step_count).
    held = frames["attempt_blocked"]
    assert len(held) > 1
    assert not all(np.array_equal(held[0], f) for f in held)
```

- [ ] **Step 3: Run pytest to confirm the new test fails**

Run: `cd /scratch/gilbreth/wang4433/babysteps && python -m pytest tests/test_render_modules.py::test_pushcube_render_episode_emits_three_phase_frames -v`

Expected: **FAIL** with `AssertionError` on `assert not all(np.array_equal(...))` — current phase 2 holds the same frame.

Do **not** commit yet — leave the test red until the implementation lands.

---

## Task 2: `prop_action` gains a `pos_scale` parameter

**Files:**
- Modify: `babysteps/render/common.py:47-56`
- Test: `tests/test_render_modules.py` (append new test functions)

- [ ] **Step 1: Write failing unit tests for `prop_action(pos_scale=...)`**

Append to `tests/test_render_modules.py` (anywhere after the imports, e.g. just before the `# ---------- Stub env ----` block):

```python
# ---------- prop_action(pos_scale=...) unit tests ---------- #


def test_prop_action_default_pos_scale_matches_legacy():
    """Default pos_scale must reproduce the legacy 0.1-scaled action so
    existing render callers are unaffected."""
    from babysteps.render.common import POS_SCALE, prop_action
    tcp = np.array([0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    target = np.array([0.02, 0.0, 0.25], dtype=np.float64)
    action = prop_action(tcp, target, gripper_cmd=-1.0)
    # pos_err = (0.02, 0, 0); divided by POS_SCALE=0.1 → (0.2, 0, 0); clipped → (0.2, 0, 0).
    expected_x = float(np.clip(0.02 / POS_SCALE, -1.0, 1.0))
    assert action[0] == pytest.approx(expected_x, abs=1e-6)
    assert action[1] == pytest.approx(0.0, abs=1e-6)
    assert action[2] == pytest.approx(0.0, abs=1e-6)
    assert action[6] == pytest.approx(-1.0, abs=1e-6)


def test_prop_action_larger_pos_scale_yields_smaller_action():
    """A larger pos_scale damps the action (smaller magnitude for the
    same pos_err) — this is the mechanism that drops the contact
    impulse during the push phase."""
    from babysteps.render.common import prop_action
    tcp = np.array([0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    target = np.array([0.02, 0.0, 0.25], dtype=np.float64)
    action_default = prop_action(tcp, target, gripper_cmd=-1.0)
    action_damped = prop_action(tcp, target, gripper_cmd=-1.0, pos_scale=0.5)
    # Same direction, smaller magnitude.
    assert abs(action_damped[0]) < abs(action_default[0])
    assert action_damped[0] == pytest.approx(0.02 / 0.5, abs=1e-6)


def test_prop_action_saturation_still_clips():
    """pos_scale affects normalization, not the ±1 saturation cap."""
    from babysteps.render.common import prop_action
    tcp = np.array([0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    target = np.array([1.0, 0.0, 0.25], dtype=np.float64)  # very far
    action = prop_action(tcp, target, gripper_cmd=-1.0, pos_scale=0.5)
    assert action[0] == pytest.approx(1.0, abs=1e-6)  # clipped


def test_prop_action_pos_scale_is_keyword_only_or_positional():
    """Backward-compatible signature: pos_scale is a kwarg with a default,
    so existing callers in pickcube/stackcube/turnfaucet/crossview render
    modules continue to work without changes."""
    from babysteps.render.common import prop_action
    tcp = np.array([0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    target = np.array([0.02, 0.0, 0.25], dtype=np.float64)
    # Legacy call style — must not raise.
    _ = prop_action(tcp, target)
    _ = prop_action(tcp, target, gripper_cmd=0.5)
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run:
```bash
cd /scratch/gilbreth/wang4433/babysteps && \
python -m pytest tests/test_render_modules.py::test_prop_action_default_pos_scale_matches_legacy \
                  tests/test_render_modules.py::test_prop_action_larger_pos_scale_yields_smaller_action \
                  tests/test_render_modules.py::test_prop_action_saturation_still_clips \
                  tests/test_render_modules.py::test_prop_action_pos_scale_is_keyword_only_or_positional -v
```

Expected: at minimum `test_prop_action_larger_pos_scale_yields_smaller_action` and `test_prop_action_saturation_still_clips` **FAIL** with `TypeError: prop_action() got an unexpected keyword argument 'pos_scale'`. (`test_prop_action_default_pos_scale_matches_legacy` may pass by accident — that is fine.)

- [ ] **Step 3: Implement the `pos_scale` parameter**

Edit `babysteps/render/common.py`, replacing the current `prop_action` function (lines 47-56):

```python
def prop_action(
    tcp_xyzw: np.ndarray,
    target_xyz: np.ndarray,
    gripper_cmd: float = -1.0,
    pos_scale: float = POS_SCALE,
) -> np.ndarray:
    """Proportional 7-dim action toward target_xyz with explicit gripper cmd.

    Default gripper_cmd=-1 (closed) matches PushSkill's behavior. The
    optional `pos_scale` lets per-phase callers damp the action
    magnitude — a larger pos_scale → smaller saturated velocity → lower
    contact impulse (used in PushCube's descend + push phases to stop
    the cube from flying)."""
    pos_err = target_xyz - tcp_xyzw[0:3]
    action = np.zeros(7, dtype=np.float32)
    action[0:3] = np.clip(pos_err / pos_scale, -1.0, 1.0).astype(np.float32)
    action[6] = np.float32(gripper_cmd)
    return action
```

- [ ] **Step 4: Run the prop_action tests to verify they pass**

Run:
```bash
cd /scratch/gilbreth/wang4433/babysteps && \
python -m pytest tests/test_render_modules.py -k "prop_action" -v
```

Expected: all 4 `prop_action` tests **PASS**.

- [ ] **Step 5: Run full render-module tests to verify no regression**

Run:
```bash
cd /scratch/gilbreth/wang4433/babysteps && \
python -m pytest tests/test_render_modules.py -v
```

Expected: all tests pass **except** the phase-2 test from Task 1, which still fails (we'll fix it in Tasks 3-7).

- [ ] **Step 6: Commit**

```bash
cd /scratch/gilbreth/wang4433/babysteps && \
git add babysteps/render/common.py tests/test_render_modules.py && \
git commit -m "feat(render): prop_action accepts optional pos_scale (damping knob)

Adds a keyword arg pos_scale=POS_SCALE so per-phase callers can damp
the saturated action magnitude. Backward compatible — default behavior
unchanged; existing callers in pickcube/stackcube/turnfaucet/crossview
render modules are unaffected.

Foundation for the PushCube render fix (spec
docs/superpowers/specs/2026-05-23-pushcube-render-real-failure-stable-push-design.md).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Per-phase POS_SCALE tuple + threading in `_execute_push`

**Files:**
- Modify: `babysteps/render/pushcube.py:34-69` (`_execute_push` function and add module constant)

- [ ] **Step 1: Add the per-phase POS_SCALE tuple**

Edit `babysteps/render/pushcube.py`. Just below the `from babysteps.skills.push import build_push_waypoints` line (after the existing imports, before the `def _execute_push` line), insert:

```python


# Per-phase pos_scale for the 4-waypoint push (approach, pre_contact_high,
# descend, push). Larger values → smaller saturated velocity → lower contact
# impulse. Approach + pre-contact stay at the legacy 0.10 (fast travel);
# descend and push are damped so contact with the cube is gentle and the
# cube does not fly. Tunable; eyeball-checked on seed 0.
_PUSHCUBE_POS_SCALE: tuple[float, ...] = (0.10, 0.10, 0.40, 0.50)
```

- [ ] **Step 2: Thread per-phase pos_scale into `_execute_push`**

In the same file, find the `action = prop_action(...)` line inside `_execute_push` (currently line 55) and replace it:

```python
        action = prop_action(tcp, target, gripper_cmd=-1.0)
```

with:

```python
        action = prop_action(
            tcp, target, gripper_cmd=-1.0,
            pos_scale=_PUSHCUBE_POS_SCALE[phase_idx],
        )
```

`phase_idx` is in `{0, 1, 2, 3}` at this point — the increment-then-break pattern just above guarantees we never index past the tuple. The tuple length matches `len(targets) == 4` (the 4 push waypoints).

- [ ] **Step 3: Run render tests**

Run:
```bash
cd /scratch/gilbreth/wang4433/babysteps && \
python -m pytest tests/test_render_modules.py -v
```

Expected: same status as before — `prop_action` tests pass, baseline-contrast tests pass, the phase-2 test from Task 1 still fails. No new regressions.

- [ ] **Step 4: Commit**

```bash
cd /scratch/gilbreth/wang4433/babysteps && \
git add babysteps/render/pushcube.py && \
git commit -m "feat(render): per-phase damped POS_SCALE for PushCube push

Adds _PUSHCUBE_POS_SCALE = (0.10, 0.10, 0.40, 0.50) and threads it into
_execute_push so the descend and push phases saturate at ~25%/20% of
max velocity. Drops the contact impulse so the cube no longer flies
when the EE makes contact.

Phases 1 and 3 (demo + retry) become smoother; phase 2 will benefit
once the held-still placeholder is removed (Tasks 5-7).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: `_execute_push` gains `max_steps` and no-progress break

**Files:**
- Modify: `babysteps/render/pushcube.py:34-69`

- [ ] **Step 1: Extend `_execute_push` signature and loop**

Replace the entire `_execute_push` function in `babysteps/render/pushcube.py` (currently lines 34-69) with:

```python
def _execute_push(
    env, waypoints, frames: list, *,
    seed: int,
    capture=render_frame,
    max_steps: int = PUSHCUBE_MAX_CONTROL_STEPS,
    no_progress_break_steps: int | None = None,
    no_progress_eps_m: float = 0.002,
) -> dict:
    """Step through waypoints capturing one frame per step. Re-resets the env
    at the start so demo / attempt / retry all begin from the same scene.

    `capture` selects the view: render_frame (third-person external camera,
    the demo view) or render_wrist_frame (first-person panda_wristcam, the
    execution view).

    `max_steps` caps the control-step budget (default PUSHCUBE_MAX_CONTROL_STEPS,
    matching the runner). Phase 2's blocked attempt uses a shorter budget so
    the stalled clip stays a few seconds rather than the full ~10s cap.

    `no_progress_break_steps` (default None = disabled) exits the loop when
    the TCP has moved less than `no_progress_eps_m` for that many
    consecutive steps — used in phase 2 to detect 'arm stalled against the
    obstacle' and end the clip early."""
    obs, _ = env.reset(seed=int(seed))
    targets = [np.asarray(wp[0:3], dtype=np.float64) for wp in waypoints]
    phase_idx = 0
    success = False

    frames.append(capture(env))
    prev_tcp_xyz: np.ndarray | None = None
    stalled_steps = 0
    for _ in range(max_steps):
        tcp, cube_xy, _, _ = read_obs(obs)
        target = targets[phase_idx]
        if np.linalg.norm(target - tcp[0:3]) < PHASE_TOL_M:
            phase_idx += 1
            if phase_idx >= len(targets):
                break
            target = targets[phase_idx]
        action = prop_action(
            tcp, target, gripper_cmd=-1.0,
            pos_scale=_PUSHCUBE_POS_SCALE[phase_idx],
        )
        obs, _r, term, trunc, info = env.step(action)
        frames.append(capture(env))
        term_b = bool(to_np(term).item()) if hasattr(term, "cpu") else bool(term)
        trunc_b = bool(to_np(trunc).item()) if hasattr(trunc, "cpu") else bool(trunc)
        succ = info.get("success", False) if hasattr(info, "get") else False
        success = bool(to_np(succ).item()) if hasattr(succ, "cpu") else bool(succ)
        if success or term_b or trunc_b:
            break
        # No-progress detection (phase 2: arm stalls against obstacle).
        if no_progress_break_steps is not None:
            tcp_now = np.asarray(tcp[0:3], dtype=np.float64)
            if prev_tcp_xyz is None:
                prev_tcp_xyz = tcp_now
                stalled_steps = 0
            else:
                if float(np.linalg.norm(tcp_now - prev_tcp_xyz)) < no_progress_eps_m:
                    stalled_steps += 1
                else:
                    stalled_steps = 0
                    prev_tcp_xyz = tcp_now
                if stalled_steps >= no_progress_break_steps:
                    break

    tcp, final_cube_xy, _, _ = read_obs(obs)
    return {
        "final_obj_xy": (float(final_cube_xy[0]), float(final_cube_xy[1])),
        "success": bool(success),
    }
```

- [ ] **Step 2: Run render tests**

Run:
```bash
cd /scratch/gilbreth/wang4433/babysteps && \
python -m pytest tests/test_render_modules.py -v
```

Expected: same as before — `prop_action` tests + baseline-contrast tests + most existing tests pass; the phase-2 test from Task 1 still fails (phase 2 is still using the held-still placeholder).

- [ ] **Step 3: Commit**

```bash
cd /scratch/gilbreth/wang4433/babysteps && \
git add babysteps/render/pushcube.py && \
git commit -m "feat(render): _execute_push gains max_steps + no-progress break

Adds optional max_steps, no_progress_break_steps, and no_progress_eps_m
kwargs. Default behaviour for existing callers is unchanged
(max_steps=PUSHCUBE_MAX_CONTROL_STEPS, no_progress disabled).

These are the knobs phase 2 will use to keep the stalled-against-wall
clip short and end it as soon as the TCP stops moving.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Obstacle build / move / park helpers

**Files:**
- Modify: `babysteps/render/pushcube.py` (add helpers and constants near the top)

The helpers must no-op when the env doesn't support actor building, so that sim-free tests using `_StubEnv` (which has no `scene` attribute) continue to work.

- [ ] **Step 1: Add obstacle constants and helpers**

Edit `babysteps/render/pushcube.py`. Add the following block immediately after the `_PUSHCUBE_POS_SCALE` definition added in Task 3 (and before `def _execute_push`):

```python


# Obstacle (phase-2 blocked-side wall) — half-extents in meters.
_OBSTACLE_HALF_W: float = 0.020   # along the approach axis (0.04 m total)
_OBSTACLE_HALF_T: float = 0.0025  # perpendicular to approach (0.005 m total)
_OBSTACLE_HALF_H: float = 0.050   # vertical (0.10 m total) — clears EE travel z
_OBSTACLE_PARK_Z: float = -0.50   # below table plane; invisible / out of the way
_OBSTACLE_BLOCK_MARGIN_M: float = 0.025  # gap between cube edge and wall face


def _get_or_build_obstacle(env):
    """Spawn (once per env) a static red box obstacle, parked below the
    table. Returns None when the env does not support actor building
    (sim-free stub envs).

    Cached on `env._babysteps_obstacle` so repeated render_episode calls
    on the same env reuse the same actor rather than accumulating walls.
    """
    cached = getattr(env, "_babysteps_obstacle", None)
    if cached is not None:
        return cached
    scene = getattr(env.unwrapped, "scene", None)
    if scene is None or not hasattr(scene, "create_actor_builder"):
        return None  # sim-free stub env: helpers no-op below
    import sapien

    builder = scene.create_actor_builder()
    half = [_OBSTACLE_HALF_W, _OBSTACLE_HALF_T, _OBSTACLE_HALF_H]
    builder.add_box_collision(half_size=half)
    builder.add_box_visual(
        half_size=half,
        material=sapien.render.RenderMaterial(base_color=[0.78, 0.20, 0.20, 1.0]),
    )
    builder.initial_pose = sapien.Pose(p=[0.0, 0.0, _OBSTACLE_PARK_Z])
    actor = builder.build_static(name="approach_obstacle")
    try:
        env._babysteps_obstacle = actor
    except AttributeError:
        # Some env wrappers reject attribute assignment; fall back to
        # caching on env.unwrapped (best-effort).
        env.unwrapped._babysteps_obstacle = actor
    return actor


def _move_obstacle_to_block(obstacle, cube_xy, cube_z, intent) -> None:
    """Place the obstacle on the blocked side of the cube, on the EE's
    approach path. No-op when obstacle is None."""
    if obstacle is None:
        return
    import sapien
    from babysteps.envs.scene import approach_to_unit
    from babysteps.skills.push import CUBE_HALF_SIZE
    unit = approach_to_unit(intent.approach_direction)
    margin = CUBE_HALF_SIZE + _OBSTACLE_BLOCK_MARGIN_M
    x = float(cube_xy[0]) + float(unit[0]) * margin
    y = float(cube_xy[1]) + float(unit[1]) * margin
    z = float(cube_z) + _OBSTACLE_HALF_H  # rest on table, half-height up
    obstacle.set_pose(sapien.Pose(
        p=[x, y, z],
        q=[1.0, 0.0, 0.0, 0.0],
    ))


def _park_obstacle(obstacle) -> None:
    """Move the obstacle far below the table — invisible, no contact.
    No-op when obstacle is None."""
    if obstacle is None:
        return
    import sapien
    obstacle.set_pose(sapien.Pose(
        p=[0.0, 0.0, _OBSTACLE_PARK_Z],
        q=[1.0, 0.0, 0.0, 0.0],
    ))
```

- [ ] **Step 2: Verify sim-free imports still work**

Run:
```bash
cd /scratch/gilbreth/wang4433/babysteps && \
python -c "from babysteps.render.pushcube import _get_or_build_obstacle, _move_obstacle_to_block, _park_obstacle; print('ok')"
```

Expected: prints `ok`. No `sapien` import should fire at module load (the `import sapien` is inside each function body), so the login-node import succeeds without GPU.

- [ ] **Step 3: Run render tests**

Run:
```bash
cd /scratch/gilbreth/wang4433/babysteps && \
python -m pytest tests/test_render_modules.py -v
```

Expected: same as Task 4 — the phase-2 test from Task 1 still fails; everything else passes. The new helpers are defined but not yet called.

- [ ] **Step 4: Commit**

```bash
cd /scratch/gilbreth/wang4433/babysteps && \
git add babysteps/render/pushcube.py && \
git commit -m "feat(render): obstacle build/move/park helpers (defensive no-op)

Adds _get_or_build_obstacle, _move_obstacle_to_block, _park_obstacle and
the obstacle dimension constants. Helpers no-op when env.unwrapped has no
scene.create_actor_builder (the sim-free stub env), so tests/test_render_modules.py
continues to work. SAPIEN is imported lazily inside each function so the
login-node import path stays clean.

Obstacle: 0.04 x 0.005 x 0.10 m red box; parked at z=-0.50 by default;
moved to the blocked side via approach_to_unit when phase 2 starts.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Rewire `render_episode` phase 2 + caption

**Files:**
- Modify: `babysteps/render/pushcube.py:136-192` (`render_episode` function)

- [ ] **Step 1: Rewrite `render_episode` to spawn obstacle, execute phase 2 for real, update caption**

Find the `def render_episode(env, adapter: BaseTaskAdapter, seed: int, fps: int)` function (around line 136-192) and replace its **entire body** with:

```python
def render_episode(
    env, adapter: BaseTaskAdapter, seed: int, fps: int,
) -> tuple[dict, dict]:
    """Run the three-phase BABYSTEPS demo for PushCube and return per-phase
    frame lists and title metadata.

    Phase 1 (demo): execute the oracle's correct intent.
    Phase 2 (attempt_blocked): place a static red wall on the demo's
        approach side; execute the demo-derived push waypoints; the arm
        physically stalls against the wall and the cube is unmoved.
    Phase 3 (retry): execute the revised (orthogonal-approach) intent.

    Returns:
        frames: {"demo": [...], "attempt_blocked": [...], "retry": [...]}
        titles: {"demo": (title, subtitle), ...}
    """
    short_id = f"seed {seed:04d}"

    # Spawn the obstacle once per env (cached). Parked below the table by
    # default so phases 1 and 3 are unaffected.
    obstacle = _get_or_build_obstacle(env)
    _park_obstacle(obstacle)

    s = _pushcube_setup(env, adapter, seed)
    correct_intent = s["correct_intent"]
    initial_intent = s["initial_intent"]
    scene_exec = s["scene_exec"]
    demo_frames = s["demo_frames"]

    # === Phase 2 — ATTEMPT (approach physically obstructed) ===
    # Move the wall onto the demo's approach side, then drive the
    # demo-derived waypoints. The arm reaches the approach standoff,
    # hits the wall, and the no-progress break ends the clip.
    _move_obstacle_to_block(
        obstacle, s["scene"].cube_xy, s["scene"].cube_z, initial_intent,
    )
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

    # === Phase 3 — RETRY with revised approach (selective) ===
    revised_intent, revision = adapter.revise_intent(
        initial_intent, s["attribution"], scene_exec,
    )
    wp_retry = build_push_waypoints(scene_exec, revised_intent)
    retry_frames: list = []
    out_retry = _execute_push(
        env, wp_retry, retry_frames, seed=seed, capture=render_wrist_frame,
    )

    demo_title = (
        f"{short_id}  phase 1/3: demo proxy",
        f"contact_region={correct_intent.contact_region}, "
        f"approach={correct_intent.approach_direction}",
    )
    a1_title = (
        f"{short_id}  phase 2/3: approach_blocked",
        f"approach_direction={initial_intent.approach_direction} "
        f"physically obstructed → arm stalls",
    )
    retry_title = (
        f"{short_id}  phase 3/3: retry (success={out_retry['success']})",
        f"approach_substitution: "
        f"{initial_intent.approach_direction} → "
        f"{revised_intent.approach_direction}  |  "
        f"frozen (preserved): {', '.join(revision.frozen_factors)}",
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

- [ ] **Step 2: Run the previously-failing phase-2 test, expect it to pass**

Run:
```bash
cd /scratch/gilbreth/wang4433/babysteps && \
python -m pytest tests/test_render_modules.py::test_pushcube_render_episode_emits_three_phase_frames -v
```

Expected: **PASS**. Phase 2 now drives `_execute_push`, which calls `env.step` repeatedly, each step incrementing `_step_count`, so wrist frames vary across the clip.

- [ ] **Step 3: Run all render-module tests**

Run:
```bash
cd /scratch/gilbreth/wang4433/babysteps && \
python -m pytest tests/test_render_modules.py -v
```

Expected: all tests pass, including `test_pushcube_render_titles_contain_phase_label`, `test_pushcube_render_retry_subtitle_shows_frozen_factors`, and `test_pushcube_render_demo_thirdperson_exec_wristcam`.

The caption test `test_pushcube_render_titles_contain_phase_label` only checks for `"phase 2/3"` — still present. Wrist-camera test asserts shape `(4, 4, 3)` for `attempt_blocked` frames — still true.

- [ ] **Step 4: Commit**

```bash
cd /scratch/gilbreth/wang4433/babysteps && \
git add babysteps/render/pushcube.py && \
git commit -m "fix(render): PushCube phase 2 executes against a real obstacle

Replaces the held-still placeholder (fps*2 copies of one frame) with a
real attempt: a static red wall is moved onto the demo's approach side,
the demo-derived push waypoints are driven, and the arm visibly stalls.
The clip ends when TCP has been still for 20 steps (no_progress_break)
or at step 120 — whichever comes first.

Phase 2 caption updated: 'planner_failed' (a runner-side flag) replaced
with 'physically obstructed -> arm stalls' (what the viewer actually
sees). The pipeline-level synthetic AttemptResult(planner_failed=True)
in _pushcube_setup is unchanged, so attribution/revision still wire up.

Brings render/pushcube.py into compliance with the render/CLAUDE.md
rule that phase 2 must show a real failure, not a held-still
placeholder.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Rewire `render_baseline_contrast` phase 2

`render_baseline_contrast` (same file, used for the selective-vs-full-replan side-by-side videos) has the same held-still placeholder. Fixing it keeps the two renders symmetric and prevents the same complaint from recurring on baseline figures.

**Files:**
- Modify: `babysteps/render/pushcube.py:195-279` (`render_baseline_contrast` function)

- [ ] **Step 1: Rewrite the phase 2 block inside `render_baseline_contrast`**

Find the phase 2 block inside `render_baseline_contrast` (around lines 218-221):

```python
    # === Phase 2 — ATTEMPT 1 (planner_failed, held still) ===
    # Execution phases are observed in the first-person panda_wristcam view.
    obs, _ = env.reset(seed=seed)
    attempt1_frames = [render_wrist_frame(env)] * (fps * 2)
```

Replace with:

```python
    # === Phase 2 — ATTEMPT 1 (approach physically obstructed) ===
    obstacle = _get_or_build_obstacle(env)
    _move_obstacle_to_block(
        obstacle, s["scene"].cube_xy, s["scene"].cube_z, initial_intent,
    )
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

- [ ] **Step 2: Update the phase 2 caption in `render_baseline_contrast`**

In the same function, find the `a1_title = (...)` block (around lines 254-258):

```python
    a1_title = (
        f"{short_id}  phase 2/4: approach_blocked",
        f"approach_direction={initial_intent.approach_direction} "
        f"is blocked → planner_failed",
    )
```

Replace with:

```python
    a1_title = (
        f"{short_id}  phase 2/4: approach_blocked",
        f"approach_direction={initial_intent.approach_direction} "
        f"physically obstructed → arm stalls",
    )
```

- [ ] **Step 3: Run all render-module tests**

Run:
```bash
cd /scratch/gilbreth/wang4433/babysteps && \
python -m pytest tests/test_render_modules.py -v
```

Expected: all tests pass, including the baseline-contrast tests `test_pushcube_baseline_contrast_emits_four_phases` and `test_pushcube_baseline_contrast_perturbs_contact_region`. They assert keys, frame counts ≥1, and the presence of `"approach_substitution"` / `"contact_region"` / `"full_replan"` in subtitles — all unchanged.

- [ ] **Step 4: Commit**

```bash
cd /scratch/gilbreth/wang4433/babysteps && \
git add babysteps/render/pushcube.py && \
git commit -m "fix(render): PushCube baseline-contrast phase 2 mirrors render_episode

Same obstacle + real-execution rewire as render_episode so the
selective-vs-full-replan baseline videos do not also show a held-still
phase 2.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Full sim-free pytest run

**Files:** none modified — verification only.

- [ ] **Step 1: Run the full sim-free suite**

Run:
```bash
cd /scratch/gilbreth/wang4433/babysteps && \
python -m pytest tests/ -q
```

Expected: all tests pass. The suite should report something like `343 passed in 1.3s` (or `347 passed` if the 4 new `prop_action` tests bring the total up). Either way: **zero failures**.

If anything fails, do not proceed to Task 9. Investigate, fix in-place, re-run.

- [ ] **Step 2: Verify no runtime imports of `sapien` at module load**

Run:
```bash
cd /scratch/gilbreth/wang4433/babysteps && \
python -c "
import sys
import babysteps.render.pushcube
print('sapien imported at module load?', 'sapien' in sys.modules)
"
```

Expected: `sapien imported at module load? False`. The `import sapien` calls inside `_get_or_build_obstacle`, `_move_obstacle_to_block`, and `_park_obstacle` are function-local, so simply importing the render module on the login node must not trigger a Vulkan-bound import.

---

## Task 9: GPU render smoke check

**Files:** new MP4s under `renders/pushcube/videos_maniskill/` (not committed in this task — eyeball-check only).

The login node has no Vulkan; this step requires a GPU. Submit via Slurm or run interactively with `srun`. Either path is fine — the user will trigger this with their preferred Slurm pattern.

- [ ] **Step 1: Submit the render job**

Use the existing batch script:
```bash
cd /scratch/gilbreth/wang4433/babysteps && \
sbatch slurm/render_pushcube.sbatch
```

The script renders 2 episodes (`--n_episodes 2 --seed_start 0`) and writes to `renders/pushcube/videos_maniskill/`. Wait for `squeue -u $USER` to show the job complete (~5 min, A100 partition).

If `sbatch` is unavailable in this session, fall back to interactive:
```bash
cd /scratch/gilbreth/wang4433/babysteps && \
srun --account=rpaleja --partition=a100-40gb --gres=gpu:1 --mem=115G --time=00:20:00 bash -lc '
  cd /scratch/gilbreth/wang4433/babysteps &&
  source /apps/external/conda/2025.09/etc/profile.d/conda.sh &&
  conda activate handover &&
  LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" \
  python scripts/render_stage0_maniskill.py \
    --task PushCube-v1 --n_episodes 2 --seed_start 0 \
    --out_dir renders/pushcube
'
```

- [ ] **Step 2: Inspect the new MP4s**

Run:
```bash
ls -lh /scratch/gilbreth/wang4433/babysteps/renders/pushcube/videos_maniskill/ && \
date  # confirm timestamps are post-implementation
```

Expected: 6 MP4s (2 seeds × 3 phases), newly written (timestamps after Task 6 commit).

Open each pair:

```
renders/pushcube/videos_maniskill/pushcube_blocked_approach_seed_0000__1_demo.mp4
renders/pushcube/videos_maniskill/pushcube_blocked_approach_seed_0000__2_attempt_blocked.mp4
renders/pushcube/videos_maniskill/pushcube_blocked_approach_seed_0000__3_retry.mp4
```

Visual checks (per the spec's acceptance criteria):

| Phase | What you should see |
|---|---|
| 1 (demo) | Robot pushes cube smoothly toward goal — no flying / skipping. Cube travels along the ground, decelerates near goal. |
| 2 (attempt_blocked) | Robot arm approaches blocked side, **visibly hits a red wall**, stalls. Clip ends after ~2-4 seconds. Cube has not moved. |
| 3 (retry) | Robot pushes from an orthogonal side, cube travels to goal — same smoothness as phase 1, no flying. |

- [ ] **Step 3: If any visual check fails, tune and re-run**

| Symptom | Knob to adjust | File / line |
|---|---|---|
| Cube still flies during demo or retry | Increase `_PUSHCUBE_POS_SCALE` for descend (idx 2) and push (idx 3): try `(0.10, 0.10, 0.60, 0.70)` | `babysteps/render/pushcube.py` |
| EE goes around or over the wall | Increase `_OBSTACLE_HALF_H` to 0.075 (taller wall) | `babysteps/render/pushcube.py` |
| EE clips through the wall | Increase `_OBSTACLE_HALF_T` to 0.005 (thicker wall) | `babysteps/render/pushcube.py` |
| Wall too close to cube, blocks the push itself | Increase `_OBSTACLE_BLOCK_MARGIN_M` to 0.04 | `babysteps/render/pushcube.py` |
| Phase 2 clip too long (no-progress doesn't fire) | Decrease `no_progress_break_steps` to 15 in `render_episode` (and `render_baseline_contrast`) | `babysteps/render/pushcube.py` |
| Phase 2 clip too short (cuts before wall contact) | Increase `max_steps` to 180 | `babysteps/render/pushcube.py` |

After any tune, re-submit `sbatch slurm/render_pushcube.sbatch`, re-inspect. Repeat until all three visual checks pass.

- [ ] **Step 4: Commit any tuning changes**

If tuning was needed:
```bash
cd /scratch/gilbreth/wang4433/babysteps && \
git add babysteps/render/pushcube.py && \
git commit -m "tune(render): PushCube obstacle / damping values from GPU smoke

Eyeball-tuned on seeds 0 and 1 after the first render smoke. See spec
docs/superpowers/specs/2026-05-23-pushcube-render-real-failure-stable-push-design.md.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

If no tuning was needed, no commit here — proceed to Task 10.

---

## Task 10: Final verification + summary

**Files:** none modified.

- [ ] **Step 1: Re-run the full sim-free suite**

Run:
```bash
cd /scratch/gilbreth/wang4433/babysteps && \
python -m pytest tests/ -q
```

Expected: all tests pass — same status as Task 8.

- [ ] **Step 2: Confirm git log**

Run:
```bash
cd /scratch/gilbreth/wang4433/babysteps && \
git log --oneline -n 10
```

Expected commit sequence (most recent first):
1. (optional) `tune(render): ...`
2. `fix(render): PushCube baseline-contrast phase 2 mirrors render_episode`
3. `fix(render): PushCube phase 2 executes against a real obstacle`
4. `feat(render): obstacle build/move/park helpers (defensive no-op)`
5. `feat(render): _execute_push gains max_steps + no-progress break`
6. `feat(render): per-phase damped POS_SCALE for PushCube push`
7. `feat(render): prop_action accepts optional pos_scale (damping knob)`
8. `design(render): pushcube real failure + stable push (render-only)`

- [ ] **Step 3: Report completion**

Summary to the user:
- Spec: `docs/superpowers/specs/2026-05-23-pushcube-render-real-failure-stable-push-design.md`
- Plan: `docs/superpowers/plans/2026-05-23-pushcube-render-real-failure-stable-push.md`
- Commits: 6-7 small commits as above, all on `master`.
- Sim-free suite: green.
- GPU smoke: 6 MP4s in `renders/pushcube/videos_maniskill/`, all three visual checks pass.
- Follow-up (not part of this plan): decide whether to git-commit the new MP4s, and whether to backport the damping to `pushcube_runner.py`.

---

## Self-review

**Spec coverage:**

- ✓ Damped per-phase POS_SCALE (spec §Component 1) — Tasks 2, 3.
- ✓ Obstacle build / lifecycle / dimensions (spec §Component 2) — Task 5; Tasks 6, 7 wire it into phases 2.
- ✓ Phase 2 real execution + `max_steps` + no-progress break (spec §Component 3) — Tasks 4, 6, 7.
- ✓ Caption update (spec §Captions) — Tasks 6, 7.
- ✓ Sim-free pytest stays green (spec §Testing) — Tasks 1, 8.
- ✓ Render smoke (spec §Testing) — Task 9.
- ✓ Render-only scope (spec §Scope / non-goals) — no plan task touches `pushcube_runner.py`, `skills/push.py`, or any adapter; explicitly verified by the absence of those paths in every "Files" block.

**Placeholder scan:** No "TBD", "TODO", or "implement later". Every step has either the exact code to write or the exact command to run with expected output.

**Type consistency:** `_PUSHCUBE_POS_SCALE`, `_OBSTACLE_HALF_W/T/H`, `_OBSTACLE_PARK_Z`, `_OBSTACLE_BLOCK_MARGIN_M` defined once in Task 3/5 and referenced consistently in Tasks 5, 6, 7, 9. `_get_or_build_obstacle` / `_move_obstacle_to_block` / `_park_obstacle` names match across definition (Task 5) and call sites (Tasks 6, 7). The `_execute_push` kwargs `max_steps`, `no_progress_break_steps`, `no_progress_eps_m` defined in Task 4 and called with the same names in Tasks 6, 7.
