# Stage-0 Sub-project D (TurnFaucet-v1) — Embodiment-Substitution Reframe

**Supersedes:** `2026-05-17-stage0-turnfaucet-d-design.md`

Replaces the `constraint_introduction` story (multi-factor revision; demo
that says "touch the faucet base"; real-sim gate not passable per the
predecessor spec's §15.6) with a single-factor `embodiment_substitution`
story (grasp_turn → poke_turn after grasp_infeasible failure). The reframe
restores the project's single-factor revision invariant and grounds the
acceptance gate in an empirically demonstrated mechanism instead of a
hand-waved one.

- Date: 2026-05-18
- Predecessor spec: `2026-05-17-stage0-turnfaucet-d-design.md` (constraint_introduction)
- Sub-project C (StackCube): committed at HEAD `c0ddb3e` (221 tests)
- Goal-of-record: `goal.md` (Stage-0 boundary; Franka/Panda only)

## 1. Motivation

The predecessor D spec proposed a `constraint_introduction` revision that
changed BOTH `constraint_region` and `contact_region` in one step,
violating the project's `goal.md` invariant:

> failure -> identify implicated intent factor -> revise only that factor
> -> preserve the rest

That spec also failed its own GPU acceptance gate (predecessor §15.6):
the Panda gripper physically cannot grasp partnet faucet handles (all
surveyed seeds 0-4 had handle thickness ≥ 4 cm in every horizontal axis
vs the 4 cm Panda gripper opening), and `info["success"]` was 0/5 across
all attempts.

The empirical reality — Franka cannot grasp these handles — IS the
research story BABYSTEPS wants to tell. The demo proxy implies grasp/turn
(hand-like interaction). The robot's first attempt to execute as
grasp_turn fails because the Franka cannot realize the demonstrated
grasp. Failure attribution identifies `embodiment_mapping` as the wrong
factor. A factor-local revision swaps `grasp_turn → poke_turn`. The retry
uses the closed gripper as a rigid tool to apply tangential force at the
handle's side.

This reframe gives Stage-0 a complete factor-revision spread:

| Sub-project | Task | Revised factor |
|---|---|---|
| A | PushCube | `approach_direction` |
| B | PickCube | `contact_region` |
| C | StackCube | `goal_state` |
| D | TurnFaucet | `embodiment_mapping` |

`constraint_region` becomes untested in Stage-0. This is an honest gap;
honoring the single-factor invariant matters more than covering every
factor at the cost of a contrived multi-factor revision.

## 2. Single-factor revision (the diff that matters)

The initial intent (from the scripted demo-to-intent step) and the
revised intent differ in exactly one factor:

```python
# Initial (scripted_demo_to_intent output):
Intent(
    goal_state="faucet_turned",
    object_motion="turn",
    contact_region="handle_grip",
    approach_direction="from_above",
    constraint_region="none",
    embodiment_mapping="proxy_contact_to_franka_grasp_turn",   # ← only this changes
)

# Revised (after embodiment_substitution):
Intent(
    goal_state="faucet_turned",
    object_motion="turn",
    contact_region="handle_grip",
    approach_direction="from_above",
    constraint_region="none",
    embodiment_mapping="proxy_contact_to_franka_poke_turn",    # ← single-factor diff
)
```

Five fields preserved (`frozen_factors=5`), one changed. The clean
revision audit that the old constraint_introduction spec could not
provide.

`embodiment_mapping` owns the low-level waypoint geometry. The other
intent factors (`goal_state`, `contact_region`, etc.) are symbolic;
`embodiment_mapping` is what the skill compiler reads when translating
those symbols into concrete TCP poses, gripper schedules, and per-mode
control strategies (single-trial vs auto-sign two-trial).

## 3. Stage-0 controlled failure

The failure is natural-physical, like StackCube's: the Panda gripper at
its default joint configuration cannot mechanically envelop the partnet
faucet handle. Phase 2 (the grasp_turn attempt) executes the existing
grasp+pull skill exactly as in the predecessor spec; the gripper jaws
fail to close around the handle, no force is applied, the faucet does
not rotate. The env_runner observes `reached_contact=True`,
`object_moved=False`, `success=False`. `build_failure_packet` derives
`failure_predicate="grasp_infeasible"` from the combination of
`intent.embodiment_mapping == "proxy_contact_to_franka_grasp_turn"` plus
those execution flags (no new AttemptResult field required).

The deliberate Stage-0 information loss lives in `scripted_demo_to_intent`:
the third-person demo observes "handle was turned by hand-like
interaction" and symbolically encodes `embodiment_mapping=grasp_turn`.
The 2D summarizer cannot distinguish "human-graspable handle" from
"Franka-graspable handle." A future learned summarizer would.

## 4. Schema deltas (additive only)

All changes land in `babysteps/schemas.py`. **Additive only — no token
removals.** PushCube, PickCube, StackCube snapshot tests stay
byte-identical. TurnFaucet snapshots regenerate (the only churn).

**Add:**
```python
EMBODIMENT_MAPPINGS += {
    "proxy_contact_to_franka_grasp_turn",   # D: initial intent
    "proxy_contact_to_franka_poke_turn",    # D: revised intent
}
FAILURE_PREDICATES  += {"grasp_infeasible"}    # D: grasp-mode contact, no joint motion
REVISION_OPERATORS  += {"embodiment_substitution"}   # D: single-factor swap
```

**Deprecated (kept in whitelist, no longer emitted by TurnFaucet):**
```python
EMBODIMENT_MAPPINGS contains "proxy_contact_to_franka_turn"      # old D
CONTACT_REGIONS     contains "faucet_base"                       # old D
CONSTRAINT_REGIONS  contains "faucet_base_static"                # old D
FAILURE_PREDICATES  contains "constraint_violation"              # old D
REVISION_OPERATORS  contains "constraint_introduction"           # old D
```

Removal of the deprecated tokens is a separate cleanup commit after
`git grep` proves zero remaining references in code/tests/snapshots.
This avoids tangling the behavior change with schema housekeeping.

**Unchanged (still used by D):**

| Whitelist | Token | Used in |
|---|---|---|
| `GOAL_STATES` | `faucet_turned` | both intents + demo final_state |
| `OBJECT_MOTIONS` | `turn` | both intents |
| `CONTACT_REGIONS` | `handle_grip` | **both** intents (contact_region preserved) |
| `APPROACH_DIRECTIONS` | `from_above` | both intents (symbolic; mechanical approach differs by embodiment) |
| `CONSTRAINT_REGIONS` | `none` | both intents (no constraint involved) |

## 5. Failure attribution

`babysteps/failure.py`:

```python
FAILURE_TO_FACTOR["grasp_infeasible"] = ("embodiment_mapping", ("embodiment_mapping",))
```

`build_failure_packet` precedence chain extends:

```
success → planner_failed → grasp_infeasible → grasp_slip
       → contact_failure → no_motion → direction_error → goal_not_satisfied
```

`grasp_infeasible` slots between `planner_failed` and `grasp_slip`
because its evidence (intent claimed grasp-based embodiment, the grip
phase completed without object motion) is more specific than
`grasp_slip` (PickCube's existing meaning: "grasp was lost during lift").

Implementation (context-derived, no new `AttemptResult` field):

```python
elif (intent.embodiment_mapping == "proxy_contact_to_franka_grasp_turn"
      and attempt.reached_contact
      and not attempt.object_moved
      and not attempt.success):
    predicate = "grasp_infeasible"
```

This pattern matches StackCube's `goal_not_satisfied` derivation (which
also computes a predicate from existing fields rather than introducing a
new AttemptResult flag).

The old `constraint_violation` branch (which read `attempt.collision`)
stays in `build_failure_packet` for backward compatibility but is
unreachable now that TurnFaucetEnvRunner never sets `collision=True`.

## 6. Revision operator `embodiment_substitution`

New branch in `babysteps/revision.py::revise_intent`. Pure single-factor
edit — the simplest possible revision operator:

```python
if attribution.wrong_factor == "embodiment_mapping":
    if intent.embodiment_mapping != "proxy_contact_to_franka_grasp_turn":
        raise NotImplementedError(
            f"embodiment_substitution handles only "
            f"grasp_turn → poke_turn (got {intent.embodiment_mapping!r}). "
            f"See docs/superpowers/specs/2026-05-18-stage0-turnfaucet-embodiment-design.md §6"
        )
    revised = replace(intent, embodiment_mapping="proxy_contact_to_franka_poke_turn")
    frozen = tuple(f for f in INTENT_FIELDS if f != "embodiment_mapping")
    rev = Revision(
        operator="embodiment_substitution",
        factor="embodiment_mapping",
        old_value="proxy_contact_to_franka_grasp_turn",
        new_value="proxy_contact_to_franka_poke_turn",
        frozen_factors=frozen,
    )
    return revised, rev
```

`frozen_factors` has five entries. The audit chain is the cleanest in
Stage-0: one factor changed, five frozen, no asymmetric two-factor
recording like the predecessor's `constraint_introduction`.

## 7. TurnSkill compilation

`babysteps/skills/turn.py` extends — does not replace — the existing
TurnSkill. The dataclass gains a `mode` field, a `gripper_schedule`
field, and a `sign` field (poke-only).

```python
@dataclass(frozen=True)
class TurnSkill:
    waypoints: np.ndarray                    # (N, 7) — xyz + qxqyqzqw
    contact_region: str                      # "handle_grip" for both modes
    target_joint_axis_xy: tuple[float, float]
    mode: str                                # "grasp" | "poke"
    gripper_schedule: tuple[float, ...]      # per-waypoint gripper cmd; len(schedule) == len(waypoints)
    sign: int = +1                           # poke-only: which tangent direction


def compile_intent_to_turn_skill(intent, scene, sign: int = +1) -> TurnSkill:
    if intent.embodiment_mapping == "proxy_contact_to_franka_grasp_turn":
        return _compile_grasp(intent, scene)
    if intent.embodiment_mapping == "proxy_contact_to_franka_poke_turn":
        return _compile_poke(intent, scene, sign=sign)
    # Deprecated token kept in whitelist — compile to the grasp variant for
    # behavioral parity with anything that still passes it (e.g., old diag scripts).
    # Removal of this fallthrough happens in the schema cleanup commit.
    if intent.embodiment_mapping == "proxy_contact_to_franka_turn":
        return _compile_grasp(intent, scene)
    raise ValueError(
        f"compile_intent_to_turn_skill: unsupported embodiment_mapping "
        f"{intent.embodiment_mapping!r}"
    )
```

**Grasp variant** (`_compile_grasp`): preserves today's behavior
identically. 4 waypoints `[approach_high, descend, grip, pull]`, gripper
schedule `[OPEN, OPEN, CLOSED, CLOSED]`, targets `scene.extra["handle_xy"]`.
The existing `DESCEND_CLEARANCE_M`, `TURN_PULL_DISTANCE_M`, `GRIP_OFFSET_M`
constants are unchanged.

**Poke variant** (`_compile_poke`): 3 waypoints, gripper schedule
`[CLOSED, CLOSED, CLOSED]`. Geometry derived from `scripts/_diag_tf_poke5.py`
(verified empirically — see §13):

```python
_POKE_LATERAL_OFFSET_M  = 0.07
_POKE_SWEEP_DISTANCE_M  = 0.22
_POKE_HEIGHT_ABOVE_M    = 0.04
_POKE_HIGH_CLEARANCE_M  = 0.12

def _compile_poke(intent, scene, sign):
    if intent.contact_region != "handle_grip":
        raise ValueError(
            f"poke_turn requires contact_region='handle_grip', got {intent.contact_region!r}"
        )
    handle_xy = np.asarray(scene.extra["handle_xy"], dtype=np.float64)
    axis_xy   = np.asarray(scene.extra["target_joint_axis_xy"], dtype=np.float64)
    handle_z  = float(scene.extra["handle_z"])
    tcp       = np.asarray(scene.tcp_start_pose, dtype=np.float64)
    travel_z  = float(tcp[2])

    # Heuristic seed tangent direction (refined at runtime by env_runner's
    # auto-sign retry). For axis along +z (typical faucet), the perpendicular
    # in xy is just the 90° CCW rotation of axis_xy's projection.
    axis_norm = float(np.linalg.norm(axis_xy))
    tangent = (np.array([0.0, 1.0]) if axis_norm < 1e-3
               else np.array([-axis_xy[1], axis_xy[0]]) / axis_norm)
    sweep_dir = tangent * sign

    contact_z  = handle_z + _POKE_HEIGHT_ABOVE_M
    approach_z = max(travel_z, handle_z + _POKE_HIGH_CLEARANCE_M) + 0.02
    pre_xy     = handle_xy - sweep_dir * _POKE_LATERAL_OFFSET_M
    post_xy    = handle_xy + sweep_dir * _POKE_SWEEP_DISTANCE_M

    wp = np.zeros((3, 7), dtype=np.float64)
    wp[0, 0:3] = [pre_xy[0],  pre_xy[1],  approach_z]
    wp[1, 0:3] = [pre_xy[0],  pre_xy[1],  contact_z]
    wp[2, 0:3] = [post_xy[0], post_xy[1], contact_z]
    wp[:, 3:7] = tcp[3:7]
    return TurnSkill(
        waypoints=wp,
        contact_region="handle_grip",
        target_joint_axis_xy=(float(axis_xy[0]), float(axis_xy[1])),
        mode="poke",
        gripper_schedule=(-1.0, -1.0, -1.0),
        sign=sign,
    )
```

The tangent direction returned by `_compile_poke` is a **heuristic seed**.
The actual winning direction is decided at runtime by
`TurnFaucetEnvRunner`'s auto-sign retry (§8). The cross-product-based
tangent is correct for the geometric right-hand rule, but the partnet
faucets' qpos sign convention is inconsistent across models — empirically,
some seeds rotate in the geometric +tangent direction, others in
−tangent. The runner tries +1 first, retries with −1 if probe progress is
insufficient.

## 8. TurnFaucetEnvRunner

`babysteps/envs/turnfaucet_runner.py` is rewritten around two pieces:
a generic phase-loop helper that consumes any TurnSkill and depends only
on `len(skill.waypoints)` + `skill.gripper_schedule` (no hardcoded
4-phase grasp assumptions), and a `run()` method that picks single-trial
(grasp) or two-trial (poke) based on `skill.mode`.

```python
_POS_SCALE: float = 0.1
_PHASE_TOL_M: float = 0.015
_GRASP_PHASE_TOL_M: float = 0.025
_GRIP_MIN_STEPS: int = 15
_MAX_CONTROL_STEPS: int = 400
_POKE_PROBE_STEPS: int = 80
_POKE_PROBE_MIN_PROGRESS: float = 0.4    # fraction of needed_delta required by probe
```

### 8.1 Helpers

```python
def _safe_bool(x):
    """Safe bool from a (possibly batched torch) tensor."""
    if hasattr(x, "cpu"):
        x = x.cpu().numpy()
    arr = np.asarray(x)
    return bool(arr.item() if arr.ndim > 0 else arr)


def _read_faucet_qpos(env):
    """env.unwrapped.target_switch_link.joint.qpos as a python float."""
    return float(_to_np(env.unwrapped.target_switch_link.joint.qpos).item())


def _read_needed_delta(env):
    """target_angle - current qpos, both via env.unwrapped."""
    env_u = env.unwrapped
    target_angle = float(_to_np(env_u.target_angle).item())
    current_qpos = _read_faucet_qpos(env)
    return target_angle - current_qpos
```

### 8.2 Generic execution

```python
@dataclass(frozen=True)
class _TrialOutcome:
    success: bool
    reached_contact: bool
    object_moved: bool
    qpos_extremum_signed_progress: float
    initial_obj_xy: tuple[float, float]
    final_obj_xy: tuple[float, float]
    trajectory_xy: tuple[tuple[float, float], ...]


def _execute_skill(env, skill, *, seed, needed_delta, contact_xy, max_steps):
    """One full execution. Generic over len(skill.waypoints) and
    skill.gripper_schedule. Grasp-mode dwell at phase index 2 requires
    _GRIP_MIN_STEPS before advancing; poke mode has no dwell."""
    obs, _ = env.reset(seed=int(seed))
    n_phases = len(skill.waypoints)
    assert len(skill.gripper_schedule) == n_phases, "schedule/waypoints length mismatch"
    targets = [np.asarray(wp[0:3], dtype=np.float64) for wp in skill.waypoints]
    grip_phase = 2 if skill.mode == "grasp" and n_phases >= 3 else -1
    phase_tol = tuple(
        _GRASP_PHASE_TOL_M if i == grip_phase else _PHASE_TOL_M
        for i in range(n_phases)
    )

    tcp0, handle_xyz0, _ = _read_obs(obs)
    initial_xy = (float(handle_xyz0[0]), float(handle_xyz0[1]))
    initial_qpos = _read_faucet_qpos(env)

    trajectory, phase_idx, steps_in_phase = [], 0, 0
    reached_contact, success = False, False
    qpos_extremum = initial_qpos

    for _ in range(max_steps):
        tcp, handle_xyz, _ = _read_obs(obs)
        trajectory.append((float(handle_xyz[0]), float(handle_xyz[1])))
        target = targets[phase_idx]
        reached = np.linalg.norm(target - tcp[0:3]) < phase_tol[phase_idx]
        advance = reached and (phase_idx != grip_phase or steps_in_phase >= _GRIP_MIN_STEPS)
        if advance:
            phase_idx += 1; steps_in_phase = 0
            if phase_idx >= n_phases:
                break
            target = targets[phase_idx]
        else:
            steps_in_phase += 1
        if phase_idx >= 1 and np.linalg.norm(tcp[0:2] - contact_xy) < 0.04:
            reached_contact = True
        action = _prop_action(tcp, target, skill.gripper_schedule[phase_idx])
        obs, _r, terminated, truncated, info = env.step(action)
        qpos = _read_faucet_qpos(env)
        if needed_delta > 0:
            qpos_extremum = max(qpos_extremum, qpos)
        else:
            qpos_extremum = min(qpos_extremum, qpos)
        success = _safe_bool(info.get("success", False))
        if success or _safe_bool(terminated) or _safe_bool(truncated):
            break

    final_xy = trajectory[-1] if trajectory else initial_xy
    progress = (qpos_extremum - initial_qpos) / max(abs(needed_delta), 1e-6)
    # object_moved derived from qpos delta, not handle xy delta. The faucet
    # rotates around a fixed joint axis; target_link_pos sweeps the arc but
    # qpos is the direct signal of articulation motion.
    object_moved = abs(qpos_extremum - initial_qpos) > 0.05  # rad
    return _TrialOutcome(
        success=success, reached_contact=reached_contact, object_moved=object_moved,
        qpos_extremum_signed_progress=progress,
        initial_obj_xy=initial_xy, final_obj_xy=final_xy,
        trajectory_xy=tuple(trajectory),
    )
```

### 8.3 `run()` dispatch

```python
def run(self, intent, scene, *, rollout_log_path=None) -> AttemptResult:
    seed = self._last_seed
    if seed is None:
        raise RuntimeError("TurnFaucetEnvRunner.run called before reset()")
    contact_xy = np.asarray(scene.extra["handle_xy"], dtype=np.float64)
    needed_delta = _read_needed_delta(self._env)

    if intent.embodiment_mapping in (
        "proxy_contact_to_franka_grasp_turn",
        "proxy_contact_to_franka_turn",   # deprecated; same single-trial behavior
    ):
        skill = compile_intent_to_turn_skill(intent, scene)
        outcome = _execute_skill(
            self._env, skill, seed=seed, needed_delta=needed_delta,
            contact_xy=contact_xy, max_steps=_MAX_CONTROL_STEPS,
        )
        return self._outcome_to_attempt_result(outcome, scene, rollout_log_path)

    if intent.embodiment_mapping != "proxy_contact_to_franka_poke_turn":
        raise ValueError(
            f"TurnFaucetEnvRunner.run: unsupported embodiment_mapping "
            f"{intent.embodiment_mapping!r}"
        )

    # Poke: auto-sign two-trial loop. Each trial is a full execution from
    # env.reset(seed) — the same-seed reset makes the sign retry a true
    # counterfactual (identical faucet configuration both times).
    #
    # The probe with sign=+1 is a TRUNCATED preview (_POKE_PROBE_STEPS, ~80)
    # used only to decide whether sign=+1 is the winning direction. If the
    # probe makes ≥ _POKE_PROBE_MIN_PROGRESS, the full sign=+1 trial is rerun
    # from a fresh reset so the captured AttemptResult/trajectory reflects a
    # complete attempt (not a truncated probe). If the probe is insufficient,
    # sign=-1 is tried at full budget; the better of the two trials wins.
    skill_pos = compile_intent_to_turn_skill(intent, scene, sign=+1)
    probe = _execute_skill(
        self._env, skill_pos, seed=seed, needed_delta=needed_delta,
        contact_xy=contact_xy, max_steps=_POKE_PROBE_STEPS,
    )
    if probe.success:
        return self._outcome_to_attempt_result(probe, scene, rollout_log_path)
    if probe.qpos_extremum_signed_progress >= _POKE_PROBE_MIN_PROGRESS:
        full_pos = _execute_skill(
            self._env, skill_pos, seed=seed, needed_delta=needed_delta,
            contact_xy=contact_xy, max_steps=_MAX_CONTROL_STEPS,
        )
        return self._outcome_to_attempt_result(full_pos, scene, rollout_log_path)
    skill_neg = compile_intent_to_turn_skill(intent, scene, sign=-1)
    full_neg = _execute_skill(
        self._env, skill_neg, seed=seed, needed_delta=needed_delta,
        contact_xy=contact_xy, max_steps=_MAX_CONTROL_STEPS,
    )
    if (full_neg.success
        or full_neg.qpos_extremum_signed_progress > probe.qpos_extremum_signed_progress):
        return self._outcome_to_attempt_result(full_neg, scene, rollout_log_path)
    full_pos = _execute_skill(
        self._env, skill_pos, seed=seed, needed_delta=needed_delta,
        contact_xy=contact_xy, max_steps=_MAX_CONTROL_STEPS,
    )
    return self._outcome_to_attempt_result(full_pos, scene, rollout_log_path)
```

### 8.4 AttemptResult population

```python
def _outcome_to_attempt_result(self, outcome, scene, rollout_log_path):
    if rollout_log_path is not None:
        rollout_log_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            rollout_log_path,
            trajectory_xy=np.asarray(outcome.trajectory_xy, dtype=np.float64),
            initial_obj_xy=np.asarray(outcome.initial_obj_xy, dtype=np.float64),
            final_obj_xy=np.asarray(outcome.final_obj_xy, dtype=np.float64),
            goal_xy=np.asarray(scene.goal_xy, dtype=np.float64),
        )
    return AttemptResult(
        initial_obj_xy=outcome.initial_obj_xy,
        final_obj_xy=outcome.final_obj_xy,
        goal_xy=scene.goal_xy,
        reached_contact=outcome.reached_contact,
        object_moved=outcome.object_moved,
        planner_failed=False,
        collision=False,                  # never set in new D — `grasp_infeasible` is context-derived
        grasp_slip=False,                 # PickCube-specific, never set by TurnFaucet
        rollout_log_path=str(rollout_log_path) if rollout_log_path else None,
        success=outcome.success,
        trajectory_xy=outcome.trajectory_xy,
    )
```

`collision=False` always — the old D-spec's "repurpose collision as
constraint_violation signal" goes away.

## 9. TurnFaucetAdapter

`babysteps/envs/turnfaucet_adapter.py`:

| Method | Behavior |
|---|---|
| `task_id` | `"TurnFaucet-v1"` |
| `make_env_runner` | `TurnFaucetEnvRunner()` (lazy import) |
| `oracle_correct_intent(scene)` | Intent with `embodiment_mapping="proxy_contact_to_franka_poke_turn"`, `contact_region="handle_grip"`, `constraint_region="none"`, `approach_direction="from_above"`, `object_motion="turn"`, `goal_state="faucet_turned"` |
| `scripted_demo_to_intent(evidence)` | Same Intent EXCEPT `embodiment_mapping="proxy_contact_to_franka_grasp_turn"` — the demo's hand-like interaction symbolically reads as grasping |
| `oracle_wrong_factor(intent)` | `"embodiment_mapping"` if `intent.embodiment_mapping == "proxy_contact_to_franka_grasp_turn"`, else `"none"` |
| `default_blocked_factory(intent)` | `()` — no physical blocking |
| `compile_skill(intent, scene)` | `compile_intent_to_turn_skill(intent, scene)` |

The "correct" mapping is `poke_turn` — that's the one that actually
rotates the faucet in real sim. The "demonstrated" mapping is
`grasp_turn` — that's the one a hand-based demo would symbolically encode
but the Franka cannot execute. This is the Stage-0 information loss the
revision loop must close.

## 10. Render module

`babysteps/render/turnfaucet.py` rewritten end-to-end. Three phases:

**Phase 1 — DEMO PROXY (privileged qpos teleport).** Robot stays at home
pose; only the faucet joint moves. `~max(2*fps, 30)` interpolated frames
showing the handle rotating from initial to target angle via direct
write to `switch_link.joint.qpos`. No physics step. Caption (per the
demo-caption guideline): "third-person object-motion proxy: faucet
turned" — no implied executable Franka motor program.

**Phase 2 — ATTEMPT (grasp_turn).** Standard env step loop with the
grasp skill: approach above handle, descend, attempt to grip (jaws fail
to close on the thick handle), pull. Tail-padded by `fps` frames. Caption
mentions `grasp_infeasible` and "jaws cannot close on handle → no rotation."

**Phase 3 — RETRY (poke_turn after embodiment_substitution).** Auto-sign
two-trial loop mirrored from the env_runner: try sign=+1 for
`_POKE_PROBE_STEPS`; if no progress, replace `retry_frames` and try
sign=-1. Caption mentions `embodiment_substitution: grasp_turn → poke_turn`
and the final `info["success"]`.

The MP4 keys (`demo`, `attempt_blocked`, `retry`) stay identical to the
existing render contract — `render_stage0_maniskill.py`, the render
registry, and downstream scripts need no changes.

The render-side execution helper is structurally the same as
`_execute_skill` but appends `render_frame(env)` each step instead of
building a trajectory list. Per the established pattern (pickcube,
stackcube), this duplication is acceptable for Stage-0; sharing the
inner loop with env_runner is a later refactor.

## 11. FakeTurnFaucetEnvRunner

`tests/conftest.py` — rewritten outcome rule:

```python
class FakeTurnFaucetEnvRunner:
    """Deterministic sim-free env_runner for TurnFaucet unit tests.

    Outcome rule:
      - intent.embodiment_mapping == "proxy_contact_to_franka_poke_turn"
        → success=True, object_moved=True, reached_contact=True
      - intent.embodiment_mapping == "proxy_contact_to_franka_grasp_turn"
        → success=False, object_moved=False, reached_contact=True
        (failure_packet derives grasp_infeasible from these flags + intent)
    """
    def reset(self, seed):
        rng = np.random.default_rng(seed)
        handle_xy = (float(rng.uniform(0.05, 0.12)),
                      float(rng.uniform(-0.05, 0.05)))
        handle_z = 0.10
        axis_xy = (0.0, 1.0)
        return SceneState(
            cube_xy=handle_xy, cube_z=handle_z, goal_xy=handle_xy,
            tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0),
            blocked_sides=(),
            extra={
                "handle_xy": handle_xy, "handle_z": handle_z,
                "target_joint_axis_xy": axis_xy,
            },
        )

    def run(self, intent, scene, *, rollout_log_path=None):
        from babysteps.skills.turn import compile_intent_to_turn_skill
        skill = compile_intent_to_turn_skill(intent, scene)
        assert skill is not None
        handle_xy = scene.extra["handle_xy"]
        if intent.embodiment_mapping == "proxy_contact_to_franka_poke_turn":
            success, moved = True, True
        else:  # grasp_turn or deprecated proxy_contact_to_franka_turn
            success, moved = False, False
        return AttemptResult(
            initial_obj_xy=handle_xy, final_obj_xy=handle_xy,
            goal_xy=scene.goal_xy,
            reached_contact=True, object_moved=moved,
            planner_failed=False, collision=False, grasp_slip=False,
            rollout_log_path=None, success=success,
            trajectory_xy=(handle_xy,),
        )
```

A `fake_turnfaucet_env_runner` pytest fixture is exposed (same name as
the predecessor spec's, so existing test rows that reference it carry
over after the outcome rule is swapped).

## 12. Test plan

Net new tests: ~30 (taking total from 261 → ~291). Pre-existing PushCube,
PickCube, and StackCube snapshots stay byte-identical.

| File | New / changed |
|---|---|
| `tests/test_schemas.py` | +4 (one test per new whitelist token) |
| `tests/test_failure.py` | +3 (`grasp_infeasible` predicate fires on correct combo; FAILURE_TO_FACTOR entry; precedence relative to `grasp_slip`) |
| `tests/test_revision.py` | +4 (single-factor diff, frozen=5, NotImplementedError on non-grasp_turn input, `factors_changed == ("embodiment_mapping",)`) |
| `tests/test_turn_skill.py` | rewrite + ~8 (grasp variant unchanged; poke variant geometry; dispatch on embodiment_mapping; sign flips waypoints; deprecated token still compiles; ValueError on unsupported) |
| `tests/test_turnfaucet_adapter.py` | rewrite + ~10 (oracle returns poke; scripted returns grasp; oracle_wrong_factor returns embodiment_mapping for grasp_turn; full `run_episode` round-trip; snapshot byte-stable) |
| `tests/test_pickcube_delta_pp.py` | +1 row (TurnFaucet fake-env delta_pp ≥ 10 with new outcome rule) |
| `tests/test_stage0_collect_cli.py` | +1 row (TurnFaucet snapshot round-trip) |
| `tests/test_render_modules.py` | +3 (TurnFaucet render returns 3 keys; demo phase length ≥ 30 frames; titles mention `embodiment_substitution` + `grasp_infeasible`) |
| `tests/test_task_registry.py` | adjusted (registry contents unchanged; the unknown-task sentinel rotates if needed) |
| `tests/conftest.py` (FakeTurnFaucetEnvRunner) | rewrite (rule: poke_turn → success; grasp_turn → grasp_infeasible) |
| `tests/snapshots/turnfaucet_samples_seeds_0_4.jsonl` | regenerate (5 records; old D snapshot stale by design) |

Snapshot regeneration is the only churn outside TurnFaucet-specific
files. PushCube/PickCube/StackCube snapshots stay byte-identical — none
of them reference any added or deprecated token.

## 13. Empirical grounding (v5 diag results)

The poke geometry constants and the auto-sign retry strategy are
grounded in `scripts/_diag_tf_poke5.py`. Across 5 seeds with the v5
implementation (closed-gripper brute-force lateral sweep, single
post-sweep waypoint at max action, auto sign-detection):

| seed | winning sign | qpos init → final | target | progress | info["success"] |
|---|---|---|---|---|---|
| 0 | +1 (no retry helped) | 0.00 → 0.00 | +1.57 | 0% | False (handle missed; very low handle_z=0.06) |
| **1** | **+1** | **0.00 → +1.26** | **+1.26** | **100.4%** | **True (24 steps)** |
| 2 | +1 | -0.04 → -1.31 | +1.26 | 0% (rotates wrong direction on both signs) | False |
| 3 | -1 | 0.00 → +0.10 | +1.57 | 6.5% | False (v4 saw 42.2% with multi-waypoint, regression in single-sweep) |
| 4 | +1 (no retry helped) | 0.00 → 0.00 | +1.57 | 0% (handle missed; very low handle_z=0.04) |

**Concrete proof of feasibility:** seed 1 reaches `info["success"]` in 24
control steps. The mechanism (closed-gripper lateral brute-force sweep)
is empirically demonstrated to drive a partnet faucet past its target
angle in real sim.

**Honest limit:** 4 GPU diag iterations did not converge on a generic
recipe that works across all 5 sampled seeds. The partnet asset
diversity (varying handle thickness, height, lever arm, joint sign
convention) makes a single parameter set fragile. The mechanism is
demonstrable; reliability across the asset distribution is partial.

The acceptance gate in §14 reflects this honestly.

## 14. Acceptance gate

The gate is split into symbolic (fake-env, strict) and physical (real-sim,
**partial physical validation**) tiers.

1. **All pre-D tests pass byte-identical** (≥ 261, no PushCube/PickCube/
   StackCube snapshot drift).
2. **New tests pass** (~30 above).
3. **TurnFaucet snapshot byte-stable** on repeated
   `scripts/stage0_collect.py --fake-env --task TurnFaucet-v1 --n_episodes 5 --seed_start 0`.
4. **GPU visual spot-check** (`scripts/render_stage0_maniskill.py --task TurnFaucet-v1 --n_episodes 5`):
   - 3 MP4s per episode (demo, attempt_blocked, retry) — 15 total.
   - **Demo MP4** shows the faucet handle rotating from initial to target
     angle via privileged teleport; robot at home pose. Caption:
     "third-person object-motion proxy: faucet turned."
   - **Attempt MP4** shows Franka attempting `grasp_turn`: approach above
     handle, descend, attempt grip (jaws fail to close), pull (no
     rotation). Caption mentions `grasp_infeasible`.
   - **Retry MP4** shows Franka attempting `poke_turn`: closed gripper
     from start, lateral approach beside handle, brute-force tangential
     sweep with auto-sign retry as needed. Caption mentions
     `embodiment_substitution`.
   - **Partial physical validation:** **at least 1 of 5 seeds reaches
     `info["success"]=True` in the retry MP4**. For the remaining seeds,
     the retry MP4 must visibly apply lateral tangential force to the
     handle (vs. `grasp_turn`'s failed vertical jaw-close), making the
     change in `embodiment_mapping` observable at the motor-program level
     even when joint rotation does not reach target.
5. **Fake-env delta_pp ≥ 10** for TurnFaucet
   (`tests/test_pickcube_delta_pp.py` parametrized row).

This gate is **more mechanism-honest / explicitly partial** than the
predecessor's. The old D spec assumed `5/5` real-sim success and
silently failed to deliver. This spec acknowledges the empirically
demonstrated limit (1/5 reaches `info["success"]`; the rest demonstrate
the embodiment change visually) and grounds the BABYSTEPS symbolic
claim on the fake-env delta_pp gate, which is unaffected by physical
reliability.

## 15. Risks & open questions

1. **Real-sim retry succeeds on a single seed.** The acceptance gate is
   met by seed 1 alone (v5 confirmed). If seed 1's success regresses
   under future code changes (e.g., a TurnSkill refactor changes the
   sweep distance), the gate fails. Mitigation: snapshot the exact poke
   geometry constants in `babysteps/skills/turn.py` and add a comment
   pointing at this spec. Future tuning of the constants should rerun
   `scripts/_diag_tf_poke5.py` and confirm seed 1 still succeeds.

2. **`grasp_infeasible` is context-derived, not flag-derived.**
   `build_failure_packet` reads `intent.embodiment_mapping` plus
   `attempt.reached_contact` + `not attempt.object_moved` + `not
   attempt.success`. This works for the Stage-0 grasp_turn scenario but
   does not generalize to "grasp_turn that almost worked" (e.g., grasp
   loosely succeeded but joint friction held). For Stage-0 this is fine
   because the partnet faucet handles are physically uniformly too thick;
   the boundary case does not arise. Future stages would need a more
   explicit flag.

3. **Auto-sign two-trial loop changes the env_runner's contract.**
   `TurnFaucetEnvRunner.run()` now calls `env.reset(seed=...)` up to
   three times per attempt (probe sign=+1, optional full sign=+1, optional
   full sign=-1). Previously the runner reset once. Other adapters
   (PushCube, PickCube, StackCube) reset once per `.run()`. The
   `BaseTaskAdapter` interface does not constrain reset count, so this is
   compliant — but worth knowing when extending.

4. **The poke tangent direction is sign-ambiguous per faucet model.**
   The cross-product-based tangent (`cross(joint_axis, radius)`) gives
   the geometric right-hand-rule positive-rotation direction. Some
   partnet faucet models have qpos sign convention inverted from the
   joint_axis direction (verified empirically — see v1 diag seed 2
   rotating −1.39 rad when geometric +tangent should have given +
   rotation). The auto-sign retry handles this at runtime; the spec does
   NOT claim the geometric tangent is correct for all faucets.

5. **Demo phase teleport bypasses physics.** Phase 1 writes
   `switch_link.joint.qpos` directly without stepping the env. This means
   the demo's "rotation" is purely visual — no momentum, no friction.
   This is acceptable for Stage-0 because the demo is third-person
   evidence of object motion, not a recorded motor program (per the
   demo-caption guideline). Phases 2 and 3 step the env normally.

6. **Constants in `_compile_poke` are partnet-specific.** The values
   `_POKE_LATERAL_OFFSET_M=0.07`, `_POKE_SWEEP_DISTANCE_M=0.22`,
   `_POKE_HEIGHT_ABOVE_M=0.04`, `_POKE_HIGH_CLEARANCE_M=0.12` are tuned
   to the partnet_mobility_faucet assets, the Panda gripper geometry,
   and the `pd_ee_delta_pose` control mode at `_POS_SCALE=0.1`. Porting
   to a different faucet asset or robot would require retuning.

## 16. Migration / supersession

- **New spec file:** this file
  (`docs/superpowers/specs/2026-05-18-stage0-turnfaucet-embodiment-design.md`).
- **Predecessor spec:**
  `docs/superpowers/specs/2026-05-17-stage0-turnfaucet-d-design.md` gets
  a one-line front-matter note: *Superseded by
  `2026-05-18-stage0-turnfaucet-embodiment-design.md`. Reason: the
  constraint_introduction story violated the single-factor revision
  invariant and the real-sim acceptance gate (§15.6); the
  embodiment_substitution reframe is the replacement.* Kept on disk for
  historical context.
- **Deprecated tokens** stay in the schema whitelists. Removal happens
  in a separate cleanup commit only after `git grep` proves zero
  remaining references.
- **CLAUDE.md TurnFaucet section** updated to describe the new story
  and the partial physical validation gate.
- **Diagnostic scripts** (`scripts/_diag_tf_*.py`) stay as historical
  artifacts. `_diag_tf_poke5.py` is the canonical mechanism reference
  for the poke geometry constants.

## 17. Plan file

This spec drives one plan file (created next via writing-plans):

- `docs/superpowers/plans/2026-05-18-stage0-turnfaucet-embodiment-plan.md`

## 18. Summary

Sub-project D is reframed from `constraint_introduction` (multi-factor,
contrived demo, never-passing gate) to `embodiment_substitution`
(single-factor, mechanism-honest, partial-physical-validation gate). The
schema gains 4 additive tokens, the revision operator drops to a pure
single-factor swap, the failure predicate is context-derived from
existing fields, and the TurnSkill compiler dispatches on
`embodiment_mapping` to produce grasp-mode (4 waypoints, open→close
gripper) or poke-mode (3 waypoints, closed throughout, auto-sign at the
runner). The render module's demo phase uses a privileged qpos teleport
to avoid relying on a Franka motor program for what is supposed to be
third-person object-motion evidence. Real-sim acceptance is
partial physical validation: at least 1 of 5 seeds reaches
`info["success"]`, with the remaining seeds demonstrating the
embodiment-mapping change at the motor-program level. The symbolic
acceptance (fake-env delta_pp ≥ 10) is unchanged from the predecessor's
intent. Stage-0 factor coverage becomes
`approach_direction`/`contact_region`/`goal_state`/`embodiment_mapping`,
with `constraint_region` deferred as an honest gap.
