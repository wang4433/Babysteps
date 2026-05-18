# Stage-0 Sub-project D (TurnFaucet-v1) — Design Spec

> **Superseded by** `2026-05-18-stage0-turnfaucet-embodiment-design.md`.
> Reason: the `constraint_introduction` story violated the single-factor
> revision invariant (changed two factors at once) and the real-sim
> acceptance gate (§15.6: 0/5 seeds reached `info["success"]`). The
> `embodiment_substitution` reframe is the replacement. This document
> is kept on disk for historical context.

**Sub-project D** of the Stage-0 four-scene roadmap. Replaces §7.3's
OpenCabinetDrawer-v1 (which is `SUPPORTED_ROBOTS=["fetch"]` — incompatible
with the project's Franka-only invariant) with `TurnFaucet-v1`, the only
Franka-compatible articulated tabletop task in ManiSkill 3. The §7.3
intent factors (`constraint_region` + `contact_region`) are preserved;
only the task name and vocabulary change.

- Date: 2026-05-17
- Predecessor spec: `2026-05-17-stage0-four-scene-roadmap-design.md` (§7.3, §4–6)
- Sub-project C (StackCube): committed at HEAD `c0ddb3e` (221 tests)
- Goal-of-record: `goal.md` (Stage-0 boundary; Franka/Panda only)

## 1. Motivation

§7.3 of the four-scene roadmap probes the `constraint_region` intent
factor (plus `contact_region` as a secondary revision). The original
spec assumed OpenCabinetDrawer-v1; investigation 2026-05-17 confirmed
that task is `mobile_manipulation` with `SUPPORTED_ROBOTS = ["fetch"]`,
incompatible with the project's "robot is always Franka/Panda"
invariant. TurnFaucet-v1 is the only Franka-compatible articulated
tabletop task in ManiSkill 3 (cf. memory note
`franka-only-articulated-tasks`).

The same failure narrative shape works: demo summarizes the gripper
target as the *body* of the articulated object (the faucet base, not
the rotating handle); attempt 1 touches the wrong spot, joint doesn't
move; revision introduces a `constraint_region` AND swaps the
`contact_region` to the handle; retry rotates the handle. This is the
only Stage-0 revision operator (`constraint_introduction`) that
legitimately revises two factors at once.

## 2. Stage-0 controlled failure

Natural failure, like StackCube. The wrong-contact waypoints physically
target the faucet base (a non-articulated static link); the joint
doesn't rotate; the env_runner detects "tried to actuate but
`angle_dist` didn't decrease" and sets `constraint_violation=True`.
`default_blocked_factory` returns `()`. The deliberate
under-specification lives entirely in `scripted_demo_to_intent`
(returns `contact_region="faucet_base"`, `constraint_region="none"`).

## 3. Acceptance Gate (mirrors C's)

1. All pre-D tests pass byte-identical (≥221).
2. New tests: TurnFaucetAdapter parity + TurnSkill geometry +
   revision.constraint_introduction branch + new
   FailurePredicate / failure attribution + schema additions
   (~35 tests).
3. Snapshot test: `tests/snapshots/turnfaucet_samples_seeds_0_4.jsonl`
   exists and is byte-stable under repeated
   `scripts/stage0_collect.py --fake-env --task TurnFaucet-v1
   --n_episodes 5 --seed_start 0`.
4. GPU visual spot-check:
   `render_stage0_maniskill.py --task TurnFaucet-v1 --n_episodes 2`
   produces three MP4s per episode showing (i) demo grips handle and
   rotates the faucet, (ii) attempt-1 touches the faucet base with no
   rotation, (iii) retry grips handle and rotates.
   - **Asset prerequisite:** `partnet_mobility_faucet` must be
     downloaded via `python -m mani_skill.utils.download_asset
     partnet_mobility_faucet` (~few hundred MB) before this command
     can succeed.
5. `report.md` summarizer reports `delta_pp >= 10` for TurnFaucet via
   fake-env (Pick4Pass M-BABY-1 bar).

## 4. Schema deltas

All deltas land in `babysteps/schemas.py`. Additive only — PushCube,
PickCube, StackCube records do not contain any of these tokens, so
their snapshot tests stay byte-identical.

```python
GOAL_STATES         += {"faucet_turned"}                       # D
OBJECT_MOTIONS      += {"turn"}                                # D
CONTACT_REGIONS     += {"faucet_base", "handle_grip"}          # D
CONSTRAINT_REGIONS  += {"faucet_base_static"}                  # D
FAILURE_PREDICATES  += {"constraint_violation"}                # D
REVISION_OPERATORS  += {"constraint_introduction"}             # D
EMBODIMENT_MAPPINGS += {"proxy_contact_to_franka_turn"}        # D
```

Note: `CONTACT_REGIONS` was previously `{"minus_x_face", "plus_x_face",
"minus_y_face", "plus_y_face"}` (cardinal cube faces for Push/Pick/
Stack). Adding `faucet_base` / `handle_grip` introduces a second
"family" of contact_region values. The PickCube revision
(`_pick_unblocked_face` in `revision.py`) iterates over a hardcoded
`_FACE_FALLBACK_ORDER` of the four cardinal faces — those new tokens
must NOT appear in that fallback order, or they'd be selected as
"unblocked" cube faces by accident. Verified inline: `_FACE_FALLBACK_ORDER`
is a module-private constant in `revision.py` and is not derived from
`CONTACT_REGIONS`, so the addition is safe.

## 5. Failure attribution

`FAILURE_TO_FACTOR` gains one entry:

```python
"constraint_violation":  ("constraint_region", ("constraint_region", "contact_region")),
```

`build_failure_packet`'s precedence chain extends:

```
success → planner_failed → constraint_violation → grasp_slip
       → contact_failure → no_motion → direction_error → goal_not_satisfied
```

`constraint_violation` lands between `planner_failed` and `grasp_slip`
because its evidence (the executor's `constraint_violation=True` flag)
is more specific than `grasp_slip` (which says "grasp lost") or
`no_motion` (which says "object stationary" without explaining why).

Implementation: in `build_failure_packet`, after the `planner_failed`
branch and before the `grasp_slip` branch, add:

```python
elif attempt.collision and not attempt.object_moved:
    # Sub-project D: constraint_violation when the gripper contacted a
    # non-articulating link and tried to actuate it. The env_runner
    # marks this case by setting collision=True (proxy for "touched
    # something that didn't move").
    predicate = "constraint_violation"
```

(We reuse the existing `AttemptResult.collision` flag to carry the
signal — see §9. Adding a new dedicated field would break the snapshot
contract; piggy-backing on `collision` keeps the records compatible.)

The execution_trace dict already includes `collision`, so no new
field is exposed.

## 6. Revision operator `constraint_introduction`

New branch in `babysteps/revision.py::revise_intent`:

```python
if attribution.wrong_factor == "constraint_region":
    # Stage-0's constraint_introduction is the only operator that
    # touches two factors at once. It is triggered when the demo
    # under-specified BOTH the contact_region (faucet_base) AND the
    # constraint_region (none). Refinement adds the constraint AND
    # swaps the contact to handle_grip. Strict-extension: handles only
    # the (none, faucet_base) → (faucet_base_static, handle_grip)
    # transition.
    if intent.constraint_region != "none" or intent.contact_region != "faucet_base":
        raise NotImplementedError(
            f"constraint_introduction does not handle transitions from "
            f"(constraint_region={intent.constraint_region!r}, "
            f"contact_region={intent.contact_region!r}). (Stage-0 supports "
            f"only the (none, faucet_base) → (faucet_base_static, "
            f"handle_grip) refinement per "
            f"docs/superpowers/specs/2026-05-17-stage0-turnfaucet-d-design.md §6)"
        )
    revised = replace(
        intent,
        constraint_region="faucet_base_static",
        contact_region="handle_grip",
    )
    # frozen excludes BOTH revised factors
    frozen = tuple(
        f for f in INTENT_FIELDS
        if f not in ("constraint_region", "contact_region")
    )
    rev = Revision(
        operator="constraint_introduction",
        factor="constraint_region",     # primary factor for audit
        old_value="none",
        new_value="faucet_base_static",
        frozen_factors=frozen,
    )
    return revised, rev
```

Note the asymmetry with previous operators: `Revision.factor` /
`old_value` / `new_value` record only the primary factor
(`constraint_region`). The contact_region change is implicit in the
revised Intent and is captured by `_diff_intents` (which `run_episode`
calls to populate `metrics["factors_changed"]`). For `frozen_factors_preserved`
to be `True`, both `constraint_region` and `contact_region` must be in
the `attribution.revise` tuple — verified above by the
`FAILURE_TO_FACTOR["constraint_violation"]` entry.

## 7. `TurnFaucetAdapter`

New file `babysteps/envs/turnfaucet_adapter.py`:

| Method | Behavior |
|---|---|
| `task_id` | `"TurnFaucet-v1"` |
| `make_env_runner` | `TurnFaucetEnvRunner()` (lazy import) |
| `oracle_correct_intent` | `Intent(goal_state="faucet_turned", object_motion="turn", contact_region="handle_grip", approach_direction="from_above", constraint_region="faucet_base_static", embodiment_mapping="proxy_contact_to_franka_turn")` |
| `default_blocked_factory` | `()` — no physical blocking |
| `oracle_wrong_factor` | `"constraint_region"` if `intent.contact_region == "faucet_base"`, else `"none"` |
| `scripted_demo_to_intent` | DELIBERATELY returns `contact_region="faucet_base"`, `constraint_region="none"`, with `object_motion="turn"`, `approach_direction="from_above"`, `embodiment_mapping="proxy_contact_to_franka_turn"`, `goal_state="faucet_turned"` (the goal isn't under-specified; only contact_region + constraint_region are) |
| `compile_skill` | `compile_intent_to_turn_skill(intent, scene)` |

`scripted_demo_to_intent` ignores `evidence.contact_region_label`
entirely (always returns `"faucet_base"`) — this is the Stage-0
information loss: the 2D summarizer can't distinguish the handle from
the body. A future learned summarizer would.

## 8. `babysteps/skills/turn.py` — `TurnSkill` + `compile_intent_to_turn_skill`

```python
# Constants
DESCEND_CLEARANCE_M: float = 0.03      # gap above contact z before close
TURN_PULL_DISTANCE_M: float = 0.05     # pull stroke once gripper closed

@dataclass(frozen=True)
class TurnSkill:
    """A compiled approach-grip-pull trajectory.

    waypoints is (4, 7): approach above contact_xy at travel_z, descend
    above contact at clearance, grip at contact_z, pull along
    target_joint_axis direction. Columns are [x, y, z, qx, qy, qz, qw].

    contact_region is one of {"faucet_base", "handle_grip"} and is
    used by the env_runner for failure attribution (whether the
    constraint_violation flag should fire).
    """
    waypoints: np.ndarray
    contact_region: str
    target_joint_axis_xy: tuple[float, float]


def compile_intent_to_turn_skill(intent, scene):
    if intent.contact_region == "faucet_base":
        contact_xy = scene.extra["faucet_base_xy"]
        contact_z = scene.extra["faucet_base_z"]
    elif intent.contact_region == "handle_grip":
        contact_xy = scene.extra["handle_xy"]
        contact_z = scene.extra["handle_z"]
    else:
        raise ValueError(
            f"compile_intent_to_turn_skill: contact_region must be one "
            f"of {{'faucet_base', 'handle_grip'}}, got {intent.contact_region!r}"
        )
    axis_xy = scene.extra["target_joint_axis_xy"]
    pull_xy = (contact_xy[0] + axis_xy[0] * TURN_PULL_DISTANCE_M,
               contact_xy[1] + axis_xy[1] * TURN_PULL_DISTANCE_M)
    tcp = np.asarray(scene.tcp_start_pose, dtype=np.float64)
    travel_z = float(tcp[2])

    wp = np.zeros((4, 7), dtype=np.float64)
    wp[0, 0:2] = contact_xy
    wp[0, 2] = travel_z
    wp[1, 0:2] = contact_xy
    wp[1, 2] = contact_z + DESCEND_CLEARANCE_M
    wp[2, 0:2] = contact_xy
    wp[2, 2] = contact_z
    wp[3, 0:2] = pull_xy
    wp[3, 2] = contact_z
    wp[:, 3:7] = tcp[3:7]
    return TurnSkill(
        waypoints=wp,
        contact_region=intent.contact_region,
        target_joint_axis_xy=axis_xy,
    )
```

## 9. `babysteps/envs/turnfaucet_runner.py` — `TurnFaucetEnvRunner`

Mirrors `StackCubeEnvRunner`. Differences:

- `reset()` populates `SceneState.extra` with:
  - `handle_xy` — from `obs["extra"]["target_link_pos"][0:2]`
  - `handle_z`  — from `obs["extra"]["target_link_pos"][2]`
  - `faucet_base_xy` — Stage-0 approximation: `(handle_xy[0] - 0.05, handle_xy[1])`. The real faucet's body root has a different pose, but for Stage-0 the constant offset suffices because the runner's only use of this is "where to send the gripper if intent.contact_region == faucet_base." The real body geometry varies per faucet model; the constant offset gives a deterministic wrong target.
  - `faucet_base_z` — table_z (≈ 0.0); Stage-0 assumes the base is grounded.
  - `target_joint_axis_xy` — first two components of `obs["extra"]["target_joint_axis"]`. For the pull direction.
- `scene.goal_xy` = `handle_xy` (convenience).
- `scene.cube_xy` / `scene.cube_z` — reused as "object xy/z to track"; for TurnFaucet, these are populated with `handle_xy` / `handle_z` so existing
  metric paths (object_displacement uses scene.cube_xy via AttemptResult) work without TurnFaucet-specific branches.
- `run()`:
  - Compile skill, execute 4 phases with gripper schedule `[OPEN, OPEN, CLOSED, CLOSED]`.
  - At end of execution: read `info["success"]` (faucet rotated past target).
  - Compute `initial_angle_dist` from obs at reset, `final_angle_dist` from obs after execution.
  - If `intent.contact_region == "faucet_base" AND not info["success"] AND angle_dist did not decrease`: set `collision=True` (Stage-0 proxy signal for `constraint_violation` predicate per §5).
  - Otherwise `collision=False`.
- `object_moved`: based on whether the faucet handle position changed (delta on `target_link_pos`). For the faucet, when the handle rotates, its position changes too.
- `grasp_slip = False, planner_failed = False` always.

## 10. ManiSkill TurnFaucet-v1 facts (verified by reading
`mani_skill/envs/tasks/tabletop/turn_faucet.py`)

- `SUPPORTED_ROBOTS = ["panda", "panda_wristcam", "fetch"]`. Default
  `panda_wristcam`. We use `panda_wristcam` (matches default).
- `obs["extra"]` keys: `tcp_pose`, `target_angle_diff`, `target_joint_axis`
  (3D unit vector), `target_link_pos` (3D point — center of mass of
  target switch link). State mode also exposes `angle_dist`.
- Success: `angle_dist < 0` (current angle exceeds target).
- **Asset requirement:** `partnet_mobility_faucet` (`asset_download_ids`
  in the @register_env decorator). Must be downloaded with
  `python -m mani_skill.utils.download_asset partnet_mobility_faucet`
  before the real env_runner can be instantiated.

## 11. `FakeTurnFaucetEnvRunner` (in `tests/conftest.py`)

```python
class FakeTurnFaucetEnvRunner:
    """Deterministic, sim-free env_runner for TurnFaucet unit tests.

    Outcome:
      - intent.contact_region == "handle_grip" AND
        intent.constraint_region == "faucet_base_static"
        → success=True, faucet rotated
      - intent.contact_region == "faucet_base" (anything else)
        → success=False, collision=True (constraint_violation signal)
    """
    def reset(self, seed):
        rng = np.random.default_rng(seed)
        # Synthetic handle / base positions per seed.
        handle_xy = (
            float(rng.uniform(0.05, 0.12)),
            float(rng.uniform(-0.05, 0.05)),
        )
        handle_z = 0.10
        base_xy = (handle_xy[0] - 0.05, handle_xy[1])
        # Pull axis: +y, deterministic.
        axis_xy = (0.0, 1.0)
        return SceneState(
            cube_xy=handle_xy, cube_z=handle_z,
            goal_xy=handle_xy,
            tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0),
            blocked_sides=(),
            extra={
                "handle_xy": handle_xy,
                "handle_z": handle_z,
                "faucet_base_xy": base_xy,
                "faucet_base_z": 0.0,
                "target_joint_axis_xy": axis_xy,
            },
        )

    def run(self, intent, scene):
        from babysteps.skills.turn import compile_intent_to_turn_skill
        skill = compile_intent_to_turn_skill(intent, scene)
        assert skill is not None
        handle_xy = scene.extra["handle_xy"]
        if (intent.contact_region == "handle_grip"
                and intent.constraint_region == "faucet_base_static"):
            success, collision = True, False
            final_xy = handle_xy  # cube_xy alias; faucet rotated in place
        else:
            success, collision = False, True
            # Cube_xy stays at original (faucet didn't move).
            final_xy = handle_xy
        synthetic_traj = (handle_xy, final_xy)
        return AttemptResult(
            initial_obj_xy=handle_xy, final_obj_xy=final_xy,
            goal_xy=scene.goal_xy,
            reached_contact=True,
            object_moved=success,         # only "moves" when faucet turns
            planner_failed=False, collision=collision, grasp_slip=False,
            rollout_log_path=None, success=success,
            trajectory_xy=synthetic_traj,
        )
```

A `fake_turnfaucet_env_runner` pytest fixture is added.

## 12. Render module `babysteps/render/turnfaucet.py`

`render_episode(env, adapter, seed, fps) -> (frames_dict, titles_dict)`.

- Phase 1 (`"demo"`): oracle intent (handle_grip, faucet_base_static)
  executed. Faucet rotates.
- Phase 2 (`"attempt_blocked"`): scripted intent (faucet_base, none)
  actually stepped. Gripper touches base, no rotation. Tail-padded by
  `fps` frames.
- Phase 3 (`"retry"`): revised intent (handle_grip,
  faucet_base_static). Faucet rotates.

Reads obs via local `_read_turn_obs` (target_link_pos + target_joint_axis).
Imports `STACKCUBE_MAX_CONTROL_STEPS` (reused; the cap is 400). The
title strings mention `constraint_region` (demo subtitle) and
`constraint_introduction` (retry subtitle).

## 13. CLI integration

One-row additions to two existing registries:

```python
# babysteps/envs/task_registry.py
def _turnfaucet_entry() -> TaskEntry:
    from babysteps.envs.turnfaucet_adapter import TurnFaucetAdapter
    def _make_fake():
        from tests.conftest import FakeTurnFaucetEnvRunner
        return FakeTurnFaucetEnvRunner()
    return TaskEntry(
        adapter_cls=TurnFaucetAdapter,
        fake_runner_factory=_make_fake,
        episode_id_prefix="turnfaucet_wrong_contact",
    )

TASK_REGISTRY["TurnFaucet-v1"] = _turnfaucet_entry()

# babysteps/render/__init__.py
def _turnfaucet_render() -> RenderEpisodeFn:
    from babysteps.render.turnfaucet import render_episode
    return render_episode

RENDER_REGISTRY["TurnFaucet-v1"] = _turnfaucet_render
```

The CLI scripts (stage0_collect, render_stage0_maniskill) need no
code changes — they dispatch through the registries via `--task`.

## 14. Test plan

Net new tests: ~38 (taking total from 221 to ~259).

| File | New tests |
|---|---|
| `tests/test_schemas.py` | +7 (each new whitelist token) |
| `tests/test_failure.py` | +2 (`constraint_violation` predicate precedence; FAILURE_TO_FACTOR entry) |
| `tests/test_revision.py` | +4 (constraint_introduction happy path; NotImplementedError on unknown (constraint, contact) pair; two-factor revision; frozen-factors audit covers both) |
| `tests/test_turn_skill.py` (NEW) | ~6 (per-contact_region waypoint geometry, raises on unknown, axis-derived pull direction) |
| `tests/test_turnfaucet_adapter.py` (NEW) | ~12 (parity tests + scripted_demo_to_intent always returns faucet_base + run_episode round-trip + snapshot) |
| `tests/test_pickcube_delta_pp.py` | +1 (TurnFaucet fake-env delta_pp >= 10) |
| `tests/test_stage0_collect_cli.py` | +1 row in existing parametrize (TurnFaucet-v1 + snapshot) |
| `tests/test_render_modules.py` | +3 (TurnFaucet render: keys + phase-2 stepping + titles mention constraint_region/introduction) |
| `tests/test_task_registry.py` | +2 (`get_task_entry("TurnFaucet-v1")` and updated `test_registry_contains_all_stage0_tasks` to include TurnFaucet-v1; the unknown-task sentinel rotates again — pick something definitely-future like `Bogus-v1`) |
| `tests/snapshots/turnfaucet_samples_seeds_0_4.jsonl` (NEW) | — (5 records) |

All pre-existing snapshots (PushCube + PickCube + StackCube) must
remain byte-identical.

## 15. Risks & open questions

1. **Real-physics outcome of `faucet_base` attempt is faucet-model-dependent.**
   The base position approximation (`handle_xy - (0.05, 0)`) may or may
   not actually contact a non-articulating link depending on the faucet
   model loaded. If the gripper "misses" the base entirely (lands in
   empty space), the `collision=True` signal won't fire from physics
   alone. Mitigation: the env_runner's logic sets `collision=True`
   based on `(contact_region == faucet_base AND not info["success"])`,
   which is a Stage-0 proxy, not a true physics signal. This is the
   same "controlled-failure proxy" pattern PickCube uses for slip.
2. **`collision` field repurposed.** `AttemptResult.collision` was
   previously always `False` for all tasks. Using it as the
   `constraint_violation` signal repurposes a previously-unused field.
   No snapshot drift because all prior snapshots have `collision: False`
   in every record; the new TurnFaucet records will have
   `collision: true` in failed-attempt records. This is additive.
3. **`constraint_introduction` is a two-factor revision.** The
   `metrics["factors_changed"]` field will include both
   `constraint_region` and `contact_region` for TurnFaucet revised
   episodes. For `frozen_factors_preserved` to be `True`,
   `attribution.revise` must contain both factors — verified above by
   the `FAILURE_TO_FACTOR["constraint_violation"]` entry's tuple.
4. **Asset download is a new prerequisite.** The fake-env tests don't
   need it, but the GPU spot-check (gate item 4) does. CLAUDE.md will
   note the download command. The asset is ~few hundred MB.
5. **TurnSkill's "pull" approximation.** Real faucet rotation requires
   tangential pull around the joint axis with the gripper closed. The
   Stage-0 skill does a straight pull along `target_joint_axis_xy`
   for a fixed distance (`TURN_PULL_DISTANCE_M = 0.05`). This may or
   may not actually rotate the faucet enough to clear the success
   threshold on a given seed. The fake-env synthesizes success
   deterministically; real-sim performance is best-effort and will
   be confirmed by the GPU spot-check.

## 16. Plan file

This spec drives one plan file (created next via writing-plans):

- `docs/superpowers/plans/2026-05-17-stage0-turnfaucet-d-plan.md`

## 17. Summary

Sub-project D lands TurnFaucet-v1 as the fourth Stage-0 adapter,
covering the `constraint_region` intent factor via the
`constraint_introduction` two-factor revision operator. The dispatch
registries and CLI scripts receive one-row additions; the bulk of the
work is the TurnFaucetAdapter + TurnFaucetEnvRunner + TurnSkill +
render module + the snapshot-stable test suite, plus the new
`constraint_violation` failure predicate and the
`constraint_introduction` revision operator. The original §7.3
narrative is preserved with faucet vocabulary substituting for
drawer/cabinet vocabulary.
