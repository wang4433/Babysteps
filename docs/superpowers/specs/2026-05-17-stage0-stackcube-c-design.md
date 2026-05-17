# Stage-0 Sub-project C (StackCube-v1) — Design Spec

**Sub-project C** of the Stage-0 four-scene roadmap. Lifts §7.2 of
`2026-05-17-stage0-four-scene-roadmap-design.md` from "scoped only" into
an implementation-ready spec by pinning the open implementation
questions (demo-to-intent shape, env_runner failure mechanism, skill
compilation per goal_state, render flow, test plan).

- Date: 2026-05-17
- Predecessor spec: `2026-05-17-stage0-four-scene-roadmap-design.md` (§7.2, §4–6)
- Sub-project B (PickCube): committed at HEAD `c5b419a` (CLI + render wired; 184 tests)
- Goal-of-record: `goal.md` (Stage-0 boundary; object-centric intent schema)

## 1. Motivation

§7.2 of the four-scene roadmap probes the `goal_state` intent factor.
The demo's 2D-trajectory summarization under-specifies "place cubeA on
cubeB" as "translate cubeA to cubeB.xy". Failure → goal_refinement →
retry stacks. This is the only one of the four scenes where the demo
*sharpens* an intent rather than substituting a wrong cardinal value —
the most novel BABYSTEPS demonstration.

## 2. Stage-0 controlled failure (the cleanest of the three so far)

Unlike PushCube's compile-time `planner_failed` or PickCube's runtime
`grasp_slip`, StackCube's failure is *natural*: the wrong-goal waypoints
physically miss the stack. The env_runner needs no `blocked_sides`
mechanism. `default_blocked_factory` returns `()`. The "controlled"
aspect lives entirely in `scripted_demo_to_intent`'s deliberate
under-specification.

## 3. Acceptance Gate (mirrors B's)

1. All pre-C tests pass byte-identical (≥184).
2. New tests: StackCubeAdapter parity + StackSkill geometry +
   revision.goal_refinement branch + schema additions (~37 tests).
3. Snapshot test: `tests/snapshots/stackcube_samples_seeds_0_4.jsonl`
   exists and is byte-stable under repeated
   `scripts/stage0_collect.py --fake-env --task StackCube-v1
   --n_episodes 5 --seed_start 0`.
4. GPU visual spot-check:
   `render_stage0_maniskill.py --task StackCube-v1 --n_episodes 2`
   produces three MP4s per episode showing (i) demo picks cubeA and
   places on cubeB (successful stack), (ii) attempt-1 drops cubeA next
   to cubeB (collision-driven side scatter), (iii) retry stacks.
5. `report.md` summarizer reports `delta_pp >= 10` for StackCube via
   fake-env (Pick4Pass M-BABY-1 bar).

## 4. Schema deltas

All deltas land in `babysteps/schemas.py`. None of them invalidate
PushCube or PickCube records — both whitelists are strict subsets.

```python
GOAL_STATES         += {"cubeA_on_cubeB"}                      # C
OBJECT_MOTIONS      += {"place_on"}                            # C
EMBODIMENT_MAPPINGS += {"proxy_contact_to_franka_pick_and_place"}  # C
REVISION_OPERATORS  += {"goal_refinement"}                     # C
# No new FAILURE_PREDICATES — goal_not_satisfied already exists.
# No new CONTACT_REGIONS, APPROACH_DIRECTIONS, CONSTRAINT_REGIONS.
```

Existing snapshot tests (`pushcube_samples_seeds_0_4.jsonl`,
`pickcube_samples_seeds_0_4.jsonl`) must remain byte-identical.

## 5. Failure attribution

**No changes.** `FAILURE_TO_FACTOR["goal_not_satisfied"] = ("goal_state",
("goal_state",))` already present and correct. `build_failure_packet`'s
precedence (success → planner_failed → grasp_slip → contact → motion →
direction → goal) already routes "cube ended up near goal but not
stacked" to `goal_not_satisfied`.

## 6. Revision operator `goal_refinement`

New branch in `babysteps/revision.py::revise_intent`:

```python
if attribution.wrong_factor == "goal_state":
    if intent.goal_state == "cube_at_target":
        new = "cubeA_on_cubeB"
    else:
        raise NotImplementedError(
            f"goal_refinement does not handle transitions from "
            f"goal_state {intent.goal_state!r}. (Stage-0 supports only "
            f"the cube_at_target → cubeA_on_cubeB refinement per "
            f"docs/superpowers/specs/2026-05-17-stage0-stackcube-c-design.md §6)"
        )
    revised = replace(intent, goal_state=new)
    frozen = tuple(f for f in INTENT_FIELDS if f != "goal_state")
    rev = Revision(
        operator="goal_refinement",
        factor="goal_state",
        old_value="cube_at_target",
        new_value="cubeA_on_cubeB",
        frozen_factors=frozen,
    )
    return revised, rev
```

Other `wrong_factor` values continue to dispatch to existing branches
(approach_substitution, contact_substitution) or raise
NotImplementedError unchanged.

## 7. `StackCubeAdapter`

New file `babysteps/envs/stackcube_adapter.py`:

| Method | Behavior |
|---|---|
| `task_id` | `"StackCube-v1"` |
| `make_env_runner()` | `StackCubeEnvRunner()` (lazy import) |
| `oracle_correct_intent(scene)` | `Intent(goal_state="cubeA_on_cubeB", object_motion="place_on", contact_region="minus_x_face", approach_direction="from_above", constraint_region="none", embodiment_mapping="proxy_contact_to_franka_pick_and_place")` |
| `default_blocked_factory(intent)` | `()` — no physical blocking; failure is from wrong-goal waypoints |
| `oracle_wrong_factor(initial_intent, scene_executor)` | `"goal_state"` if `initial_intent.goal_state == "cube_at_target"`, else `"none"` |
| `scripted_demo_to_intent(evidence)` | Always returns `goal_state="cube_at_target"` (the under-specification). `object_motion` derived from 2D trajectory dominant axis (translate_+x / etc.). Other factors fixed to the cubeA_on_cubeB-compatible defaults except `goal_state`. |
| `compile_skill(intent, scene)` | `compile_intent_to_stack_skill(intent, scene)` |

`scripted_demo_to_intent` rationale: the demo's `object_trajectory` is 2D
`((cubeA_initial_xy), (cubeB.xy))`. The 3D pick-and-place is invisible to
the 2D summarizer. This is the Stage-0 controlled information loss; a
future Stage-1 summarizer with 3D / video input would see the place.

## 8. `babysteps/skills/stack.py` — `StackSkill` + `compile_intent_to_stack_skill`

```python
# Constants
CUBE_HALF_SIZE: float = 0.02            # matches ManiSkill StackCube-v1
DESCEND_CLEARANCE_M: float = 0.02       # gap above target z before close
PLACE_CLEARANCE_M: float = 0.005        # gap above cubeB top before release

@dataclass(frozen=True)
class StackSkill:
    """A compiled stack-and-place trajectory.

    waypoints is (N, 7) where N=4 for cube_at_target and N=5 for
    cubeA_on_cubeB. Columns are [x, y, z, qx, qy, qz, qw]."""
    waypoints: np.ndarray
    cubeA_z: float
    cubeB_top_z: float
    goal_state: str


def compile_intent_to_stack_skill(intent, scene):
    if intent.goal_state == "cube_at_target":
        return StackSkill(
            waypoints=_build_translate_waypoints(scene),
            cubeA_z=scene.cube_z,
            cubeB_top_z=scene.extra["cubeB_top_z"],
            goal_state="cube_at_target",
        )
    if intent.goal_state == "cubeA_on_cubeB":
        return StackSkill(
            waypoints=_build_place_on_waypoints(scene),
            cubeA_z=scene.cube_z,
            cubeB_top_z=scene.extra["cubeB_top_z"],
            goal_state="cubeA_on_cubeB",
        )
    raise ValueError(
        f"compile_intent_to_stack_skill: goal_state must be one of "
        f"{{cube_at_target, cubeA_on_cubeB}}, got {intent.goal_state!r}"
    )
```

Waypoint shapes (all use scene.tcp_start_pose for quaternion in last 4
columns):

**`_build_translate_waypoints` (4 rows, cube_at_target):**
0. approach_high: `(cubeA.xy, travel_z)`
1. descend: `(cubeA.xy, cubeA_z + DESCEND_CLEARANCE_M)`
2. grasp: `(cubeA.xy, cubeA_z)`
3. translate_release: `(cubeB.xy, cubeA_z + DESCEND_CLEARANCE_M)`

→ Cube grasped, lifted only slightly, carried to cubeB.xy, released at
low z. Physics: cubeA collides with cubeB and either bounces aside or
lands directly on it (rare). Either way `is_cubeA_on_cubeB ∧
is_cubeA_static ∧ ¬is_cubeA_grasped` is unreliable — usually `False`.

**`_build_place_on_waypoints` (5 rows, cubeA_on_cubeB):**
0. approach_high: `(cubeA.xy, travel_z)`
1. descend: `(cubeA.xy, cubeA_z + DESCEND_CLEARANCE_M)`
2. grasp: `(cubeA.xy, cubeA_z)`
3. lift_above_cubeB: `(cubeB.xy, travel_z)`
4. place_on: `(cubeB.xy, cubeB_top_z + CUBE_HALF_SIZE + PLACE_CLEARANCE_M)`

→ Cube lifted high over cubeB, descended just above cubeB's top,
released. Settles on top.

## 9. `babysteps/envs/stackcube_runner.py` — `StackCubeEnvRunner`

Mirrors `PickCubeEnvRunner` structurally. Differences:

- 5-phase loop for cubeA_on_cubeB, 4-phase for cube_at_target. Phase
  count derived from `skill.waypoints.shape[0]`.
- Gripper schedule:
  - `cube_at_target` (4 phases): `[OPEN, OPEN, CLOSED, OPEN]` — release
    as part of the translate-release phase.
  - `cubeA_on_cubeB` (5 phases): `[OPEN, OPEN, CLOSED, CLOSED, OPEN]` —
    keep grasp through the lift, release only at place_on.
- `reset(seed)` populates `SceneState`:
  - `cube_xy = cubeA.xy`, `cube_z = cubeA.z`
  - `goal_xy = cubeB.xy` (convenience: existing scene-reading callers
    work; the "goal" in StackCube is cubeB's xy)
  - `extra = {"cubeB_xy": cubeB.xy, "cubeB_z": cubeB.z, "cubeB_top_z":
    cubeB.z + 2 * CUBE_HALF_SIZE}`
  - `blocked_sides = ()` always
- `AttemptResult.success` taken directly from `info["success"]` (which
  StackCube-v1 produces from the `cubeA_on_cubeB ∧ static ∧ ¬grasped`
  evaluation — see §10 of this spec).
- `object_moved = ||final_cubeA_xy - initial_cubeA_xy|| > 0.005`.
- `reached_contact = True` once the gripper has reached the cubeA
  position (heuristic: gripper xy within 0.04m and z within 0.04m of
  cubeA at any time during phases 1-2).
- `grasp_slip = False, planner_failed = False, collision = False` —
  none of these are the controlled failure mechanism for StackCube.

The PD-control + waypoint-following machinery from `PickCubeEnvRunner`
is reused near-verbatim (`_to_np`, `_raw_to_xyzw`, `_read_obs` adapted
for `cubeA_pose`/`cubeB_pose`, `_prop_action`).

## 10. ManiSkill StackCube-v1 facts (verified by reading
`mani_skill/envs/tasks/tabletop/stack_cube.py` in `~/.conda/envs/handover/lib/python3.10/site-packages/`)

- `obs["extra"]` keys: `tcp_pose`, `cubeA_pose`, `cubeB_pose`,
  `tcp_to_cubeA_pos`, `tcp_to_cubeB_pos`, `cubeA_to_cubeB_pos`.
- Cube half-size: 0.02 (4 cm cubes).
- Success criterion (`evaluate`):
  - xy_flag: `||cubeA_xy - cubeB_xy|| <= sqrt(0.02² + 0.02²) + 0.005 ≈ 0.0333`
  - z_flag: `|cubeA_z - cubeB_z - 0.04| <= 0.005` (cubeA center one cube-width above cubeB center)
  - is_cubeA_static: linear velocity < 1e-2
  - is_cubeA_grasped = False (gripper released)
  - `success = xy_flag ∧ z_flag ∧ static ∧ ¬grasped`
- **No `goal_pos` key** (unlike PushCube/PickCube). The "goal" is
  implicit (cubeB's pose + cube width).

## 11. `FakeStackCubeEnvRunner` (in `tests/conftest.py`)

Deterministic sim-free runner. Pattern mirrors `FakePickEnvRunner`.

```python
class FakeStackCubeEnvRunner:
    """Deterministic, sim-free env_runner for StackCube unit tests.

    Stage-0 controlled-failure mechanism: outcome is keyed entirely off
    intent.goal_state:
      - cubeA_on_cubeB → success=True, final_obj_xy=cubeB_xy
      - cube_at_target (or anything else) → success=False,
        final_obj_xy=cubeB_xy + (0.025, 0)  (cubeA slid off after collision)
    """
    def reset(self, seed):
        rng = np.random.default_rng(seed)
        r = float(rng.uniform(0.05, 0.12))
        theta = (seed % 4) * (np.pi / 2)
        cubeB_xy = (r * np.cos(theta), r * np.sin(theta))
        cubeB_z = 0.02
        return SceneState(
            cube_xy=(0.0, 0.0),
            cube_z=0.02,
            goal_xy=cubeB_xy,
            tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0),
            blocked_sides=(),
            extra={
                "cubeB_xy": cubeB_xy,
                "cubeB_z": cubeB_z,
                "cubeB_top_z": cubeB_z + 0.04,
            },
        )

    def run(self, intent, scene):
        skill = compile_intent_to_stack_skill(intent, scene)
        assert skill is not None
        cubeA_init = scene.cube_xy
        cubeB_xy = scene.extra["cubeB_xy"]
        if intent.goal_state == "cubeA_on_cubeB":
            final_xy = cubeB_xy
            success = True
        else:
            final_xy = (cubeB_xy[0] + 0.025, cubeB_xy[1])  # slid off
            success = False
        synthetic_traj = tuple(
            (cubeA_init[0] + (final_xy[0] - cubeA_init[0]) * t,
             cubeA_init[1] + (final_xy[1] - cubeA_init[1]) * t)
            for t in np.linspace(0.0, 1.0, 8)
        )
        return AttemptResult(
            initial_obj_xy=cubeA_init, final_obj_xy=final_xy,
            goal_xy=scene.goal_xy,
            reached_contact=True, object_moved=True,
            planner_failed=False, collision=False, grasp_slip=False,
            rollout_log_path=None, success=success,
            trajectory_xy=synthetic_traj,
        )
```

A `fake_stack_env_runner` pytest fixture is added alongside the other
two fake fixtures.

## 12. Render module `babysteps/render/stackcube.py`

`render_episode(env, adapter, seed, fps) -> (frames_dict, titles_dict)`.

Same contract as the existing `pushcube.py` / `pickcube.py`.

- Phase 1 (`"demo"`): oracle intent (cubeA_on_cubeB) executed. Captures
  the full pick-lift-place trajectory.
- Phase 2 (`"attempt_blocked"`): scripted intent (cube_at_target)
  actually stepped. Cube ends up scattered next to cubeB. Tail-padded by
  `fps` frames so the failed state is visible.
- Phase 3 (`"retry"`): revised intent (cubeA_on_cubeB) executed.
  Successful stack.

All three phases step the env (matching PickCube; unlike PushCube's
phase-2 freeze). Phase key `"attempt_blocked"` is kept for consistency
with the other render modules — a comment notes the name is historical
from B/A's blocked-approach narrative; for StackCube, the failure is
under-specification, not blocking.

Imports `STACKCUBE_MAX_CONTROL_STEPS` (new constant in
`babysteps/render/common.py`, value 400 — matches PickCube's runner
cap, sufficient for the 5-phase trajectory).

Title strings mention `goal_state` (demo subtitle) and
`goal_refinement` (retry subtitle).

## 13. CLI integration

One-row additions to two existing registries (no new dispatcher code):

```python
# babysteps/envs/task_registry.py
def _stackcube_entry() -> TaskEntry:
    from babysteps.envs.stackcube_adapter import StackCubeAdapter
    def _make_fake() -> EnvRunner:
        from tests.conftest import FakeStackCubeEnvRunner
        return FakeStackCubeEnvRunner()
    return TaskEntry(
        adapter_cls=StackCubeAdapter,
        fake_runner_factory=_make_fake,
        episode_id_prefix="stackcube_underspec_goal",
    )

TASK_REGISTRY["StackCube-v1"] = _stackcube_entry()

# babysteps/render/__init__.py
def _stackcube_render() -> RenderEpisodeFn:
    from babysteps.render.stackcube import render_episode
    return render_episode

RENDER_REGISTRY["StackCube-v1"] = _stackcube_render
```

`scripts/stage0_collect.py`, `scripts/stage0_summarize.py`,
`scripts/render_stage0_maniskill.py` need no code changes — they all
already dispatch through the registries via `--task` argparse
`choices=sorted(TASK_REGISTRY.keys())`.

`tests/test_task_registry.py::test_task_registry_matches_render_registry`
will continue to pass once both registry entries land.

## 14. Test plan

Net new tests: ~37 (taking total from 184 to ~221).

| File | New tests |
|---|---|
| `tests/test_schemas.py` | +4 (new whitelist tokens) |
| `tests/test_revision.py` | +3 (goal_refinement happy path; NotImplementedError for unknown source; factor-frozen invariant) |
| `tests/test_failure.py` | 0 (no new predicate) |
| `tests/test_stack_skill.py` (NEW) | ~10 (per-goal_state waypoint geometry, defensive checks, cube/goal positioning, raises on unknown goal_state) |
| `tests/test_stackcube_adapter.py` (NEW) | ~15 (parity tests mirroring `test_pickcube_adapter.py`; scripted_demo_to_intent always under-specifies; default_blocked_factory returns (); oracle_wrong_factor logic; snapshot) |
| `tests/test_pickcube_delta_pp.py` | +1 (StackCube fake-env delta_pp >= 10) |
| `tests/test_stage0_collect_cli.py` | +1 row in existing parametrize (StackCube-v1 + snapshot) |
| `tests/test_render_modules.py` | +3 (StackCube render: keys + phase-2-actually-steps-env + titles mention goal_state/refinement) |
| `tests/test_task_registry.py` | +1 (`get_task_entry("StackCube-v1")` matches StackCubeAdapter) |
| `tests/snapshots/stackcube_samples_seeds_0_4.jsonl` (NEW) | — (5 records, byte-stable) |

Both pre-existing snapshots (PushCube + PickCube) must remain
byte-identical.

## 15. Risks & open questions

1. **Real-physics outcome of cube_at_target attempt is non-deterministic.**
   Releasing cubeA at low z near cubeB causes a collision; physics may
   land cubeA on top (rare success), beside cubeB (typical), or off the
   table (occasional). The success criterion correctly reports False in
   all of these. The MP4 may look chaotic. Mitigation: tail-pad phase 2;
   expect "scattered next to cubeB" as the canonical visual.
2. **`object_motion = "place_on"` is decorative for Stage-0.** The skill
   compiler dispatches on `goal_state`, not `object_motion`. The oracle
   uses `place_on`; the scripted summarizer uses `translate_<axis>`.
   After goal_refinement, `revised_intent.object_motion` stays as
   `translate_<axis>` (frozen). This is consistent with factor-local
   revision but slightly inelegant. Stage-1+ may introduce a paired
   operator that revises both.
3. **Cube-half-size hardcoded twice.** `babysteps/skills/stack.py`
   defines `CUBE_HALF_SIZE = 0.02`; ManiSkill StackCube defines its own
   `self.cube_half_size`. Stage-0 takes the constant on faith. If
   ManiSkill ever changes the StackCube cube size, the skill geometry
   needs an update. Mitigation: an integration test (skipped on
   non-Vulkan machines) could read the env's half-size and assert
   equality. Out of scope for this plan.
4. **`is_cubeA_static` requirement adds variance.** ManiSkill's success
   check requires cubeA's linear velocity < 1e-2. After release, cubeA
   may wobble briefly. The render loop should step a few extra frames
   after the final waypoint to let the physics settle. Mitigation: tail
   padding on phase 3 too (not just phase 2).

## 16. Plan file

This spec drives one plan file (created next via writing-plans):

- `docs/superpowers/plans/2026-05-17-stage0-stackcube-c-plan.md`

## 17. Summary

Sub-project C lands StackCube-v1 as the third adapter in the Stage-0
loop, demonstrating the third distinct failure mode (`goal_state`
under-specification) and the third revision operator (`goal_refinement`,
a strict-extension goal sharpener). The dispatch registries and CLI
scripts receive one-row additions; the bulk of the work is the
StackCubeAdapter + StackCubeEnvRunner + StackSkill + render module +
the snapshot-stable test suite.
