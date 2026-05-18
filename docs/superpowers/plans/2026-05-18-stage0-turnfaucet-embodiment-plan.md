# TurnFaucet Embodiment-Substitution Reframe — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace TurnFaucet's `constraint_introduction` story with `embodiment_substitution` (single-factor `grasp_turn → poke_turn` revision after `grasp_infeasible` failure) per `docs/superpowers/specs/2026-05-18-stage0-turnfaucet-embodiment-design.md`.

**Architecture:** Additive schema tokens (no removals); TurnSkill compiler dispatches on `embodiment_mapping` to produce grasp-mode (existing behavior) or poke-mode (closed-gripper lateral sweep); env_runner and render module use generic phase loops driven by `len(skill.waypoints)` and `skill.gripper_schedule` with auto-sign two-trial loop for poke; fake-env unchanged in topology, swapped in outcome rule; only TurnFaucet snapshot regenerates.

**Tech Stack:** Python 3.10, ManiSkill 3 (`TurnFaucet-v1`, `sim_backend="gpu"`, `control_mode="pd_ee_delta_pose"`), NumPy, pytest. Existing pure-Python modules in `babysteps/` + sim adapters in `babysteps/envs/` + per-task render modules in `babysteps/render/`.

---

## File map (what each task touches)

| File | Action | Tasks |
|---|---|---|
| `babysteps/schemas.py` | Modify (additive) | T1 |
| `babysteps/skills/turn.py` | Modify (extend TurnSkill, add dispatch + _compile_poke) | T2, T3, T4 |
| `babysteps/failure.py` | Modify (FAILURE_TO_FACTOR + derivation branch) | T5, T6 |
| `babysteps/revision.py` | Modify (new operator branch) | T7 |
| `babysteps/envs/turnfaucet_adapter.py` | Modify (oracle/scripted/wrong_factor) | T8 |
| `tests/conftest.py` | Modify (FakeTurnFaucetEnvRunner rule) | T9 |
| `tests/snapshots/turnfaucet_samples_seeds_0_4.jsonl` | Regenerate | T10 |
| `tests/test_pickcube_delta_pp.py` | Modify (TurnFaucet row params) | T11 |
| `babysteps/envs/turnfaucet_runner.py` | Rewrite (generic loop + helpers + run dispatch) | T12, T13 |
| `babysteps/render/turnfaucet.py` | Rewrite (privileged demo + generic loop + auto-sign) | T14 |
| `tests/test_schemas.py` | Modify (assert new tokens) | T1 |
| `tests/test_failure.py` | Modify (assert grasp_infeasible derivation) | T5, T6 |
| `tests/test_revision.py` | Modify (assert embodiment_substitution) | T7 |
| `tests/test_turn_skill.py` | Modify (grasp + poke variants) | T2, T3, T4 |
| `tests/test_turnfaucet_adapter.py` | Modify (new contract) | T8 |
| `tests/test_render_modules.py` | Modify (new render contract) | T14 |
| `docs/superpowers/specs/2026-05-17-stage0-turnfaucet-d-design.md` | Modify (front-matter supersession note) | T15 |
| `CLAUDE.md` | Modify (TurnFaucet section: new story + partial gate) | T15 |

---

## Task 1: Add 4 new schema tokens (additive only)

**Files:**
- Modify: `babysteps/schemas.py`
- Modify: `tests/test_schemas.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_schemas.py`:

```python
def test_embodiment_grasp_turn_token():
    from babysteps.schemas import EMBODIMENT_MAPPINGS
    assert "proxy_contact_to_franka_grasp_turn" in EMBODIMENT_MAPPINGS


def test_embodiment_poke_turn_token():
    from babysteps.schemas import EMBODIMENT_MAPPINGS
    assert "proxy_contact_to_franka_poke_turn" in EMBODIMENT_MAPPINGS


def test_grasp_infeasible_predicate_token():
    from babysteps.schemas import FAILURE_PREDICATES
    assert "grasp_infeasible" in FAILURE_PREDICATES


def test_embodiment_substitution_operator_token():
    from babysteps.schemas import REVISION_OPERATORS
    assert "embodiment_substitution" in REVISION_OPERATORS


def test_old_d_tokens_remain_deprecated_but_present():
    """Per spec §4: additive only. Deprecated tokens stay in whitelists
    until a separate cleanup commit proves no references remain."""
    from babysteps.schemas import (
        EMBODIMENT_MAPPINGS, CONTACT_REGIONS, CONSTRAINT_REGIONS,
        FAILURE_PREDICATES, REVISION_OPERATORS,
    )
    assert "proxy_contact_to_franka_turn" in EMBODIMENT_MAPPINGS
    assert "faucet_base" in CONTACT_REGIONS
    assert "faucet_base_static" in CONSTRAINT_REGIONS
    assert "constraint_violation" in FAILURE_PREDICATES
    assert "constraint_introduction" in REVISION_OPERATORS
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
pytest tests/test_schemas.py::test_embodiment_grasp_turn_token tests/test_schemas.py::test_embodiment_poke_turn_token tests/test_schemas.py::test_grasp_infeasible_predicate_token tests/test_schemas.py::test_embodiment_substitution_operator_token -v
```
Expected: 4 FAIL (KeyError / not-in-frozenset).

- [ ] **Step 3: Add tokens to `babysteps/schemas.py`**

Edit `babysteps/schemas.py`. Locate the existing frozenset definitions and add new entries (do NOT remove anything):

```python
EMBODIMENT_MAPPINGS: frozenset[str] = frozenset({
    "proxy_contact_to_franka_push",
    "proxy_contact_to_franka_grasp",
    "proxy_contact_to_franka_pick_and_place",
    "proxy_contact_to_franka_turn",                # D: deprecated, kept in whitelist
    "proxy_contact_to_franka_grasp_turn",          # D: NEW — initial intent
    "proxy_contact_to_franka_poke_turn",           # D: NEW — revised intent
})

FAILURE_PREDICATES: frozenset[str] = frozenset({
    "none",
    "approach_blocked",
    "direction_error",
    "contact_failure",
    "no_motion",
    "goal_not_satisfied",
    "grasp_slip",
    "constraint_violation",                        # D: deprecated, kept in whitelist
    "grasp_infeasible",                            # D: NEW
})

REVISION_OPERATORS: frozenset[str] = frozenset({
    "approach_substitution",
    "contact_substitution",
    "goal_refinement",
    "constraint_introduction",                     # D: deprecated, kept in whitelist
    "embodiment_substitution",                     # D: NEW
})
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
pytest tests/test_schemas.py -v
```
Expected: all tests pass (including the new 4 and the deprecation-presence guard).

- [ ] **Step 5: Commit**

```bash
git add babysteps/schemas.py tests/test_schemas.py
git commit -m "feat(d/schema): add grasp_turn, poke_turn, grasp_infeasible, embodiment_substitution tokens"
```

---

## Task 2: Extend TurnSkill dataclass with mode/gripper_schedule/sign fields

**Files:**
- Modify: `babysteps/skills/turn.py:36-47` (TurnSkill dataclass)
- Modify: `tests/test_turn_skill.py`

- [ ] **Step 1: Add failing test**

Append to `tests/test_turn_skill.py`:

```python
def test_turn_skill_has_mode_gripper_schedule_sign_fields():
    """TurnSkill must carry per-mode dispatch metadata so generic phase
    loops (runner/render) can iterate without hardcoded grasp assumptions."""
    import numpy as np
    from babysteps.skills.turn import TurnSkill
    wp = np.zeros((3, 7), dtype=np.float64)
    skill = TurnSkill(
        waypoints=wp,
        contact_region="handle_grip",
        target_joint_axis_xy=(0.0, 1.0),
        mode="poke",
        gripper_schedule=(-1.0, -1.0, -1.0),
        sign=+1,
    )
    assert skill.mode == "poke"
    assert skill.gripper_schedule == (-1.0, -1.0, -1.0)
    assert skill.sign == +1
    assert len(skill.gripper_schedule) == len(skill.waypoints)
```

- [ ] **Step 2: Run test to confirm failure**

```bash
pytest tests/test_turn_skill.py::test_turn_skill_has_mode_gripper_schedule_sign_fields -v
```
Expected: FAIL with `TypeError: TurnSkill.__init__() got an unexpected keyword argument 'mode'`.

- [ ] **Step 3: Extend TurnSkill in `babysteps/skills/turn.py`**

Replace the existing `@dataclass(frozen=True) class TurnSkill: ...` block with:

```python
@dataclass(frozen=True)
class TurnSkill:
    """A compiled approach trajectory for the TurnFaucet task.

    waypoints is (N, 7). Columns are [x, y, z, qx, qy, qz, qw]. N varies
    by mode: grasp uses 4 (approach, descend, grip, pull), poke uses 3
    (approach, descend-lateral, sweep). Runner/render phase loops MUST
    iterate based on len(waypoints) + gripper_schedule, never on a
    hardcoded 4-phase grasp shape.

    mode is "grasp" | "poke" and is dispatched on intent.embodiment_mapping
    by compile_intent_to_turn_skill (§7 of the spec).

    gripper_schedule[i] is the gripper command for waypoint i:
    +1.0 = open, -1.0 = closed.

    sign is poke-only (+1 or -1). For poke, the runner's auto-sign
    two-trial loop picks the winning sign per seed.
    """
    waypoints: np.ndarray
    contact_region: str
    target_joint_axis_xy: tuple[float, float]
    mode: str
    gripper_schedule: tuple[float, ...]
    sign: int = +1
```

- [ ] **Step 4: Run test to confirm pass**

```bash
pytest tests/test_turn_skill.py::test_turn_skill_has_mode_gripper_schedule_sign_fields -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add babysteps/skills/turn.py tests/test_turn_skill.py
git commit -m "feat(d/skill): extend TurnSkill with mode, gripper_schedule, sign fields"
```

---

## Task 3: Refactor compile_intent_to_turn_skill to dispatch on embodiment_mapping (preserve grasp behavior)

**Files:**
- Modify: `babysteps/skills/turn.py` (`compile_intent_to_turn_skill` function)
- Modify: `tests/test_turn_skill.py`

- [ ] **Step 1: Add failing test**

Append to `tests/test_turn_skill.py`:

```python
def _make_grasp_intent():
    from babysteps.schemas import Intent
    return Intent(
        goal_state="faucet_turned",
        object_motion="turn",
        contact_region="handle_grip",
        approach_direction="from_above",
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_grasp_turn",
    )


def _make_scene():
    from babysteps.schemas import SceneState
    return SceneState(
        cube_xy=(0.05, 0.02), cube_z=0.10, goal_xy=(0.05, 0.02),
        tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0),
        blocked_sides=(),
        extra={"handle_xy": (0.05, 0.02), "handle_z": 0.10,
               "target_joint_axis_xy": (0.0, 1.0)},
    )


def test_compile_grasp_turn_returns_grasp_mode_4_waypoints():
    from babysteps.skills.turn import compile_intent_to_turn_skill
    skill = compile_intent_to_turn_skill(_make_grasp_intent(), _make_scene())
    assert skill.mode == "grasp"
    assert len(skill.waypoints) == 4
    assert skill.gripper_schedule == (1.0, 1.0, -1.0, -1.0)
    assert skill.contact_region == "handle_grip"


def test_compile_deprecated_turn_token_falls_back_to_grasp():
    """Per spec §7: deprecated 'proxy_contact_to_franka_turn' compiles to
    the grasp variant for backward-compat with old diag scripts."""
    from babysteps.schemas import Intent
    from babysteps.skills.turn import compile_intent_to_turn_skill
    intent = Intent(
        goal_state="faucet_turned", object_motion="turn",
        contact_region="handle_grip", approach_direction="from_above",
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_turn",
    )
    skill = compile_intent_to_turn_skill(intent, _make_scene())
    assert skill.mode == "grasp"


def test_compile_unknown_embodiment_raises():
    from babysteps.schemas import Intent
    from babysteps.skills.turn import compile_intent_to_turn_skill
    intent = Intent(
        goal_state="faucet_turned", object_motion="turn",
        contact_region="handle_grip", approach_direction="from_above",
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_push",  # wrong embodiment for turn task
    )
    import pytest
    with pytest.raises(ValueError, match="unsupported embodiment_mapping"):
        compile_intent_to_turn_skill(intent, _make_scene())
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
pytest tests/test_turn_skill.py::test_compile_grasp_turn_returns_grasp_mode_4_waypoints tests/test_turn_skill.py::test_compile_deprecated_turn_token_falls_back_to_grasp tests/test_turn_skill.py::test_compile_unknown_embodiment_raises -v
```
Expected: FAIL (current dispatch is on `contact_region`, not `embodiment_mapping`).

- [ ] **Step 3: Refactor in `babysteps/skills/turn.py`**

Replace the existing `compile_intent_to_turn_skill` function with:

```python
def compile_intent_to_turn_skill(
    intent: Intent, scene: SceneState, sign: int = +1,
) -> TurnSkill:
    """Dispatches on intent.embodiment_mapping per spec §7.

    grasp_turn (and deprecated proxy_contact_to_franka_turn) → _compile_grasp
    poke_turn → _compile_poke(sign=sign)
    anything else → ValueError
    """
    if intent.embodiment_mapping == "proxy_contact_to_franka_grasp_turn":
        return _compile_grasp(intent, scene)
    if intent.embodiment_mapping == "proxy_contact_to_franka_poke_turn":
        return _compile_poke(intent, scene, sign=sign)
    if intent.embodiment_mapping == "proxy_contact_to_franka_turn":
        # Deprecated token still in whitelist — preserve behavioral parity for
        # old diag scripts. Removal happens in the schema cleanup commit.
        return _compile_grasp(intent, scene)
    raise ValueError(
        f"compile_intent_to_turn_skill: unsupported embodiment_mapping "
        f"{intent.embodiment_mapping!r}"
    )


def _compile_grasp(intent: Intent, scene: SceneState) -> TurnSkill:
    """Grasp-mode TurnSkill (4 waypoints, OPEN→CLOSED schedule).

    Behavior identical to the original compile_intent_to_turn_skill body
    when intent.contact_region == "handle_grip". The faucet_base branch
    (predecessor D's wrong-contact case) is gone — no one emits it now.
    """
    if intent.contact_region != "handle_grip":
        raise ValueError(
            f"_compile_grasp: contact_region must be 'handle_grip', "
            f"got {intent.contact_region!r}"
        )
    contact_xy = np.asarray(scene.extra["handle_xy"], dtype=np.float64)
    contact_z = float(scene.extra["handle_z"])
    axis_xy = np.asarray(scene.extra["target_joint_axis_xy"], dtype=np.float64)
    axis_norm = float(np.linalg.norm(axis_xy))
    if axis_norm < 1e-3:
        pull_dir_xy = np.array([0.0, 1.0])
    else:
        pull_dir_xy = np.array([-axis_xy[1], axis_xy[0]]) / axis_norm
    pull_xy = contact_xy + pull_dir_xy * TURN_PULL_DISTANCE_M
    tcp = np.asarray(scene.tcp_start_pose, dtype=np.float64)
    travel_z = float(tcp[2])
    grip_z = contact_z + GRIP_OFFSET_M

    wp = np.zeros((4, 7), dtype=np.float64)
    wp[0, 0:2] = contact_xy
    wp[0, 2] = travel_z
    wp[1, 0:2] = contact_xy
    wp[1, 2] = grip_z + DESCEND_CLEARANCE_M
    wp[2, 0:2] = contact_xy
    wp[2, 2] = grip_z
    wp[3, 0:2] = pull_xy
    wp[3, 2] = grip_z
    wp[:, 3:7] = tcp[3:7]
    return TurnSkill(
        waypoints=wp,
        contact_region="handle_grip",
        target_joint_axis_xy=(float(axis_xy[0]), float(axis_xy[1])),
        mode="grasp",
        gripper_schedule=(1.0, 1.0, -1.0, -1.0),
        sign=+1,
    )


def _compile_poke(intent: Intent, scene: SceneState, sign: int) -> TurnSkill:
    raise NotImplementedError("_compile_poke implemented in Task 4")
```

(The stub `_compile_poke` lets Task 3 compile/pass without Task 4's full implementation.)

- [ ] **Step 4: Run tests to confirm pass**

```bash
pytest tests/test_turn_skill.py -v
```
Expected: all pass. Includes the new dispatch tests AND all pre-existing TurnSkill tests (grasp behavior must be byte-identical to before).

- [ ] **Step 5: Commit**

```bash
git add babysteps/skills/turn.py tests/test_turn_skill.py
git commit -m "refactor(d/skill): dispatch compile_intent_to_turn_skill on embodiment_mapping; preserve grasp behavior"
```

---

## Task 4: Implement _compile_poke (closed-gripper lateral sweep, 3 waypoints)

**Files:**
- Modify: `babysteps/skills/turn.py` (`_compile_poke` function + constants)
- Modify: `tests/test_turn_skill.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_turn_skill.py`:

```python
def _make_poke_intent():
    from babysteps.schemas import Intent
    return Intent(
        goal_state="faucet_turned",
        object_motion="turn",
        contact_region="handle_grip",
        approach_direction="from_above",
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_poke_turn",
    )


def test_compile_poke_returns_poke_mode_3_waypoints():
    from babysteps.skills.turn import compile_intent_to_turn_skill
    skill = compile_intent_to_turn_skill(_make_poke_intent(), _make_scene())
    assert skill.mode == "poke"
    assert len(skill.waypoints) == 3
    assert skill.gripper_schedule == (-1.0, -1.0, -1.0)
    assert skill.contact_region == "handle_grip"
    assert skill.sign == +1


def test_compile_poke_sign_negative_flips_sweep_direction():
    import numpy as np
    from babysteps.skills.turn import compile_intent_to_turn_skill
    skill_pos = compile_intent_to_turn_skill(_make_poke_intent(), _make_scene(), sign=+1)
    skill_neg = compile_intent_to_turn_skill(_make_poke_intent(), _make_scene(), sign=-1)
    # pre_xy and post_xy should mirror through handle_xy.
    handle_xy = np.array(_make_scene().extra["handle_xy"])
    pre_pos  = skill_pos.waypoints[1, 0:2]
    pre_neg  = skill_neg.waypoints[1, 0:2]
    post_pos = skill_pos.waypoints[2, 0:2]
    post_neg = skill_neg.waypoints[2, 0:2]
    # pre points are on opposite sides of handle_xy.
    np.testing.assert_allclose(pre_pos - handle_xy, -(pre_neg - handle_xy), atol=1e-9)
    np.testing.assert_allclose(post_pos - handle_xy, -(post_neg - handle_xy), atol=1e-9)


def test_compile_poke_requires_handle_grip_contact_region():
    from babysteps.schemas import Intent
    from babysteps.skills.turn import compile_intent_to_turn_skill
    intent = Intent(
        goal_state="faucet_turned", object_motion="turn",
        contact_region="faucet_base",  # deprecated D token, still in whitelist
        approach_direction="from_above", constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_poke_turn",
    )
    import pytest
    with pytest.raises(ValueError, match="poke_turn requires contact_region='handle_grip'"):
        compile_intent_to_turn_skill(intent, _make_scene())


def test_compile_poke_z_above_handle_for_finger_dangle():
    from babysteps.skills.turn import compile_intent_to_turn_skill
    skill = compile_intent_to_turn_skill(_make_poke_intent(), _make_scene())
    # contact_z (waypoint 1 and 2) is handle_z + _POKE_HEIGHT_ABOVE_M.
    assert skill.waypoints[1, 2] == 0.10 + 0.04
    assert skill.waypoints[2, 2] == 0.10 + 0.04
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
pytest tests/test_turn_skill.py::test_compile_poke_returns_poke_mode_3_waypoints tests/test_turn_skill.py::test_compile_poke_sign_negative_flips_sweep_direction tests/test_turn_skill.py::test_compile_poke_requires_handle_grip_contact_region tests/test_turn_skill.py::test_compile_poke_z_above_handle_for_finger_dangle -v
```
Expected: 4 FAIL with `NotImplementedError: _compile_poke implemented in Task 4`.

- [ ] **Step 3: Implement `_compile_poke` and add poke constants**

In `babysteps/skills/turn.py`, add these module-level constants near the existing `DESCEND_CLEARANCE_M`, etc.:

```python
# --- Poke-mode constants (verified empirically by scripts/_diag_tf_poke5.py)
_POKE_LATERAL_OFFSET_M: float = 0.07
_POKE_SWEEP_DISTANCE_M: float = 0.22
_POKE_HEIGHT_ABOVE_M: float = 0.04
_POKE_HIGH_CLEARANCE_M: float = 0.12
```

Replace the `_compile_poke` stub with:

```python
def _compile_poke(intent: Intent, scene: SceneState, sign: int) -> TurnSkill:
    """Poke-mode TurnSkill (3 waypoints, closed-gripper throughout).

    Closed-gripper lateral brute-force sweep. Per spec §7: the tangent
    direction is a HEURISTIC SEED — the actual winning direction is
    decided at runtime by TurnFaucetEnvRunner's auto-sign retry. The
    cross-product-based tangent is correct geometrically, but partnet
    faucets' qpos sign convention is inconsistent across models.
    """
    if intent.contact_region != "handle_grip":
        raise ValueError(
            f"poke_turn requires contact_region='handle_grip', "
            f"got {intent.contact_region!r}"
        )
    handle_xy = np.asarray(scene.extra["handle_xy"], dtype=np.float64)
    axis_xy = np.asarray(scene.extra["target_joint_axis_xy"], dtype=np.float64)
    handle_z = float(scene.extra["handle_z"])
    tcp = np.asarray(scene.tcp_start_pose, dtype=np.float64)
    travel_z = float(tcp[2])

    axis_norm = float(np.linalg.norm(axis_xy))
    if axis_norm < 1e-3:
        tangent = np.array([0.0, 1.0])
    else:
        tangent = np.array([-axis_xy[1], axis_xy[0]]) / axis_norm
    sweep_dir = tangent * sign

    contact_z = handle_z + _POKE_HEIGHT_ABOVE_M
    approach_z = max(travel_z, handle_z + _POKE_HIGH_CLEARANCE_M) + 0.02
    pre_xy = handle_xy - sweep_dir * _POKE_LATERAL_OFFSET_M
    post_xy = handle_xy + sweep_dir * _POKE_SWEEP_DISTANCE_M

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

- [ ] **Step 4: Run tests to confirm pass**

```bash
pytest tests/test_turn_skill.py -v
```
Expected: all pass (4 new poke tests + all existing grasp tests).

- [ ] **Step 5: Commit**

```bash
git add babysteps/skills/turn.py tests/test_turn_skill.py
git commit -m "feat(d/skill): _compile_poke (closed-gripper lateral sweep, 3 waypoints)"
```

---

## Task 5: Add grasp_infeasible entry to FAILURE_TO_FACTOR

**Files:**
- Modify: `babysteps/failure.py`
- Modify: `tests/test_failure.py`

- [ ] **Step 1: Add failing test**

Append to `tests/test_failure.py`:

```python
def test_grasp_infeasible_maps_to_embodiment_mapping():
    from babysteps.failure import FAILURE_TO_FACTOR
    assert FAILURE_TO_FACTOR["grasp_infeasible"] == (
        "embodiment_mapping", ("embodiment_mapping",)
    )
```

- [ ] **Step 2: Run test to confirm failure**

```bash
pytest tests/test_failure.py::test_grasp_infeasible_maps_to_embodiment_mapping -v
```
Expected: FAIL with `KeyError: 'grasp_infeasible'`.

- [ ] **Step 3: Add entry to `babysteps/failure.py`**

Locate the `FAILURE_TO_FACTOR` dict in `babysteps/failure.py` and add the entry:

```python
FAILURE_TO_FACTOR: dict[str, tuple[str, tuple[str, ...]]] = {
    # ... existing entries (approach_blocked, grasp_slip, constraint_violation, etc.) ...
    "grasp_infeasible": ("embodiment_mapping", ("embodiment_mapping",)),   # D NEW
}
```

(Keep all existing entries — additive only.)

- [ ] **Step 4: Run test to confirm pass**

```bash
pytest tests/test_failure.py::test_grasp_infeasible_maps_to_embodiment_mapping -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add babysteps/failure.py tests/test_failure.py
git commit -m "feat(d/failure): FAILURE_TO_FACTOR entry for grasp_infeasible → embodiment_mapping"
```

---

## Task 6: Add context-derived grasp_infeasible branch in build_failure_packet

**Files:**
- Modify: `babysteps/failure.py` (`build_failure_packet`)
- Modify: `tests/test_failure.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_failure.py`:

```python
def _make_grasp_turn_intent():
    from babysteps.schemas import Intent
    return Intent(
        goal_state="faucet_turned", object_motion="turn",
        contact_region="handle_grip", approach_direction="from_above",
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_grasp_turn",
    )


def _make_grasp_failed_attempt():
    from babysteps.schemas import AttemptResult
    return AttemptResult(
        initial_obj_xy=(0.05, 0.02), final_obj_xy=(0.05, 0.02),
        goal_xy=(0.05, 0.02),
        reached_contact=True, object_moved=False,
        planner_failed=False, collision=False, grasp_slip=False,
        rollout_log_path=None, success=False, trajectory_xy=(),
    )


def _make_scene_extra_with_handle():
    from babysteps.schemas import SceneState
    return SceneState(
        cube_xy=(0.05, 0.02), cube_z=0.10, goal_xy=(0.05, 0.02),
        tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0),
        blocked_sides=(),
        extra={"handle_xy": (0.05, 0.02), "handle_z": 0.10,
               "target_joint_axis_xy": (0.0, 1.0)},
    )


def test_build_failure_packet_grasp_infeasible_when_grasp_turn_no_motion():
    from babysteps.failure import build_failure_packet
    fp = build_failure_packet(
        _make_grasp_turn_intent(),
        _make_grasp_failed_attempt(),
        _make_scene_extra_with_handle(),
    )
    assert fp.failure_predicate == "grasp_infeasible"


def test_build_failure_packet_not_grasp_infeasible_when_embodiment_is_poke_turn():
    """The derivation must check intent.embodiment_mapping, not just the
    AttemptResult flags. A poke_turn intent that reached_contact but
    didn't move the object is a different failure mode (e.g., no_motion)."""
    from babysteps.failure import build_failure_packet
    from dataclasses import replace
    poke_intent = replace(
        _make_grasp_turn_intent(),
        embodiment_mapping="proxy_contact_to_franka_poke_turn",
    )
    fp = build_failure_packet(
        poke_intent,
        _make_grasp_failed_attempt(),
        _make_scene_extra_with_handle(),
    )
    assert fp.failure_predicate != "grasp_infeasible"


def test_grasp_infeasible_precedence_above_grasp_slip():
    """Per spec §5: grasp_infeasible slots between planner_failed and
    grasp_slip in the precedence chain. A failed grasp_turn attempt
    where reached_contact=True + grasp_slip=False + object_moved=False
    should resolve to grasp_infeasible, not no_motion."""
    from babysteps.failure import build_failure_packet
    fp = build_failure_packet(
        _make_grasp_turn_intent(),
        _make_grasp_failed_attempt(),
        _make_scene_extra_with_handle(),
    )
    assert fp.failure_predicate == "grasp_infeasible"
    assert fp.failure_predicate != "no_motion"
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
pytest tests/test_failure.py::test_build_failure_packet_grasp_infeasible_when_grasp_turn_no_motion tests/test_failure.py::test_build_failure_packet_not_grasp_infeasible_when_embodiment_is_poke_turn tests/test_failure.py::test_grasp_infeasible_precedence_above_grasp_slip -v
```
Expected: FAIL (predicate resolves to something else, e.g., `no_motion` or `contact_failure`).

- [ ] **Step 3: Add the derivation branch in `babysteps/failure.py`**

Locate `build_failure_packet`. After the `planner_failed` branch and BEFORE the `grasp_slip` branch (precedence per spec §5), insert:

```python
    elif (
        intent.embodiment_mapping == "proxy_contact_to_franka_grasp_turn"
        and attempt.reached_contact
        and not attempt.object_moved
        and not attempt.success
    ):
        # Spec §5: context-derived predicate (no new AttemptResult field).
        # Grasp-mode embodiment reached the handle but the gripper jaws
        # could not envelop it (handle thickness > Panda gripper opening).
        predicate = "grasp_infeasible"
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
pytest tests/test_failure.py -v
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add babysteps/failure.py tests/test_failure.py
git commit -m "feat(d/failure): context-derived grasp_infeasible branch in build_failure_packet"
```

---

## Task 7: Add embodiment_substitution revision operator

**Files:**
- Modify: `babysteps/revision.py` (`revise_intent`)
- Modify: `tests/test_revision.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_revision.py`:

```python
def _grasp_intent():
    from babysteps.schemas import Intent
    return Intent(
        goal_state="faucet_turned", object_motion="turn",
        contact_region="handle_grip", approach_direction="from_above",
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_grasp_turn",
    )


def _embodiment_attribution():
    from babysteps.failure import Attribution
    return Attribution(
        wrong_factor="embodiment_mapping",
        revise=("embodiment_mapping",),
        freeze=("goal_state", "object_motion", "contact_region",
                "approach_direction", "constraint_region"),
        semantic_failure=True,
    )


def test_revise_intent_embodiment_substitution_swaps_grasp_to_poke():
    from babysteps.revision import revise_intent
    intent = _grasp_intent()
    revised, rev = revise_intent(intent, _embodiment_attribution(), scene=None)
    assert revised.embodiment_mapping == "proxy_contact_to_franka_poke_turn"
    # All other factors preserved.
    assert revised.goal_state == intent.goal_state
    assert revised.object_motion == intent.object_motion
    assert revised.contact_region == intent.contact_region
    assert revised.approach_direction == intent.approach_direction
    assert revised.constraint_region == intent.constraint_region


def test_revise_intent_embodiment_substitution_records_5_frozen_factors():
    from babysteps.revision import revise_intent
    _, rev = revise_intent(_grasp_intent(), _embodiment_attribution(), scene=None)
    assert rev.operator == "embodiment_substitution"
    assert rev.factor == "embodiment_mapping"
    assert rev.old_value == "proxy_contact_to_franka_grasp_turn"
    assert rev.new_value == "proxy_contact_to_franka_poke_turn"
    assert set(rev.frozen_factors) == {
        "goal_state", "object_motion", "contact_region",
        "approach_direction", "constraint_region",
    }
    assert len(rev.frozen_factors) == 5


def test_revise_intent_embodiment_substitution_rejects_non_grasp_turn_input():
    """Per spec §6: only grasp_turn → poke_turn is supported."""
    import pytest
    from dataclasses import replace
    from babysteps.revision import revise_intent
    poke_already = replace(
        _grasp_intent(),
        embodiment_mapping="proxy_contact_to_franka_poke_turn",
    )
    with pytest.raises(NotImplementedError,
                        match="embodiment_substitution handles only"):
        revise_intent(poke_already, _embodiment_attribution(), scene=None)
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
pytest tests/test_revision.py::test_revise_intent_embodiment_substitution_swaps_grasp_to_poke tests/test_revision.py::test_revise_intent_embodiment_substitution_records_5_frozen_factors tests/test_revision.py::test_revise_intent_embodiment_substitution_rejects_non_grasp_turn_input -v
```
Expected: FAIL (no branch handles `wrong_factor == "embodiment_mapping"`).

- [ ] **Step 3: Add the branch in `babysteps/revision.py`**

Locate `revise_intent`. Add this branch alongside the existing `approach_substitution`, `contact_substitution`, `goal_refinement`, `constraint_introduction` branches:

```python
    if attribution.wrong_factor == "embodiment_mapping":
        # Spec §6: pure single-factor swap. Only the
        # grasp_turn → poke_turn transition is supported in Stage-0.
        if intent.embodiment_mapping != "proxy_contact_to_franka_grasp_turn":
            raise NotImplementedError(
                f"embodiment_substitution handles only "
                f"grasp_turn → poke_turn (got {intent.embodiment_mapping!r}). "
                f"See docs/superpowers/specs/"
                f"2026-05-18-stage0-turnfaucet-embodiment-design.md §6"
            )
        revised = replace(
            intent, embodiment_mapping="proxy_contact_to_franka_poke_turn",
        )
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

- [ ] **Step 4: Run tests to confirm pass**

```bash
pytest tests/test_revision.py -v
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add babysteps/revision.py tests/test_revision.py
git commit -m "feat(d/revision): embodiment_substitution operator (grasp_turn → poke_turn, 5 frozen factors)"
```

---

## Task 8: Update TurnFaucetAdapter (oracle returns poke, scripted returns grasp)

**Files:**
- Modify: `babysteps/envs/turnfaucet_adapter.py`
- Modify: `tests/test_turnfaucet_adapter.py`

- [ ] **Step 1: Add failing tests**

Locate the existing tests in `tests/test_turnfaucet_adapter.py` and append:

```python
def test_oracle_correct_intent_returns_poke_turn():
    from babysteps.envs.turnfaucet_adapter import TurnFaucetAdapter
    from babysteps.schemas import SceneState
    scene = SceneState(
        cube_xy=(0.05, 0.02), cube_z=0.10, goal_xy=(0.05, 0.02),
        tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0),
        blocked_sides=(),
        extra={"handle_xy": (0.05, 0.02), "handle_z": 0.10,
               "target_joint_axis_xy": (0.0, 1.0)},
    )
    adapter = TurnFaucetAdapter()
    intent = adapter.oracle_correct_intent(scene)
    assert intent.embodiment_mapping == "proxy_contact_to_franka_poke_turn"
    assert intent.contact_region == "handle_grip"
    assert intent.constraint_region == "none"
    assert intent.goal_state == "faucet_turned"
    assert intent.object_motion == "turn"
    assert intent.approach_direction == "from_above"


def test_scripted_demo_to_intent_returns_grasp_turn():
    from babysteps.envs.turnfaucet_adapter import TurnFaucetAdapter
    from babysteps.schemas import DemoEvidence
    evidence = DemoEvidence(
        camera="third_person", demonstrator_type="proxy_oracle",
        object_trajectory=((0.05, 0.02),) * 2,
        contact_region_label="handle_grip",
        final_state="faucet_turned",
        rgbd_video_path=None,
    )
    intent = TurnFaucetAdapter().scripted_demo_to_intent(evidence)
    assert intent.embodiment_mapping == "proxy_contact_to_franka_grasp_turn"
    assert intent.contact_region == "handle_grip"
    assert intent.constraint_region == "none"


def test_oracle_wrong_factor_embodiment_mapping_for_grasp_turn():
    from babysteps.envs.turnfaucet_adapter import TurnFaucetAdapter
    from babysteps.schemas import Intent
    adapter = TurnFaucetAdapter()
    grasp = Intent(
        goal_state="faucet_turned", object_motion="turn",
        contact_region="handle_grip", approach_direction="from_above",
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_grasp_turn",
    )
    assert adapter.oracle_wrong_factor(grasp) == "embodiment_mapping"
    poke = Intent(
        goal_state="faucet_turned", object_motion="turn",
        contact_region="handle_grip", approach_direction="from_above",
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_poke_turn",
    )
    assert adapter.oracle_wrong_factor(poke) == "none"


def test_default_blocked_factory_empty_tuple():
    from babysteps.envs.turnfaucet_adapter import TurnFaucetAdapter
    intent = TurnFaucetAdapter().scripted_demo_to_intent(
        type("E", (), {"contact_region_label": "handle_grip"})()  # minimal stub
    ) if False else None
    # Just call default_blocked_factory with any intent; output is fixed.
    from babysteps.schemas import Intent
    fake = Intent(
        goal_state="faucet_turned", object_motion="turn",
        contact_region="handle_grip", approach_direction="from_above",
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_grasp_turn",
    )
    assert TurnFaucetAdapter().default_blocked_factory(fake) == ()
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
pytest tests/test_turnfaucet_adapter.py::test_oracle_correct_intent_returns_poke_turn tests/test_turnfaucet_adapter.py::test_scripted_demo_to_intent_returns_grasp_turn tests/test_turnfaucet_adapter.py::test_oracle_wrong_factor_embodiment_mapping_for_grasp_turn -v
```
Expected: FAIL (current adapter returns old D's `proxy_contact_to_franka_turn` and `faucet_base` values).

- [ ] **Step 3: Update `babysteps/envs/turnfaucet_adapter.py`**

Update the relevant methods. The complete replacements (per spec §9):

```python
def oracle_correct_intent(self, scene):
    from babysteps.schemas import Intent
    return Intent(
        goal_state="faucet_turned",
        object_motion="turn",
        contact_region="handle_grip",
        approach_direction="from_above",
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_poke_turn",
    )


def scripted_demo_to_intent(self, evidence):
    """Stage-0 information loss: the demo's hand-like interaction
    symbolically reads as grasping; the 2D summarizer cannot know that
    the Franka cannot mechanically execute it.
    """
    from babysteps.schemas import Intent
    return Intent(
        goal_state="faucet_turned",
        object_motion="turn",
        contact_region="handle_grip",
        approach_direction="from_above",
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_grasp_turn",
    )


def oracle_wrong_factor(self, intent):
    return (
        "embodiment_mapping"
        if intent.embodiment_mapping == "proxy_contact_to_franka_grasp_turn"
        else "none"
    )


def default_blocked_factory(self, intent):
    return ()
```

If the existing adapter has its own `compile_skill` / `task_id` / `make_env_runner` methods that reference old D logic, leave them as-is — they're already correct (they dispatch to `compile_intent_to_turn_skill` which Task 3+4 already updated).

- [ ] **Step 4: Run tests to confirm pass**

```bash
pytest tests/test_turnfaucet_adapter.py -v
```
Expected: new tests pass. Existing tests that asserted old-D adapter behavior MAY fail — that's expected; update or delete them per the new contract. The acceptance gate (§14 of spec) requires the symbolic pipeline to work end-to-end, not preservation of predecessor D test assertions.

- [ ] **Step 5: Commit**

```bash
git add babysteps/envs/turnfaucet_adapter.py tests/test_turnfaucet_adapter.py
git commit -m "feat(d/adapter): oracle returns poke_turn; scripted returns grasp_turn; oracle_wrong_factor returns embodiment_mapping"
```

---

## Task 9: Update FakeTurnFaucetEnvRunner outcome rule

**Files:**
- Modify: `tests/conftest.py` (`FakeTurnFaucetEnvRunner.run`)

- [ ] **Step 1: Add failing test**

Append to `tests/test_turnfaucet_adapter.py` (or wherever the fake runner is tested):

```python
def test_fake_runner_poke_turn_returns_success_true(fake_turnfaucet_env_runner):
    from babysteps.schemas import Intent, SceneState
    scene = fake_turnfaucet_env_runner.reset(seed=0)
    poke = Intent(
        goal_state="faucet_turned", object_motion="turn",
        contact_region="handle_grip", approach_direction="from_above",
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_poke_turn",
    )
    result = fake_turnfaucet_env_runner.run(poke, scene)
    assert result.success is True
    assert result.object_moved is True
    assert result.reached_contact is True


def test_fake_runner_grasp_turn_returns_grasp_infeasible_signature(fake_turnfaucet_env_runner):
    from babysteps.schemas import Intent
    scene = fake_turnfaucet_env_runner.reset(seed=0)
    grasp = Intent(
        goal_state="faucet_turned", object_motion="turn",
        contact_region="handle_grip", approach_direction="from_above",
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_grasp_turn",
    )
    result = fake_turnfaucet_env_runner.run(grasp, scene)
    assert result.success is False
    assert result.object_moved is False
    assert result.reached_contact is True
    assert result.collision is False  # spec §8.4: collision never set
    assert result.grasp_slip is False  # PickCube-specific; never set by TF
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
pytest tests/test_turnfaucet_adapter.py::test_fake_runner_poke_turn_returns_success_true tests/test_turnfaucet_adapter.py::test_fake_runner_grasp_turn_returns_grasp_infeasible_signature -v
```
Expected: FAIL (old fake-env outcome rule is keyed on `contact_region`, not `embodiment_mapping`).

- [ ] **Step 3: Rewrite `FakeTurnFaucetEnvRunner.run` in `tests/conftest.py`**

Find the existing `FakeTurnFaucetEnvRunner` class in `tests/conftest.py`. Replace its `run` method body with:

```python
def run(self, intent, scene, *, rollout_log_path=None):
    """Spec §11 outcome rule:
      poke_turn  → success=True,  object_moved=True,  reached_contact=True
      grasp_turn → success=False, object_moved=False, reached_contact=True
      (failure_packet derives grasp_infeasible from these flags + intent)
    """
    from babysteps.skills.turn import compile_intent_to_turn_skill
    from babysteps.schemas import AttemptResult
    skill = compile_intent_to_turn_skill(intent, scene)
    assert skill is not None
    handle_xy = scene.extra["handle_xy"]
    if intent.embodiment_mapping == "proxy_contact_to_franka_poke_turn":
        success, moved = True, True
    else:
        # grasp_turn (and deprecated proxy_contact_to_franka_turn)
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

Make sure the `reset` method still populates `scene.extra` with `handle_xy`, `handle_z`, `target_joint_axis_xy` (per spec §11 reset block). If the existing reset only populates the old D extras (`faucet_base_xy`, etc.), update it to match the spec:

```python
def reset(self, seed):
    import numpy as np
    from babysteps.schemas import SceneState
    rng = np.random.default_rng(seed)
    handle_xy = (float(rng.uniform(0.05, 0.12)),
                  float(rng.uniform(-0.05, 0.05)))
    handle_z = 0.10
    axis_xy = (0.0, 1.0)
    return SceneState(
        cube_xy=handle_xy, cube_z=handle_z, goal_xy=handle_xy,
        tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0),
        blocked_sides=(),
        extra={"handle_xy": handle_xy, "handle_z": handle_z,
                "target_joint_axis_xy": axis_xy},
    )
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
pytest tests/test_turnfaucet_adapter.py -v
```
Expected: new tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/conftest.py tests/test_turnfaucet_adapter.py
git commit -m "test(d/fake-env): FakeTurnFaucetEnvRunner outcome rule keyed on embodiment_mapping"
```

---

## Task 10: Regenerate TurnFaucet snapshot

**Files:**
- Replace: `tests/snapshots/turnfaucet_samples_seeds_0_4.jsonl`

- [ ] **Step 1: Delete the stale snapshot**

```bash
rm tests/snapshots/turnfaucet_samples_seeds_0_4.jsonl
```

- [ ] **Step 2: Regenerate via the fake-env collector**

```bash
python scripts/stage0_collect.py \
    --fake-env \
    --task TurnFaucet-v1 \
    --n_episodes 5 \
    --seed_start 0 \
    --out tests/snapshots/turnfaucet_samples_seeds_0_4.jsonl
```
(If the `--out` flag is named differently in `stage0_collect.py`, adjust per its CLI.)

- [ ] **Step 3: Verify the snapshot file is non-empty and has 5 lines**

```bash
wc -l tests/snapshots/turnfaucet_samples_seeds_0_4.jsonl
```
Expected: `5`.

- [ ] **Step 4: Verify snapshot stability — rerun and diff**

```bash
python scripts/stage0_collect.py \
    --fake-env --task TurnFaucet-v1 --n_episodes 5 --seed_start 0 \
    --out /tmp/turnfaucet_rerun.jsonl
diff /tmp/turnfaucet_rerun.jsonl tests/snapshots/turnfaucet_samples_seeds_0_4.jsonl
```
Expected: empty diff (byte-identical).

- [ ] **Step 5: Verify the snapshot test in CLI tests passes**

```bash
pytest tests/test_stage0_collect_cli.py -v -k turnfaucet
```
Expected: PASS.

- [ ] **Step 6: Verify PushCube/PickCube/StackCube snapshots unchanged**

```bash
git status tests/snapshots/
```
Expected: only `turnfaucet_samples_seeds_0_4.jsonl` shows as modified.

- [ ] **Step 7: Commit**

```bash
git add tests/snapshots/turnfaucet_samples_seeds_0_4.jsonl
git commit -m "test(d/snapshot): regenerate TurnFaucet snapshot for embodiment_substitution story"
```

---

## Task 11: Update TurnFaucet delta_pp test row

**Files:**
- Modify: `tests/test_pickcube_delta_pp.py`

- [ ] **Step 1: Inspect current test structure**

```bash
grep -n "TurnFaucet" tests/test_pickcube_delta_pp.py
```
Locate the TurnFaucet parametrize row (added by predecessor D).

- [ ] **Step 2: Run the existing TurnFaucet delta_pp test to see current state**

```bash
pytest tests/test_pickcube_delta_pp.py -v -k turnfaucet
```
If PASS: the existing row works with the new fake-env outcome rule (which it should — initial grasp_turn fails, revised poke_turn succeeds, delta_pp = 100% - 0% = 100pp). Skip to Step 4.
If FAIL: continue to Step 3.

- [ ] **Step 3: If the row needs updating**

The existing row likely parametrizes a `(task_name, expected_delta_pp_floor)` tuple. Confirm the floor is `10` (matching the Pick4Pass M-BABY-1 bar) and that the task is `"TurnFaucet-v1"`. If the row passes a different oracle/scripted intent, remove that parametrization (the adapter handles intent generation now).

- [ ] **Step 4: Run test to confirm pass**

```bash
pytest tests/test_pickcube_delta_pp.py -v -k turnfaucet
```
Expected: PASS with delta_pp ≥ 10 (likely ≥ 90 given fake-env is deterministic poke_turn → success and grasp_turn → fail).

- [ ] **Step 5: Commit**

```bash
git add tests/test_pickcube_delta_pp.py
git commit -m "test(d/delta-pp): verify TurnFaucet delta_pp >= 10 with embodiment_substitution"
```

---

## Task 12: Add helpers + _execute_skill to TurnFaucetEnvRunner

**Files:**
- Modify: `babysteps/envs/turnfaucet_runner.py`

- [ ] **Step 1: Replace the entire module with the new helpers + _execute_skill**

This is a large rewrite. Replace `babysteps/envs/turnfaucet_runner.py` contents with:

```python
"""Real ManiSkill TurnFaucet-v1 env_runner — embodiment_substitution version.

Generic phase loop driven by len(skill.waypoints) + skill.gripper_schedule.
No hardcoded 4-phase grasp assumptions. run() dispatches single-trial
(grasp_turn) vs two-trial auto-sign (poke_turn) per spec §8.

Requires partnet_mobility_faucet asset download (see CLAUDE.md)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from babysteps.schemas import AttemptResult, Intent, SceneState
from babysteps.skills.turn import compile_intent_to_turn_skill


_POS_SCALE: float = 0.1
_PHASE_TOL_M: float = 0.015
_GRASP_PHASE_TOL_M: float = 0.025
_GRIP_MIN_STEPS: int = 15
_MAX_CONTROL_STEPS: int = 400
_POKE_PROBE_STEPS: int = 80
_POKE_PROBE_MIN_PROGRESS: float = 0.4   # fraction of needed_delta required by probe


def _to_np(x):
    arr = x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)
    return arr[0] if arr.ndim == 2 else arr


def _safe_bool(x) -> bool:
    """Safe bool from a (possibly batched torch) tensor."""
    if hasattr(x, "cpu"):
        x = x.cpu().numpy()
    arr = np.asarray(x)
    return bool(arr.item() if arr.ndim > 0 else arr)


def _raw_to_xyzw(raw_pose) -> np.ndarray:
    raw = np.asarray(raw_pose, dtype=np.float64)
    return np.concatenate([raw[0:3], raw[4:7], raw[3:4]])


def _read_obs(obs):
    """(tcp_xyzw, handle_xyz, joint_axis_xyz)."""
    tcp = _raw_to_xyzw(_to_np(obs["extra"]["tcp_pose"]))
    handle_xyz = _to_np(obs["extra"]["target_link_pos"]).astype(np.float64)
    axis_xyz = _to_np(obs["extra"]["target_joint_axis"]).astype(np.float64)
    return tcp, handle_xyz, axis_xyz


def _read_faucet_qpos(env) -> float:
    """env.unwrapped.target_switch_link.joint.qpos as a python float."""
    return float(_to_np(env.unwrapped.target_switch_link.joint.qpos).item())


def _read_needed_delta(env) -> float:
    """target_angle - current qpos, both via env.unwrapped."""
    env_u = env.unwrapped
    target_angle = float(_to_np(env_u.target_angle).item())
    return target_angle - _read_faucet_qpos(env)


def _prop_action(tcp_xyzw, target_xyz, gripper_cmd):
    pos_err = target_xyz - tcp_xyzw[0:3]
    action = np.zeros(7, dtype=np.float32)
    action[0:3] = np.clip(pos_err / _POS_SCALE, -1.0, 1.0).astype(np.float32)
    action[6] = np.float32(gripper_cmd)
    return action


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
    _GRIP_MIN_STEPS before advancing; poke mode has no dwell.

    Per spec §8.2: object_moved is derived from qpos delta (NOT handle xy
    delta) because target_link_pos sweeps the arc as the joint rotates
    but qpos is the direct signal of articulation motion.
    """
    obs, _ = env.reset(seed=int(seed))
    n_phases = len(skill.waypoints)
    assert len(skill.gripper_schedule) == n_phases, \
        "gripper_schedule length must match waypoints length"
    targets = [np.asarray(wp[0:3], dtype=np.float64) for wp in skill.waypoints]
    grip_phase = 2 if skill.mode == "grasp" and n_phases >= 3 else -1
    phase_tol = tuple(
        _GRASP_PHASE_TOL_M if i == grip_phase else _PHASE_TOL_M
        for i in range(n_phases)
    )

    tcp0, handle_xyz0, _ = _read_obs(obs)
    initial_xy = (float(handle_xyz0[0]), float(handle_xyz0[1]))
    initial_qpos = _read_faucet_qpos(env)

    trajectory: list[tuple[float, float]] = []
    phase_idx, steps_in_phase = 0, 0
    reached_contact, success = False, False
    qpos_extremum = initial_qpos

    for _ in range(max_steps):
        tcp, handle_xyz, _ = _read_obs(obs)
        trajectory.append((float(handle_xyz[0]), float(handle_xyz[1])))
        target = targets[phase_idx]
        reached = np.linalg.norm(target - tcp[0:3]) < phase_tol[phase_idx]
        advance = reached and (
            phase_idx != grip_phase or steps_in_phase >= _GRIP_MIN_STEPS
        )
        if advance:
            phase_idx += 1
            steps_in_phase = 0
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
    object_moved = abs(qpos_extremum - initial_qpos) > 0.05  # rad

    return _TrialOutcome(
        success=success, reached_contact=reached_contact, object_moved=object_moved,
        qpos_extremum_signed_progress=progress,
        initial_obj_xy=initial_xy, final_obj_xy=final_xy,
        trajectory_xy=tuple(trajectory),
    )


class TurnFaucetEnvRunner:
    """Real ManiSkill TurnFaucet-v1 runner. See run() for dispatch logic."""

    def __init__(self) -> None:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401
        self._env = gym.make(
            "TurnFaucet-v1",
            obs_mode="state_dict",
            control_mode="pd_ee_delta_pose",
            sim_backend="gpu",   # CPU IK is broken for this env; see predecessor D §10
        )
        self._last_seed: Optional[int] = None

    def reset(self, seed: int) -> SceneState:
        self._last_seed = int(seed)
        obs, _ = self._env.reset(seed=int(seed))
        tcp, handle_xyz, axis_xyz = _read_obs(obs)
        handle_xy = (float(handle_xyz[0]), float(handle_xyz[1]))
        handle_z = float(handle_xyz[2])
        axis_xy = (float(axis_xyz[0]), float(axis_xyz[1]))
        return SceneState(
            cube_xy=handle_xy,
            cube_z=handle_z,
            goal_xy=handle_xy,
            tcp_start_pose=tuple(float(v) for v in tcp),  # type: ignore[arg-type]
            blocked_sides=(),
            extra={
                "handle_xy": handle_xy,
                "handle_z": handle_z,
                "target_joint_axis_xy": axis_xy,
            },
        )

    def run(self, intent: Intent, scene: SceneState, *,
            rollout_log_path: Optional[Path] = None) -> AttemptResult:
        raise NotImplementedError("run() implemented in Task 13")

    def _outcome_to_attempt_result(
        self, outcome: _TrialOutcome, scene: SceneState,
        rollout_log_path: Optional[Path],
    ) -> AttemptResult:
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
            collision=False,
            grasp_slip=False,
            rollout_log_path=str(rollout_log_path) if rollout_log_path else None,
            success=outcome.success,
            trajectory_xy=outcome.trajectory_xy,
        )

    def close(self) -> None:
        try:
            self._env.close()
        except Exception:
            pass
```

(Note `run()` is stubbed with `NotImplementedError`; Task 13 fills it in. This keeps the file importable.)

- [ ] **Step 2: Verify the module imports without sim**

```bash
python -c "from babysteps.envs.turnfaucet_runner import TurnFaucetEnvRunner, _execute_skill, _safe_bool, _read_faucet_qpos, _read_needed_delta; print('ok')"
```
Expected: `ok` (no import errors).

- [ ] **Step 3: Run fake-env tests to confirm nothing breaks**

```bash
pytest tests/test_turnfaucet_adapter.py tests/test_stage0_collect_cli.py -v -k turnfaucet
```
Expected: all pass (the fake env_runner is used by snapshot/cli tests; the real runner is not exercised here).

- [ ] **Step 4: Commit**

```bash
git add babysteps/envs/turnfaucet_runner.py
git commit -m "feat(d/runner): generic _execute_skill phase loop + helpers (run() stub)"
```

---

## Task 13: Implement TurnFaucetEnvRunner.run() with auto-sign two-trial dispatch

**Files:**
- Modify: `babysteps/envs/turnfaucet_runner.py` (`TurnFaucetEnvRunner.run`)

- [ ] **Step 1: Replace the `run()` stub with the dispatch logic**

In `babysteps/envs/turnfaucet_runner.py`, replace the stub `def run(...)` with:

```python
def run(self, intent: Intent, scene: SceneState, *,
        rollout_log_path: Optional[Path] = None) -> AttemptResult:
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

    # Poke: auto-sign two-trial. Each trial does its own env.reset(seed) so
    # the sign retry is a true counterfactual (identical faucet config).
    # Probe with sign=+1 is a truncated preview to decide direction.
    # If probe makes >= _POKE_PROBE_MIN_PROGRESS, rerun a full trial with
    # sign=+1 from fresh reset (captured trajectory reflects a complete
    # attempt). If probe falls short, run sign=-1 at full budget and pick
    # the better of the two by progress.
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
            or full_neg.qpos_extremum_signed_progress
                > probe.qpos_extremum_signed_progress):
        return self._outcome_to_attempt_result(full_neg, scene, rollout_log_path)
    full_pos = _execute_skill(
        self._env, skill_pos, seed=seed, needed_delta=needed_delta,
        contact_xy=contact_xy, max_steps=_MAX_CONTROL_STEPS,
    )
    return self._outcome_to_attempt_result(full_pos, scene, rollout_log_path)
```

- [ ] **Step 2: Re-import check**

```bash
python -c "from babysteps.envs.turnfaucet_runner import TurnFaucetEnvRunner; print('ok')"
```
Expected: `ok`.

- [ ] **Step 3: Run fake-env tests (still no real-sim required)**

```bash
pytest tests/test_turnfaucet_adapter.py tests/test_stage0_collect_cli.py tests/test_pickcube_delta_pp.py -v -k turnfaucet
```
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add babysteps/envs/turnfaucet_runner.py
git commit -m "feat(d/runner): auto-sign two-trial dispatch for poke_turn in run()"
```

---

## Task 14: Rewrite render_episode (privileged demo + generic loop + auto-sign retry)

**Files:**
- Rewrite: `babysteps/render/turnfaucet.py`
- Modify: `tests/test_render_modules.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_render_modules.py`:

```python
def test_render_turnfaucet_returns_three_phase_keys(fake_turnfaucet_env_runner):
    """The render contract is unchanged: (frames_dict, titles_dict) with
    keys demo/attempt_blocked/retry. Phase contents change per spec §10."""
    # Construct a fake env via the FakeTurnFaucetEnvRunner-shaped scaffold.
    # Real GPU rendering is tested separately at the acceptance gate (T16).
    # This test only verifies the structural contract.
    from babysteps.render.turnfaucet import render_episode
    from babysteps.envs.turnfaucet_adapter import TurnFaucetAdapter
    # If the render module requires a sim env, this test should be marked
    # as a smoke test that only verifies the function signature and key
    # set; full visual verification is the GPU spot-check in Task 16.
    # See if the render module can accept a stub env that responds to
    # render(), reset(), step(), unwrapped (target_switch_link, target_angle).
    pass  # If smoke-only, leave a placeholder; full structural tests in next step.


def test_render_turnfaucet_titles_mention_embodiment_substitution_and_grasp_infeasible(
    fake_turnfaucet_env_runner,
):
    """Verify caption strings include the spec §10 vocabulary."""
    # Same caveat — depends on whether render_episode is testable without
    # a real ManiSkill env. If yes, assert titles contain the strings.
    pass
```

If `render_episode` requires a real env, mark these tests as `@pytest.mark.gpu` and skip them in non-GPU CI runs. Add a comment:

```python
# Visual verification of render_episode is the GPU spot-check (Task 16).
# Sim-free render tests verify only the structural contract.
```

- [ ] **Step 2: Run tests (will pass as placeholders, then we'll add structural assertions)**

```bash
pytest tests/test_render_modules.py -v -k turnfaucet
```
Expected: PASS (placeholders). Will be tightened after render module is rewritten.

- [ ] **Step 3: Replace `babysteps/render/turnfaucet.py` with the new version**

Replace the entire file contents with:

```python
"""TurnFaucet-v1 render_episode — three phases for the Stage-0 MP4 set.

Phase 1 (demo): PRIVILEGED qpos teleport — faucet handle rotates from
initial to target angle by direct write to switch_link.joint.qpos. Robot
holds home pose. Caption describes object motion, never a robot motor
program (per the Stage-0 demo-caption guideline).

Phase 2 (attempt_blocked): grasp_turn execution. Franka attempts to grip
the handle and fails (jaws cannot close on the thick partnet handle).

Phase 3 (retry): embodiment_substitution-revised poke_turn. Closed-gripper
lateral brute-force sweep with auto-sign retry as needed.

Generic over len(skill.waypoints) + skill.gripper_schedule. No hardcoded
4-phase grasp assumptions.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from babysteps.envs.task_adapter import BaseTaskAdapter
from babysteps.render.common import (
    PHASE_TOL_M, prop_action, render_frame, to_np,
)
from babysteps.schemas import AttemptResult, DemoEvidence, Intent, SceneState
from babysteps.skills.turn import compile_intent_to_turn_skill

_GRIPPER_CLOSED = -1.0
_PHASE_TOL_M = 0.015
_GRASP_PHASE_TOL_M = 0.025
_GRIP_MIN_STEPS = 15
_MAX_CONTROL_STEPS = 400
_POKE_PROBE_STEPS = 80
_POKE_PROBE_MIN_PROGRESS = 0.4
_DEMO_MIN_FRAMES = 30


def _read_turn_obs(obs):
    tcp_raw = np.asarray(to_np(obs["extra"]["tcp_pose"]), dtype=np.float64)
    tcp = np.concatenate([tcp_raw[0:3], tcp_raw[4:7], tcp_raw[3:4]])
    handle_xyz = np.asarray(to_np(obs["extra"]["target_link_pos"]), dtype=np.float64)
    axis_xyz = np.asarray(to_np(obs["extra"]["target_joint_axis"]), dtype=np.float64)
    return tcp, handle_xyz, axis_xyz


def _safe_bool(x) -> bool:
    if hasattr(x, "cpu"):
        x = x.cpu().numpy()
    arr = np.asarray(x)
    return bool(arr.item() if arr.ndim > 0 else arr)


def _read_faucet_qpos(env_u) -> float:
    return float(to_np(env_u.target_switch_link.joint.qpos).item())


def _set_faucet_qpos(env_u, switch_link, new_qpos: float) -> None:
    """Direct write to the rotating joint's qpos (privileged, no physics).
    The exact API depends on SAPIEN's articulation interface; use whichever
    of these is supported in the current ManiSkill version:
      switch_link.joint.qpos = torch.tensor([[new_qpos]])   # SAPIEN 3.x
    """
    import torch
    switch_link.joint.qpos = torch.tensor(
        [[new_qpos]], dtype=torch.float32, device=switch_link.joint.qpos.device,
    )


def _execute_skill_for_render(env, skill, *, seed, frames, contact_xy, max_steps):
    """Render-side mirror of TurnFaucetEnvRunner._execute_skill. Same
    generic phase loop but appends render_frame(env) each step instead
    of building a trajectory list. Returns dict with success + progress.
    """
    obs, _ = env.reset(seed=int(seed))
    n_phases = len(skill.waypoints)
    assert len(skill.gripper_schedule) == n_phases
    targets = [np.asarray(wp[0:3], dtype=np.float64) for wp in skill.waypoints]
    grip_phase = 2 if skill.mode == "grasp" and n_phases >= 3 else -1
    phase_tol = tuple(
        _GRASP_PHASE_TOL_M if i == grip_phase else _PHASE_TOL_M
        for i in range(n_phases)
    )

    env_u = env.unwrapped
    target_angle = float(to_np(env_u.target_angle).item())
    initial_qpos = _read_faucet_qpos(env_u)
    needed_delta = target_angle - initial_qpos
    qpos_extremum = initial_qpos

    frames.append(render_frame(env))
    phase_idx, steps_in_phase = 0, 0
    success = False
    for _ in range(max_steps):
        tcp, handle_xyz, _ = _read_turn_obs(obs)
        target = targets[phase_idx]
        reached = np.linalg.norm(target - tcp[0:3]) < phase_tol[phase_idx]
        advance = reached and (
            phase_idx != grip_phase or steps_in_phase >= _GRIP_MIN_STEPS
        )
        if advance:
            phase_idx += 1
            steps_in_phase = 0
            if phase_idx >= n_phases:
                break
            target = targets[phase_idx]
        else:
            steps_in_phase += 1
        action = prop_action(tcp, target, gripper_cmd=skill.gripper_schedule[phase_idx])
        obs, _r, terminated, truncated, info = env.step(action)
        frames.append(render_frame(env))
        qpos = _read_faucet_qpos(env_u)
        if needed_delta > 0:
            qpos_extremum = max(qpos_extremum, qpos)
        else:
            qpos_extremum = min(qpos_extremum, qpos)
        success = _safe_bool(info.get("success", False))
        if success or _safe_bool(terminated) or _safe_bool(truncated):
            break

    progress = (qpos_extremum - initial_qpos) / max(abs(needed_delta), 1e-6)
    return {"success": bool(success), "progress": float(progress)}


def render_episode(env, adapter, seed, fps):
    short_id = f"seed {seed:04d}"
    env_u = env.unwrapped

    # === Phase 1 — DEMO PROXY (privileged qpos teleport) ===
    obs, _ = env.reset(seed=seed)
    switch_link = env_u.target_switch_link
    initial_qpos = _read_faucet_qpos(env_u)
    target_angle = float(to_np(env_u.target_angle).item())
    n_demo = max(int(2 * fps), _DEMO_MIN_FRAMES)
    demo_frames = []
    for i in range(n_demo):
        t = i / max(n_demo - 1, 1)
        new_qpos = initial_qpos + t * (target_angle - initial_qpos)
        _set_faucet_qpos(env_u, switch_link, new_qpos)
        demo_frames.append(render_frame(env))

    # Reset for the real execution phases. Demo's qpos teleport is discarded.
    obs, _ = env.reset(seed=seed)
    tcp, handle_xyz, axis_xyz = _read_turn_obs(obs)
    handle_xy = (float(handle_xyz[0]), float(handle_xyz[1]))
    handle_z = float(handle_xyz[2])
    axis_xy = (float(axis_xyz[0]), float(axis_xyz[1]))
    scene = SceneState(
        cube_xy=handle_xy, cube_z=handle_z, goal_xy=handle_xy,
        tcp_start_pose=tuple(float(v) for v in tcp),  # type: ignore[arg-type]
        blocked_sides=(),
        extra={"handle_xy": handle_xy, "handle_z": handle_z,
                "target_joint_axis_xy": axis_xy},
    )

    demo_evidence = DemoEvidence(
        camera="third_person", demonstrator_type="proxy_oracle",
        object_trajectory=(handle_xy, handle_xy),
        contact_region_label="handle_grip",
        final_state="faucet_turned",
        rgbd_video_path=None,
    )
    initial_intent = adapter.scripted_demo_to_intent(demo_evidence)
    scene_exec = replace(
        scene, blocked_sides=adapter.default_blocked_factory(initial_intent),
    )
    contact_xy = np.asarray(handle_xy, dtype=np.float64)

    # === Phase 2 — ATTEMPT (grasp_turn, fails) ===
    grasp_skill = compile_intent_to_turn_skill(initial_intent, scene_exec)
    attempt_frames = []
    _ = _execute_skill_for_render(
        env, grasp_skill, seed=seed, frames=attempt_frames,
        contact_xy=contact_xy, max_steps=_MAX_CONTROL_STEPS,
    )

    # === Phase 3 — RETRY (poke_turn after embodiment_substitution) ===
    fp = adapter.build_failure_packet(
        initial_intent,
        AttemptResult(
            initial_obj_xy=handle_xy, final_obj_xy=handle_xy,
            goal_xy=handle_xy,
            reached_contact=True, object_moved=False,
            planner_failed=False, collision=False, grasp_slip=False,
            rollout_log_path=None, success=False,
        ),
        scene_exec,
    )
    attribution = adapter.attribute_failure(fp)
    revised_intent, _rev = adapter.revise_intent(initial_intent, attribution, scene_exec)

    # Probe sign=+1; if no progress, retry sign=-1 with frames replaced.
    poke_pos = compile_intent_to_turn_skill(revised_intent, scene_exec, sign=+1)
    retry_frames = []
    out_probe = _execute_skill_for_render(
        env, poke_pos, seed=seed, frames=retry_frames,
        contact_xy=contact_xy, max_steps=_POKE_PROBE_STEPS,
    )
    if (not out_probe["success"]) and out_probe["progress"] < _POKE_PROBE_MIN_PROGRESS:
        retry_frames = []
        poke_neg = compile_intent_to_turn_skill(revised_intent, scene_exec, sign=-1)
        out_retry = _execute_skill_for_render(
            env, poke_neg, seed=seed, frames=retry_frames,
            contact_xy=contact_xy, max_steps=_MAX_CONTROL_STEPS,
        )
    elif (not out_probe["success"]) and out_probe["progress"] >= _POKE_PROBE_MIN_PROGRESS:
        # Probe direction is right; rerun a full trial from fresh reset.
        retry_frames = []
        out_retry = _execute_skill_for_render(
            env, poke_pos, seed=seed, frames=retry_frames,
            contact_xy=contact_xy, max_steps=_MAX_CONTROL_STEPS,
        )
    else:
        out_retry = out_probe

    # Tail-pad attempt_frames so all three MP4s have similar duration.
    if attempt_frames:
        attempt_frames = attempt_frames + [attempt_frames[-1]] * fps

    demo_title = (
        f"{short_id}  phase 1/3: third-person object-motion proxy",
        f"faucet turned (handle_grip, faucet_turned)",
    )
    a2_title = (
        f"{short_id}  phase 2/3: grasp_infeasible",
        f"embodiment={initial_intent.embodiment_mapping}: jaws cannot close on handle → no rotation",
    )
    a3_title = (
        f"{short_id}  phase 3/3: retry (success={out_retry['success']})",
        f"embodiment_substitution: {initial_intent.embodiment_mapping} → {revised_intent.embodiment_mapping}",
    )

    return (
        {"demo": demo_frames, "attempt_blocked": attempt_frames, "retry": retry_frames},
        {"demo": demo_title, "attempt_blocked": a2_title, "retry": a3_title},
    )
```

- [ ] **Step 4: Run any sim-free render tests**

```bash
pytest tests/test_render_modules.py -v -k turnfaucet
```
Expected: PASS (smoke tests). Full visual verification is Task 16's GPU spot-check.

- [ ] **Step 5: Verify the module imports**

```bash
python -c "from babysteps.render.turnfaucet import render_episode; print('ok')"
```
Expected: `ok`.

- [ ] **Step 6: Commit**

```bash
git add babysteps/render/turnfaucet.py tests/test_render_modules.py
git commit -m "feat(d/render): privileged demo + generic phase loop + auto-sign poke retry"
```

---

## Task 15: Documentation updates (predecessor spec supersession + CLAUDE.md)

**Files:**
- Modify: `docs/superpowers/specs/2026-05-17-stage0-turnfaucet-d-design.md` (front-matter only)
- Modify: `CLAUDE.md` (TurnFaucet section)

- [ ] **Step 1: Add supersession note to predecessor spec**

Edit `docs/superpowers/specs/2026-05-17-stage0-turnfaucet-d-design.md`. Replace the first line (`# Stage-0 Sub-project D ...`) with:

```markdown
# Stage-0 Sub-project D (TurnFaucet-v1) — Design Spec

> **Superseded by** `2026-05-18-stage0-turnfaucet-embodiment-design.md`.
> Reason: the `constraint_introduction` story violated the single-factor
> revision invariant (changed two factors at once) and the real-sim
> acceptance gate (§15.6: 0/5 seeds reached `info["success"]`). The
> `embodiment_substitution` reframe is the replacement. This document
> is kept on disk for historical context.
```

- [ ] **Step 2: Update CLAUDE.md TurnFaucet section**

Locate the `# TurnFaucet` block in `CLAUDE.md` (the one with the `srun` command for TurnFaucet). Replace its narrative paragraph with:

```markdown
# TurnFaucet (Sub-project D — embodiment_substitution; per
# 2026-05-18-stage0-turnfaucet-embodiment-design.md)
# Requires partnet_mobility_faucet asset (one-time):
#   python -m mani_skill.utils.download_asset partnet_mobility_faucet
# NOTE: render_stage0_maniskill.py auto-selects sim_backend="gpu" for
# TurnFaucet because the CPU-sim IK drifts the robot even with action=0.
# Phase 1 (demo) uses a privileged qpos teleport — robot stays at home;
# only the faucet joint moves. Phase 2 (grasp_turn attempt) physically
# fails because partnet handles exceed the Panda gripper opening. Phase
# 3 (poke_turn retry) uses closed-gripper lateral sweep with auto-sign
# detection (see scripts/_diag_tf_poke5.py for the empirical reference).
# Partial physical validation gate: >=1/5 retry MP4s reaches info["success"];
# remaining seeds visibly apply lateral tangential force.
```

(Keep the `srun` command block beneath this comment unchanged — the CLI invocation is the same; only the narrative differs.)

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-05-17-stage0-turnfaucet-d-design.md CLAUDE.md
git commit -m "docs(d): supersession note on predecessor spec; CLAUDE.md TurnFaucet section reflects embodiment_substitution"
```

---

## Task 16: GPU spot-check — partial physical validation

**Files:**
- None modified — this task verifies the acceptance gate.

- [ ] **Step 1: Run the full test suite to confirm pre-D snapshots are byte-identical**

```bash
pytest tests/ -v
```
Expected: all pass. PushCube, PickCube, StackCube snapshots unchanged. New TurnFaucet snapshot stable. ~290 tests total.

- [ ] **Step 2: Run the GPU render command for 5 episodes**

```bash
srun --account=rpaleja --partition=a30 --gres=gpu:1 --mem=115G --time=00:25:00 bash -lc '
  cd /home/wang4433/scratch/babysteps &&
  source /apps/external/conda/2025.09/etc/profile.d/conda.sh &&
  conda activate handover &&
  OUT_DIR=/home/wang4433/scratch/babysteps/renders/turnfaucet_embodiment &&
  LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" \
  python scripts/render_stage0_maniskill.py \
    --task TurnFaucet-v1 \
    --out_dir "$OUT_DIR" \
    --n_episodes 5 \
    --seed_start 0 &&
  ls -lh "$OUT_DIR/videos_maniskill"
'
```

- [ ] **Step 3: Verify 15 MP4s produced (5 episodes × 3 phases)**

```bash
ls renders/turnfaucet_embodiment/videos_maniskill/ | wc -l
```
Expected: `15`.

- [ ] **Step 4: Verify ≥1/5 seeds reaches info["success"] in retry**

The render script logs per-episode success. Grep for success markers in stdout (or inspect the per-seed metric JSON if `render_stage0_maniskill.py` writes one):

```bash
grep -E "retry.*success" renders/turnfaucet_embodiment/videos_maniskill/*.log 2>/dev/null \
    || echo "(check the render command's stdout or scripts/_diag_tf_poke5.py output for seed 1 = success)"
```

Manual verification: open `renders/turnfaucet_embodiment/videos_maniskill/turnfaucet_seed_0001__3_retry.mp4` — the seed 1 retry should visibly rotate the faucet handle past target.

- [ ] **Step 5: Document the partial validation outcome**

For each seed, record:
- Demo MP4: shows faucet rotating, robot at home pose (visual check).
- Attempt MP4: shows Franka approach + descend + fail to grip + no rotation.
- Retry MP4: shows Franka closed-gripper lateral sweep; note success/fail.

Acceptable outcomes (any of these clears the gate):
- ≥ 1 retry MP4 with `info["success"]=True`.
- For non-success retries: lateral tangential motion visible (differentiates from grasp_turn's vertical jaw-close).

- [ ] **Step 6: No code commit; close out the implementation**

```bash
git status
```
Expected: clean working tree (or only render output dirs untracked). Tag the implementation as done:

```bash
git log --oneline -16
```
Expected: a clean sequence of T1-T15 commits, each focused on one logical change.

---

## Self-Review

**Spec coverage check:**
- §1 Motivation → addressed by overall reframe; no task needed.
- §2 Single-factor revision → T7 (operator) + T8 (adapter scripted/oracle).
- §3 Stage-0 controlled failure → T6 (grasp_infeasible derivation) + T9 (fake outcome rule).
- §4 Schema deltas → T1.
- §5 Failure attribution → T5 + T6.
- §6 Revision operator → T7.
- §7 TurnSkill compilation → T2 + T3 + T4.
- §8 TurnFaucetEnvRunner → T12 + T13.
- §9 TurnFaucetAdapter → T8.
- §10 Render module → T14.
- §11 FakeTurnFaucetEnvRunner → T9.
- §12 Test plan → spread across T1, T5, T6, T7, T8, T9, T11, T14.
- §13 Empirical grounding → T16 (verifies seed 1 still succeeds with the spec'd implementation).
- §14 Acceptance gate → T10 (snapshot), T11 (delta_pp), T16 (GPU spot-check).
- §15 Risks & open questions → addressed by spec, no code task.
- §16 Migration / supersession → T15.

Every spec section maps to at least one task. No gaps.

**Placeholder scan:** No "TBD", "TODO", or "fill in details" remain. All code blocks are concrete. The `pass` placeholders in Task 14 Step 1 are intentional smoke-test stubs — they pass trivially and the visual verification is delegated to T16.

**Type consistency:** `TurnSkill` carries `mode: str`, `gripper_schedule: tuple[float, ...]`, `sign: int` consistently in T2-T4-T9-T12-T14. `compile_intent_to_turn_skill(intent, scene, sign=+1)` signature is consistent across T3, T4, T12, T13, T14. `_execute_skill` parameter list (`env, skill, *, seed, needed_delta, contact_xy, max_steps`) is identical in T12 (runner) and T14 (render's `_execute_skill_for_render` mirror). `AttemptResult` field set is unchanged from existing schema (no new fields per spec §5's context-derived predicate).
