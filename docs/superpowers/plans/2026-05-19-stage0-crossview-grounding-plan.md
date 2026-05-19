# Cross-View Grounding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Stage-0 cross-view task where Robot B mis-grounds a demonstrated push direction (observed from a rotated observer view), fails with a direction error, and BABYSTEPS revises only the new `direction_grounding` factor to recover.

**Architecture:** Additive 7th intent factor `direction_grounding` (defaulted + omitted-when-default so the four existing tasks' snapshots stay byte-identical). The demo trajectory is rotated into the observer frame by a new default-identity `observe_demo` adapter hook (−yaw); a thin `CrossViewPushEnvRunner` resolves the stored intent back to world frame (+yaw for `observer_frame`, identity for `actor_frame`) before delegating to unchanged PushCube physics. The −yaw observation and +yaw resolution cancel, so `actor_frame` (identity) pushes the wrong way and the revised `observer_frame` pushes correctly.

**Tech Stack:** Python 3, NumPy, ManiSkill 3 (real runner only on GPU/Vulkan nodes), pytest. All unit tests are sim-free.

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `babysteps/schemas.py` | `DIRECTION_GROUNDINGS`, `Intent.direction_grounding`, `grounding_substitution` token | Modify |
| `babysteps/envs/scene.py` | `motion_to_unit`, `rotate_xy`, `rotate_motion_token`, `resolve_grounded_motion`, `world_resolved_intent` | Modify |
| `babysteps/revision.py` | `grounding_substitution` branch | Modify |
| `babysteps/envs/task_adapter.py` | default-identity `observe_demo` hook | Modify |
| `babysteps/episode.py` | `_diff_intents` over 7 fields; call `observe_demo` in `generate_proxy_demo` | Modify |
| `babysteps/envs/crossview_adapter.py` | `CrossViewPushAdapter` + `OBSERVER_YAWS`/`observer_yaw_for_seed` | Create |
| `babysteps/envs/crossview_runner.py` | `CrossViewPushEnvRunner` | Create |
| `babysteps/envs/task_registry.py` | `CrossViewPush-v1` entry | Modify |
| `tests/conftest.py` | `FakeCrossViewEnvRunner` | Modify |
| `babysteps/render/crossview.py` | observer-view 3-phase render flow | Create |
| `babysteps/render/__init__.py` | `CrossViewPush-v1` render entry | Modify |
| `scripts/render_stage0_maniskill.py` | `gym.make(adapter.task_id, …)` so registry key ≠ gym id works | Modify |
| `tests/test_crossview.py` | all sim-free unit + end-to-end tests | Create |
| `tests/snapshots/crossview_samples_seeds_0_4.jsonl` | committed end-to-end snapshot | Create |

**Run all tests with:** `cd /scratch/gilbreth/wang4433/babysteps && python -m pytest -q`

---

### Task 1: Schema whitelists — `DIRECTION_GROUNDINGS` + `grounding_substitution`

**Files:**
- Modify: `babysteps/schemas.py`
- Test: `tests/test_crossview.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_crossview.py`:

```python
"""Cross-view grounding (Sub-project E) unit + end-to-end tests."""
from __future__ import annotations

import numpy as np
import pytest

from babysteps import schemas


def test_direction_groundings_whitelist():
    assert schemas.DIRECTION_GROUNDINGS == frozenset(
        {"actor_frame", "observer_frame", "object_frame", "world_frame"}
    )


def test_grounding_substitution_operator_registered():
    assert "grounding_substitution" in schemas.REVISION_OPERATORS
    # Existing operators preserved.
    assert "approach_substitution" in schemas.REVISION_OPERATORS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_crossview.py -q`
Expected: FAIL — `AttributeError: module ... has no attribute 'DIRECTION_GROUNDINGS'`.

- [ ] **Step 3: Add the whitelists**

In `babysteps/schemas.py`, after the `CONSTRAINT_REGIONS` block (around line 60), add:

```python
DIRECTION_GROUNDINGS: frozenset[str] = frozenset({
    "actor_frame",       # E: cross-view — egocentric (identity) grounding; the bug
    "observer_frame",    # E: cross-view — account for the observer camera yaw; the fix
    "object_frame",      # E: reserved (later cut)
    "world_frame",       # E: default for non-cross-view tasks; inert
})
```

In the `REVISION_OPERATORS` frozenset, add the new token:

```python
    "grounding_substitution",          # E: cross-view — swap actor_frame → observer_frame
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_crossview.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add babysteps/schemas.py tests/test_crossview.py
git commit -m "feat(crossview): DIRECTION_GROUNDINGS whitelist + grounding_substitution token"
```

---

### Task 2: `Intent.direction_grounding` field (defaulted, validated, omit-when-default)

**Files:**
- Modify: `babysteps/schemas.py` (`Intent`)
- Test: `tests/test_crossview.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_crossview.py`:

```python
def test_intent_direction_grounding_defaults_and_omits():
    base = dict(
        goal_state="cube_at_target", object_motion="translate_+x",
        contact_region="minus_x_face", approach_direction="from_minus_x",
        constraint_region="none", embodiment_mapping="proxy_contact_to_franka_push",
    )
    # Default value is world_frame and is OMITTED from to_dict (snapshot-safe).
    i_default = schemas.Intent(**base)
    assert i_default.direction_grounding == "world_frame"
    assert "direction_grounding" not in i_default.to_dict()

    # Non-default value IS serialized and round-trips.
    i_actor = schemas.Intent(**base, direction_grounding="actor_frame")
    d = i_actor.to_dict()
    assert d["direction_grounding"] == "actor_frame"
    assert schemas.Intent.from_dict(d) == i_actor

    # A dict without the key reads back as the default.
    assert schemas.Intent.from_dict(i_default.to_dict()).direction_grounding == "world_frame"


def test_intent_direction_grounding_validated():
    with pytest.raises(ValueError):
        schemas.Intent(
            goal_state="cube_at_target", object_motion="translate_+x",
            contact_region="minus_x_face", approach_direction="from_minus_x",
            constraint_region="none",
            embodiment_mapping="proxy_contact_to_franka_push",
            direction_grounding="banana",
        )
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_crossview.py -q`
Expected: FAIL — `Intent.__init__() got an unexpected keyword argument 'direction_grounding'`.

- [ ] **Step 3: Add the field**

In `babysteps/schemas.py`, in the `Intent` dataclass, add the field **after** `embodiment_mapping` (it must be last because it carries a default):

```python
    embodiment_mapping: str
    direction_grounding: str = "world_frame"
```

Add validation at the end of `__post_init__`:

```python
        _validate(self.embodiment_mapping, EMBODIMENT_MAPPINGS, "embodiment_mapping")
        _validate(self.direction_grounding, DIRECTION_GROUNDINGS, "direction_grounding")
```

Replace `to_dict` and `from_dict`:

```python
    def to_dict(self) -> dict:
        d = {f: getattr(self, f) for f in INTENT_FIELDS}
        # Omit when default so the six-factor tasks serialize byte-identically
        # to their locked snapshots (additive-schema discipline).
        if self.direction_grounding != "world_frame":
            d["direction_grounding"] = self.direction_grounding
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Intent":
        kwargs = {f: d[f] for f in INTENT_FIELDS}
        if "direction_grounding" in d:
            kwargs["direction_grounding"] = d["direction_grounding"]
        return cls(**kwargs)
```

> Note: `INTENT_FIELDS` stays the six-tuple. `direction_grounding` is intentionally excluded from it; the metrics audit (Task 5) compares the 7th field separately.

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_crossview.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Verify existing snapshots are untouched**

Run: `python -m pytest -q`
Expected: PASS — all pre-existing tests (incl. the four task snapshots) still pass, because `to_dict` omits the defaulted field.

- [ ] **Step 6: Commit**

```bash
git add babysteps/schemas.py tests/test_crossview.py
git commit -m "feat(crossview): Intent.direction_grounding (defaulted, omit-when-default)"
```

---

### Task 3: `scene.py` grounding-resolution helpers

**Files:**
- Modify: `babysteps/envs/scene.py`
- Test: `tests/test_crossview.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_crossview.py`:

```python
from babysteps.envs import scene as scenemod


def test_rotate_motion_token_cardinal():
    # 180° flips signs; 90° CCW maps +x->+y, +y->-x.
    assert scenemod.rotate_motion_token("translate_+x", 180) == "translate_-x"
    assert scenemod.rotate_motion_token("translate_+y", 180) == "translate_-y"
    assert scenemod.rotate_motion_token("translate_+x", 90) == "translate_+y"
    assert scenemod.rotate_motion_token("translate_+y", 90) == "translate_-x"
    assert scenemod.rotate_motion_token("translate_+x", 0) == "translate_+x"


def test_resolve_grounded_motion():
    # actor_frame ignores yaw (identity = the bug).
    assert scenemod.resolve_grounded_motion("translate_-x", "actor_frame", 180) == "translate_-x"
    # observer_frame applies the yaw (the fix): -x observed under 180° -> +x world.
    assert scenemod.resolve_grounded_motion("translate_-x", "observer_frame", 180) == "translate_+x"
    with pytest.raises(NotImplementedError):
        scenemod.resolve_grounded_motion("translate_+x", "object_frame", 90)


def test_world_resolved_intent_recovers_world_face():
    from babysteps.schemas import Intent
    # Observer saw -x (the demo, viewed under 180°). actor_frame keeps it wrong;
    # observer_frame recovers world +x and its contact face minus_x_face.
    observed = Intent(
        goal_state="cube_at_target", object_motion="translate_-x",
        contact_region="plus_x_face", approach_direction="from_plus_x",
        constraint_region="none", embodiment_mapping="proxy_contact_to_franka_push",
        direction_grounding="observer_frame",
    )
    world = scenemod.world_resolved_intent(observed, 180)
    assert world.object_motion == "translate_+x"
    assert world.contact_region == "minus_x_face"
    assert world.approach_direction == "from_minus_x"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_crossview.py -q`
Expected: FAIL — `AttributeError: module 'babysteps.envs.scene' has no attribute 'rotate_motion_token'`.

- [ ] **Step 3: Implement the helpers**

In `babysteps/envs/scene.py`, add `from dataclasses import replace` and `from babysteps.schemas import Intent, SceneState` (extend the existing import) at the top, then add to `__all__`:

```python
    "motion_to_unit",
    "rotate_xy",
    "rotate_motion_token",
    "resolve_grounded_motion",
    "world_resolved_intent",
```

Add at the end of the module:

```python
# ---------- cross-view grounding (Sub-project E) ----------------------- #

_MOTION_UNIT: dict[str, np.ndarray] = {
    "translate_+x": np.array([1.0, 0.0]),
    "translate_-x": np.array([-1.0, 0.0]),
    "translate_+y": np.array([0.0, 1.0]),
    "translate_-y": np.array([0.0, -1.0]),
}


def motion_to_unit(token: str) -> np.ndarray:
    """xy unit vector for a cardinal OBJECT_MOTIONS token."""
    if token not in _MOTION_UNIT:
        raise ValueError(f"motion_to_unit: {token!r} is not a cardinal translate token")
    return _MOTION_UNIT[token].copy()


def rotate_xy(p: tuple[float, float], yaw_deg: int) -> tuple[float, float]:
    """Rotate an xy point CCW by a multiple of 90°. Raises on other angles."""
    yaw = int(yaw_deg) % 360
    x, y = float(p[0]), float(p[1])
    if yaw == 0:
        return (x, y)
    if yaw == 90:
        return (-y, x)
    if yaw == 180:
        return (-x, -y)
    if yaw == 270:
        return (y, -x)
    raise ValueError(f"rotate_xy supports only multiples of 90°, got {yaw_deg}")


def rotate_motion_token(token: str, yaw_deg: int) -> str:
    """Rotate a cardinal motion token CCW by a multiple of 90°."""
    u = motion_to_unit(token)
    rx, ry = rotate_xy((float(u[0]), float(u[1])), yaw_deg)
    return goal_direction_to_motion(np.array([rx, ry], dtype=np.float64))


def resolve_grounded_motion(
    observed_motion: str, grounding: str, observer_yaw_deg: int,
) -> str:
    """Map an observer-relative cardinal motion to the world frame.

    actor_frame    → apply 0° (identity; the egocentric bug)
    observer_frame → apply observer_yaw_deg (the fix)
    object/world   → reserved (later cut) → NotImplementedError
    """
    if grounding == "actor_frame":
        applied = 0
    elif grounding == "observer_frame":
        applied = int(observer_yaw_deg)
    else:
        raise NotImplementedError(
            f"resolve_grounded_motion supports actor_frame/observer_frame in "
            f"Stage-0; got {grounding!r}"
        )
    return rotate_motion_token(observed_motion, applied)


def world_resolved_intent(intent: "Intent", observer_yaw_deg: int) -> "Intent":
    """Resolve an observer-relative intent into a plain world-frame push intent.

    Only `direction_grounding` controls the resolution; the resulting world
    object_motion / contact_region / approach_direction are derived from it.
    The returned intent's `direction_grounding` is set to world_frame (inert)."""
    world_motion = resolve_grounded_motion(
        intent.object_motion, intent.direction_grounding, observer_yaw_deg,
    )
    world_face = direction_to_face(motion_to_unit(world_motion))
    world_approach = face_to_approach(world_face)
    return replace(
        intent,
        object_motion=world_motion,
        contact_region=world_face,
        approach_direction=world_approach,
        direction_grounding="world_frame",
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_crossview.py -q`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add babysteps/envs/scene.py tests/test_crossview.py
git commit -m "feat(crossview): scene grounding-resolution helpers"
```

---

### Task 4: `grounding_substitution` revision branch

**Files:**
- Modify: `babysteps/revision.py`
- Test: `tests/test_crossview.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_crossview.py`:

```python
from babysteps.failure import Attribution
from babysteps.revision import revise_intent
from babysteps.schemas import INTENT_FIELDS, Intent, SceneState


def _cv_initial_intent() -> Intent:
    return Intent(
        goal_state="cube_at_target", object_motion="translate_-x",
        contact_region="plus_x_face", approach_direction="from_plus_x",
        constraint_region="none", embodiment_mapping="proxy_contact_to_franka_push",
        direction_grounding="actor_frame",
    )


def _dummy_scene() -> SceneState:
    return SceneState(
        cube_xy=(0.0, 0.0), cube_z=0.02, goal_xy=(0.1, 0.0),
        tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0),
        blocked_sides=(), extra={"observer_yaw_deg": 180},
    )


def test_grounding_substitution_flips_only_grounding():
    intent = _cv_initial_intent()
    attribution = Attribution(
        semantic_failure=True, wrong_factor="direction_grounding",
        freeze=INTENT_FIELDS, revise=("direction_grounding",),
    )
    revised, rev = revise_intent(intent, attribution, _dummy_scene())
    assert revised.direction_grounding == "observer_frame"
    # Every six-tuple factor is unchanged.
    for f in INTENT_FIELDS:
        assert getattr(revised, f) == getattr(intent, f)
    assert rev.operator == "grounding_substitution"
    assert rev.factor == "direction_grounding"
    assert rev.old_value == "actor_frame" and rev.new_value == "observer_frame"


def test_grounding_substitution_rejects_non_actor_frame():
    intent = _cv_initial_intent()
    intent = Intent.from_dict({**intent.to_dict(), "direction_grounding": "observer_frame"})
    attribution = Attribution(
        semantic_failure=True, wrong_factor="direction_grounding",
        freeze=INTENT_FIELDS, revise=("direction_grounding",),
    )
    with pytest.raises(NotImplementedError):
        revise_intent(intent, attribution, _dummy_scene())
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_crossview.py -q`
Expected: FAIL — the final `raise NotImplementedError(... 'approach_direction', 'contact_region' ...)` fires for `direction_grounding`.

- [ ] **Step 3: Add the branch**

In `babysteps/revision.py`, add this branch **before** the final catch-all `raise NotImplementedError` (after the `embodiment_mapping` branch):

```python
    if attribution.wrong_factor == "direction_grounding":
        # Sub-project E: pure single-factor swap. Only actor_frame →
        # observer_frame is supported in Stage-0.
        if intent.direction_grounding != "actor_frame":
            raise NotImplementedError(
                f"grounding_substitution handles only actor_frame → "
                f"observer_frame (got {intent.direction_grounding!r}). See "
                f"docs/superpowers/specs/2026-05-19-stage0-crossview-grounding-design.md §6"
            )
        revised = replace(intent, direction_grounding="observer_frame")
        # INTENT_FIELDS is the six-tuple; direction_grounding (the revised
        # factor) is excluded from it, so every frozen factor is preserved.
        rev = Revision(
            operator="grounding_substitution",
            factor="direction_grounding",
            old_value="actor_frame",
            new_value="observer_frame",
            frozen_factors=INTENT_FIELDS,
        )
        return revised, rev
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_crossview.py -q`
Expected: PASS (9 tests).

- [ ] **Step 5: Commit**

```bash
git add babysteps/revision.py tests/test_crossview.py
git commit -m "feat(crossview): grounding_substitution revision branch"
```

---

### Task 5: `observe_demo` hook + `_diff_intents` over 7 fields

**Files:**
- Modify: `babysteps/envs/task_adapter.py` (`BaseTaskAdapter.observe_demo`)
- Modify: `babysteps/episode.py` (`generate_proxy_demo`, `_diff_intents`)
- Test: `tests/test_crossview.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_crossview.py`:

```python
from babysteps.episode import _diff_intents


def test_diff_intents_detects_grounding_change():
    a = _cv_initial_intent()                                  # actor_frame
    b = Intent.from_dict({**a.to_dict(), "direction_grounding": "observer_frame"})
    assert _diff_intents(a, b) == ("direction_grounding",)
    # And no false positive when nothing changes.
    assert _diff_intents(a, a) == ()


def test_base_observe_demo_is_identity():
    from babysteps.envs.pushcube_adapter import PushCubeAdapter
    adapter = PushCubeAdapter()
    traj = ((0.0, 0.0), (0.1, 0.0))
    correct = Intent(
        goal_state="cube_at_target", object_motion="translate_+x",
        contact_region="minus_x_face", approach_direction="from_minus_x",
        constraint_region="none", embodiment_mapping="proxy_contact_to_franka_push",
    )
    scene = _dummy_scene()
    out_traj, out_contact = adapter.observe_demo(traj, correct, scene)
    assert out_traj == traj
    assert out_contact == "minus_x_face"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_crossview.py -q`
Expected: FAIL — `_diff_intents` returns `()` for the grounding change (only checks `INTENT_FIELDS`); `observe_demo` does not exist.

- [ ] **Step 3a: Add the default hook**

In `babysteps/envs/task_adapter.py`, add a concrete method to `BaseTaskAdapter` (in the "overridable hooks" section, after `revise_intent`):

```python
    def observe_demo(
        self, object_trajectory, correct_intent, scene,
    ):
        """How the demo is *observed* before intent extraction. Default is
        identity: the proxy demo is observed in the same frame it was executed.
        Returns (object_trajectory, contact_region_label). Override for tasks
        whose demo view differs from the execution view (cross-view)."""
        return object_trajectory, correct_intent.contact_region
```

- [ ] **Step 3b: Call the hook in `generate_proxy_demo`**

In `babysteps/episode.py`, replace the tail of `generate_proxy_demo` (from `traj = demo_attempt.trajectory_xy` onward):

```python
    traj = demo_attempt.trajectory_xy
    if not traj:
        traj = (demo_attempt.initial_obj_xy, demo_attempt.final_obj_xy)
    observed_traj, contact_label = adapter.observe_demo(traj, correct, scene)
    return DemoEvidence(
        camera="third_person",
        demonstrator_type="proxy_oracle",
        object_trajectory=observed_traj,
        contact_region_label=contact_label,
        final_state=correct.goal_state,
        rgbd_video_path=None,
    )
```

- [ ] **Step 3c: Widen `_diff_intents`**

In `babysteps/episode.py`, replace `_diff_intents`:

```python
_DIFF_FIELDS: tuple[str, ...] = INTENT_FIELDS + ("direction_grounding",)


def _diff_intents(a: Intent, b: Intent) -> tuple[str, ...]:
    return tuple(f for f in _DIFF_FIELDS if getattr(a, f) != getattr(b, f))
```

- [ ] **Step 4: Run to verify it passes (and existing snapshots hold)**

Run: `python -m pytest -q`
Expected: PASS — new tests pass; the four existing task snapshots are unchanged (default `observe_demo` reproduces today's `DemoEvidence`; `_diff_intents` adds no factor when `direction_grounding` is equal).

- [ ] **Step 5: Commit**

```bash
git add babysteps/envs/task_adapter.py babysteps/episode.py tests/test_crossview.py
git commit -m "feat(crossview): observe_demo hook + _diff_intents over 7 fields"
```

---

### Task 6: `CrossViewPushAdapter`

**Files:**
- Create: `babysteps/envs/crossview_adapter.py`
- Test: `tests/test_crossview.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_crossview.py`:

```python
def test_crossview_adapter_methods():
    from babysteps.envs.crossview_adapter import CrossViewPushAdapter, observer_yaw_for_seed
    from babysteps.schemas import DemoEvidence

    adapter = CrossViewPushAdapter()
    assert adapter.task_id == "PushCube-v1"

    # scripted_demo_to_intent always grounds in actor_frame (the bug).
    evidence = DemoEvidence(
        camera="third_person", demonstrator_type="proxy_oracle",
        object_trajectory=((0.0, 0.0), (-0.1, 0.0)),     # observer saw -x
        contact_region_label="plus_x_face", final_state="cube_at_target",
        rgbd_video_path=None,
    )
    intent = adapter.scripted_demo_to_intent(evidence)
    assert intent.direction_grounding == "actor_frame"
    assert intent.object_motion == "translate_-x"

    # oracle_wrong_factor: direction_grounding iff rotated + actor_frame.
    rotated = SceneState(
        cube_xy=(0.0, 0.0), cube_z=0.02, goal_xy=(0.1, 0.0),
        tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0),
        blocked_sides=(), extra={"observer_yaw_deg": 180},
    )
    assert adapter.oracle_wrong_factor(intent, rotated) == "direction_grounding"
    unrotated = SceneState(
        cube_xy=(0.0, 0.0), cube_z=0.02, goal_xy=(0.1, 0.0),
        tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0),
        blocked_sides=(), extra={"observer_yaw_deg": 0},
    )
    assert adapter.oracle_wrong_factor(intent, unrotated) == "none"

    # default_blocked_factory is empty (failure is the frame bug, not a block).
    assert adapter.default_blocked_factory(intent) == ()

    # observe_demo rotates the demo into observer frame (-yaw).
    observed_traj, contact = adapter.observe_demo(
        ((0.0, 0.0), (0.1, 0.0)), _cv_world_oracle(), rotated,
    )
    # world +x viewed under 180° appears as -x.
    assert observed_traj[-1][0] < 0
    assert contact == "plus_x_face"

    # observer schedule is deterministic and never 0 (so failures always fire).
    assert observer_yaw_for_seed(0) in (90, 180, 270)


def test_crossview_attribute_failure_maps_to_grounding():
    from babysteps.envs.crossview_adapter import CrossViewPushAdapter
    from babysteps.schemas import FailurePacket
    adapter = CrossViewPushAdapter()
    intent = _cv_initial_intent()
    for predicate in ("direction_error", "goal_not_satisfied"):
        fp = FailurePacket(
            chosen_intent=intent, execution_trace={}, failure_predicate=predicate,
            object_displacement=0.1, direction_alignment=-1.0,
        )
        attr = adapter.attribute_failure(fp)
        assert attr.wrong_factor == "direction_grounding"
        assert attr.revise == ("direction_grounding",)


def _cv_world_oracle() -> Intent:
    return Intent(
        goal_state="cube_at_target", object_motion="translate_+x",
        contact_region="minus_x_face", approach_direction="from_minus_x",
        constraint_region="none", embodiment_mapping="proxy_contact_to_franka_push",
        direction_grounding="actor_frame",
    )
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_crossview.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'babysteps.envs.crossview_adapter'`.

- [ ] **Step 3: Create the adapter**

Create `babysteps/envs/crossview_adapter.py`:

```python
"""CrossViewPush adapter — Sub-project E (cross-view grounding).

Reuses PushCube physics; the cross-view-ness lives entirely in:
  * observe_demo        → rotate the demo trajectory into the observer frame (-yaw)
  * scripted_demo_to_intent → grounds in actor_frame (the egocentric bug)
  * compile_skill       → resolve observer-relative intent to world via direction_grounding
  * attribute_failure   → direction_error / goal_not_satisfied → direction_grounding
The revision (grounding_substitution) is inherited from the shared reviser.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from babysteps.demo import trajectory_to_motion
from babysteps.envs.scene import (
    direction_to_face,
    face_to_approach,
    goal_direction_to_motion,
    rotate_xy,
    world_resolved_intent,
)
from babysteps.envs.task_adapter import BaseTaskAdapter, EnvRunner
from babysteps.failure import Attribution
from babysteps.schemas import (
    INTENT_FIELDS,
    DemoEvidence,
    FailurePacket,
    Intent,
    SceneState,
)
from babysteps.skills.push import compile_intent_to_push_skill

# Deterministic per-seed observer yaw schedule. All non-zero so the
# egocentric (actor_frame) grounding always produces a wrong push.
OBSERVER_YAWS: tuple[int, ...] = (90, 180, 270)


def observer_yaw_for_seed(seed: int) -> int:
    return OBSERVER_YAWS[int(seed) % len(OBSERVER_YAWS)]


class CrossViewPushAdapter(BaseTaskAdapter):
    task_id = "PushCube-v1"   # underlying gym env; registry key is CrossViewPush-v1

    def make_env_runner(self) -> EnvRunner:
        from babysteps.envs.crossview_runner import CrossViewPushEnvRunner
        return CrossViewPushEnvRunner()

    def oracle_correct_intent(self, scene: SceneState) -> Intent:
        # World-correct push, grounded in actor_frame so that
        # world_resolved_intent (identity for actor_frame) leaves it unchanged
        # → the demo succeeds for any observer yaw.
        goal_vec = np.array(scene.goal_xy) - np.array(scene.cube_xy)
        face = direction_to_face(goal_vec)
        return Intent(
            goal_state="cube_at_target",
            object_motion=goal_direction_to_motion(goal_vec),
            contact_region=face,
            approach_direction=face_to_approach(face),
            constraint_region="none",
            embodiment_mapping="proxy_contact_to_franka_push",
            direction_grounding="actor_frame",
        )

    def observe_demo(self, object_trajectory, correct_intent, scene):
        """Rotate the world demo trajectory into the observer frame (-yaw)."""
        yaw = int(scene.extra["observer_yaw_deg"])
        origin = object_trajectory[0]

        def _obs(p):
            d = (p[0] - origin[0], p[1] - origin[1])
            rd = rotate_xy(d, -yaw)
            return (origin[0] + rd[0], origin[1] + rd[1])

        observed = tuple(_obs(p) for p in object_trajectory)
        disp = (object_trajectory[-1][0] - origin[0],
                object_trajectory[-1][1] - origin[1])
        obs_disp = rotate_xy(disp, -yaw)
        observed_contact = direction_to_face(np.array(obs_disp, dtype=np.float64))
        return observed, observed_contact

    def scripted_demo_to_intent(self, evidence: DemoEvidence) -> Intent:
        contact = evidence.contact_region_label
        motion = trajectory_to_motion(evidence.object_trajectory)
        return Intent(
            goal_state="cube_at_target",
            object_motion=motion,
            contact_region=contact,
            approach_direction=face_to_approach(contact),
            constraint_region="none",
            embodiment_mapping="proxy_contact_to_franka_push",
            direction_grounding="actor_frame",   # the egocentric bug
        )

    def default_blocked_factory(self, intent: Intent) -> tuple[str, ...]:
        return ()

    def oracle_wrong_factor(
        self, initial_intent: Intent, scene_executor: SceneState,
    ) -> str:
        yaw = int(scene_executor.extra.get("observer_yaw_deg", 0))
        if yaw % 360 != 0 and initial_intent.direction_grounding == "actor_frame":
            return "direction_grounding"
        return "none"

    def compile_skill(self, intent: Intent, scene: SceneState) -> Any:
        yaw = int(scene.extra.get("observer_yaw_deg", 0))
        return compile_intent_to_push_skill(world_resolved_intent(intent, yaw), scene)

    def attribute_failure(self, fp: FailurePacket) -> Attribution:
        # In this task there is no goal ambiguity and the object_motion content
        # is correct (only mis-framed); any wrong-place failure is a grounding
        # error. Both the opposite (direction_error) and orthogonal
        # (goal_not_satisfied) cases attribute to direction_grounding.
        if fp.failure_predicate in ("direction_error", "goal_not_satisfied"):
            return Attribution(
                semantic_failure=True,
                wrong_factor="direction_grounding",
                freeze=INTENT_FIELDS,
                revise=("direction_grounding",),
            )
        from babysteps import failure as failure_mod
        return failure_mod.attribute_failure(fp)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_crossview.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add babysteps/envs/crossview_adapter.py tests/test_crossview.py
git commit -m "feat(crossview): CrossViewPushAdapter (observe_demo, actor_frame, grounding attribution)"
```

---

### Task 7: `CrossViewPushEnvRunner`

**Files:**
- Create: `babysteps/envs/crossview_runner.py`
- Test: covered by Task 9's end-to-end (real runner needs GPU; no isolated unit test here)

- [ ] **Step 1: Create the runner**

Create `babysteps/envs/crossview_runner.py`:

```python
"""Real ManiSkill cross-view runner — thin wrapper over PushCubeEnvRunner.

reset() injects the per-seed observer yaw into SceneState.extra. run()
resolves the stored (observer-relative) intent to world frame via
direction_grounding before delegating to unchanged PushCube physics.
Needs a GPU/Vulkan node (inherited from PushCubeEnvRunner)."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Optional

from babysteps.envs.crossview_adapter import observer_yaw_for_seed
from babysteps.envs.pushcube_runner import PushCubeEnvRunner
from babysteps.envs.scene import world_resolved_intent
from babysteps.schemas import AttemptResult, Intent, SceneState


class CrossViewPushEnvRunner(PushCubeEnvRunner):
    def reset(self, seed: int) -> SceneState:
        scene = super().reset(seed)
        yaw = observer_yaw_for_seed(seed)
        return replace(scene, extra={**scene.extra, "observer_yaw_deg": yaw})

    def run(
        self, intent: Intent, scene: SceneState,
        *, rollout_log_path: Optional[Path] = None,
    ) -> AttemptResult:
        yaw = int(scene.extra["observer_yaw_deg"])
        world_intent = world_resolved_intent(intent, yaw)
        return super().run(world_intent, scene, rollout_log_path=rollout_log_path)
```

- [ ] **Step 2: Verify it imports cleanly (sim-free)**

Run: `python -c "from babysteps.envs.crossview_runner import CrossViewPushEnvRunner; print('ok')"`
Expected: prints `ok` (class import does not construct the env, so no Vulkan needed).

- [ ] **Step 3: Commit**

```bash
git add babysteps/envs/crossview_runner.py
git commit -m "feat(crossview): CrossViewPushEnvRunner (resolve-then-delegate)"
```

---

### Task 8: Register `CrossViewPush-v1` + `FakeCrossViewEnvRunner`

**Files:**
- Modify: `tests/conftest.py` (`FakeCrossViewEnvRunner`)
- Modify: `babysteps/envs/task_registry.py` (entry)
- Test: `tests/test_crossview.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_crossview.py`:

```python
def test_crossview_registry_entry():
    from babysteps.envs.task_registry import get_task_entry
    from babysteps.envs.crossview_adapter import CrossViewPushAdapter
    entry = get_task_entry("CrossViewPush-v1")
    assert entry.adapter_cls is CrossViewPushAdapter
    assert entry.episode_id_prefix == "crossview_grounding"
    runner = entry.fake_runner_factory()
    assert runner.__class__.__name__ == "FakeCrossViewEnvRunner"


def test_fake_crossview_runner_failure_then_success():
    from tests.conftest import FakeCrossViewEnvRunner
    from babysteps.envs.crossview_adapter import CrossViewPushAdapter
    runner = FakeCrossViewEnvRunner()
    adapter = CrossViewPushAdapter()
    scene = runner.reset(0)
    assert scene.extra["observer_yaw_deg"] in (90, 180, 270)

    # The actor_frame initial intent pushes the wrong way.
    from babysteps.episode import generate_proxy_demo
    demo = generate_proxy_demo(runner, scene, adapter)
    initial = adapter.scripted_demo_to_intent(demo)
    assert initial.direction_grounding == "actor_frame"
    a1 = runner.run(initial, scene)
    assert a1.success is False and a1.object_moved is True

    # observer_frame recovers the correct push.
    from babysteps.schemas import Intent
    revised = Intent.from_dict({**initial.to_dict(), "direction_grounding": "observer_frame"})
    a2 = runner.run(revised, scene)
    assert a2.success is True
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_crossview.py -q`
Expected: FAIL — `KeyError`/`ImportError` for `CrossViewPush-v1` / `FakeCrossViewEnvRunner`.

- [ ] **Step 3a: Add `FakeCrossViewEnvRunner` to `tests/conftest.py`**

At the top of `tests/conftest.py`, add to the imports:

```python
from dataclasses import replace  # noqa: E402
```

Append at the end of `tests/conftest.py`:

```python
class FakeCrossViewEnvRunner(FakeEnvRunner):
    """Deterministic, sim-free cross-view runner.

    reset injects the per-seed observer yaw. run resolves the stored intent to
    world frame (via direction_grounding) then reuses FakeEnvRunner physics:
    actor_frame (identity) pushes along the observer-relative — i.e. wrong —
    direction; observer_frame recovers the world push and reaches the goal.
    """

    OBSERVER_YAWS: tuple[int, ...] = (90, 180, 270)

    def reset(self, seed: int) -> SceneState:
        base = super().reset(seed)
        yaw = self.OBSERVER_YAWS[int(seed) % len(self.OBSERVER_YAWS)]
        scene = replace(base, extra={**base.extra, "observer_yaw_deg": yaw})
        self._scenes_by_seed[seed] = scene
        return scene

    def run(self, intent: Intent, scene: SceneState) -> AttemptResult:
        from babysteps.envs.scene import world_resolved_intent
        yaw = int(scene.extra["observer_yaw_deg"])
        return super().run(world_resolved_intent(intent, yaw), scene)


@pytest.fixture
def fake_crossview_env_runner() -> "FakeCrossViewEnvRunner":
    return FakeCrossViewEnvRunner()
```

- [ ] **Step 3b: Register the task**

In `babysteps/envs/task_registry.py`, add an entry function after `_stackcube_entry` (mirror the existing pattern):

```python
def _crossview_entry() -> TaskEntry:
    from babysteps.envs.crossview_adapter import CrossViewPushAdapter

    def _make_fake() -> EnvRunner:
        from tests.conftest import FakeCrossViewEnvRunner
        return FakeCrossViewEnvRunner()

    return TaskEntry(
        adapter_cls=CrossViewPushAdapter,
        fake_runner_factory=_make_fake,
        episode_id_prefix="crossview_grounding",
    )
```

Then add `"CrossViewPush-v1": _crossview_entry,` to the `TASK_REGISTRY` dict (find it lower in the file alongside `"PushCube-v1": _pushcube_entry,` etc.).

> If `TASK_REGISTRY` is built by calling the entry functions (e.g.
> `{"PushCube-v1": _pushcube_entry()}`), match that exact form instead —
> read the bottom of `task_registry.py` and follow whichever pattern is there.

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_crossview.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/conftest.py babysteps/envs/task_registry.py tests/test_crossview.py
git commit -m "feat(crossview): register CrossViewPush-v1 + FakeCrossViewEnvRunner"
```

---

### Task 9: End-to-end loop test + committed snapshot + single-factor invariant

**Files:**
- Test: `tests/test_crossview.py`
- Create: `tests/snapshots/crossview_samples_seeds_0_4.jsonl`

- [ ] **Step 1: Write the end-to-end test (no snapshot file yet)**

Append to `tests/test_crossview.py`:

```python
import pathlib

_SNAPSHOT = pathlib.Path(__file__).parent / "snapshots" / "crossview_samples_seeds_0_4.jsonl"


def _run_crossview_episodes(n=5):
    from babysteps.episode import run_episode
    from babysteps.envs.crossview_adapter import CrossViewPushAdapter
    from tests.conftest import FakeCrossViewEnvRunner
    adapter = CrossViewPushAdapter()
    adapter._env_runner = FakeCrossViewEnvRunner()   # inject fake (skip make_env_runner)
    records = [
        run_episode(
            episode_id=f"crossview_grounding_seed_{s:04d}", seed=s, adapter=adapter,
        )
        for s in range(n)
    ]
    adapter.close()
    return records


def test_crossview_loop_revises_only_grounding_and_recovers():
    for r in _run_crossview_episodes():
        m = r.metrics
        # Failure attributed to and corrected via direction_grounding.
        assert r.failure_packet["wrong_factor"] == "direction_grounding"
        assert m["oracle_wrong_factor"] == "direction_grounding"
        assert m["factor_attribution_correct"] is True
        # Single-factor invariant: exactly direction_grounding changed.
        assert tuple(m["factors_changed"]) == ("direction_grounding",)
        assert m["frozen_factors_preserved"] is True
        # Recovery: initial fails, retry succeeds.
        assert m["initial_success"] is False
        assert m["retry_success"] is True


def test_crossview_snapshot_byte_stable():
    lines = [r.to_jsonl_line() for r in _run_crossview_episodes()]
    expected = _SNAPSHOT.read_text().splitlines()
    assert lines == expected
```

- [ ] **Step 2: Run to verify the invariant test passes but snapshot fails**

Run: `python -m pytest tests/test_crossview.py -k crossview_loop -q`
Expected: PASS (invariant test). The snapshot test will fail until Step 3 (file missing).

- [ ] **Step 3: Generate the snapshot from the now-trusted loop**

Run:

```bash
python - <<'PY'
import pathlib, sys
sys.path.insert(0, ".")
from babysteps.episode import run_episode
from babysteps.envs.crossview_adapter import CrossViewPushAdapter
from tests.conftest import FakeCrossViewEnvRunner
adapter = CrossViewPushAdapter()
adapter._env_runner = FakeCrossViewEnvRunner()
out = pathlib.Path("tests/snapshots/crossview_samples_seeds_0_4.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)
lines = [
    run_episode(episode_id=f"crossview_grounding_seed_{s:04d}", seed=s, adapter=adapter).to_jsonl_line()
    for s in range(5)
]
out.write_text("\n".join(lines) + "\n")
adapter.close()
print("wrote", out, len(lines), "lines")
PY
```

Expected: `wrote tests/snapshots/crossview_samples_seeds_0_4.jsonl 5 lines`.

- [ ] **Step 4: Run to verify both pass**

Run: `python -m pytest tests/test_crossview.py -q`
Expected: PASS (all crossview tests, including the snapshot).

- [ ] **Step 5: Sanity-check the snapshot content**

Run: `python -c "import json; d=[json.loads(l) for l in open('tests/snapshots/crossview_samples_seeds_0_4.jsonl')]; print(d[0]['revision']['operator'], d[0]['execution']['initial_intent'].get('direction_grounding'), d[0]['retry']['final_intent'].get('direction_grounding'))"`
Expected: `grounding_substitution actor_frame observer_frame`.

- [ ] **Step 6: Commit**

```bash
git add tests/test_crossview.py tests/snapshots/crossview_samples_seeds_0_4.jsonl
git commit -m "test(crossview): end-to-end loop + single-factor invariant + snapshot"
```

---

### Task 10: Render module + dispatch

**Files:**
- Create: `babysteps/render/crossview.py`
- Modify: `babysteps/render/__init__.py`
- Modify: `scripts/render_stage0_maniskill.py`
- Test: `tests/test_crossview.py` (sim-free structural check)

- [ ] **Step 1: Write the failing structural test**

Append to `tests/test_crossview.py`:

```python
def test_crossview_render_is_registered():
    from babysteps.render import get_render_fn
    fn = get_render_fn("CrossViewPush-v1")
    assert callable(fn)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_crossview.py -k render -q`
Expected: FAIL — `KeyError: no render module for task 'CrossViewPush-v1'`.

- [ ] **Step 3a: Create the render module**

Create `babysteps/render/crossview.py`:

```python
"""CrossViewPush render_episode — three phases for the Stage-0 MP4 set.

Phase 1 (demo): the world-correct oracle push (the demonstration).
Phase 2 (attempt): the actor_frame (egocentric) push — cube moves the WRONG
way (this is the cross-view failure, NOT a held-still planner failure).
Phase 3 (retry): grounding_substitution → observer_frame → correct push.

Geometry reuses build_push_waypoints on world-resolved intents."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from babysteps.envs.crossview_adapter import observer_yaw_for_seed
from babysteps.envs.scene import world_resolved_intent
from babysteps.render.common import (
    PUSHCUBE_MAX_CONTROL_STEPS,
    PHASE_TOL_M,
    prop_action,
    read_obs,
    render_frame,
    to_np,
)
from babysteps.schemas import AttemptResult, DemoEvidence, SceneState
from babysteps.skills.push import build_push_waypoints


def _execute_push(env, waypoints, frames: list, *, seed: int) -> dict:
    obs, _ = env.reset(seed=int(seed))
    targets = [np.asarray(wp[0:3], dtype=np.float64) for wp in waypoints]
    phase_idx = 0
    success = False
    frames.append(render_frame(env))
    for _ in range(PUSHCUBE_MAX_CONTROL_STEPS):
        tcp, cube_xy, _, _ = read_obs(obs)
        target = targets[phase_idx]
        if np.linalg.norm(target - tcp[0:3]) < PHASE_TOL_M:
            phase_idx += 1
            if phase_idx >= len(targets):
                break
            target = targets[phase_idx]
        action = prop_action(tcp, target, gripper_cmd=-1.0)
        obs, _r, term, trunc, info = env.step(action)
        frames.append(render_frame(env))
        term_b = bool(to_np(term).item()) if hasattr(term, "cpu") else bool(term)
        trunc_b = bool(to_np(trunc).item()) if hasattr(trunc, "cpu") else bool(trunc)
        succ = info.get("success", False) if hasattr(info, "get") else False
        success = bool(to_np(succ).item()) if hasattr(succ, "cpu") else bool(succ)
        if success or term_b or trunc_b:
            break
    tcp, final_cube_xy, _, _ = read_obs(obs)
    return {
        "final_obj_xy": (float(final_cube_xy[0]), float(final_cube_xy[1])),
        "success": bool(success),
    }


def render_episode(env, adapter, seed: int, fps: int) -> tuple[dict, dict]:
    short_id = f"seed {seed:04d}"
    yaw = observer_yaw_for_seed(seed)

    obs, _ = env.reset(seed=seed)
    tcp_xyzw, cube_xy0, goal_xy, cube_z = read_obs(obs)
    scene = SceneState(
        cube_xy=(float(cube_xy0[0]), float(cube_xy0[1])),
        cube_z=cube_z,
        goal_xy=(float(goal_xy[0]), float(goal_xy[1])),
        tcp_start_pose=tuple(float(v) for v in tcp_xyzw),  # type: ignore[arg-type]
        blocked_sides=(),
        extra={"observer_yaw_deg": yaw},
    )

    # === Phase 1 — DEMO (world-correct oracle push) ===
    correct = adapter.oracle_correct_intent(scene)
    wp_demo = build_push_waypoints(scene, world_resolved_intent(correct, yaw))
    demo_frames: list = []
    out_demo = _execute_push(env, wp_demo, demo_frames, seed=seed)

    observed_traj, contact_label = adapter.observe_demo(
        (scene.cube_xy, out_demo["final_obj_xy"]), correct, scene,
    )
    evidence = DemoEvidence(
        camera="observer_view", demonstrator_type="proxy_oracle",
        object_trajectory=observed_traj, contact_region_label=contact_label,
        final_state=correct.goal_state, rgbd_video_path=None,
    )
    initial = adapter.scripted_demo_to_intent(evidence)
    scene_exec = replace(scene, blocked_sides=adapter.default_blocked_factory(initial))

    # === Phase 2 — ATTEMPT 1 (actor_frame, wrong push) ===
    wp_a = build_push_waypoints(scene_exec, world_resolved_intent(initial, yaw))
    attempt_frames: list = []
    out_a = _execute_push(env, wp_a, attempt_frames, seed=seed)

    # === Phase 3 — RETRY (observer_frame) ===
    fp = adapter.build_failure_packet(
        initial,
        AttemptResult(
            initial_obj_xy=scene.cube_xy, final_obj_xy=out_a["final_obj_xy"],
            goal_xy=scene.goal_xy, reached_contact=True, object_moved=True,
            planner_failed=False, collision=False, grasp_slip=False,
            rollout_log_path=None, success=out_a["success"],
        ),
        scene_exec,
    )
    attribution = adapter.attribute_failure(fp)
    revised, _rev = adapter.revise_intent(initial, attribution, scene_exec)
    wp_r = build_push_waypoints(scene_exec, world_resolved_intent(revised, yaw))
    retry_frames: list = []
    out_r = _execute_push(env, wp_r, retry_frames, seed=seed)

    demo_title = (
        f"{short_id}  phase 1/3: demo (observer view, yaw={yaw}deg)",
        f"object moved to target; observed contact_region={contact_label}",
    )
    a1_title = (
        f"{short_id}  phase 2/3: attempt (direction_grounding=actor_frame)",
        f"egocentric grounding pushes wrong way (success={out_a['success']})",
    )
    retry_title = (
        f"{short_id}  phase 3/3: retry (success={out_r['success']})",
        "grounding_substitution: actor_frame -> observer_frame",
    )
    return (
        {"demo": demo_frames, "attempt_blocked": attempt_frames, "retry": retry_frames},
        {"demo": demo_title, "attempt_blocked": a1_title, "retry": retry_title},
    )
```

> Observer-camera note: this first cut renders all phases from PushCube's
> default `render_mode="rgb_array"` camera; the demo banner states the
> observer yaw. A literal rotated observer camera (ManiSkill
> `human_render_camera_configs`) is a deferred visual enhancement — the
> data record (Task 9) carries `observer_yaw_deg` and is the claim-bearing
> artifact, not the MP4 viewpoint.

- [ ] **Step 3b: Register the render fn**

In `babysteps/render/__init__.py`, add after `_turnfaucet_render`:

```python
def _crossview_render() -> RenderEpisodeFn:
    from babysteps.render.crossview import render_episode
    return render_episode
```

And add to `RENDER_REGISTRY`:

```python
    "CrossViewPush-v1": _crossview_render,
```

- [ ] **Step 3c: Make the render script use the gym task id**

In `scripts/render_stage0_maniskill.py`, change the env creation so the registry key (`CrossViewPush-v1`) maps to the real gym env (`PushCube-v1`) via the adapter:

```python
        env = gym.make(
            adapter.task_id,
            obs_mode="state_dict",
            control_mode="pd_ee_delta_pose",
            sim_backend="cpu",
            render_mode="rgb_array",
        )
```

(Replaces `gym.make(args.task, ...)`. For the four existing tasks `adapter.task_id == args.task`, so behavior is unchanged.)

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_crossview.py -k render -q`
Expected: PASS. Then `python -m pytest -q` → all pass.

- [ ] **Step 5: Commit**

```bash
git add babysteps/render/crossview.py babysteps/render/__init__.py scripts/render_stage0_maniskill.py tests/test_crossview.py
git commit -m "feat(crossview): observer-view render module + dispatch (gym.make via adapter.task_id)"
```

---

### Task 11: GPU physical-validation render + data cut (acceptance)

**Files:** none changed — this task runs the real sim on a GPU node and records results.

- [ ] **Step 1: Render the 3-phase MP4s on a GPU node**

```bash
srun --account=rpaleja --partition=a100-40gb --gres=gpu:1 --mem=115G --time=00:20:00 bash -lc '
  cd /scratch/gilbreth/wang4433/babysteps &&
  source /apps/external/conda/2025.09/etc/profile.d/conda.sh &&
  conda activate handover &&
  OUT_DIR=/scratch/gilbreth/wang4433/render_crossview &&
  LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" \
  python scripts/render_stage0_maniskill.py \
    --task CrossViewPush-v1 --out_dir "$OUT_DIR" \
    --n_episodes 3 --seed_start 0 &&
  ls -lh "$OUT_DIR/videos_maniskill"
'
```

Expected: 3 episodes × 3 MP4s = 9 files. **Acceptance criterion 3:** for ≥2 seeds, the `3_retry` MP4 reaches `info["success"]` (printed during the run) and the `2_attempt_blocked` MP4 visibly pushes the cube the wrong way.

- [ ] **Step 2: Collect the data cut (real sim, ~24 seeds)**

```bash
srun --account=rpaleja --partition=a100-40gb --gres=gpu:1 --mem=115G --time=00:30:00 bash -lc '
  cd /scratch/gilbreth/wang4433/babysteps &&
  source /apps/external/conda/2025.09/etc/profile.d/conda.sh &&
  conda activate handover &&
  OUT_DIR=/scratch/gilbreth/wang4433/data_crossview &&
  LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" \
  python scripts/stage0_collect.py --task CrossViewPush-v1 \
    --n_episodes 24 --seed_start 0 --out_dir "$OUT_DIR" &&
  python scripts/stage0_summarize.py --samples "$OUT_DIR/samples.jsonl" --out_dir "$OUT_DIR"
'
```

> If `scripts/stage0_collect.py`'s flags differ, read its `argparse` block
> and match them — the four existing tasks already collect via this script,
> so the same invocation shape applies with `--task CrossViewPush-v1`.

- [ ] **Step 3: Verify the acceptance gate from the report**

Open `/scratch/gilbreth/wang4433/data_crossview/report.json` and confirm:
- `delta_pp >= 10.0` and `passed_acceptance == true`
- `intent_factor_attribution_accuracy == 1.0`
- `frozen_factor_preservation_rate == 1.0`
- `unnecessary_factor_change_rate == 0.0`

- [ ] **Step 4: Record results in CLAUDE.md**

Add a `CrossViewPush` section to `CLAUDE.md` (mirror the existing per-task render blocks): the srun command from Step 1, the expected 9-MP4 output naming (`crossview_grounding_seed_NNNN__{1_demo,2_attempt_blocked,3_retry}.mp4`), and the observed gate numbers from Step 3.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(crossview): GPU validation + data-cut gate results in CLAUDE.md"
```

---

## Self-Review

**1. Spec coverage** (against `2026-05-19-stage0-crossview-grounding-design.md`):
- §2 single-factor mechanism → Tasks 3 (`world_resolved_intent`), 5 (`observe_demo`), 9 (invariant assertion). ✓
- §3 schema (additive, omit-when-default) → Tasks 1, 2; snapshot-stability re-checked in Tasks 2 & 5. ✓
- §4 resolution helper → Task 3. ✓
- §5 adapter (all 7 rows incl. attribute_failure override) → Task 6; runner → Task 7. ✓
- §6 reviser (`grounding_substitution`, frozen=INTENT_FIELDS) + untouched shared table → Task 4 + Task 6 override. ✓
- §7 observer camera + render → Task 10 (literal rotated camera deferred, flagged). ✓ (partial, documented)
- §8 data/metrics + full-replanning baseline → Task 11. ⚠️ **Gap:** the spec wires a `full_replanning` baseline for the selectivity contrast; this plan implements only the BABYSTEPS-selective path + its metrics. The baseline is deferred to Milestone 3 (consistent with `milestone1_locked_claim.md §4`). Acceptance criterion 5 (the baseline contrast) is therefore **not** covered by this plan — noted so it isn't a surprise.
- §9 testing → Tasks 1–9 (sim-free), Task 10 (structural), Task 11 (GPU). ✓
- §11 acceptance gate → Task 11. ✓ (criteria 1–4; criterion 5 deferred per above)

**2. Placeholder scan:** No "TBD"/"handle edge cases"/"similar to". The two "if the script differs, read it" notes (Tasks 8, 11) point at exact files/patterns rather than hiding content.

**3. Type consistency:** `world_resolved_intent(intent, yaw)`, `resolve_grounded_motion(observed, grounding, yaw)`, `observe_demo(object_trajectory, correct_intent, scene) -> (traj, contact_label)`, `observer_yaw_for_seed(seed)`, `Revision(operator, factor, old_value, new_value, frozen_factors)`, `Attribution(semantic_failure, wrong_factor, freeze, revise)` — all match their definitions in `schemas.py`/`failure.py` and are used identically across Tasks 3–10.

**One spec gap intentionally carried:** the `full_replanning` baseline (§8/criterion 5) is out of scope for this plan and deferred to Milestone 3.
