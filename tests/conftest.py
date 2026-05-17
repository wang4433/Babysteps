"""Shared test fixtures and import-path setup."""
from __future__ import annotations

import pathlib
import sys

# Make the project root importable without `pip install -e .` (handy on Gilbreth).
_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from babysteps.envs.scene import direction_to_face, face_to_push_unit  # noqa: E402
from babysteps.skills.pick import compile_intent_to_pick_skill  # noqa: E402
from babysteps.skills.push import compile_intent_to_push_skill  # noqa: E402
from babysteps.schemas import AttemptResult, Intent, SceneState  # noqa: E402


class FakeEnvRunner:
    """Deterministic, sim-free env_runner for unit tests.

    `reset(seed)` returns a synthetic SceneState. `run(intent, scene)`
    consults the skill compiler for feasibility:
      - If `compile_intent_to_push_skill` returns None → planner_failed=True.
      - Else, the push is "physically" simulated:
          - the cube travels along `face_to_push_unit(contact_region)`,
          - by `min(0.6 * cube→goal distance, 0.15)` (Pick4Pass calibration),
          - and success is declared iff the final cube xy is within
            `GOAL_RADIUS` of `scene.goal_xy` (PushCube's ManiSkill criterion).
    """

    GOAL_RADIUS: float = 0.025      # ManiSkill PushCube success threshold
    PUSH_TRAVEL_SCALE: float = 0.6
    PUSH_TRAVEL_MAX_M: float = 0.15

    def __init__(self) -> None:
        self._scenes_by_seed: dict[int, SceneState] = {}

    def reset(self, seed: int) -> SceneState:
        if seed not in self._scenes_by_seed:
            # Deterministic synthetic scene per seed. Cube at origin, goal
            # placed along one of the four cardinal axes selected by `seed % 4`.
            rng = np.random.default_rng(seed)
            r = float(rng.uniform(0.10, 0.18))
            theta = (seed % 4) * (np.pi / 2)
            goal_xy = (r * np.cos(theta), r * np.sin(theta))
            self._scenes_by_seed[seed] = SceneState(
                cube_xy=(0.0, 0.0),
                cube_z=0.02,
                goal_xy=(float(goal_xy[0]), float(goal_xy[1])),
                tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0),
                blocked_sides=(),
            )
        return self._scenes_by_seed[seed]

    def run(self, intent: Intent, scene: SceneState) -> AttemptResult:
        skill = compile_intent_to_push_skill(intent, scene)
        if skill is None:
            return AttemptResult(
                initial_obj_xy=scene.cube_xy,
                final_obj_xy=scene.cube_xy,
                goal_xy=scene.goal_xy,
                reached_contact=False,
                object_moved=False,
                planner_failed=True,
                collision=False,
                grasp_slip=False,
                rollout_log_path=None,
                success=False,
                trajectory_xy=(),
            )

        cube = np.asarray(scene.cube_xy, dtype=np.float64)
        goal = np.asarray(scene.goal_xy, dtype=np.float64)
        goal_vec = goal - cube
        push_unit = face_to_push_unit(intent.contact_region)

        # Ideal physics for unit tests: when the contact_region matches the
        # cube→goal direction face, the push reaches the goal exactly. When
        # the contact_region is wrong, the cube moves a fixed distance along
        # the (wrong) push direction. This isolates the loop semantics from
        # PD-tracking calibration.
        correct_face = direction_to_face(goal_vec)
        if intent.contact_region == correct_face:
            final = goal.copy()
        else:
            final = cube + push_unit * 0.10

        synthetic_traj = tuple(
            (float(cube[0] + (final[0] - cube[0]) * t),
             float(cube[1] + (final[1] - cube[1]) * t))
            for t in np.linspace(0.0, 1.0, 8)
        )
        dist_to_goal = float(np.linalg.norm(final - goal))
        moved_dist = float(np.linalg.norm(final - cube))
        success = dist_to_goal <= self.GOAL_RADIUS
        object_moved = moved_dist > 0.005
        return AttemptResult(
            initial_obj_xy=tuple(float(v) for v in cube),     # type: ignore[arg-type]
            final_obj_xy=tuple(float(v) for v in final),       # type: ignore[arg-type]
            goal_xy=scene.goal_xy,
            reached_contact=True,
            object_moved=object_moved,
            planner_failed=False,
            collision=False,
            grasp_slip=False,
            rollout_log_path=None,
            success=success,
            trajectory_xy=synthetic_traj,
        )

    def close(self) -> None:
        pass


@pytest.fixture
def fake_env_runner() -> FakeEnvRunner:
    return FakeEnvRunner()


class FakePickEnvRunner:
    """Deterministic, sim-free env_runner for PickCube unit tests.

    Compiles the PickSkill (always succeeds — Stage-0 PickSkill never
    returns None). Stage-0 controlled-failure mechanism:
      - If `intent.contact_region in scene.blocked_sides` → grasp_slip:
        AttemptResult(grasp_slip=True, reached_contact=True,
                      object_moved=False, final_obj_xy=cube_xy,
                      success=False).
      - Else → successful lift: AttemptResult(success=True,
                                              reached_contact=True,
                                              object_moved=True,
                                              final_obj_xy=goal_xy).
    """

    def __init__(self) -> None:
        self._scenes_by_seed: dict[int, SceneState] = {}

    def reset(self, seed: int) -> SceneState:
        if seed not in self._scenes_by_seed:
            # Same deterministic scene generator as the PushCube fake.
            # PickCube cares about goal_xy (lift destination) so the seed
            # mapping is identical — the contact_region change is what
            # exercises BABYSTEPS, not the scene geometry.
            rng = np.random.default_rng(seed)
            r = float(rng.uniform(0.10, 0.18))
            theta = (seed % 4) * (np.pi / 2)
            goal_xy = (r * np.cos(theta), r * np.sin(theta))
            self._scenes_by_seed[seed] = SceneState(
                cube_xy=(0.0, 0.0),
                cube_z=0.02,
                goal_xy=(float(goal_xy[0]), float(goal_xy[1])),
                tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0),
                blocked_sides=(),
            )
        return self._scenes_by_seed[seed]

    def run(self, intent: Intent, scene: SceneState) -> AttemptResult:
        # Compile-time check: should never return None for PickSkill, but
        # be defensive so a future regression here doesn't silently pass.
        skill = compile_intent_to_pick_skill(intent, scene)
        assert skill is not None

        cube = np.asarray(scene.cube_xy, dtype=np.float64)
        goal = np.asarray(scene.goal_xy, dtype=np.float64)

        if intent.contact_region in scene.blocked_sides:
            # Stage-0 controlled grasp_slip. Synthetic trajectory: cube
            # rises briefly (not captured in 2D xy) then falls back.
            return AttemptResult(
                initial_obj_xy=tuple(float(v) for v in cube),     # type: ignore[arg-type]
                final_obj_xy=tuple(float(v) for v in cube),       # type: ignore[arg-type]
                goal_xy=scene.goal_xy,
                reached_contact=True,
                object_moved=False,
                planner_failed=False,
                collision=False,
                grasp_slip=True,
                rollout_log_path=None,
                success=False,
                trajectory_xy=(
                    tuple(float(v) for v in cube),     # start
                    tuple(float(v) for v in cube),     # end (z lift not in 2D)
                ),
            )

        # Successful pick + lift: cube ends at goal_xy.
        synthetic_traj = tuple(
            (float(cube[0] + (goal[0] - cube[0]) * t),
             float(cube[1] + (goal[1] - cube[1]) * t))
            for t in np.linspace(0.0, 1.0, 8)
        )
        return AttemptResult(
            initial_obj_xy=tuple(float(v) for v in cube),     # type: ignore[arg-type]
            final_obj_xy=tuple(float(v) for v in goal),        # type: ignore[arg-type]
            goal_xy=scene.goal_xy,
            reached_contact=True,
            object_moved=True,
            planner_failed=False,
            collision=False,
            grasp_slip=False,
            rollout_log_path=None,
            success=True,
            trajectory_xy=synthetic_traj,
        )

    def close(self) -> None:
        pass


@pytest.fixture
def fake_pick_env_runner() -> FakePickEnvRunner:
    return FakePickEnvRunner()


@pytest.fixture
def collect_main():
    """Import scripts/stage0_collect.py's `main` function fresh.

    Deletes any prior `stage0_collect` from sys.modules first so each test
    gets a clean import. Needed because stage0_collect.py mutates sys.path
    at import time — without the cache-clear, a second invocation in the
    same test session reuses the cached module."""
    import importlib
    # Ensure scripts/ is importable. _ROOT is conftest's repo root.
    scripts_dir = pathlib.Path(__file__).resolve().parent.parent / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    if "stage0_collect" in sys.modules:
        del sys.modules["stage0_collect"]
    return importlib.import_module("stage0_collect").main


@pytest.fixture
def correct_intent_for_scene():
    """Helper: build the *correct* intent for a given scene (oracle access)."""
    def _build(scene: SceneState) -> Intent:
        goal_vec = np.array(scene.goal_xy) - np.array(scene.cube_xy)
        face = direction_to_face(goal_vec)
        approach = {
            "minus_x_face": "from_minus_x",
            "plus_x_face":  "from_plus_x",
            "minus_y_face": "from_minus_y",
            "plus_y_face":  "from_plus_y",
        }[face]
        # Match object_motion to goal direction (dominant axis sign).
        if abs(goal_vec[0]) >= abs(goal_vec[1]):
            motion = "translate_+x" if goal_vec[0] >= 0 else "translate_-x"
        else:
            motion = "translate_+y" if goal_vec[1] >= 0 else "translate_-y"
        return Intent(
            goal_state="cube_at_target",
            object_motion=motion,
            contact_region=face,
            approach_direction=approach,
            constraint_region="none",
            embodiment_mapping="proxy_contact_to_franka_push",
        )
    return _build
