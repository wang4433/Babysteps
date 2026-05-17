"""StackCube-v1 adapter — the third concrete BaseTaskAdapter.

Pulls every StackCube-specific decision behind one class:
  * make_env_runner       → StackCubeEnvRunner (Task 5)
  * oracle_correct_intent → cubeA_on_cubeB / place_on / pick-and-place
  * default_blocked_factory → () — no physical blocking; the Stage-0
                              controlled failure is from wrong-goal
                              waypoints, not from blocked_sides
  * oracle_wrong_factor   → "goal_state" if intent.goal_state ==
                            "cube_at_target", else "none"
  * scripted_demo_to_intent → DELIBERATELY returns goal_state=
                              "cube_at_target". The 2D trajectory
                              summarization can't see vertical motion,
                              so the demo's true stacking is hidden.
  * compile_skill         → wraps skills.stack.compile_intent_to_stack_skill

Hook defaults (build_failure_packet / attribute_failure / revise_intent)
are inherited unchanged from BaseTaskAdapter — the goal_refinement
operator (Task 2) lives in revision.py and dispatches on
attribution.wrong_factor='goal_state' automatically."""
from __future__ import annotations

import numpy as np

from babysteps.envs.task_adapter import BaseTaskAdapter, EnvRunner
from babysteps.schemas import CONTACT_REGIONS, DemoEvidence, Intent, SceneState
from babysteps.skills.stack import compile_intent_to_stack_skill


# Default contact_region for the oracle — pick-and-place doesn't depend
# strongly on which face is grasped (parallel-jaw + lift); pick the same
# canonical value as PickCubeAdapter so the snapshot files have a
# consistent contact_region across tasks.
_DEFAULT_CONTACT_REGION: str = "minus_x_face"


def _dominant_axis_motion(traj: tuple[tuple[float, float], ...]) -> str:
    """Pick translate_<axis> from the (initial, final) trajectory's dominant
    component. Matches the convention used by PushCubeAdapter."""
    if len(traj) < 2:
        return "translate_+x"   # degenerate; arbitrary default
    dx = traj[-1][0] - traj[0][0]
    dy = traj[-1][1] - traj[0][1]
    if abs(dx) >= abs(dy):
        return "translate_+x" if dx >= 0 else "translate_-x"
    return "translate_+y" if dy >= 0 else "translate_-y"


class StackCubeAdapter(BaseTaskAdapter):
    task_id = "StackCube-v1"

    def make_env_runner(self) -> EnvRunner:
        from babysteps.envs.stackcube_runner import StackCubeEnvRunner
        return StackCubeEnvRunner()

    def oracle_correct_intent(self, scene: SceneState) -> Intent:
        return Intent(
            goal_state="cubeA_on_cubeB",
            object_motion="place_on",
            contact_region=_DEFAULT_CONTACT_REGION,
            approach_direction="from_above",
            constraint_region="none",
            embodiment_mapping="proxy_contact_to_franka_pick_and_place",
        )

    def default_blocked_factory(self, intent: Intent) -> tuple[str, ...]:
        # No physical blocking — the Stage-0 controlled failure comes
        # from the under-specified goal_state in scripted_demo_to_intent.
        return ()

    def oracle_wrong_factor(
        self, initial_intent: Intent, scene_executor: SceneState,
    ) -> str:
        if initial_intent.goal_state == "cube_at_target":
            return "goal_state"
        return "none"

    def scripted_demo_to_intent(self, evidence: DemoEvidence) -> Intent:
        contact = evidence.contact_region_label
        if contact not in CONTACT_REGIONS:
            raise ValueError(
                f"DemoEvidence.contact_region_label must be one of "
                f"{sorted(CONTACT_REGIONS)}, got {contact!r}"
            )
        motion = _dominant_axis_motion(evidence.object_trajectory)
        return Intent(
            goal_state="cube_at_target",   # DELIBERATELY under-specified
            object_motion=motion,
            contact_region=contact,
            approach_direction="from_above",
            constraint_region="none",
            embodiment_mapping="proxy_contact_to_franka_pick_and_place",
        )

    def compile_skill(self, intent: Intent, scene: SceneState):
        return compile_intent_to_stack_skill(intent, scene)
