"""PushCube-v1 adapter — the first concrete BaseTaskAdapter.

Pulls every PushCube-specific decision behind one class:
  * make_env_runner       → PushCubeEnvRunner
  * oracle_correct_intent → uses goal direction to derive face/approach/motion
  * default_blocked_factory → (intent.approach_direction,)
  * oracle_wrong_factor   → "approach_direction" if intent's approach is in
                            blocked_sides, else "none"
  * scripted_demo_to_intent → contact_region_label + trajectory → Intent

Hook defaults (build_failure_packet / attribute_failure / revise_intent) are
inherited unchanged from BaseTaskAdapter."""
from __future__ import annotations

import numpy as np

from babysteps.demo import trajectory_to_motion
from babysteps.envs.scene import direction_to_face, face_to_approach
from babysteps.envs.task_adapter import BaseTaskAdapter, EnvRunner
from babysteps.schemas import CONTACT_REGIONS, DemoEvidence, Intent, SceneState


class PushCubeAdapter(BaseTaskAdapter):
    task_id = "PushCube-v1"

    def make_env_runner(self) -> EnvRunner:
        from babysteps.envs.pushcube_runner import PushCubeEnvRunner
        return PushCubeEnvRunner()

    def oracle_correct_intent(self, scene: SceneState) -> Intent:
        goal_vec = np.array(scene.goal_xy) - np.array(scene.cube_xy)
        face = direction_to_face(goal_vec)
        approach = face_to_approach(face)
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

    def default_blocked_factory(self, intent: Intent) -> tuple[str, ...]:
        return (intent.approach_direction,)

    def oracle_wrong_factor(
        self, initial_intent: Intent, scene_executor: SceneState,
    ) -> str:
        if initial_intent.approach_direction in scene_executor.blocked_sides:
            return "approach_direction"
        return "none"

    def scripted_demo_to_intent(self, evidence: DemoEvidence) -> Intent:
        contact = evidence.contact_region_label
        if contact not in CONTACT_REGIONS:
            raise ValueError(
                f"DemoEvidence.contact_region_label must be one of "
                f"{sorted(CONTACT_REGIONS)}, got {contact!r}"
            )
        motion = trajectory_to_motion(evidence.object_trajectory)
        return Intent(
            goal_state="cube_at_target",
            object_motion=motion,
            contact_region=contact,
            approach_direction=face_to_approach(contact),
            constraint_region="none",
            embodiment_mapping="proxy_contact_to_franka_push",
        )

    def task_valid_tokens(self) -> dict[str, tuple[str, ...]]:
        return {
            "approach_direction": (
                "from_minus_x", "from_plus_x", "from_minus_y", "from_plus_y",
            ),
            "contact_region": (
                "minus_x_face", "plus_x_face", "minus_y_face", "plus_y_face",
            ),
        }
