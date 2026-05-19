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
