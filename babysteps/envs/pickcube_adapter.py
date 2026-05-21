"""PickCube-v1 adapter — the second concrete BaseTaskAdapter.

Pulls every PickCube-specific decision behind one class:
  * make_env_runner       → PickCubeEnvRunner (Sub-project B Task 12)
  * oracle_correct_intent → top-down grasp with the first unblocked face
  * default_blocked_factory → (intent.contact_region,) — the executor
                              flags the demonstrated gripper-axis as
                              slip-prone, the Stage-0 controlled failure
  * oracle_wrong_factor   → "contact_region" if the intent's contact
                            face is in blocked_sides, else "none"
  * scripted_demo_to_intent → contact_region_label → Intent (object_motion
                              and approach_direction are PickCube-fixed:
                              "lift_up" and "from_above")
  * compile_skill         → wraps skills.pick.compile_intent_to_pick_skill

Hook defaults (build_failure_packet / attribute_failure / revise_intent)
are inherited unchanged from BaseTaskAdapter — the new `grasp_slip`
predicate, `contact_substitution` operator, and FAILURE_TO_FACTOR row
all live in the shared modules (failure.py, revision.py)."""
from __future__ import annotations

from babysteps.envs.task_adapter import BaseTaskAdapter, EnvRunner
from babysteps.schemas import CONTACT_REGIONS, DemoEvidence, Intent, SceneState
from babysteps.skills.pick import compile_intent_to_pick_skill

# Deterministic fallback order for the oracle when several faces are
# unblocked. Matches the order used by revision._pick_unblocked_face so
# tests and traces line up.
_FACE_PREFERENCE_ORDER: tuple[str, ...] = (
    "minus_x_face", "plus_x_face", "minus_y_face", "plus_y_face",
)


class PickCubeAdapter(BaseTaskAdapter):
    task_id = "PickCube-v1"

    def make_env_runner(self) -> EnvRunner:
        from babysteps.envs.pickcube_runner import PickCubeEnvRunner
        return PickCubeEnvRunner()

    def oracle_correct_intent(self, scene: SceneState) -> Intent:
        blocked = set(scene.blocked_sides)
        contact = next(
            (f for f in _FACE_PREFERENCE_ORDER if f not in blocked),
            None,
        )
        if contact is None:
            raise RuntimeError(
                f"oracle_correct_intent: every cardinal face is blocked "
                f"({sorted(blocked)!r}); no graspable gripper-axis exists"
            )
        return Intent(
            goal_state="cube_lifted_at_target",
            object_motion="lift_up",
            contact_region=contact,
            approach_direction="from_above",
            constraint_region="none",
            embodiment_mapping="proxy_contact_to_franka_grasp",
        )

    def default_blocked_factory(self, intent: Intent) -> tuple[str, ...]:
        return (intent.contact_region,)

    def oracle_wrong_factor(
        self, initial_intent: Intent, scene_executor: SceneState,
    ) -> str:
        if initial_intent.contact_region in scene_executor.blocked_sides:
            return "contact_region"
        return "none"

    def scripted_demo_to_intent(self, evidence: DemoEvidence) -> Intent:
        contact = evidence.contact_region_label
        if contact not in CONTACT_REGIONS:
            raise ValueError(
                f"DemoEvidence.contact_region_label must be one of "
                f"{sorted(CONTACT_REGIONS)}, got {contact!r}"
            )
        return Intent(
            goal_state="cube_lifted_at_target",
            object_motion="lift_up",
            contact_region=contact,
            approach_direction="from_above",
            constraint_region="none",
            embodiment_mapping="proxy_contact_to_franka_grasp",
        )

    def compile_skill(self, intent: Intent, scene: SceneState):
        return compile_intent_to_pick_skill(intent, scene)

    def task_valid_tokens(self) -> dict[str, tuple[str, ...]]:
        # PickCube's controlled fault is a slip-prone contact face.
        return {
            "contact_region": (
                "minus_x_face", "plus_x_face", "minus_y_face", "plus_y_face",
            ),
        }
