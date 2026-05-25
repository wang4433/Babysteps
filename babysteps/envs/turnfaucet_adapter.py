"""TurnFaucet-v1 adapter — the fourth concrete BaseTaskAdapter.

Pulls every TurnFaucet-specific decision behind one class:
  * make_env_runner       → TurnFaucetEnvRunner (Task 6)
  * oracle_correct_intent → handle_grip / none / poke_turn
                            (the mechanically feasible mapping)
  * default_blocked_factory → () — no physical blocking; the Stage-0
                              controlled failure is the infeasible
                              embodiment_mapping in scripted_demo_to_intent
  * oracle_wrong_factor   → "embodiment_mapping" if
                            intent.embodiment_mapping ==
                            "proxy_contact_to_franka_grasp_turn",
                            else "none"
  * scripted_demo_to_intent → DELIBERATELY returns
                              embodiment_mapping="proxy_contact_to_franka_grasp_turn".
                              The 2D summarizer observes "hand-like interaction"
                              and encodes grasp; it cannot know that the Franka
                              cannot mechanically envelop the faucet handle.

Hook defaults (build_failure_packet / attribute_failure / revise_intent)
are inherited unchanged from BaseTaskAdapter — the grasp_infeasible predicate
(Task 5-6) and embodiment_substitution operator (Task 7) live in failure.py /
revision.py and dispatch automatically."""
from __future__ import annotations

from babysteps.envs.task_adapter import BaseTaskAdapter, EnvRunner
from babysteps.schemas import DemoEvidence, Intent, SceneState


class TurnFaucetAdapter(BaseTaskAdapter):
    task_id = "TurnFaucet-v1"

    def make_env_runner(self) -> EnvRunner:
        from babysteps.envs.turnfaucet_runner import TurnFaucetEnvRunner
        return TurnFaucetEnvRunner()

    def oracle_correct_intent(self, scene: SceneState) -> Intent:
        return Intent(
            goal_state="faucet_turned",
            object_motion="turn",
            contact_region="handle_grip",
            approach_direction="from_above",
            constraint_region="none",
            embodiment_mapping="proxy_contact_to_franka_poke_turn",
        )

    def default_blocked_factory(self, intent: Intent) -> tuple[str, ...]:
        return ()

    def oracle_wrong_factor(
        self,
        initial_intent: Intent,
        scene_executor: SceneState | None = None,
    ) -> str:
        if initial_intent.embodiment_mapping == "proxy_contact_to_franka_grasp_turn":
            return "embodiment_mapping"
        return "none"

    def scripted_demo_to_intent(self, evidence: DemoEvidence) -> Intent:
        # Stage-0 information loss: the demo's hand-like interaction
        # symbolically reads as grasping; the 2D summarizer cannot know that
        # the Franka cannot mechanically execute it.
        return Intent(
            goal_state="faucet_turned",
            object_motion="turn",
            contact_region="handle_grip",
            approach_direction="from_above",
            constraint_region="none",
            embodiment_mapping="proxy_contact_to_franka_grasp_turn",  # DELIBERATELY infeasible
        )
