"""TurnFaucet-v1 adapter — the fourth concrete BaseTaskAdapter.

Pulls every TurnFaucet-specific decision behind one class:
  * make_env_runner       → TurnFaucetEnvRunner (Task 6)
  * oracle_correct_intent → handle_grip / faucet_base_static / turn
  * default_blocked_factory → () — no physical blocking; the Stage-0
                              controlled failure is the under-specified
                              (contact_region, constraint_region) pair
                              in scripted_demo_to_intent
  * oracle_wrong_factor   → "constraint_region" if intent.contact_region
                            == "faucet_base", else "none"
  * scripted_demo_to_intent → DELIBERATELY returns (contact_region=
                              "faucet_base", constraint_region="none").
                              The 2D summarizer can't distinguish the
                              handle from the body.
  * compile_skill         → wraps skills.turn.compile_intent_to_turn_skill

Hook defaults (build_failure_packet / attribute_failure / revise_intent)
are inherited unchanged from BaseTaskAdapter — the new
constraint_violation predicate (Task 2) and constraint_introduction
operator (Task 3) live in failure.py / revision.py and dispatch
automatically."""
from __future__ import annotations

from babysteps.envs.task_adapter import BaseTaskAdapter, EnvRunner
from babysteps.schemas import DemoEvidence, Intent, SceneState
from babysteps.skills.turn import compile_intent_to_turn_skill


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
            constraint_region="faucet_base_static",
            embodiment_mapping="proxy_contact_to_franka_turn",
        )

    def default_blocked_factory(self, intent: Intent) -> tuple[str, ...]:
        return ()

    def oracle_wrong_factor(
        self, initial_intent: Intent, scene_executor: SceneState,
    ) -> str:
        if initial_intent.contact_region == "faucet_base":
            return "constraint_region"
        return "none"

    def scripted_demo_to_intent(self, evidence: DemoEvidence) -> Intent:
        # Deliberately ignores evidence.contact_region_label —
        # the 2D summarizer can't distinguish the handle from the body.
        return Intent(
            goal_state="faucet_turned",
            object_motion="turn",
            contact_region="faucet_base",      # DELIBERATELY under-specified
            approach_direction="from_above",
            constraint_region="none",           # DELIBERATELY missing
            embodiment_mapping="proxy_contact_to_franka_turn",
        )

    def compile_skill(self, intent: Intent, scene: SceneState):
        return compile_intent_to_turn_skill(intent, scene)
