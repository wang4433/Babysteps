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
