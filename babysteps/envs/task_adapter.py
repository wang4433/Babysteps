"""TaskAdapter interface — the boundary that the episode loop and CLI scripts
sit behind, so they don't depend on any concrete task.

Sub-projects B (PickCube) and C (StackCube) each add one concrete adapter
without touching the loop or the orchestration code.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Protocol

from babysteps.failure import Attribution
from babysteps.schemas import (
    AttemptResult,
    DemoEvidence,
    FailurePacket,
    Intent,
    Revision,
    SceneState,
)


class EnvRunner(Protocol):
    """Minimal env_runner contract. Implementations: the fake in
    tests/conftest.py and the real ManiSkill PushCubeEnvRunner. Future tasks
    add their own runners; the adapter constructs them in make_env_runner.

    This is the canonical EnvRunner Protocol; episode.run_episode imports
    it from here. (It was relocated from babysteps.episode in Plan Task 6.)"""

    def reset(self, seed: int) -> SceneState: ...
    def run(self, intent: Intent, scene: SceneState) -> AttemptResult: ...
    def close(self) -> None: ...


class BaseTaskAdapter(ABC):
    """One concrete adapter per ManiSkill task. `episode.run_episode` and the
    CLI scripts depend ONLY on this base class.

    Six abstract methods are task-specific (must be implemented). Three
    overridable hooks (`build_failure_packet`, `attribute_failure`,
    `revise_intent`) default to the shared modules; override only when the
    task's failure precedence or revision operators differ.

    Two concrete lifecycle helpers (`env_runner`, `close`) cache the env
    runner so the per-episode allocation cost stays at one — the same as
    pre-A — even though the episode loop calls `env_runner()` every
    iteration."""

    task_id: str  # subclasses set, e.g. "PushCube-v1"

    @property
    def gym_env_id(self) -> str:
        """Underlying ManiSkill gym env id for gym.make(). Defaults to task_id;
        override when the logical task (registry key) differs from the gym env
        (e.g. CrossViewPush is a logical task running on the PushCube-v1 env)."""
        return self.task_id

    def __init_subclass__(cls, **kw: object) -> None:
        super().__init_subclass__(**kw)
        if not isinstance(getattr(cls, "task_id", None), str):
            raise TypeError(
                f"{cls.__name__} must define task_id as a class-level str "
                f"(e.g., task_id = 'PushCube-v1')"
            )

    # ---- env_runner lifecycle (concrete; cached) ------------------------- #

    def __init__(self) -> None:
        self._env_runner: Optional[EnvRunner] = None

    def env_runner(self) -> EnvRunner:
        """Lazily construct (via make_env_runner) and cache the runner. The
        episode loop calls this once per episode; caching preserves the pre-A
        one-runner-across-all-seeds perf for expensive real-sim runners."""
        if self._env_runner is None:
            self._env_runner = self.make_env_runner()
        return self._env_runner

    def close(self) -> None:
        """Release the cached runner if any. Idempotent."""
        if self._env_runner is not None:
            self._env_runner.close()
            self._env_runner = None

    # ---- abstract: must be implemented per task --------------------------- #

    @abstractmethod
    def make_env_runner(self) -> EnvRunner:
        """Construct the concrete env_runner for this task. Called at most
        once per adapter instance — env_runner() caches the result."""

    @abstractmethod
    def oracle_correct_intent(self, scene: SceneState) -> Intent:
        """Read privileged scene state and produce the ground-truth intent
        that would succeed on this scene. Used inside generate_proxy_demo —
        NOT in the privileged-firewalled `scripted_demo_to_intent` path."""

    @abstractmethod
    def default_blocked_factory(self, intent: Intent) -> tuple[str, ...]:
        """Given the initial intent, return the privileged blocked_sides
        tuple that produces a Stage-0 controlled semantic failure."""

    @abstractmethod
    def oracle_wrong_factor(
        self, initial_intent: Intent, scene_executor: SceneState,
    ) -> str:
        """Return the name of the intent factor that is wrong under the
        executor's blocked_sides. 'none' if no factor is wrong."""

    @abstractmethod
    def scripted_demo_to_intent(self, evidence: DemoEvidence) -> Intent:
        """Privileged-firewalled scripted intent extraction. Takes ONLY a
        DemoEvidence — never a SceneState. Future stages replace this with
        DINO/VLM grounding."""

    @abstractmethod
    def compile_skill(self, intent: Intent, scene: SceneState) -> Any:
        """Compile the intent + scene into an executable skill object. The
        env_runner consumes whatever this returns. Returns None when the
        intent is infeasible (e.g., approach_direction in blocked_sides) —
        None propagates as planner_failed=True downstream."""

    # ---- overridable hooks: default delegates to shared modules ----------- #

    def build_failure_packet(
        self, intent: Intent, attempt: AttemptResult, scene: SceneState,
    ) -> FailurePacket:
        """Build the structured failure packet from the attempt. Override when
        this task's predicate precedence differs from the shared rule."""
        from babysteps import failure as failure_mod
        return failure_mod.build_failure_packet(intent, attempt, scene)

    def attribute_failure(self, fp: FailurePacket) -> Attribution:
        """Map a failure_predicate to the implicated intent factor. Override
        when this task has predicate→factor mappings beyond the shared table."""
        from babysteps import failure as failure_mod
        return failure_mod.attribute_failure(fp)

    def revise_intent(
        self, intent: Intent, attribution: Attribution, scene: SceneState,
    ) -> tuple[Intent, Revision]:
        """Produce a (revised_intent, Revision) pair. Override when this task
        introduces a new REVISION_OPERATORS entry."""
        from babysteps import revision as revision_mod
        return revision_mod.revise_intent(intent, attribution, scene)

    def observe_demo(
        self,
        object_trajectory: tuple[tuple[float, float], ...],
        correct_intent: Intent,
        scene: SceneState,
    ) -> tuple[tuple[tuple[float, float], ...], str]:
        """How the demo is *observed* before intent extraction. Default is
        identity: the proxy demo is observed in the same frame it was executed.
        Returns (object_trajectory, contact_region_label). Override for tasks
        whose demo view differs from the execution view (cross-view)."""
        return object_trajectory, correct_intent.contact_region

    def task_valid_tokens(self) -> dict[str, tuple[str, ...]]:
        """Per-factor task-valid alternative tokens for baseline resampling.

        Keys are the *task-editable* factors (those with >1 plausible token
        for this task). Values are the plausible tokens. Default: no editable
        factors. The three main-table adapters override. Used only by the M3
        baseline policies; the selective loop never calls this."""
        return {}
