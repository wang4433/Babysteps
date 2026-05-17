"""Stage-0 task dispatch table.

The single source of truth that maps a `--task` CLI arg to:
  * the concrete BaseTaskAdapter subclass to instantiate,
  * a factory for the deterministic sim-free fake env_runner (used by
    `--fake-env` and by the end-to-end snapshot tests),
  * the episode_id prefix that names this task's Stage-0 controlled
    failure (e.g. "pushcube_blocked_approach" vs "pickcube_grasp_slip").

The episode_id_prefix is held here (not on the adapter) because it's a
Stage-0 cosmetic naming convention tied to the controlled-failure
mechanism, not a task-semantic decision the adapter cares about.

Adding a new Stage-0 task (Sub-projects C, D) is a one-entry addition
here plus the corresponding render module entry in babysteps.render.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from babysteps.envs.task_adapter import BaseTaskAdapter, EnvRunner


@dataclass(frozen=True)
class TaskEntry:
    """One row of the Stage-0 task dispatch table."""
    adapter_cls: type[BaseTaskAdapter]
    fake_runner_factory: Callable[[], EnvRunner]
    episode_id_prefix: str


def _pushcube_entry() -> TaskEntry:
    # PushCubeAdapter is safe to import at module load — it does not pull
    # mani_skill (that import only fires when adapter.make_env_runner()
    # constructs the real runner, which the registry itself never does).
    # FakeEnvRunner is genuinely lazy via the _make_fake closure below so
    # tests/ stays off the import path until --fake-env is selected.
    from babysteps.envs.pushcube_adapter import PushCubeAdapter

    def _make_fake() -> EnvRunner:
        # FakeEnvRunner lives in tests/conftest.py; the registry deliberately
        # references it so the same code-path that powers `--fake-env` is the
        # one the snapshot tests exercise. The import is local to keep
        # tests/ off the runtime import path.
        from tests.conftest import FakeEnvRunner
        return FakeEnvRunner()

    return TaskEntry(
        adapter_cls=PushCubeAdapter,
        fake_runner_factory=_make_fake,
        episode_id_prefix="pushcube_blocked_approach",
    )


def _pickcube_entry() -> TaskEntry:
    # PickCubeAdapter is safe to import at module load — it does not pull
    # mani_skill (that import only fires when adapter.make_env_runner()
    # constructs the real runner, which the registry itself never does).
    # FakePickEnvRunner is genuinely lazy via the _make_fake closure below so
    # tests/ stays off the import path until --fake-env is selected.
    from babysteps.envs.pickcube_adapter import PickCubeAdapter

    def _make_fake() -> EnvRunner:
        from tests.conftest import FakePickEnvRunner
        return FakePickEnvRunner()

    return TaskEntry(
        adapter_cls=PickCubeAdapter,
        fake_runner_factory=_make_fake,
        episode_id_prefix="pickcube_grasp_slip",
    )


def _stackcube_entry() -> TaskEntry:
    # StackCubeAdapter is safe to import at module load — it does not
    # pull mani_skill (deferred to make_env_runner()). FakeStackCube-
    # EnvRunner is lazy via the _make_fake closure.
    from babysteps.envs.stackcube_adapter import StackCubeAdapter

    def _make_fake() -> EnvRunner:
        from tests.conftest import FakeStackCubeEnvRunner
        return FakeStackCubeEnvRunner()

    return TaskEntry(
        adapter_cls=StackCubeAdapter,
        fake_runner_factory=_make_fake,
        episode_id_prefix="stackcube_underspec_goal",
    )


TASK_REGISTRY: dict[str, TaskEntry] = {
    "PushCube-v1": _pushcube_entry(),
    "PickCube-v1": _pickcube_entry(),
    "StackCube-v1": _stackcube_entry(),
}


def get_task_entry(task_id: str) -> TaskEntry:
    """Return the TaskEntry for `task_id` or raise KeyError listing the
    known tasks (handy when the user mistypes `--task`)."""
    if task_id not in TASK_REGISTRY:
        known = sorted(TASK_REGISTRY.keys())
        raise KeyError(
            f"unknown task {task_id!r}; known tasks: {known}"
        )
    return TASK_REGISTRY[task_id]
