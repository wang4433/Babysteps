"""Tests for babysteps.envs.task_adapter — the BaseTaskAdapter ABC and the
EnvRunner Protocol that every concrete adapter implements."""
from __future__ import annotations

import pytest

from babysteps.envs.task_adapter import BaseTaskAdapter, EnvRunner
from babysteps.failure import Attribution
from babysteps.schemas import (
    AttemptResult, DemoEvidence, FailurePacket, Intent, Revision, SceneState,
)


# ---------- real runners honour the EnvRunner Protocol ------------------ #


def test_real_runners_accept_rollout_seed():
    """Every concrete real runner's run() must accept the keyword-only
    `rollout_seed` declared by the EnvRunner Protocol — episode.run_episode
    passes it on every retry. Sim-free CI uses fake runners, so this inspect
    guard is what catches a real-runner signature drift before a GPU job does.
    """
    import inspect

    from babysteps.envs.crossview_runner import CrossViewPushEnvRunner
    from babysteps.envs.pickcube_runner import PickCubeEnvRunner
    from babysteps.envs.pushcube_runner import PushCubeEnvRunner
    from babysteps.envs.stackcube_runner import StackCubeEnvRunner
    from babysteps.envs.turnfaucet_runner import TurnFaucetEnvRunner

    for cls in (
        PushCubeEnvRunner, PickCubeEnvRunner, StackCubeEnvRunner,
        TurnFaucetEnvRunner, CrossViewPushEnvRunner,
    ):
        params = inspect.signature(cls.run).parameters
        assert "rollout_seed" in params, (
            f"{cls.__name__}.run is missing the rollout_seed keyword required "
            f"by the EnvRunner Protocol"
        )
        assert params["rollout_seed"].kind is inspect.Parameter.KEYWORD_ONLY


# ---------- BaseTaskAdapter is abstract --------------------------------- #


def test_base_adapter_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseTaskAdapter()  # type: ignore[abstract]


def test_partial_subclass_cannot_be_instantiated():
    class HalfAdapter(BaseTaskAdapter):
        task_id = "TestTask-v0"
        def make_env_runner(self):  # noqa: D401
            raise NotImplementedError
        # Missing: oracle_correct_intent, default_blocked_factory,
        # oracle_wrong_factor, scripted_demo_to_intent.
    with pytest.raises(TypeError):
        HalfAdapter()  # type: ignore[abstract]


# ---------- env_runner caching + close ---------------------------------- #


class _CountingAdapter(BaseTaskAdapter):
    """Subclass that tracks how many times make_env_runner is invoked."""
    task_id = "CountTask-v0"

    def __init__(self):
        super().__init__()
        self.make_calls = 0

    def make_env_runner(self):
        self.make_calls += 1
        class _Runner:
            def reset(self, seed): return _ok_scene()
            def run(self, intent, scene): return _ok_attempt()
            def close(self): pass
        return _Runner()

    def oracle_correct_intent(self, scene): return _ok_intent()
    def default_blocked_factory(self, intent): return ()
    def oracle_wrong_factor(self, intent, scene): return "none"
    def scripted_demo_to_intent(self, evidence): return _ok_intent()


def test_env_runner_caches_after_first_call():
    a = _CountingAdapter()
    assert a.make_calls == 0
    r1 = a.env_runner()
    r2 = a.env_runner()
    r3 = a.env_runner()
    assert a.make_calls == 1
    assert r1 is r2 is r3


def test_close_idempotent_and_releases_runner():
    a = _CountingAdapter()
    a.env_runner()       # construct
    assert a._env_runner is not None
    a.close()
    assert a._env_runner is None
    a.close()            # second call is a no-op
    assert a._env_runner is None


def test_close_then_env_runner_reallocates():
    a = _CountingAdapter()
    a.env_runner()
    a.close()
    a.env_runner()
    assert a.make_calls == 2


# ---------- Concrete stub used to exercise the hook defaults ------------ #


class _StubAdapter(BaseTaskAdapter):
    task_id = "StubTask-v0"

    def make_env_runner(self):
        raise NotImplementedError

    def oracle_correct_intent(self, scene):
        return _ok_intent()

    def default_blocked_factory(self, intent):
        return ()

    def oracle_wrong_factor(self, intent, scene):
        return "none"

    def scripted_demo_to_intent(self, evidence):
        return _ok_intent()


def _ok_intent() -> Intent:
    return Intent(
        goal_state="cube_at_target",
        object_motion="translate_+x",
        contact_region="minus_x_face",
        approach_direction="from_minus_x",
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_push",
    )


def _ok_scene() -> SceneState:
    return SceneState(
        cube_xy=(0.0, 0.0), cube_z=0.02, goal_xy=(0.2, 0.0),
        tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0),
        blocked_sides=("from_minus_x",),
    )


def _ok_attempt(success: bool = False) -> AttemptResult:
    return AttemptResult(
        initial_obj_xy=(0.0, 0.0), final_obj_xy=(0.0, 0.0), goal_xy=(0.2, 0.0),
        reached_contact=False, object_moved=False,
        planner_failed=True, collision=False, grasp_slip=False,
        rollout_log_path=None, success=success,
    )


# ---------- Hook defaults delegate to shared modules -------------------- #


def test_build_failure_packet_default_delegates(monkeypatch):
    """The default hook must call babysteps.failure.build_failure_packet."""
    import babysteps.failure as failure_mod
    calls = []
    original = failure_mod.build_failure_packet

    def spy(intent, attempt, scene):
        calls.append((intent, attempt, scene))
        return original(intent, attempt, scene)

    monkeypatch.setattr(failure_mod, "build_failure_packet", spy)

    fp = _StubAdapter().build_failure_packet(_ok_intent(), _ok_attempt(), _ok_scene())
    assert isinstance(fp, FailurePacket)
    assert len(calls) == 1
    assert calls[0][0] == _ok_intent()


def test_attribute_failure_default_delegates(monkeypatch):
    import babysteps.failure as failure_mod
    calls = []
    original = failure_mod.attribute_failure

    def spy(fp):
        calls.append(fp)
        return original(fp)

    monkeypatch.setattr(failure_mod, "attribute_failure", spy)

    fp = _StubAdapter().build_failure_packet(_ok_intent(), _ok_attempt(), _ok_scene())
    attr = _StubAdapter().attribute_failure(fp)
    assert isinstance(attr, Attribution)
    assert len(calls) == 1


def test_revise_intent_default_delegates(monkeypatch):
    import babysteps.revision as revision_mod
    calls = []
    original = revision_mod.revise_intent

    def spy(intent, attribution, scene):
        calls.append((intent, attribution, scene))
        return original(intent, attribution, scene)

    monkeypatch.setattr(revision_mod, "revise_intent", spy)

    fp = _StubAdapter().build_failure_packet(_ok_intent(), _ok_attempt(), _ok_scene())
    attr = _StubAdapter().attribute_failure(fp)
    revised, rev = _StubAdapter().revise_intent(_ok_intent(), attr, _ok_scene())
    assert isinstance(revised, Intent)
    assert isinstance(rev, Revision)
    assert len(calls) == 1


# ---------- EnvRunner Protocol -------------------------------------------- #


def test_env_runner_protocol_has_required_methods():
    """Structural typing — anything with reset/run/close satisfies it."""
    class _StubRunner:
        def reset(self, seed): return _ok_scene()
        def run(self, intent, scene): return _ok_attempt()
        def close(self): pass

    runner: EnvRunner = _StubRunner()   # type-check only; no runtime check
    assert runner.reset(0) == _ok_scene()


# ---------- __init_subclass__ enforces task_id at class-definition time --- #


def test_subclass_without_task_id_raises_at_definition():
    """task_id is enforced at __init_subclass__, not at instantiation —
    catches the omission before any episode actually runs."""
    with pytest.raises(TypeError, match="task_id"):
        class _NoIdAdapter(BaseTaskAdapter):
            # task_id intentionally omitted
            def make_env_runner(self): raise NotImplementedError
            def oracle_correct_intent(self, scene): return _ok_intent()
            def default_blocked_factory(self, intent): return ()
            def oracle_wrong_factor(self, initial_intent, scene_executor): return "none"
            def scripted_demo_to_intent(self, evidence): return _ok_intent()


def test_base_task_valid_tokens_defaults_empty():
    from babysteps.envs.task_adapter import BaseTaskAdapter

    class _Stub(BaseTaskAdapter):
        task_id = "PushCube-v1"
        def make_env_runner(self): raise NotImplementedError
        def oracle_correct_intent(self, scene): raise NotImplementedError
        def default_blocked_factory(self, intent): return ()
        def oracle_wrong_factor(self, i, s): return "none"
        def scripted_demo_to_intent(self, e): raise NotImplementedError

    assert _Stub().task_valid_tokens() == {}
