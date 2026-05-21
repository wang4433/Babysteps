"""Integration test for the Stage-0 episode loop with a fake env_runner.

After Sub-project A, run_episode takes an adapter. This file uses a stub
adapter wired around the deterministic FakeEnvRunner (conftest fixture)."""
from __future__ import annotations

import inspect
from dataclasses import replace

from babysteps.episode import run_episode
from babysteps.schemas import CLAIM_BOUNDARY, DemoEvidence, EpisodeRecord


# Stub adapter that uses the conftest FakeEnvRunner via injection. We avoid
# importing PushCubeAdapter here so this test stays decoupled from any one
# concrete adapter.
def _make_stub_adapter(fake_runner, *, blocked_factory=None):
    from babysteps.envs.pushcube_adapter import PushCubeAdapter

    class _StubAdapter(PushCubeAdapter):
        def make_env_runner(self):
            return fake_runner
        if blocked_factory is not None:
            def default_blocked_factory(self, intent):
                return blocked_factory(intent)
    return _StubAdapter()


def test_run_episode_blocked_then_retry_success(fake_env_runner):
    adapter = _make_stub_adapter(fake_env_runner)
    rec = run_episode(
        episode_id="pushcube_blocked_approach_seed_0000",
        seed=0,
        adapter=adapter,
    )
    assert isinstance(rec, EpisodeRecord)
    assert rec.task == "PushCube-v1"
    assert rec.claim_boundary == CLAIM_BOUNDARY
    assert rec.demo["demonstrator_type"] == "proxy_oracle"
    assert rec.execution["success"] is False
    assert rec.failure_packet["failure_predicate"] == "approach_blocked"
    assert rec.failure_packet["wrong_factor"] == "approach_direction"
    assert rec.revision is not None
    assert rec.revision["operator"] == "approach_substitution"
    assert rec.revision["factor"] == "approach_direction"
    assert rec.retry is not None
    assert rec.retry["success"] is True
    m = rec.metrics
    assert m["initial_success"] is False
    assert m["retry_success"] is True
    assert m["num_attempts_to_success"] == 2
    assert m["factor_attribution_correct"] is True
    assert m["frozen_factors_preserved"] is True
    assert m["factors_changed"] == ["approach_direction"]


def test_run_episode_top_level_keys_match_goal_md(fake_env_runner):
    rec = run_episode(
        episode_id="pushcube_blocked_approach_seed_0000",
        seed=0,
        adapter=_make_stub_adapter(fake_env_runner),
    )
    d = rec.to_dict()
    expected = {
        "episode_id", "stage", "task", "claim_boundary",
        "demo", "execution", "failure_packet", "revision", "retry", "metrics",
    }
    assert set(d.keys()) == expected
    assert "goal_xy" not in rec.demo
    assert "blocked_sides" not in rec.demo


def test_run_episode_jsonl_roundtrip(fake_env_runner):
    rec = run_episode(
        episode_id="pushcube_blocked_approach_seed_0000",
        seed=0,
        adapter=_make_stub_adapter(fake_env_runner),
    )
    line = rec.to_jsonl_line()
    rt = EpisodeRecord.from_jsonl_line(line)
    assert rt.episode_id == rec.episode_id
    assert rt.failure_packet["failure_predicate"] == "approach_blocked"


def test_run_episode_scripted_demo_to_intent_called_with_only_demo_evidence(
    fake_env_runner,
):
    """Privileged-firewall enforcement: adapter.scripted_demo_to_intent
    must be called with a DemoEvidence and nothing else."""
    from babysteps.envs.pushcube_adapter import PushCubeAdapter

    call_args = []

    class _SpyAdapter(PushCubeAdapter):
        def make_env_runner(self):
            return fake_env_runner
        def scripted_demo_to_intent(self, evidence):
            call_args.append(evidence)
            return super().scripted_demo_to_intent(evidence)

    run_episode(
        episode_id="pushcube_blocked_approach_seed_0000",
        seed=0,
        adapter=_SpyAdapter(),
    )
    assert len(call_args) == 1
    assert isinstance(call_args[0], DemoEvidence)


def test_run_episode_already_succeeds_no_revision(fake_env_runner):
    """If blocked_sides is empty, the initial intent succeeds and no
    revision/retry is recorded."""
    rec = run_episode(
        episode_id="pushcube_unblocked_seed_0000",
        seed=0,
        adapter=_make_stub_adapter(
            fake_env_runner,
            blocked_factory=lambda intent: (),   # never blocks
        ),
    )
    assert rec.execution["success"] is True
    assert rec.failure_packet["failure_predicate"] == "none"
    assert rec.revision is None
    assert rec.retry is None
    assert rec.metrics["initial_success"] is True
    assert rec.metrics["num_attempts_to_success"] == 1


def test_run_episode_multiple_seeds_all_succeed(fake_env_runner):
    adapter = _make_stub_adapter(fake_env_runner)
    for seed in range(4):
        rec = run_episode(
            episode_id=f"pushcube_blocked_approach_seed_{seed:04d}",
            seed=seed,
            adapter=adapter,
        )
        assert rec.execution["success"] is False
        assert rec.retry["success"] is True, (
            f"seed {seed} did not recover; revised approach was "
            f"{rec.revision['new_value']!r}"
        )


def test_run_episode_signature_takes_adapter_keyword():
    """Guard against accidentally restoring the old env_runner= kwarg."""
    sig = inspect.signature(run_episode)
    assert "adapter" in sig.parameters
    assert "env_runner" not in sig.parameters
    assert "blocked_sides_factory" not in sig.parameters


def test_run_episode_one_shot_policy_has_no_retry(fake_env_runner):
    from babysteps.policies import one_shot
    adapter = _make_stub_adapter(fake_env_runner)
    rec = run_episode(
        episode_id="t", seed=1, adapter=adapter, policy=one_shot)
    assert rec.retry is None
    assert rec.revision is None


def test_run_episode_default_policy_is_selective(fake_env_runner):
    params = inspect.signature(run_episode).parameters
    assert "policy" in params


def test_env_runner_run_accepts_rollout_seed(fake_env_runner):
    # The fake runner must accept the optional kwarg (deterministic: ignored).
    from babysteps.schemas import Intent, SceneState
    scene = fake_env_runner.reset(0)
    intent = Intent(
        goal_state="cube_at_target", object_motion="translate_+x",
        contact_region="plus_x_face", approach_direction="from_minus_x",
        constraint_region="none", embodiment_mapping="proxy_contact_to_franka_push")
    r1 = fake_env_runner.run(intent, scene, rollout_seed=1)
    r2 = fake_env_runner.run(intent, scene, rollout_seed=2)
    assert r1.success == r2.success  # deterministic fake env


def test_baseline_metrics_correct_factor_fixed_is_inclusion():
    # correct_factor_fixed must be True when the implicated factor was changed,
    # even if the new value differs from the oracle's canonical token.
    from babysteps.episode import _baseline_metrics
    from babysteps.schemas import Intent
    base = Intent(
        goal_state="cube_at_target", object_motion="translate_+x",
        contact_region="minus_x_face", approach_direction="from_minus_x",
        constraint_region="none", embodiment_mapping="proxy_contact_to_franka_push")
    # revised changed approach_direction to a DIFFERENT (non-oracle) unblocked side
    revised = replace(base, approach_direction="from_plus_y")
    toks = {"approach_direction": ("from_minus_x", "from_plus_x", "from_plus_y"),
            "contact_region": ("minus_x_face", "plus_x_face", "minus_y_face", "plus_y_face")}
    m = _baseline_metrics(base, revised, "approach_direction", toks)
    assert m["correct_factor_fixed"] is True          # implicated factor was edited
    assert m["harmful_revision"] is False             # contact_region (should-preserve) untouched
    assert m["n_should_preserve"] == 1 and m["n_preserved"] == 1
    # same-intent retry: nothing changed → not fixed
    m2 = _baseline_metrics(base, base, "approach_direction", toks)
    assert m2["correct_factor_fixed"] is False
