"""Integration test for the Stage-0 episode loop with a fake env_runner.

This test exercises the full path:
  demo proxy → demo_to_intent → skill compile (blocked) → planner_failed →
  attribute → revise → retry succeeds.
"""
from __future__ import annotations

import json

import pytest

from babysteps.episode import run_episode
from babysteps.schemas import CLAIM_BOUNDARY, EpisodeRecord


def test_run_episode_blocked_then_retry_success(fake_env_runner):
    rec = run_episode(
        episode_id="pushcube_blocked_approach_seed_0000",
        seed=0,
        env_runner=fake_env_runner,
    )
    assert isinstance(rec, EpisodeRecord)
    assert rec.claim_boundary == CLAIM_BOUNDARY
    assert rec.demo["demonstrator_type"] == "proxy_oracle"
    # Attempt 1 was blocked by the approach-direction feasibility flag.
    assert rec.execution["success"] is False
    assert rec.failure_packet["failure_predicate"] == "approach_blocked"
    assert rec.failure_packet["wrong_factor"] == "approach_direction"
    # The revision changed exactly approach_direction.
    assert rec.revision is not None
    assert rec.revision["operator"] == "approach_substitution"
    assert rec.revision["factor"] == "approach_direction"
    # Retry succeeded.
    assert rec.retry is not None
    assert rec.retry["success"] is True
    # Metrics carry the diagnostic columns.
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
        env_runner=fake_env_runner,
    )
    d = rec.to_dict()
    expected = {
        "episode_id", "stage", "task", "claim_boundary",
        "demo", "execution", "failure_packet", "revision", "retry", "metrics",
    }
    assert set(d.keys()) == expected
    # Demo dict has no privileged fields.
    assert "goal_xy" not in rec.demo
    assert "blocked_sides" not in rec.demo


def test_run_episode_jsonl_roundtrip(fake_env_runner):
    rec = run_episode(
        episode_id="pushcube_blocked_approach_seed_0000",
        seed=0,
        env_runner=fake_env_runner,
    )
    line = rec.to_jsonl_line()
    rt = EpisodeRecord.from_jsonl_line(line)
    assert rt.episode_id == rec.episode_id
    assert rt.failure_packet["failure_predicate"] == "approach_blocked"


def test_run_episode_demo_to_intent_called_with_only_demo_evidence(monkeypatch, fake_env_runner):
    """Privileged-firewall enforcement: demo_to_intent must be called with a
    DemoEvidence and nothing else (no SceneState parameter)."""
    from babysteps import episode as episode_mod
    from babysteps.schemas import DemoEvidence

    call_args = []
    original = episode_mod.demo_to_intent

    def spy(arg):
        call_args.append(arg)
        return original(arg)

    monkeypatch.setattr(episode_mod, "demo_to_intent", spy)

    run_episode(
        episode_id="pushcube_blocked_approach_seed_0000",
        seed=0,
        env_runner=fake_env_runner,
    )
    assert len(call_args) == 1
    assert isinstance(call_args[0], DemoEvidence)


def test_run_episode_already_succeeds_no_revision(fake_env_runner):
    """If the initial intent already succeeds (e.g., blocked_sides_factory
    returns empty), the record carries no revision and no retry."""
    rec = run_episode(
        episode_id="pushcube_unblocked_seed_0000",
        seed=0,
        env_runner=fake_env_runner,
        blocked_sides_factory=lambda intent: (),   # never blocks
    )
    assert rec.execution["success"] is True
    assert rec.failure_packet["failure_predicate"] == "none"
    assert rec.revision is None
    assert rec.retry is None
    assert rec.metrics["initial_success"] is True
    assert rec.metrics["num_attempts_to_success"] == 1


def test_run_episode_multiple_seeds_all_succeed(fake_env_runner):
    """The blocked-then-retry path works for several seeds (goal direction
    rotates with seed in the fake env)."""
    for seed in range(4):
        rec = run_episode(
            episode_id=f"pushcube_blocked_approach_seed_{seed:04d}",
            seed=seed,
            env_runner=fake_env_runner,
        )
        assert rec.execution["success"] is False
        assert rec.retry["success"] is True, (
            f"seed {seed} did not recover; revised approach was "
            f"{rec.revision['new_value']!r}"
        )
