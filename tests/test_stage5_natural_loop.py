"""Sim-free guards for the Stage-5 NATURAL-failure, seed-decoupled loop.

These assert the headline property the rework exists to demonstrate: under a
demo/exec direction MISMATCH with NO artificial block, failure is driven by the
stale intent (not a block), and ONLY revisers that consume the execution
feedback recover — the open-loop revisers do not.

Runs on the login node via tests.conftest.FakeEnvRunner, whose reset() varies
the goal direction by seed%4 (0:+x, 1:+y, 2:-x, 3:-y) and whose run() succeeds
iff contact_region matches the scene's goal direction. No GPU/Vulkan/mani_skill.
"""
from __future__ import annotations

from babysteps.envs.pushcube_adapter import PushCubeAdapter
from scripts.stage5_natural_loop_eval import REVISERS, run_natural_episode
from tests.conftest import FakeEnvRunner

# FakeEnvRunner direction by seed%4: 0:+x, 1:+y, 2:-x, 3:-y.
DEMO_PLUS_X = 400      # 400 % 4 == 0  -> +x goal
EXEC_MINUS_X = 2       # 2 % 4 == 2    -> -x goal   (180deg mismatch)
EXEC_PLUS_X = 8        # 8 % 4 == 0    -> +x goal   (matches demo)
EXEC_PLUS_Y = 5        # 5 % 4 == 1    -> +y goal   (90deg perpendicular mismatch)
ALL_REVISERS = ["same_intent", "rule_orthogonal", "feedback_flip",
                "feedback_residual", "oracle_value"]


def _episode(demo_seed, exec_seed):
    adapter = PushCubeAdapter()
    runner = FakeEnvRunner()
    return run_natural_episode(
        adapter, runner, demo_seed=demo_seed, exec_seed=exec_seed,
        demo_motion=None, exec_motion=None, revisers=ALL_REVISERS)


def test_mismatch_fails_naturally_without_a_block():
    """Demo +x intent on a -x exec scene, NO block -> natural failure."""
    r = _episode(DEMO_PLUS_X, EXEC_MINUS_X)
    assert r["direction_mismatch"] is True
    assert r["initial_success"] is False
    # The failure is a real wrong-direction push, not the block path.
    assert r["failure_predicate"] in ("direction_error", "goal_not_satisfied")
    assert r["failure_predicate"] != "approach_blocked"


def test_displacement_vec_is_populated_on_failure():
    """The 2D execution-feedback vector exists (it is None only in the block
    path / when nothing moved)."""
    r = _episode(DEMO_PLUS_X, EXEC_MINUS_X)
    vec = r["displacement_vec"]
    assert vec is not None and len(vec) == 2
    # Cube was pushed +x while the goal was -x -> non-trivial x drift.
    assert abs(vec[0]) > 1e-3


def test_open_loop_revisers_do_not_recover():
    """same_intent and the orthogonal rule both fail the binary x-flip."""
    r = _episode(DEMO_PLUS_X, EXEC_MINUS_X)
    assert r["revisers"]["same_intent"]["final_success"] is False
    # rule_orthogonal picks a y-face (90deg), not the opposite x-face -> fails.
    assert r["revisers"]["rule_orthogonal"]["final_success"] is False
    assert r["revisers"]["rule_orthogonal"]["new_contact"] in (
        "minus_y_face", "plus_y_face")


def test_feedback_and_oracle_revisers_recover():
    """Revisers that consume execution feedback flip to the opposite face."""
    r = _episode(DEMO_PLUS_X, EXEC_MINUS_X)
    for name in ("feedback_flip", "feedback_residual", "oracle_value"):
        assert r["revisers"][name]["final_success"] is True, name
        assert r["revisers"][name]["new_contact"] == "plus_x_face", name


def test_perpendicular_mismatch_needs_residual_not_flip():
    """4-way: a +x demo on a +y exec is a 90deg mismatch. The reverse-only
    heuristic (feedback_flip) picks an x-axis face and FAILS; the goal-relative
    residual / oracle pick the +y face and recover. This is why 4-way needs a
    goal-relative feedback signal, not just 'reverse on failure'."""
    r = _episode(DEMO_PLUS_X, EXEC_PLUS_Y)
    assert r["direction_mismatch"] is True
    assert r["initial_success"] is False
    # reverse-only -> x-axis face -> wrong for a +y goal
    assert r["revisers"]["feedback_flip"]["final_success"] is False
    assert r["revisers"]["feedback_flip"]["new_contact"] in ("minus_x_face", "plus_x_face")
    # goal-relative residual and oracle recover the perpendicular case
    for name in ("feedback_residual", "oracle_value"):
        assert r["revisers"][name]["final_success"] is True, name
        assert r["revisers"][name]["new_contact"] == "minus_y_face", name


def test_matched_direction_succeeds_at_attempt_one():
    """Control: when demo and exec share the direction, no failure, no revision."""
    r = _episode(DEMO_PLUS_X, EXEC_PLUS_X)
    assert r["direction_mismatch"] is False
    assert r["initial_success"] is True
    for name in ALL_REVISERS:
        assert r["revisers"][name]["final_success"] is True
        assert r["revisers"][name]["revised"] is False


def test_reviser_registry_matches():
    assert set(REVISERS) == set(ALL_REVISERS)
