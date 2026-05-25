"""Tests for babysteps.envs.pushcube_adapter — the first concrete adapter.

Proves byte-equivalent behaviour with the pre-A episode/demo/execution
helpers. The snapshot test is added in Task 9 once the episode refactor
is wired through; the per-method parity tests below land in Task 5."""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from babysteps.envs.pushcube_adapter import PushCubeAdapter
from babysteps.envs.task_adapter import BaseTaskAdapter
from babysteps.schemas import DemoEvidence, Intent, SceneState


# ---------- Class-level checks ----------------------------------------- #


def test_pushcube_adapter_subclass_of_base():
    assert issubclass(PushCubeAdapter, BaseTaskAdapter)


def test_pushcube_adapter_task_id():
    assert PushCubeAdapter.task_id == "PushCube-v1"
    assert PushCubeAdapter().task_id == "PushCube-v1"


# ---------- oracle_correct_intent parity ------------------------------- #


def _scene_with_goal(goal_xy: tuple[float, float]) -> SceneState:
    return SceneState(
        cube_xy=(0.0, 0.0),
        cube_z=0.02,
        goal_xy=goal_xy,
        tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0),
        blocked_sides=(),
    )


@pytest.mark.parametrize("goal_xy,expected_face,expected_approach,expected_motion", [
    ((0.15, 0.0),  "minus_x_face", "from_minus_x", "translate_+x"),
    ((-0.15, 0.0), "plus_x_face",  "from_plus_x",  "translate_-x"),
    ((0.0, 0.15),  "minus_y_face", "from_minus_y", "translate_+y"),
    ((0.0, -0.15), "plus_y_face",  "from_plus_y",  "translate_-y"),
])
def test_oracle_correct_intent_per_cardinal(
    goal_xy, expected_face, expected_approach, expected_motion,
):
    scene = _scene_with_goal(goal_xy)
    intent = PushCubeAdapter().oracle_correct_intent(scene)
    assert intent.contact_region == expected_face
    assert intent.approach_direction == expected_approach
    assert intent.object_motion == expected_motion
    assert intent.goal_state == "cube_at_target"
    assert intent.constraint_region == "none"
    assert intent.embodiment_mapping == "proxy_contact_to_franka_push"


# ---------- default_blocked_factory parity ----------------------------- #


@pytest.mark.parametrize("approach", [
    "from_minus_x", "from_plus_x", "from_minus_y", "from_plus_y",
])
def test_default_blocked_factory_blocks_the_intent_approach(approach):
    intent = Intent(
        goal_state="cube_at_target",
        object_motion="translate_+x",
        contact_region="minus_x_face",
        approach_direction=approach,
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_push",
    )
    assert PushCubeAdapter().default_blocked_factory(intent) == (approach,)


# ---------- oracle_wrong_factor parity --------------------------------- #


def test_oracle_wrong_factor_when_intent_approach_in_blocked():
    scene = SceneState(
        cube_xy=(0.0, 0.0), cube_z=0.02, goal_xy=(0.15, 0.0),
        tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0),
        blocked_sides=("from_minus_x",),
    )
    intent = PushCubeAdapter().oracle_correct_intent(_scene_with_goal((0.15, 0.0)))
    assert PushCubeAdapter().oracle_wrong_factor(intent, scene) == "approach_direction"


def test_oracle_wrong_factor_when_intent_approach_unblocked():
    scene = SceneState(
        cube_xy=(0.0, 0.0), cube_z=0.02, goal_xy=(0.15, 0.0),
        tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0),
        blocked_sides=("from_plus_y",),  # unrelated to intent's approach
    )
    intent = PushCubeAdapter().oracle_correct_intent(_scene_with_goal((0.15, 0.0)))
    assert PushCubeAdapter().oracle_wrong_factor(intent, scene) == "none"


# ---------- scripted_demo_to_intent parity ----------------------------- #


def _evidence(traj, face) -> DemoEvidence:
    return DemoEvidence(
        camera="third_person",
        demonstrator_type="proxy_oracle",
        object_trajectory=tuple(tuple(p) for p in traj),
        contact_region_label=face,
        final_state="cube_at_target",
        rgbd_video_path=None,
    )


def test_scripted_demo_to_intent_signature_takes_only_demo_evidence():
    """Privileged-firewall regression guard, moved from test_demo.py."""
    sig = inspect.signature(PushCubeAdapter.scripted_demo_to_intent)
    params = list(sig.parameters.values())
    # self + evidence = 2.
    assert len(params) == 2, (
        f"scripted_demo_to_intent must take ONLY (self, DemoEvidence); "
        f"got params: {[p.name for p in params]}"
    )
    ev_param = params[1]
    annot = ev_param.annotation
    annot_name = annot if isinstance(annot, str) else getattr(annot, "__name__", str(annot))
    assert annot is DemoEvidence or annot_name == "DemoEvidence", (
        f"second parameter must be annotated DemoEvidence, got {annot!r}"
    )


def test_scripted_demo_to_intent_plus_x():
    ev = _evidence([(0.0, 0.0), (0.10, 0.0)], "minus_x_face")
    intent = PushCubeAdapter().scripted_demo_to_intent(ev)
    assert intent.object_motion == "translate_+x"
    assert intent.contact_region == "minus_x_face"
    assert intent.approach_direction == "from_minus_x"
    assert intent.goal_state == "cube_at_target"
    assert intent.constraint_region == "none"
    assert intent.embodiment_mapping == "proxy_contact_to_franka_push"


def test_scripted_demo_to_intent_minus_y():
    ev = _evidence([(0.0, 0.0), (0.0, -0.1)], "plus_y_face")
    intent = PushCubeAdapter().scripted_demo_to_intent(ev)
    assert intent.object_motion == "translate_-y"
    assert intent.contact_region == "plus_y_face"
    assert intent.approach_direction == "from_plus_y"


def test_scripted_demo_to_intent_rejects_unknown_contact_region():
    ev = _evidence([(0.0, 0.0), (0.1, 0.0)], "not_a_face")
    with pytest.raises(ValueError, match="contact_region"):
        PushCubeAdapter().scripted_demo_to_intent(ev)


# ---------- Hook inheritance check ------------------------------------- #


def test_hooks_inherited_from_base():
    """PushCubeAdapter does not override the three hooks."""
    assert PushCubeAdapter.build_failure_packet is BaseTaskAdapter.build_failure_packet
    assert PushCubeAdapter.attribute_failure is BaseTaskAdapter.attribute_failure
    assert PushCubeAdapter.revise_intent is BaseTaskAdapter.revise_intent


# ---------- Snapshot acceptance test ----------------------------------- #


def test_pushcube_task_valid_tokens():
    from babysteps.envs.pushcube_adapter import PushCubeAdapter
    toks = PushCubeAdapter().task_valid_tokens()
    # PushCube edits approach_direction and contact_region only.
    assert set(toks) == {"approach_direction", "contact_region"}
    assert set(toks["contact_region"]) == {
        "minus_x_face", "plus_x_face", "minus_y_face", "plus_y_face",
    }
    assert set(toks["approach_direction"]) == {
        "from_minus_x", "from_plus_x", "from_minus_y", "from_plus_y",
    }


def test_pushcube_adapter_samples_jsonl_matches_pre_a_snapshot(fake_env_runner):
    """The byte-equality regression bar for Sub-project A.

    Captures the same five episodes the pre-A code produced and asserts
    the JSONL stream is byte-for-byte identical. If this test diffs, the
    refactor has changed observable output and Sub-project A is not done.
    """
    from pathlib import Path
    from babysteps.envs.pushcube_adapter import PushCubeAdapter
    from babysteps.episode import run_episode

    class _FakeAdapter(PushCubeAdapter):
        def make_env_runner(self):
            return fake_env_runner

    adapter = _FakeAdapter()
    actual_lines = []
    for seed in range(5):
        rec = run_episode(
            episode_id=f"pushcube_blocked_approach_seed_{seed:04d}",
            seed=seed,
            adapter=adapter,
        )
        actual_lines.append(rec.to_jsonl_line())
    actual = "\n".join(actual_lines) + "\n"

    snapshot_path = (
        Path(__file__).parent / "snapshots" / "pushcube_samples_seeds_0_4.jsonl"
    )
    expected = snapshot_path.read_text()
    assert actual == expected, (
        "PushCubeAdapter samples.jsonl drifted from the pre-A snapshot. "
        f"Snapshot at: {snapshot_path}. "
        "If this drift is intentional, re-capture with "
        "`python scripts/stage0_collect.py --out_dir /tmp/baseline "
        "--n_episodes 5 --seed_start 0 --fake-env` and copy the "
        "samples.jsonl into the snapshots/ dir."
    )
