"""Tests for babysteps/envs/turnfaucet_adapter.py.

Mirrors test_stackcube_adapter.py's shape: parity tests + snapshot."""
from __future__ import annotations

from pathlib import Path

import pytest

from babysteps.envs.task_adapter import BaseTaskAdapter
from babysteps.envs.turnfaucet_adapter import TurnFaucetAdapter
from babysteps.schemas import DemoEvidence, Intent, SceneState


def _scene_with_extra():
    return SceneState(
        cube_xy=(0.1, 0.0), cube_z=0.10, goal_xy=(0.1, 0.0),
        tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0),
        blocked_sides=(),
        extra={
            "handle_xy": (0.1, 0.0), "handle_z": 0.10,
            "faucet_base_xy": (0.05, 0.0), "faucet_base_z": 0.0,
            "target_joint_axis_xy": (0.0, 1.0),
        },
    )


def test_task_id_is_turnfaucet_v1():
    assert TurnFaucetAdapter.task_id == "TurnFaucet-v1"


def test_is_subclass_of_basetaskadapter():
    assert issubclass(TurnFaucetAdapter, BaseTaskAdapter)


def test_oracle_correct_intent_is_handle_grip_poke_turn():
    """oracle_correct_intent emits the mechanically feasible embodiment:
    poke_turn with no constraint_region (constraint_introduction removed)."""
    adapter = TurnFaucetAdapter()
    intent = adapter.oracle_correct_intent(_scene_with_extra())
    assert intent.contact_region == "handle_grip"
    assert intent.constraint_region == "none"
    assert intent.goal_state == "faucet_turned"
    assert intent.object_motion == "turn"
    assert intent.approach_direction == "from_above"
    assert intent.embodiment_mapping == "proxy_contact_to_franka_poke_turn"


def test_default_blocked_factory_is_empty():
    intent = Intent(
        goal_state="faucet_turned", object_motion="turn",
        contact_region="handle_grip", approach_direction="from_above",
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_grasp_turn",
    )
    adapter = TurnFaucetAdapter()
    assert adapter.default_blocked_factory(intent) == ()


def test_oracle_wrong_factor_for_grasp_turn_intent():
    """grasp_turn is the infeasible initial intent → wrong factor is embodiment_mapping."""
    intent = Intent(
        goal_state="faucet_turned", object_motion="turn",
        contact_region="handle_grip", approach_direction="from_above",
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_grasp_turn",
    )
    adapter = TurnFaucetAdapter()
    assert adapter.oracle_wrong_factor(intent, _scene_with_extra()) == "embodiment_mapping"


def test_oracle_wrong_factor_for_poke_turn_correct_intent():
    """poke_turn is the correct (oracle) intent → no wrong factor."""
    intent = Intent(
        goal_state="faucet_turned", object_motion="turn",
        contact_region="handle_grip", approach_direction="from_above",
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_poke_turn",
    )
    adapter = TurnFaucetAdapter()
    assert adapter.oracle_wrong_factor(intent, _scene_with_extra()) == "none"


def test_scripted_demo_to_intent_returns_grasp_turn_infeasible_embodiment():
    """Stage-0 information loss: scripted_demo_to_intent always returns
    embodiment_mapping=grasp_turn — the 2D summarizer encodes the
    hand-like demo interaction as grasping without knowing the Franka
    cannot mechanically envelop the handle."""
    evidence = DemoEvidence(
        camera="third_person",
        demonstrator_type="proxy_oracle",
        object_trajectory=((0.0, 0.0), (0.1, 0.0)),
        contact_region_label="handle_grip",   # demo's true label
        final_state="faucet_turned",
        rgbd_video_path=None,
    )
    adapter = TurnFaucetAdapter()
    intent = adapter.scripted_demo_to_intent(evidence)
    assert intent.contact_region == "handle_grip"       # preserved (not under-specified)
    assert intent.constraint_region == "none"            # no constraint
    assert intent.goal_state == "faucet_turned"
    assert intent.object_motion == "turn"
    assert intent.embodiment_mapping == "proxy_contact_to_franka_grasp_turn"  # infeasible


def test_scripted_demo_to_intent_ignores_contact_region_label():
    """The label could be anything; the summarizer always produces the
    same infeasible grasp_turn mapping regardless of demo label."""
    evidence_handle = DemoEvidence(
        camera="third_person", demonstrator_type="proxy_oracle",
        object_trajectory=((0.0, 0.0), (0.1, 0.0)),
        contact_region_label="handle_grip",
        final_state="faucet_turned", rgbd_video_path=None,
    )
    evidence_minus_x = DemoEvidence(
        camera="third_person", demonstrator_type="proxy_oracle",
        object_trajectory=((0.0, 0.0), (0.1, 0.0)),
        contact_region_label="minus_x_face",
        final_state="faucet_turned", rgbd_video_path=None,
    )
    adapter = TurnFaucetAdapter()
    i1 = adapter.scripted_demo_to_intent(evidence_handle)
    i2 = adapter.scripted_demo_to_intent(evidence_minus_x)
    assert i1 == i2
    assert i1.embodiment_mapping == "proxy_contact_to_franka_grasp_turn"


def test_compile_skill_delegates_to_turn_skill():
    from babysteps.skills.turn import TurnSkill
    intent = Intent(
        goal_state="faucet_turned", object_motion="turn",
        contact_region="handle_grip", approach_direction="from_above",
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_poke_turn",
    )
    adapter = TurnFaucetAdapter()
    skill = adapter.compile_skill(intent, _scene_with_extra())
    assert isinstance(skill, TurnSkill)


def test_adapter_inherits_default_hooks():
    assert (
        TurnFaucetAdapter.build_failure_packet
        is BaseTaskAdapter.build_failure_packet
    )
    assert (
        TurnFaucetAdapter.attribute_failure
        is BaseTaskAdapter.attribute_failure
    )
    assert (
        TurnFaucetAdapter.revise_intent
        is BaseTaskAdapter.revise_intent
    )


# ---------- end-to-end episode loop test ------------------------------ #


def test_full_episode_via_fake_runner_recovers_via_embodiment_substitution(
    fake_turnfaucet_env_runner,
):
    """End-to-end episode loop: scripted intent has grasp_turn (infeasible);
    failure → grasp_infeasible; revision → poke_turn; retry succeeds.

    NOTE: this test requires FakeTurnFaucetEnvRunner (T9) to return
    collision=False, object_moved=False for grasp_turn intents so that
    build_failure_packet fires grasp_infeasible instead of constraint_violation.
    Until T9 lands, this test is expected to FAIL.
    """
    from babysteps.episode import run_episode

    class _Adapter(TurnFaucetAdapter):
        def make_env_runner(self):
            return fake_turnfaucet_env_runner

    rec = run_episode(
        episode_id="turnfaucet_embodiment_seed_0000",
        seed=0,
        adapter=_Adapter(),
    )
    assert rec.metrics["initial_success"] is False
    assert rec.metrics["retry_success"] is True
    assert rec.metrics["factor_attribution_correct"] is True
    assert rec.metrics["frozen_factors_preserved"] is True
    # Single factor changed (embodiment_substitution is single-factor).
    assert set(rec.metrics["factors_changed"]) == {"embodiment_mapping"}
    assert rec.revision is not None
    assert rec.revision["operator"] == "embodiment_substitution"
    assert rec.revision["factor"] == "embodiment_mapping"
    assert rec.revision["old_value"] == "proxy_contact_to_franka_grasp_turn"
    assert rec.revision["new_value"] == "proxy_contact_to_franka_poke_turn"


# ---------- Snapshot acceptance test --------------------------------- #


def test_turnfaucet_adapter_samples_jsonl_matches_snapshot(
    fake_turnfaucet_env_runner,
):
    from babysteps.episode import run_episode

    class _Adapter(TurnFaucetAdapter):
        def make_env_runner(self):
            return fake_turnfaucet_env_runner

    adapter = _Adapter()
    actual_lines = []
    for seed in range(5):
        rec = run_episode(
            episode_id=f"turnfaucet_wrong_contact_seed_{seed:04d}",
            seed=seed,
            adapter=adapter,
        )
        actual_lines.append(rec.to_jsonl_line())
    actual = "\n".join(actual_lines) + "\n"

    snapshot_path = (
        Path(__file__).parent / "snapshots" / "turnfaucet_samples_seeds_0_4.jsonl"
    )
    if not snapshot_path.exists():
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(actual)
        pytest.skip(
            f"Captured initial snapshot at {snapshot_path}. Re-run to "
            f"verify byte-equality."
        )
    expected = snapshot_path.read_text()
    assert actual == expected, (
        "TurnFaucetAdapter samples.jsonl drifted from the snapshot. "
        f"Snapshot at: {snapshot_path}. "
        "If intentional, delete the snapshot file and re-run this test "
        "to re-capture."
    )


# ---------- New embodiment-substitution contract tests (T8) ----------- #


def test_oracle_correct_intent_returns_poke_turn():
    from babysteps.envs.turnfaucet_adapter import TurnFaucetAdapter
    from babysteps.schemas import SceneState
    scene = SceneState(
        cube_xy=(0.05, 0.02), cube_z=0.10, goal_xy=(0.05, 0.02),
        tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0),
        blocked_sides=(),
        extra={"handle_xy": (0.05, 0.02), "handle_z": 0.10,
               "target_joint_axis_xy": (0.0, 1.0)},
    )
    adapter = TurnFaucetAdapter()
    intent = adapter.oracle_correct_intent(scene)
    assert intent.embodiment_mapping == "proxy_contact_to_franka_poke_turn"
    assert intent.contact_region == "handle_grip"
    assert intent.constraint_region == "none"
    assert intent.goal_state == "faucet_turned"
    assert intent.object_motion == "turn"
    assert intent.approach_direction == "from_above"


def test_scripted_demo_to_intent_returns_grasp_turn():
    from babysteps.envs.turnfaucet_adapter import TurnFaucetAdapter
    from babysteps.schemas import DemoEvidence
    evidence = DemoEvidence(
        camera="third_person", demonstrator_type="proxy_oracle",
        object_trajectory=((0.05, 0.02),) * 2,
        contact_region_label="handle_grip",
        final_state="faucet_turned",
        rgbd_video_path=None,
    )
    intent = TurnFaucetAdapter().scripted_demo_to_intent(evidence)
    assert intent.embodiment_mapping == "proxy_contact_to_franka_grasp_turn"
    assert intent.contact_region == "handle_grip"
    assert intent.constraint_region == "none"


def test_oracle_wrong_factor_embodiment_mapping_for_grasp_turn():
    from babysteps.envs.turnfaucet_adapter import TurnFaucetAdapter
    from babysteps.schemas import Intent
    adapter = TurnFaucetAdapter()
    grasp = Intent(
        goal_state="faucet_turned", object_motion="turn",
        contact_region="handle_grip", approach_direction="from_above",
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_grasp_turn",
    )
    assert adapter.oracle_wrong_factor(grasp) == "embodiment_mapping"
    poke = Intent(
        goal_state="faucet_turned", object_motion="turn",
        contact_region="handle_grip", approach_direction="from_above",
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_poke_turn",
    )
    assert adapter.oracle_wrong_factor(poke) == "none"


def test_fake_runner_poke_turn_returns_success_true(fake_turnfaucet_env_runner):
    from babysteps.schemas import Intent
    scene = fake_turnfaucet_env_runner.reset(seed=0)
    poke = Intent(
        goal_state="faucet_turned", object_motion="turn",
        contact_region="handle_grip", approach_direction="from_above",
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_poke_turn",
    )
    result = fake_turnfaucet_env_runner.run(poke, scene)
    assert result.success is True
    assert result.object_moved is True
    assert result.reached_contact is True


def test_fake_runner_grasp_turn_returns_grasp_infeasible_signature(fake_turnfaucet_env_runner):
    from babysteps.schemas import Intent
    scene = fake_turnfaucet_env_runner.reset(seed=0)
    grasp = Intent(
        goal_state="faucet_turned", object_motion="turn",
        contact_region="handle_grip", approach_direction="from_above",
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_grasp_turn",
    )
    result = fake_turnfaucet_env_runner.run(grasp, scene)
    assert result.success is False
    assert result.object_moved is False
    assert result.reached_contact is True
    assert result.collision is False  # spec §8.4: collision never set in new D
    assert result.grasp_slip is False  # PickCube-specific; never set by TF
