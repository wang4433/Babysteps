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


def test_oracle_correct_intent_is_handle_grip_with_constraint():
    adapter = TurnFaucetAdapter()
    intent = adapter.oracle_correct_intent(_scene_with_extra())
    assert intent.contact_region == "handle_grip"
    assert intent.constraint_region == "faucet_base_static"
    assert intent.goal_state == "faucet_turned"
    assert intent.object_motion == "turn"
    assert intent.approach_direction == "from_above"
    assert intent.embodiment_mapping == "proxy_contact_to_franka_turn"


def test_default_blocked_factory_is_empty():
    intent = Intent(
        goal_state="faucet_turned", object_motion="turn",
        contact_region="faucet_base", approach_direction="from_above",
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_turn",
    )
    adapter = TurnFaucetAdapter()
    assert adapter.default_blocked_factory(intent) == ()


def test_oracle_wrong_factor_for_faucet_base_contact():
    intent = Intent(
        goal_state="faucet_turned", object_motion="turn",
        contact_region="faucet_base", approach_direction="from_above",
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_turn",
    )
    adapter = TurnFaucetAdapter()
    assert adapter.oracle_wrong_factor(intent, _scene_with_extra()) == "constraint_region"


def test_oracle_wrong_factor_for_correct_intent():
    intent = Intent(
        goal_state="faucet_turned", object_motion="turn",
        contact_region="handle_grip", approach_direction="from_above",
        constraint_region="faucet_base_static",
        embodiment_mapping="proxy_contact_to_franka_turn",
    )
    adapter = TurnFaucetAdapter()
    assert adapter.oracle_wrong_factor(intent, _scene_with_extra()) == "none"


def test_scripted_demo_to_intent_always_under_specifies_both():
    """Stage-0 controlled mechanism: scripted_demo_to_intent always
    returns contact_region=faucet_base AND constraint_region=none."""
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
    assert intent.contact_region == "faucet_base"   # under-specified
    assert intent.constraint_region == "none"        # under-specified
    assert intent.goal_state == "faucet_turned"
    assert intent.object_motion == "turn"
    assert intent.embodiment_mapping == "proxy_contact_to_franka_turn"


def test_scripted_demo_to_intent_ignores_contact_region_label():
    """The label could be anything; the summarizer doesn't use it."""
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
    assert i1.contact_region == "faucet_base"


def test_compile_skill_delegates_to_turn_skill():
    from babysteps.skills.turn import TurnSkill
    intent = Intent(
        goal_state="faucet_turned", object_motion="turn",
        contact_region="handle_grip", approach_direction="from_above",
        constraint_region="faucet_base_static",
        embodiment_mapping="proxy_contact_to_franka_turn",
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


def test_full_episode_via_fake_runner_recovers_via_constraint_introduction(
    fake_turnfaucet_env_runner,
):
    from babysteps.episode import run_episode

    class _Adapter(TurnFaucetAdapter):
        def make_env_runner(self):
            return fake_turnfaucet_env_runner

    rec = run_episode(
        episode_id="turnfaucet_wrong_contact_seed_0000",
        seed=0,
        adapter=_Adapter(),
    )
    assert rec.metrics["initial_success"] is False
    assert rec.metrics["retry_success"] is True
    assert rec.metrics["factor_attribution_correct"] is True
    assert rec.metrics["frozen_factors_preserved"] is True
    # Two factors changed (constraint_introduction is two-factor).
    assert set(rec.metrics["factors_changed"]) == {"constraint_region", "contact_region"}
    assert rec.revision is not None
    assert rec.revision["operator"] == "constraint_introduction"
    assert rec.revision["factor"] == "constraint_region"
    assert rec.revision["old_value"] == "none"
    assert rec.revision["new_value"] == "faucet_base_static"


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
