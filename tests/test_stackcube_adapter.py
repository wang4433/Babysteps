"""Tests for babysteps/envs/stackcube_adapter.py.

Mirrors test_pickcube_adapter.py's shape: parity tests + snapshot test.
The snapshot bootstraps on first run, then enforces byte-equality."""
from __future__ import annotations

from pathlib import Path

import pytest

from babysteps.envs.stackcube_adapter import StackCubeAdapter
from babysteps.envs.task_adapter import BaseTaskAdapter
from babysteps.schemas import (
    CONTACT_REGIONS,
    DemoEvidence,
    Intent,
    SceneState,
)


# ---------- adapter API / parity tests ------------------------------- #


def test_task_id_is_stackcube_v1():
    assert StackCubeAdapter.task_id == "StackCube-v1"


def test_is_subclass_of_basetaskadapter():
    assert issubclass(StackCubeAdapter, BaseTaskAdapter)


def test_oracle_correct_intent_is_cubeA_on_cubeB():
    """The oracle knows the correct goal; uses 'cubeA_on_cubeB' with
    'place_on' motion and the pick-and-place embodiment."""
    scene = SceneState(
        cube_xy=(0.0, 0.0), cube_z=0.02, goal_xy=(0.1, 0.0),
        tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0),
        blocked_sides=(),
        extra={"cubeB_xy": (0.1, 0.0), "cubeB_z": 0.02, "cubeB_top_z": 0.06},
    )
    adapter = StackCubeAdapter()
    intent = adapter.oracle_correct_intent(scene)
    assert intent.goal_state == "cubeA_on_cubeB"
    assert intent.object_motion == "place_on"
    assert intent.embodiment_mapping == "proxy_contact_to_franka_pick_and_place"
    assert intent.approach_direction == "from_above"
    assert intent.constraint_region == "none"
    assert intent.contact_region in CONTACT_REGIONS


def test_default_blocked_factory_is_empty():
    """StackCube's controlled failure is from wrong-goal waypoints, not
    blocking — so default_blocked_factory always returns ()."""
    intent = Intent(
        goal_state="cube_at_target", object_motion="translate_+x",
        contact_region="minus_x_face", approach_direction="from_above",
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_pick_and_place",
    )
    adapter = StackCubeAdapter()
    assert adapter.default_blocked_factory(intent) == ()


def test_oracle_wrong_factor_for_under_specified_intent():
    """When the initial intent has goal_state=cube_at_target (the
    deliberately under-specified value), oracle_wrong_factor returns
    'goal_state'."""
    intent = Intent(
        goal_state="cube_at_target", object_motion="translate_+x",
        contact_region="minus_x_face", approach_direction="from_above",
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_pick_and_place",
    )
    scene = SceneState(
        cube_xy=(0.0, 0.0), cube_z=0.02, goal_xy=(0.1, 0.0),
        tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0),
        blocked_sides=(),
    )
    adapter = StackCubeAdapter()
    assert adapter.oracle_wrong_factor(intent, scene) == "goal_state"


def test_oracle_wrong_factor_for_already_correct_intent():
    """If the initial intent already has goal_state=cubeA_on_cubeB,
    nothing is wrong — return 'none'."""
    intent = Intent(
        goal_state="cubeA_on_cubeB", object_motion="place_on",
        contact_region="minus_x_face", approach_direction="from_above",
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_pick_and_place",
    )
    scene = SceneState(
        cube_xy=(0.0, 0.0), cube_z=0.02, goal_xy=(0.1, 0.0),
        tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0),
        blocked_sides=(),
    )
    adapter = StackCubeAdapter()
    assert adapter.oracle_wrong_factor(intent, scene) == "none"


def test_scripted_demo_to_intent_always_under_specifies_goal():
    """The Stage-0 controlled mechanism: scripted_demo_to_intent always
    returns goal_state='cube_at_target' regardless of the demo's true
    final state."""
    evidence = DemoEvidence(
        camera="third_person",
        demonstrator_type="proxy_oracle",
        object_trajectory=((0.0, 0.0), (0.10, 0.0)),
        contact_region_label="minus_x_face",
        # The demo's TRUE final_state is cubeA_on_cubeB — but the
        # summarizer doesn't see the vertical component.
        final_state="cubeA_on_cubeB",
        rgbd_video_path=None,
    )
    adapter = StackCubeAdapter()
    intent = adapter.scripted_demo_to_intent(evidence)
    assert intent.goal_state == "cube_at_target"   # under-specified
    # object_motion derived from dominant 2D axis (cubeA → cubeB.xy).
    assert intent.object_motion in {
        "translate_+x", "translate_-x", "translate_+y", "translate_-y",
    }


def test_scripted_demo_to_intent_object_motion_matches_trajectory():
    """object_motion reflects the dominant axis of the 2D trajectory."""
    # +x dominant
    evidence_px = DemoEvidence(
        camera="third_person", demonstrator_type="proxy_oracle",
        object_trajectory=((0.0, 0.0), (0.10, 0.01)),
        contact_region_label="minus_x_face", final_state="cubeA_on_cubeB",
        rgbd_video_path=None,
    )
    # -y dominant
    evidence_my = DemoEvidence(
        camera="third_person", demonstrator_type="proxy_oracle",
        object_trajectory=((0.0, 0.0), (0.01, -0.10)),
        contact_region_label="minus_x_face", final_state="cubeA_on_cubeB",
        rgbd_video_path=None,
    )
    adapter = StackCubeAdapter()
    assert adapter.scripted_demo_to_intent(evidence_px).object_motion == "translate_+x"
    assert adapter.scripted_demo_to_intent(evidence_my).object_motion == "translate_-y"


def test_scripted_demo_to_intent_uses_pick_and_place_embodiment():
    evidence = DemoEvidence(
        camera="third_person", demonstrator_type="proxy_oracle",
        object_trajectory=((0.0, 0.0), (0.1, 0.0)),
        contact_region_label="minus_x_face", final_state="cubeA_on_cubeB",
        rgbd_video_path=None,
    )
    adapter = StackCubeAdapter()
    intent = adapter.scripted_demo_to_intent(evidence)
    assert intent.embodiment_mapping == "proxy_contact_to_franka_pick_and_place"


def test_scripted_demo_to_intent_rejects_bad_contact_region():
    """Invalid contact_region_label raises ValueError (defensive
    consistent with Push/Pick adapters)."""
    evidence = DemoEvidence(
        camera="third_person", demonstrator_type="proxy_oracle",
        object_trajectory=((0.0, 0.0), (0.1, 0.0)),
        contact_region_label="bogus_face",
        final_state="cubeA_on_cubeB", rgbd_video_path=None,
    )
    adapter = StackCubeAdapter()
    with pytest.raises(ValueError):
        adapter.scripted_demo_to_intent(evidence)


def test_adapter_inherits_default_hooks():
    """StackCubeAdapter does not override the three optional hooks —
    failure attribution and revision use the shared modules unchanged."""
    assert (
        StackCubeAdapter.build_failure_packet
        is BaseTaskAdapter.build_failure_packet
    )
    assert (
        StackCubeAdapter.attribute_failure
        is BaseTaskAdapter.attribute_failure
    )
    assert (
        StackCubeAdapter.revise_intent
        is BaseTaskAdapter.revise_intent
    )


# ---------- end-to-end episode loop test ------------------------------ #


def test_full_episode_via_fake_runner_recovers_via_goal_refinement(
    fake_stack_env_runner,
):
    """One round-trip through run_episode: scripted intent under-specifies →
    goal_not_satisfied → goal_refinement → revised retry succeeds."""
    from babysteps.episode import run_episode

    class _Adapter(StackCubeAdapter):
        def make_env_runner(self):
            return fake_stack_env_runner

    rec = run_episode(
        episode_id="stackcube_underspec_goal_seed_0000",
        seed=0,
        adapter=_Adapter(),
    )
    assert rec.metrics["initial_success"] is False
    assert rec.metrics["retry_success"] is True
    assert rec.metrics["factor_attribution_correct"] is True
    assert rec.metrics["frozen_factors_preserved"] is True
    assert rec.metrics["factors_changed"] == ["goal_state"]
    assert rec.revision is not None
    assert rec.revision["operator"] == "goal_refinement"
    assert rec.revision["factor"] == "goal_state"
    assert rec.revision["old_value"] == "cube_at_target"
    assert rec.revision["new_value"] == "cubeA_on_cubeB"


# ---------- Snapshot acceptance test --------------------------------- #


def test_stackcube_adapter_samples_jsonl_matches_snapshot(fake_stack_env_runner):
    """Generates 5 episodes via fake runner and asserts byte-equality with
    tests/snapshots/stackcube_samples_seeds_0_4.jsonl.

    First-run convenience: if the snapshot does not exist, capture it and
    skip. Subsequent runs verify byte-equality. The same snapshot is
    asserted from the CLI side by tests/test_stage0_collect_cli.py."""
    from babysteps.episode import run_episode

    class _Adapter(StackCubeAdapter):
        def make_env_runner(self):
            return fake_stack_env_runner

    adapter = _Adapter()
    actual_lines = []
    for seed in range(5):
        rec = run_episode(
            episode_id=f"stackcube_underspec_goal_seed_{seed:04d}",
            seed=seed,
            adapter=adapter,
        )
        actual_lines.append(rec.to_jsonl_line())
    actual = "\n".join(actual_lines) + "\n"

    snapshot_path = (
        Path(__file__).parent / "snapshots" / "stackcube_samples_seeds_0_4.jsonl"
    )
    if not snapshot_path.exists():
        # First-run convenience: capture the snapshot.
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(actual)
        pytest.skip(
            f"Captured initial snapshot at {snapshot_path}. Re-run to "
            f"verify byte-equality."
        )
    expected = snapshot_path.read_text()
    assert actual == expected, (
        "StackCubeAdapter samples.jsonl drifted from the snapshot. "
        f"Snapshot at: {snapshot_path}. "
        "If intentional, delete the snapshot file and re-run this test "
        "to re-capture."
    )
