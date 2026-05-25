"""Tests for babysteps.envs.pickcube_adapter — the second concrete adapter.

Parity tests mirror test_pushcube_adapter.py. Snapshot test asserts a fresh
PickCube JSONL against tests/snapshots/pickcube_samples_seeds_0_4.jsonl.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from babysteps.envs.pickcube_adapter import PickCubeAdapter
from babysteps.envs.task_adapter import BaseTaskAdapter
from babysteps.schemas import DemoEvidence, Intent, SceneState


# ---------- Class-level checks ----------------------------------------- #


def test_pickcube_adapter_subclass_of_base():
    assert issubclass(PickCubeAdapter, BaseTaskAdapter)


def test_pickcube_adapter_task_id():
    assert PickCubeAdapter.task_id == "PickCube-v1"
    assert PickCubeAdapter().task_id == "PickCube-v1"


# ---------- oracle_correct_intent -------------------------------------- #


def _scene(blocked: tuple[str, ...] = ()) -> SceneState:
    return SceneState(
        cube_xy=(0.0, 0.0),
        cube_z=0.02,
        goal_xy=(0.12, 0.04),
        tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0),
        blocked_sides=blocked,
    )


def test_oracle_correct_intent_defaults_to_minus_x_face_when_unblocked():
    intent = PickCubeAdapter().oracle_correct_intent(_scene())
    assert intent.contact_region == "minus_x_face"
    assert intent.approach_direction == "from_above"
    assert intent.object_motion == "lift_up"
    assert intent.goal_state == "cube_lifted_at_target"
    assert intent.embodiment_mapping == "proxy_contact_to_franka_grasp"
    assert intent.constraint_region == "none"


def test_oracle_correct_intent_skips_blocked_faces_in_preference_order():
    intent = PickCubeAdapter().oracle_correct_intent(
        _scene(blocked=("minus_x_face", "plus_x_face")),
    )
    # Falls through to the next preferred cardinal.
    assert intent.contact_region == "minus_y_face"


def test_oracle_correct_intent_raises_when_all_blocked():
    with pytest.raises(RuntimeError, match="every cardinal face is blocked"):
        PickCubeAdapter().oracle_correct_intent(
            _scene(blocked=("minus_x_face", "plus_x_face",
                            "minus_y_face", "plus_y_face")),
        )


# ---------- default_blocked_factory ------------------------------------ #


@pytest.mark.parametrize("contact", [
    "minus_x_face", "plus_x_face", "minus_y_face", "plus_y_face",
])
def test_default_blocked_factory_blocks_the_intent_contact(contact):
    intent = Intent(
        goal_state="cube_lifted_at_target",
        object_motion="lift_up",
        contact_region=contact,
        approach_direction="from_above",
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_grasp",
    )
    assert PickCubeAdapter().default_blocked_factory(intent) == (contact,)


# ---------- oracle_wrong_factor ---------------------------------------- #


def test_oracle_wrong_factor_when_intent_contact_in_blocked():
    intent = PickCubeAdapter().oracle_correct_intent(_scene())
    scene = _scene(blocked=(intent.contact_region,))
    assert PickCubeAdapter().oracle_wrong_factor(intent, scene) == "contact_region"


def test_oracle_wrong_factor_when_intent_contact_unblocked():
    intent = PickCubeAdapter().oracle_correct_intent(_scene())
    scene = _scene(blocked=("plus_y_face",))  # unrelated face
    assert PickCubeAdapter().oracle_wrong_factor(intent, scene) == "none"


# ---------- scripted_demo_to_intent ------------------------------------ #


def _evidence(contact: str, final_state: str = "cube_lifted_at_target") -> DemoEvidence:
    return DemoEvidence(
        camera="third_person",
        demonstrator_type="proxy_oracle",
        object_trajectory=((0.0, 0.0), (0.12, 0.04)),
        contact_region_label=contact,
        final_state=final_state,
        rgbd_video_path=None,
    )


def test_scripted_demo_to_intent_signature_takes_only_demo_evidence():
    """Privileged-firewall regression guard (mirrors PushCube guard)."""
    sig = inspect.signature(PickCubeAdapter.scripted_demo_to_intent)
    params = list(sig.parameters.values())
    assert len(params) == 2
    ev_param = params[1]
    annot = ev_param.annotation
    annot_name = annot if isinstance(annot, str) else getattr(annot, "__name__", str(annot))
    assert annot is DemoEvidence or annot_name == "DemoEvidence"


@pytest.mark.parametrize("contact", [
    "minus_x_face", "plus_x_face", "minus_y_face", "plus_y_face",
])
def test_scripted_demo_to_intent_per_contact(contact):
    intent = PickCubeAdapter().scripted_demo_to_intent(_evidence(contact))
    assert intent.contact_region == contact
    assert intent.object_motion == "lift_up"
    assert intent.approach_direction == "from_above"
    assert intent.goal_state == "cube_lifted_at_target"
    assert intent.embodiment_mapping == "proxy_contact_to_franka_grasp"
    assert intent.constraint_region == "none"


def test_scripted_demo_to_intent_rejects_unknown_contact():
    with pytest.raises(ValueError, match="contact_region"):
        PickCubeAdapter().scripted_demo_to_intent(_evidence("not_a_face"))


# ---------- Hook inheritance ------------------------------------------- #


def test_hooks_inherited_from_base():
    assert PickCubeAdapter.build_failure_packet is BaseTaskAdapter.build_failure_packet
    assert PickCubeAdapter.attribute_failure is BaseTaskAdapter.attribute_failure
    assert PickCubeAdapter.revise_intent is BaseTaskAdapter.revise_intent


# ---------- End-to-end episode loop ------------------------------------ #


def test_run_episode_demo_then_blocked_then_grasp_slip_then_revised_retry(
    fake_pick_env_runner,
):
    """The Stage-0 happy path for PickCube:
      1. Demo (no blockers) → success.
      2. Initial intent reused with executor blocking the demonstrated contact.
      3. Attempt 1 → grasp_slip.
      4. Attribution → wrong_factor = contact_region.
      5. Revision → contact_substitution to the 90°-orthogonal face.
      6. Retry → success.
    """
    from babysteps.episode import run_episode

    class _Adapter(PickCubeAdapter):
        def make_env_runner(self):
            return fake_pick_env_runner

    rec = run_episode(
        episode_id="pickcube_grasp_slip_seed_0000",
        seed=0,
        adapter=_Adapter(),
    )
    assert rec.task == "PickCube-v1"
    assert rec.execution["success"] is False
    assert rec.failure_packet["failure_predicate"] == "grasp_slip"
    assert rec.failure_packet["wrong_factor"] == "contact_region"
    assert rec.revision is not None
    assert rec.revision["operator"] == "contact_substitution"
    assert rec.revision["factor"] == "contact_region"
    assert rec.retry is not None
    assert rec.retry["success"] is True


def test_run_episode_metrics_track_factor_local_revision(fake_pick_env_runner):
    """Non-regression invariant: the revised intent differs from the initial
    in EXACTLY one factor (contact_region)."""
    from babysteps.episode import run_episode

    class _Adapter(PickCubeAdapter):
        def make_env_runner(self):
            return fake_pick_env_runner

    rec = run_episode(
        episode_id="pickcube_grasp_slip_seed_0001",
        seed=1,
        adapter=_Adapter(),
    )
    assert rec.metrics["factors_changed"] == ["contact_region"]
    assert rec.metrics["frozen_factors_preserved"] is True
    assert rec.metrics["factor_attribution_correct"] is True


# ---------- Snapshot acceptance test ----------------------------------- #


def test_pickcube_adapter_samples_jsonl_matches_snapshot(fake_pick_env_runner):
    """Generates 5 episodes via fake runner and asserts byte-equality with
    tests/snapshots/pickcube_samples_seeds_0_4.jsonl. The snapshot is
    captured (and re-captured intentionally) by running
    `python scripts/stage0_collect.py --task PickCube-v1 --fake-env
    --out_dir /tmp/baseline --n_episodes 5 --seed_start 0` and copying
    the produced samples.jsonl into tests/snapshots/."""
    from babysteps.episode import run_episode

    class _Adapter(PickCubeAdapter):
        def make_env_runner(self):
            return fake_pick_env_runner

    adapter = _Adapter()
    actual_lines = []
    for seed in range(5):
        rec = run_episode(
            episode_id=f"pickcube_grasp_slip_seed_{seed:04d}",
            seed=seed,
            adapter=adapter,
        )
        actual_lines.append(rec.to_jsonl_line())
    actual = "\n".join(actual_lines) + "\n"

    snapshot_path = (
        Path(__file__).parent / "snapshots" / "pickcube_samples_seeds_0_4.jsonl"
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
        "PickCubeAdapter samples.jsonl drifted from the snapshot. "
        f"Snapshot at: {snapshot_path}. "
        "If intentional, delete the snapshot file and re-run this test "
        "to re-capture."
    )
