"""Cross-view grounding (Sub-project E) unit + end-to-end tests."""
from __future__ import annotations

import numpy as np
import pytest

from babysteps import schemas


def test_direction_groundings_whitelist():
    assert schemas.DIRECTION_GROUNDINGS == frozenset(
        {"actor_frame", "observer_frame", "object_frame", "world_frame"}
    )


def test_grounding_substitution_operator_registered():
    assert "grounding_substitution" in schemas.REVISION_OPERATORS
    # Existing operators preserved.
    assert "approach_substitution" in schemas.REVISION_OPERATORS


def test_intent_direction_grounding_defaults_and_omits():
    base = dict(
        goal_state="cube_at_target", object_motion="translate_+x",
        contact_region="minus_x_face", approach_direction="from_minus_x",
        constraint_region="none", embodiment_mapping="proxy_contact_to_franka_push",
    )
    # Default value is world_frame and is OMITTED from to_dict (snapshot-safe).
    i_default = schemas.Intent(**base)
    assert i_default.direction_grounding == "world_frame"
    assert "direction_grounding" not in i_default.to_dict()

    # Non-default value IS serialized and round-trips.
    i_actor = schemas.Intent(**base, direction_grounding="actor_frame")
    d = i_actor.to_dict()
    assert d["direction_grounding"] == "actor_frame"
    assert schemas.Intent.from_dict(d) == i_actor

    # A dict without the key reads back as the default.
    assert schemas.Intent.from_dict(i_default.to_dict()).direction_grounding == "world_frame"


def test_intent_direction_grounding_validated():
    with pytest.raises(ValueError):
        schemas.Intent(
            goal_state="cube_at_target", object_motion="translate_+x",
            contact_region="minus_x_face", approach_direction="from_minus_x",
            constraint_region="none",
            embodiment_mapping="proxy_contact_to_franka_push",
            direction_grounding="banana",
        )


from babysteps.envs import scene as scenemod


def test_rotate_motion_token_cardinal():
    # 180° flips signs; 90° CCW maps +x->+y, +y->-x.
    assert scenemod.rotate_motion_token("translate_+x", 180) == "translate_-x"
    assert scenemod.rotate_motion_token("translate_+y", 180) == "translate_-y"
    assert scenemod.rotate_motion_token("translate_+x", 90) == "translate_+y"
    assert scenemod.rotate_motion_token("translate_+y", 90) == "translate_-x"
    assert scenemod.rotate_motion_token("translate_+x", 0) == "translate_+x"


def test_resolve_grounded_motion():
    # actor_frame ignores yaw (identity = the bug).
    assert scenemod.resolve_grounded_motion("translate_-x", "actor_frame", 180) == "translate_-x"
    # observer_frame applies the yaw (the fix): -x observed under 180° -> +x world.
    assert scenemod.resolve_grounded_motion("translate_-x", "observer_frame", 180) == "translate_+x"
    with pytest.raises(NotImplementedError):
        scenemod.resolve_grounded_motion("translate_+x", "object_frame", 90)


def test_world_resolved_intent_recovers_world_face():
    from babysteps.schemas import Intent
    # Observer saw -x (the demo, viewed under 180°). actor_frame keeps it wrong;
    # observer_frame recovers world +x and its contact face minus_x_face.
    observed = Intent(
        goal_state="cube_at_target", object_motion="translate_-x",
        contact_region="plus_x_face", approach_direction="from_plus_x",
        constraint_region="none", embodiment_mapping="proxy_contact_to_franka_push",
        direction_grounding="observer_frame",
    )
    world = scenemod.world_resolved_intent(observed, 180)
    assert world.object_motion == "translate_+x"
    assert world.contact_region == "minus_x_face"
    assert world.approach_direction == "from_minus_x"


from babysteps.failure import Attribution
from babysteps.revision import revise_intent
from babysteps.schemas import INTENT_FIELDS, Intent, SceneState


def _cv_initial_intent() -> Intent:
    return Intent(
        goal_state="cube_at_target", object_motion="translate_-x",
        contact_region="plus_x_face", approach_direction="from_plus_x",
        constraint_region="none", embodiment_mapping="proxy_contact_to_franka_push",
        direction_grounding="actor_frame",
    )


def _dummy_scene() -> SceneState:
    return SceneState(
        cube_xy=(0.0, 0.0), cube_z=0.02, goal_xy=(0.1, 0.0),
        tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0),
        blocked_sides=(), extra={"observer_yaw_deg": 180},
    )


def test_grounding_substitution_flips_only_grounding():
    intent = _cv_initial_intent()
    attribution = Attribution(
        semantic_failure=True, wrong_factor="direction_grounding",
        freeze=INTENT_FIELDS, revise=("direction_grounding",),
    )
    revised, rev = revise_intent(intent, attribution, _dummy_scene())
    assert revised.direction_grounding == "observer_frame"
    # Every six-tuple factor is unchanged.
    for f in INTENT_FIELDS:
        assert getattr(revised, f) == getattr(intent, f)
    assert rev.operator == "grounding_substitution"
    assert rev.factor == "direction_grounding"
    assert rev.old_value == "actor_frame" and rev.new_value == "observer_frame"


def test_grounding_substitution_rejects_non_actor_frame():
    intent = _cv_initial_intent()
    intent = Intent.from_dict({**intent.to_dict(), "direction_grounding": "observer_frame"})
    attribution = Attribution(
        semantic_failure=True, wrong_factor="direction_grounding",
        freeze=INTENT_FIELDS, revise=("direction_grounding",),
    )
    with pytest.raises(NotImplementedError):
        revise_intent(intent, attribution, _dummy_scene())


from babysteps.episode import _diff_intents


def test_diff_intents_detects_grounding_change():
    a = _cv_initial_intent()                                  # actor_frame
    b = Intent.from_dict({**a.to_dict(), "direction_grounding": "observer_frame"})
    assert _diff_intents(a, b) == ("direction_grounding",)
    # And no false positive when nothing changes.
    assert _diff_intents(a, a) == ()


def test_base_observe_demo_is_identity():
    from babysteps.envs.pushcube_adapter import PushCubeAdapter
    adapter = PushCubeAdapter()
    traj = ((0.0, 0.0), (0.1, 0.0))
    correct = Intent(
        goal_state="cube_at_target", object_motion="translate_+x",
        contact_region="minus_x_face", approach_direction="from_minus_x",
        constraint_region="none", embodiment_mapping="proxy_contact_to_franka_push",
    )
    scene = _dummy_scene()
    out_traj, out_contact = adapter.observe_demo(traj, correct, scene)
    assert out_traj == traj
    assert out_contact == "minus_x_face"


def test_crossview_adapter_methods():
    from babysteps.envs.crossview_adapter import CrossViewPushAdapter, observer_yaw_for_seed
    from babysteps.schemas import DemoEvidence

    adapter = CrossViewPushAdapter()
    assert adapter.task_id == "PushCube-v1"

    # scripted_demo_to_intent always grounds in actor_frame (the bug).
    evidence = DemoEvidence(
        camera="third_person", demonstrator_type="proxy_oracle",
        object_trajectory=((0.0, 0.0), (-0.1, 0.0)),     # observer saw -x
        contact_region_label="plus_x_face", final_state="cube_at_target",
        rgbd_video_path=None,
    )
    intent = adapter.scripted_demo_to_intent(evidence)
    assert intent.direction_grounding == "actor_frame"
    assert intent.object_motion == "translate_-x"

    # oracle_wrong_factor: direction_grounding iff rotated + actor_frame.
    rotated = SceneState(
        cube_xy=(0.0, 0.0), cube_z=0.02, goal_xy=(0.1, 0.0),
        tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0),
        blocked_sides=(), extra={"observer_yaw_deg": 180},
    )
    assert adapter.oracle_wrong_factor(intent, rotated) == "direction_grounding"
    unrotated = SceneState(
        cube_xy=(0.0, 0.0), cube_z=0.02, goal_xy=(0.1, 0.0),
        tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0),
        blocked_sides=(), extra={"observer_yaw_deg": 0},
    )
    assert adapter.oracle_wrong_factor(intent, unrotated) == "none"

    # default_blocked_factory is empty (failure is the frame bug, not a block).
    assert adapter.default_blocked_factory(intent) == ()

    # observe_demo rotates the demo into observer frame (-yaw).
    observed_traj, contact = adapter.observe_demo(
        ((0.0, 0.0), (0.1, 0.0)), _cv_world_oracle(), rotated,
    )
    # world +x viewed under 180° appears as -x.
    assert observed_traj[-1][0] < 0
    assert contact == "plus_x_face"

    # observer schedule is deterministic and never 0 (so failures always fire).
    assert observer_yaw_for_seed(0) in (90, 180, 270)


def test_crossview_attribute_failure_maps_to_grounding():
    from babysteps.envs.crossview_adapter import CrossViewPushAdapter
    from babysteps.schemas import FailurePacket
    adapter = CrossViewPushAdapter()
    intent = _cv_initial_intent()
    for predicate in ("direction_error", "goal_not_satisfied"):
        fp = FailurePacket(
            chosen_intent=intent, execution_trace={}, failure_predicate=predicate,
            object_displacement=0.1, direction_alignment=-1.0,
        )
        attr = adapter.attribute_failure(fp)
        assert attr.wrong_factor == "direction_grounding"
        assert attr.revise == ("direction_grounding",)


def _cv_world_oracle() -> Intent:
    return Intent(
        goal_state="cube_at_target", object_motion="translate_+x",
        contact_region="minus_x_face", approach_direction="from_minus_x",
        constraint_region="none", embodiment_mapping="proxy_contact_to_franka_push",
        direction_grounding="actor_frame",
    )
