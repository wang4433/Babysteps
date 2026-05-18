"""Tests for babysteps.schemas — pure data contracts.

Asserts the data shapes from
docs/superpowers/specs/2026-05-15-stage0-pushcube-blocked-design.md §5
and the JSONL line shape matches goal.md "Episode Data Format".
"""
from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

from babysteps.schemas import (
    APPROACH_DIRECTIONS,
    CLAIM_BOUNDARY,
    CONTACT_REGIONS,
    EMBODIMENT_MAPPINGS,
    FAILURE_PREDICATES,
    GOAL_STATES,
    INTENT_FIELDS,
    OBJECT_MOTIONS,
    REVISION_OPERATORS,
    AttemptResult,
    DemoEvidence,
    EpisodeRecord,
    FailurePacket,
    Intent,
    Revision,
    SceneState,
)


# ---------- Module-level constants ---------------------------------------- #


def test_intent_fields_exact():
    assert INTENT_FIELDS == (
        "goal_state",
        "object_motion",
        "contact_region",
        "approach_direction",
        "constraint_region",
        "embodiment_mapping",
    )


def test_claim_boundary_string():
    assert CLAIM_BOUNDARY == "third_person_demo_proxy_not_human_demo"


def test_whitelists_disjoint_and_present():
    assert "minus_x_face" in CONTACT_REGIONS
    assert "from_minus_x" in APPROACH_DIRECTIONS
    assert "from_above" in APPROACH_DIRECTIONS
    assert "translate_+x" in OBJECT_MOTIONS
    assert "proxy_contact_to_franka_push" in EMBODIMENT_MAPPINGS
    assert {"approach_blocked", "none"} <= FAILURE_PREDICATES
    assert "approach_substitution" in REVISION_OPERATORS


def test_whitelists_pickcube_additions():
    """B (PickCube) additions — see four-scene roadmap spec §4."""
    assert "lift_up" in OBJECT_MOTIONS
    assert "proxy_contact_to_franka_grasp" in EMBODIMENT_MAPPINGS
    assert "cube_lifted_at_target" in GOAL_STATES
    assert "grasp_slip" in FAILURE_PREDICATES
    assert "contact_substitution" in REVISION_OPERATORS
    # CONTACT_REGIONS contains the 4 cardinal cube faces (Pick reuses them).
    # D (TurnFaucet) adds "faucet_base" and "handle_grip" to the same set.
    assert {"minus_x_face", "plus_x_face", "minus_y_face", "plus_y_face"} <= CONTACT_REGIONS


# ---------- Intent ------------------------------------------------------- #


def _ok_intent() -> Intent:
    return Intent(
        goal_state="cube_at_target",
        object_motion="translate_+x",
        contact_region="minus_x_face",
        approach_direction="from_minus_x",
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_push",
    )


def test_intent_roundtrip():
    i = _ok_intent()
    assert Intent.from_dict(i.to_dict()) == i


def _ok_pickcube_intent() -> Intent:
    return Intent(
        goal_state="cube_lifted_at_target",
        object_motion="lift_up",
        contact_region="minus_x_face",  # x-axis-aligned gripper
        approach_direction="from_above",
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_grasp",
    )


def test_pickcube_intent_roundtrip():
    i = _ok_pickcube_intent()
    assert Intent.from_dict(i.to_dict()) == i


def test_intent_is_frozen():
    i = _ok_intent()
    with pytest.raises(FrozenInstanceError):
        i.contact_region = "plus_x_face"  # type: ignore[misc]


def test_intent_rejects_unknown_contact_region():
    with pytest.raises(ValueError, match="contact_region"):
        Intent(
            goal_state="cube_at_target",
            object_motion="translate_+x",
            contact_region="not_a_face",
            approach_direction="from_minus_x",
            constraint_region="none",
            embodiment_mapping="proxy_contact_to_franka_push",
        )


def test_intent_rejects_unknown_approach_direction():
    with pytest.raises(ValueError, match="approach_direction"):
        Intent(
            goal_state="cube_at_target",
            object_motion="translate_+x",
            contact_region="minus_x_face",
            approach_direction="from_somewhere",
            constraint_region="none",
            embodiment_mapping="proxy_contact_to_franka_push",
        )


def test_intent_rejects_unknown_object_motion():
    with pytest.raises(ValueError, match="object_motion"):
        Intent(
            goal_state="cube_at_target",
            object_motion="spin_like_a_top",
            contact_region="minus_x_face",
            approach_direction="from_minus_x",
            constraint_region="none",
            embodiment_mapping="proxy_contact_to_franka_push",
        )


def test_intent_rejects_unknown_embodiment_mapping():
    with pytest.raises(ValueError, match="embodiment_mapping"):
        Intent(
            goal_state="cube_at_target",
            object_motion="translate_+x",
            contact_region="minus_x_face",
            approach_direction="from_minus_x",
            constraint_region="none",
            embodiment_mapping="cyborg_telekinesis",
        )


# ---------- DemoEvidence ------------------------------------------------- #


def _ok_demo_evidence() -> DemoEvidence:
    return DemoEvidence(
        camera="third_person",
        demonstrator_type="proxy_oracle",
        object_trajectory=((0.0, 0.0), (0.05, 0.0), (0.10, 0.0)),
        contact_region_label="minus_x_face",
        final_state="cube_at_target",
        rgbd_video_path=None,
    )


def test_demo_evidence_roundtrip_null_video():
    d = _ok_demo_evidence()
    rt = DemoEvidence.from_dict(d.to_dict())
    assert rt == d
    assert rt.rgbd_video_path is None


def test_demo_evidence_roundtrip_with_video():
    d = DemoEvidence(
        camera="third_person",
        demonstrator_type="proxy_oracle",
        object_trajectory=((0.0, 0.0),),
        contact_region_label="minus_x_face",
        final_state="cube_at_target",
        rgbd_video_path="data/demos/x/demo_rgbd.mp4",
    )
    assert DemoEvidence.from_dict(d.to_dict()) == d


# ---------- SceneState --------------------------------------------------- #


def _ok_scene() -> SceneState:
    return SceneState(
        cube_xy=(0.0, 0.0),
        cube_z=0.02,
        goal_xy=(0.2, 0.05),
        tcp_start_pose=(0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0),
        blocked_sides=("from_minus_x",),
    )


def test_scene_roundtrip_tuple_blocked_sides():
    s = _ok_scene()
    rt = SceneState.from_dict(s.to_dict())
    assert rt == s
    assert isinstance(rt.blocked_sides, tuple)


def test_scene_roundtrip_with_extra():
    s = SceneState(
        cube_xy=(0.0, 0.0),
        cube_z=0.02,
        goal_xy=(0.2, 0.05),
        tcp_start_pose=(0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0),
        blocked_sides=(),
        extra={"gripper_width": 0.08, "base_cube_xy": [0.1, 0.0]},
    )
    rt = SceneState.from_dict(s.to_dict())
    assert rt.extra == {"gripper_width": 0.08, "base_cube_xy": [0.1, 0.0]}


def test_scene_empty_extra_omitted_from_json():
    """Empty extra must NOT appear as a key in to_dict — this is what
    preserves byte-for-byte JSON equality for pre-A PushCube records."""
    s = SceneState(
        cube_xy=(0.0, 0.0), cube_z=0.02, goal_xy=(0.2, 0.05),
        tcp_start_pose=(0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0),
        blocked_sides=(),
    )
    d = s.to_dict()
    assert "extra" not in d
    # And the default is an empty dict, round-trippable.
    rt = SceneState.from_dict(d)
    assert rt.extra == {}


def test_scene_default_extra_is_empty_dict():
    s = SceneState(
        cube_xy=(0.0, 0.0), cube_z=0.02, goal_xy=(0.2, 0.05),
        tcp_start_pose=(0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0),
        blocked_sides=(),
    )
    assert s.extra == {}


# ---------- AttemptResult ------------------------------------------------ #


def _ok_attempt() -> AttemptResult:
    return AttemptResult(
        initial_obj_xy=(0.0, 0.0),
        final_obj_xy=(0.0, 0.0),
        goal_xy=(0.2, 0.05),
        reached_contact=False,
        object_moved=False,
        planner_failed=True,
        collision=False,
        grasp_slip=False,
        rollout_log_path=None,
        success=False,
    )


def test_attempt_roundtrip():
    a = _ok_attempt()
    assert AttemptResult.from_dict(a.to_dict()) == a


# ---------- FailurePacket ------------------------------------------------ #


def test_failure_packet_roundtrip_preserves_intent():
    fp = FailurePacket(
        chosen_intent=_ok_intent(),
        execution_trace={
            "reached_contact": False, "object_moved": False, "collision": False,
            "planner_failed": True,   "grasp_slip": False,
        },
        failure_predicate="approach_blocked",
        object_displacement=0.0,
        direction_alignment=None,
    )
    rt = FailurePacket.from_dict(fp.to_dict())
    assert rt.chosen_intent == fp.chosen_intent
    assert rt.failure_predicate == "approach_blocked"
    assert rt.execution_trace == fp.execution_trace


def test_failure_packet_rejects_unknown_predicate():
    with pytest.raises(ValueError, match="failure_predicate"):
        FailurePacket(
            chosen_intent=_ok_intent(),
            execution_trace={},
            failure_predicate="cube_exploded",
            object_displacement=None,
            direction_alignment=None,
        )


# ---------- Revision ----------------------------------------------------- #


def test_revision_roundtrip_frozen_factors_is_tuple():
    r = Revision(
        operator="approach_substitution",
        factor="approach_direction",
        old_value="from_minus_x",
        new_value="from_plus_x",
        frozen_factors=(
            "goal_state", "object_motion", "contact_region",
            "constraint_region", "embodiment_mapping",
        ),
    )
    rt = Revision.from_dict(r.to_dict())
    assert rt == r
    assert isinstance(rt.frozen_factors, tuple)


def test_revision_rejects_unknown_operator():
    with pytest.raises(ValueError, match="operator"):
        Revision(
            operator="rebuild_universe",
            factor="approach_direction",
            old_value="from_minus_x",
            new_value="from_plus_x",
            frozen_factors=(),
        )


# ---------- EpisodeRecord ------------------------------------------------ #


def _ok_episode_record() -> EpisodeRecord:
    return EpisodeRecord(
        episode_id="pushcube_blocked_approach_seed_0001",
        stage="stage_0",
        task="PushCube-v1",
        claim_boundary=CLAIM_BOUNDARY,
        demo={
            "camera": "third_person",
            "rgbd_video": None,
            "object_trajectory": [[0.0, 0.0], [0.05, 0.0], [0.10, 0.0]],
            "contact_region_label": "minus_x_face",
            "final_state": "cube_at_target",
            "demonstrator_type": "proxy_oracle",
        },
        execution={
            "camera": "robot_first_person",
            "robot": "Franka",
            "initial_intent": _ok_intent().to_dict(),
            "success": False,
        },
        failure_packet={
            "failure_predicate": "approach_blocked",
            "wrong_factor": "approach_direction",
            "oracle_wrong_factor": "approach_direction",
            "execution_trace": {
                "reached_contact": False, "object_moved": False, "collision": False,
                "planner_failed": True,   "grasp_slip": False,
            },
        },
        revision={
            "operator": "approach_substitution",
            "factor": "approach_direction",
            "old_value": "from_minus_x",
            "new_value": "from_plus_x",
            "frozen_factors": [
                "goal_state", "object_motion", "contact_region",
                "constraint_region", "embodiment_mapping",
            ],
        },
        retry={"success": True, "num_retries": 1},
        metrics={
            "initial_success": False,
            "retry_success": True,
            "num_attempts_to_success": 2,
            "failure_type": "approach_blocked",
            "wrong_factor_predicted": "approach_direction",
            "oracle_wrong_factor": "approach_direction",
            "factor_attribution_correct": True,
            "factors_changed": ["approach_direction"],
            "frozen_factors_preserved": True,
        },
    )


def test_episode_record_jsonl_roundtrip():
    rec = _ok_episode_record()
    line = rec.to_jsonl_line()
    assert "\n" not in line
    parsed = json.loads(line)
    rt = EpisodeRecord.from_jsonl_line(line)
    assert rt.episode_id == rec.episode_id
    assert rt.demo == rec.demo
    # Privileged-firewall canary: the demo dict has no goal_xy.
    assert "goal_xy" not in rec.demo


def test_episode_record_top_level_keys_match_goal_md():
    """Snapshot guard — these keys appear in goal.md §"Episode Data Format"."""
    rec = _ok_episode_record()
    d = rec.to_dict()
    expected = {
        "episode_id", "stage", "task", "claim_boundary",
        "demo", "execution", "failure_packet", "revision", "retry", "metrics",
    }
    assert set(d.keys()) == expected


def test_episode_record_no_failure_path():
    """If the first attempt succeeds, revision and retry are None and the
    record still round-trips."""
    rec = EpisodeRecord(
        episode_id="pushcube_no_fail_seed_0000",
        stage="stage_0",
        task="PushCube-v1",
        claim_boundary=CLAIM_BOUNDARY,
        demo=_ok_episode_record().demo,
        execution=_ok_episode_record().execution | {"success": True},
        failure_packet={
            "failure_predicate": "none",
            "wrong_factor": None,
            "oracle_wrong_factor": "approach_direction",
            "execution_trace": {},
        },
        revision=None,
        retry=None,
        metrics={"initial_success": True, "retry_success": None,
                 "num_attempts_to_success": 1, "failure_type": "none",
                 "wrong_factor_predicted": None, "oracle_wrong_factor": "approach_direction",
                 "factor_attribution_correct": None, "factors_changed": [],
                 "frozen_factors_preserved": None},
    )
    line = rec.to_jsonl_line()
    rt = EpisodeRecord.from_jsonl_line(line)
    assert rt.revision is None
    assert rt.retry is None


# ---------- Sub-project C (StackCube) whitelist additions ----------- #


def test_goal_states_contains_cubeA_on_cubeB():
    from babysteps.schemas import GOAL_STATES
    assert "cubeA_on_cubeB" in GOAL_STATES


def test_object_motions_contains_place_on():
    from babysteps.schemas import OBJECT_MOTIONS
    assert "place_on" in OBJECT_MOTIONS


def test_embodiment_mappings_contains_pick_and_place():
    from babysteps.schemas import EMBODIMENT_MAPPINGS
    assert "proxy_contact_to_franka_pick_and_place" in EMBODIMENT_MAPPINGS


def test_revision_operators_contains_goal_refinement():
    from babysteps.schemas import REVISION_OPERATORS
    assert "goal_refinement" in REVISION_OPERATORS


# ---------- Sub-project D (TurnFaucet) whitelist additions ----------- #


def test_goal_states_contains_faucet_turned():
    from babysteps.schemas import GOAL_STATES
    assert "faucet_turned" in GOAL_STATES


def test_object_motions_contains_turn():
    from babysteps.schemas import OBJECT_MOTIONS
    assert "turn" in OBJECT_MOTIONS


def test_contact_regions_contains_faucet_base():
    from babysteps.schemas import CONTACT_REGIONS
    assert "faucet_base" in CONTACT_REGIONS


def test_contact_regions_contains_handle_grip():
    from babysteps.schemas import CONTACT_REGIONS
    assert "handle_grip" in CONTACT_REGIONS


def test_constraint_regions_contains_faucet_base_static():
    from babysteps.schemas import CONSTRAINT_REGIONS
    assert "faucet_base_static" in CONSTRAINT_REGIONS


def test_failure_predicates_contains_constraint_violation():
    from babysteps.schemas import FAILURE_PREDICATES
    assert "constraint_violation" in FAILURE_PREDICATES


def test_revision_operators_contains_constraint_introduction():
    from babysteps.schemas import REVISION_OPERATORS
    assert "constraint_introduction" in REVISION_OPERATORS


def test_embodiment_mappings_contains_franka_turn():
    from babysteps.schemas import EMBODIMENT_MAPPINGS
    assert "proxy_contact_to_franka_turn" in EMBODIMENT_MAPPINGS


# ---------- Sub-project D reframe (embodiment_substitution) tokens ------- #


def test_embodiment_grasp_turn_token():
    from babysteps.schemas import EMBODIMENT_MAPPINGS
    assert "proxy_contact_to_franka_grasp_turn" in EMBODIMENT_MAPPINGS


def test_embodiment_poke_turn_token():
    from babysteps.schemas import EMBODIMENT_MAPPINGS
    assert "proxy_contact_to_franka_poke_turn" in EMBODIMENT_MAPPINGS


def test_grasp_infeasible_predicate_token():
    from babysteps.schemas import FAILURE_PREDICATES
    assert "grasp_infeasible" in FAILURE_PREDICATES


def test_embodiment_substitution_operator_token():
    from babysteps.schemas import REVISION_OPERATORS
    assert "embodiment_substitution" in REVISION_OPERATORS


def test_old_d_tokens_remain_deprecated_but_present():
    """Per spec §4: additive only. Deprecated tokens stay in whitelists
    until a separate cleanup commit proves no references remain."""
    from babysteps.schemas import (
        EMBODIMENT_MAPPINGS, CONTACT_REGIONS, CONSTRAINT_REGIONS,
        FAILURE_PREDICATES, REVISION_OPERATORS,
    )
    assert "proxy_contact_to_franka_turn" in EMBODIMENT_MAPPINGS
    assert "faucet_base" in CONTACT_REGIONS
    assert "faucet_base_static" in CONSTRAINT_REGIONS
    assert "constraint_violation" in FAILURE_PREDICATES
    assert "constraint_introduction" in REVISION_OPERATORS
