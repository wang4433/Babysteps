"""Stage-0 episode loop: orchestrates demo → intent → execute → fail → revise → retry.

Pure orchestration: no simulator import, no I/O. The env_runner is injected;
the loop never reaches past its `reset` / `run` methods.

A single `run_episode(...)` call produces one `EpisodeRecord` matching the
shape mandated by `goal.md` §"Episode Data Format" — see `test_episode.py`
for the snapshot guard.
"""
from __future__ import annotations

from typing import Callable, Optional, Protocol

import numpy as np

from babysteps.demo import demo_to_intent
from babysteps.envs.scene import direction_to_face, face_to_approach
from babysteps.failure import attribute_failure, build_failure_packet
from babysteps.revision import revise_intent
from babysteps.schemas import (
    CLAIM_BOUNDARY,
    INTENT_FIELDS,
    AttemptResult,
    DemoEvidence,
    EpisodeRecord,
    Intent,
    SceneState,
)


class EnvRunner(Protocol):
    """Minimal env_runner contract. Both the fake (`tests/conftest.py`) and
    the real ManiSkill runner (`babysteps/envs/pushcube_runner.py`) implement
    this shape."""

    def reset(self, seed: int) -> SceneState: ...
    def run(self, intent: Intent, scene: SceneState) -> AttemptResult: ...
    def close(self) -> None: ...


BlockedSidesFactory = Callable[[Intent], tuple[str, ...]]


def _default_blocked_sides_factory(intent: Intent) -> tuple[str, ...]:
    """The Stage-0 default: block the demo's preferred approach direction —
    deterministic given the demo's intent, independent of seed."""
    return (intent.approach_direction,)


# ---------- demo proxy generation -------------------------------------- #


def _oracle_correct_intent_for_scene(scene: SceneState) -> Intent:
    """Reads privileged state to construct the correct intent. Used only
    inside `generate_proxy_demo` — the GENERATOR, not the EXTRACTOR. The
    privileged-firewall (goal.md §5) is on `demo_to_intent`, not here."""
    goal_vec = np.array(scene.goal_xy) - np.array(scene.cube_xy)
    face = direction_to_face(goal_vec)
    approach = face_to_approach(face)
    if abs(goal_vec[0]) >= abs(goal_vec[1]):
        motion = "translate_+x" if goal_vec[0] >= 0 else "translate_-x"
    else:
        motion = "translate_+y" if goal_vec[1] >= 0 else "translate_-y"
    return Intent(
        goal_state="cube_at_target",
        object_motion=motion,
        contact_region=face,
        approach_direction=approach,
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_push",
    )


def generate_proxy_demo(env_runner: EnvRunner, scene: SceneState) -> DemoEvidence:
    """Run the oracle scripted push on `scene` (with blocked_sides=()) and
    pack the result into DemoEvidence. The DemoEvidence carries only
    demo-visible quantities — no goal_xy, no blocked_sides."""
    correct = _oracle_correct_intent_for_scene(scene)
    unblocked = SceneState(
        cube_xy=scene.cube_xy, cube_z=scene.cube_z, goal_xy=scene.goal_xy,
        tcp_start_pose=scene.tcp_start_pose, blocked_sides=(),
    )
    demo_attempt = env_runner.run(correct, unblocked)
    traj = demo_attempt.trajectory_xy
    if not traj:
        # Fallback: build a 2-point trajectory from start/end.
        traj = (demo_attempt.initial_obj_xy, demo_attempt.final_obj_xy)
    return DemoEvidence(
        camera="third_person",
        demonstrator_type="proxy_oracle",
        object_trajectory=traj,
        contact_region_label=correct.contact_region,
        final_state="cube_at_target",
        rgbd_video_path=None,
    )


# ---------- per-episode metrics ---------------------------------------- #


def _compute_metrics(
    *,
    initial_success: bool,
    retry_success: Optional[bool],
    failure_predicate: str,
    wrong_factor_predicted: Optional[str],
    oracle_wrong_factor: str,
    factors_changed: tuple[str, ...],
) -> dict:
    """Per-episode metrics for the eval/summarize step."""
    if not initial_success and retry_success is True:
        num_attempts = 2
    elif initial_success:
        num_attempts = 1
    else:
        num_attempts = 2  # failed twice — capped at MAX_ATTEMPTS

    attribution_correct: Optional[bool] = (
        None if wrong_factor_predicted is None
        else wrong_factor_predicted == oracle_wrong_factor
    )

    # frozen_factors_preserved: True iff factors_changed is a subset of
    # the predicted-revise set. For Stage-0 (single-factor revision) this
    # is just "factors_changed has length ≤ 1 and equals (wrong_factor,)".
    frozen_preserved: Optional[bool]
    if wrong_factor_predicted is None:
        frozen_preserved = None
    else:
        frozen_preserved = (
            tuple(factors_changed) == (wrong_factor_predicted,)
            or len(factors_changed) == 0
        )

    return {
        "initial_success":           bool(initial_success),
        "retry_success":             retry_success,
        "num_attempts_to_success":   int(num_attempts),
        "failure_type":              failure_predicate,
        "wrong_factor_predicted":    wrong_factor_predicted,
        "oracle_wrong_factor":       oracle_wrong_factor,
        "factor_attribution_correct": attribution_correct,
        "factors_changed":           list(factors_changed),
        "frozen_factors_preserved":  frozen_preserved,
    }


def _diff_intents(a: Intent, b: Intent) -> tuple[str, ...]:
    """Names of factors that differ between two Intents."""
    return tuple(f for f in INTENT_FIELDS if getattr(a, f) != getattr(b, f))


# ---------- the loop --------------------------------------------------- #


def run_episode(
    *,
    episode_id: str,
    seed: int,
    env_runner: EnvRunner,
    blocked_sides_factory: BlockedSidesFactory = _default_blocked_sides_factory,
) -> EpisodeRecord:
    """One Stage-0 PushCube blocked-approach episode.

    Steps:
      1. Reset → SceneState (blocked_sides=()).
      2. Generate proxy demo via oracle scripted push.
      3. Derive initial intent from demo evidence (no SceneState read).
      4. Build executor scene with blocked_sides_factory(initial_intent).
      5. Attempt 1: env_runner.run(initial_intent, scene_executor).
      6. Build failure packet → attribute → if semantic, revise.
      7. Attempt 2: env_runner.run(revised_intent, scene_executor).
      8. Pack EpisodeRecord with metrics.
    """
    # 1.
    scene_initial = env_runner.reset(seed)
    # 2.
    demo_evidence = generate_proxy_demo(env_runner, scene_initial)
    # 3. PRIVILEGED FIREWALL: demo_to_intent takes ONLY the DemoEvidence.
    initial_intent = demo_to_intent(demo_evidence)
    # 4.
    scene_executor = SceneState(
        cube_xy=scene_initial.cube_xy,
        cube_z=scene_initial.cube_z,
        goal_xy=scene_initial.goal_xy,
        tcp_start_pose=scene_initial.tcp_start_pose,
        blocked_sides=blocked_sides_factory(initial_intent),
    )
    # Oracle wrong factor: by Stage-0 construction it's the factor blocked.
    oracle_wrong_factor = (
        "approach_direction"
        if initial_intent.approach_direction in scene_executor.blocked_sides
        else "none"
    )
    # 5.
    attempt_1 = env_runner.run(initial_intent, scene_executor)
    failure_packet = build_failure_packet(initial_intent, attempt_1, scene_executor)

    # Demo / execution / failure_packet dicts always present.
    demo_dict = {
        "camera": demo_evidence.camera,
        "rgbd_video": demo_evidence.rgbd_video_path,
        "object_trajectory": [list(p) for p in demo_evidence.object_trajectory],
        "contact_region_label": demo_evidence.contact_region_label,
        "final_state": demo_evidence.final_state,
        "demonstrator_type": demo_evidence.demonstrator_type,
    }
    execution_dict = {
        "camera": "robot_first_person",
        "robot": "Franka",
        "initial_intent": initial_intent.to_dict(),
        "success": bool(attempt_1.success),
    }

    if failure_packet.failure_predicate == "none":
        # Happy initial path.
        fp_dict = {
            "failure_predicate": "none",
            "wrong_factor": None,
            "oracle_wrong_factor": oracle_wrong_factor,
            "execution_trace": dict(failure_packet.execution_trace),
        }
        metrics = _compute_metrics(
            initial_success=True,
            retry_success=None,
            failure_predicate="none",
            wrong_factor_predicted=None,
            oracle_wrong_factor=oracle_wrong_factor,
            factors_changed=(),
        )
        return EpisodeRecord(
            episode_id=episode_id,
            stage="stage_0",
            task="PushCube-v1",
            claim_boundary=CLAIM_BOUNDARY,
            demo=demo_dict,
            execution=execution_dict,
            failure_packet=fp_dict,
            revision=None,
            retry=None,
            metrics=metrics,
        )

    # 6. Attribute.
    attribution = attribute_failure(failure_packet)
    # 7. Revise (Stage 0 only handles approach_direction; other factors raise
    # NotImplementedError — caught here and reported as an unrevised failure).
    try:
        revised_intent, revision_record = revise_intent(
            initial_intent, attribution, scene_executor,
        )
    except NotImplementedError as exc:
        fp_dict = {
            "failure_predicate": failure_packet.failure_predicate,
            "wrong_factor": attribution.wrong_factor,
            "oracle_wrong_factor": oracle_wrong_factor,
            "execution_trace": dict(failure_packet.execution_trace),
            "revision_error": str(exc),
        }
        metrics = _compute_metrics(
            initial_success=False, retry_success=False,
            failure_predicate=failure_packet.failure_predicate,
            wrong_factor_predicted=attribution.wrong_factor,
            oracle_wrong_factor=oracle_wrong_factor,
            factors_changed=(),
        )
        return EpisodeRecord(
            episode_id=episode_id, stage="stage_0", task="PushCube-v1",
            claim_boundary=CLAIM_BOUNDARY,
            demo=demo_dict, execution=execution_dict, failure_packet=fp_dict,
            revision=None, retry=None, metrics=metrics,
        )

    # 8. Retry.
    attempt_2 = env_runner.run(revised_intent, scene_executor)
    factors_changed = _diff_intents(initial_intent, revised_intent)

    fp_dict = {
        "failure_predicate": failure_packet.failure_predicate,
        "wrong_factor": attribution.wrong_factor,
        "oracle_wrong_factor": oracle_wrong_factor,
        "execution_trace": dict(failure_packet.execution_trace),
        "object_displacement": failure_packet.object_displacement,
        "direction_alignment": failure_packet.direction_alignment,
        "freeze": list(attribution.freeze),
        "revise": list(attribution.revise),
    }
    revision_dict = revision_record.to_dict()
    retry_dict = {
        "success": bool(attempt_2.success),
        "num_retries": 1,
        "final_intent": revised_intent.to_dict(),
    }
    metrics = _compute_metrics(
        initial_success=bool(attempt_1.success),
        retry_success=bool(attempt_2.success),
        failure_predicate=failure_packet.failure_predicate,
        wrong_factor_predicted=attribution.wrong_factor,
        oracle_wrong_factor=oracle_wrong_factor,
        factors_changed=factors_changed,
    )

    return EpisodeRecord(
        episode_id=episode_id,
        stage="stage_0",
        task="PushCube-v1",
        claim_boundary=CLAIM_BOUNDARY,
        demo=demo_dict,
        execution=execution_dict,
        failure_packet=fp_dict,
        revision=revision_dict,
        retry=retry_dict,
        metrics=metrics,
    )
