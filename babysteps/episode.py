"""Stage-0 episode loop: orchestrates demo → intent → execute → fail → revise → retry.

Pure orchestration: no simulator import, no I/O, no PushCube assumptions. The
adapter is injected; every task-specific decision (skill compilation, scripted
demo→intent, blocked-sides factory, oracle wrong-factor labelling, failure
attribution, intent revision) is dispatched through the adapter.

A single `run_episode(...)` call produces one `EpisodeRecord` matching the
shape mandated by `goal.md` §"Episode Data Format" — see `test_episode.py`
for the snapshot guard.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Optional

from babysteps.envs.task_adapter import BaseTaskAdapter, EnvRunner
from babysteps.schemas import (
    CLAIM_BOUNDARY,
    INTENT_FIELDS,
    AttemptResult,
    DemoEvidence,
    EpisodeRecord,
    Intent,
    SceneState,
)


# ---------- demo proxy generation -------------------------------------- #


def generate_proxy_demo(
    env_runner: EnvRunner, scene: SceneState, adapter: BaseTaskAdapter,
) -> DemoEvidence:
    """Run the adapter's oracle scripted skill on `scene` (with blocked_sides=())
    and pack the result into DemoEvidence. The DemoEvidence carries only
    demo-visible quantities — no goal_xy, no blocked_sides."""
    correct = adapter.oracle_correct_intent(scene)
    unblocked = replace(scene, blocked_sides=())
    demo_attempt = env_runner.run(correct, unblocked)
    traj = demo_attempt.trajectory_xy
    if not traj:
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
        num_attempts = 2

    attribution_correct: Optional[bool] = (
        None if wrong_factor_predicted is None
        else wrong_factor_predicted == oracle_wrong_factor
    )

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
    return tuple(f for f in INTENT_FIELDS if getattr(a, f) != getattr(b, f))


# ---------- the loop --------------------------------------------------- #


def run_episode(
    *,
    episode_id: str,
    seed: int,
    adapter: BaseTaskAdapter,
) -> EpisodeRecord:
    """One Stage-0 blocked-approach episode for the adapter's task.

    Steps:
      1. Construct env_runner via adapter (cached). Reset → SceneState (blocked_sides=()).
      2. Generate proxy demo via adapter's oracle.
      3. Derive initial intent via adapter.scripted_demo_to_intent.
      4. Build executor scene with adapter.default_blocked_factory(initial_intent).
      5. Attempt 1: env_runner.run(initial_intent, scene_executor).
      6. Build failure packet via adapter, attribute, revise.
      7. Attempt 2: env_runner.run(revised_intent, scene_executor).
      8. Pack EpisodeRecord with metrics. task = adapter.task_id.

    The adapter owns the env_runner lifecycle. Callers should call
    adapter.close() once they're done with this adapter instance.
    """
    env_runner = adapter.env_runner()      # cached on the adapter
    scene_initial = env_runner.reset(seed)
    demo_evidence = generate_proxy_demo(env_runner, scene_initial, adapter)
    initial_intent = adapter.scripted_demo_to_intent(demo_evidence)
    scene_executor = replace(
        scene_initial,
        blocked_sides=adapter.default_blocked_factory(initial_intent),
    )
    oracle_wrong_factor = adapter.oracle_wrong_factor(
        initial_intent, scene_executor,
    )
    attempt_1 = env_runner.run(initial_intent, scene_executor)
    failure_packet = adapter.build_failure_packet(
        initial_intent, attempt_1, scene_executor,
    )

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
            task=adapter.task_id,
            claim_boundary=CLAIM_BOUNDARY,
            demo=demo_dict,
            execution=execution_dict,
            failure_packet=fp_dict,
            revision=None,
            retry=None,
            metrics=metrics,
        )

    attribution = adapter.attribute_failure(failure_packet)
    try:
        revised_intent, revision_record = adapter.revise_intent(
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
            episode_id=episode_id, stage="stage_0", task=adapter.task_id,
            claim_boundary=CLAIM_BOUNDARY,
            demo=demo_dict, execution=execution_dict, failure_packet=fp_dict,
            revision=None, retry=None, metrics=metrics,
        )

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
        task=adapter.task_id,
        claim_boundary=CLAIM_BOUNDARY,
        demo=demo_dict,
        execution=execution_dict,
        failure_packet=fp_dict,
        revision=revision_dict,
        retry=retry_dict,
        metrics=metrics,
    )
