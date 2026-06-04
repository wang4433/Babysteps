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

import hashlib
import random
from dataclasses import replace
from typing import Any, Callable, Optional

from babysteps.envs.task_adapter import BaseTaskAdapter, EnvRunner
from babysteps.policies import RetryContext, RetryPolicy, babysteps_selective
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
    observed_traj, contact_label = adapter.observe_demo(traj, correct, scene)
    return DemoEvidence(
        camera="third_person",
        demonstrator_type="proxy_oracle",
        object_trajectory=observed_traj,
        contact_region_label=contact_label,
        # Task-aware: read from the oracle intent so each adapter sets its
        # own goal_state label (PushCube: "cube_at_target"; PickCube:
        # "cube_lifted_at_target"). PushCube's value is unchanged →
        # snapshot byte-equality preserved.
        final_state=correct.goal_state,
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
    frozen_factors: Optional[tuple[str, ...]] = None,
) -> dict:
    """Per-episode metrics for the eval/summarize step.

    `frozen_factors` (from Revision.frozen_factors) is the authoritative list
    of factors that must not change.  When supplied (any revision that
    produces a Revision record), frozen_factors_preserved is True iff no
    factor in `frozen_factors` appears in `factors_changed`.  This handles
    both single-factor operators (approach_substitution, contact_substitution,
    goal_refinement) and multi-factor operators (constraint_introduction, which
    revises TWO factors at once).

    When `frozen_factors` is None (no revision was attempted), the metric falls
    back to the legacy single-factor heuristic so existing records are
    unaffected.
    """
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
    elif frozen_factors is not None:
        # Authoritative check: none of the frozen factors should have changed.
        frozen_set = set(frozen_factors)
        frozen_preserved = not any(f in frozen_set for f in factors_changed)
    else:
        # Legacy single-factor fallback (used when no Revision record exists).
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


_DIFF_FIELDS: tuple[str, ...] = INTENT_FIELDS + ("direction_grounding",)


def _diff_intents(a: Intent, b: Intent) -> tuple[str, ...]:
    return tuple(f for f in _DIFF_FIELDS if getattr(a, f) != getattr(b, f))


def _stable_hash(seed: int, salt: str) -> int:
    """Deterministic 32-bit seed from (episode seed, salt)."""
    h = hashlib.sha256(f"{seed}:{salt}".encode()).hexdigest()
    return int(h[:8], 16)


def _baseline_metrics(
    initial: Intent,
    revised: Intent,
    oracle_wrong_factor: str,
    task_valid_tokens: dict,
) -> dict:
    """Label-based baseline metrics (spec §5). Computed only for baseline runs.

    - correct_factor_fixed: the true wrong factor was among the factors the
      retry changed (recovery to a valid alternative need not equal a single
      canonical oracle token).
    - should_preserve: editable factors other than the true wrong factor.
    - harmful_revision: any should-preserve factor changed (was correct → wrong).
    """
    editable = tuple(f for f in INTENT_FIELDS if f in task_valid_tokens)
    should_preserve = tuple(f for f in editable if f != oracle_wrong_factor)
    changed = set(_diff_intents(initial, revised))
    correct_factor_fixed = oracle_wrong_factor in changed
    n_preserved = sum(1 for f in should_preserve if f not in changed)
    harmful = any(f in changed for f in should_preserve)
    return {
        "correct_factor_fixed": bool(correct_factor_fixed),
        "harmful_revision": bool(harmful),
        "n_should_preserve": int(len(should_preserve)),
        "n_preserved": int(n_preserved),
    }


# ---------- the loop --------------------------------------------------- #


def run_episode(
    *,
    episode_id: str,
    seed: int,
    adapter: BaseTaskAdapter,
    policy: RetryPolicy = babysteps_selective,
    record_baseline_metrics: bool = False,
    demo_features_provider: Optional[Callable[[int], Any]] = None,
    initial_intent_provider: Optional[Callable[[int, Intent], Intent]] = None,
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

    Stage-5 P1 hook
    ---------------
    `demo_features_provider`, when given, is a callable
    ``provider(seed: int) -> Z`` that supplies the demo-encoded feature
    vector consumed by latent revision policies. It overrides the
    default handcrafted `extract_episode_features` path (M2a behavior).
    The default `None` preserves the M2a / Stage-4 byte-for-byte path.

    `initial_intent_provider`, when given, is a callable
    ``provider(seed: int, scripted_intent: Intent) -> Intent`` that
    REPLACES the scripted/demo-derived attempt-1 intent with one decoded
    from demo-view vision (Stage-5 latent-input sever, "Sever A"). The
    scripted intent is passed as the base so trivially-constant factors are
    preserved; the provider overwrites only the factors it can ground. This
    hook changes the attempt-1 intent source, not the execution camera: the
    retry loop still runs from the executing robot's first-person evidence.
    The default `None` keeps the scripted path (snapshot byte-equality).
    """
    env_runner = adapter.env_runner()      # cached on the adapter
    scene_initial = env_runner.reset(seed)
    demo_evidence = generate_proxy_demo(env_runner, scene_initial, adapter)
    initial_intent = adapter.scripted_demo_to_intent(demo_evidence)
    if initial_intent_provider is not None:
        initial_intent = initial_intent_provider(seed, initial_intent)
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
    oracle_correct_intent = adapter.oracle_correct_intent(scene_executor)
    # Stage-4 M2a: pre-compute the handcrafted demo encoding so a latent
    # revision policy can read it from RetryContext. Sim-free policies
    # (one_shot, babysteps_selective, …) ignore the field. We import
    # inside the function so episode.py keeps its no-Stage-4 default
    # import surface; the import is cheap and runs once per episode.
    #
    # Stage-5 P1: `demo_features_provider`, when given, overrides this
    # handcrafted path with a vision-grounded vector (e.g. cached DINOv2
    # features keyed by seed). Default `None` preserves M2a behavior.
    if demo_features_provider is not None:
        try:
            demo_features = demo_features_provider(seed)
        except Exception:
            # Provider failure should not abort the episode; fall back
            # to None (latent_revision_factory degrades to a no-op
            # revision in that branch).
            demo_features = None
    else:
        try:
            from babysteps.stage4.features import extract_episode_features as _ef
            demo_features = _ef({"demo": {
                "object_trajectory": [list(p) for p in demo_evidence.object_trajectory],
                "contact_region_label": demo_evidence.contact_region_label,
                "final_state": demo_evidence.final_state,
            }})
        except Exception:
            # Stage-4 not importable here (e.g. fake adapter without contact
            # label in whitelist); leave demo_features None.
            demo_features = None
    ctx = RetryContext(
        initial_intent=initial_intent,
        attribution=attribution,
        scene=scene_executor,
        oracle_correct_intent=oracle_correct_intent,
        oracle_wrong_factor=oracle_wrong_factor,
        task_valid_tokens=adapter.task_valid_tokens(),
        rng=random.Random(_stable_hash(seed, "policy")),
        revise_fn=adapter.revise_intent,
        demo_features=demo_features,
        failure_predicate=failure_packet.failure_predicate,
        failure_packet=failure_packet,
    )

    try:
        proposal = policy(ctx)
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

    if proposal is None:
        # one_shot: no retry. Record the failed initial attempt only.
        fp_dict = {
            "failure_predicate": failure_packet.failure_predicate,
            "wrong_factor": attribution.wrong_factor,
            "oracle_wrong_factor": oracle_wrong_factor,
            "execution_trace": dict(failure_packet.execution_trace),
        }
        metrics = _compute_metrics(
            initial_success=bool(attempt_1.success), retry_success=None,
            failure_predicate=failure_packet.failure_predicate,
            wrong_factor_predicted=attribution.wrong_factor,
            oracle_wrong_factor=oracle_wrong_factor,
            factors_changed=(),
        )
        if record_baseline_metrics:
            metrics.update(_baseline_metrics(
                initial_intent, initial_intent,
                oracle_wrong_factor, adapter.task_valid_tokens()))
        return EpisodeRecord(
            episode_id=episode_id, stage="stage_0", task=adapter.task_id,
            claim_boundary=CLAIM_BOUNDARY,
            demo=demo_dict, execution=execution_dict, failure_packet=fp_dict,
            revision=None, retry=None, metrics=metrics,
        )

    revised_intent, revision_record = proposal
    attempt_2 = env_runner.run(
        revised_intent, scene_executor,
        rollout_seed=_stable_hash(seed, "attempt_2"),
    )
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
        frozen_factors=revision_record.frozen_factors,
    )
    if record_baseline_metrics:
        metrics.update(_baseline_metrics(
            initial_intent, revised_intent,
            oracle_wrong_factor, adapter.task_valid_tokens()))

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
