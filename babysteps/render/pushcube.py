"""PushCube-v1 render_episode — three phases for the Stage-0 MP4 set.

Phase 1 (demo): execute the oracle's correct intent in a fresh seed, capture
all frames.
Phase 2 (attempt_blocked): the demo's approach is blocked; the skill
compiler returns None → planner_failed. The render captures a 'held still'
loop (fps * 2 copies of one initial frame) to convey 'nothing happened'.
Phase 3 (retry): the revised intent (orthogonal approach) succeeds.

Identical semantics to the pre-extraction `_execute_push` /
`_build_waypoints` / main() flow in scripts/render_stage0_maniskill.py."""
from __future__ import annotations

import random
from dataclasses import replace

import numpy as np

from babysteps.envs.task_adapter import BaseTaskAdapter
from babysteps.policies import RetryContext, full_replan_analogue
from babysteps.render.common import (
    PUSHCUBE_MAX_CONTROL_STEPS,
    PHASE_TOL_M,
    prop_action,
    read_obs,
    render_frame,
    to_np,
)
from babysteps.schemas import AttemptResult, DemoEvidence, SceneState
from babysteps.skills.push import build_push_waypoints


def _execute_push(env, waypoints, frames: list, *, seed: int) -> dict:
    """Step through waypoints capturing one frame per step. Re-resets the env
    at the start so demo / attempt / retry all begin from the same scene."""
    obs, _ = env.reset(seed=int(seed))
    targets = [np.asarray(wp[0:3], dtype=np.float64) for wp in waypoints]
    phase_idx = 0
    success = False

    frames.append(render_frame(env))
    for _ in range(PUSHCUBE_MAX_CONTROL_STEPS):
        tcp, cube_xy, _, _ = read_obs(obs)
        target = targets[phase_idx]
        if np.linalg.norm(target - tcp[0:3]) < PHASE_TOL_M:
            phase_idx += 1
            if phase_idx >= len(targets):
                break
            target = targets[phase_idx]
        action = prop_action(tcp, target, gripper_cmd=-1.0)
        obs, _r, term, trunc, info = env.step(action)
        frames.append(render_frame(env))
        term_b = bool(to_np(term).item()) if hasattr(term, "cpu") else bool(term)
        trunc_b = bool(to_np(trunc).item()) if hasattr(trunc, "cpu") else bool(trunc)
        succ = info.get("success", False) if hasattr(info, "get") else False
        success = bool(to_np(succ).item()) if hasattr(succ, "cpu") else bool(succ)
        if success or term_b or trunc_b:
            break

    tcp, final_cube_xy, _, _ = read_obs(obs)
    return {
        "final_obj_xy": (float(final_cube_xy[0]), float(final_cube_xy[1])),
        "success": bool(success),
    }


def _pushcube_setup(env, adapter: BaseTaskAdapter, seed: int) -> dict:
    """Phase-1 demo execution + intent/attribution derivation.

    Shared by render_episode (canonical three-phase) and
    render_baseline_contrast (selective-vs-full_replan retries). Runs the
    oracle demo, derives the initial intent, builds the blocked executor
    scene, and attributes the synthetic planner_failed.
    """
    obs, _ = env.reset(seed=seed)
    tcp_xyzw, cube_xy0, goal_xy, cube_z = read_obs(obs)
    scene = SceneState(
        cube_xy=(float(cube_xy0[0]), float(cube_xy0[1])),
        cube_z=cube_z,
        goal_xy=(float(goal_xy[0]), float(goal_xy[1])),
        tcp_start_pose=tuple(float(v) for v in tcp_xyzw),  # type: ignore[arg-type]
        blocked_sides=(),
    )
    correct_intent = adapter.oracle_correct_intent(scene)
    wp_demo = build_push_waypoints(scene, correct_intent)
    demo_frames: list = []
    out_demo = _execute_push(env, wp_demo, demo_frames, seed=seed)

    demo_evidence = DemoEvidence(
        camera="third_person",
        demonstrator_type="proxy_oracle",
        object_trajectory=(
            (float(cube_xy0[0]), float(cube_xy0[1])),
            out_demo["final_obj_xy"],
        ),
        contact_region_label=correct_intent.contact_region,
        final_state=correct_intent.goal_state,
        rgbd_video_path=None,
    )
    initial_intent = adapter.scripted_demo_to_intent(demo_evidence)
    scene_exec = replace(
        scene, blocked_sides=adapter.default_blocked_factory(initial_intent),
    )

    # Synthetic AttemptResult: planner_failed means no env stepping occurred,
    # so initial_obj_xy and final_obj_xy are both the scene's initial state.
    # The fp/attribution pipeline only uses the predicate flags to derive the
    # wrong_factor — the cube positions are inert here.
    fp = adapter.build_failure_packet(
        initial_intent,
        AttemptResult(
            initial_obj_xy=scene.cube_xy, final_obj_xy=scene.cube_xy,
            goal_xy=scene.goal_xy,
            reached_contact=False, object_moved=False,
            planner_failed=True, collision=False, grasp_slip=False,
            rollout_log_path=None, success=False,
        ),
        scene_exec,
    )
    attribution = adapter.attribute_failure(fp)
    return {
        "scene": scene,
        "scene_exec": scene_exec,
        "correct_intent": correct_intent,
        "initial_intent": initial_intent,
        "attribution": attribution,
        "demo_frames": demo_frames,
    }


def render_episode(
    env, adapter: BaseTaskAdapter, seed: int, fps: int,
) -> tuple[dict, dict]:
    """Run the three-phase BABYSTEPS demo for PushCube and return per-phase
    frame lists and title metadata.

    Returns:
        frames: {"demo": [...], "attempt_blocked": [...], "retry": [...]}
        titles: {"demo": (title, subtitle), ...}
    """
    short_id = f"seed {seed:04d}"
    s = _pushcube_setup(env, adapter, seed)
    correct_intent = s["correct_intent"]
    initial_intent = s["initial_intent"]
    scene_exec = s["scene_exec"]
    demo_frames = s["demo_frames"]

    # === Phase 2 — ATTEMPT 1 (planner_failed, held still) ===
    obs, _ = env.reset(seed=seed)
    attempt1_frames = [render_frame(env)] * (fps * 2)

    # === Phase 3 — RETRY with revised approach (selective) ===
    revised_intent, revision = adapter.revise_intent(
        initial_intent, s["attribution"], scene_exec,
    )
    wp_retry = build_push_waypoints(scene_exec, revised_intent)
    retry_frames: list = []
    out_retry = _execute_push(env, wp_retry, retry_frames, seed=seed)

    demo_title = (
        f"{short_id}  phase 1/3: demo proxy",
        f"contact_region={correct_intent.contact_region}, "
        f"approach={correct_intent.approach_direction}",
    )
    a1_title = (
        f"{short_id}  phase 2/3: approach_blocked",
        f"approach_direction={initial_intent.approach_direction} "
        f"is blocked → planner_failed",
    )
    retry_title = (
        f"{short_id}  phase 3/3: retry (success={out_retry['success']})",
        f"approach_substitution: "
        f"{initial_intent.approach_direction} → "
        f"{revised_intent.approach_direction}  |  "
        f"frozen (preserved): {', '.join(revision.frozen_factors)}",
    )
    return (
        {"demo": demo_frames,
         "attempt_blocked": attempt1_frames,
         "retry": retry_frames},
        {"demo": demo_title,
         "attempt_blocked": a1_title,
         "retry": retry_title},
    )


def render_baseline_contrast(
    env, adapter: BaseTaskAdapter, seed: int, fps: int,
) -> tuple[dict, dict]:
    """Render the PushCube baseline contrast: the same demo + blocked attempt,
    then two retries side-by-side —

      retry_selective:    babysteps_selective — revises approach_direction
                          only, keeps contact_region → pushes toward goal.
      retry_full_replan:  full_replan_analogue — fixes approach_direction AND
                          perturbs contact_region (a collateral edit), so the
                          push goes the wrong way and recovery fails.

    Both retries use the real policies from babysteps.policies, so the clip
    shows the measured behaviour, not a hand-staged failure.
    """
    short_id = f"seed {seed:04d}"
    s = _pushcube_setup(env, adapter, seed)
    correct_intent = s["correct_intent"]
    initial_intent = s["initial_intent"]
    scene_exec = s["scene_exec"]
    attribution = s["attribution"]
    demo_frames = s["demo_frames"]

    # === Phase 2 — ATTEMPT 1 (planner_failed, held still) ===
    obs, _ = env.reset(seed=seed)
    attempt1_frames = [render_frame(env)] * (fps * 2)

    # === Phase 3a — SELECTIVE retry (approach_direction only) ===
    sel_intent, _ = adapter.revise_intent(initial_intent, attribution, scene_exec)
    sel_frames: list = []
    out_sel = _execute_push(
        env, build_push_waypoints(scene_exec, sel_intent), sel_frames, seed=seed,
    )

    # === Phase 3b — FULL_REPLAN retry (approach fixed + contact_region perturbed) ===
    ctx = RetryContext(
        initial_intent=initial_intent,
        attribution=attribution,
        scene=scene_exec,
        oracle_correct_intent=adapter.oracle_correct_intent(scene_exec),
        oracle_wrong_factor=adapter.oracle_wrong_factor(initial_intent, scene_exec),
        task_valid_tokens=adapter.task_valid_tokens(),
        rng=random.Random(seed),
        revise_fn=adapter.revise_intent,
    )
    fr_intent, _ = full_replan_analogue(ctx)
    fr_frames: list = []
    out_fr = _execute_push(
        env, build_push_waypoints(scene_exec, fr_intent), fr_frames, seed=seed,
    )

    demo_title = (
        f"{short_id}  phase 1/4: demo proxy",
        f"contact_region={correct_intent.contact_region}, "
        f"approach={correct_intent.approach_direction}",
    )
    a1_title = (
        f"{short_id}  phase 2/4: approach_blocked",
        f"approach_direction={initial_intent.approach_direction} "
        f"is blocked → planner_failed",
    )
    sel_title = (
        f"{short_id}  phase 3a/4: babysteps_selective (success={out_sel['success']})",
        f"approach_substitution: {initial_intent.approach_direction} → "
        f"{sel_intent.approach_direction}; contact_region preserved "
        f"(={sel_intent.contact_region})",
    )
    fr_title = (
        f"{short_id}  phase 3b/4: full_replan_analogue (success={out_fr['success']})",
        f"approach fixed BUT contact_region: {initial_intent.contact_region} -> "
        f"{fr_intent.contact_region} (collateral edit → wrong-way push)",
    )
    return (
        {"demo": demo_frames,
         "attempt_blocked": attempt1_frames,
         "retry_selective": sel_frames,
         "retry_full_replan": fr_frames},
        {"demo": demo_title,
         "attempt_blocked": a1_title,
         "retry_selective": sel_title,
         "retry_full_replan": fr_title},
    )
