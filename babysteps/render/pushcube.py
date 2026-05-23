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
    render_wrist_frame,
    to_np,
)
from babysteps.schemas import AttemptResult, DemoEvidence, SceneState
from babysteps.skills.push import build_push_waypoints


# Per-phase pos_scale for the 4-waypoint push (approach, pre_contact_high,
# descend, push). Larger values → smaller saturated velocity → lower contact
# impulse. Approach + pre-contact stay at the legacy 0.10 (fast travel);
# descend and push are damped so contact with the cube is gentle and the
# cube does not fly. Tunable; eyeball-checked on seed 0.
_PUSHCUBE_POS_SCALE: tuple[float, ...] = (0.10, 0.10, 0.40, 0.50)


# Obstacle (phase-2 blocked-side wall) — half-extents in meters.
_OBSTACLE_HALF_W: float = 0.020   # along the approach axis (0.04 m total)
_OBSTACLE_HALF_T: float = 0.0025  # perpendicular to approach (0.005 m total)
_OBSTACLE_HALF_H: float = 0.050   # vertical (0.10 m total) — clears EE travel z
_OBSTACLE_PARK_Z: float = -0.50   # below table plane; invisible / out of the way
_OBSTACLE_BLOCK_MARGIN_M: float = 0.025  # gap between cube edge and wall face


def _get_or_build_obstacle(env):
    """Spawn (once per env) a static red box obstacle, parked below the
    table. Returns None when the env does not support actor building
    (sim-free stub envs).

    Cached on `env._babysteps_obstacle` so repeated render_episode calls
    on the same env reuse the same actor rather than accumulating walls.
    """
    cached = getattr(env, "_babysteps_obstacle",
                     getattr(env.unwrapped, "_babysteps_obstacle", None))
    if cached is not None:
        return cached
    scene = getattr(env.unwrapped, "scene", None)
    if scene is None or not hasattr(scene, "create_actor_builder"):
        return None  # sim-free stub env: helpers no-op below
    import sapien

    builder = scene.create_actor_builder()
    half = [_OBSTACLE_HALF_W, _OBSTACLE_HALF_T, _OBSTACLE_HALF_H]
    builder.add_box_collision(half_size=half)
    builder.add_box_visual(
        half_size=half,
        material=sapien.render.RenderMaterial(base_color=[0.78, 0.20, 0.20, 1.0]),
    )
    builder.initial_pose = sapien.Pose(p=[0.0, 0.0, _OBSTACLE_PARK_Z])
    actor = builder.build_static(name="approach_obstacle")
    try:
        env._babysteps_obstacle = actor
    except AttributeError:
        # Some env wrappers reject attribute assignment; fall back to
        # caching on env.unwrapped (best-effort).
        env.unwrapped._babysteps_obstacle = actor
    return actor


def _move_obstacle_to_block(obstacle, cube_xy, cube_z, intent) -> None:
    """Place the obstacle on the blocked side of the cube, on the EE's
    approach path. No-op when obstacle is None."""
    if obstacle is None:
        return
    import sapien
    from babysteps.envs.scene import approach_to_unit
    from babysteps.skills.push import CUBE_HALF_SIZE
    unit = approach_to_unit(intent.approach_direction)
    margin = CUBE_HALF_SIZE + _OBSTACLE_BLOCK_MARGIN_M
    x = float(cube_xy[0]) + float(unit[0]) * margin
    y = float(cube_xy[1]) + float(unit[1]) * margin
    z = float(cube_z) + _OBSTACLE_HALF_H  # base at cube_z, center at cube_z + half_h
    obstacle.set_pose(sapien.Pose(
        p=[x, y, z],
        q=[1.0, 0.0, 0.0, 0.0],
    ))


def _park_obstacle(obstacle) -> None:
    """Move the obstacle far below the table — invisible, no contact.
    No-op when obstacle is None."""
    if obstacle is None:
        return
    import sapien
    obstacle.set_pose(sapien.Pose(
        p=[0.0, 0.0, _OBSTACLE_PARK_Z],
        q=[1.0, 0.0, 0.0, 0.0],
    ))


def _execute_push(
    env, waypoints, frames: list, *,
    seed: int,
    capture=render_frame,
    max_steps: int = PUSHCUBE_MAX_CONTROL_STEPS,
    no_progress_break_steps: int | None = None,
    no_progress_eps_m: float = 0.002,
) -> dict:
    """Step through waypoints capturing one frame per step. Re-resets the env
    at the start so demo / attempt / retry all begin from the same scene.

    `capture` selects the view: render_frame (third-person external camera,
    the demo view) or render_wrist_frame (first-person panda_wristcam, the
    execution view).

    `max_steps` caps the control-step budget (default PUSHCUBE_MAX_CONTROL_STEPS,
    matching the runner). Phase 2's blocked attempt uses a shorter budget so
    the stalled clip stays a few seconds rather than the full ~10s cap.

    `no_progress_break_steps` (default None = disabled) exits the loop when
    the TCP has moved less than `no_progress_eps_m` for that many
    consecutive steps — used in phase 2 to detect 'arm stalled against the
    obstacle' and end the clip early."""
    obs, _ = env.reset(seed=int(seed))
    targets = [np.asarray(wp[0:3], dtype=np.float64) for wp in waypoints]
    phase_idx = 0
    success = False

    frames.append(capture(env))
    prev_tcp_xyz: np.ndarray | None = None
    stalled_steps = 0
    for _ in range(max_steps):
        tcp, cube_xy, _, _ = read_obs(obs)
        target = targets[phase_idx]
        if np.linalg.norm(target - tcp[0:3]) < PHASE_TOL_M:
            phase_idx += 1
            if phase_idx >= len(targets):
                break
            target = targets[phase_idx]
        action = prop_action(
            tcp, target, gripper_cmd=-1.0,
            pos_scale=_PUSHCUBE_POS_SCALE[phase_idx],
        )
        obs, _r, term, trunc, info = env.step(action)
        frames.append(capture(env))
        term_b = bool(to_np(term).item()) if hasattr(term, "cpu") else bool(term)
        trunc_b = bool(to_np(trunc).item()) if hasattr(trunc, "cpu") else bool(trunc)
        succ = info.get("success", False) if hasattr(info, "get") else False
        success = bool(to_np(succ).item()) if hasattr(succ, "cpu") else bool(succ)
        if success or term_b or trunc_b:
            break
        # No-progress detection (phase 2: arm stalls against obstacle).
        if no_progress_break_steps is not None:
            tcp_now = np.asarray(tcp[0:3], dtype=np.float64)
            if prev_tcp_xyz is None:
                prev_tcp_xyz = tcp_now
                stalled_steps = 0
            else:
                if float(np.linalg.norm(tcp_now - prev_tcp_xyz)) < no_progress_eps_m:
                    stalled_steps += 1
                else:
                    stalled_steps = 0
                    prev_tcp_xyz = tcp_now
                if stalled_steps >= no_progress_break_steps:
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
    # Execution phases are observed in the first-person panda_wristcam view.
    obs, _ = env.reset(seed=seed)
    attempt1_frames = [render_wrist_frame(env)] * (fps * 2)

    # === Phase 3 — RETRY with revised approach (selective) ===
    revised_intent, revision = adapter.revise_intent(
        initial_intent, s["attribution"], scene_exec,
    )
    wp_retry = build_push_waypoints(scene_exec, revised_intent)
    retry_frames: list = []
    out_retry = _execute_push(
        env, wp_retry, retry_frames, seed=seed, capture=render_wrist_frame,
    )

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
    # Execution phases are observed in the first-person panda_wristcam view.
    obs, _ = env.reset(seed=seed)
    attempt1_frames = [render_wrist_frame(env)] * (fps * 2)

    # === Phase 3a — SELECTIVE retry (approach_direction only) ===
    sel_intent, _ = adapter.revise_intent(initial_intent, attribution, scene_exec)
    sel_frames: list = []
    out_sel = _execute_push(
        env, build_push_waypoints(scene_exec, sel_intent), sel_frames, seed=seed,
        capture=render_wrist_frame,
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
        capture=render_wrist_frame,
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
