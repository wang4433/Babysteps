"""CrossViewPush render_episode — three phases for the Stage-0 MP4 set.

Phase 1 (demo): the world-correct oracle push (the demonstration).
Phase 2 (attempt): the actor_frame (egocentric) push — cube moves the WRONG
way (this is the cross-view failure, NOT a held-still planner failure).
Phase 3 (retry): grounding_substitution → observer_frame → correct push.

Geometry reuses build_push_waypoints on world-resolved intents."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from babysteps.envs.crossview_adapter import observer_yaw_for_seed
from babysteps.envs.scene import world_resolved_intent
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


def render_episode(env, adapter, seed: int, fps: int) -> tuple[dict, dict]:
    short_id = f"seed {seed:04d}"
    yaw = observer_yaw_for_seed(seed)

    obs, _ = env.reset(seed=seed)
    tcp_xyzw, cube_xy0, goal_xy, cube_z = read_obs(obs)
    scene = SceneState(
        cube_xy=(float(cube_xy0[0]), float(cube_xy0[1])),
        cube_z=cube_z,
        goal_xy=(float(goal_xy[0]), float(goal_xy[1])),
        tcp_start_pose=tuple(float(v) for v in tcp_xyzw),  # type: ignore[arg-type]
        blocked_sides=(),
        extra={"observer_yaw_deg": yaw},
    )

    # === Phase 1 — DEMO (world-correct oracle push) ===
    correct = adapter.oracle_correct_intent(scene)
    wp_demo = build_push_waypoints(scene, world_resolved_intent(correct, yaw))
    demo_frames: list = []
    out_demo = _execute_push(env, wp_demo, demo_frames, seed=seed)

    observed_traj, contact_label = adapter.observe_demo(
        (scene.cube_xy, out_demo["final_obj_xy"]), correct, scene,
    )
    evidence = DemoEvidence(
        camera="observer_view", demonstrator_type="proxy_oracle",
        object_trajectory=observed_traj, contact_region_label=contact_label,
        final_state=correct.goal_state, rgbd_video_path=None,
    )
    initial = adapter.scripted_demo_to_intent(evidence)
    scene_exec = replace(scene, blocked_sides=adapter.default_blocked_factory(initial))

    # === Phase 2 — ATTEMPT 1 (actor_frame, wrong push) ===
    wp_a = build_push_waypoints(scene_exec, world_resolved_intent(initial, yaw))
    attempt_frames: list = []
    out_a = _execute_push(env, wp_a, attempt_frames, seed=seed)

    # === Phase 3 — RETRY (observer_frame) ===
    fp = adapter.build_failure_packet(
        initial,
        AttemptResult(
            initial_obj_xy=scene.cube_xy, final_obj_xy=out_a["final_obj_xy"],
            goal_xy=scene.goal_xy, reached_contact=True, object_moved=True,
            planner_failed=False, collision=False, grasp_slip=False,
            rollout_log_path=None, success=out_a["success"],
        ),
        scene_exec,
    )
    attribution = adapter.attribute_failure(fp)
    revised, _rev = adapter.revise_intent(initial, attribution, scene_exec)
    wp_r = build_push_waypoints(scene_exec, world_resolved_intent(revised, yaw))
    retry_frames: list = []
    out_r = _execute_push(env, wp_r, retry_frames, seed=seed)

    # NOTE: all phases render from PushCube's default (world) camera. The
    # observer yaw is applied to the grounding math (observe_demo / resolution),
    # not to a physically rotated SAPIEN camera — so the banner says "world
    # camera" with the observer yaw as metadata, not "observer view".
    demo_title = (
        f"{short_id}  phase 1/3: demo (world camera, observer yaw={yaw}deg)",
        f"object moved to target; observed contact_region={contact_label}",
    )
    a1_title = (
        f"{short_id}  phase 2/3: attempt (direction_grounding=actor_frame)",
        f"egocentric grounding pushes wrong way (success={out_a['success']})",
    )
    retry_title = (
        f"{short_id}  phase 3/3: retry (success={out_r['success']})",
        "grounding_substitution: actor_frame -> observer_frame",
    )
    return (
        {"demo": demo_frames, "attempt_blocked": attempt_frames, "retry": retry_frames},
        {"demo": demo_title, "attempt_blocked": a1_title, "retry": retry_title},
    )
