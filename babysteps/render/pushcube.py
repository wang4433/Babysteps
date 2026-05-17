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

from dataclasses import replace

import numpy as np

from babysteps.envs.task_adapter import BaseTaskAdapter
from babysteps.render.common import (
    MAX_CONTROL_STEPS,
    PHASE_TOL_M,
    prop_action,
    read_obs,
    render_frame,
)
from babysteps.schemas import AttemptResult, DemoEvidence, SceneState


def _build_waypoints(scene: SceneState, intent) -> np.ndarray:
    """4-waypoint PushCube trajectory (approach, pre-contact high, pre-contact
    low, push-end). Identical to the inline _build_waypoints in
    scripts/render_stage0_maniskill.py before this extraction — see
    babysteps.skills.push for the canonical version used by the env_runner."""
    from babysteps.envs.scene import approach_to_unit, face_to_push_unit
    cube_xy = np.asarray(scene.cube_xy, dtype=np.float64)
    goal_xy = np.asarray(scene.goal_xy, dtype=np.float64)
    tcp = np.asarray(scene.tcp_start_pose, dtype=np.float64)
    travel_z = float(tcp[2])
    push_z = float(scene.cube_z)
    push_unit = face_to_push_unit(intent.contact_region)
    approach_unit = approach_to_unit(intent.approach_direction)
    standoff = 0.02 + 0.005
    approach_standoff = 0.10
    pre_contact_xy = cube_xy - push_unit * standoff
    approach_xy = cube_xy + approach_unit * approach_standoff
    cube_to_goal = float(np.linalg.norm(goal_xy - cube_xy))
    push_travel = min(0.6 * cube_to_goal, 0.15)
    push_end_xy = cube_xy + push_unit * push_travel

    wp = np.zeros((4, 7), dtype=np.float64)
    wp[0, 0:2] = approach_xy
    wp[0, 2] = travel_z
    wp[1, 0:2] = pre_contact_xy
    wp[1, 2] = travel_z
    wp[2, 0:2] = pre_contact_xy
    wp[2, 2] = push_z
    wp[3, 0:2] = push_end_xy
    wp[3, 2] = push_z
    wp[:, 3:7] = tcp[3:7]
    return wp


def _execute_push(env, waypoints, frames: list, *, seed: int) -> dict:
    """Step through waypoints capturing one frame per step. Re-resets the env
    at the start so demo / attempt / retry all begin from the same scene."""
    obs, _ = env.reset(seed=int(seed))
    targets = [np.asarray(wp[0:3], dtype=np.float64) for wp in waypoints]
    phase_idx = 0
    success = False

    frames.append(render_frame(env))
    for _ in range(MAX_CONTROL_STEPS):
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
        term_b = bool(term) if not hasattr(term, "cpu") \
            else bool(term.cpu().numpy().item())
        trunc_b = bool(trunc) if not hasattr(trunc, "cpu") \
            else bool(trunc.cpu().numpy().item())
        succ = info.get("success", False) if hasattr(info, "get") else False
        success = bool(succ) if not hasattr(succ, "cpu") \
            else bool(succ.cpu().numpy().item())
        if success or term_b or trunc_b:
            break

    tcp, final_cube_xy, _, _ = read_obs(obs)
    return {
        "final_obj_xy": (float(final_cube_xy[0]), float(final_cube_xy[1])),
        "success": bool(success),
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

    # === Phase 1 — DEMO PROXY ===
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
    wp_demo = _build_waypoints(scene, correct_intent)
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

    # === Phase 2 — ATTEMPT 1 (planner_failed, held still) ===
    obs, _ = env.reset(seed=seed)
    attempt1_frames = [render_frame(env)] * (fps * 2)

    # === Phase 3 — RETRY with revised approach ===
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
    revised_intent, _rev = adapter.revise_intent(
        initial_intent, attribution, scene_exec,
    )
    wp_retry = _build_waypoints(scene_exec, revised_intent)
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
        f"{revised_intent.approach_direction}",
    )
    return (
        {"demo": demo_frames,
         "attempt_blocked": attempt1_frames,
         "retry": retry_frames},
        {"demo": demo_title,
         "attempt_blocked": a1_title,
         "retry": retry_title},
    )
