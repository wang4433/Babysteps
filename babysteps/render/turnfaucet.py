"""TurnFaucet-v1 render_episode — three phases for the Stage-0 MP4 set.

Phase 1 (demo): oracle intent (handle_grip + faucet_base_static).
Faucet rotates.
Phase 2 (attempt_blocked): scripted intent (faucet_base + none).
Gripper touches the static body, no rotation. Tail-padded.
Phase 3 (retry): constraint_introduction-revised intent (handle_grip
+ faucet_base_static). Faucet rotates.

Like PickCube and StackCube (unlike PushCube), all three phases step
the env."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from babysteps.envs.task_adapter import BaseTaskAdapter
from babysteps.render.common import (
    PHASE_TOL_M,
    STACKCUBE_MAX_CONTROL_STEPS,
    prop_action,
    render_frame,
    to_np,
)
from babysteps.schemas import AttemptResult, DemoEvidence, Intent, SceneState
from babysteps.skills.turn import compile_intent_to_turn_skill


_GRIPPER_OPEN = 1.0
_GRIPPER_CLOSED = -1.0


def _read_turn_obs(obs):
    """(tcp_xyzw, handle_xyz, axis_xyz) from TurnFaucet obs."""
    tcp_raw = np.asarray(to_np(obs["extra"]["tcp_pose"]), dtype=np.float64)
    tcp = np.concatenate([tcp_raw[0:3], tcp_raw[4:7], tcp_raw[3:4]])
    handle_xyz = np.asarray(to_np(obs["extra"]["target_link_pos"]), dtype=np.float64)
    axis_xyz = np.asarray(to_np(obs["extra"]["target_joint_axis"]), dtype=np.float64)
    return tcp, handle_xyz, axis_xyz


def _execute_turn(env, intent, scene, frames, *, seed):
    skill = compile_intent_to_turn_skill(intent, scene)
    obs, _ = env.reset(seed=int(seed))
    targets = [np.asarray(wp[0:3], dtype=np.float64) for wp in skill.waypoints]
    phase_gripper = (_GRIPPER_OPEN, _GRIPPER_OPEN, _GRIPPER_CLOSED, _GRIPPER_CLOSED)

    phase_idx = 0
    success = False
    frames.append(render_frame(env))
    for _ in range(STACKCUBE_MAX_CONTROL_STEPS):
        tcp, _h, _a = _read_turn_obs(obs)
        target = targets[phase_idx]
        if np.linalg.norm(target - tcp[0:3]) < PHASE_TOL_M:
            phase_idx += 1
            if phase_idx >= len(targets):
                break
            target = targets[phase_idx]
        action = prop_action(tcp, target, gripper_cmd=phase_gripper[phase_idx])
        obs, _r, term, trunc, info = env.step(action)
        frames.append(render_frame(env))
        term_b = bool(to_np(term).item()) if hasattr(term, "cpu") else bool(term)
        trunc_b = bool(to_np(trunc).item()) if hasattr(trunc, "cpu") else bool(trunc)
        succ = info.get("success", False) if hasattr(info, "get") else False
        success = bool(to_np(succ).item()) if hasattr(succ, "cpu") else bool(succ)
        if success or term_b or trunc_b:
            break
    return {"success": bool(success)}


def render_episode(env, adapter, seed, fps):
    short_id = f"seed {seed:04d}"

    # === Phase 1 — DEMO (oracle's handle_grip + faucet_base_static) ===
    obs, _ = env.reset(seed=seed)
    tcp, handle_xyz, axis_xyz = _read_turn_obs(obs)
    handle_xy = (float(handle_xyz[0]), float(handle_xyz[1]))
    handle_z = float(handle_xyz[2])
    base_xy = (handle_xy[0] - 0.05, handle_xy[1])
    axis_xy = (float(axis_xyz[0]), float(axis_xyz[1]))
    scene = SceneState(
        cube_xy=handle_xy, cube_z=handle_z, goal_xy=handle_xy,
        tcp_start_pose=tuple(float(v) for v in tcp),  # type: ignore[arg-type]
        blocked_sides=(),
        extra={
            "handle_xy": handle_xy, "handle_z": handle_z,
            "faucet_base_xy": base_xy, "faucet_base_z": 0.0,
            "target_joint_axis_xy": axis_xy,
        },
    )
    correct_intent = adapter.oracle_correct_intent(scene)
    demo_frames: list = []
    _ = _execute_turn(env, correct_intent, scene, demo_frames, seed=seed)

    demo_evidence = DemoEvidence(
        camera="third_person",
        demonstrator_type="proxy_oracle",
        object_trajectory=(handle_xy, handle_xy),
        contact_region_label="handle_grip",
        final_state="faucet_turned",
        rgbd_video_path=None,
    )
    initial_intent = adapter.scripted_demo_to_intent(demo_evidence)
    scene_exec = replace(
        scene, blocked_sides=adapter.default_blocked_factory(initial_intent),
    )

    # === Phase 2 — ATTEMPT 1 (faucet_base + none; collision, no rotation) ===
    attempt1_frames: list = []
    _ = _execute_turn(env, initial_intent, scene_exec, attempt1_frames, seed=seed)

    # === Phase 3 — RETRY (constraint_introduction-revised) ===
    fp = adapter.build_failure_packet(
        initial_intent,
        AttemptResult(
            initial_obj_xy=scene.cube_xy, final_obj_xy=scene.cube_xy,
            goal_xy=scene.goal_xy,
            reached_contact=True, object_moved=False,
            planner_failed=False, collision=True, grasp_slip=False,
            rollout_log_path=None, success=False,
        ),
        scene_exec,
    )
    attribution = adapter.attribute_failure(fp)
    revised_intent, _rev = adapter.revise_intent(initial_intent, attribution, scene_exec)
    retry_frames: list = []
    out_retry = _execute_turn(env, revised_intent, scene_exec, retry_frames, seed=seed)

    demo_title = (
        f"{short_id}  phase 1/3: demo proxy",
        f"contact_region={correct_intent.contact_region}, "
        f"constraint_region={correct_intent.constraint_region}",
    )
    a1_title = (
        f"{short_id}  phase 2/3: constraint_violation",
        f"contact_region={initial_intent.contact_region} (faucet body) → no rotation",
    )
    retry_title = (
        f"{short_id}  phase 3/3: retry (success={out_retry['success']})",
        f"constraint_introduction: "
        f"({initial_intent.constraint_region}, {initial_intent.contact_region}) → "
        f"({revised_intent.constraint_region}, {revised_intent.contact_region})",
    )

    if attempt1_frames:
        tail = [attempt1_frames[-1]] * fps
        attempt1_frames = attempt1_frames + tail

    return (
        {"demo": demo_frames,
         "attempt_blocked": attempt1_frames,
         "retry": retry_frames},
        {"demo": demo_title,
         "attempt_blocked": a1_title,
         "retry": retry_title},
    )
