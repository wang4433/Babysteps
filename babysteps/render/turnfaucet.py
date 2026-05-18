"""TurnFaucet-v1 render_episode — three phases for the Stage-0 MP4 set.

Phase 1 (demo): PRIVILEGED qpos teleport — faucet handle rotates from
initial to target angle by direct write to switch_link.joint.qpos. Robot
holds home pose. Caption describes object motion, never a robot motor
program (per the Stage-0 demo-caption guideline).

Phase 2 (attempt_blocked): grasp_turn execution. Franka attempts to grip
the handle and fails (jaws cannot close on the thick partnet handle).

Phase 3 (retry): embodiment_substitution-revised poke_turn. Closed-gripper
lateral brute-force sweep with auto-sign retry as needed.

Generic over len(skill.waypoints) + skill.gripper_schedule. No hardcoded
4-phase grasp assumptions.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from babysteps.envs.task_adapter import BaseTaskAdapter
from babysteps.render.common import (
    PHASE_TOL_M, prop_action, render_frame, to_np,
)
from babysteps.schemas import AttemptResult, DemoEvidence, Intent, SceneState
from babysteps.skills.turn import compile_intent_to_turn_skill

_GRIPPER_CLOSED = -1.0
_PHASE_TOL_M = 0.015
_GRASP_PHASE_TOL_M = 0.025
_GRIP_MIN_STEPS = 15
_MAX_CONTROL_STEPS = 400
_POKE_PROBE_STEPS = 80
_POKE_PROBE_MIN_PROGRESS = 0.4
_DEMO_MIN_FRAMES = 30


def _read_turn_obs(obs):
    tcp_raw = np.asarray(to_np(obs["extra"]["tcp_pose"]), dtype=np.float64)
    tcp = np.concatenate([tcp_raw[0:3], tcp_raw[4:7], tcp_raw[3:4]])
    handle_xyz = np.asarray(to_np(obs["extra"]["target_link_pos"]), dtype=np.float64)
    axis_xyz = np.asarray(to_np(obs["extra"]["target_joint_axis"]), dtype=np.float64)
    return tcp, handle_xyz, axis_xyz


def _safe_bool(x) -> bool:
    if hasattr(x, "cpu"):
        x = x.cpu().numpy()
    arr = np.asarray(x)
    return bool(arr.item() if arr.ndim > 0 else arr)


def _read_faucet_qpos(env_u) -> float:
    return float(to_np(env_u.target_switch_link.joint.qpos).item())


def _set_faucet_qpos(env_u, switch_link, new_qpos: float) -> None:
    """Direct write to the rotating joint's qpos (privileged, no physics).

    ManiSkill's ArticulationJoint.qpos setter (CPU path) requires
    joint.articulation to be non-None, but for a Link.merge()-ed
    switch_link articulation is None.  We bypass it by writing the
    underlying physx articulation qpos array directly.

    For stub/fake joints (in sim-free tests) that expose a plain qpos
    setter we fall back to direct assignment.
    """
    joint = switch_link.joint
    if hasattr(joint, "_physx_articulations") and joint._physx_articulations:
        # Real ManiSkill CPU/GPU joint — write via physx articulation array.
        pa = joint._physx_articulations[0]
        q = pa.qpos.copy()
        idx = int(joint.active_index)
        q[idx] = float(new_qpos)
        pa.qpos = q
    else:
        # Stub joint (sim-free tests): simple setter assignment.
        joint.qpos = np.array([[new_qpos]], dtype=np.float64)


def _execute_skill_for_render(env, skill, *, seed, frames, contact_xy, max_steps):
    """Render-side mirror of TurnFaucetEnvRunner._execute_skill. Same
    generic phase loop but appends render_frame(env) each step instead
    of building a trajectory list. Returns dict with success + progress.
    """
    obs, _ = env.reset(seed=int(seed))
    n_phases = len(skill.waypoints)
    assert len(skill.gripper_schedule) == n_phases
    targets = [np.asarray(wp[0:3], dtype=np.float64) for wp in skill.waypoints]
    grip_phase = 2 if skill.mode == "grasp" and n_phases >= 3 else -1
    phase_tol = tuple(
        _GRASP_PHASE_TOL_M if i == grip_phase else _PHASE_TOL_M
        for i in range(n_phases)
    )

    env_u = env.unwrapped
    target_angle = float(to_np(env_u.target_angle).item())
    initial_qpos = _read_faucet_qpos(env_u)
    needed_delta = target_angle - initial_qpos
    qpos_extremum = initial_qpos

    frames.append(render_frame(env))
    phase_idx, steps_in_phase = 0, 0
    success = False
    for _ in range(max_steps):
        tcp, handle_xyz, _ = _read_turn_obs(obs)
        target = targets[phase_idx]
        reached = np.linalg.norm(target - tcp[0:3]) < phase_tol[phase_idx]
        advance = reached and (
            phase_idx != grip_phase or steps_in_phase >= _GRIP_MIN_STEPS
        )
        if advance:
            phase_idx += 1
            steps_in_phase = 0
            if phase_idx >= n_phases:
                break
            target = targets[phase_idx]
        else:
            steps_in_phase += 1
        action = prop_action(tcp, target, gripper_cmd=skill.gripper_schedule[phase_idx])
        obs, _r, terminated, truncated, info = env.step(action)
        frames.append(render_frame(env))
        qpos = _read_faucet_qpos(env_u)
        if needed_delta > 0:
            qpos_extremum = max(qpos_extremum, qpos)
        else:
            qpos_extremum = min(qpos_extremum, qpos)
        success = _safe_bool(info.get("success", False))
        if success or _safe_bool(terminated) or _safe_bool(truncated):
            break

    progress = (qpos_extremum - initial_qpos) / max(abs(needed_delta), 1e-6)
    return {"success": bool(success), "progress": float(progress)}


def render_episode(env, adapter, seed, fps):
    short_id = f"seed {seed:04d}"
    env_u = env.unwrapped

    # === Phase 1 — DEMO PROXY (privileged qpos teleport) ===
    obs, _ = env.reset(seed=seed)
    switch_link = env_u.target_switch_link
    initial_qpos = _read_faucet_qpos(env_u)
    target_angle = float(to_np(env_u.target_angle).item())
    n_demo = max(int(2 * fps), _DEMO_MIN_FRAMES)
    demo_frames = []
    for i in range(n_demo):
        t = i / max(n_demo - 1, 1)
        new_qpos = initial_qpos + t * (target_angle - initial_qpos)
        _set_faucet_qpos(env_u, switch_link, new_qpos)
        demo_frames.append(render_frame(env))

    # Reset for the real execution phases. Demo's qpos teleport is discarded.
    obs, _ = env.reset(seed=seed)
    tcp, handle_xyz, axis_xyz = _read_turn_obs(obs)
    handle_xy = (float(handle_xyz[0]), float(handle_xyz[1]))
    handle_z = float(handle_xyz[2])
    axis_xy = (float(axis_xyz[0]), float(axis_xyz[1]))
    scene = SceneState(
        cube_xy=handle_xy, cube_z=handle_z, goal_xy=handle_xy,
        tcp_start_pose=tuple(float(v) for v in tcp),  # type: ignore[arg-type]
        blocked_sides=(),
        extra={"handle_xy": handle_xy, "handle_z": handle_z,
                "target_joint_axis_xy": axis_xy},
    )

    demo_evidence = DemoEvidence(
        camera="third_person", demonstrator_type="proxy_oracle",
        object_trajectory=(handle_xy, handle_xy),
        contact_region_label="handle_grip",
        final_state="faucet_turned",
        rgbd_video_path=None,
    )
    initial_intent = adapter.scripted_demo_to_intent(demo_evidence)
    scene_exec = replace(
        scene, blocked_sides=adapter.default_blocked_factory(initial_intent),
    )
    contact_xy = np.asarray(handle_xy, dtype=np.float64)

    # === Phase 2 — ATTEMPT (grasp_turn, fails) ===
    grasp_skill = compile_intent_to_turn_skill(initial_intent, scene_exec)
    attempt_frames = []
    _ = _execute_skill_for_render(
        env, grasp_skill, seed=seed, frames=attempt_frames,
        contact_xy=contact_xy, max_steps=_MAX_CONTROL_STEPS,
    )

    # === Phase 3 — RETRY (poke_turn after embodiment_substitution) ===
    fp = adapter.build_failure_packet(
        initial_intent,
        AttemptResult(
            initial_obj_xy=handle_xy, final_obj_xy=handle_xy,
            goal_xy=handle_xy,
            reached_contact=True, object_moved=False,
            planner_failed=False, collision=False, grasp_slip=False,
            rollout_log_path=None, success=False,
        ),
        scene_exec,
    )
    attribution = adapter.attribute_failure(fp)
    revised_intent, _rev = adapter.revise_intent(initial_intent, attribution, scene_exec)

    # Probe sign=+1; if no progress, retry sign=-1 with frames replaced.
    poke_pos = compile_intent_to_turn_skill(revised_intent, scene_exec, sign=+1)
    retry_frames = []
    out_probe = _execute_skill_for_render(
        env, poke_pos, seed=seed, frames=retry_frames,
        contact_xy=contact_xy, max_steps=_POKE_PROBE_STEPS,
    )
    if (not out_probe["success"]) and out_probe["progress"] < _POKE_PROBE_MIN_PROGRESS:
        retry_frames = []
        poke_neg = compile_intent_to_turn_skill(revised_intent, scene_exec, sign=-1)
        out_retry = _execute_skill_for_render(
            env, poke_neg, seed=seed, frames=retry_frames,
            contact_xy=contact_xy, max_steps=_MAX_CONTROL_STEPS,
        )
    elif (not out_probe["success"]) and out_probe["progress"] >= _POKE_PROBE_MIN_PROGRESS:
        retry_frames = []
        out_retry = _execute_skill_for_render(
            env, poke_pos, seed=seed, frames=retry_frames,
            contact_xy=contact_xy, max_steps=_MAX_CONTROL_STEPS,
        )
    else:
        out_retry = out_probe

    # Tail-pad attempt_frames so all three MP4s have similar duration.
    if attempt_frames:
        attempt_frames = attempt_frames + [attempt_frames[-1]] * fps

    demo_title = (
        f"{short_id}  phase 1/3: third-person object-motion proxy",
        f"faucet turned (handle_grip, faucet_turned)",
    )
    a2_title = (
        f"{short_id}  phase 2/3: grasp_infeasible",
        f"embodiment={initial_intent.embodiment_mapping}: jaws cannot close on handle → no rotation",
    )
    a3_title = (
        f"{short_id}  phase 3/3: retry (success={out_retry['success']})",
        f"embodiment_substitution: {initial_intent.embodiment_mapping} → {revised_intent.embodiment_mapping}",
    )

    return (
        {"demo": demo_frames, "attempt_blocked": attempt_frames, "retry": retry_frames},
        {"demo": demo_title, "attempt_blocked": a2_title, "retry": a3_title},
    )
