"""StackCube-v1 render_episode — three phases for the Stage-0 MP4 set.

Phase 1 (demo): execute the oracle's correct intent (cubeA_on_cubeB).
Successful stack.
Phase 2 (attempt_blocked): execute the scripted-summarizer's under-
specified intent (cube_at_target). Cube drops at cubeB.xy at low z,
collides with cubeB, scatters. The viewer sees the failure happen.
Phase 3 (retry): execute the goal_refinement-revised intent
(cubeA_on_cubeB). Successful stack.

Like PickCube (and unlike PushCube), all three phases actually step
the env — Stage-0's StackCube failure happens at execution time, not
compile time. The phase key 'attempt_blocked' is kept for consistency
with the other render modules; the term is historical (from B/A's
blocked-approach narrative). For StackCube the failure is under-
specification, not blocking.
"""
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
from babysteps.skills.stack import (
    CUBE_HALF_SIZE,
    compile_intent_to_stack_skill,
)


_GRIPPER_OPEN = 1.0
_GRIPPER_CLOSED = -1.0


def _read_stack_obs(obs):
    """(tcp_xyzw, cubeA_xy, cubeA_z, cubeB_xy, cubeB_z) from a StackCube obs.

    Local to this render module — the StackCube obs has cubeA_pose and
    cubeB_pose, not the cube/goal pair that babysteps.render.common.read_obs
    parses."""
    tcp_raw = to_np(obs["extra"]["tcp_pose"])
    tcp_raw = np.asarray(tcp_raw, dtype=np.float64)
    tcp = np.concatenate([tcp_raw[0:3], tcp_raw[4:7], tcp_raw[3:4]])
    cubeA_full = np.asarray(to_np(obs["extra"]["cubeA_pose"]), dtype=np.float64)
    cubeA_xy = cubeA_full[0:2]
    cubeA_z = float(cubeA_full[2])
    cubeB_full = np.asarray(to_np(obs["extra"]["cubeB_pose"]), dtype=np.float64)
    cubeB_xy = cubeB_full[0:2]
    cubeB_z = float(cubeB_full[2])
    return tcp, cubeA_xy, cubeA_z, cubeB_xy, cubeB_z


# Stage-5 dual-camera / goal_state — post-place gripper retract. After the
# stack/place completes, the gripper is co-located with the just-placed cubes
# and occludes the stack-vs-near relation from EVERY exterior viewpoint (the
# camera sweep, job 10969709, falsified the high-oblique fix). Lifting the arm
# up-and-back clears it so the final frames show the clean placement — matching
# the 0.99 armless config ceiling. Default OFF keeps the Stage-0 MP4 render
# byte-identical.
_RETRACT_DXYZ = (-0.08, 0.0, 0.30)   # world-frame TCP delta: up 30cm, back 8cm
_RETRACT_STEPS = 45                  # P-control steps toward the cleared pose
_RETRACT_DWELL = 12                  # hold frames so the last frames are clean/stable


def _execute_stack(
    env, intent: Intent, scene: SceneState, frames: list, *, seed: int,
    retract: bool = False,
) -> dict:
    """Step the env through StackSkill's waypoints + per-phase gripper
    schedule. Mirrors PickCube's _execute_pick but with cubeA/cubeB obs
    and goal_state-dispatched waypoint count.

    ``retract`` (Stage-5 goal_state experiment): after the place completes,
    lift the open gripper up-and-back and dwell, so the appended final frames
    show the placed cubes WITHOUT the occluding gripper. The env must have
    enough TimeLimit headroom (see the probe's --retract path, which bumps
    max_episode_steps). Default False -> byte-identical to the committed render.
    """
    skill = compile_intent_to_stack_skill(intent, scene)
    obs, _ = env.reset(seed=int(seed))
    targets = [np.asarray(wp[0:3], dtype=np.float64) for wp in skill.waypoints]

    n_phases = len(targets)
    if n_phases == 4:
        phase_gripper = (
            _GRIPPER_OPEN, _GRIPPER_OPEN, _GRIPPER_CLOSED, _GRIPPER_OPEN,
        )
    elif n_phases == 5:
        phase_gripper = (
            _GRIPPER_OPEN, _GRIPPER_OPEN, _GRIPPER_CLOSED,
            _GRIPPER_CLOSED, _GRIPPER_OPEN,
        )
    else:
        raise RuntimeError(
            f"_execute_stack: unexpected waypoint count {n_phases}; "
            "expected 4 or 5"
        )

    phase_idx = 0
    success = False
    frames.append(render_frame(env))
    for _ in range(STACKCUBE_MAX_CONTROL_STEPS):
        tcp, _cubeA_xy, _cubeA_z, _cubeB_xy, _cubeB_z = _read_stack_obs(obs)
        target = targets[phase_idx]
        if np.linalg.norm(target - tcp[0:3]) < PHASE_TOL_M:
            phase_idx += 1
            if phase_idx >= n_phases:
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

    if retract:
        # Lift the open gripper clear of the placed cubes and hold, so the
        # appended final frames show the clean stack-vs-near relation. Stepping
        # continues the sim (placed cubes stay put); the caller provides the
        # TimeLimit headroom.
        tcp, *_ = _read_stack_obs(obs)
        clear_target = tcp[0:3] + np.asarray(_RETRACT_DXYZ, dtype=np.float64)
        for _ in range(_RETRACT_STEPS):
            tcp, *_ = _read_stack_obs(obs)
            if np.linalg.norm(clear_target - tcp[0:3]) < PHASE_TOL_M:
                break
            action = prop_action(tcp, clear_target, gripper_cmd=_GRIPPER_OPEN)
            obs, _r, _term, _trunc, _info = env.step(action)
            frames.append(render_frame(env))
        for _ in range(_RETRACT_DWELL):  # hold the cleared pose: stable clean tail
            tcp, *_ = _read_stack_obs(obs)
            action = prop_action(tcp, clear_target, gripper_cmd=_GRIPPER_OPEN)
            obs, _r, _term, _trunc, _info = env.step(action)
            frames.append(render_frame(env))

    return {"success": bool(success)}


def render_episode(
    env, adapter: BaseTaskAdapter, seed: int, fps: int,
) -> tuple[dict, dict]:
    """Three-phase BABYSTEPS render for StackCube."""
    short_id = f"seed {seed:04d}"

    # === Phase 1 — DEMO PROXY (oracle's correct intent: cubeA_on_cubeB) ===
    obs, _ = env.reset(seed=seed)
    tcp, cubeA_xy0, cubeA_z, cubeB_xy, cubeB_z = _read_stack_obs(obs)
    cubeB_top_z = cubeB_z + 2 * CUBE_HALF_SIZE
    scene = SceneState(
        cube_xy=(float(cubeA_xy0[0]), float(cubeA_xy0[1])),
        cube_z=cubeA_z,
        goal_xy=(float(cubeB_xy[0]), float(cubeB_xy[1])),
        tcp_start_pose=tuple(float(v) for v in tcp),  # type: ignore[arg-type]
        blocked_sides=(),
        extra={
            "cubeB_xy": (float(cubeB_xy[0]), float(cubeB_xy[1])),
            "cubeB_z": cubeB_z,
            "cubeB_top_z": cubeB_top_z,
        },
    )
    correct_intent = adapter.oracle_correct_intent(scene)
    demo_frames: list = []
    _ = _execute_stack(env, correct_intent, scene, demo_frames, seed=seed)

    # Build the DemoEvidence the loop would build (2D trajectory hides
    # vertical motion — this is the Stage-0 controlled information loss).
    demo_evidence = DemoEvidence(
        camera="third_person",
        demonstrator_type="proxy_oracle",
        object_trajectory=(
            (float(cubeA_xy0[0]), float(cubeA_xy0[1])),
            (float(cubeB_xy[0]), float(cubeB_xy[1])),
        ),
        contact_region_label=correct_intent.contact_region,
        final_state=correct_intent.goal_state,
        rgbd_video_path=None,
    )
    initial_intent = adapter.scripted_demo_to_intent(demo_evidence)
    scene_exec = replace(
        scene, blocked_sides=adapter.default_blocked_factory(initial_intent),
    )

    # === Phase 2 — ATTEMPT 1 (cube_at_target; collides with cubeB) ===
    attempt1_frames: list = []
    _ = _execute_stack(env, initial_intent, scene_exec, attempt1_frames, seed=seed)

    # === Phase 3 — RETRY with goal_refinement-revised intent ===
    # Synthetic AttemptResult: goal_not_satisfied means the cube didn't
    # reach its (sharpened) target. The fp/attribution pipeline only
    # uses the predicate flags to derive wrong_factor.
    fp = adapter.build_failure_packet(
        initial_intent,
        AttemptResult(
            initial_obj_xy=scene.cube_xy, final_obj_xy=scene.goal_xy,
            goal_xy=scene.goal_xy,
            reached_contact=True, object_moved=True,
            planner_failed=False, collision=False, grasp_slip=False,
            rollout_log_path=None, success=False,
        ),
        scene_exec,
    )
    attribution = adapter.attribute_failure(fp)
    revised_intent, _rev = adapter.revise_intent(
        initial_intent, attribution, scene_exec,
    )
    retry_frames: list = []
    out_retry = _execute_stack(
        env, revised_intent, scene_exec, retry_frames, seed=seed,
    )

    demo_title = (
        f"{short_id}  phase 1/3: demo proxy",
        f"goal_state={correct_intent.goal_state}, "
        f"object_motion={correct_intent.object_motion}",
    )
    a1_title = (
        f"{short_id}  phase 2/3: goal_under-specified",
        f"goal_state={initial_intent.goal_state} → cube drops at cubeB.xy "
        f"(collides, scatters)",
    )
    retry_title = (
        f"{short_id}  phase 3/3: retry (success={out_retry['success']})",
        f"goal_refinement: "
        f"{initial_intent.goal_state} → {revised_intent.goal_state}",
    )

    # Tail-pad attempt1 so the scattered cube is on-screen for at least
    # fps frames (mirrors PickCube's slip-visibility padding).
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
