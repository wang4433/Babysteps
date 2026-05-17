"""PickCube-v1 render_episode — three phases for the Stage-0 MP4 set.

Phase 1 (demo): execute the oracle's correct intent (top-down grasp with
the demonstrated contact_region). Capture frames.
Phase 2 (attempt_blocked): execute the initial intent in the executor
scene where the demonstrated contact_region is in blocked_sides. The
PickCubeEnvRunner's slip mechanism opens the gripper at lift-time, so
the cube falls back — we render those frames; the grasp_slip is
visually obvious in the MP4.
Phase 3 (retry): execute the revised intent (orthogonal contact_region).

Unlike PushCube, phase 2 here is NOT held-still: the failure happens at
execution time, not compile time, so the viewer needs to see the
attempted-then-failed lift."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from babysteps.envs.task_adapter import BaseTaskAdapter
from babysteps.render.common import (
    PICKCUBE_MAX_CONTROL_STEPS,
    PHASE_TOL_M,
    prop_action,
    read_obs,
    render_frame,
    to_np,
)
from babysteps.schemas import AttemptResult, DemoEvidence, Intent, SceneState
from babysteps.skills.pick import compile_intent_to_pick_skill


_GRIPPER_OPEN = 1.0
_GRIPPER_CLOSED = -1.0


def _execute_pick(
    env, intent: Intent, scene: SceneState, frames: list, *,
    seed: int,
) -> dict:
    """Step the env through PickSkill's 4 waypoints + gripper schedule.

    The slip behavior (gripper open at lift) is keyed off
    `intent.contact_region in scene.blocked_sides`, mirroring
    PickCubeEnvRunner.run."""
    skill = compile_intent_to_pick_skill(intent, scene)
    obs, _ = env.reset(seed=int(seed))
    targets = [np.asarray(wp[0:3], dtype=np.float64) for wp in skill.waypoints]
    slip = intent.contact_region in scene.blocked_sides
    lift_gripper = _GRIPPER_OPEN if slip else _GRIPPER_CLOSED
    phase_gripper = (
        _GRIPPER_OPEN, _GRIPPER_OPEN, _GRIPPER_CLOSED, lift_gripper,
    )

    phase_idx = 0
    success = False
    frames.append(render_frame(env))
    for _ in range(PICKCUBE_MAX_CONTROL_STEPS):
        tcp, _cube_xy, _, _ = read_obs(obs)
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

    return {"success": bool(success and not slip)}


def render_episode(
    env, adapter: BaseTaskAdapter, seed: int, fps: int,
) -> tuple[dict, dict]:
    """Three-phase BABYSTEPS render for PickCube."""
    short_id = f"seed {seed:04d}"

    # === Phase 1 — DEMO PROXY (oracle's correct intent) ===
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
    demo_frames: list = []
    _ = _execute_pick(env, correct_intent, scene, demo_frames, seed=seed)

    # Build the DemoEvidence the loop would build.
    demo_evidence = DemoEvidence(
        camera="third_person",
        demonstrator_type="proxy_oracle",
        object_trajectory=(
            (float(cube_xy0[0]), float(cube_xy0[1])),
            (float(scene.goal_xy[0]), float(scene.goal_xy[1])),
        ),
        contact_region_label=correct_intent.contact_region,
        final_state=correct_intent.goal_state,
        rgbd_video_path=None,
    )
    initial_intent = adapter.scripted_demo_to_intent(demo_evidence)
    scene_exec = replace(
        scene, blocked_sides=adapter.default_blocked_factory(initial_intent),
    )

    # === Phase 2 — ATTEMPT 1 (grasp_slip, actually executed) ===
    attempt1_frames: list = []
    _ = _execute_pick(env, initial_intent, scene_exec, attempt1_frames, seed=seed)

    # === Phase 3 — RETRY with revised contact_region ===
    # Synthetic AttemptResult: grasp_slip happened during phase 2, but for
    # the pipeline (build_failure_packet → attribute_failure → revise_intent)
    # only the flags matter, not the cube positions. Setting grasp_slip=True
    # routes to the "grasp_slip" predicate → wrong_factor "contact_region".
    fp = adapter.build_failure_packet(
        initial_intent,
        AttemptResult(
            initial_obj_xy=scene.cube_xy, final_obj_xy=scene.cube_xy,
            goal_xy=scene.goal_xy,
            reached_contact=True, object_moved=False,
            planner_failed=False, collision=False, grasp_slip=True,
            rollout_log_path=None, success=False,
        ),
        scene_exec,
    )
    attribution = adapter.attribute_failure(fp)
    revised_intent, _rev = adapter.revise_intent(
        initial_intent, attribution, scene_exec,
    )
    retry_frames: list = []
    out_retry = _execute_pick(
        env, revised_intent, scene_exec, retry_frames, seed=seed,
    )

    demo_title = (
        f"{short_id}  phase 1/3: demo proxy",
        f"contact_region={correct_intent.contact_region}, "
        f"approach={correct_intent.approach_direction}",
    )
    a1_title = (
        f"{short_id}  phase 2/3: grasp_slip",
        f"contact_region={initial_intent.contact_region} "
        f"is slip-prone → lift opens, cube drops",
    )
    retry_title = (
        f"{short_id}  phase 3/3: retry (success={out_retry['success']})",
        f"contact_substitution: "
        f"{initial_intent.contact_region} → {revised_intent.contact_region}",
    )

    # Pad attempt1 with a tail of the last frame so the slip is on-screen for
    # at least fps*1 frames (otherwise the lift can finish too fast to see).
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
