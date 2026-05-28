"""PushCube-v1 render_episode — three phases for the Stage-0 MP4 set.

Phase 1 (demo): execute the oracle's correct intent in a fresh seed, capture
all frames.
Phase 2 (attempt_blocked): a small grey-brown clutter object is placed on the
demo's approach side (scene-clutter mismatch — see redesign_failure_paradigm.md
§"Phase 1"); the demo-derived push waypoints are driven, and the arm
visibly stalls against the clutter. The clip ends when TCP has been still
for N steps or at the max-steps budget.
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


# Opposite contact face for PushCube's natural-failure paper-figure render.
# Used by `render_natural_failure_episode` to flip exactly one factor
# (contact_region) of the oracle correct intent: the skill compiler routes
# push direction through `face_to_push_unit(intent.contact_region)`, so this
# flip produces a wrong-way push with no obstacle. The PickCube-style
# orthogonal flip in scene.ORTHOGONAL_FACE is a 90° rotation; here we want
# the geometric opposite so the cube visibly moves AWAY from the goal.
_OPPOSITE_FACE: dict[str, str] = {
    "minus_x_face": "plus_x_face",
    "plus_x_face":  "minus_x_face",
    "minus_y_face": "plus_y_face",
    "plus_y_face":  "minus_y_face",
}


# Obstacle (phase-2 blocked-side clutter object) — half-extents in meters.
# Sized as a small grey-brown clutter object (5 cm × 5 cm × 8 cm), looks like
# a mug or small container sitting on the table rather than a red barrier. The
# scene-clutter mismatch is the demo→execution drift category for PushCube;
# see redesign_failure_paradigm.md §"Phase 1". Earlier sizes: 4 cm × 0.5 cm ×
# 10 cm (invisible at overview), then 4 cm × 15 cm × 15 cm (a red wall — too
# obviously synthetic).
_OBSTACLE_HALF_W: float = 0.025   # along the approach axis (0.05 m total)
_OBSTACLE_HALF_T: float = 0.025   # perpendicular to approach (0.05 m total)
_OBSTACLE_HALF_H: float = 0.04    # vertical (0.08 m total) — sits on the table
_OBSTACLE_PARK_Z: float = -0.50   # below table plane; invisible / out of the way
_OBSTACLE_BLOCK_MARGIN_M: float = 0.025  # gap between cube edge and clutter face


def _get_or_build_obstacle(env):
    """Spawn (once per env) a static grey-brown clutter box, parked below the
    table. Returns None when the env does not support actor building
    (sim-free stub envs).

    Cached on `env._babysteps_obstacle` so repeated render_episode calls
    on the same env reuse the same actor rather than accumulating clutter.
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
        # Neutral grey-brown — reads as a small container / mug on the table,
        # not a synthetic red barrier.
        material=sapien.render.RenderMaterial(base_color=[0.55, 0.45, 0.35, 1.0]),
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
    """Place the clutter object on the blocked side of the cube, on the EE's
    approach path, sitting on the table surface. No-op when obstacle is None."""
    if obstacle is None:
        return
    import sapien
    from babysteps.envs.scene import approach_to_unit
    from babysteps.skills.push import CUBE_HALF_SIZE
    unit = approach_to_unit(intent.approach_direction)
    margin = CUBE_HALF_SIZE + _OBSTACLE_BLOCK_MARGIN_M
    x = float(cube_xy[0]) + float(unit[0]) * margin
    y = float(cube_xy[1]) + float(unit[1]) * margin
    # Sit on the table: table top is at cube_z - CUBE_HALF_SIZE; place the
    # clutter so its base sits there and its center is half_h above.
    z = float(cube_z) - CUBE_HALF_SIZE + _OBSTACLE_HALF_H
    # The clutter is symmetric in xy (5 cm × 5 cm cross-section) so no
    # orientation correction is needed for y-approach seeds. Kept as
    # identity for clarity; the prior 90°-around-z rotation was needed only
    # for the highly asymmetric wall shape.
    q = [1.0, 0.0, 0.0, 0.0]
    obstacle.set_pose(sapien.Pose(p=[x, y, z], q=q))


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

    Phase 1 (demo): execute the oracle's correct intent.
    Phase 2 (attempt_blocked): place a small grey-brown clutter object on the
        demo's approach side; execute the demo-derived push waypoints; the arm
        physically stalls against the clutter and the cube is unmoved.
    Phase 3 (retry): execute the revised (orthogonal-approach) intent.

    Returns:
        frames: {"demo": [...], "attempt_blocked": [...], "retry": [...]}
        titles: {"demo": (title, subtitle), ...}
    """
    short_id = f"seed {seed:04d}"

    # Spawn the obstacle once per env (cached). Parked below the table by
    # default so phases 1 and 3 are unaffected.
    obstacle = _get_or_build_obstacle(env)
    _park_obstacle(obstacle)

    s = _pushcube_setup(env, adapter, seed)
    correct_intent = s["correct_intent"]
    initial_intent = s["initial_intent"]
    scene_exec = s["scene_exec"]
    demo_frames = s["demo_frames"]

    # === Phase 2 — ATTEMPT (approach physically obstructed) ===
    # Move the clutter object onto the demo's approach side, then drive the
    # demo-derived waypoints. The arm reaches the approach standoff,
    # hits the clutter, and the no-progress break ends the clip.
    #
    # Deliberate divergence from the collection path: render needs a
    # visible failure (MP4 for reviewers), so we spawn a physical clutter
    # object and step the env. The collection path in
    # babysteps/envs/pushcube_runner.py instead returns
    # planner_failed=True without stepping — fast for 1k-episode runs and
    # consistent with the research claim, which operates at the
    # intent/attribution/revision layer, not the control layer.
    # Do not unify.
    _move_obstacle_to_block(
        obstacle, s["scene"].cube_xy, s["scene"].cube_z, initial_intent,
    )
    wp_attempt = build_push_waypoints(scene_exec, initial_intent)
    attempt1_frames: list = []
    _ = _execute_push(
        env, wp_attempt, attempt1_frames, seed=seed,
        capture=render_wrist_frame,
        max_steps=120,
        no_progress_break_steps=20,
        no_progress_eps_m=0.002,
    )

    # === Phase 3 — RETRY with revised approach (selective) ===
    # Clutter stays on the table so the retry scene matches the attempt
    # scene — the robot approaches from a different direction, not a
    # different scene.
    revised_intent, revision = adapter.revise_intent(
        initial_intent, s["attribution"], scene_exec,
    )
    wp_retry = build_push_waypoints(scene_exec, revised_intent)
    retry_frames: list = []
    out_retry = _execute_push(
        env, wp_retry, retry_frames, seed=seed, capture=render_wrist_frame,
    )
    _park_obstacle(obstacle)

    demo_title = (
        f"{short_id}  phase 1/3: demo proxy",
        f"contact_region={correct_intent.contact_region}, "
        f"approach={correct_intent.approach_direction}",
    )
    a1_title = (
        f"{short_id}  phase 2/3: approach_blocked",
        f"approach_direction={initial_intent.approach_direction} "
        f"physically obstructed → arm stalls",
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

    # Spawn the obstacle once per env (cached). Parked below the table by
    # default so the demo phase is unaffected even if the previous call
    # left the obstacle on the table.
    obstacle = _get_or_build_obstacle(env)
    _park_obstacle(obstacle)

    s = _pushcube_setup(env, adapter, seed)
    correct_intent = s["correct_intent"]
    initial_intent = s["initial_intent"]
    scene_exec = s["scene_exec"]
    attribution = s["attribution"]
    demo_frames = s["demo_frames"]

    # === Phase 2 — ATTEMPT 1 (approach physically obstructed) ===
    _move_obstacle_to_block(
        obstacle, s["scene"].cube_xy, s["scene"].cube_z, initial_intent,
    )
    wp_attempt = build_push_waypoints(scene_exec, initial_intent)
    attempt1_frames: list = []
    _ = _execute_push(
        env, wp_attempt, attempt1_frames, seed=seed,
        capture=render_wrist_frame,
        max_steps=120,
        no_progress_break_steps=20,
        no_progress_eps_m=0.002,
    )

    # === Phase 3a — SELECTIVE retry (approach_direction only) ===
    # Clutter stays on the table for both retries — same scene, different
    # approach strategy.
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
    _park_obstacle(obstacle)

    demo_title = (
        f"{short_id}  phase 1/4: demo proxy",
        f"contact_region={correct_intent.contact_region}, "
        f"approach={correct_intent.approach_direction}",
    )
    a1_title = (
        f"{short_id}  phase 2/4: approach_blocked",
        f"approach_direction={initial_intent.approach_direction} "
        f"physically obstructed → arm stalls",
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


def render_policy_episode(
    env,
    adapter: BaseTaskAdapter,
    seed: int,
    *,
    policy_name: str,
    policy_callable,
    demo_features_provider=None,
    fps: int = 20,
) -> tuple[list, tuple[str, str]]:
    """Render one continuous PushCube episode under a single retry policy.

    Phases run in order and frames are concatenated into one list:
      1. demo (oracle correct intent, no obstacle)
      2. attempt_blocked (initial intent + clutter in place → arm stalls)
      3. retry (policy_callable's revised intent vs the same clutter)

    Returns (frames, (title, subtitle)) where the title encodes seed +
    policy_name and the subtitle records the revision (or "no_revision"
    for one_shot-style policies). Designed for the Stage-5 P1 iconic
    contrast renders, where the same (seed, demo, attempt) prefix is
    re-rendered per policy so each MP4 is a self-contained "full episode"
    clip.

    Parameters
    ----------
    policy_name : str
        Short tag burned into the title (e.g. "latent",
        "oracle_factor_revision", "babysteps_selective",
        "same_intent_retry"). Identity only — does not switch behaviour.
    policy_callable : RetryPolicy
        A `(RetryContext) -> Optional[(Intent, Revision)]` function. May
        be the LatentPack closure from
        `babysteps.stage4.latent_policy.latent_revision_factory`.
    demo_features_provider : Optional[Callable[[int], Any]]
        If provided, `demo_features_provider(seed)` is called and the
        result is attached to `RetryContext.demo_features`. Required for
        the latent policy; pass None for all others.

    Notes
    -----
    `fps` is accepted for signature parity with `render_episode` and
    `render_baseline_contrast`; frame capture cadence is governed by
    `_execute_push`, not by this argument.
    """
    short_id = f"seed {seed:04d}"

    # Spawn / park obstacle (no-op if already cached on env).
    obstacle = _get_or_build_obstacle(env)
    _park_obstacle(obstacle)

    s = _pushcube_setup(env, adapter, seed)
    correct_intent = s["correct_intent"]
    initial_intent = s["initial_intent"]
    scene_exec = s["scene_exec"]
    attribution = s["attribution"]
    demo_frames = s["demo_frames"]

    # === Phase 2 — initial intent vs the clutter ===
    _move_obstacle_to_block(
        obstacle, s["scene"].cube_xy, s["scene"].cube_z, initial_intent,
    )
    wp_attempt = build_push_waypoints(scene_exec, initial_intent)
    attempt_frames: list = []
    _ = _execute_push(
        env, wp_attempt, attempt_frames, seed=seed,
        capture=render_wrist_frame,
        max_steps=120,
        no_progress_break_steps=20,
        no_progress_eps_m=0.002,
    )
    # Note: leave the clutter in place for the retry — every policy must
    # face the same physical obstacle.

    # === Phase 3 — policy retry ===
    fp = adapter.build_failure_packet(
        initial_intent,
        AttemptResult(
            initial_obj_xy=s["scene"].cube_xy,
            final_obj_xy=s["scene"].cube_xy,
            goal_xy=s["scene"].goal_xy,
            reached_contact=False, object_moved=False,
            planner_failed=True, collision=False, grasp_slip=False,
            rollout_log_path=None, success=False,
        ),
        scene_exec,
    )
    demo_features = (
        demo_features_provider(seed) if demo_features_provider is not None else None
    )
    ctx = RetryContext(
        initial_intent=initial_intent,
        attribution=attribution,
        scene=scene_exec,
        oracle_correct_intent=adapter.oracle_correct_intent(scene_exec),
        oracle_wrong_factor=adapter.oracle_wrong_factor(initial_intent, scene_exec),
        task_valid_tokens=adapter.task_valid_tokens(),
        rng=random.Random(seed),
        revise_fn=adapter.revise_intent,
        demo_features=demo_features,
        failure_predicate=fp.failure_predicate,
        failure_packet=fp,
    )
    out = policy_callable(ctx)
    retry_frames: list = []
    if out is None:
        # one_shot-style: no retry. Capture a single still frame so the
        # concatenated clip ends cleanly rather than truncating mid-stream.
        retry_frames.append(render_wrist_frame(env))
        retry_intent = initial_intent
        revision_subtitle = "no_revision (one_shot)"
        retry_success = False
    else:
        retry_intent, revision = out
        out_exec = _execute_push(
            env,
            build_push_waypoints(scene_exec, retry_intent),
            retry_frames,
            seed=seed,
            capture=render_wrist_frame,
        )
        retry_success = bool(out_exec["success"])
        if revision.factor == "none":
            revision_subtitle = "no_revision (same_intent_retry)"
        else:
            revision_subtitle = (
                f"{revision.factor}: {revision.old_value} → {revision.new_value}"
            )

    title = (
        f"{short_id}  policy: {policy_name}  (success={retry_success})",
        f"demo({correct_intent.contact_region}/{correct_intent.approach_direction})"
        f"  →  blocked({initial_intent.approach_direction})"
        f"  →  retry({revision_subtitle})",
    )
    frames = list(demo_frames) + list(attempt_frames) + list(retry_frames)
    return frames, title


def _natural_wrong_intent(correct_intent):
    """Single-factor flip of contact_region to the geometric opposite face.

    Models the PushCube vision-inference failure mode for the paper figure:
    the demo viewpoint hides which face the proxy contacted, so the encoder
    picks the opposite face. The skill compiler routes push direction
    through `face_to_push_unit(intent.contact_region)`, so executing this
    misgrounded intent visibly pushes the cube AWAY from the goal — no
    obstacle needed.
    """
    return replace(
        correct_intent,
        contact_region=_OPPOSITE_FACE[correct_intent.contact_region],
    )


def render_natural_failure_episode(
    env, adapter: BaseTaskAdapter, seed: int, fps: int,
) -> tuple[dict, dict]:
    """Paper-figure render: three phases, no obstacle, natural wrong-intent.

    Phase 1 (demo, third-person): execute the oracle correct intent — cube
        reaches the goal. This is the input the vision encoder would see.
    Phase 2 (attempt, wrist): execute a single-factor misgrounded intent
        (contact_region flipped to the opposite face). The push physically
        goes the wrong way and the cube moves AWAY from the goal.
    Phase 3 (retry, wrist): execute the revised intent (contact_region
        flipped back to the correct face). Cube reaches the goal. The
        revision is a single-factor edit — the other five factors are
        preserved.

    Returns the same `(frames_dict, titles_dict)` shape as `render_episode`
    but with phase keys ("demo", "attempt", "retry"). No obstacle / clutter
    is spawned: blocked_sides is empty throughout, so this render path does
    not depend on the obstacle-management helpers used by the Stage-0
    clutter render.
    """
    short_id = f"seed {seed:04d}"

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
    wrong_intent = _natural_wrong_intent(correct_intent)

    # === Phase 1 — demo (third-person, oracle correct intent) ===
    wp_demo = build_push_waypoints(scene, correct_intent)
    demo_frames: list = []
    _ = _execute_push(env, wp_demo, demo_frames, seed=seed)

    # === Phase 2 — attempt (wrist, misgrounded contact_region) ===
    # `build_push_waypoints` resolves push direction from contact_region,
    # so flipping the face is sufficient to make the cube go the wrong way
    # — no obstacle, no clutter, no scene mutation.
    wp_attempt = build_push_waypoints(scene, wrong_intent)
    attempt_frames: list = []
    out_attempt = _execute_push(
        env, wp_attempt, attempt_frames, seed=seed,
        capture=render_wrist_frame,
    )

    # === Phase 3 — retry (wrist, revised contact_region) ===
    # The Stage-0 contact_substitution heuristic prefers a 90°-orthogonal
    # face, which does not recover the goal-pushing face here. For the
    # paper figure we construct the inverse-of-misgrounding revision
    # directly — equivalent to a contact_substitution operator whose
    # candidate ordering opens with the opposite face. The general
    # revision pipeline's measured success is validated by the M3
    # baselines and the P1 latent policy, not by this figure.
    revised_intent = replace(
        wrong_intent, contact_region=correct_intent.contact_region,
    )
    wp_retry = build_push_waypoints(scene, revised_intent)
    retry_frames: list = []
    out_retry = _execute_push(
        env, wp_retry, retry_frames, seed=seed,
        capture=render_wrist_frame,
    )

    frozen = ", ".join(
        f for f in (
            "goal_state", "object_motion", "approach_direction",
            "constraint_region", "embodiment_mapping",
        )
    )
    demo_title = (
        f"{short_id}  phase 1/3: demo (oracle intent)",
        f"contact_region={correct_intent.contact_region}, "
        f"approach={correct_intent.approach_direction}",
    )
    attempt_title = (
        f"{short_id}  phase 2/3: attempt (misgrounded intent, "
        f"success={out_attempt['success']})",
        f"inferred contact_region={wrong_intent.contact_region} "
        f"(opposite face) → cube pushed away from goal",
    )
    retry_title = (
        f"{short_id}  phase 3/3: retry (success={out_retry['success']})",
        f"contact_region: {wrong_intent.contact_region} → "
        f"{revised_intent.contact_region}  |  frozen (preserved): {frozen}",
    )
    return (
        {"demo": demo_frames,
         "attempt": attempt_frames,
         "retry": retry_frames},
        {"demo": demo_title,
         "attempt": attempt_title,
         "retry": retry_title},
    )
