"""Render Stage-0 episodes as real ManiSkill RGB MP4s.

For each seed, runs the full BABYSTEPS loop (demo proxy → blocked attempt →
revised retry) in PushCube-v1, capturing `env.render()` frames per phase, and
writes one MP4 per phase to `<out_dir>/videos_maniskill/`.

This script needs Vulkan. On the Gilbreth login node it works via Mesa's
software Vulkan rasterizer (lavapipe) — slow but real. On a GPU compute node
it can use the NVIDIA Vulkan ICD and runs much faster.

Recommended invocation (login-node, lavapipe; ~minutes per episode):

    cd /home/wang4433/scratch/babysteps
    conda activate handover
    LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" \\
    VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json \\
    CUDA_VISIBLE_DEVICES="" \\
      python scripts/render_stage0_maniskill.py \\
        --out_dir datasets/stage0_pushcube_blocked --n_episodes 5 --seed_start 0
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Make the project root importable without `pip install -e .`.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Push-skill geometry (Pick4Pass calibration).
_POS_SCALE = 0.1
_PHASE_TOL_M = 0.015
_MAX_CONTROL_STEPS = 300


def _to_np(x):
    """Batched ManiSkill tensor (1, N) → flat numpy (N,)."""
    arr = x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)
    return arr[0] if arr.ndim == 2 else arr


def _raw_to_xyzw(raw_pose) -> np.ndarray:
    raw = np.asarray(raw_pose, dtype=np.float64)
    return np.concatenate([raw[0:3], raw[4:7], raw[3:4]])


def _read_obs(obs):
    tcp = _raw_to_xyzw(_to_np(obs["extra"]["tcp_pose"]))
    cube_full = _to_np(obs["extra"]["obj_pose"])
    cube_xy = cube_full[0:2].astype(np.float64)
    cube_z = float(cube_full[2])
    goal_xy = _to_np(obs["extra"]["goal_pos"])[0:2].astype(np.float64)
    return tcp, cube_xy, goal_xy, cube_z


def _prop_action(tcp_xyzw: np.ndarray, target_xyz: np.ndarray) -> np.ndarray:
    pos_err = target_xyz - tcp_xyzw[0:3]
    action = np.zeros(7, dtype=np.float32)
    action[0:3] = np.clip(pos_err / _POS_SCALE, -1.0, 1.0).astype(np.float32)
    action[6] = np.float32(-1.0)
    return action


def _render_frame(env) -> np.ndarray:
    """Get a (H, W, 3) uint8 RGB frame from env.render()."""
    f = env.render()
    if hasattr(f, "cpu"):
        f = f.cpu().numpy()
    f = np.asarray(f)
    if f.ndim == 4:
        f = f[0]
    if f.dtype != np.uint8:
        f = (255.0 * np.clip(f, 0.0, 1.0)).astype(np.uint8) if f.max() <= 1.0 \
            else f.astype(np.uint8)
    return f


def _execute_push(env, waypoints, capture_frames: list, *, seed: int) -> dict:
    """Step the env through the 3 waypoints, capturing one RGB frame per step.

    Re-resets the env with `seed` so demo / attempt / retry all start from the
    identical initial state — otherwise `env.reset()` would randomize and the
    pre-computed waypoints (built from a previous scene observation) would be
    aimed at the wrong cube position.
    """
    obs, _ = env.reset(seed=int(seed))
    tcp, cube_xy0, goal_xy, _ = _read_obs(obs)
    initial_obj_xy = (float(cube_xy0[0]), float(cube_xy0[1]))

    targets = [np.asarray(wp[0:3], dtype=np.float64) for wp in waypoints]
    phase_idx = 0
    success = False

    capture_frames.append(_render_frame(env))
    for _ in range(_MAX_CONTROL_STEPS):
        tcp, cube_xy, _, _ = _read_obs(obs)
        target = targets[phase_idx]
        if np.linalg.norm(target - tcp[0:3]) < _PHASE_TOL_M:
            phase_idx += 1
            if phase_idx >= len(targets):
                break
            target = targets[phase_idx]
        action = _prop_action(tcp, target)
        obs, _r, term, trunc, info = env.step(action)
        capture_frames.append(_render_frame(env))
        term_b = bool(_to_np(term).item()) if hasattr(term, "cpu") else bool(term)
        trunc_b = bool(_to_np(trunc).item()) if hasattr(trunc, "cpu") else bool(trunc)
        succ = info.get("success", False) if hasattr(info, "get") else False
        success = bool(_to_np(succ).item()) if hasattr(succ, "cpu") else bool(succ)
        if success or term_b or trunc_b:
            break

    tcp, final_cube_xy, _, _ = _read_obs(obs)
    return {
        "initial_obj_xy": initial_obj_xy,
        "final_obj_xy": (float(final_cube_xy[0]), float(final_cube_xy[1])),
        "success": bool(success),
    }


def _annotate_frame(frame: np.ndarray, title: str, subtitle: str = "") -> np.ndarray:
    """Add a black banner with the title across the top of the frame.

    Pure numpy + PIL — no matplotlib to keep this script slim."""
    from PIL import Image, ImageDraw, ImageFont
    img = Image.fromarray(frame)
    W, H = img.size
    banner_h = 60 if subtitle else 36
    canvas = Image.new("RGB", (W, H + banner_h), (16, 16, 16))
    canvas.paste(img, (0, banner_h))
    draw = ImageDraw.Draw(canvas)
    try:
        font_big = ImageFont.truetype(
            "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf", 16,
        )
        font_sm = ImageFont.truetype(
            "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf", 12,
        )
    except Exception:
        font_big = ImageFont.load_default()
        font_sm = ImageFont.load_default()
    draw.text((10, 6), title, fill=(255, 255, 255), font=font_big)
    if subtitle:
        draw.text((10, 30), subtitle, fill=(200, 200, 200), font=font_sm)
    return np.asarray(canvas)


def _save_mp4(frames: list, out_path: Path, fps: int) -> None:
    import imageio.v2 as imageio
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(out_path), fps=fps, codec="libx264", quality=8,
        macro_block_size=1,
    )
    for fr in frames:
        writer.append_data(fr)
    writer.close()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--n_episodes", type=int, default=5)
    p.add_argument("--seed_start", type=int, default=0)
    p.add_argument("--fps", type=int, default=20)
    args = p.parse_args(argv)

    try:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401
    except Exception as exc:
        print(
            f"ManiSkill import failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2

    from babysteps.demo import demo_to_intent
    from babysteps.envs.scene import direction_to_face, face_to_approach, face_to_push_unit
    from babysteps.skills.push import (
        CUBE_HALF_SIZE, PRE_CONTACT_STANDOFF,
        PUSH_TRAVEL_SCALE, PUSH_TRAVEL_MAX_M,
    )
    from babysteps.failure import attribute_failure, build_failure_packet
    from babysteps.revision import revise_intent
    from babysteps.schemas import AttemptResult, DemoEvidence, Intent, SceneState
    from babysteps.episode import (
        _default_blocked_sides_factory,
        _oracle_correct_intent_for_scene,
        generate_proxy_demo,
    )

    videos_dir = args.out_dir / "videos_maniskill"
    videos_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(
        "PushCube-v1",
        obs_mode="state_dict",
        control_mode="pd_ee_delta_pose",
        sim_backend="cpu",
        render_mode="rgb_array",
    )

    try:
        for i in range(args.n_episodes):
            seed = args.seed_start + i
            episode_id = f"pushcube_blocked_approach_seed_{seed:04d}"
            print(f"[{i + 1}/{args.n_episodes}] {episode_id}", flush=True)

            # === Phase 1: DEMO PROXY ===
            obs, _ = env.reset(seed=seed)
            tcp_xyzw, cube_xy0, goal_xy, cube_z = _read_obs(obs)
            tcp_start = tuple(float(v) for v in tcp_xyzw)
            scene = SceneState(
                cube_xy=(float(cube_xy0[0]), float(cube_xy0[1])),
                cube_z=cube_z,
                goal_xy=(float(goal_xy[0]), float(goal_xy[1])),
                tcp_start_pose=tcp_start,    # type: ignore[arg-type]
                blocked_sides=(),
            )
            correct_intent = _oracle_correct_intent_for_scene(scene)
            print(f"   demo intent: contact_region={correct_intent.contact_region} "
                  f"approach_direction={correct_intent.approach_direction}; "
                  f"cube_xy={scene.cube_xy} goal_xy={scene.goal_xy}")
            wp_demo = _build_waypoints(scene, correct_intent)
            demo_frames: list = []
            out = _execute_push(env, wp_demo, demo_frames, seed=seed)

            # Synthesize DemoEvidence (we use the oracle intent's contact_region;
            # the trajectory is what the oracle actually achieved).
            demo_evidence = DemoEvidence(
                camera="third_person",
                demonstrator_type="proxy_oracle",
                object_trajectory=(out["initial_obj_xy"], out["final_obj_xy"]),
                contact_region_label=correct_intent.contact_region,
                final_state="cube_at_target",
                rgbd_video_path=None,
            )

            # === Derive initial intent + blocked-sides ===
            initial_intent = demo_to_intent(demo_evidence)
            scene_exec = SceneState(
                cube_xy=scene.cube_xy, cube_z=scene.cube_z,
                goal_xy=scene.goal_xy, tcp_start_pose=scene.tcp_start_pose,
                blocked_sides=_default_blocked_sides_factory(initial_intent),
            )

            # === Phase 2: ATTEMPT 1 — planner_failed (approach blocked) ===
            # No env stepping — render a held-still RGB frame and repeat.
            obs, _ = env.reset(seed=seed)
            attempt1_frames = [_render_frame(env)] * (args.fps * 2)

            # === Phase 3: RETRY with revised approach ===
            failure_packet = build_failure_packet(
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
            attribution = attribute_failure(failure_packet)
            revised_intent, _rev = revise_intent(
                initial_intent, attribution, scene_exec,
            )
            print(f"   revised intent: approach_direction "
                  f"{initial_intent.approach_direction} → "
                  f"{revised_intent.approach_direction}")
            wp_retry = _build_waypoints(scene_exec, revised_intent)
            retry_frames: list = []
            retry_out = _execute_push(env, wp_retry, retry_frames, seed=seed)
            print(f"   demo_success={out['success']} retry_success={retry_out['success']}")

            # === Annotate + write three MP4s per episode ===
            # Titles fit ≤ 512 px at font size 16 (~30 chars).
            short_id = f"seed {seed:04d}"
            demo_title = f"{short_id}  phase 1/3: demo proxy"
            demo_sub = (
                f"contact_region={correct_intent.contact_region}, "
                f"approach={correct_intent.approach_direction}"
            )
            a1_title = f"{short_id}  phase 2/3: approach_blocked"
            a1_sub = (
                f"approach_direction={initial_intent.approach_direction} "
                f"is blocked → planner_failed"
            )
            retry_title = (
                f"{short_id}  phase 3/3: retry "
                f"(success={retry_out['success']})"
            )
            retry_sub = (
                f"approach_substitution: "
                f"{initial_intent.approach_direction} → "
                f"{revised_intent.approach_direction}"
            )

            demo_path = videos_dir / f"{episode_id}__1_demo.mp4"
            a1_path = videos_dir / f"{episode_id}__2_attempt_blocked.mp4"
            retry_path = videos_dir / f"{episode_id}__3_retry.mp4"

            _save_mp4(
                [_annotate_frame(fr, demo_title, demo_sub) for fr in demo_frames],
                demo_path, args.fps,
            )
            _save_mp4(
                [_annotate_frame(fr, a1_title, a1_sub) for fr in attempt1_frames],
                a1_path, args.fps,
            )
            _save_mp4(
                [_annotate_frame(fr, retry_title, retry_sub) for fr in retry_frames],
                retry_path, args.fps,
            )
            for vp in (demo_path, a1_path, retry_path):
                kb = vp.stat().st_size // 1024
                print(f"   wrote {vp.name}  ({kb} KB)")
    finally:
        env.close()

    print(f"\nDone. MP4s in {videos_dir}")
    return 0


def _build_waypoints(scene, intent) -> np.ndarray:
    """Local copy of build_push_waypoints — kept inline so this script is
    fully self-contained and doesn't drag in babysteps.skills.push's whitelist
    of cube_z conventions."""
    from babysteps.envs.scene import face_to_push_unit
    cube_xy = np.asarray(scene.cube_xy, dtype=np.float64)
    goal_xy = np.asarray(scene.goal_xy, dtype=np.float64)
    tcp = np.asarray(scene.tcp_start_pose, dtype=np.float64)
    travel_z = float(tcp[2])
    push_z = float(scene.cube_z)
    push_unit = face_to_push_unit(intent.contact_region)
    standoff = 0.02 + 0.005
    pre_contact_xy = cube_xy - push_unit * standoff
    cube_to_goal = float(np.linalg.norm(goal_xy - cube_xy))
    push_travel = min(0.6 * cube_to_goal, 0.15)
    push_end_xy = cube_xy + push_unit * push_travel

    wp = np.zeros((3, 7), dtype=np.float64)
    wp[0, 0:2] = pre_contact_xy
    wp[0, 2] = travel_z
    wp[1, 0:2] = pre_contact_xy
    wp[1, 2] = push_z
    wp[2, 0:2] = push_end_xy
    wp[2, 2] = push_z
    wp[:, 3:7] = tcp[3:7]
    return wp


if __name__ == "__main__":
    sys.exit(main())
