"""Shared utilities for per-task render modules.

Pulled out of scripts/render_stage0_maniskill.py so both pushcube.py
and pickcube.py (and future stackcube.py / drawer.py) can reuse them
without circular dependency on the script."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

# Phase-control constants — match the PD calibration of the env_runners.
POS_SCALE: float = 0.1
PHASE_TOL_M: float = 0.015
# Per-task control-step caps. Render modules use the cap that matches
# their task's real env_runner so render visuals do not run longer than
# what the production pipeline would have produced.
PUSHCUBE_MAX_CONTROL_STEPS: int = 300   # matches PushCubeEnvRunner._MAX_CONTROL_STEPS
PICKCUBE_MAX_CONTROL_STEPS: int = 400   # matches PickCubeEnvRunner._MAX_CONTROL_STEPS
MAX_CONTROL_STEPS: int = 400   # back-compat alias for callers that don't care


def to_np(x):
    """Convert a possibly-batched torch/cuda tensor to a flat numpy view."""
    arr = x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)
    return arr[0] if arr.ndim == 2 else arr


def raw_to_xyzw(raw_pose) -> np.ndarray:
    """ManiSkill's pose comes as [x, y, z, qw, qx, qy, qz]; we want xyzw."""
    raw = np.asarray(raw_pose, dtype=np.float64)
    return np.concatenate([raw[0:3], raw[4:7], raw[3:4]])


def read_obs(obs) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """(tcp_xyzw, cube_xy, goal_xy, cube_z) from a PushCube/PickCube obs."""
    tcp = raw_to_xyzw(to_np(obs["extra"]["tcp_pose"]))
    cube_full = to_np(obs["extra"]["obj_pose"])
    cube_xy = cube_full[0:2].astype(np.float64)
    cube_z = float(cube_full[2])
    goal_xy = to_np(obs["extra"]["goal_pos"])[0:2].astype(np.float64)
    return tcp, cube_xy, goal_xy, cube_z


def prop_action(
    tcp_xyzw: np.ndarray, target_xyz: np.ndarray, gripper_cmd: float = -1.0,
) -> np.ndarray:
    """Proportional 7-dim action toward target_xyz with explicit gripper cmd.
    Default gripper_cmd=-1 (closed) matches PushSkill's behavior."""
    pos_err = target_xyz - tcp_xyzw[0:3]
    action = np.zeros(7, dtype=np.float32)
    action[0:3] = np.clip(pos_err / POS_SCALE, -1.0, 1.0).astype(np.float32)
    action[6] = np.float32(gripper_cmd)
    return action


def render_frame(env) -> np.ndarray:
    """One (H, W, 3) uint8 RGB frame from env.render()."""
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


def annotate_frame(
    frame: np.ndarray, title: str, subtitle: str = "",
) -> np.ndarray:
    """Add a black banner with title (and optional subtitle) above frame."""
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


def save_mp4(frames: Iterable[np.ndarray], out_path: Path, fps: int) -> None:
    """Write `frames` to `out_path` as H.264 MP4.

    `frames` is materialized to a list up front so that a partially-
    consumed generator (easy to produce with a chained expression) does
    not silently yield a truncated MP4 with no error."""
    import imageio.v2 as imageio
    frames = list(frames)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(out_path), fps=fps, codec="libx264", quality=8,
        macro_block_size=1,
    )
    for fr in frames:
        writer.append_data(fr)
    writer.close()
