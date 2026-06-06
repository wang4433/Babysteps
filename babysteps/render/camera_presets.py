"""Shared high-oblique camera presets for the Stage-5 dual-camera demo render.

The Stage-5 dual-camera setup renders the third-person demo from a GLOBAL
high-oblique view (and, for the dual-stream extractor, a second external CONTACT
view). These presets are the canonical eye/target poses, shared by the GPU
render driver (`scripts/stage5_render_demo_frames.py`) and the goal_state probe
(`scripts/stage5_goal_state_probe.py`) so the two never drift.

Design contract (do not break):
  * The oblique presets KEEP world-z as image-up (up=(0,0,1)), so a stacked
    tower reads as vertical extent — the cue that distinguishes StackCube
    `goal_state` (stack-on vs place-near).
  * They are deliberately NOT nadir. Pure top-down (the legacy
    `_topdown_camera_configs`, eye=[0,0,0.65], up=(1,0,0)) collapses the height
    that DEFINES stacking, leaving only XY-footprint coincidence — a
    near-tautology with the success label (the reviewer trap). Nadir lives in
    the render script, not here.

Sim-free: the pose tuples + `camera_elevation_deg` need no simulator; only
`look_at_pose_list` / `oblique_camera_configs` import mani_skill (lazily).
"""
from __future__ import annotations

import numpy as np

# eye/target in world meters. "default" = no override (ManiSkill render_camera).
CAMERA_PRESETS: dict[str, tuple | None] = {
    "default": None,  # ManiSkill render_camera (low oblique ~15deg)
    "oblique_high": ((0.45, 0.0, 0.60), (0.0, 0.0, 0.05)),     # front-elevated ~51deg
    "oblique_higher": ((0.25, 0.0, 0.70), (0.0, 0.0, 0.05)),   # steep front ~69deg (NOT nadir)
    "oblique_corner": ((0.40, 0.40, 0.70), (0.0, 0.0, 0.05)),  # corner-elevated
}


def camera_elevation_deg(eye, target) -> float:
    """Elevation angle (deg above the table plane) of an eye->target view.

    90deg = straight-down nadir, 0deg = horizontal. Pure geometry (sim-free).
    """
    e = np.asarray(eye, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    d = e - t
    horiz = float(np.hypot(d[0], d[1]))
    return float(np.degrees(np.arctan2(abs(d[2]), horiz))) if horiz > 1e-9 else 90.0


def look_at_pose_list(eye, target, up=(0.0, 0.0, 1.0)) -> list:
    """[x,y,z,qw,qx,qy,qz] pose list for gym.make human_render_camera_configs.

    GPU: imports mani_skill lazily so this module stays login-node importable.
    """
    from mani_skill.utils import sapien_utils

    pose = sapien_utils.look_at(eye=list(eye), target=list(target), up=tuple(up))
    raw = pose.raw_pose
    raw = raw[0] if getattr(raw, "ndim", 1) == 2 else raw
    return [float(v) for v in (
        raw.cpu().numpy() if hasattr(raw, "cpu") else np.asarray(raw)
    )]


def oblique_camera_configs(camera: str = "default", *, uid: str = "render_camera",
                           eye=None, target=None):
    """human_render_camera_configs override {uid: {"pose": ...}} for a preset.

    Returns None for "default" (no override). An explicit eye/target pair
    overrides the named preset (pose sweep). Raises on an unknown/None preset.
    """
    if eye is not None and target is not None:
        return {uid: {"pose": look_at_pose_list(eye, target)}}
    if camera == "default":
        return None
    if camera not in CAMERA_PRESETS or CAMERA_PRESETS[camera] is None:
        raise ValueError(f"unknown/None camera preset {camera!r}")
    return {uid: {"pose": look_at_pose_list(*CAMERA_PRESETS[camera])}}
