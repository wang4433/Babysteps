"""Pure scene geometry helpers — sim-agnostic.

Note: `SceneState` lives in babysteps.schemas (re-exported here for callers
that group it semantically with the helpers).

The face/approach vocabulary:
  - contact_region = which cube face the EE *touches*. To push the cube toward
    +x, the EE must touch the cube's -x face.
  - approach_direction = which path the EE took (or will take) to reach the
    face. By the Stage-0 convention, the demo's preferred approach pairs with
    its contact face (e.g. minus_x_face ↔ from_minus_x). approach_direction is
    a semantic feasibility tag — the physical waypoint geometry depends only
    on contact_region (see test_execution.py).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from babysteps.schemas import Intent, SceneState  # re-export

__all__ = [
    "SceneState",
    "direction_to_face",
    "face_to_approach",
    "face_to_push_unit",
    "approach_to_unit",
    "goal_direction_to_motion",
    "OPPOSITE_APPROACH",
    "ORTHOGONAL_FACE",
    "motion_to_unit",
    "rotate_xy",
    "rotate_motion_token",
    "resolve_grounded_motion",
    "world_resolved_intent",
]


# Mapping: cube face EE contacts → unit vector of cube travel when pushed.
_PUSH_UNIT_BY_FACE: dict[str, np.ndarray] = {
    "minus_x_face": np.array([1.0, 0.0]),
    "plus_x_face":  np.array([-1.0, 0.0]),
    "minus_y_face": np.array([0.0, 1.0]),
    "plus_y_face":  np.array([0.0, -1.0]),
}

# Mapping: cube face EE contacts → demo-default approach_direction.
_APPROACH_BY_FACE: dict[str, str] = {
    "minus_x_face": "from_minus_x",
    "plus_x_face":  "from_plus_x",
    "minus_y_face": "from_minus_y",
    "plus_y_face":  "from_plus_y",
}

# Mapping: approach_direction → xy unit vector pointing FROM cube TOWARD the
# side the EE should fly to first (before descending to the contact face).
# "from_above" maps to (0, 0): the high pre-contact waypoint sits directly
# above the cube center rather than offset to a side.
_APPROACH_UNIT: dict[str, np.ndarray] = {
    "from_minus_x": np.array([-1.0, 0.0]),
    "from_plus_x":  np.array([1.0, 0.0]),
    "from_minus_y": np.array([0.0, -1.0]),
    "from_plus_y":  np.array([0.0, 1.0]),
    "from_above":   np.array([0.0, 0.0]),
}

OPPOSITE_APPROACH: dict[str, str] = {
    "from_minus_x": "from_plus_x",
    "from_plus_x":  "from_minus_x",
    "from_minus_y": "from_plus_y",
    "from_plus_y":  "from_minus_y",
}

# Sub-project B (PickCube): the orthogonal (90°-around-z-rotated) gripper
# axis used by `contact_substitution`. minus_x_face → minus_y_face means
# "rotate the gripper from x-aligned to y-aligned." Pairs are
# distance-preserving (the EE only has to rotate the wrist, not move).
ORTHOGONAL_FACE: dict[str, str] = {
    "minus_x_face": "minus_y_face",
    "plus_x_face":  "plus_y_face",
    "minus_y_face": "minus_x_face",
    "plus_y_face":  "plus_x_face",
}


def direction_to_face(goal_vec_xy: np.ndarray) -> str:
    """Snap a (dx, dy) cube→goal vector to the cube face that should be
    contacted to produce that motion. Dominant axis wins; ties go to x."""
    v = np.asarray(goal_vec_xy, dtype=np.float64)
    if abs(v[0]) >= abs(v[1]):
        return "minus_x_face" if v[0] >= 0 else "plus_x_face"
    return "minus_y_face" if v[1] >= 0 else "plus_y_face"


def face_to_approach(face: str) -> str:
    """The default approach_direction the demo proxy reports for a given
    contact face (paired by Stage-0 convention)."""
    if face not in _APPROACH_BY_FACE:
        raise ValueError(f"unknown face {face!r}")
    return _APPROACH_BY_FACE[face]


def face_to_push_unit(face: str) -> np.ndarray:
    """Unit-vector direction of cube travel when EE contacts `face`."""
    if face not in _PUSH_UNIT_BY_FACE:
        raise ValueError(f"unknown face {face!r}")
    return _PUSH_UNIT_BY_FACE[face].copy()


def approach_to_unit(approach: str) -> np.ndarray:
    """Unit-vector (in cube-centred xy) pointing toward the side the EE should
    fly to before descending. Returns (0, 0) for "from_above"."""
    if approach not in _APPROACH_UNIT:
        raise ValueError(f"unknown approach {approach!r}")
    return _APPROACH_UNIT[approach].copy()


def goal_direction_to_motion(goal_vec_xy: np.ndarray) -> str:
    """Snap a (dx, dy) net-displacement vector to one of the four cardinal
    OBJECT_MOTIONS values. Dominant axis wins."""
    v = np.asarray(goal_vec_xy, dtype=np.float64)
    if abs(v[0]) >= abs(v[1]):
        return "translate_+x" if v[0] >= 0 else "translate_-x"
    return "translate_+y" if v[1] >= 0 else "translate_-y"


# ---------- cross-view grounding (Sub-project E) ----------------------- #

_MOTION_UNIT: dict[str, np.ndarray] = {
    "translate_+x": np.array([1.0, 0.0]),
    "translate_-x": np.array([-1.0, 0.0]),
    "translate_+y": np.array([0.0, 1.0]),
    "translate_-y": np.array([0.0, -1.0]),
}


def motion_to_unit(token: str) -> np.ndarray:
    """xy unit vector for a cardinal OBJECT_MOTIONS token."""
    if token not in _MOTION_UNIT:
        raise ValueError(f"motion_to_unit: {token!r} is not a cardinal translate token")
    return _MOTION_UNIT[token].copy()


def rotate_xy(p: tuple[float, float], yaw_deg: int) -> tuple[float, float]:
    """Rotate an xy point CCW by a multiple of 90°. Raises on other angles."""
    yaw = int(yaw_deg) % 360
    x, y = float(p[0]), float(p[1])
    if yaw == 0:
        return (x, y)
    if yaw == 90:
        return (-y, x)
    if yaw == 180:
        return (-x, -y)
    if yaw == 270:
        return (y, -x)
    raise ValueError(f"rotate_xy supports only multiples of 90°, got {yaw_deg}")


def rotate_motion_token(token: str, yaw_deg: int) -> str:
    """Rotate a cardinal motion token CCW by a multiple of 90°."""
    u = motion_to_unit(token)
    rx, ry = rotate_xy((float(u[0]), float(u[1])), yaw_deg)
    return goal_direction_to_motion(np.array([rx, ry], dtype=np.float64))


def resolve_grounded_motion(
    observed_motion: str, grounding: str, observer_yaw_deg: int,
) -> str:
    """Map an observer-relative cardinal motion to the world frame.

    actor_frame    → apply 0° (identity; the egocentric bug)
    observer_frame → apply observer_yaw_deg (the fix)
    object/world   → reserved (later cut) → NotImplementedError
    """
    if grounding == "actor_frame":
        applied = 0
    elif grounding == "observer_frame":
        applied = int(observer_yaw_deg)
    else:
        raise NotImplementedError(
            f"resolve_grounded_motion supports actor_frame/observer_frame in "
            f"Stage-0; got {grounding!r}"
        )
    return rotate_motion_token(observed_motion, applied)


def world_resolved_intent(intent: "Intent", observer_yaw_deg: int) -> "Intent":
    """Resolve an observer-relative intent into a plain world-frame push intent.

    Only `direction_grounding` controls the resolution; the resulting world
    object_motion / contact_region / approach_direction are derived from it.
    The returned intent's `direction_grounding` is set to world_frame (inert)."""
    world_motion = resolve_grounded_motion(
        intent.object_motion, intent.direction_grounding, observer_yaw_deg,
    )
    world_face = direction_to_face(motion_to_unit(world_motion))
    world_approach = face_to_approach(world_face)
    return replace(
        intent,
        object_motion=world_motion,
        contact_region=world_face,
        approach_direction=world_approach,
        direction_grounding="world_frame",
    )
