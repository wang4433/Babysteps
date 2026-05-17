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

import numpy as np

from babysteps.schemas import SceneState  # re-export

__all__ = [
    "SceneState",
    "direction_to_face",
    "face_to_approach",
    "face_to_push_unit",
    "goal_direction_to_motion",
    "OPPOSITE_APPROACH",
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

OPPOSITE_APPROACH: dict[str, str] = {
    "from_minus_x": "from_plus_x",
    "from_plus_x":  "from_minus_x",
    "from_minus_y": "from_plus_y",
    "from_plus_y":  "from_minus_y",
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


def goal_direction_to_motion(goal_vec_xy: np.ndarray) -> str:
    """Snap a (dx, dy) net-displacement vector to one of the four cardinal
    OBJECT_MOTIONS values. Dominant axis wins."""
    v = np.asarray(goal_vec_xy, dtype=np.float64)
    if abs(v[0]) >= abs(v[1]):
        return "translate_+x" if v[0] >= 0 else "translate_-x"
    return "translate_+y" if v[1] >= 0 else "translate_-y"
