"""Stage-5 — sim-free attribution dataset (PURE geometry, no GPU/Vulkan).

Builds ``(modalities -> revisable factor)`` training/eval examples for the
*distilled multimodal attributor* (build-order step 4) directly from the
deterministic PokeCube LOTO geometry — the same ``face_to_push_unit`` /
``motion_to_unit`` table the real loop uses. Imports ZERO gitignored
``models/`` or ``datasets/stage5/`` artifacts, so the unit suite runs on the
login node.

Two example families:

* **clean contact_region failures** — the natural PokeCube mis-grounding: the
  demo grounds the goal direction correctly but the *contacted face* is wrong.
  ``true_factor = contact_region``. The 6 (direction × wrong-face) cases give 6
  distinct residual directions, each a clean 1:1 cue for the correct face. This
  is exactly the **positional shortcut** a residual-only head would learn.

* **hard negatives (object_motion misread)** — Class A from the design. The
  perception reads the *wrong goal direction*, so the robot pokes the face that
  is *correct for the misread direction*. The resulting residual is
  **byte-identical** to a clean contact_region failure (same residual, same
  contacted face, same trajectory); the ONLY difference is the inferred
  ``object_motion`` token in the intent. ``true_factor = object_motion``.

The hard negatives are what make "which factor?" a non-trivial label: a head
that reads only the residual sees an identical input for a matched
clean/hard-negative pair and cannot separate them. Only the symbolic
intent-context modality (the inferred ``object_motion`` token) breaks the tie —
which is the whole point of mandating a multimodal interface rather than a
residual-only rule. See ``attribution_head.py`` for the consumer.

HONESTY NOTE: this is a *contact-region baseline / diagnostic proof-of-concept*
on synthetic geometry. It demonstrates that the architecture CAN use a
non-positional modality to defeat the residual shortcut; whether that
distinguishing signal arises naturally from real pixels/features is a GPU /
real-data question (build-order step 5), not something these synthetic examples
prove.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from babysteps.envs.scene import face_to_approach, face_to_push_unit, motion_to_unit
from babysteps.schemas import Intent

# The 3 reachable PokeCube LOTO directions (−x is reach-dead, excluded) and the
# oracle correct face per direction. Mirrors stage5_pokecube_loto_eval.py:57-60.
LOTO_DIRECTIONS: tuple[str, ...] = ("+x", "+y", "-y")
_DIR_TO_FACE: dict[str, str] = {
    "+x": "minus_x_face", "+y": "minus_y_face", "-y": "plus_y_face"}
_DIR_TO_MOTION: dict[str, str] = {
    "+x": "translate_+x", "+y": "translate_+y", "-y": "translate_-y"}
# The 3 reachable contact faces (one per direction); plus_x_face is reach-dead
# and intentionally excluded (mirrors LOTO_FACES in stage5_pokecube_loto_eval.py).
REACHABLE_FACES: tuple[str, ...] = tuple(_DIR_TO_FACE.values())
_FACE_ORDER: tuple[str, ...] = (
    "minus_x_face", "plus_x_face", "minus_y_face", "plus_y_face")

# Default PokeCube failure predicate (fallback for direct Example construction).
DEFAULT_PREDICATE: str = "direction_error"

PUSH_DISTANCE: float = 0.10   # nominal poke displacement (m), matches the env.


@dataclass(frozen=True)
class Example:
    """One ``(modalities -> factor)`` attribution example.

    ``residual_xy`` / ``trajectory_xy`` are the OBSERVABLE execution feedback
    (non-privileged). ``initial_intent`` carries the symbolic context (the
    inferred ``object_motion`` token is the hard-negative separator).
    ``true_factor`` is the LABEL the attributor predicts; ``correct_value`` is
    the would-be value fix (consumed downstream by the editor, NOT by the
    attribution head)."""
    residual_xy: tuple[float, float]
    trajectory_xy: tuple[tuple[float, float], ...]
    initial_intent: Intent
    true_factor: str
    correct_value: str
    predicate: str = DEFAULT_PREDICATE
    obs_feat: Optional[tuple[float, ...]] = None
    factor_menu: tuple[str, ...] = ("goal_state", "object_motion",
                                    "contact_region", "approach_direction",
                                    "constraint_region", "embodiment_mapping")
    meta: dict = field(default_factory=dict)


def _poke_intent(*, object_motion: str, contact_region: str) -> Intent:
    """Canonical PokeCube intent with a given (object_motion, contact_region).

    The other factors are held at their PokeCube defaults — the attribution head
    only reads object_motion + contact_region for context, but a full valid
    Intent keeps the example faithful and JSON-roundtrippable."""
    return Intent(
        goal_state="cube_at_target",
        object_motion=object_motion,
        contact_region=contact_region,
        approach_direction=face_to_approach(contact_region),
        constraint_region="none",
        embodiment_mapping="proxy_contact_to_franka_push",
    )


def _trajectory(cube0: np.ndarray, final: np.ndarray, k: int = 5
                ) -> tuple[tuple[float, float], ...]:
    """Linear cube path cube0 -> final, k points (observable execution trace)."""
    pts = [cube0 + (final - cube0) * (i / (k - 1)) for i in range(k)]
    return tuple((float(p[0]), float(p[1])) for p in pts)


def _predicate_for(direction: str, wrong_face: str) -> str:
    """The failure predicate the real loop emits for a wrong-face poke (matches
    ``babysteps.failure.build_failure_packet`` direction-alignment branch): a
    cube pushed OPPOSITE the goal axis -> ``direction_error``; a cube pushed
    PERPENDICULAR (alignment ~0) -> ``goal_not_satisfied``. Keying the predicate
    on (direction, wrong_face) keeps it IDENTICAL within a matched clean/hardneg
    pair (no factor leak) while matching the deployed predicate MIX (4
    perpendicular + 2 opposite over the LOTO grid), so the distilled head's
    residual block is domain-faithful for the recovery gate."""
    goal = motion_to_unit(_DIR_TO_MOTION[direction])
    push = face_to_push_unit(wrong_face)
    return "direction_error" if float(np.dot(push, goal)) < -1e-6 \
        else "goal_not_satisfied"


def _noise_vec(direction: str, wrong_face: str, replicate: int,
               base_seed: int, noise: float) -> np.ndarray:
    """Deterministic observation noise keyed by (direction, wrong_face,
    replicate, base_seed).

    A clean example and its matched hard negative share the SAME
    (direction, wrong_face, replicate) key, so they get IDENTICAL noise and
    therefore a BYTE-IDENTICAL residual at any noise level. This makes the
    headline (residual-only collapses on hard negatives) immune to the "it's
    just a small-noise artifact" objection. Uses a stable integer seed (NOT
    Python ``hash``, which is salted per process)."""
    if noise <= 0:
        return np.zeros(2, dtype=np.float64)
    d_idx = LOTO_DIRECTIONS.index(direction) if direction in LOTO_DIRECTIONS else 0
    f_idx = _FACE_ORDER.index(wrong_face) if wrong_face in _FACE_ORDER else 0
    s = (int(base_seed) * 1_000_003 + int(replicate) * 10_007
         + d_idx * 131 + f_idx) % (2 ** 32)
    return np.random.default_rng(s).normal(0.0, noise, size=2)


def _example(*, direction: str, wrong_face: str, object_motion: str,
             true_factor: str, correct_value: str, cube0: np.ndarray,
             replicate: int, base_seed: int, noise: float,
             kind: str) -> Example:
    """Assemble one example from the deterministic geometry.

    The cube ends at ``cube0 + PUSH_DISTANCE * push_unit(wrong_face)``; the goal
    is ``cube0 + PUSH_DISTANCE * motion_unit(direction)``; the residual is
    ``goal - final`` plus deterministic per-(direction, wrong_face, replicate)
    observation noise (so a matched clean/hard-negative pair stays
    byte-identical). ``object_motion`` is set independently of ``direction`` so
    a hard negative can carry a MISREAD motion token while the residual matches
    a clean example."""
    goal_unit = motion_to_unit(_DIR_TO_MOTION[direction])
    push_unit = face_to_push_unit(wrong_face)
    goal = cube0 + PUSH_DISTANCE * goal_unit
    final = cube0 + PUSH_DISTANCE * push_unit
    residual = goal - final + _noise_vec(
        direction, wrong_face, replicate, base_seed, noise)
    return Example(
        residual_xy=(float(residual[0]), float(residual[1])),
        trajectory_xy=_trajectory(cube0, final),
        initial_intent=_poke_intent(object_motion=object_motion,
                                    contact_region=wrong_face),
        true_factor=true_factor,
        correct_value=correct_value,
        predicate=_predicate_for(direction, wrong_face),
        meta={"kind": kind, "direction": direction, "wrong_face": wrong_face},
    )


def make_contactregion_examples(
    *, directions: tuple[str, ...] = LOTO_DIRECTIONS, n_per_case: int = 1,
    noise: float = 0.0, seed: int = 0, cube0: tuple[float, float] = (0.0, 0.0),
) -> list[Example]:
    """Clean contact_region failures: correct goal direction, WRONG contacted
    face. For each direction d and each reachable face w != correct(d), the cube
    travels along w's push direction and undershoots the goal, leaving a residual
    that points 1:1 to the correct face. ``true_factor = contact_region``."""
    c0 = np.asarray(cube0, dtype=np.float64)
    out: list[Example] = []
    for d in directions:
        correct = _DIR_TO_FACE[d]
        motion = _DIR_TO_MOTION[d]            # CORRECT inferred motion
        for w in REACHABLE_FACES:            # 3 reachable faces (−x reach-dead)
            if w == correct:
                continue
            for r in range(n_per_case):
                out.append(_example(
                    direction=d, wrong_face=w, object_motion=motion,
                    true_factor="contact_region", correct_value=correct,
                    cube0=c0, replicate=r, base_seed=seed, noise=noise,
                    kind="clean"))
    return out


def make_hardneg_objmotion_examples(
    *, directions: tuple[str, ...] = LOTO_DIRECTIONS, n_per_case: int = 1,
    noise: float = 0.0, seed: int = 0, cube0: tuple[float, float] = (0.0, 0.0),
) -> list[Example]:
    """Class-A hard negatives: the goal direction is MISREAD.

    For a true direction ``d_true`` and a misread ``d_inf != d_true``, the robot
    pokes ``correct(d_inf)`` (the right face *for the misread motion*). The cube
    moves along that face's push direction, so the residual equals
    ``0.10*(motion(d_true) - push(correct(d_inf)))`` — IDENTICAL to the clean
    example ``(direction=d_true, wrong_face=correct(d_inf))`` (same residual,
    same contacted face, same trajectory). The ONLY difference is
    ``initial_intent.object_motion`` (misread ``d_inf`` vs correct ``d_true``).
    ``true_factor = object_motion`` (the misread goal direction)."""
    c0 = np.asarray(cube0, dtype=np.float64)
    out: list[Example] = []
    for d_true in directions:
        for d_inf in directions:
            if d_inf == d_true:
                continue
            w_face = _DIR_TO_FACE[d_inf]       # correct face FOR the misread dir
            for r in range(n_per_case):
                out.append(_example(
                    direction=d_true, wrong_face=w_face,
                    object_motion=_DIR_TO_MOTION[d_inf],   # MISREAD motion token
                    true_factor="object_motion",
                    correct_value=_DIR_TO_MOTION[d_true],
                    cube0=c0, replicate=r, base_seed=seed, noise=noise,
                    kind="hardneg_objmotion"))
    return out


def make_dataset(
    *, n_per_case: int = 1, noise: float = 0.0, include_hardneg: bool = True,
    seed: int = 0,
) -> list[Example]:
    """Combined clean (+ hard-negative) dataset.

    Clean and hard negatives use the SAME ``seed`` so a matched pair (same
    direction, wrong_face, replicate) gets identical observation noise and a
    byte-identical residual — the residual-only arm therefore sees a truly
    identical input for the pair (the headline cannot be a noise artifact)."""
    ex = make_contactregion_examples(
        n_per_case=n_per_case, noise=noise, seed=seed)
    if include_hardneg:
        ex += make_hardneg_objmotion_examples(
            n_per_case=n_per_case, noise=noise, seed=seed)
    return ex
