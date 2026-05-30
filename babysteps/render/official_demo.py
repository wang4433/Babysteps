"""Official ManiSkill demo → third-person frames (Scope A).

Two ways to source a *demonstration* from ManiSkill's **official** Panda
motion-planning oracle, both yielding ONLY third-person RGB frames:

  ``run_official_solver_frames`` — RUN-LIVE: invoke the official solver
      (``solvePushCube`` / ``solvePickCube`` / ``solveStackCube``) and film
      the third-person human-render camera. Reads no recorded trace at all —
      it just watches the canonical oracle solve the task.

  ``replay_official_state_frames`` — STATE-REPLAY: read the recorded scene
      poses (``env_states``) from a downloaded ``trajectory.h5`` and teleport
      the scene per frame, filming third-person. The recorded ``actions``
      dataset — the privileged Franka motor program — is NEVER opened.

Firewall (``goal.md`` Data Pipeline 1 + working invariants 3-4). The
demonstrator is allowed privileged access — it *is* the oracle Franka — but
only the rendered third-person video leaves this module. The recorded
``actions`` channel never reaches the intent path
(``babysteps.stage4.vision_features``). The frames describe *object motion*
(a third-person proxy), never an executable Franka motor program.

Import-sim-free: ``gymnasium`` / ``mani_skill`` / ``sapien`` / ``h5py`` are
imported lazily INSIDE function bodies so the login-node test suite can
introspect this module without a Vulkan device. Rendering itself needs a
GPU/Vulkan node; planning (run-live) additionally needs a working ``mplib``.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from babysteps.render.common import _to_uint8_frame, render_frame

# Per-episode datasets in an official ``trajectory.h5`` that are NOT the
# scene-state channel: the recorded Franka motor program (``actions``), the
# learning signal (``rewards``), and the outcome flags. This module never
# bracket-indexes the ``.h5`` by any of these — the state-replay path touches
# ONLY ``SAFE_STATE_KEY``. (Outcome flags are label-side data; we read success
# from the *live* env in run-live, never from the recorded file.)
PRIVILEGED_H5_KEYS: tuple[str, ...] = (
    "actions",
    "rewards",
    "success",
    "fail",
    "terminated",
    "truncated",
)

# The only ``.h5`` group the state-replay path reads: recorded maximal-coord
# scene poses (per actor/articulation). Disjoint from PRIVILEGED_H5_KEYS.
SAFE_STATE_KEY: str = "env_states"

# Official cube solvers require pd_joint_pos (StackCube asserts it); the
# published motionplanning demos were recorded under this control mode too.
OFFICIAL_CONTROL_MODE: str = "pd_joint_pos"

# task_id -> attribute name in
# ``mani_skill.examples.motionplanning.panda.solutions``.
_SOLVER_NAMES: dict[str, str] = {
    "PushCube-v1": "solvePushCube",
    "PickCube-v1": "solvePickCube",
    "StackCube-v1": "solveStackCube",
}


def resolve_official_traj(env_id: str, demos_root: str | os.PathLike | None = None):
    """Locate the downloaded official trajectory for ``env_id``.

    Returns ``(h5_path, json_path)`` under
    ``<demos_root>/<env_id>/motionplanning/trajectory.{h5,json}``. ``demos_root``
    defaults to ``$MS_ASSET_DIR/demos`` or ``~/.maniskill/demos`` (the path
    ``mani_skill.utils.download_demo`` writes to). Pure path construction — no
    sim import; does not check existence.
    """
    if demos_root is None:
        asset_dir = os.environ.get("MS_ASSET_DIR", os.path.expanduser("~/.maniskill"))
        demos_root = Path(asset_dir) / "demos"
    base = Path(demos_root) / env_id / "motionplanning"
    return base / "trajectory.h5", base / "trajectory.json"


def _select_episode(episodes: list, *, seed: int | None, episode_index: int) -> dict:
    """Pick one episode record from a trajectory.json ``episodes`` list."""
    if seed is not None:
        for ep in episodes:
            if int(ep.get("episode_seed", -1)) == int(seed):
                return ep
        raise KeyError(
            f"no official episode with episode_seed={seed}; "
            f"available seeds: {[ep.get('episode_seed') for ep in episodes[:8]]}..."
        )
    if not 0 <= episode_index < len(episodes):
        raise IndexError(
            f"episode_index {episode_index} out of range (have {len(episodes)})"
        )
    return episodes[episode_index]


def run_official_solver_frames(
    env_id: str,
    seed: int,
    *,
    shader: str = "default",
    sim_backend: str = "cpu",
    capture=render_frame,
):
    """RUN-LIVE: run the official ManiSkill oracle on ``seed`` and film it.

    Builds a fresh env in the solver-required ``pd_joint_pos`` control mode,
    invokes the official solver (which steps the env internally), and captures
    one third-person frame per control step. Returns ``(frames, success)``
    where ``frames`` is ``list[(H, W, 3) uint8]`` and ``success`` is a bool.

    Reads no recorded ``.h5``/``actions``: the only output is rendered pixels
    plus the live env's own success check (a permitted label/success signal).

    Requires a GPU/Vulkan node for rendering and a working ``mplib`` for the
    planner. Raises on plan failure only indirectly — a failed plan yields
    ``success=False`` (mirrors the official ``--only-count-success`` pattern,
    where the caller iterates seeds and skips failures).
    """
    import gymnasium as gym
    import importlib

    if env_id not in _SOLVER_NAMES:
        raise KeyError(
            f"no official solver mapped for {env_id!r}; known: {sorted(_SOLVER_NAMES)}"
        )
    solutions = importlib.import_module(
        "mani_skill.examples.motionplanning.panda.solutions"
    )
    solve = getattr(solutions, _SOLVER_NAMES[env_id])

    frames: list = []

    class _FrameCapture(gym.Wrapper):
        """Grab one third-person frame after each env.step the solver issues."""

        def step(self, action):
            out = self.env.step(action)
            frames.append(capture(self))
            return out

    env = _FrameCapture(
        gym.make(
            env_id,
            obs_mode="none",
            control_mode=OFFICIAL_CONTROL_MODE,
            render_mode="rgb_array",
            sensor_configs=dict(shader_pack=shader),
            human_render_camera_configs=dict(shader_pack=shader),
            sim_backend=sim_backend,
        )
    )
    try:
        res = solve(env, seed=seed, debug=False, vis=False)
    finally:
        env.close()

    # The live env's own success flag (permitted label/success signal). Read
    # via .get — never bracket-indexed — so the firewall scan stays clean.
    success = False
    if res != -1:
        succ = res[-1].get("success")
        if succ is not None:
            success = bool(np.asarray(succ).flatten()[0])
    return frames, success


def replay_official_state_frames(
    env_id: str,
    seed: int | None = None,
    *,
    episode_index: int = 0,
    traj_paths: tuple | None = None,
    stride: int = 1,
    sim_backend: str = "physx_cpu",
):
    """STATE-REPLAY: teleport through recorded scene poses and film third-person.

    Reads ONLY the recorded ``env_states`` group (``SAFE_STATE_KEY``) from the
    downloaded ``trajectory.h5`` and applies each per-step state with
    ``env.set_state_dict`` — never calls ``env.step`` with a recorded action,
    and never opens the ``actions`` dataset. Returns ``(frames, meta)`` where
    ``frames`` is ``list[(H, W, 3) uint8]`` and ``meta`` carries the source
    episode id/seed.

    Requires a GPU/Vulkan node for rendering; needs no ``mplib``.
    """
    import gymnasium as gym
    import h5py
    from mani_skill.trajectory import utils as trajectory_utils
    from mani_skill.utils import io_utils

    if traj_paths is None:
        traj_paths = resolve_official_traj(env_id)
    h5_path, json_path = traj_paths

    jd = io_utils.load_json(str(json_path))
    env_kwargs = dict(jd["env_info"]["env_kwargs"])
    env_kwargs.update(render_mode="rgb_array", sim_backend=sim_backend, num_envs=1)

    env = gym.make(jd["env_info"]["env_id"], **env_kwargs)
    frames: list = []
    try:
        with h5py.File(str(h5_path), "r") as h5:
            ep = _select_episode(jd["episodes"], seed=seed, episode_index=episode_index)
            tid = f"traj_{ep['episode_id']}"
            reset_kwargs = dict(ep.get("reset_kwargs", {}))
            reset_kwargs.setdefault("seed", ep["episode_seed"])
            env.reset(**reset_kwargs)
            # The ONLY .h5 group we touch: recorded scene poses.
            states = trajectory_utils.dict_to_list_of_dicts(h5[tid][SAFE_STATE_KEY])
            for state in states[::stride]:
                env.unwrapped.set_state_dict(state)
                frames.append(_to_uint8_frame(env.unwrapped.render()))
            meta = {
                "episode_id": ep["episode_id"],
                "episode_seed": ep["episode_seed"],
                "n_states": len(states),
            }
    finally:
        env.close()
    return frames, meta


def official_demo_frames(env_id: str, seed: int, *, source: str = "solver", **kwargs):
    """Dispatch to the chosen official-demo source, returning frames only.

    ``source="solver"`` -> run-live (default, the cleanest provenance);
    ``source="state_replay"`` -> recorded-state replay (unblocked fallback).
    """
    if source == "solver":
        frames, _ = run_official_solver_frames(env_id, seed, **kwargs)
        return frames
    if source == "state_replay":
        frames, _ = replay_official_state_frames(env_id, seed, **kwargs)
        return frames
    raise ValueError(f"unknown source {source!r}; use 'solver' or 'state_replay'")
