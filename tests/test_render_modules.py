"""Tests for per-task render modules (babysteps.render.{pushcube,pickcube}).

These tests use a stub env that mimics gymnasium's reset/step/render API
with deterministic obs and frames, so we exercise the per-task phase
logic (waypoint dispatch, gripper schedule, frame counts) without
needing ManiSkill or a GPU."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest


# ---------- Stub env --------------------------------------------------- #


@dataclass
class _StubObs:
    """A dict-like obs with extra.{tcp_pose, obj_pose, goal_pos}."""
    tcp: np.ndarray
    cube: np.ndarray
    goal: np.ndarray

    def __getitem__(self, key: str):
        if key == "extra":
            # Match ManiSkill's pose convention: [x,y,z,qw,qx,qy,qz].
            tcp_raw = np.concatenate([self.tcp[0:3], np.array([1.0]),
                                      self.tcp[3:6]])
            cube_full = np.array([self.cube[0], self.cube[1], 0.02])
            goal_full = np.array([self.goal[0], self.goal[1], 0.02])
            return {"tcp_pose": tcp_raw, "obj_pose": cube_full,
                    "goal_pos": goal_full}
        raise KeyError(key)


class _StubEnv:
    """Drop-in stand-in for gym.make('PushCube-v1') / 'PickCube-v1'.

    reset(seed) places the cube at the origin and the goal at +x. The TCP
    'tracks' the action target deterministically each step (just integrates
    the action) so phase transitions happen predictably."""

    def __init__(self) -> None:
        self.tcp = np.array([0.0, 0.0, 0.25, 0.0, 0.0, 0.0], dtype=np.float64)
        self.cube = np.array([0.0, 0.0], dtype=np.float64)
        self.goal = np.array([0.12, 0.0], dtype=np.float64)
        self._step_count = 0

    def reset(self, seed: int = 0):
        self.tcp = np.array([0.0, 0.0, 0.25, 0.0, 0.0, 0.0], dtype=np.float64)
        self.cube = np.array([0.0, 0.0], dtype=np.float64)
        self.goal = np.array([0.12, 0.0], dtype=np.float64)
        self._step_count = 0
        return _StubObs(self.tcp, self.cube, self.goal), {}

    def step(self, action):
        # Integrate xyz error directly so target is reached in ~10 steps per phase.
        self.tcp[0:3] = self.tcp[0:3] + 0.02 * np.asarray(action[0:3])
        self._step_count += 1
        return (
            _StubObs(self.tcp, self.cube, self.goal),
            0.0, False, False,
            {"success": False},
        )

    def render(self):
        # Return a tiny deterministic RGB frame (8x8x3 uint8).
        return (np.ones((8, 8, 3), dtype=np.uint8) * (self._step_count % 256))

    def close(self):
        pass


# ---------- PushCube render tests -------------------------------------- #


def test_pushcube_render_episode_emits_three_phase_frames():
    """render_episode returns frames dict with demo/attempt_blocked/retry."""
    from babysteps.render.pushcube import render_episode
    from babysteps.envs.pushcube_adapter import PushCubeAdapter

    env = _StubEnv()
    adapter = PushCubeAdapter()
    frames, titles = render_episode(env, adapter, seed=0, fps=4)

    assert set(frames.keys()) == {"demo", "attempt_blocked", "retry"}
    assert set(titles.keys()) == {"demo", "attempt_blocked", "retry"}
    # All three phases must produce at least one frame.
    assert len(frames["demo"]) >= 1
    assert len(frames["attempt_blocked"]) >= 1  # PushCube: held-still loop
    assert len(frames["retry"]) >= 1
    # PushCube's attempt_blocked is a held-still synthesis (planner_failed).
    # Confirm the held frames don't trigger env.step — same frame N times.
    held = frames["attempt_blocked"]
    assert all(np.array_equal(held[0], f) for f in held)


def test_pushcube_render_titles_contain_phase_label():
    from babysteps.render.pushcube import render_episode
    from babysteps.envs.pushcube_adapter import PushCubeAdapter
    env = _StubEnv()
    _, titles = render_episode(env, PushCubeAdapter(), seed=0, fps=4)
    assert "phase 1/3" in titles["demo"][0]
    assert "phase 2/3" in titles["attempt_blocked"][0]
    assert "phase 3/3" in titles["retry"][0]


# ---------- PickCube render tests -------------------------------------- #


def test_pickcube_render_episode_emits_three_phase_frames():
    from babysteps.render.pickcube import render_episode
    from babysteps.envs.pickcube_adapter import PickCubeAdapter

    env = _StubEnv()
    adapter = PickCubeAdapter()
    frames, titles = render_episode(env, adapter, seed=0, fps=4)

    assert set(frames.keys()) == {"demo", "attempt_blocked", "retry"}
    assert set(titles.keys()) == {"demo", "attempt_blocked", "retry"}
    # All three phases must step the env (grasp_slip is execution-time).
    assert len(frames["demo"]) >= 2
    assert len(frames["attempt_blocked"]) >= 2
    assert len(frames["retry"]) >= 2


def test_pickcube_render_phase2_actually_steps_env():
    """Unlike PushCube (held still), PickCube must step the env in phase 2
    so the grasp_slip is visible (gripper closes, lifts, releases). Detect
    by checking the stub env's step_count incremented across phase 2 frames."""
    from babysteps.render.pickcube import render_episode
    from babysteps.envs.pickcube_adapter import PickCubeAdapter

    env = _StubEnv()
    frames, _ = render_episode(env, PickCubeAdapter(), seed=0, fps=4)
    held = frames["attempt_blocked"]
    # In the stub env each step bumps the frame intensity, so consecutive
    # frames differ iff env.step was called. PushCube's phase-2 frames are
    # all identical; PickCube's must differ.
    assert not all(np.array_equal(held[0], f) for f in held), (
        "PickCube phase 2 should step the env to surface grasp_slip; "
        "saw all-identical frames (PushCube-style hold)."
    )


def test_pickcube_render_titles_mention_contact_region():
    from babysteps.render.pickcube import render_episode
    from babysteps.envs.pickcube_adapter import PickCubeAdapter
    _, titles = render_episode(_StubEnv(), PickCubeAdapter(), seed=0, fps=4)
    # Demo subtitle should mention contact_region (which face was grasped).
    assert "contact_region" in titles["demo"][1]
    # Retry subtitle should mention contact_substitution.
    assert "contact_substitution" in titles["retry"][1]
