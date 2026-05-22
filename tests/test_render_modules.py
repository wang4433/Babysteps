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
        # Third-person external camera: an 8x8x3 frame (the demo view).
        return (np.ones((8, 8, 3), dtype=np.uint8) * (self._step_count % 256))

    # --- first-person panda_wristcam capture path --- #
    @property
    def unwrapped(self):
        return self

    def get_sensor_images(self):
        # First-person wrist camera: a distinctly-shaped 4x4x3 frame, batched
        # (B,H,W,3) like ManiSkill, so tests can tell the wrist view apart from
        # the 8x8 third-person render() view purely by frame shape.
        rgb = np.ones((1, 4, 4, 3), dtype=np.uint8) * 50
        return {"hand_camera": {"rgb": rgb}}

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


def test_pushcube_render_demo_thirdperson_exec_wristcam():
    """Stage-0 camera split: the demo phase is captured from the third-person
    external camera (8x8 render() frames); the execution phases (blocked
    attempt + retry) are captured from the first-person panda_wristcam
    hand_camera (4x4 get_sensor_images() frames)."""
    from babysteps.render.pushcube import render_episode
    from babysteps.envs.pushcube_adapter import PushCubeAdapter

    frames, _ = render_episode(_StubEnv(), PushCubeAdapter(), seed=0, fps=4)
    # Demo: third-person external camera (8x8).
    assert all(f.shape == (8, 8, 3) for f in frames["demo"])
    # Execution phases: first-person wrist camera (4x4).
    assert all(f.shape == (4, 4, 3) for f in frames["attempt_blocked"])
    assert all(f.shape == (4, 4, 3) for f in frames["retry"])


def test_pushcube_render_titles_contain_phase_label():
    from babysteps.render.pushcube import render_episode
    from babysteps.envs.pushcube_adapter import PushCubeAdapter
    env = _StubEnv()
    _, titles = render_episode(env, PushCubeAdapter(), seed=0, fps=4)
    assert "phase 1/3" in titles["demo"][0]
    assert "phase 2/3" in titles["attempt_blocked"][0]
    assert "phase 3/3" in titles["retry"][0]


def test_pushcube_render_retry_subtitle_shows_frozen_factors():
    """The retry subtitle must name the frozen (preserved) factors so the
    single-factor revision invariant is visible in the rendered caption.
    For PushCube the revised factor is approach_direction, so frozen_factors
    includes goal_state and object_motion (all INTENT_FIELDS minus approach_direction)."""
    from babysteps.render.pushcube import render_episode
    from babysteps.envs.pushcube_adapter import PushCubeAdapter

    env = _StubEnv()
    _, titles = render_episode(env, PushCubeAdapter(), seed=0, fps=4)
    _title, subtitle = titles["retry"]
    assert "frozen" in subtitle.lower()
    assert "goal_state" in subtitle
    assert "object_motion" in subtitle


def test_pushcube_baseline_contrast_emits_four_phases():
    """The baseline-contrast render shares demo + blocked attempt, then emits
    two retries: babysteps_selective (preserves contact_region) and
    full_replan_analogue (perturbs contact_region → wrong-way push)."""
    from babysteps.render.pushcube import render_baseline_contrast
    from babysteps.envs.pushcube_adapter import PushCubeAdapter

    frames, titles = render_baseline_contrast(_StubEnv(), PushCubeAdapter(), seed=0, fps=4)
    keys = {"demo", "attempt_blocked", "retry_selective", "retry_full_replan"}
    assert set(frames.keys()) == keys
    assert set(titles.keys()) == keys
    for k in keys:
        assert len(frames[k]) >= 1
    # Selective revises approach only; full_replan also perturbs contact_region.
    assert "approach_substitution" in titles["retry_selective"][1]
    assert "contact_region" in titles["retry_full_replan"][1]
    assert "full_replan" in titles["retry_full_replan"][0]


def test_pushcube_baseline_contrast_perturbs_contact_region():
    """The full_replan retry's contact_region must differ from the selective
    retry's (the collateral edit that breaks the push). Asserted on the token
    values encoded in the subtitles."""
    from babysteps.render.pushcube import render_baseline_contrast
    from babysteps.envs.pushcube_adapter import PushCubeAdapter

    _, titles = render_baseline_contrast(_StubEnv(), PushCubeAdapter(), seed=0, fps=4)
    # full_replan subtitle encodes "contact_region: <old> -> <new>"; old != new.
    sub = titles["retry_full_replan"][1]
    assert "->" in sub or "→" in sub


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


# ---------- StackCube render tests ------------------------------------ #


class _StubStackEnv:
    """Stand-in for gym.make('StackCube-v1') used in stackcube render tests.

    Like _StubEnv but the obs has cubeA_pose + cubeB_pose (no goal_pos).
    The TCP integrates the action's xyz so phase transitions happen in
    ~10 stub steps."""

    def __init__(self) -> None:
        self.tcp = np.array([0.0, 0.0, 0.25, 0.0, 0.0, 0.0], dtype=np.float64)
        self.cubeA = np.array([0.0, 0.0], dtype=np.float64)
        self.cubeB = np.array([0.10, 0.0], dtype=np.float64)
        self._step_count = 0

    def reset(self, seed: int = 0):
        self.tcp = np.array([0.0, 0.0, 0.25, 0.0, 0.0, 0.0], dtype=np.float64)
        self.cubeA = np.array([0.0, 0.0], dtype=np.float64)
        self.cubeB = np.array([0.10, 0.0], dtype=np.float64)
        self._step_count = 0
        return _StubStackObs(self.tcp, self.cubeA, self.cubeB), {}

    def step(self, action):
        self.tcp[0:3] = self.tcp[0:3] + 0.02 * np.asarray(action[0:3])
        self._step_count += 1
        return (
            _StubStackObs(self.tcp, self.cubeA, self.cubeB),
            0.0, False, False,
            {"success": False},
        )

    def render(self):
        return (np.ones((8, 8, 3), dtype=np.uint8) * (self._step_count % 256))

    def close(self):
        pass


from dataclasses import dataclass as _dc


@_dc
class _StubStackObs:
    tcp: np.ndarray
    cubeA: np.ndarray
    cubeB: np.ndarray

    def __getitem__(self, key: str):
        if key == "extra":
            tcp_raw = np.concatenate([self.tcp[0:3], np.array([1.0]),
                                      self.tcp[3:6]])
            cubeA_full = np.array([self.cubeA[0], self.cubeA[1], 0.02])
            cubeB_full = np.array([self.cubeB[0], self.cubeB[1], 0.02])
            return {
                "tcp_pose": tcp_raw,
                "cubeA_pose": cubeA_full,
                "cubeB_pose": cubeB_full,
            }
        raise KeyError(key)


def test_stackcube_render_episode_emits_three_phase_frames():
    from babysteps.render.stackcube import render_episode
    from babysteps.envs.stackcube_adapter import StackCubeAdapter

    env = _StubStackEnv()
    adapter = StackCubeAdapter()
    frames, titles = render_episode(env, adapter, seed=0, fps=4)

    assert set(frames.keys()) == {"demo", "attempt_blocked", "retry"}
    assert set(titles.keys()) == {"demo", "attempt_blocked", "retry"}
    assert len(frames["demo"]) >= 2
    assert len(frames["attempt_blocked"]) >= 2
    assert len(frames["retry"]) >= 2


def test_stackcube_render_phase2_actually_steps_env():
    """Like PickCube and unlike PushCube — phase 2 steps the env so the
    failed translate-and-drop is visible. Detect by checking the stub
    env's step_count varies the frame intensity."""
    from babysteps.render.stackcube import render_episode
    from babysteps.envs.stackcube_adapter import StackCubeAdapter

    env = _StubStackEnv()
    frames, _ = render_episode(env, StackCubeAdapter(), seed=0, fps=4)
    held = frames["attempt_blocked"]
    assert not all(np.array_equal(held[0], f) for f in held), (
        "StackCube phase 2 should step the env to surface the failed "
        "translate-and-drop; saw all-identical frames."
    )


def test_stackcube_render_titles_mention_goal_state():
    from babysteps.render.stackcube import render_episode
    from babysteps.envs.stackcube_adapter import StackCubeAdapter
    _, titles = render_episode(_StubStackEnv(), StackCubeAdapter(), seed=0, fps=4)
    # Demo subtitle should mention goal_state="cubeA_on_cubeB" (the oracle).
    assert "cubeA_on_cubeB" in titles["demo"][1]
    # Retry subtitle should mention goal_refinement.
    assert "goal_refinement" in titles["retry"][1]


# ---------- TurnFaucet render tests ---------------------------------- #


class _StubJoint:
    """Minimal stub for SAPIEN's articulation joint, supporting qpos read/write."""

    def __init__(self, initial_qpos: float = 0.0) -> None:
        # Store as a 2-D numpy array so to_np(joint.qpos) returns a scalar
        # after the arr.ndim==2 branch (arr[0] → 1-D, then .item() works).
        self._qpos = np.array([[initial_qpos]], dtype=np.float64)

    @property
    def qpos(self):
        return self._qpos

    @qpos.setter
    def qpos(self, value):
        # Accept torch tensors (as written by _set_faucet_qpos) or numpy arrays.
        if hasattr(value, "cpu"):
            value = value.cpu().numpy()
        self._qpos = np.asarray(value, dtype=np.float64)
        if self._qpos.ndim == 1:
            self._qpos = self._qpos[np.newaxis, :]

    @property
    def device(self):
        return "cpu"


class _StubSwitchLink:
    """Stub for env.unwrapped.target_switch_link."""

    def __init__(self, initial_qpos: float = 0.0) -> None:
        self.joint = _StubJoint(initial_qpos)


class _StubTurnEnvUnwrapped:
    """Stub for env.unwrapped — carries target_switch_link and target_angle."""

    def __init__(self, switch_link: _StubSwitchLink, target_angle: float) -> None:
        self.target_switch_link = switch_link
        self.target_angle = np.array(target_angle, dtype=np.float64)


class _StubTurnEnv:
    """Stand-in for gym.make('TurnFaucet-v1').

    Obs has tcp_pose, target_link_pos (handle xyz), and
    target_joint_axis (3D). Also exposes env.unwrapped with
    target_switch_link.joint.qpos and target_angle for the
    privileged-demo phase in the new render_episode."""

    def __init__(self) -> None:
        self.tcp = np.array([0.0, 0.0, 0.25, 0.0, 0.0, 0.0], dtype=np.float64)
        self.handle = np.array([0.10, 0.0, 0.10], dtype=np.float64)
        self.axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        self._step_count = 0
        # Privileged state for demo-phase: handle rotates from 0 to π/4.
        self._switch_link = _StubSwitchLink(initial_qpos=0.0)
        self.unwrapped = _StubTurnEnvUnwrapped(
            switch_link=self._switch_link,
            target_angle=np.pi / 4,
        )

    def reset(self, seed: int = 0):
        self.tcp = np.array([0.0, 0.0, 0.25, 0.0, 0.0, 0.0], dtype=np.float64)
        self.handle = np.array([0.10, 0.0, 0.10], dtype=np.float64)
        self.axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        self._step_count = 0
        # Reset qpos to initial each time (demo teleport is discarded on reset).
        self._switch_link.joint.qpos = np.array([[0.0]])
        return _StubTurnObs(self.tcp, self.handle, self.axis), {}

    def step(self, action):
        self.tcp[0:3] = self.tcp[0:3] + 0.02 * np.asarray(action[0:3])
        self._step_count += 1
        return (
            _StubTurnObs(self.tcp, self.handle, self.axis),
            0.0, False, False,
            {"success": False},
        )

    def render(self):
        return (np.ones((8, 8, 3), dtype=np.uint8) * (self._step_count % 256))

    def close(self):
        pass


@_dc
class _StubTurnObs:
    tcp: np.ndarray
    handle: np.ndarray
    axis: np.ndarray

    def __getitem__(self, key: str):
        if key == "extra":
            tcp_raw = np.concatenate([self.tcp[0:3], np.array([1.0]),
                                      self.tcp[3:6]])
            return {
                "tcp_pose": tcp_raw,
                "target_link_pos": self.handle,
                "target_joint_axis": self.axis,
            }
        raise KeyError(key)


def test_turnfaucet_render_episode_emits_three_phase_frames():
    from babysteps.render.turnfaucet import render_episode
    from babysteps.envs.turnfaucet_adapter import TurnFaucetAdapter

    env = _StubTurnEnv()
    adapter = TurnFaucetAdapter()
    frames, titles = render_episode(env, adapter, seed=0, fps=4)

    assert set(frames.keys()) == {"demo", "attempt_blocked", "retry"}
    assert set(titles.keys()) == {"demo", "attempt_blocked", "retry"}
    assert len(frames["demo"]) >= 2
    assert len(frames["attempt_blocked"]) >= 2
    assert len(frames["retry"]) >= 2


def test_turnfaucet_render_phase2_actually_steps_env():
    """Phase 2 = grasp_turn attempt that steps the env (jaws can't close on
    the thick handle, but the control loop still steps). Detect by checking
    that consecutive attempt_blocked frames differ (stub env bumps intensity
    each step, so identical frames => env was not stepped)."""
    from babysteps.render.turnfaucet import render_episode
    from babysteps.envs.turnfaucet_adapter import TurnFaucetAdapter

    env = _StubTurnEnv()
    frames, _ = render_episode(env, TurnFaucetAdapter(), seed=0, fps=4)
    held = frames["attempt_blocked"]
    assert not all(np.array_equal(held[0], f) for f in held), (
        "TurnFaucet phase 2 (grasp_turn attempt) should step the env; "
        "saw all-identical frames."
    )


def test_turnfaucet_render_titles_mention_embodiment_substitution_and_grasp_infeasible():
    """Phase 2 title must mention 'grasp_infeasible'; phase 3 title must
    mention 'embodiment_substitution' (new D-embodiment contract per spec §10)."""
    from babysteps.render.turnfaucet import render_episode
    from babysteps.envs.turnfaucet_adapter import TurnFaucetAdapter
    _, titles = render_episode(_StubTurnEnv(), TurnFaucetAdapter(), seed=0, fps=4)
    # Phase 2: grasp_infeasible is the named failure predicate.
    assert "grasp_infeasible" in titles["attempt_blocked"][0]
    # Phase 3: embodiment_substitution is the revision operator.
    assert "embodiment_substitution" in titles["retry"][1]
    # The embodiment tokens (grasp_turn → poke_turn) must appear in phase 3.
    assert "grasp_turn" in titles["retry"][1]
    assert "poke_turn" in titles["retry"][1]
