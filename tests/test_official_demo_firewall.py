"""Firewall tests for the official-demo render path (Scope A).

`babysteps.render.official_demo` sources a demonstration from ManiSkill's
official Panda oracle — either run-live (run the solver, film it) or
state-replay (teleport through recorded scene poses). Both must yield ONLY
third-person RGB frames; the recorded Franka motor program (the `.h5`
`actions` channel) must never reach the intent path
(`babysteps.stage4.vision_features`).

These are static source-introspection + signature checks, mirroring
`tests/test_stage4_features.py` and `tests/test_stage5_render_demo_frames_smoke.py`.
The `official_demo` checks are pure-Python (no torch / no sim) and run on the
login node; the encoder-side checks `importorskip("torch")` since
`vision_features` imports torch unconditionally. No GPU, no Vulkan, no h5py.
"""
from __future__ import annotations

import inspect

import pytest


# Tokens that would indicate the recorded action/state channel (or the sim /
# planner) has leaked into a module that must stay pixels-only. Chosen to avoid
# innocent substrings: the encoder legitimately contains no "action"/"reward"
# (it has "extr-action"? no — verified absent), so we scan plurals/specifics.
_PRIVILEGED_TOKENS = (
    "qpos",
    "qvel",
    "actions",
    "set_state",
    "env_state",
    "tcp_pose",
    ".h5",
    "h5py",
    "pd_joint",
    "mplib",
)


def test_vision_encoder_source_has_no_privileged_tokens():
    """The encoder must not reference any action/state/sim/planner token."""
    from babysteps.stage4 import vision_features

    src = inspect.getsource(vision_features).lower()
    for forbidden in _PRIVILEGED_TOKENS:
        assert forbidden not in src, (
            f"firewall violation: {forbidden!r} in vision_features.py"
        )


def test_vision_encoder_signature_rejects_privileged_params():
    """`extract_vision_features` must accept only pixels + encoder knobs."""
    from babysteps.stage4.vision_features import extract_vision_features

    params = set(inspect.signature(extract_vision_features).parameters)
    assert params <= {
        "demo_frames",
        "encoder",
        "pool",
        "device",
        "resolution",
        "vjepa_n_frames",  # V-JEPA clip length — an encoder knob, not a data channel
        "vjepa_crop",      # V-JEPA crop size — an encoder knob, not a data channel
        "_encoder",
    }, f"encoder signature widened to accept extra params: {params}"
    privileged = {
        "actions",
        "qpos",
        "qvel",
        "env_states",
        "control",
        "obs",
        "rewards",
        "success",
        "initial_intent",
    }
    assert params.isdisjoint(privileged), (
        f"encoder signature exposes privileged params: {params & privileged}"
    )


def test_official_demo_safe_and_privileged_keys_are_disjoint():
    """The state-replay path reads only SAFE_STATE_KEY, not any privileged key."""
    from babysteps.render.official_demo import PRIVILEGED_H5_KEYS, SAFE_STATE_KEY

    assert SAFE_STATE_KEY == "env_states"
    assert "actions" in PRIVILEGED_H5_KEYS
    assert set(PRIVILEGED_H5_KEYS).isdisjoint({SAFE_STATE_KEY})


def test_official_demo_never_indexes_privileged_h5_keys():
    """The module must not bracket-index the .h5 by any privileged key.

    The state-replay path is only permitted to read `h5[tid]["env_states"]`.
    Reading `h5[...]["actions"]` (or rewards/outcome flags) would pull the
    recorded Franka motor program off disk — exactly the privileged channel
    the demo->intent firewall forbids.
    """
    from babysteps.render import official_demo

    src = inspect.getsource(official_demo)
    for key in official_demo.PRIVILEGED_H5_KEYS:
        for pattern in (f'["{key}"]', f"['{key}']"):
            assert pattern not in src, (
                f"firewall violation: official_demo bracket-indexes {pattern} "
                f"(the recorded {key!r} channel must never be read)"
            )


def test_official_demo_imports_sim_free():
    """Importing the module must not pull the sim/planner/h5py at module load.

    The renderer is GPU-only and run-live needs mplib, but the module must
    stay importable on the login node so this firewall suite can introspect
    it without a Vulkan device. All such imports must be lazy (inside the
    function bodies).
    """
    import importlib
    import sys

    # Drop any prior import so we observe this module's own load side effects.
    for name in ("babysteps.render.official_demo",):
        sys.modules.pop(name, None)
    importlib.import_module("babysteps.render.official_demo")

    for sim_mod in ("mani_skill", "sapien", "h5py", "mplib", "gymnasium"):
        assert sim_mod not in sys.modules, (
            f"official_demo imported {sim_mod!r} at module load; "
            f"the sim/planner/h5py import must stay lazy (inside the functions)."
        )


def test_official_demo_public_surface():
    """Pin the public functions and the control-mode the solvers require."""
    from babysteps.render import official_demo

    for fn in (
        "run_official_solver_frames",
        "replay_official_state_frames",
        "official_demo_frames",
        "resolve_official_traj",
    ):
        assert hasattr(official_demo, fn), f"missing public function {fn!r}"
    # Official cube solvers require pd_joint_pos (StackCube asserts it).
    assert official_demo.OFFICIAL_CONTROL_MODE == "pd_joint_pos"


def test_encoder_consumes_frames_only_not_h5_payload():
    """End-to-end firewall: a fake official .h5 payload carries action/qpos/qvel
    arrays alongside frames; only the frames may be handed to the encoder, and
    the encoder produces a feature vector from those pixels alone."""
    torch = pytest.importorskip("torch")
    import numpy as np

    from babysteps.stage4.vision_features import extract_vision_features

    class _FakeEncoder(torch.nn.Module):
        def __init__(self, d: int = 768):
            super().__init__()
            self.d = d

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            T = x.shape[0]
            base = torch.arange(self.d, dtype=torch.float32) / self.d
            per_t = x.mean(dim=(1, 2, 3)).unsqueeze(-1)
            return base.unsqueeze(0) * per_t

    # A stand-in for one official .h5 episode: privileged channels PLUS frames.
    fake_h5_episode = {
        "actions": np.zeros((5, 8), dtype=np.float32),
        "qpos": np.zeros((6, 9), dtype=np.float32),
        "qvel": np.zeros((6, 9), dtype=np.float32),
        "env_states": np.zeros((6, 13), dtype=np.float32),
        "frames": [128 * np.ones((64, 64, 3), dtype=np.uint8) for _ in range(5)],
    }

    # Only the rendered frames cross into the encoder — never the privileged keys.
    z = extract_vision_features(
        fake_h5_episode["frames"],
        device="cpu",
        _encoder=_FakeEncoder(d=768),
    )
    assert isinstance(z, np.ndarray)
    assert z.shape == (768,)
    assert z.dtype == np.float32
