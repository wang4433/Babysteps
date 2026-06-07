"""Tests for babysteps.envs.pokecube_adapter — the SECOND contact_region family
(build-order step 3).

PokeCube is the leave-one-task-family-out partner for PushCube: it must share
contact_region candidate SEMANTICS + revision RULE BYTE-FOR-BYTE (the poke-vs-
push difference is execution-only, in the GPU runner). These tests guard that
shared contract sim-free, so an accidental divergence is caught before it
silently invalidates the LOTO claim.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

from babysteps.envs.pokecube_adapter import PokeCubeAdapter
from babysteps.envs.pushcube_adapter import PushCubeAdapter
from babysteps.envs.task_adapter import BaseTaskAdapter
from babysteps.schemas import DemoEvidence, SceneState
from babysteps.stage5.revision_policy import candidates_for

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

PUSH = PushCubeAdapter()
POKE = PokeCubeAdapter()


def _scene(goal_xy):
    return SceneState(cube_xy=(0.0, 0.0), cube_z=0.02, goal_xy=goal_xy,
                      tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0),
                      blocked_sides=())


def test_pokecube_is_pushcube_subclass_with_own_task_id():
    assert issubclass(PokeCubeAdapter, PushCubeAdapter)
    assert issubclass(PokeCubeAdapter, BaseTaskAdapter)
    assert POKE.task_id == "PokeCube-v1"


def test_oracle_intent_identical_to_pushcube():
    # Same scene → identical intent (contact_region + all factors) for both
    # families; this is the shared-semantics contract.
    for goal in [(0.15, 0.0), (-0.15, 0.0), (0.0, 0.15), (0.0, -0.15),
                 (0.12, 0.05)]:
        sc = _scene(goal)
        assert (POKE.oracle_correct_intent(sc).to_dict()
                == PUSH.oracle_correct_intent(sc).to_dict())


def test_contact_region_vocab_byte_identical():
    assert (candidates_for("PokeCube-v1", "contact_region")
            == candidates_for("PushCube-v1", "contact_region"))
    assert (POKE.task_valid_tokens()["contact_region"]
            == PUSH.task_valid_tokens()["contact_region"])


def test_scripted_demo_to_intent_parity():
    ev = DemoEvidence(
        camera="external", demonstrator_type="scripted",
        object_trajectory=((0.0, 0.0), (0.15, 0.0)),
        contact_region_label="minus_x_face", final_state="cube_at_target",
        rgbd_video_path=None)
    assert POKE.scripted_demo_to_intent(ev).to_dict() == PUSH.scripted_demo_to_intent(ev).to_dict()


def test_unified_evaluator_routes_pokecube_to_fake_runner():
    # PokeCube is a Stage-5-only task (no Stage-0 render/collect scaffolding), so
    # it is NOT in the Stage-0 TASK_REGISTRY; the unified evaluator routes it
    # directly. This guards that routing + the shared push-physics fake.
    umt = importlib.import_module("stage5_unified_maintable_eval")
    adapter, runner = umt._make_runner_adapter("PokeCube-v1", fake=True)
    from tests.conftest import FakeEnvRunner, FakePokeEnvRunner
    assert isinstance(adapter, PokeCubeAdapter)
    assert isinstance(runner, FakePokeEnvRunner)
    assert isinstance(runner, FakeEnvRunner)  # shares push physics sim-free
