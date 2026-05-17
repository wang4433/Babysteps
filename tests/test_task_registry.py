"""Tests for babysteps.envs.task_registry — Stage-0 task dispatch."""
from __future__ import annotations

import pytest

from babysteps.envs.task_registry import (
    TASK_REGISTRY,
    TaskEntry,
    get_task_entry,
)


def test_registry_contains_all_stage0_tasks():
    """PushCube-v1 (A), PickCube-v1 (B), and StackCube-v1 (C) must be present."""
    assert set(TASK_REGISTRY.keys()) == {
        "PushCube-v1", "PickCube-v1", "StackCube-v1",
    }


def test_get_task_entry_pushcube():
    from babysteps.envs.pushcube_adapter import PushCubeAdapter
    entry = get_task_entry("PushCube-v1")
    assert isinstance(entry, TaskEntry)
    assert entry.adapter_cls is PushCubeAdapter
    assert entry.episode_id_prefix == "pushcube_blocked_approach"


def test_get_task_entry_pickcube():
    from babysteps.envs.pickcube_adapter import PickCubeAdapter
    entry = get_task_entry("PickCube-v1")
    assert isinstance(entry, TaskEntry)
    assert entry.adapter_cls is PickCubeAdapter
    assert entry.episode_id_prefix == "pickcube_grasp_slip"


def test_get_task_entry_unknown_task_raises():
    with pytest.raises(KeyError) as exc:
        get_task_entry("OpenCabinetDrawer-v1")
    msg = str(exc.value)
    assert "OpenCabinetDrawer-v1" in msg
    assert "PushCube-v1" in msg
    assert "PickCube-v1" in msg
    assert "StackCube-v1" in msg


def test_fake_runner_factory_pushcube():
    """Factory returns an env_runner whose .run/.reset/.close work without
    a real simulator (i.e., one of the FakeEnvRunner classes)."""
    entry = get_task_entry("PushCube-v1")
    runner = entry.fake_runner_factory()
    assert hasattr(runner, "reset")
    assert hasattr(runner, "run")
    assert hasattr(runner, "close")
    runner.close()


def test_fake_runner_factory_pickcube():
    entry = get_task_entry("PickCube-v1")
    runner = entry.fake_runner_factory()
    assert hasattr(runner, "reset")
    assert hasattr(runner, "run")
    assert hasattr(runner, "close")
    runner.close()


def test_registry_entries_are_taskentry_instances():
    for task_id, entry in TASK_REGISTRY.items():
        assert isinstance(entry, TaskEntry), f"{task_id} entry is not TaskEntry"
        # adapter_cls's task_id must match the registry key.
        assert entry.adapter_cls.task_id == task_id


def test_task_registry_matches_render_registry():
    """Every task in TASK_REGISTRY must have a matching RENDER_REGISTRY entry.

    Without this guard, adding a Sub-project C task to TASK_REGISTRY but
    forgetting to add a babysteps/render/<task>.py module would mean
    `render_stage0_maniskill.py --task <new>` accepts the arg via argparse
    (it reads TASK_REGISTRY.keys()), then KeyErrors at runtime in
    get_render_fn. This test makes that a fast pytest failure instead."""
    from babysteps.render import RENDER_REGISTRY
    assert set(TASK_REGISTRY.keys()) == set(RENDER_REGISTRY.keys()), (
        f"TASK_REGISTRY tasks {sorted(TASK_REGISTRY.keys())} "
        f"!= RENDER_REGISTRY tasks {sorted(RENDER_REGISTRY.keys())}"
    )


def test_get_task_entry_stackcube():
    from babysteps.envs.stackcube_adapter import StackCubeAdapter
    entry = get_task_entry("StackCube-v1")
    assert isinstance(entry, TaskEntry)
    assert entry.adapter_cls is StackCubeAdapter
    assert entry.episode_id_prefix == "stackcube_underspec_goal"


def test_fake_runner_factory_stackcube():
    entry = get_task_entry("StackCube-v1")
    runner = entry.fake_runner_factory()
    assert hasattr(runner, "reset")
    assert hasattr(runner, "run")
    assert hasattr(runner, "close")
    runner.close()
