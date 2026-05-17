"""Tests for babysteps.envs.task_registry — Stage-0 task dispatch."""
from __future__ import annotations

import pytest

from babysteps.envs.task_registry import (
    TASK_REGISTRY,
    TaskEntry,
    get_task_entry,
)


def test_registry_contains_both_stage0_tasks():
    """PushCube-v1 (sub-project A) and PickCube-v1 (sub-project B) must be present."""
    assert set(TASK_REGISTRY.keys()) == {"PushCube-v1", "PickCube-v1"}


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
        get_task_entry("StackCube-v1")
    msg = str(exc.value)
    assert "StackCube-v1" in msg
    assert "PushCube-v1" in msg
    assert "PickCube-v1" in msg


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
