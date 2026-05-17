"""Per-task render dispatch for Stage-0 MP4 generation.

Each task contributes one `render_episode(env, adapter, seed, fps)`
function that returns three lists of RGB frames (one per BABYSTEPS
phase: demo, blocked attempt, retry) plus title metadata for the
on-frame banners. `scripts/render_stage0_maniskill.py` is a thin
dispatcher over RENDER_REGISTRY.

Render modules are sim-free in their tested surface (waypoint
selection + frame counting) and pull in mani_skill only when invoked
end-to-end."""
from __future__ import annotations

from typing import Callable


# A render_episode_fn returns:
#   ({phase_name → list[rgb_frame]}, {phase_name → (title, subtitle)})
# where phase_names are "demo", "attempt_blocked", "retry".
RenderEpisodeFn = Callable[..., tuple[dict, dict]]


def _pushcube_render() -> RenderEpisodeFn:
    from babysteps.render.pushcube import render_episode
    return render_episode


def _pickcube_render() -> RenderEpisodeFn:
    from babysteps.render.pickcube import render_episode
    return render_episode


# Lazy: each entry's import happens on first access via RENDER_REGISTRY[task_id]().
# This keeps importing babysteps.render cheap when only one task is needed.
RENDER_REGISTRY: dict[str, Callable[[], RenderEpisodeFn]] = {
    "PushCube-v1": _pushcube_render,
    "PickCube-v1": _pickcube_render,
}


def get_render_fn(task_id: str) -> RenderEpisodeFn:
    if task_id not in RENDER_REGISTRY:
        known = sorted(RENDER_REGISTRY.keys())
        raise KeyError(f"no render module for task {task_id!r}; known: {known}")
    return RENDER_REGISTRY[task_id]()
