"""Pure collection planners for the Stage-4 varied-intent cut (sim-free).

Two strategies, matching the spec:
  * stratified_seed_plan  — Approach A (PushCube): assign each contiguous seed
    a target class so classes are perfectly balanced. The driver injects the
    cube pose to realise the assigned class.
  * select_balanced_seeds — Approach B (StackCube): given a stream of
    (seed, observed_class) from native resets, keep the first
    `episodes_per_class` seeds of each class (in seed order). Raises if any
    class cannot be filled.
"""
from __future__ import annotations

from typing import Iterable


def stratified_seed_plan(
    classes: tuple[str, ...], episodes_per_class: int, seed_start: int = 0,
) -> list[tuple[int, str]]:
    """Contiguous (seed, target_class) assignments, balanced by construction.

    Seeds run seed_start .. seed_start + len(classes)*episodes_per_class - 1.
    Class order is round-robin so an interrupted run stays roughly balanced."""
    plan: list[tuple[int, str]] = []
    seed = seed_start
    for _ in range(episodes_per_class):
        for cls in classes:
            plan.append((seed, cls))
            seed += 1
    return plan


def select_balanced_seeds(
    stream: Iterable[tuple[int, str]],
    classes: tuple[str, ...],
    episodes_per_class: int,
) -> list[int]:
    """Keep the first `episodes_per_class` seeds of each class, in seed order.

    `stream` is (seed, observed_class) from native resets. Returns the kept
    seeds sorted ascending. Raises ValueError if any class is underfilled when
    the stream is exhausted."""
    want = set(classes)
    kept: dict[str, list[int]] = {c: [] for c in classes}
    for seed, cls in stream:
        if cls in want and len(kept[cls]) < episodes_per_class:
            kept[cls].append(seed)
        if all(len(kept[c]) >= episodes_per_class for c in classes):
            break
    short = {c: len(v) for c, v in kept.items() if len(v) < episodes_per_class}
    if short:
        raise ValueError(
            f"select_balanced_seeds could not fill {short} "
            f"(wanted {episodes_per_class}/class)"
        )
    return sorted(s for v in kept.values() for s in v)
