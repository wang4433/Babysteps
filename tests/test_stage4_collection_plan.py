"""Pure collection planners (sim-free)."""
import pytest

from babysteps.stage4.collection_plan import (
    select_balanced_seeds,
    stratified_seed_plan,
)

_DIRS = ("translate_+x", "translate_-x", "translate_+y", "translate_-y")


def test_stratified_plan_is_balanced_and_deterministic():
    plan = stratified_seed_plan(_DIRS, episodes_per_class=10, seed_start=0)
    assert len(plan) == 40
    counts = {c: 0 for c in _DIRS}
    for _seed, cls in plan:
        counts[cls] += 1
    assert all(v == 10 for v in counts.values())
    seeds = [s for s, _ in plan]
    assert seeds == list(range(0, 40))
    assert stratified_seed_plan(_DIRS, 10, 0) == plan


def test_select_balanced_keeps_quota_per_class_in_seed_order():
    stream = []
    for s in range(100):
        cls = _DIRS[s % 4] if s < 80 else "translate_+x"
        stream.append((s, cls))
    kept = select_balanced_seeds(stream, _DIRS, episodes_per_class=10)
    assert len(kept) == 40
    by_cls = {c: 0 for c in _DIRS}
    for s in kept:
        cls = dict(stream)[s]
        by_cls[cls] += 1
    assert all(v == 10 for v in by_cls.values())
    assert kept == sorted(kept)


def test_select_balanced_raises_when_a_class_cannot_be_filled():
    stream = [(s, "translate_+x") for s in range(50)]
    with pytest.raises(ValueError, match="could not fill"):
        select_balanced_seeds(stream, _DIRS, episodes_per_class=10)
