# scripts/stage4_collect_varied.py
"""Stage-4 varied-intent collection driver (GPU).

PushCube (Approach A): stratified seed→motion plan; the runner injects the
cube pose per seed so object_motion is balanced by construction.
StackCube (Approach B): native resets binned by cubeA→cubeB direction; keep a
balanced seed subset (rejection sampling), then run those episodes.

Writes one EpisodeRecord per line to <out_dir>/<task>/samples.jsonl plus
report.{json,md} (via babysteps.eval), mirroring scripts/stage0_collect.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from babysteps.envs.scene import cubeA_to_cubeB_motion  # noqa: E402
from babysteps.envs.task_registry import get_task_entry  # noqa: E402
from babysteps.episode import run_episode  # noqa: E402
from babysteps.eval import compute_metrics, write_report  # noqa: E402
from babysteps.stage4.collection_plan import (  # noqa: E402
    select_balanced_seeds,
    stratified_seed_plan,
)

_DIRS = ("translate_+x", "translate_-x", "translate_+y", "translate_-y")


def _collect_pushcube(out_dir: Path, per_class: int, seed_start: int) -> int:
    entry = get_task_entry("PushCube-v1")
    adapter = entry.adapter_cls()
    runner = adapter.env_runner()
    plan = stratified_seed_plan(_DIRS, per_class, seed_start)
    records = []
    try:
        for seed, motion in plan:
            runner.set_injection(motion)
            rec = run_episode(
                episode_id=f"pushcube_varied_seed_{seed:04d}",
                seed=seed, adapter=adapter,
            )
            records.append(rec)
    finally:
        adapter.close()
    return _write(out_dir / "PushCube-v1", records)


def _collect_stackcube(
    out_dir: Path, per_class: int, seed_start: int, max_scan: int,
) -> int:
    entry = get_task_entry("StackCube-v1")
    adapter = entry.adapter_cls()
    records = []
    # Single try/finally over BOTH passes: select_balanced_seeds raises
    # ValueError on an underfillable bin (a designed failure path), and a
    # native reset can fail mid-scan — either way the GPU env must be released.
    try:
        runner = adapter.env_runner()  # cached; run_episode reuses this instance
        # Pass 1: bin native resets by cubeA→cubeB direction.
        stream = []
        for seed in range(seed_start, seed_start + max_scan):
            scene = runner.reset(seed)
            cubeB_xy = scene.extra["cubeB_xy"]
            stream.append((seed, cubeA_to_cubeB_motion(scene.cube_xy, cubeB_xy)))
        kept = select_balanced_seeds(stream, _DIRS, per_class)
        # Pass 2: full episodes on the kept seeds.
        for seed in kept:
            rec = run_episode(
                episode_id=f"stackcube_varied_seed_{seed:04d}",
                seed=seed, adapter=adapter,
            )
            records.append(rec)
    finally:
        adapter.close()
    return _write(out_dir / "StackCube-v1", records)


def _write(task_dir: Path, records) -> int:
    task_dir.mkdir(parents=True, exist_ok=True)
    with (task_dir / "samples.jsonl").open("w") as f:
        for rec in records:
            f.write(rec.to_jsonl_line() + "\n")
    metrics = compute_metrics(records)
    write_report(metrics, task_dir)
    print(f"wrote {task_dir}/samples.jsonl ({len(records)} episodes)")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=["PushCube-v1", "StackCube-v1"], required=True)
    p.add_argument("--out_dir", type=Path,
                   default=_ROOT / "datasets/stage4/varied_intent")
    p.add_argument("--per_class", type=int, default=10)
    p.add_argument("--seed_start", type=int, default=0)
    p.add_argument("--max_scan", type=int, default=400,
                   help="StackCube only: max native seeds to scan for binning.")
    args = p.parse_args(argv)
    if args.task == "PushCube-v1":
        return _collect_pushcube(args.out_dir, args.per_class, args.seed_start)
    return _collect_stackcube(
        args.out_dir, args.per_class, args.seed_start, args.max_scan)


if __name__ == "__main__":
    sys.exit(main())
