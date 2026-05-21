#!/usr/bin/env python
"""Run the Stage-0 procedural baseline sweep: methods × tasks × seeds.

Sim-free CI uses --fake-env. The real sweep runs on GPU via the task registry.
See docs/superpowers/specs/2026-05-20-stage0-baselines-design.md.

Adapter construction mirrors scripts/stage0_collect.py:
  - get_task_entry(task_id) returns a TaskEntry with adapter_cls and
    fake_runner_factory.
  - For --fake-env: subclass adapter_cls, override make_env_runner() to return
    the fake_runner_factory() result.
  - For the real env: instantiate adapter_cls() directly (needs Vulkan).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the project root importable without `pip install -e .`.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from babysteps.envs.task_adapter import BaseTaskAdapter  # noqa: E402
from babysteps.envs.task_registry import (  # noqa: E402
    TASK_REGISTRY,
    get_task_entry,
)
from babysteps.episode import run_episode  # noqa: E402
from babysteps.eval import (  # noqa: E402
    compute_comparison_table,
    compute_metrics,
    write_comparison_table,
)
from babysteps.policies import POLICIES  # noqa: E402

MAIN_TABLE_METHODS = [
    "one_shot", "same_intent_retry", "random_factor_revision",
    "babysteps_selective", "text_feedback_replan", "full_replan_analogue",
    "oracle_factor_revision",
]


def _make_adapter(task_id: str, use_fake: bool) -> BaseTaskAdapter:
    """Build the right adapter for `task_id`, wired to fake or real runner.

    Mirrors stage0_collect.py exactly: get_task_entry dispatches to the
    registered adapter class. For --fake-env, we dynamically subclass
    adapter_cls and override make_env_runner() to return the fake runner
    from fake_runner_factory(). For the real env, adapter_cls() is used
    directly (requires Vulkan/GPU).
    """
    entry = get_task_entry(task_id)
    if use_fake:
        fake = entry.fake_runner_factory()

        class _FakeAdapter(entry.adapter_cls):  # type: ignore[misc, valid-type]
            def make_env_runner(self):
                return fake

        return _FakeAdapter()
    return entry.adapter_cls()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--tasks", nargs="+", required=True,
        choices=sorted(TASK_REGISTRY.keys()),
        help="Which Stage-0 tasks to sweep over.",
    )
    ap.add_argument(
        "--methods", nargs="+", default=["all"],
        help='Which policies to evaluate; use "all" to run the full 7-method table.',
    )
    ap.add_argument("--n_episodes", type=int, default=20)
    ap.add_argument("--seed_start", type=int, default=0)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument(
        "--fake-env", action="store_true",
        help="Use the deterministic sim-free fake env_runner from "
             "tests/conftest. Useful for CI on a login node without Vulkan.",
    )
    args = ap.parse_args()

    methods = MAIN_TABLE_METHODS if args.methods == ["all"] else args.methods
    args.out_dir.mkdir(parents=True, exist_ok=True)

    by: dict[str, dict[str, dict]] = {m: {} for m in methods}
    for task in args.tasks:
        for method in methods:
            adapter = _make_adapter(task, args.fake_env)
            records = []
            try:
                for i in range(args.n_episodes):
                    seed = args.seed_start + i
                    records.append(run_episode(
                        episode_id=f"{task}_{method}_seed_{seed:04d}",
                        seed=seed,
                        adapter=adapter,
                        policy=POLICIES[method],
                        record_baseline_metrics=True,
                    ))
            finally:
                adapter.close()
            metrics = compute_metrics(records)
            run_dir = args.out_dir / method / task
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "samples.jsonl").write_text(
                "\n".join(r.to_jsonl_line() for r in records) + "\n")
            (run_dir / "report.json").write_text(
                json.dumps(metrics, indent=2, sort_keys=True) + "\n")
            by[method][task] = metrics
            print(
                f"[{method}/{task}] n={args.n_episodes} "
                f"initial_rate={metrics['initial_attempt_success_rate']:.2f} "
                f"retry_rate={metrics['retry_success_rate']:.2f} "
                f"delta_pp={metrics['delta_pp']:.1f}",
                flush=True,
            )

    table = compute_comparison_table(by, methods=methods, tasks=list(args.tasks))
    write_comparison_table(table, args.out_dir / "comparison_table.md")
    (args.out_dir / "comparison_table.json").write_text(
        json.dumps(table, indent=2, sort_keys=True) + "\n")
    print(f"wrote comparison table to {args.out_dir}/comparison_table.md")


if __name__ == "__main__":
    main()
