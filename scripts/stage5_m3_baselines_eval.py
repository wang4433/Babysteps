"""Stage-5 M3 — procedural baselines main table eval.

Runs all seven procedural retry policies from ``babysteps.policies.POLICIES``
on PushCube-v1, PickCube-v1, and StackCube-v1 over a held-out seed range.
Produces the paper's main comparison table.

Each (task, policy, seed) triple calls ``episode.run_episode`` exactly once
through the task's real ManiSkill env_runner (or ``--fake`` for sim-free
smoke tests).

Example::

    python scripts/stage5_m3_baselines_eval.py \\
        --tasks PushCube-v1 PickCube-v1 StackCube-v1 \\
        --eval-seeds 100-149 \\
        --out-dir reports/stage5/m3_baselines

    # sim-free smoke:
    python scripts/stage5_m3_baselines_eval.py \\
        --tasks PushCube-v1 PickCube-v1 StackCube-v1 \\
        --eval-seeds 100-104 --fake \\
        --out-dir reports/stage5/m3_baselines_smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from babysteps.episode import run_episode  # noqa: E402
from babysteps.envs.task_registry import get_task_entry  # noqa: E402
from babysteps.policies import POLICIES  # noqa: E402


_SUPPORTED_TASKS = ("PushCube-v1", "PickCube-v1", "StackCube-v1")

_DEFAULT_POLICIES = (
    "one_shot",
    "same_intent_retry",
    "random_factor_revision",
    "babysteps_selective",
    "text_feedback_replan",
    "full_replan_analogue",
    "oracle_factor_revision",
)


def _parse_seed_range(s: str) -> list[int]:
    if "-" in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(s)]


def _make_adapter(task: str, *, fake: bool):
    entry = get_task_entry(task)
    adapter = entry.adapter_cls()
    if fake:
        adapter._env_runner_override = entry.fake_runner_factory()
        original_make = adapter.make_env_runner
        def _fake_make():
            return adapter._env_runner_override
        adapter.make_env_runner = _fake_make
    return adapter


def _eval_policy(adapter, policy_fn, *, prefix: str,
                 seeds: list[int]) -> dict:
    rows = []
    for seed in seeds:
        rec = run_episode(
            episode_id=f"m3_{prefix}_seed_{seed:04d}",
            seed=seed,
            adapter=adapter,
            policy=policy_fn,
        )
        initial_success = bool(rec.metrics.get("initial_success", False))
        retry_success = rec.metrics.get("retry_success")
        if retry_success is None:
            final_success = initial_success
        else:
            final_success = bool(retry_success)

        revision_dict = rec.revision
        revised_factor = None
        if revision_dict is not None:
            revised_factor = revision_dict.get("factor")

        rows.append({
            "seed": seed,
            "initial_success": initial_success,
            "retry_success": retry_success,
            "final_success": final_success,
            "revised_factor": revised_factor,
        })
    n = len(rows)
    return {
        "n": n,
        "initial_success_rate": sum(r["initial_success"] for r in rows) / n,
        "final_success_rate": sum(r["final_success"] for r in rows) / n,
        "rows": rows,
    }


def _write_main_table(out_dir: Path, all_results: dict) -> None:
    """Write the paper's main comparison table as Markdown + JSON."""
    tasks = sorted(all_results.keys())
    policies = list(_DEFAULT_POLICIES)

    lines = [
        "# Stage-5 M3 — Procedural Baselines Main Table",
        "",
        "| policy |",
    ]
    header_cols = " policy |"
    sep_cols = "---|"
    for task in tasks:
        short = task.replace("-v1", "")
        header_cols += f" {short} |"
        sep_cols += "---|"
    lines = [
        "# Stage-5 M3 — Procedural Baselines Main Table",
        "",
        f"| {header_cols}",
        f"| {sep_cols}",
    ]
    for pol in policies:
        row = f"| `{pol}` |"
        for task in tasks:
            res = all_results.get(task, {}).get(pol)
            if res is None:
                row += " - |"
            else:
                row += f" {res['final_success_rate']:.3f} |"
        lines.append(row)

    lines.extend(["", "## Delta-pp vs same_intent_retry", ""])
    for task in tasks:
        sir = all_results.get(task, {}).get("same_intent_retry", {}).get("final_success_rate", 0)
        lines.append(f"### {task}")
        lines.append("")
        lines.append("| policy | final | Δpp vs retry |")
        lines.append("|---|---|---|")
        for pol in policies:
            res = all_results.get(task, {}).get(pol)
            if res is None:
                continue
            fsr = res["final_success_rate"]
            delta = (fsr - sir) * 100
            lines.append(f"| `{pol}` | {fsr:.3f} | {delta:+.1f} |")
        lines.append("")

    (out_dir / "main_table.md").write_text("\n".join(lines) + "\n")

    summary = {}
    for task in tasks:
        summary[task] = {}
        for pol in policies:
            res = all_results.get(task, {}).get(pol)
            if res:
                summary[task][pol] = res["final_success_rate"]
    (out_dir / "main_table.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Stage-5 M3 procedural baselines eval.")
    p.add_argument("--tasks", nargs="+", default=list(_SUPPORTED_TASKS),
                   choices=_SUPPORTED_TASKS)
    p.add_argument("--policies", default=",".join(_DEFAULT_POLICIES),
                   help="Comma-separated policy names.")
    p.add_argument("--eval-seeds", required=True,
                   help="Seed range A-B (inclusive) or single int.")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--fake", action="store_true",
                   help="Use fake env_runner (sim-free smoke test).")
    args = p.parse_args(argv)

    seeds = _parse_seed_range(args.eval_seeds)
    policy_names = [n.strip() for n in args.policies.split(",") if n.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict[str, dict]] = {}

    for task in args.tasks:
        print(f"\n{'='*60}")
        print(f"  {task}  (seeds={args.eval_seeds}, fake={args.fake})")
        print(f"{'='*60}")

        task_results = {}
        for pol_name in policy_names:
            if pol_name not in POLICIES:
                print(f"  SKIP unknown policy: {pol_name}")
                continue

            adapter = _make_adapter(task, fake=args.fake)
            try:
                print(f"\n  --- {pol_name} ---")
                res = _eval_policy(
                    adapter, POLICIES[pol_name],
                    prefix=f"{task.replace('-', '_')}_{pol_name}",
                    seeds=seeds,
                )
                print(f"  initial: {res['initial_success_rate']:.3f}  "
                      f"final: {res['final_success_rate']:.3f}")
                task_results[pol_name] = res
            finally:
                if hasattr(adapter, "close"):
                    adapter.close()

        all_results[task] = task_results

        task_dir = args.out_dir / task
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "results.json").write_text(
            json.dumps({
                "task": task, "fake": args.fake,
                "eval_seeds": seeds,
                "per_policy": task_results,
            }, indent=2, sort_keys=True) + "\n"
        )

    _write_main_table(args.out_dir, all_results)
    print(f"\nwrote {args.out_dir}/main_table.{{md,json}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
