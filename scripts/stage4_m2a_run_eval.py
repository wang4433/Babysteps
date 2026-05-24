"""Stage-4 M2a A3 — paired eval: latent_revision vs babysteps_selective.

For a held-out seed range (disjoint from the training cut), runs
`episode.run_episode` once per (seed, policy) pair through the task's
real env_runner (or a fake one for sim-free smoke). Compares per-policy
success rates and reports Δpp (latent − babysteps_selective).

This is the smallest meaningful M2a end-to-end number: does the latent
revision loop close on the task? G4 (`Δpp(latent vs failure-agnostic
retry) ≥ 10`) and G5 (`Δpp(latent vs oracle) ≤ 5pp`) are computed
from these per-policy numbers when run against the real env on enough
seeds.

Sim-free smoke (FakeEnvRunner): `--fake` flag.
Real eval: run from a GPU node; the adapter dispatches to ManiSkill.

Example::

    # Sim-free smoke (FakeEnvRunner via the test conftest)
    python scripts/stage4_m2a_run_eval.py \\
        --task PushCube-v1 \\
        --pack-dir models/stage4/m2a/PushCube-v1 \\
        --out-dir reports/stage4/m2a_a3_smoke \\
        --eval-seeds 100-119 --fake

    # Real eval (on a GPU node, ManiSkill via the adapter)
    python scripts/stage4_m2a_run_eval.py \\
        --task PushCube-v1 \\
        --pack-dir models/stage4/m2a/PushCube-v1 \\
        --out-dir reports/stage4/m2a_a3_pushcube \\
        --eval-seeds 100-149
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
from babysteps.policies import (  # noqa: E402
    babysteps_selective, oracle_factor_revision, same_intent_retry,
)
from babysteps.stage4.latent_policy import (  # noqa: E402
    latent_revision_factory, load_latent_pack,
)


_TASK_TO_ADAPTER_MODULE = {
    "PushCube-v1": ("babysteps.envs.pushcube_adapter", "PushCubeAdapter"),
    "StackCube-v1": ("babysteps.envs.stackcube_adapter", "StackCubeAdapter"),
}


def _parse_seed_range(s: str) -> list[int]:
    if "-" in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(s)]


def _make_real_adapter(task: str):
    module_name, cls_name = _TASK_TO_ADAPTER_MODULE[task]
    import importlib
    cls = getattr(importlib.import_module(module_name), cls_name)
    return cls()


def _make_fake_adapter(task: str):
    """Stub adapter that uses the test conftest FakeEnvRunner.

    Mirrors the pattern in tests/test_episode.py::_make_stub_adapter,
    but here we wrap a real task adapter so .scripted_demo_to_intent
    etc. fire correctly.
    """
    from tests.conftest import (
        FakeEnvRunner, FakePickEnvRunner, FakeStackCubeEnvRunner,
    )
    fakes = {
        "PushCube-v1": FakeEnvRunner,
        "PickCube-v1": FakePickEnvRunner,
        "StackCube-v1": FakeStackCubeEnvRunner,
    }
    base = _make_real_adapter(task)
    fake_runner = fakes[task]()
    class _StubAdapter(base.__class__):
        def make_env_runner(self):
            return fake_runner
    return _StubAdapter()


def _eval_policy(adapter, policy, *, episode_prefix: str,
                 seeds: list[int]) -> dict:
    """Returns per-seed list of (success, success_at_initial, success_at_retry)."""
    rows = []
    for seed in seeds:
        rec = run_episode(
            episode_id=f"{episode_prefix}_seed_{seed:04d}",
            seed=seed,
            adapter=adapter,
            policy=policy,
        )
        initial_success = bool(rec.metrics.get("initial_success", False))
        retry_success = rec.metrics.get("retry_success")
        # Final success = retry if retry happened, else initial
        if retry_success is None:
            final_success = initial_success
        else:
            final_success = bool(retry_success)
        rows.append({
            "seed": seed,
            "initial_success": initial_success,
            "retry_success": retry_success,
            "final_success": final_success,
        })
    n = len(rows)
    return {
        "n": n,
        "initial_success_rate": sum(r["initial_success"] for r in rows) / n,
        "final_success_rate": sum(r["final_success"] for r in rows) / n,
        "rows": rows,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Stage-4 M2a A3 paired eval.")
    p.add_argument("--task", required=True,
                   choices=sorted(_TASK_TO_ADAPTER_MODULE))
    p.add_argument("--pack-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--eval-seeds", required=True,
                   help="Seed range A-B (inclusive) or single int.")
    p.add_argument("--fake", action="store_true",
                   help="Use FakeEnvRunner via tests.conftest (sim-free smoke).")
    p.add_argument("--policies", default="latent,babysteps_selective,same_intent_retry,oracle_factor_revision",
                   help="Comma-separated list to evaluate.")
    args = p.parse_args(argv)

    seeds = _parse_seed_range(args.eval_seeds)
    pack = load_latent_pack(args.pack_dir)
    print(f"loaded LatentPack from {args.pack_dir}; "
          f"centroids for factors {sorted(pack.centroids.keys())}")

    policy_factory = {
        "latent": lambda: latent_revision_factory(pack),
        "babysteps_selective": lambda: babysteps_selective,
        "same_intent_retry": lambda: same_intent_retry,
        "oracle_factor_revision": lambda: oracle_factor_revision,
    }
    policy_names = [n.strip() for n in args.policies.split(",") if n.strip()]

    results = {}
    for name in policy_names:
        print(f"\n=== {name} (n_seeds={len(seeds)}, fake={args.fake}) ===")
        adapter = _make_fake_adapter(args.task) if args.fake else _make_real_adapter(args.task)
        try:
            res = _eval_policy(
                adapter, policy_factory[name](),
                episode_prefix=f"m2a_a3_{args.task.replace('-', '_')}_{name}",
                seeds=seeds,
            )
            print(f"  initial success: {res['initial_success_rate']:.3f}")
            print(f"  final success:   {res['final_success_rate']:.3f}")
            results[name] = res
        finally:
            if hasattr(adapter, "close"):
                adapter.close()

    # Δpp (final success) vs babysteps_selective and same_intent_retry
    summary = {}
    base_bs = results.get("babysteps_selective", {}).get("final_success_rate")
    base_sir = results.get("same_intent_retry", {}).get("final_success_rate")
    base_oracle = results.get("oracle_factor_revision", {}).get("final_success_rate")
    if "latent" in results:
        lat = results["latent"]["final_success_rate"]
        summary["latent_final_success_rate"] = lat
        if base_sir is not None:
            summary["delta_pp_vs_same_intent_retry"] = (lat - base_sir) * 100  # G4
        if base_bs is not None:
            summary["delta_pp_vs_babysteps_selective"] = (lat - base_bs) * 100
        if base_oracle is not None:
            summary["delta_pp_vs_oracle"] = (lat - base_oracle) * 100  # G5

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "a3_results.json").write_text(json.dumps({
        "config": {
            "task": args.task, "fake": args.fake,
            "eval_seeds": [int(s) for s in seeds],
            "pack_dir": str(args.pack_dir),
        },
        "per_policy": results,
        "summary": summary,
    }, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {args.out_dir}/a3_results.json")
    print(f"summary: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
