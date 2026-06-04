"""Stage-5 P1 — paired eval: vision-grounded latent_revision vs baselines.

Stage-5 analogue of scripts/stage4_m2a_run_eval.py. For a held-out seed
range (disjoint from the training cut), runs `episode.run_episode` once
per (seed, policy) pair through the task's real env_runner. Unlike M2a,
the latent policy reads Z from a pre-cached spatial_mean DINOv2 feature
file (Stage-5 P1 S3 output) instead of the 20-dim handcrafted
``babysteps.stage4.features.extract_episode_features``.

The feature injection point is the new `demo_features_provider` hook on
`run_episode`: a callable `seed -> np.ndarray` that overrides the
default handcrafted Z. The vision-grounded LatentPack (S5 Half A) is
loaded as-is — the only difference from M2a at inference time is which
Z the IntentHead sees.

G4 (Δpp(latent vs failure-agnostic retry) ≥ 10pp) and G5
(Δpp(latent vs oracle) ≥ -5pp) are computed from these per-policy
numbers when run against the real env on enough seeds.

Example::

    python scripts/stage5_p1_run_eval.py \\
        --task PushCube-v1 \\
        --pack-dir models/stage5/p1_vision/PushCube-v1 \\
        --features-dir datasets/stage5/varied_intent/PushCube-v1/features/ \\
        --out-dir reports/stage5/p1_vision_g4_g5/PushCube-v1 \\
        --eval-seeds 100-149
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

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
from babysteps.stage5.latent_intent import build_latent_intent  # noqa: E402


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
    """Stub adapter for sim-free smoke (same pattern as M2a A3)."""
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


def _make_vision_provider(features_dir: Path):
    """Returns `provider(seed) -> np.ndarray` loading cached DINOv2 features.

    The `_dinov2.npy` suffix matches the layout written by
    scripts/stage5_cache_dinov2.py. Missing files raise FileNotFoundError;
    run_episode catches provider exceptions and falls back to None, which
    the latent_revision policy degrades to a no-op revision branch.
    """
    features_dir = Path(features_dir)
    def _provider(seed: int) -> np.ndarray:
        p = features_dir / f"seed_{seed:04d}_dinov2.npy"
        return np.load(p).astype(np.float32)
    return _provider


def _eval_policy(adapter, policy, *, episode_prefix: str, seeds: list[int],
                 demo_features_provider=None, initial_intent_provider=None) -> dict:
    """Returns per-seed list of (success, success_at_initial, success_at_retry)."""
    rows = []
    for seed in seeds:
        rec = run_episode(
            episode_id=f"{episode_prefix}_seed_{seed:04d}",
            seed=seed,
            adapter=adapter,
            policy=policy,
            demo_features_provider=demo_features_provider,
            initial_intent_provider=initial_intent_provider,
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
    p = argparse.ArgumentParser(description="Stage-5 P1 paired eval (vision-grounded).")
    p.add_argument("--task", required=True,
                   choices=sorted(_TASK_TO_ADAPTER_MODULE))
    p.add_argument("--pack-dir", type=Path, required=True,
                   help="Vision-grounded LatentPack directory (S5 Half A output).")
    p.add_argument("--features-dir", type=Path, required=True,
                   help="Directory of cached demo-view seed_NNNN_dinov2.npy "
                        "files (S5 S3 output).")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--eval-seeds", required=True,
                   help="Seed range A-B (inclusive) or single int.")
    p.add_argument("--fake", action="store_true",
                   help="Use FakeEnvRunner via tests.conftest (sim-free smoke).")
    p.add_argument("--policies",
                   default="latent,babysteps_selective,same_intent_retry,oracle_factor_revision",
                   help="Comma-separated list to evaluate.")
    p.add_argument("--latent-initial", action="store_true",
                   help="Sever A: decode attempt-1 intent from demo-view vision "
                        "(DINOv2->IntentHead->nearest-centroid) for ALL "
                        "policies, instead of the scripted demo->intent. "
                        "Makes the whole loop latent-input, not just the "
                        "latent revision.")
    args = p.parse_args(argv)

    seeds = _parse_seed_range(args.eval_seeds)
    pack = load_latent_pack(args.pack_dir)
    print(f"loaded LatentPack from {args.pack_dir}; "
          f"centroids for factors {sorted(pack.centroids.keys())}")
    vision_provider = _make_vision_provider(args.features_dir)
    print(f"vision provider: {args.features_dir}")

    # Sever A — latent attempt-1 intent from demo-view features for ALL
    # policies (whole loop latent).
    initial_provider = None
    if args.latent_initial:
        def initial_provider(seed, scripted):  # type: ignore[misc]
            try:
                return build_latent_intent(pack, vision_provider(seed), scripted)
            except Exception:
                return scripted
        print("latent-initial ON: attempt-1 intent decoded from demo-view vision (Sever A)")

    # Only the latent policy needs the vision-grounded Z; the
    # baselines (babysteps_selective, same_intent_retry,
    # oracle_factor_revision) don't read demo_features. We pass None to
    # them to preserve the M2a / Stage-4 path byte-for-byte and so the
    # baseline numbers stay directly comparable to M2a's A3 results.
    policy_factory = {
        "latent": lambda: latent_revision_factory(pack),
        "babysteps_selective": lambda: babysteps_selective,
        "same_intent_retry": lambda: same_intent_retry,
        "oracle_factor_revision": lambda: oracle_factor_revision,
    }
    policy_provider = {
        "latent": vision_provider,
        "babysteps_selective": None,
        "same_intent_retry": None,
        "oracle_factor_revision": None,
    }
    policy_names = [n.strip() for n in args.policies.split(",") if n.strip()]

    results = {}
    for name in policy_names:
        print(f"\n=== {name} (n_seeds={len(seeds)}, fake={args.fake}) ===")
        adapter = _make_fake_adapter(args.task) if args.fake else _make_real_adapter(args.task)
        try:
            res = _eval_policy(
                adapter, policy_factory[name](),
                episode_prefix=f"s5_p1_{args.task.replace('-', '_')}_{name}",
                seeds=seeds,
                demo_features_provider=policy_provider[name],
                initial_intent_provider=initial_provider,
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
    (args.out_dir / "p1_results.json").write_text(json.dumps({
        "config": {
            "task": args.task, "fake": args.fake,
            "latent_initial": bool(args.latent_initial),
            "eval_seeds": [int(s) for s in seeds],
            "pack_dir": str(args.pack_dir),
            "features_dir": str(args.features_dir),
        },
        "per_policy": results,
        "summary": summary,
    }, indent=2, sort_keys=True) + "\n")
    # Also write a short Markdown report mirroring M2a's notes.md shape.
    _write_report_md(args.out_dir / "report.md", args, results, summary)
    print(f"\nwrote {args.out_dir}/p1_results.json")
    print(f"wrote {args.out_dir}/report.md")
    print(f"summary: {summary}")
    return 0


def _write_report_md(out_path: Path, args, results: dict, summary: dict) -> None:
    """Write a human-readable G4/G5 report alongside p1_results.json."""
    lines = [
        f"# Stage-5 P1 vision-grounded eval — {args.task}",
        "",
        f"- Pack: `{args.pack_dir}`",
        f"- Features: `{args.features_dir}` (spatial_mean DINOv2)",
        f"- Eval seeds: `{args.eval_seeds}` "
        f"(n={results.get('latent', {}).get('n', '?')})",
        f"- Real env: `{not args.fake}`",
        "",
        "## Per-policy success rates",
        "",
        "| policy | initial | final |",
        "|---|---|---|",
    ]
    for name, res in results.items():
        lines.append(
            f"| `{name}` | {res['initial_success_rate']:.3f} | "
            f"{res['final_success_rate']:.3f} |"
        )
    lines.extend([
        "",
        "## G4 / G5 gates (vs M2a's handcrafted-Z baseline)",
        "",
    ])
    if "delta_pp_vs_same_intent_retry" in summary:
        g4 = summary["delta_pp_vs_same_intent_retry"]
        g4_pass = "PASS" if g4 >= 10 else "FAIL"
        lines.append(
            f"- **G4**: Δpp(latent vs same_intent_retry) = "
            f"{g4:+.2f}pp ({g4_pass}, threshold ≥ 10)"
        )
    if "delta_pp_vs_oracle" in summary:
        g5 = summary["delta_pp_vs_oracle"]
        g5_pass = "PASS" if g5 >= -5 else "FAIL"
        lines.append(
            f"- **G5**: Δpp(latent vs oracle) = "
            f"{g5:+.2f}pp ({g5_pass}, threshold ≥ -5)"
        )
    out_path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
