"""Stage-5 step 1 — VLM wrong-factor HISTOGRAM on held-out PokeCube.

The 8-condition main table reported VLM constrained-attribution accuracy ~0.017
on contact_region, but it reduced the predicted factor to a bool before
aggregating. This recovers the DISTRIBUTION: which factor does the 8B VLM
actually pick from a single third-person failure frame? (Hypothesis: it
systematically reads "the cube is in the wrong place" as a goal_state /
object_motion error, never the occluded contact_region.)

Faithful to the deployed path: the saved frame ALONE is insufficient — the
constrained prompt (vlm_attribute.build_constrained_prompt) needs the attempted
``initial_intent`` + ``failure_predicate``, which are built live, not stored in
the PNG. So each held-out (seed, direction, wrong_face) episode is RE-SOURCED
with the same ``_source_episode`` the main table used, and the real frame is fed
to the SAME ``VLMAttributor`` (reset_cost/cost_snapshot bracketing, single
third-person image, no wrist view).

GPU: real ``PokeCubeEnvRunner`` + InternVL3.5-8B.
``--mock``: ``FakePokeEnvRunner`` + ``MockVLMClient`` (login-node smoke, no
GPU/Vulkan — validates the re-source + attribute + tally pipeline only).

Example::

    python scripts/stage5_pokecube_step1_diagnostic.py \\
        --seeds 0-299 --directions +x,+y,-y --max-approach-dist 0.785 \\
        --target-n 20 \\
        --out reports/stage5/pokecube_step1_diagnostic/results.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _ROOT / "scripts"
for _p in (str(_ROOT), str(_SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from babysteps.schemas import INTENT_FIELDS  # noqa: E402
from babysteps.stage5.revision_policy import (  # noqa: E402
    AttributionObs, VLMAttributor,
)
import stage5_pokecube_maintable_eval as M  # noqa: E402


def run_diagnostic(runner, adapter, vlm, *, seeds, directions, max_approach_dist,
                   target_n, capture_frames, frames_dir):
    """Re-source each held-out episode and record the VLM's constrained factor
    pick. Mirrors the main table's 'vlm' attributor_override path exactly."""
    reachable = M._reachable_seeds(runner, seeds, directions, max_approach_dist,
                                   target_n)
    attributor = VLMAttributor(vlm)
    rows: list[dict] = []
    for s in reachable:
        for d in directions:
            correct = M._DIR_TO_FACE[d]
            for w in M.LOTO_FACES:
                if w == correct:
                    continue
                ep = M._source_episode(
                    runner, adapter, s, d, w,
                    capture_frames=capture_frames, frames_dir=frames_dir)
                obs = AttributionObs(
                    task="PokeCube-v1", factor_menu=INTENT_FIELDS,
                    failure_predicate=ep.e_fail.predicate,
                    initial_intent=ep.initial, frame_path=ep.frame_path,
                    wrist_frame_path=ep.wrist_frame_path, key=f"{s}:{d}:{w}")
                res = attributor.attribute(obs)
                rows.append({
                    "seed": s, "direction": d, "wrong_face": w,
                    "correct_face": correct,
                    "initial_success": bool(ep.initial_success),
                    "predicted_factor": res.factor,
                    "attribution_correct": bool(res.factor == "contact_region"),
                    "latency_s": float(res.latency_s),
                    "gen_tokens": int(res.cost.get("gen_tokens", 0)),
                })
    return rows, len(reachable)


def summarize(rows: list[dict]) -> dict:
    n = max(1, len(rows))
    hist = Counter((r["predicted_factor"] or "parse_fail") for r in rows)
    by_dir: dict[str, Counter] = {}
    for r in rows:
        by_dir.setdefault(r["direction"], Counter())[
            r["predicted_factor"] or "parse_fail"] += 1
    return {
        "n": len(rows),
        "attribution_accuracy": sum(r["attribution_correct"] for r in rows) / n,
        "predicted_factor_histogram": dict(hist.most_common()),
        "by_direction_histogram": {d: dict(c.most_common())
                                   for d, c in by_dir.items()},
        "mean_latency_s": sum(r["latency_s"] for r in rows) / n,
        "mean_gen_tokens": sum(r["gen_tokens"] for r in rows) / n,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", default="0-299")
    p.add_argument("--directions", default="+x,+y,-y")
    p.add_argument("--max-approach-dist", type=float, default=0.785)
    p.add_argument("--target-n", type=int, default=20)
    p.add_argument("--mock", action="store_true",
                   help="FakePoke + MockVLM (sim-free smoke; no GPU/Vulkan)")
    p.add_argument("--out", type=Path, default=Path(
        "reports/stage5/pokecube_step1_diagnostic/results.json"))
    args = p.parse_args(argv)

    directions = [d.strip() for d in args.directions.split(",") if d.strip()]
    seeds = M._parse_seeds(args.seeds)
    frames_dir = args.out.parent / "frames"

    from babysteps.envs.pokecube_adapter import PokeCubeAdapter
    adapter = PokeCubeAdapter()
    if args.mock:
        from tests.conftest import FakePokeEnvRunner
        from babysteps.stage5.vlm_attribute import MockVLMClient
        runner = FakePokeEnvRunner()
        vlm = MockVLMClient(constrained_response="contact_region")
        capture_frames = False
    else:
        from babysteps.envs.pokecube_runner import PokeCubeEnvRunner
        from babysteps.stage5.vlm_attribute import InternVLClient
        runner = PokeCubeEnvRunner(render_mode="rgb_array")
        vlm = InternVLClient()
        print("loading InternVL3.5-8B ...")
        vlm.load()
        capture_frames = True

    print(f"=== PokeCube step-1 VLM wrong-factor histogram (mock={args.mock}) "
          f"dirs={directions} target_n={args.target_n} ===")
    try:
        rows, n_reach = run_diagnostic(
            runner, adapter, vlm, seeds=seeds, directions=directions,
            max_approach_dist=args.max_approach_dist, target_n=args.target_n,
            capture_frames=capture_frames, frames_dir=frames_dir)
    finally:
        if hasattr(runner, "close"):
            try:
                runner.close()
            except Exception:
                pass

    summary = summarize(rows)
    result = {"summary": summary, "n_reachable_seeds": n_reach,
              "config": {"directions": directions, "mock": args.mock,
                         "max_approach_dist": args.max_approach_dist,
                         "target_n": args.target_n},
              "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))

    print(f"\n=== VLM predicted-factor histogram (n={summary['n']}, "
          f"attr_acc={summary['attribution_accuracy']:.3f}) ===")
    for f, c in summary["predicted_factor_histogram"].items():
        print(f"  {f:<22} {c}")
    print(f"mean latency {summary['mean_latency_s']:.3f}s  "
          f"mean gen_tokens {summary['mean_gen_tokens']:.1f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
