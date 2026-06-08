"""Stage-5 — StackCube goal_state full-vision natural loop.

The 2nd positive latent task. Unlike PushCube (instance-mismatch + residual
feedback), StackCube's natural failure is goal UNDER-SPECIFICATION: the demo is
goal-ambiguous, so an initial intent that grounds goal_state as `cube_at_target`
(place-near) executes a low-z drop that collides with cubeB and scatters
(`goal_not_satisfied`); the deterministic `goal_refinement` operator lifts it to
`cubeA_on_cubeB` (stack), which the retry executes.

The loop here decodes the INITIAL goal_state straight from the demo CLIP feature
(VisionIntentExtractor over a goal_state pack), then runs exec -> fail -> revise
-> retry on the real StackCube runner. Only goal_state varies; the other factors
are the oracle/task-constant values, so the loop isolates goal_state grounding +
revision.

Grounding<->revision spectrum (choose the demo features dir + matching pack):
  * RETRACT first_last demos  -> goal_state grounds ~0.92 -> few misgroundings,
    grounding carries it (mirrors PushCube full-vision: vision intent ~= oracle).
  * AMBIGUOUS whole-clip demos -> grounds ~0.63 -> many misgroundings, the
    failure-driven goal_refinement revision carries it.
Both close toward the stack-execution ceiling (~0.82, the runner's intrinsic
pick-and-place reliability — NOT 1.0; the metric is recovery vs no-recovery).

Revisers:
  * same_intent  — retry the identical (possibly mis-grounded) intent.  [open-loop]
  * operator     — goal_refinement: cube_at_target -> cubeA_on_cubeB (the single
                   valid strict-extension; deterministic slot-local edit).
  * oracle_value — set goal_state to the oracle (cubeA_on_cubeB).       [upper bound]
goal_state revision is a single deterministic transition (no continuous feedback
signal, unlike PushCube's residual head), so `operator` IS the principled
slot-local revision; we do not add a degenerate "learned" head here.

GPU/Vulkan for the real StackCube runner; the core loop is sim-free and
unit-tested against a fake runner whose success keys on intent.goal_state.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from babysteps.failure import Attribution  # noqa: E402
from babysteps.revision import revise_intent as rule_revise_intent  # noqa: E402
from babysteps.schemas import INTENT_FIELDS, Intent  # noqa: E402

_GOAL_STATE_IDX = INTENT_FIELDS.index("goal_state")


# ---------- revisers (signature (initial, fp, scene, adapter) -> Intent) ----- #

def _rev_same_intent(initial: Intent, fp, scene, adapter) -> Intent:
    return initial


def _rev_operator(initial: Intent, fp, scene, adapter) -> Intent:
    """goal_refinement: the Stage-0 deterministic slot-local edit. Only defined
    for the under-specified cube_at_target -> cubeA_on_cubeB strict-extension; a
    no-op if the initial goal_state is already cubeA_on_cubeB (correct decode
    that nonetheless failed at execution -> nothing this operator can fix)."""
    if initial.goal_state != "cube_at_target":
        return initial
    attr = Attribution(
        semantic_failure=True, wrong_factor="goal_state",
        freeze=tuple(f for f in INTENT_FIELDS if f != "goal_state"),
        revise=("goal_state",),
    )
    revised, _rec = rule_revise_intent(initial, attr, scene)
    return revised


def _rev_oracle_value(initial: Intent, fp, scene, adapter) -> Intent:
    """Upper bound: set goal_state to the exec-scene oracle (cubeA_on_cubeB)."""
    return replace(initial, goal_state=adapter.oracle_correct_intent(scene).goal_state)


REVISERS = {
    "same_intent": _rev_same_intent,
    "operator": _rev_operator,
    "oracle_value": _rev_oracle_value,
}


# ---------- the natural-failure episode ------------------------------------- #

def run_goalstate_episode(adapter, runner, extractor, *, seed: int,
                          demo_features_dir, suffix: str,
                          revisers: list[str], reviser_fns: dict | None = None,
                          demo_class: str = "stack",
                          tuple_sink: list | None = None) -> dict:
    """One demo -> vision-decode goal_state -> exec -> fail -> revise -> retry.

    The demo is the TRUE task (a stack); its goal_state is vision-decoded from
    the cached demo feature. If the decode is the under-specified cube_at_target,
    execution collides/scatters (goal_not_satisfied) and the revisers attempt
    recovery. Only goal_state varies off the oracle intent.
    """
    fns = reviser_fns or REVISERS
    scene = runner.reset(seed)

    feat_path = Path(demo_features_dir) / f"seed_{seed:04d}_{demo_class}_{suffix}.npy"
    if not feat_path.exists():
        raise FileNotFoundError(f"missing demo feature for seed {seed}: {feat_path}")
    decoded_gs = extractor.decode_factor(np.load(feat_path), _GOAL_STATE_IDX)

    oracle = adapter.oracle_correct_intent(scene)
    true_gs = oracle.goal_state  # cubeA_on_cubeB (the task is always a stack)
    initial_intent = replace(oracle, goal_state=decoded_gs)

    attempt_1 = runner.run(initial_intent, scene)
    fp = adapter.build_failure_packet(initial_intent, attempt_1, scene)

    out = {
        "seed": seed,
        "decoded_goal_state": decoded_gs,
        "true_goal_state": true_gs,
        "vision_decode_correct": bool(decoded_gs == true_gs),
        "initial_success": bool(attempt_1.success),
        "failure_predicate": fp.failure_predicate,
        "revisers": {},
    }

    # Pooled shared-policy training tuple (one per failed episode). goal_state has
    # NO continuous feedback (residual_xy=null) — coverage for the shared scorer,
    # but EXCLUDED from any learned-choice metric (its single valid transition is
    # the deterministic goal_refinement operator). Mirrors the PushCube
    # natural-loop --dump-tuples. correct_value is a sim-derived TRAINING label
    # (allowed off the demo->intent path, CLAUDE.md inv #4).
    if tuple_sink is not None and not attempt_1.success:
        tuple_sink.append({
            "task": "StackCube-v1", "factor": "goal_state",
            "seed": seed,
            "current_value": decoded_gs, "correct_value": true_gs,
            "failure_predicate": fp.failure_predicate,
            "residual_xy": None,
            "candidates": ["cube_at_target", "cubeA_on_cubeB"],
        })

    for name in revisers:
        if attempt_1.success:
            out["revisers"][name] = {"final_success": True, "revised": False,
                                     "new_goal_state": initial_intent.goal_state}
            continue
        revised = fns[name](initial_intent, fp, scene, adapter)
        attempt_2 = runner.run(revised, scene)
        out["revisers"][name] = {
            "final_success": bool(attempt_2.success),
            "revised": revised.goal_state != initial_intent.goal_state,
            "new_goal_state": revised.goal_state,
        }
    return out


def summarize(rows: list[dict], revisers: list[str]) -> dict:
    n = len(rows)
    fails = [r for r in rows if not r["initial_success"]]
    summ = {
        "n": n, "n_initial_fail": len(fails),
        "vision_decode_acc": (sum(r["vision_decode_correct"] for r in rows) / max(1, n)),
        "initial_success_rate": sum(r["initial_success"] for r in rows) / max(1, n),
        "final_success_rate": {},
        "final_success_rate_on_initial_fail": {},
    }
    for name in revisers:
        summ["final_success_rate"][name] = (
            sum(r["revisers"][name]["final_success"] for r in rows) / max(1, n))
        summ["final_success_rate_on_initial_fail"][name] = (
            sum(r["revisers"][name]["final_success"] for r in fails) / max(1, len(fails)))
    return summ


def _parse_seed_range(s: str) -> list[int]:
    if "-" in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(s)]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pack-dir", type=Path, required=True,
                   help="goal_state LatentPack (stage5_train_goalstate_pack.py).")
    p.add_argument("--demo-features-dir", type=Path, required=True,
                   help="Cached eval demo features (seed_NNNN_<class>_<suffix>.npy "
                        "from stage5_goal_state_probe.py --dump-features on EVAL seeds).")
    p.add_argument("--feature-suffix", type=str, default="dinov2_fl")
    p.add_argument("--demo-class", type=str, default="stack",
                   help="Which dumped clip is the eval demo (the true task = stack).")
    p.add_argument("--eval-seeds", default="200-249",
                   help="Held-out EXEC/demo seed range (disjoint from pack-train).")
    p.add_argument("--revisers", default="same_intent,operator,oracle_value")
    p.add_argument("--dump-tuples", type=Path, default=None,
                   help="Write pooled shared-policy training tuples (one per "
                        "failed episode) to this jsonl. Coverage for the shared "
                        "RevisionPolicy; goal_state has no residual feedback.")
    p.add_argument("--out-dir", type=Path,
                   default=Path("reports/stage5/goalstate_loop/StackCube-v1"))
    args = p.parse_args(argv)

    revisers = [r.strip() for r in args.revisers.split(",") if r.strip()]
    unknown = [r for r in revisers if r not in REVISERS]
    if unknown:
        p.error(f"unknown reviser(s): {unknown}")
    seeds = _parse_seed_range(args.eval_seeds)

    from babysteps.envs.stackcube_adapter import StackCubeAdapter
    from babysteps.envs.stackcube_runner import StackCubeEnvRunner
    from babysteps.stage4.vision_intent import VisionIntentExtractor

    adapter = StackCubeAdapter()
    runner = StackCubeEnvRunner()
    template = adapter.oracle_correct_intent(runner.reset(seeds[0]))
    extractor = VisionIntentExtractor.from_pack(args.pack_dir, template)
    print(f"goal_state loop: pack={args.pack_dir} feats={args.demo_features_dir} "
          f"suffix={args.feature_suffix} seeds={args.eval_seeds}")

    tuple_sink: list | None = [] if args.dump_tuples is not None else None
    rows = []
    for s in seeds:
        rows.append(run_goalstate_episode(
            adapter, runner, extractor, seed=s,
            demo_features_dir=args.demo_features_dir, suffix=args.feature_suffix,
            revisers=revisers, demo_class=args.demo_class,
            tuple_sink=tuple_sink))

    summ = summarize(rows, revisers)
    print(f"=== StackCube goal_state loop (n={summ['n']}, "
          f"n_initial_fail={summ['n_initial_fail']}) ===")
    print(f"  vision-decode accuracy: {summ['vision_decode_acc']:.3f}")
    print(f"  initial success: {summ['initial_success_rate']:.3f}")
    for name in revisers:
        print(f"  final[{name:14s}] all={summ['final_success_rate'][name]:.3f}  "
              f"on_fail={summ['final_success_rate_on_initial_fail'][name]:.3f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "goalstate_loop_results.json").write_text(json.dumps({
        "config": {"task": "StackCube-v1", "pack_dir": str(args.pack_dir),
                   "demo_features_dir": str(args.demo_features_dir),
                   "feature_suffix": args.feature_suffix,
                   "eval_seeds": args.eval_seeds, "revisers": revisers},
        "summary": summ, "rows": rows}, indent=2) + "\n")
    print(f"\nwrote {args.out_dir}/goalstate_loop_results.json")

    if tuple_sink is not None:
        args.dump_tuples.parent.mkdir(parents=True, exist_ok=True)
        with args.dump_tuples.open("w") as f:
            for t in tuple_sink:
                f.write(json.dumps(t) + "\n")
        print(f"wrote {len(tuple_sink)} goal_state tuples to {args.dump_tuples}")

    if hasattr(adapter, "close"):
        adapter.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
