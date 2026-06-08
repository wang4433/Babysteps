"""Stage-5 LOTO eval — does the FROZEN shared RevisionPolicy transfer to a
held-out PokeCube family?

This is the scientific payoff of build-order step 5 for the contact_region cell.
The shared scorer is trained ONCE on pooled PushCube contact_region + StackCube
goal_state tuples (g_i masked; see scripts/stage5_train_shared_policy.py --gi
none), FROZEN, and applied here to PokeCube — a task it never saw. PokeCube is a
genuine leave-one-task-family-out partner: it shares contact_region's candidate
SEMANTICS + residual->face revision RULE with PushCube but differs entirely in
EXECUTION (a grasped peg pokes the cube; see babysteps/envs/pokecube_runner.py).

Per-episode protocol (encoder-free — isolates the SCORER's transfer; PokeCube has
no trained encoder BY DESIGN, that's what "held out" means):
  for a reachable seed + injected goal direction g (correct face f* = oracle):
    for each WRONG initial face w != f* (the natural mis-grounding stand-in):
      1. run(contact_region=w)  -> initial attempt FAILS (poke wrong side)
      2. residual = goal_xy - final_cube_xy   (== _observed_residual: where the
         cube SHOULD be minus where it ended; identical to the PushCube training
         tuples, so the frozen scorer reads the same signal)
      3. frozen shared scorer.decide(contact_region, current=w, candidates=3
         reachable faces, e_fail=(direction_error, residual), g_i=None) -> c
      4. re-run(contact_region=c) -> recovery success?
Conditions reported (mirror the unified main table): open_loop (keep w),
random_face, shared_scorer (frozen, the headline), oracle (f* directly = ceiling).

Reachable seeds are selected with the same all-directions reach filter the
3-way kill-gate validated (--max-approach-dist 0.785). Candidate set = the 3
reachable faces {minus_x_face(+x), minus_y_face(+y), plus_y_face(-y)}; -x is
reach-dead and excluded.

GPU/Vulkan only for the real PokeCube runner; mani_skill import is deferred. The
scorer load + aggregation are sim-free (CPU torch).

Example::

    python scripts/stage5_pokecube_loto_eval.py \\
        --scorer models/stage5/shared_policy/pooled_gi_none.pt \\
        --seeds 0-299 --target-n 20 --max-approach-dist 0.785 \\
        --out reports/stage5/pokecube_loto/results.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_SCRIPTS = _ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

# 3 reachable PokeCube contact_region candidates (−x reach-dead, excluded).
LOTO_FACES = ("minus_x_face", "minus_y_face", "plus_y_face")
# direction token (for goal-move injection) -> oracle correct face
_DIR_TO_FACE = {"+x": "minus_x_face", "+y": "minus_y_face", "-y": "plus_y_face"}
_DIR_TO_MOTION = {"+x": None, "+y": "translate_+y", "-y": "translate_-y"}


def _parse_seeds(s: str) -> list[int]:
    if "-" in s and "," not in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",") if x]


def _residual(goal_xy, final_xy):
    """goal - final : where the cube SHOULD be minus where it ended. Identical to
    residual_reviser._observed_residual (final = cube0 + observed displacement =
    AttemptResult.final_obj_xy), so the frozen scorer reads the training signal."""
    return (float(goal_xy[0]) - float(final_xy[0]),
            float(goal_xy[1]) - float(final_xy[1]))


def aggregate_loto(rows: list[dict]) -> dict:
    """Pure aggregation over per-episode rows (one per seed×direction×wrong-face).
    A row has: open_loop_success, scorer_face_correct, scorer_success,
    random_success, oracle_success."""
    n = max(1, len(rows))
    def rate(key):
        return sum(1 for r in rows if r.get(key)) / n
    return {
        "n_episodes": len(rows),
        "open_loop_success": rate("open_loop_success"),
        "random_face_success": rate("random_success"),
        "shared_scorer_face_acc": rate("scorer_face_correct"),
        "shared_scorer_recovery": rate("scorer_success"),
        "oracle_success": rate("oracle_success"),
    }


def run_loto(policy, adapter, runner, *, seeds, directions, candidates,
             max_approach_dist=None, target_n=None, seed_rng=0):
    """Core eval loop. `policy` exposes .decide(RevisionRequest)->RevisionDecision.
    Returns (rows, n_reachable_seeds). Runner-agnostic so a fake runner can drive
    a sim-free test."""
    from babysteps.stage5.revision_policy import FailureEvidence, RevisionRequest
    from babysteps.schemas import Intent
    from stage5_pokecube_killgate import _worst_waypoint_dist

    rng = random.Random(seed_rng)
    rows: list[dict] = []
    reachable = 0
    for s in seeds:
        # reachability is goal-direction independent here; gate on the native
        # +x scene (the filter already maxes over all tested directions).
        runner.set_injection(None)
        scene0 = runner.reset(int(s))
        if max_approach_dist is not None and _worst_waypoint_dist(
                scene0.cube_xy, scene0.goal_xy, list(directions)) > max_approach_dist:
            continue
        if target_n is not None and reachable >= target_n:
            break
        reachable += 1
        for d in directions:
            correct = _DIR_TO_FACE[d]
            motion = _DIR_TO_MOTION[d]
            runner.set_injection(motion)
            scene = runner.reset(int(s))

            # oracle ceiling (correct face directly)
            oracle_att = runner.run(_intent(adapter, scene, correct), scene)

            for w in candidates:
                if w == correct:
                    continue
                runner.set_injection(motion)
                scene_w = runner.reset(int(s))
                att1 = runner.run(_intent(adapter, scene_w, w), scene_w)
                resid = _residual(scene_w.goal_xy, att1.final_obj_xy)

                dec = policy.decide(RevisionRequest(
                    factor="contact_region", current_value=w,
                    candidates=tuple(candidates),
                    e_fail=FailureEvidence("direction_error", resid), g_i=None))
                corrected = dec.new_value if dec.new_value is not None else w

                runner.set_injection(motion)
                scene_c = runner.reset(int(s))
                att2 = runner.run(_intent(adapter, scene_c, corrected), scene_c)

                # random baseline: a random face among candidates
                rface = rng.choice([f for f in candidates])
                runner.set_injection(motion)
                scene_r = runner.reset(int(s))
                att_r = runner.run(_intent(adapter, scene_r, rface), scene_r)

                rows.append({
                    "seed": int(s), "direction": d,
                    "correct_face": correct, "wrong_face": w,
                    "residual_xy": [round(resid[0], 4), round(resid[1], 4)],
                    "scorer_face": corrected,
                    "scorer_face_correct": bool(corrected == correct),
                    "open_loop_success": bool(att1.success),
                    "scorer_success": bool(att2.success),
                    "random_face": rface,
                    "random_success": bool(att_r.success),
                    "oracle_success": bool(oracle_att.success),
                })
    return rows, reachable


def _intent(adapter, scene, face: str):
    """Oracle intent for the scene with contact_region OVERRIDDEN to `face`
    (and approach_direction kept consistent). All other factors from the oracle."""
    from dataclasses import replace
    from babysteps.envs.scene import face_to_approach
    base = adapter.oracle_correct_intent(scene)
    return replace(base, contact_region=face,
                   approach_direction=face_to_approach(face))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scorer", type=Path, required=True,
                   help="Frozen shared-scorer checkpoint (save_shared_scorer).")
    p.add_argument("--seeds", default="0-299")
    p.add_argument("--directions", default="+x,+y,-y")
    p.add_argument("--max-approach-dist", type=float, default=0.785)
    p.add_argument("--target-n", type=int, default=20)
    p.add_argument("--out", type=Path,
                   default=Path("reports/stage5/pokecube_loto/results.json"))
    args = p.parse_args(argv)

    from babysteps.stage5.shared_revision_policy import SharedScorerPolicy
    from babysteps.envs.pokecube_adapter import PokeCubeAdapter
    from babysteps.envs.pokecube_runner import PokeCubeEnvRunner

    policy = SharedScorerPolicy.from_pack(args.scorer)
    adapter = PokeCubeAdapter()
    runner = PokeCubeEnvRunner()
    seeds = _parse_seeds(args.seeds)
    directions = [d.strip() for d in args.directions.split(",") if d.strip()]

    print(f"=== PokeCube LOTO eval: scorer={args.scorer} dirs={directions} "
          f"target_n={args.target_n} max_approach_dist={args.max_approach_dist} ===")
    try:
        rows, n_reach = run_loto(
            policy, adapter, runner, seeds=seeds, directions=directions,
            candidates=LOTO_FACES, max_approach_dist=args.max_approach_dist,
            target_n=args.target_n)
    finally:
        if hasattr(runner, "close"):
            try:
                runner.close()
            except Exception:
                pass

    agg = aggregate_loto(rows)
    agg["n_reachable_seeds"] = n_reach
    agg["candidates"] = list(LOTO_FACES)
    agg["scorer"] = str(args.scorer)
    print(json.dumps(agg, indent=2))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({**agg, "rows": rows}, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
