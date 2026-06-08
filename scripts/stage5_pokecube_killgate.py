"""Stage-5 step-3 GPU KILL-GATE — confirm the PokeCube contact_region family.

The grasp+poke PokeCubeEnvRunner is the one genuine engineering risk in the
leave-one-task-family-out plan. Before PokeCube can serve as the held-out LOTO
cell, the runner must be able to SUCCEED when given the ORACLE (correct) contact
face. This harness measures exactly that: per poke direction, reset PokeCube
(with goal-move injection for non-+x), build the oracle intent from the scene
geometry (adapter.oracle_correct_intent), run the grasp+poke, and report
success + terminal sim diagnostics (is_peg_grasped, is_cube_placed,
head_to_cube_dist, cube_displacement).

Decision rule (the "kill-gate"):
  * native +x oracle poke success >= GATE (default 0.8)  -> family CORE confirmed
    (grasp+poke is feasible at all; PokeCube is a real controllable family).
  * >= 2 directions >= GATE                              -> multi-candidate family
    (the 4-face contact_region vocab is executable, mirroring PushCube's probe).
If +x is below the gate the family is dead regardless of the other faces; tune
the geometry constants in pokecube_runner.py and re-run before any LOTO work.

This mirrors scripts/stage5_pokecube_probe.py::probe_pushcube_4face but for the
NEW runner. GPU/Vulkan only; mani_skill import is deferred to call time so this
file imports on the login node.

Example::

    python scripts/stage5_pokecube_killgate.py --seeds 0-23 --directions +x \\
        --out reports/stage5/pokecube_killgate/findings.json
    # full 4-face landscape in one job:
    python scripts/stage5_pokecube_killgate.py --seeds 0-11 \\
        --directions +x,-x,+y,-y
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# direction token -> goal-move injection motion (None = native +x goal)
_DIR_TO_MOTION = {
    "+x": None,            # native PokeCube goal is already +x
    "-x": "translate_-x",
    "+y": "translate_+y",
    "-y": "translate_-y",
}

# Panda base x in PokeCube-v1 (poke_cube.py _load_agent: Pose(p=[-0.615,0,0])).
_BASE_XY = (-0.615, 0.0)


def _worst_waypoint_dist(cube_xy, goal_xy, directions: list[str]) -> float:
    """Max distance from the Panda base over the prepoke + poke waypoints across
    the tested directions — the reachability bottleneck (a seed only works in a
    direction if the gripper can reach BOTH that direction's standoff and poke
    pose). Mirrors pokecube_runner.run's geometry exactly so the filter predicts
    reach. goal_xy here is the NATIVE +x goal; per-direction goals are derived by
    rotating the native cube->goal offset onto each motion (matching the runner's
    set_injection, which keeps |cube->goal| fixed)."""
    import numpy as np
    from babysteps.envs.pokecube_runner import (
        _CUBE_HALF, _CONTACT_GAP, _PEG_HALF_LENGTH, _POKE_OVERSHOOT)
    from babysteps.envs.scene import face_to_push_unit, motion_to_unit
    cube = np.asarray(cube_xy, dtype=float)
    base = np.asarray(_BASE_XY, dtype=float)
    push_dist = float(np.linalg.norm(np.asarray(goal_xy, float) - cube))
    worst = 0.0
    for d in directions:
        motion = _DIR_TO_MOTION[d]
        unit = (np.array([1.0, 0.0]) if motion is None else motion_to_unit(motion))
        goal_d = cube + push_dist * unit
        # face that produces this motion -> poke_dir (= cube travel unit)
        face = {"+x": "minus_x_face", "-x": "plus_x_face",
                "+y": "minus_y_face", "-y": "plus_y_face"}[d]
        poke_dir = face_to_push_unit(face)
        prepoke = cube - poke_dir * (_CUBE_HALF + _CONTACT_GAP + _PEG_HALF_LENGTH)
        poke = goal_d - poke_dir * (_CUBE_HALF + _PEG_HALF_LENGTH - _POKE_OVERSHOOT)
        for wp in (prepoke, poke):
            worst = max(worst, float(np.linalg.norm(wp - base)))
    return worst


def _parse_seeds(s: str) -> list[int]:
    if "-" in s and "," not in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",") if x]


def run_killgate(seeds: list[int], directions: list[str], gate: float,
                 max_cube_x: float | None = None, target_n: int | None = None,
                 max_approach_dist: float | None = None) -> dict:
    """Reachability pre-filters (a seed is SKIPPED if it fails any set filter):
      * max_cube_x       — native cube_x exceeds it (crude near-cube filter).
      * max_approach_dist — worst prepoke/poke waypoint distance from the Panda
        base, over ALL tested directions, exceeds it. This is the principled
        all-directions reachability gate: it selects the SAME reachable seed set
        across directions, so +x/+y/-y are compared on identical seeds.
    target_n stops once that many reachable seeds are collected (per direction)."""
    out: dict = {"seeds": [int(s) for s in seeds], "gate": gate,
                 "max_cube_x": max_cube_x, "target_n": target_n,
                 "max_approach_dist": max_approach_dist,
                 "per_direction": {}}
    try:
        from babysteps.envs.pokecube_adapter import PokeCubeAdapter
        from babysteps.envs.pokecube_runner import PokeCubeEnvRunner
    except Exception as exc:  # pragma: no cover - GPU only
        out["error"] = f"import failed: {exc}"
        out["traceback"] = traceback.format_exc().splitlines()[-4:]
        return out

    adapter = PokeCubeAdapter()
    try:
        runner = PokeCubeEnvRunner()
    except Exception as exc:
        out["error"] = f"runner init failed: {exc}"
        out["traceback"] = traceback.format_exc().splitlines()[-4:]
        return out

    try:
        for d in directions:
            motion = _DIR_TO_MOTION[d]
            runner.set_injection(motion)
            rows = []
            succ = 0
            faces = set()
            skipped = 0
            for s in seeds:
                scene = runner.reset(int(s))
                if max_cube_x is not None and scene.cube_xy[0] > max_cube_x:
                    skipped += 1
                    continue
                if max_approach_dist is not None and _worst_waypoint_dist(
                        scene.cube_xy, scene.goal_xy, directions) > max_approach_dist:
                    skipped += 1
                    continue
                if target_n is not None and len(rows) >= target_n:
                    break
                intent = adapter.oracle_correct_intent(scene)
                faces.add(intent.contact_region)
                attempt = runner.run(intent, scene)
                diag = dict(runner.last_diag)
                succ += int(bool(attempt.success))
                rows.append({
                    "seed": int(s),
                    "success": bool(attempt.success),
                    "reached_contact": bool(attempt.reached_contact),
                    "cube_xy": [round(v, 4) for v in scene.cube_xy],
                    "goal_xy": [round(v, 4) for v in scene.goal_xy],
                    "final_obj_xy": [round(v, 4) for v in attempt.final_obj_xy],
                    "is_peg_grasped": diag.get("is_peg_grasped"),
                    "is_cube_placed": diag.get("is_cube_placed"),
                    "head_to_cube_dist": (round(diag["head_to_cube_dist"], 4)
                                          if diag.get("head_to_cube_dist") is not None else None),
                    "cube_displacement": round(diag.get("cube_displacement", 0.0), 4),
                })
            n = max(1, len(rows))
            out["per_direction"][d] = {
                "motion": motion or "native_+x",
                "oracle_faces": sorted(faces),
                "n": len(rows),
                "n_skipped_unreachable": skipped,
                "success_rate": succ / n,
                "grasp_rate": (sum(1 for r in rows if r["is_peg_grasped"]) / n),
                "contact_rate": (sum(1 for r in rows if r["reached_contact"]) / n),
                "mean_cube_displacement": (
                    sum(r["cube_displacement"] for r in rows) / n),
                "passes_gate": bool(succ / n >= gate),
                "rows": rows,
            }
    finally:
        if hasattr(runner, "close"):
            try:
                runner.close()
            except Exception:
                pass

    pd = out["per_direction"]
    out["plus_x_success"] = pd.get("+x", {}).get("success_rate")
    out["plus_x_passes_gate"] = pd.get("+x", {}).get("passes_gate")
    out["n_directions_passing"] = sum(1 for v in pd.values() if v.get("passes_gate"))
    out["family_core_confirmed"] = bool(out.get("plus_x_passes_gate"))
    out["multi_candidate_confirmed"] = out["n_directions_passing"] >= 2
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", default="0-23", help="Seed range (0-23) or list.")
    p.add_argument("--directions", default="+x",
                   help="Comma list of +x,-x,+y,-y (default +x = the gate floor).")
    p.add_argument("--gate", type=float, default=0.8)
    p.add_argument("--max-cube-x", type=float, default=None,
                   help="Reachability pre-filter: skip seeds whose native "
                        "cube_x exceeds this (far cubes are reach-dead).")
    p.add_argument("--target-n", type=int, default=None,
                   help="Stop after this many reachable seeds (per direction).")
    p.add_argument("--max-approach-dist", type=float, default=None,
                   help="All-directions reachability gate: skip seeds whose "
                        "worst prepoke/poke waypoint distance from the Panda "
                        "base (over all tested directions) exceeds this (~0.80).")
    p.add_argument("--out", type=Path,
                   default=Path("reports/stage5/pokecube_killgate/findings.json"))
    args = p.parse_args(argv)

    seeds = _parse_seeds(args.seeds)
    directions = [d.strip() for d in args.directions.split(",") if d.strip()]
    bad = [d for d in directions if d not in _DIR_TO_MOTION]
    if bad:
        p.error(f"unknown directions {bad}; choose from {sorted(_DIR_TO_MOTION)}")

    print(f"=== PokeCube kill-gate: directions={directions} seeds={args.seeds} "
          f"gate={args.gate} max_cube_x={args.max_cube_x} target_n={args.target_n} ===")
    findings = run_killgate(seeds, directions, args.gate,
                            max_cube_x=args.max_cube_x, target_n=args.target_n,
                            max_approach_dist=args.max_approach_dist)
    # compact console view (drop per-seed rows)
    compact = {k: v for k, v in findings.items() if k != "per_direction"}
    compact["per_direction"] = {
        d: {k: val for k, val in v.items() if k != "rows"}
        for d, v in findings.get("per_direction", {}).items()
    }
    print(json.dumps(compact, indent=2, default=str))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(findings, indent=2, default=str) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
