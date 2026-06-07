"""Stage-5 step-3 GPU probe — verify PokeCube mechanics + PushCube 4-face
feasibility BEFORE writing the grasp+poke runner.

Two questions the feasibility study could not answer on the login node
(mani_skill is GPU-only):

PART 1 — PokeCube-v1 mechanics. Is ``PokeCube-v1`` a real ManiSkill 3 env? If
so: does the peg start grasped or must the robot grasp it; is the goal a radius
region; is the goal pose fixed across seeds (like PushCube's native +x) so the
goal-move injection is required; what are the obs keys / action space. If the
env id is wrong, list the registered Poke/Push/Peg envs so we use the real name.

PART 2 — PushCube 4-face feasibility (the user chose "enrich to 4 faces"). Run
the EXISTING PushCubeEnvRunner(orient_control=True) with the oracle contact face
for each of the 4 injected push directions and report per-face success. If
orient_control cannot reliably push all 4 faces, the shared contact_region rule
the LOTO tests is only +x/-x (near-trivial) and we must fix the controller (or
narrow the claim) before the GPU LOTO eval.

GPU/Vulkan only. Import of mani_skill is deferred to main() so this file imports
on the login node.

Example::

    python scripts/stage5_pokecube_probe.py --seeds 0-9 \\
        --out reports/stage5/pokecube_probe/findings.json
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _parse_seeds(s: str) -> list[int]:
    if "-" in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",") if x]


def _jsonable(x):
    if isinstance(x, np.ndarray):
        return [float(v) for v in x.reshape(-1)[:8]]
    if hasattr(x, "tolist"):
        try:
            return _jsonable(np.asarray(x))
        except Exception:
            return str(x)
    return x


def probe_pokecube(seeds: list[int]) -> dict:
    """PART 1 — inspect PokeCube-v1 mechanics (best-effort, never raises)."""
    out: dict = {"env_id": "PokeCube-v1"}
    try:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401  (registers tasks)
    except Exception as exc:  # pragma: no cover - GPU only
        out["error"] = f"import failed: {exc}"
        return out

    # Find the real env id if PokeCube-v1 isn't registered verbatim.
    try:
        all_ids = sorted(gym.envs.registry.keys())
        out["related_env_ids"] = [
            k for k in all_ids
            if any(t in k for t in ("Poke", "Push", "Peg"))]
    except Exception as exc:
        out["registry_error"] = str(exc)

    try:
        env = gym.make("PokeCube-v1", obs_mode="state_dict",
                       control_mode="pd_ee_delta_pose", sim_backend="cpu")
    except Exception as exc:
        out["make_error"] = f"{exc}"
        out["make_traceback"] = traceback.format_exc().splitlines()[-3:]
        return out

    try:
        out["action_space"] = str(env.action_space)
        u = env.unwrapped
        out["unwrapped_attrs"] = [
            a for a in dir(u)
            if any(t in a.lower() for t in ("peg", "cube", "goal", "obj"))
            and not a.startswith("__")]
        rows = []
        for s in seeds:
            obs, info = env.reset(seed=int(s))
            rec: dict = {"seed": s}
            # obs is a (possibly nested) state_dict; capture a few key fields.
            if isinstance(obs, dict):
                rec["obs_keys"] = sorted(obs.keys())
                extra = obs.get("extra", {})
                if isinstance(extra, dict):
                    rec["extra_keys"] = sorted(extra.keys())
                    for k in ("obj_pose", "peg_pose", "goal_pos", "tcp_pose",
                              "cube_pose", "peg_head_pose"):
                        if k in extra:
                            rec[k] = _jsonable(extra[k])
            for attr in ("peg", "cube", "goal_region", "obj"):
                o = getattr(u, attr, None)
                pose = getattr(o, "pose", None) if o is not None else None
                p = getattr(pose, "p", None) if pose is not None else None
                if p is not None:
                    rec[f"{attr}_p"] = _jsonable(p)
            rows.append(rec)
        out["per_seed"] = rows
        # Is the goal/peg fixed across seeds? (PushCube's native goal is fixed.)
        out["note"] = ("inspect per_seed obj/peg/goal poses: if goal_pos is "
                       "identical across seeds, goal-move injection is required "
                       "(as for PushCube).")
    except Exception as exc:
        out["inspect_error"] = f"{exc}"
        out["inspect_traceback"] = traceback.format_exc().splitlines()[-3:]
    finally:
        try:
            env.close()
        except Exception:
            pass
    return out


def probe_pushcube_4face(seeds: list[int]) -> dict:
    """PART 2 — per-face PushCube success under orient_control (4-face enrich)."""
    out: dict = {"orient_control": True, "per_face": {}}
    try:
        from babysteps.envs.pushcube_adapter import PushCubeAdapter
        from babysteps.envs.pushcube_runner import PushCubeEnvRunner
    except Exception as exc:  # pragma: no cover - GPU only
        out["error"] = f"import failed: {exc}"
        return out

    motions = {
        "plus_x": "translate_+x", "minus_x": "translate_-x",
        "plus_y": "translate_+y", "minus_y": "translate_-y",
    }
    adapter = PushCubeAdapter()
    try:
        runner = PushCubeEnvRunner(orient_control=True)
    except Exception as exc:
        out["error"] = f"runner init failed: {exc}"
        return out

    try:
        for label, motion in motions.items():
            runner.set_injection(motion)
            succ = 0
            faces = set()
            for s in seeds:
                scene = runner.reset(s)
                intent = adapter.oracle_correct_intent(scene)
                faces.add(intent.contact_region)
                attempt = runner.run(intent, scene)
                succ += int(bool(attempt.success))
            out["per_face"][label] = {
                "motion": motion, "n": len(seeds),
                "success_rate": succ / max(1, len(seeds)),
                "oracle_faces": sorted(faces),
            }
    finally:
        if hasattr(runner, "close"):
            try:
                runner.close()
            except Exception:
                pass
    rates = [v["success_rate"] for v in out["per_face"].values()]
    out["min_face_success"] = min(rates) if rates else None
    out["all_faces_ok_at_0.8"] = bool(rates and min(rates) >= 0.8)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", default="0-9", help="Seed range, e.g. 0-9.")
    p.add_argument("--out", type=Path,
                   default=Path("reports/stage5/pokecube_probe/findings.json"))
    p.add_argument("--skip-pokecube", action="store_true")
    p.add_argument("--skip-pushcube", action="store_true")
    args = p.parse_args(argv)
    seeds = _parse_seeds(args.seeds)

    findings: dict = {"seeds": args.seeds}
    if not args.skip_pokecube:
        print("=== PART 1: PokeCube-v1 mechanics ===")
        findings["pokecube"] = probe_pokecube(seeds)
        print(json.dumps(findings["pokecube"], indent=2, default=str))
    if not args.skip_pushcube:
        print("\n=== PART 2: PushCube 4-face feasibility (orient_control) ===")
        findings["pushcube_4face"] = probe_pushcube_4face(seeds)
        print(json.dumps(findings["pushcube_4face"], indent=2, default=str))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(findings, indent=2, default=str) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
