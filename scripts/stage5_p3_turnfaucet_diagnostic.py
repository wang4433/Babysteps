"""Stage-5 P3 diagnostic: poke_turn failure-mode breakdown.

Re-runs the production TurnFaucetEnvRunner poke_turn dispatch (probe →
maybe full → maybe sign-retry) on seeds 100-149 (the held-out cut used by
P2 and M3) and categorizes failures so we can decide if the 4% success
rate on this band is fixable in ≤3 days or whether TurnFaucet should
drop to the appendix per redesign_failure_paradigm.md §"Phase 3".

Outputs:
  reports/stage5/turnfaucet_diagnostic/per_seed.jsonl  one JSON line per seed
  reports/stage5/turnfaucet_diagnostic/summary.md       category counts + notes

Categories (ordered by precedence on inspection):
  - success                       : info['success'] == True on the chosen trial
  - no_contact                    : reached_contact = False on every trial
  - contact_no_motion             : reached_contact = True, |qpos delta| < 0.05 rad
  - partial_rotation              : 0 < |progress| < 0.5
  - mostly_rotated                : 0.5 <= |progress| < 0.95
  - near_success_no_termination   : |progress| >= 0.95 but info['success'] = False
  - exception                     : seed raised
Plus, recorded but not in primary category: `arm_qpos_near_limit` flag
based on the final robot qpos.

Sim-free / login-node usable? **No.** Requires GPU+Vulkan (TurnFaucet
uses sim_backend='gpu'). Run via slurm/stage5_p3_turnfaucet_diagnostic.sbatch.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np


_REPO = "/scratch/gilbreth/wang4433/babysteps"
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def _to_np(x):
    arr = x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)
    return arr[0] if arr.ndim == 2 else arr


# Franka Panda joint limits (lower, upper) for arm joints 0..6. The two
# finger joints (indices 7-8) are gripper actuators and not relevant for
# arm-collapse detection.
_PANDA_ARM_LIMITS_RAD = (
    (-2.8973,  2.8973),
    (-1.7628,  1.7628),
    (-2.8973,  2.8973),
    (-3.0718, -0.0698),
    (-2.8973,  2.8973),
    (-0.0175,  3.7525),
    (-2.8973,  2.8973),
)
_ARM_LIMIT_MARGIN_RAD = 0.10  # within 0.10 rad of a hard limit → "near limit"


def _arm_near_limit(qpos) -> dict:
    arm = list(qpos)[:7]
    flags = []
    for i, (q, (lo, hi)) in enumerate(zip(arm, _PANDA_ARM_LIMITS_RAD)):
        if q <= lo + _ARM_LIMIT_MARGIN_RAD:
            flags.append({"joint": i, "qpos": float(q), "limit": float(lo), "side": "lower"})
        elif q >= hi - _ARM_LIMIT_MARGIN_RAD:
            flags.append({"joint": i, "qpos": float(q), "limit": float(hi), "side": "upper"})
    return {"any": bool(flags), "joints": flags}


def _categorize(per_seed: dict) -> str:
    if per_seed.get("error"):
        return "exception"
    if per_seed["final_success"]:
        return "success"
    reached_contact = per_seed["best_reached_contact"]
    progress = abs(per_seed["best_progress"])
    if not reached_contact:
        return "no_contact"
    if not per_seed["best_object_moved"]:
        return "contact_no_motion"
    if progress < 0.5:
        return "partial_rotation"
    if progress < 0.95:
        return "mostly_rotated"
    return "near_success_no_termination"


def _trial_dict(name: str, sign: int, max_steps: int, outcome) -> dict:
    return {
        "name": name,
        "sign": int(sign),
        "max_steps": int(max_steps),
        "success": bool(outcome.success),
        "progress": float(outcome.qpos_extremum_signed_progress),
        "reached_contact": bool(outcome.reached_contact),
        "object_moved": bool(outcome.object_moved),
    }


def run_seed(env, seed: int, scene, intent, _execute_skill, compile_skill,
             *, probe_steps: int, max_steps: int, probe_min_progress: float):
    """Replicate TurnFaucetEnvRunner.run dispatch for poke_turn, with logging."""
    from babysteps.envs.turnfaucet_runner import _read_needed_delta

    needed_delta = _read_needed_delta(env)
    contact_xy = np.asarray(scene.extra["handle_xy"], dtype=np.float64)

    skill_pos = compile_skill(intent, scene, sign=+1)

    trials = []

    probe = _execute_skill(
        env, skill_pos, seed=seed, needed_delta=needed_delta,
        contact_xy=contact_xy, max_steps=probe_steps,
    )
    trials.append(_trial_dict("probe_pos", +1, probe_steps, probe))
    chosen = probe

    if probe.success:
        pass
    elif probe.qpos_extremum_signed_progress >= probe_min_progress:
        full_pos = _execute_skill(
            env, skill_pos, seed=seed, needed_delta=needed_delta,
            contact_xy=contact_xy, max_steps=max_steps,
        )
        trials.append(_trial_dict("full_pos", +1, max_steps, full_pos))
        chosen = full_pos
    else:
        skill_neg = compile_skill(intent, scene, sign=-1)
        full_neg = _execute_skill(
            env, skill_neg, seed=seed, needed_delta=needed_delta,
            contact_xy=contact_xy, max_steps=max_steps,
        )
        trials.append(_trial_dict("full_neg", -1, max_steps, full_neg))
        if (full_neg.success
                or full_neg.qpos_extremum_signed_progress
                    > probe.qpos_extremum_signed_progress):
            chosen = full_neg
        else:
            full_pos2 = _execute_skill(
                env, skill_pos, seed=seed, needed_delta=needed_delta,
                contact_xy=contact_xy, max_steps=max_steps,
            )
            trials.append(_trial_dict("full_pos_fallback", +1, max_steps, full_pos2))
            chosen = full_pos2

    # End-of-episode env state (from the chosen / last-executed trial).
    final_robot_qpos = _to_np(env.unwrapped.agent.robot.qpos).tolist()
    final_faucet_qpos = float(_to_np(env.unwrapped.target_switch_link.joint.qpos).item())
    target_angle = float(_to_np(env.unwrapped.target_angle).item())
    arm_limit_check = _arm_near_limit(final_robot_qpos)

    # "best" trial = the one chosen by the dispatch (matches production AttemptResult).
    return {
        "seed": int(seed),
        "needed_delta_rad": float(needed_delta),
        "target_angle_rad": float(target_angle),
        "final_faucet_qpos_rad": float(final_faucet_qpos),
        "trials": trials,
        "final_success": bool(chosen.success),
        "best_progress": float(chosen.qpos_extremum_signed_progress),
        "best_reached_contact": bool(chosen.reached_contact),
        "best_object_moved": bool(chosen.object_moved),
        "final_robot_qpos": [float(q) for q in final_robot_qpos],
        "arm_near_limit": arm_limit_check,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed_start", type=int, default=100)
    p.add_argument("--seed_end", type=int, default=149)  # inclusive
    p.add_argument("--out_dir", type=str,
                   default="reports/stage5/turnfaucet_diagnostic")
    args = p.parse_args()

    import gymnasium as gym
    import mani_skill.envs  # noqa: F401 — registers TurnFaucet-v1
    from babysteps.envs.turnfaucet_runner import (
        _execute_skill, _read_obs, _poke_geometry_extra,
        _POKE_PROBE_STEPS, _MAX_CONTROL_STEPS, _POKE_PROBE_MIN_PROGRESS,
    )
    from babysteps.skills.turn import compile_intent_to_turn_skill
    from babysteps.schemas import Intent, SceneState

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_seed_path = out_dir / "per_seed.jsonl"
    summary_path = out_dir / "summary.md"

    seeds = list(range(args.seed_start, args.seed_end + 1))
    print(f"[diag] running {len(seeds)} seeds ({args.seed_start}..{args.seed_end})", flush=True)

    env = gym.make(
        "TurnFaucet-v1", obs_mode="state_dict",
        control_mode="pd_ee_delta_pose", sim_backend="gpu",
    )

    poke_intent = Intent(
        goal_state="faucet_turned",
        object_motion="turn",
        contact_region="handle_grip",
        approach_direction="from_above",
        constraint_region="faucet_base_static",
        embodiment_mapping="proxy_contact_to_franka_poke_turn",
    )

    results: list[dict] = []
    start = time.time()

    with per_seed_path.open("w") as f:
        for i, seed in enumerate(seeds):
            t0 = time.time()
            try:
                # Build scene from a fresh reset (matches production
                # TurnFaucetEnvRunner.reset()).
                obs, _ = env.reset(seed=int(seed))
                tcp, handle_xyz, axis_xyz = _read_obs(obs)
                handle_xy = (float(handle_xyz[0]), float(handle_xyz[1]))
                handle_z = float(handle_xyz[2])
                axis_xy = (float(axis_xyz[0]), float(axis_xyz[1]))
                extra = {
                    "handle_xy": handle_xy,
                    "handle_z": handle_z,
                    "target_joint_axis_xy": axis_xy,
                }
                # v1 poke geometry (OBB centre + true tangent) — identical to
                # TurnFaucetEnvRunner.reset() so this diagnostic stays a faithful
                # reproduction of the production poke dispatch.
                extra.update(_poke_geometry_extra(env, obs))
                scene = SceneState(
                    cube_xy=handle_xy, cube_z=handle_z, goal_xy=handle_xy,
                    tcp_start_pose=tuple(float(v) for v in tcp),
                    blocked_sides=(),
                    extra=extra,
                )
                per_seed = run_seed(
                    env, seed, scene, poke_intent,
                    _execute_skill, compile_intent_to_turn_skill,
                    probe_steps=_POKE_PROBE_STEPS,
                    max_steps=_MAX_CONTROL_STEPS,
                    probe_min_progress=_POKE_PROBE_MIN_PROGRESS,
                )
            except Exception as e:
                per_seed = {
                    "seed": int(seed),
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
            per_seed["category"] = _categorize(per_seed)
            per_seed["wall_s"] = round(time.time() - t0, 2)
            results.append(per_seed)
            f.write(json.dumps(per_seed) + "\n")
            f.flush()
            elapsed = time.time() - start
            avg = elapsed / (i + 1)
            eta = avg * (len(seeds) - i - 1)
            arm_flag = "*" if per_seed.get("arm_near_limit", {}).get("any") else " "
            print(
                f"  [{i+1:>2}/{len(seeds)}] seed {seed:3d}: "
                f"category={per_seed['category']:>28s} {arm_flag}  "
                f"({elapsed/60:.1f}m elapsed, ETA {eta/60:.1f}m)",
                flush=True,
            )

    try:
        env.close()
    except Exception:
        pass

    cats = Counter(r["category"] for r in results)
    arm_near_limit_count = sum(
        1 for r in results if r.get("arm_near_limit", {}).get("any")
    )
    total = len(results)

    order = [
        "success",
        "no_contact",
        "contact_no_motion",
        "partial_rotation",
        "mostly_rotated",
        "near_success_no_termination",
        "exception",
    ]
    lines = [
        "# TurnFaucet poke_turn diagnostic — failure-mode breakdown",
        "",
        f"Seeds {args.seed_start}..{args.seed_end} ({total} total).",
        f"Source: `scripts/stage5_p3_turnfaucet_diagnostic.py`.",
        f"Dispatch: probe(+1, {_POKE_PROBE_STEPS} steps) → if progress ≥ "
        f"{_POKE_PROBE_MIN_PROGRESS:.2f} full(+1, {_MAX_CONTROL_STEPS}) → "
        f"else full(-1, {_MAX_CONTROL_STEPS}) → fallback full(+1, {_MAX_CONTROL_STEPS}).",
        "",
        "## Category counts",
        "",
        "| category | count | rate |",
        "| --- | --- | --- |",
    ]
    for cat in order:
        if cat in cats:
            n = cats[cat]
            lines.append(f"| {cat} | {n} | {n/total:.0%} |")
    # Any category not in `order` (defensive).
    for cat, n in cats.items():
        if cat not in order:
            lines.append(f"| {cat} | {n} | {n/total:.0%} |")
    lines.extend([
        "",
        f"`arm_near_limit` flag (final robot qpos within "
        f"{_ARM_LIMIT_MARGIN_RAD:.2f} rad of any hard limit): "
        f"{arm_near_limit_count} / {total} "
        f"({arm_near_limit_count/total:.0%}).",
        "",
        "## How to read this",
        "",
        "- `success` — what we want; trial reached `info['success']=True`.",
        "- `no_contact` — gripper never reached the handle. Fix: improve "
        "handle-localization / waypoint geometry (the `compute_geometry` "
        "helper in `scripts/_diag_tf_poke5.py` is the empirical reference).",
        "- `contact_no_motion` — reached handle, qpos didn't move > 0.05 rad. "
        "Fix: contact force / sign / EE pose during sweep.",
        "- `partial_rotation` — pushed the handle but stalled. Fix: longer "
        "sweep distance, multi-step contact, lower friction at TCP.",
        "- `mostly_rotated` / `near_success_no_termination` — close to target; "
        "may just need a longer trial budget or a tighter success threshold.",
        "- `arm_near_limit` flag — separate signal; co-occurs with any of "
        "the above when the IK has wedged the arm against a joint stop. If "
        "this co-occurs with `no_contact` it is the most-likely root cause.",
        "",
        "## Decision gate (per redesign_failure_paradigm.md §\"Phase 3\")",
        "",
        "If the top-1 category points to a fix that is plausibly ≤3 days of "
        "engineering (waypoint geometry, force/direction, trial budget), "
        "proceed with Phase 3. Otherwise drop TurnFaucet to appendix and "
        "use the 4-task paper narrative.",
        "",
        "Per-seed details: see `per_seed.jsonl`.",
    ])
    summary_path.write_text("\n".join(lines))

    print("\n=== SUMMARY ===")
    for cat in order + sorted(c for c in cats if c not in order):
        if cat in cats:
            n = cats[cat]
            print(f"  {cat:30s}  {n:3d}  ({n/total:.0%})")
    print(f"  arm_near_limit (any cat)         {arm_near_limit_count:3d}  "
          f"({arm_near_limit_count/total:.0%})")
    print(f"\nWrote {per_seed_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
