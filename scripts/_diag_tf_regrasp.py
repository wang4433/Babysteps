"""TurnFaucet oracle MVP-2: center-grasp + axis-aligned spin + RE-GRASP RATCHET.

The embodiment story (user direction 2026-05-29): the source embodiment turns
the faucet with a CONTINUOUS grasp-turn; the Franka wrist (joint-7, +-2.9 rad)
cannot rotate continuously that far, so it must adapt by RE-GRASPING: spin to
the wrist limit, release, rewind the wrist, regrip, spin again -- ratcheting the
handle to target. stiffness=0 means each partial turn persists.

Over arc6 (off-axis rim grasp + 6-DOF pose arc, which slipped/wedged), this uses
the RELIABLE center grasp (grasp+spin already turned vertical knobs ~50% in one
spin) and generalizes the spin to the true joint axis:
    action[3:6] = sign * axis_world * SPIN     (root frame == world; base is
                                                identity-oriented)
For a vertical axis this is action[5] (the working grasp+spin); for a horizontal
axis it commands rotation about world-x/y.

State machine per seed: approach -> descend -> [grip -> spin-until-stall ->
release -> rewind]* until success or budget. Stall = qpos not advancing toward
target over a window (wrist saturated) -> trigger re-grasp.

SCRATCH diagnostic (GPU-only, not imported by package/tests).
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, "/scratch/gilbreth/wang4433/babysteps")

import numpy as np
import gymnasium as gym
import mani_skill.envs  # noqa: F401

from babysteps.render.common import to_np

SEEDS = [int(s) for s in os.environ.get("DIAG_SEEDS", "100,101,102,103,104").split(",")]
MAX_STEPS = int(os.environ.get("DIAG_MAX_STEPS", "600"))
CLEAR = float(os.environ.get("DIAG_CLEAR", "0.12"))
GRIP_Z = float(os.environ.get("DIAG_GRIPZ", "0.0"))
SPIN = float(os.environ.get("DIAG_SPIN", "0.8"))
POS_SCALE = 0.1
ROT_SCALE = 0.1
N_APPROACH = 40
N_DESCEND = 40
N_GRIP = 18
SPIN_STEPS = int(os.environ.get("DIAG_SPINSTEPS", "90"))
STALL_WIN = 25
STALL_EPS = float(os.environ.get("DIAG_STALLEPS", "0.02"))
N_RELEASE = 12
N_REWIND = int(os.environ.get("DIAG_NREWIND", "45"))
N_REDESCEND = 18
REWIND_LIFT = float(os.environ.get("DIAG_REWLIFT", "0.06"))  # lift clear of handle before rewinding
MAX_CYCLES = int(os.environ.get("DIAG_MAXCYC", "6"))


def _f(x):
    return float(to_np(x).item()) if hasattr(x, "item") or hasattr(x, "cpu") else float(x)


def tcp_xyz(obs):
    return to_np(obs["extra"]["tcp_pose"])[0:3].astype(np.float64)


def act(cur_p, tgt_p, rotvec=(0, 0, 0), grip=-1.0):
    a = np.zeros(7, dtype=np.float32)
    a[0:3] = np.clip((tgt_p - cur_p) / POS_SCALE, -1.0, 1.0)
    a[3:6] = np.clip(np.asarray(rotvec) / ROT_SCALE, -1.0, 1.0)
    a[6] = np.float32(grip)
    return a


def _succ(info):
    s = info.get("success", False)
    return bool(to_np(s).item()) if hasattr(s, "cpu") else bool(s)


def run_trial(env, seed, sign, max_steps):
    obs, _ = env.reset(seed=int(seed))
    env_u = env.unwrapped
    sl = env_u.target_switch_link
    target_angle = _f(env_u.target_angle)
    theta_init = _f(sl.joint.qpos)
    needed = target_angle - theta_init
    cmass = to_np(obs["extra"]["target_link_pos"]).astype(np.float64)
    axis = to_np(obs["extra"]["target_joint_axis"]).astype(np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    above = cmass + np.array([0, 0, CLEAR])
    grip_pt = cmass + np.array([0, 0, GRIP_Z])
    spin_vec = sign * axis * SPIN
    rewind_vec = -sign * axis * SPIN

    qpos_ext = theta_init
    success = False
    step = 0
    cycles = 0
    log = []

    def do(target, steps, rotvec, grip):
        nonlocal step, obs, qpos_ext, success
        for _ in range(steps):
            if step >= max_steps:
                return "budget"
            cur = tcp_xyz(obs)
            a = act(cur, target, rotvec, grip)
            obs, _r, term, trunc, info = env.step(a)
            step += 1
            th = _f(sl.joint.qpos)
            qpos_ext = max(qpos_ext, th) if needed > 0 else min(qpos_ext, th)
            if _succ(info):
                success = True
                return "success"
            if (bool(to_np(term).item()) if hasattr(term, "cpu") else bool(term)):
                return "term"
        return "done"

    do(above, N_APPROACH, (0, 0, 0), 1.0)
    do(grip_pt, N_DESCEND, (0, 0, 0), 1.0)

    while step < max_steps and not success and cycles < MAX_CYCLES:
        do(grip_pt, N_GRIP, (0, 0, 0), -1.0)            # (re)grip
        # spin until stall / cap / success
        spin_used = 0
        th_start = _f(sl.joint.qpos)
        recent = [th_start]
        while step < max_steps and not success and spin_used < SPIN_STEPS:
            r = do(grip_pt, 1, spin_vec, -1.0)
            spin_used += 1
            th = _f(sl.joint.qpos)
            recent.append(th)
            if len(recent) > STALL_WIN:
                recent.pop(0)
                # progress toward target over the window
                adv = (recent[-1] - recent[0]) * (1 if needed > 0 else -1)
                if adv < STALL_EPS:
                    break  # wrist saturated -> re-grasp
            if r in ("success", "budget", "term"):
                break
        th_end = _f(sl.joint.qpos)
        log.append((cycles, round(th_start, 3), round(th_end, 3), spin_used))
        if success or step >= max_steps:
            break
        # Re-grasp WITHOUT disturbing the handle: open, lift clear, rewind the
        # wrist up off the handle, then descend back to regrip. (stiffness=0 so
        # the partial turn persists; the prior in-place rewind swept the open
        # fingers across the handle and knocked it back -- grip-loss on 16/36.)
        rewind_pt = grip_pt + np.array([0, 0, REWIND_LIFT])
        do(rewind_pt, N_RELEASE, (0, 0, 0), 1.0)        # open + lift clear
        do(rewind_pt, N_REWIND, rewind_vec, 1.0)        # rewind wrist (clear of handle)
        do(grip_pt, N_REDESCEND, (0, 0, 0), 1.0)        # descend back to handle (open)
        cycles += 1

    final = _f(sl.joint.qpos)
    prog = (qpos_ext - theta_init) / needed if abs(needed) > 1e-6 else 0.0
    return {"success": success, "sign": sign, "steps": step,
            "theta_init": theta_init, "final": final, "target": target_angle,
            "needed": needed, "qpos_ext": qpos_ext, "progress": prog,
            "axis": axis.tolist(), "cycles": cycles + 1, "cyclelog": log}


def run_seed(env, seed):
    r1 = run_trial(env, seed, +1, MAX_STEPS)
    if r1["success"] or r1["progress"] > 0.4:
        return {"seed": seed, "winning_sign": +1, **r1}
    r2 = run_trial(env, seed, -1, MAX_STEPS)
    best = r1 if r1["progress"] >= r2["progress"] else r2
    return {"seed": seed, "winning_sign": best["sign"], **best}


def main():
    import json
    from pathlib import Path

    seed_start = os.environ.get("DIAG_SEED_START")
    seeds = (list(range(int(seed_start), int(os.environ["DIAG_SEED_END"]) + 1))
             if seed_start else SEEDS)
    out_dir = os.environ.get("DIAG_OUT", "")
    env = gym.make("TurnFaucet-v1", obs_mode="state_dict",
                   control_mode="pd_ee_delta_pose", sim_backend="gpu",
                   max_episode_steps=MAX_STEPS + 5)
    print(f"[regrasp] n_seeds={len(seeds)} max_steps={MAX_STEPS} spin={SPIN} "
          f"spinsteps={SPIN_STEPS} nrewind={N_REWIND} maxcyc={MAX_CYCLES}", flush=True)
    results = []
    f = None
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        f = open(Path(out_dir) / "per_seed.jsonl", "w")
    for s in seeds:
        try:
            r = run_seed(env, s)
        except Exception as e:
            import traceback; traceback.print_exc()
            r = {"seed": s, "success": False, "error": str(e)}
        results.append(r)
        if f is not None:
            f.write(json.dumps({k: v for k, v in r.items() if k != "cyclelog"}) + "\n")
            f.flush()
        if "error" in r:
            print(f"  seed {s}: ERROR {r['error']}", flush=True)
        else:
            ax = r["axis"]
            print(f"  seed {s:3d}: success={r['success']!s:5} sign={r['winning_sign']:+d} "
                  f"steps={r['steps']:3d} cyc={r['cycles']} "
                  f"axis=({ax[0]:+.2f},{ax[1]:+.2f},{ax[2]:+.2f}) "
                  f"qpos {r['theta_init']:+.2f}->{r['final']:+.2f} target={r['target']:+.2f} "
                  f"prog={100*r['progress']:5.1f}%", flush=True)
            print(f"           cyclelog: {r['cyclelog']}", flush=True)
    env.close()
    if f is not None:
        f.close()
    n = sum(1 for r in results if r.get("success"))
    half = sum(1 for r in results if r.get("progress", 0) > 0.5 and not r.get("success"))
    print(f"\n=== REGRASP SUMMARY: {n}/{len(results)} success "
          f"({n/len(results):.0%}), {half}/{len(results)} >=50% (non-success) ===", flush=True)
    if out_dir:
        Path(out_dir, "summary.md").write_text(
            f"# TurnFaucet re-grasp ratchet oracle\n\n"
            f"{n}/{len(results)} success ({n/len(results):.0%}); "
            f"{half}/{len(results)} >=50% progress (non-success).\n\n"
            f"Params: spin={SPIN} spinsteps={SPIN_STEPS} nrewind={N_REWIND} "
            f"maxcyc={MAX_CYCLES} max_steps={MAX_STEPS}.\n"
            f"Baseline poke: 2/50 (4%). Per-seed: per_seed.jsonl.\n")


if __name__ == "__main__":
    main()
