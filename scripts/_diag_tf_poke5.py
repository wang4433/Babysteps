"""TurnFaucet diag v5: v1 BRUTE-FORCE single-waypoint sweep + auto sign.

v4 found that multi-step sub-waypoints slow the gripper (per-step action
drops from ~1.0 to ~0.3) which removes the impulse force that v1 relied
on. Return to v1's single-waypoint LONG sweep and add per-seed auto sign
detection with an EARLY-EXIT probe: if qpos hasn't moved toward target
by PROBE_MIN_DELTA within PROBE_STEPS, abort and retry with opposite sign.

This is the cleanest test of the v1 mechanism + sign detection. If it
doesn't get >= 2/5 seeds to info['success'], we're at the empirical
ceiling and need to relax the acceptance gate.
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, "/scratch/gilbreth/wang4433/babysteps")

import numpy as np
import gymnasium as gym
import mani_skill.envs  # noqa: F401
from mani_skill.utils.geometry.trimesh_utils import get_component_mesh

from babysteps.render.common import to_np, prop_action

SEEDS = [int(s) for s in os.environ.get("DIAG_SEEDS", "0,1,2,3,4").split(",")]
PHASE_TOL_M = 0.015
MAX_STEPS_PER_TRIAL = 220

LATERAL_OFFSET_M = 0.07
SWEEP_DISTANCE_M = 0.22   # large sweep, single waypoint, brute force
HEIGHT_ABOVE_HANDLE_M = 0.04
HIGH_CLEARANCE_M = 0.12
PROBE_STEPS = 80
PROBE_MIN_DELTA = 0.15


def get_handle_obb_world(switch_link):
    comp = switch_link._objs[0]
    mesh_local = get_component_mesh(comp, to_world_frame=False)
    if mesh_local is None:
        raise RuntimeError("no handle mesh")
    obb_local = mesh_local.bounding_box_oriented
    link_T_batched = switch_link.pose.to_transformation_matrix()
    link_T = link_T_batched[0].cpu().numpy() if hasattr(link_T_batched, "cpu") else np.asarray(link_T_batched)[0]
    obb_T_world = link_T @ np.array(obb_local.primitive.transform)
    return obb_T_world, np.array(obb_local.primitive.extents)


def compute_geometry(switch_link, obs):
    obb_T, _extents = get_handle_obb_world(switch_link)
    handle_center = obb_T[:3, 3]
    joint_anchor = to_np(switch_link.joint.get_global_pose().p)
    joint_axis = to_np(obs["extra"]["target_joint_axis"])
    jn = float(np.linalg.norm(joint_axis))
    if jn < 1e-6:
        raise RuntimeError("degenerate joint axis")
    joint_axis = joint_axis / jn
    radius = handle_center - joint_anchor
    tangent_3d = np.cross(joint_axis, radius)
    tn = float(np.linalg.norm(tangent_3d))
    if tn < 1e-4:
        tangent_3d = np.array([0.0, 1.0, 0.0])
    else:
        tangent_3d = tangent_3d / tn
    return (
        handle_center[0:2].astype(np.float64),
        tangent_3d[0:2].astype(np.float64),
        float(handle_center[2]),
    )


def build_brute_waypoints(handle_xy, tangent_xy, handle_z, tcp_init_z, sign):
    """v1-style brute-force: 4 waypoints, single sweep at max action."""
    sweep_dir = tangent_xy * sign
    contact_z = handle_z + HEIGHT_ABOVE_HANDLE_M
    approach_z = max(tcp_init_z, handle_z + HIGH_CLEARANCE_M) + 0.02
    pre_xy = handle_xy - sweep_dir * LATERAL_OFFSET_M
    post_xy = handle_xy + sweep_dir * SWEEP_DISTANCE_M
    waypoints = [
        np.array([pre_xy[0], pre_xy[1], approach_z]),
        np.array([pre_xy[0], pre_xy[1], contact_z]),
        np.array([post_xy[0], post_xy[1], contact_z]),   # single brute sweep
    ]
    return waypoints, approach_z, contact_z


def run_trial(seed, sign, enable_early_exit=True):
    env = gym.make(
        "TurnFaucet-v1", obs_mode="state_dict",
        control_mode="pd_ee_delta_pose",
        sim_backend="gpu", render_mode="rgb_array",
    )
    obs, _ = env.reset(seed=int(seed))
    env_u = env.unwrapped
    switch_link = env_u.target_switch_link
    target_angle = float(to_np(env_u.target_angle).item())
    initial_qpos = float(to_np(switch_link.joint.qpos).item())
    needed_delta = target_angle - initial_qpos

    handle_xy, tangent_xy, handle_z = compute_geometry(switch_link, obs)
    tcp_raw0 = to_np(obs["extra"]["tcp_pose"])
    tcp_init_z = float(tcp_raw0[2])
    waypoints, approach_z, contact_z = build_brute_waypoints(
        handle_xy, tangent_xy, handle_z, tcp_init_z, sign,
    )

    phase_idx = 0
    qpos_extremum = initial_qpos
    success = False
    last_step = 0
    aborted = False

    for step in range(MAX_STEPS_PER_TRIAL):
        last_step = step
        tcp_raw = to_np(obs["extra"]["tcp_pose"])
        tcp_xyz = tcp_raw[:3]
        target = waypoints[phase_idx]
        dist = float(np.linalg.norm(target - tcp_xyz))
        if dist < PHASE_TOL_M:
            phase_idx += 1
            if phase_idx >= len(waypoints):
                break
            target = waypoints[phase_idx]
        tcp_xyzw = np.concatenate([tcp_raw[0:3], tcp_raw[4:7], tcp_raw[3:4]])
        action = prop_action(tcp_xyzw, target, gripper_cmd=-1.0)
        obs, _r, term, trunc, info = env.step(action)
        qpos = float(to_np(switch_link.joint.qpos).item())
        if needed_delta > 0:
            qpos_extremum = max(qpos_extremum, qpos)
        else:
            qpos_extremum = min(qpos_extremum, qpos)
        succ = info.get("success", False)
        succ_b = bool(to_np(succ).item()) if hasattr(succ, "cpu") else bool(succ)
        if succ_b:
            success = True
            break

        # Early exit: if probe phase complete and minimum progress not made,
        # abort to save GPU time for the sign retry.
        if enable_early_exit and step == PROBE_STEPS:
            probe_progress = (qpos_extremum - initial_qpos) / needed_delta if abs(needed_delta) > 1e-6 else 0.0
            if probe_progress < PROBE_MIN_DELTA / abs(needed_delta):
                aborted = True
                break

    final_qpos = float(to_np(switch_link.joint.qpos).item())
    progress = (qpos_extremum - initial_qpos) / needed_delta if abs(needed_delta) > 1e-6 else 0.0
    env.close()

    return {
        "success": success, "aborted": aborted,
        "qpos_init": initial_qpos, "qpos_final": final_qpos,
        "qpos_extremum": qpos_extremum, "target_angle": target_angle,
        "needed_delta": needed_delta, "steps": last_step + 1, "progress": progress,
        "handle_xy": handle_xy.tolist(), "tangent_xy": tangent_xy.tolist(),
        "handle_z": handle_z, "approach_z": approach_z, "contact_z": contact_z,
    }


def run_seed(seed):
    print(f"=== seed {seed} ===")
    r1 = run_trial(seed, sign=+1, enable_early_exit=True)
    print(f"  sign=+1: handle=({r1['handle_xy'][0]:+.3f},{r1['handle_xy'][1]:+.3f},{r1['handle_z']:+.3f}) "
          f"tangent=({r1['tangent_xy'][0]:+.3f},{r1['tangent_xy'][1]:+.3f}) "
          f"approach_z={r1['approach_z']:.3f} contact_z={r1['contact_z']:.3f}")
    print(f"    qpos {r1['qpos_init']:+.3f}->{r1['qpos_final']:+.3f}, "
          f"extremum={r1['qpos_extremum']:+.3f}, target={r1['target_angle']:+.3f}, "
          f"progress={100*r1['progress']:.1f}%, success={r1['success']}, "
          f"aborted={r1['aborted']}, steps={r1['steps']}")

    if r1["success"] or r1["progress"] > 0.7:
        return {"seed": seed, "winning_sign": +1, **r1}

    print(f"  sign=+1 made {100*r1['progress']:.1f}%; trying sign=-1...")
    r2 = run_trial(seed, sign=-1, enable_early_exit=False)
    print(f"  sign=-1:")
    print(f"    qpos {r2['qpos_init']:+.3f}->{r2['qpos_final']:+.3f}, "
          f"extremum={r2['qpos_extremum']:+.3f}, target={r2['target_angle']:+.3f}, "
          f"progress={100*r2['progress']:.1f}%, success={r2['success']}, "
          f"steps={r2['steps']}")

    if r2["progress"] > r1["progress"]:
        return {"seed": seed, "winning_sign": -1, **r2}
    else:
        return {"seed": seed, "winning_sign": +1, **r1}


def main():
    results = []
    for s in SEEDS:
        try:
            r = run_seed(s)
            results.append(r)
        except Exception as e:
            import traceback
            print(f"!! seed {s} FAILED: {e}")
            traceback.print_exc()
            results.append({"seed": s, "success": False, "error": str(e)})
        print()

    n_success = sum(1 for r in results if r.get("success"))
    n_partial = sum(1 for r in results if r.get("progress", 0) > 0.5 and not r.get("success"))
    print(f"=== SUMMARY: {n_success}/{len(results)} info['success'], {n_partial}/{len(results)} >=50% progress ===")
    for r in results:
        if "error" in r:
            print(f"  seed {r['seed']}: ERROR {r['error']}")
        else:
            print(f"  seed {r['seed']:2d}: success={r['success']:<5} winning_sign={r.get('winning_sign'):+d}  "
                  f"qpos {r['qpos_init']:+.3f}->{r['qpos_final']:+.3f}  "
                  f"target={r['target_angle']:+.3f}  extremum={r['qpos_extremum']:+.3f}  "
                  f"({100*r['progress']:5.1f}% of needed delta)")


if __name__ == "__main__":
    main()
