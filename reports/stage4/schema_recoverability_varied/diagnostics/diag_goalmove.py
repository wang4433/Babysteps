"""Goal-move diagnostic: keep cube native, move the goal. Reports per
seed x direction whether the cube moves in the right direction. Also discovers
the goal-region handle up front (printed before the runner is used, so we learn
the right attribute name even if the runner's goal_region access crashes)."""
import sys
sys.path.insert(0, "/scratch/gilbreth/wang4433/babysteps")
import numpy as np

# --- 1) discover the goal handle on a bare env (printed first, flushed) ---
import gymnasium as gym
import mani_skill.envs  # noqa: F401
e = gym.make("PushCube-v1", obs_mode="state_dict",
             control_mode="pd_ee_delta_pose", sim_backend="cpu")
e.reset(seed=0)
u = e.unwrapped
print("goal-ish attrs:", [a for a in dir(u) if "goal" in a.lower()], flush=True)
gr = getattr(u, "goal_region", None)
print("goal_region type:", type(gr).__name__ if gr is not None else None, flush=True)
e.close()

# --- 2) test goal-move via the runner ---
from babysteps.envs.pushcube_runner import PushCubeEnvRunner
from babysteps.envs.pushcube_adapter import PushCubeAdapter
from babysteps.demo import trajectory_to_motion

r = PushCubeEnvRunner()
a = PushCubeAdapter()
dirs = ("translate_+x", "translate_-x", "translate_+y", "translate_-y")
hdr = (f"{'seed':>4} {'target':>12} {'intent':>12} {'succ':>5} {'moved':>5} "
       f"{'dx':>7} {'dy':>7} {'traj_motion':>12} {'dir_ok':>6} "
       f"{'cube_init':>20} {'goal':>20}")
print(hdr, flush=True)
print("-" * len(hdr), flush=True)
for seed in (0, 1, 2, 7, 13):
    for m in dirs:
        r.set_injection(m)
        s = r.reset(seed)
        i = a.oracle_correct_intent(s)
        res = r.run(i, s)
        init = np.array(res.initial_obj_xy)
        fin = np.array(res.final_obj_xy)
        disp = fin - init
        traj = res.trajectory_xy
        tm = trajectory_to_motion(traj) if len(traj) >= 2 else "NA"
        dir_ok = (tm == m and res.object_moved)
        ci = tuple(round(float(v), 3) for v in s.cube_xy)
        g = tuple(round(float(v), 3) for v in s.goal_xy)
        print(f"{seed:>4} {m:>12} {i.object_motion:>12} {str(res.success):>5} "
              f"{str(res.object_moved):>5} {disp[0]:>7.3f} {disp[1]:>7.3f} "
              f"{tm:>12} {str(dir_ok):>6} {str(ci):>20} {str(g):>20}", flush=True)
r.close()
print("DIAG DONE", flush=True)
