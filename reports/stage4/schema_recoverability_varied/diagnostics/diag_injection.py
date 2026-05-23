"""Diagnostic: does the cube-move injection MOVE the cube in the right
direction (probe-usable), or not move at all (broken)? Reports movement, not
just success. Sim-free firewall N/A — this is a GPU diagnostic."""
import sys
sys.path.insert(0, "/scratch/gilbreth/wang4433/babysteps")
import numpy as np
from babysteps.envs.pushcube_runner import PushCubeEnvRunner
from babysteps.envs.pushcube_adapter import PushCubeAdapter
from babysteps.demo import trajectory_to_motion

r = PushCubeEnvRunner()
a = PushCubeAdapter()
dirs = ("translate_+x", "translate_-x", "translate_+y", "translate_-y")

hdr = (f"{'seed':>4} {'target':>12} {'intent':>12} {'succ':>5} {'moved':>5} "
       f"{'dx':>7} {'dy':>7} {'traj_motion':>12} {'label_ok':>8} "
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
        label_ok = (tm == m)
        ci = tuple(round(float(v), 3) for v in s.cube_xy)
        g = tuple(round(float(v), 3) for v in s.goal_xy)
        print(f"{seed:>4} {m:>12} {i.object_motion:>12} {str(res.success):>5} "
              f"{str(res.object_moved):>5} {disp[0]:>7.3f} {disp[1]:>7.3f} "
              f"{tm:>12} {str(label_ok):>8} {str(ci):>20} {str(g):>20}",
              flush=True)
r.close()
print("DIAG DONE", flush=True)
