"""Census of TurnFaucet switch-joint axis orientation per (model, switch).

For the vertical-axis subset (Sub-project D productionization step 7): reset the
env across many seeds, record for each the faucet model id, target switch-link
name, and the world-frame joint axis, and classify vertical / tilted /
horizontal. Aggregates to a whitelist of (model, switch) combos whose handle
rotates about a (near-)vertical axis -- where the re-grasp ratchet works (32%).

Output: reports/stage5/turnfaucet_diagnostic/axis_census.json
  { "by_combo": {"<model>/<switch>": {"class":..., "n":k, "axis":[...]}},
    "vertical_combos": [...], "seed_axis": {"<seed>": "<class>"} }

SCRATCH diagnostic (GPU-only).
"""
from __future__ import annotations

import os
import sys
import json
import collections
from pathlib import Path
sys.path.insert(0, "/scratch/gilbreth/wang4433/babysteps")

import numpy as np
import gymnasium as gym
import mani_skill.envs  # noqa: F401

from babysteps.render.common import to_np

N_SEEDS = int(os.environ.get("DIAG_NSEEDS", "400"))
OUT = os.environ.get("DIAG_OUT", "reports/stage5/turnfaucet_diagnostic/axis_census.json")


def classify(axis):
    az = abs(float(axis[2]))
    return "vertical" if az > 0.85 else ("horizontal" if az < 0.3 else "tilted")


def main():
    env = gym.make("TurnFaucet-v1", obs_mode="state_dict",
                   control_mode="pd_ee_delta_pose", sim_backend="gpu")
    by_combo = {}
    seed_axis = {}
    cls_count = collections.Counter()
    for seed in range(N_SEEDS):
        obs, _ = env.reset(seed=seed)
        u = env.unwrapped
        try:
            model_id = u._faucets[0].name.split("-")[0]
        except Exception:
            model_id = "?"
        try:
            switch = u._target_switch_links[0].name
        except Exception:
            switch = "?"
        axis = to_np(obs["extra"]["target_joint_axis"]).astype(float).tolist()
        c = classify(axis)
        cls_count[c] += 1
        seed_axis[str(seed)] = c
        key = f"{model_id}/{switch}"
        rec = by_combo.setdefault(key, {"class": c, "n": 0, "axis": axis})
        rec["n"] += 1
        if seed < 8:
            print(f"  seed {seed}: {key} axis={[round(a,2) for a in axis]} -> {c}", flush=True)
    env.close()

    vertical = sorted(k for k, v in by_combo.items() if v["class"] == "vertical")
    vertical_models = sorted({k.split("/")[0] for k in vertical})
    out = {
        "n_seeds": N_SEEDS,
        "n_combos": len(by_combo),
        "class_counts": dict(cls_count),
        "by_combo": by_combo,
        "vertical_combos": vertical,
        "vertical_models": vertical_models,
        "seed_axis": seed_axis,
    }
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps(out, indent=2))
    print(f"\nclasses over {N_SEEDS} resets: {dict(cls_count)}", flush=True)
    print(f"distinct (model/switch) combos: {len(by_combo)}", flush=True)
    print(f"vertical combos: {len(vertical)} across {len(vertical_models)} models", flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
