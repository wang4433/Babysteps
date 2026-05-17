"""ManiSkill PushCube-v1 loadability smoke test.

Exits 0 if the env loads and resets in the active conda env;
exits 1 with a one-line error report otherwise. Not a unit test — runs only
when explicitly invoked. The Stage-0 collect script (`stage0_collect.py`)
imports `mani_skill` lazily and surfaces the same error message.
"""
from __future__ import annotations

import sys


def main() -> int:
    try:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401 — registers PushCube-v1
        env = gym.make(
            "PushCube-v1",
            obs_mode="state_dict",
            control_mode="pd_ee_delta_pose",
            sim_backend="cpu",
        )
        obs, info = env.reset(seed=0)
        keys = sorted(obs.keys()) if hasattr(obs, "keys") else type(obs).__name__
        print(f"OK; obs keys: {keys}")
        env.close()
        return 0
    except Exception as exc:
        print(f"SMOKE FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
