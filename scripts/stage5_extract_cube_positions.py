"""Stage-5 — extract privileged cubeA/cubeB START positions for an exact
StackCube cut (GPU). Feeds ``stage5_relation_oracle_probe.py --source dir``.

For each seed in a varied-intent ``samples.jsonl``, does a deterministic
``env.reset(seed)`` on the real StackCube env and dumps
``<out-dir>/seed_NNNN_positions.npz`` with:

    cubeA_xy0   (2,)  cubeA resting xy at reset
    cubeB_xy    (2,)  cubeB resting xy at reset

**No rollout.** The oracle-ceiling probe only needs the RESTING relative
geometry ``(cubeB - cubeA_start)`` — *not* cubeA's trajectory delta, which is
the quantity the ``object_motion`` label is derived from
(``trajectory_to_motion``) and would make the probe circular. A single reset
reproduces the exact scene the original demo started from (the cut uses native
resets; same seed → same layout), so this needs no oracle program and carries
no drift hazard from the runner's PD/dwell schedule.

This makes the relation oracle ceiling reproducible on the EXACT cut that
produced the DINOv2 0.42 (``reports/stage5/p1_vision_g1``), so the
``0.42 → 0.9x`` comparison uses the same seeds rather than the cross-cut
p2_vlm_wrist seeds 100-149.

Example::

    python scripts/stage5_extract_cube_positions.py \\
        --jsonl datasets/stage4/varied_intent/StackCube-v1/samples.jsonl \\
        --out-dir datasets/stage5/relation_oracle/StackCube-v1/

GPU/Vulkan node required (login-node reset triggers Vulkan device init).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))

# Reuse the sibling renderer's env builder + obs reader + record loaders so the
# reset path matches the cached-frame / demo path exactly. These lazy-import
# mani_skill only inside _make_env, so importing them on the login node is safe.
from stage5_render_demo_frames import (  # noqa: E402
    _load_records, _make_env, _read_stackcube_obs, _seed_from_record,
)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jsonl", type=Path, required=True,
                   help="varied-intent samples.jsonl (seeds read from episode_id).")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None, help="cap seeds (smoke).")
    args = p.parse_args(argv)

    records = _load_records(args.jsonl, args.limit)
    if not records:
        print("no records", file=sys.stderr)
        return 1
    task = records[0]["task"]
    if task != "StackCube-v1":
        print(f"only StackCube-v1 supported (got {task})", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    env = _make_env(task)
    try:
        for rec in records:
            seed = _seed_from_record(rec)
            obs, _info = env.reset(seed=int(seed))
            _tcp, cubeA_xy, _cubeA_z, cubeB_xy, _cubeB_z = _read_stackcube_obs(obs)
            out = args.out_dir / f"seed_{seed:04d}_positions.npz"
            np.savez_compressed(
                out,
                cubeA_xy0=np.asarray(cubeA_xy, dtype=np.float64),
                cubeB_xy=np.asarray(cubeB_xy, dtype=np.float64),
            )
            print(f"wrote {out} cubeA0={tuple(np.round(cubeA_xy,3))} "
                  f"cubeB={tuple(np.round(cubeB_xy,3))}", flush=True)
    finally:
        try:
            env.close()
        except Exception:
            pass

    print(f"\nextracted {len(records)} seeds → {args.out_dir}")
    print("next: python scripts/stage5_relation_oracle_probe.py --source dir "
          f"--positions-dir {args.out_dir} --jsonl {args.jsonl} "
          "--out-dir reports/stage5/relation_oracle_exactcut/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
