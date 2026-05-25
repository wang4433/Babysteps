#!/usr/bin/env python
"""Stage-5 P1 — iconic per-policy PushCube render contrast.

For each (seed, policy) pair, emits one MP4 containing demo + initial
blocked attempt + that policy's retry, concatenated into a single
"full episode" clip. Designed to populate the empty 'videos' column of
the Stage-5 P1 PushCube held-out report gallery at
reports/stage5/p1_vision_g4_g5/PushCube-v1/report_gallery/.

Output filename: pushcube_seed_NNNN__<policy>_full.mp4

Default scope (5 iconic seeds × 4 policies = 20 MP4s):
  seeds    : 100, 110, 120, 130, 143
             (100/110/120/130: clean wins for all but same_intent_retry;
              143: latent fails, oracle/babysteps succeed)
  policies : latent, oracle_factor_revision,
             babysteps_selective, same_intent_retry

Needs Vulkan. On the Gilbreth login node it falls back to Mesa lavapipe
(slow); on a GPU node it uses the NVIDIA Vulkan ICD.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Make the project root importable without `pip install -e .`.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from babysteps.envs.task_registry import get_task_entry  # noqa: E402
from babysteps.policies import (  # noqa: E402
    babysteps_selective, oracle_factor_revision, same_intent_retry,
)
from babysteps.render.common import annotate_frame, save_mp4  # noqa: E402
from babysteps.render.pushcube import render_policy_episode  # noqa: E402
from babysteps.stage4.latent_policy import (  # noqa: E402
    latent_revision_factory, load_latent_pack,
)


DEFAULT_SEEDS = (100, 110, 120, 130, 143)
DEFAULT_POLICIES = (
    "latent", "oracle_factor_revision",
    "babysteps_selective", "same_intent_retry",
)


def _parse_seeds(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _parse_policies(s: str) -> list[str]:
    out = []
    for name in s.split(","):
        name = name.strip()
        if not name:
            continue
        if name not in DEFAULT_POLICIES:
            raise ValueError(
                f"Unknown policy {name!r}. Valid: {DEFAULT_POLICIES}"
            )
        out.append(name)
    return out


def _make_features_provider(features_dir: Path):
    """Return a callable(seed) -> np.ndarray reading cached DINOv2 features."""
    features_dir = Path(features_dir)

    def _provider(seed: int):
        path = features_dir / f"seed_{seed:04d}_dinov2.npy"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing cached DINOv2 feature for seed {seed}: {path}"
            )
        return np.load(path)

    return _provider


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS),
                   help="Comma-separated seed list (default: 5 iconic seeds).")
    p.add_argument("--policies", default=",".join(DEFAULT_POLICIES),
                   help="Comma-separated policy list (default: all 4).")
    p.add_argument("--pack-dir", type=Path,
                   default=Path("models/stage5/p1_vision/PushCube-v1"),
                   help="LatentPack directory (intent_head.pt / revise_head.pt / centroids.npz / meta.json).")
    p.add_argument("--features-dir", type=Path,
                   default=Path("datasets/stage5/varied_intent/PushCube-v1/features"),
                   help="Directory of cached DINOv2 seed_NNNN_dinov2.npy files.")
    p.add_argument("--out-dir", type=Path,
                   default=Path("renders/stage5_p1_iconic/pushcube"),
                   help="Output directory for MP4s.")
    p.add_argument("--fps", type=int, default=20)
    args = p.parse_args(argv)

    seeds = _parse_seeds(args.seeds)
    policies = _parse_policies(args.policies)

    try:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401 — registers PushCube-v1
    except ImportError as exc:
        print(f"ManiSkill import failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    # Validate feature presence before launching env (cheap fail-fast).
    features_provider = _make_features_provider(args.features_dir)
    if "latent" in policies:
        for s in seeds:
            _ = features_provider(s)  # raises FileNotFoundError on miss

    # Load LatentPack once (only needed if 'latent' is in the policy list,
    # but loading is cheap and keeps the code branchless).
    if "latent" in policies:
        pack = load_latent_pack(args.pack_dir)
        latent_policy = latent_revision_factory(pack)
    else:
        latent_policy = None

    policy_callables = {
        "latent": latent_policy,
        "oracle_factor_revision": oracle_factor_revision,
        "babysteps_selective": babysteps_selective,
        "same_intent_retry": same_intent_retry,
    }
    policy_provider = {
        "latent": features_provider,
        "oracle_factor_revision": None,
        "babysteps_selective": None,
        "same_intent_retry": None,
    }

    entry = get_task_entry("PushCube-v1")
    adapter = entry.adapter
    env = gym.make("PushCube-v1", obs_mode="state_dict", render_mode="rgb_array",
                    sim_backend="gpu")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    n_total = len(seeds) * len(policies)
    written: list[Path] = []
    for seed in seeds:
        for pol in policies:
            print(f"--- seed {seed:04d} × policy {pol} "
                  f"({len(written) + 1}/{n_total}) ---", flush=True)
            frames, title = render_policy_episode(
                env, adapter, seed,
                policy_name=pol,
                policy_callable=policy_callables[pol],
                demo_features_provider=policy_provider[pol],
                fps=args.fps,
            )
            annotated = [annotate_frame(f, title[0], title[1]) for f in frames]
            out_path = args.out_dir / f"pushcube_seed_{seed:04d}__{pol}_full.mp4"
            save_mp4(annotated, out_path, fps=args.fps)
            written.append(out_path)
            print(f"  wrote {out_path}  ({len(annotated)} frames)", flush=True)

    env.close()
    print(f"\nDone. {len(written)} MP4s under {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
