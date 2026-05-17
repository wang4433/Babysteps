"""Stage-0 PushCube blocked-approach data-collection CLI.

Runs `run_episode` for `--n_episodes` seeded episodes and writes one
EpisodeRecord per line to `<out_dir>/samples.jsonl`. Then computes metrics
and writes `report.{md,json}` next to the samples.

Backends:
  - default: real `PushCubeEnvRunner` (needs Vulkan-capable compute node).
  - `--fake-env`: deterministic sim-free runner from `tests.conftest`,
    useful for verifying the data contract on the login node.

If `mani_skill` fails to import and `--fake-env` was NOT requested, this
script aborts with the import error rather than silently falling back —
that way the user sees the real failure mode.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the project root importable without `pip install -e .`.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from babysteps.episode import run_episode  # noqa: E402
from babysteps.eval import compute_metrics, write_report  # noqa: E402
from babysteps.schemas import EpisodeRecord  # noqa: E402


def _make_adapter(use_fake: bool):
    """Build a PushCubeAdapter wired to the right env runner."""
    from babysteps.envs.pushcube_adapter import PushCubeAdapter  # noqa: WPS433
    if use_fake:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from tests.conftest import FakeEnvRunner   # noqa: WPS433
        fake = FakeEnvRunner()

        class _FakePushCubeAdapter(PushCubeAdapter):
            def make_env_runner(self):
                return fake
        return _FakePushCubeAdapter()
    return PushCubeAdapter()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--n_episodes", type=int, default=5)
    p.add_argument("--seed_start", type=int, default=0)
    p.add_argument(
        "--fake-env", action="store_true",
        help="Use the deterministic sim-free FakeEnvRunner from tests/conftest. "
             "Useful for verifying the loop and JSONL shape on the login node, "
             "where Vulkan is unavailable.",
    )
    p.add_argument(
        "--rollouts-subdir", type=str, default="rollouts",
        help="Sub-directory of out_dir to hold per-episode rollout .npz files. "
             "Only the real PushCubeEnvRunner writes these.",
    )
    args = p.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    samples_path = args.out_dir / "samples.jsonl"
    # Truncate any prior run.
    samples_path.write_text("")

    adapter = _make_adapter(args.fake_env)
    records: list[EpisodeRecord] = []
    try:
        for i in range(args.n_episodes):
            seed = args.seed_start + i
            episode_id = f"pushcube_blocked_approach_seed_{seed:04d}"
            rec = run_episode(
                episode_id=episode_id,
                seed=seed,
                adapter=adapter,
            )
            records.append(rec)
            with samples_path.open("a") as f:
                f.write(rec.to_jsonl_line() + "\n")
            print(
                f"[{i + 1}/{args.n_episodes}] seed={seed} "
                f"initial_success={rec.metrics['initial_success']} "
                f"retry_success={rec.metrics['retry_success']} "
                f"failure_type={rec.metrics['failure_type']}",
                flush=True,
            )
    finally:
        adapter.close()

    metrics = compute_metrics(records)
    write_report(metrics, args.out_dir)
    print()
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed_acceptance"] else 1


if __name__ == "__main__":
    sys.exit(main())
