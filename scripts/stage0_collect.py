"""Stage-0 blocked-approach data-collection CLI.

Runs `run_episode` for `--n_episodes` seeded episodes and writes one
EpisodeRecord per line to `<out_dir>/samples.jsonl`. Then computes
metrics and writes `report.{md,json}` next to the samples.

Tasks (dispatched via babysteps.envs.task_registry):
  --task PushCube-v1  (default) — Sub-project A, approach_blocked failure.
  --task PickCube-v1            — Sub-project B, grasp_slip failure.

Backends:
  --fake-env: deterministic sim-free runner from tests/conftest. Each
              task has its own fake (FakeEnvRunner / FakePickEnvRunner)
              wired through the registry.
  (default):  real env_runner from the adapter (needs Vulkan).

If `mani_skill` fails to import and `--fake-env` was NOT requested, this
script aborts with the import error rather than silently falling back —
the user sees the real failure mode.
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

from babysteps.envs.task_adapter import BaseTaskAdapter  # noqa: E402
from babysteps.envs.task_registry import (  # noqa: E402
    TASK_REGISTRY,
    get_task_entry,
)
from babysteps.episode import run_episode  # noqa: E402
from babysteps.eval import compute_metrics, write_report  # noqa: E402
from babysteps.schemas import EpisodeRecord  # noqa: E402


def _make_adapter(task_id: str, use_fake: bool) -> BaseTaskAdapter:
    """Build the right adapter for `task_id`, wired to fake or real runner."""
    entry = get_task_entry(task_id)
    if use_fake:
        fake = entry.fake_runner_factory()

        class _FakeAdapter(entry.adapter_cls):  # type: ignore[misc, valid-type]
            def make_env_runner(self):
                return fake

        return _FakeAdapter()
    return entry.adapter_cls()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--task", type=str, default="PushCube-v1",
        choices=sorted(TASK_REGISTRY.keys()),
        help="Which Stage-0 ManiSkill task to drive. Default PushCube-v1 "
             "for backward compatibility.",
    )
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--n_episodes", type=int, default=5)
    p.add_argument("--seed_start", type=int, default=0)
    p.add_argument(
        "--fake-env", action="store_true",
        help="Use the deterministic sim-free fake env_runner from "
             "tests/conftest. Useful for verifying the loop and JSONL "
             "shape on a login node where Vulkan is unavailable.",
    )
    p.add_argument(
        "--rollouts-subdir", type=str, default="rollouts",
        help="Sub-directory of out_dir to hold per-episode rollout .npz "
             "files. Only the real env_runner writes these.",
    )
    args = p.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    samples_path = args.out_dir / "samples.jsonl"
    samples_path.write_text("")   # truncate any prior run

    entry = get_task_entry(args.task)
    adapter = _make_adapter(args.task, args.fake_env)
    records: list[EpisodeRecord] = []
    try:
        for i in range(args.n_episodes):
            seed = args.seed_start + i
            episode_id = f"{entry.episode_id_prefix}_seed_{seed:04d}"
            rec = run_episode(
                episode_id=episode_id,
                seed=seed,
                adapter=adapter,
            )
            records.append(rec)
            with samples_path.open("a") as f:
                f.write(rec.to_jsonl_line() + "\n")
            retry_val = rec.metrics['retry_success']
            retry_str = "N/A" if retry_val is None else str(retry_val)
            print(
                f"[{i + 1}/{args.n_episodes}] task={args.task} seed={seed} "
                f"initial_success={rec.metrics['initial_success']} "
                f"retry={retry_str} "
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
