"""Stage-5 P2 — render one first-person failure-attempt frame per held-out seed.

For each seed in --seeds, re-run the standard Stage-0 episode pipeline (oracle
demo → initial_intent → executor scene with blocked_sides → run the attempt)
and save a single PNG capturing the end-of-attempt RGB frame from the env's
render camera. For seeds where the planner aborts before stepping the env
(planner_failed=True, e.g. PushCube approach_blocked), the saved frame is the
post-reset initial scene — the runner never spawns the obstacle physically,
so the visual carries no obstacle; the VLM gets its blocked-ness signal from
the failure_predicate string instead.

Output:
    <out-dir>/frames/seed_NNNN_attempt.png   (one per seed)
    <out-dir>/episodes.jsonl                 (one row per seed)

The JSONL row mirrors the per-seed EpisodeRecord needed by the eval driver:
failure_packet, initial_intent, oracle_wrong_factor, rule_table_wrong_factor.

Example::

    python scripts/stage5_p2_render_failure_frames.py \\
        --task PushCube-v1 --seeds 100-149 \\
        --out-dir datasets/stage5/p2_vlm/PushCube-v1/

GPU/Vulkan required.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from babysteps.episode import generate_proxy_demo  # noqa: E402
from babysteps.envs.task_registry import get_task_entry  # noqa: E402
from babysteps.failure import attribute_failure  # noqa: E402
from babysteps.render.common import render_frame  # noqa: E402


def _parse_seed_range(s: str) -> list[int]:
    if "-" in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(s)]


def _save_png(path: Path, frame: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame.astype(np.uint8)).save(path)


def _make_render_runner(task: str):
    """Construct a task-specific env_runner with render_mode='rgb_array'.

    We bypass adapter.env_runner() (which caches a NON-render-mode runner)
    because the data-collection runner doesn't allocate a render camera —
    render_frame fails with `render_mode is not set`. The render_mode
    constructor flag is the minimum-invasive addition (default None
    preserves all existing data-collection behavior byte-for-byte).
    """
    if task == "PushCube-v1":
        from babysteps.envs.pushcube_runner import PushCubeEnvRunner
        return PushCubeEnvRunner(render_mode="rgb_array")
    if task == "PickCube-v1":
        from babysteps.envs.pickcube_runner import PickCubeEnvRunner
        return PickCubeEnvRunner(render_mode="rgb_array")
    if task == "StackCube-v1":
        from babysteps.envs.stackcube_runner import StackCubeEnvRunner
        return StackCubeEnvRunner(render_mode="rgb_array")
    raise ValueError(f"unsupported task: {task}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True,
                   choices=["PushCube-v1", "PickCube-v1", "StackCube-v1"])
    p.add_argument("--seeds", required=True,
                   help="Seed range A-B (inclusive) or single int.")
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args(argv)

    seeds = _parse_seed_range(args.seeds)
    entry = get_task_entry(args.task)
    adapter = entry.adapter_cls()
    # Use a render-enabled runner directly (NOT adapter.env_runner()).
    env_runner = _make_render_runner(args.task)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = args.out_dir / "episodes.jsonl"
    n_saved, n_failure = 0, 0

    with episodes_path.open("w") as ef:
        for seed in seeds:
            try:
                # Step 1: reset + demo + intent — same path as run_episode.
                scene_initial = env_runner.reset(seed)
                demo_evidence = generate_proxy_demo(
                    env_runner, scene_initial, adapter,
                )
                initial_intent = adapter.scripted_demo_to_intent(demo_evidence)
                scene_executor = replace(
                    scene_initial,
                    blocked_sides=adapter.default_blocked_factory(initial_intent),
                )
                # Step 2: re-reset to the executor scene so attempt is from
                # the same physical state as a true Stage-0 attempt.
                env_runner.reset(seed)
                # Step 3: run the attempt. run() resets internally, then
                # either steps the env (PickCube/StackCube/PushCube-unblocked)
                # or returns planner_failed without stepping (PushCube
                # approach_blocked). Either way, env_runner._env's state
                # after this call is the right thing to render.
                attempt = env_runner.run(initial_intent, scene_executor)
                env = env_runner._env  # noqa: SLF001 — intentional script-only access
                frame = render_frame(env)
            except Exception as exc:
                print(f"WARN: seed {seed} failed pipeline: {exc}",
                      file=sys.stderr)
                continue

            frame_path = args.out_dir / "frames" / f"seed_{seed:04d}_attempt.png"
            _save_png(frame_path, frame)
            n_saved += 1

            # Build episode record (failure packet + oracle wrong factor).
            failure_packet = adapter.build_failure_packet(
                initial_intent, attempt, scene_executor,
            )
            oracle = adapter.oracle_wrong_factor(initial_intent, scene_executor)
            try:
                attribution = attribute_failure(failure_packet)
                rule_factor = attribution.wrong_factor
            except ValueError:
                # No rule for this predicate (e.g. predicate=="none" already
                # handled by attribute_failure; defensive fallback).
                rule_factor = None
            is_failure = failure_packet.failure_predicate != "none"
            if is_failure:
                n_failure += 1
            row = {
                "seed": seed,
                "task": args.task,
                "frame_path": str(frame_path),
                "initial_intent": initial_intent.to_dict(),
                "failure_predicate": failure_packet.failure_predicate,
                "oracle_wrong_factor": oracle,
                "rule_table_wrong_factor": rule_factor,
                "is_failure": is_failure,
                "initial_success": bool(attempt.success),
            }
            ef.write(json.dumps(row, sort_keys=True) + "\n")

    env_runner.close()
    adapter.close()  # idempotent; no-op since we never called adapter.env_runner()
    print(f"saved {n_saved} frames ({n_failure} failures) → {args.out_dir}")
    print(f"wrote {episodes_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
