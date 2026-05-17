"""Render Stage-0 episodes as 2D top-down MP4s from state trajectories.

Re-runs each episode with the fake env_runner (deterministic given seed),
captures the three phase trajectories (demo, attempt 1, attempt 2), and
writes one MP4 per episode to `<out_dir>/videos/`. No simulator, no GPU,
no Vulkan needed — works on the Gilbreth login node.

For real RGB recordings (Franka arm + table), use
`scripts/render_stage0_maniskill.py` on a GPU+Vulkan compute node.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

# Make the project root importable without `pip install -e .`.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from babysteps.envs.pushcube_adapter import PushCubeAdapter  # noqa: E402
from babysteps.episode import generate_proxy_demo  # noqa: E402
from babysteps.viz import render_episode_topdown  # noqa: E402


_ADAPTER = PushCubeAdapter()


def _replay_with_recording(env_runner, seed: int) -> dict:
    """Reproduce run_episode but capture per-attempt trajectories for the
    renderer. Equivalent semantically to babysteps.episode.run_episode."""
    scene_initial = env_runner.reset(seed)
    demo_evidence = generate_proxy_demo(env_runner, scene_initial, _ADAPTER)
    initial_intent = _ADAPTER.scripted_demo_to_intent(demo_evidence)
    scene_executor = replace(
        scene_initial,
        blocked_sides=_ADAPTER.default_blocked_factory(initial_intent),
    )
    attempt_1 = env_runner.run(initial_intent, scene_executor)
    fp = _ADAPTER.build_failure_packet(initial_intent, attempt_1, scene_executor)

    revised_intent = None
    attempt_2 = None
    if fp.failure_predicate != "none":
        attribution = _ADAPTER.attribute_failure(fp)
        try:
            revised_intent, _ = _ADAPTER.revise_intent(
                initial_intent, attribution, scene_executor,
            )
            attempt_2 = env_runner.run(revised_intent, scene_executor)
        except NotImplementedError:
            pass

    return {
        "scene_initial": scene_initial,
        "scene_executor": scene_executor,
        "demo_evidence": demo_evidence,
        "initial_intent": initial_intent,
        "revised_intent": revised_intent,
        "attempt_1": attempt_1,
        "attempt_2": attempt_2,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--samples", type=Path, required=True,
        help="samples.jsonl — used only to read episode_id / seed pairs.",
    )
    p.add_argument(
        "--out_dir", type=Path, required=True,
        help="Per-episode MP4s go to <out_dir>/videos/<episode_id>.mp4",
    )
    p.add_argument("--fps", type=int, default=12)
    p.add_argument(
        "--limit", type=int, default=None,
        help="Render at most this many episodes from the front of samples.jsonl.",
    )
    args = p.parse_args(argv)

    # Fake env_runner from the test fixture — deterministic, sim-free.
    from tests.conftest import FakeEnvRunner  # noqa: WPS433
    runner = FakeEnvRunner()

    videos_dir = args.out_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    with args.samples.open() as f:
        for i, line in enumerate(f):
            if args.limit is not None and i >= args.limit:
                break
            rec = json.loads(line.strip())
            episode_id = rec["episode_id"]
            seed = int(episode_id.rsplit("_", 1)[-1])
            replay = _replay_with_recording(runner, seed)
            out_path = videos_dir / f"{episode_id}.mp4"
            print(f"  → rendering {out_path}", flush=True)
            attempt2 = replay["attempt_2"]
            render_episode_topdown(
                out_path=out_path,
                episode_id=episode_id,
                cube0_xy=replay["scene_initial"].cube_xy,
                goal_xy=replay["scene_initial"].goal_xy,
                blocked_sides=replay["scene_executor"].blocked_sides,
                demo_trajectory=list(replay["demo_evidence"].object_trajectory),
                initial_intent=replay["initial_intent"].to_dict(),
                revised_intent=(
                    replay["revised_intent"].to_dict()
                    if replay["revised_intent"] else None
                ),
                attempt1_trajectory=list(replay["attempt_1"].trajectory_xy or [replay["scene_initial"].cube_xy]),
                attempt1_planner_failed=bool(replay["attempt_1"].planner_failed),
                attempt2_trajectory=(
                    list(attempt2.trajectory_xy) if attempt2 is not None else None
                ),
                attempt2_success=(
                    bool(attempt2.success) if attempt2 is not None else None
                ),
                fps=args.fps,
            )
            written.append(out_path)

    runner.close()
    print()
    print(f"wrote {len(written)} video(s) to {videos_dir}:")
    for p in written:
        size_kb = p.stat().st_size // 1024
        print(f"  {p}  ({size_kb} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
