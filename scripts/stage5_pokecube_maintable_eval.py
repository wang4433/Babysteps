"""Stage-5 — PokeCube 6-condition FAIR-RECOVERY main table (held-out family).

The user's locked step 1: run ALL SIX unified conditions on the held-out
PokeCube family under a FAIR protocol — every condition branches from the SAME
first failure, the same controller, the same scene seed, and the same retry
budget — so the proposed shared policy is compared to the VLM baselines on
identical ground. This fills the pasted-analysis Table 2 (fair recovery) +
Table 4 (efficiency: decision-latency split / VLM tokens+calls / edit
cardinality / preservation / harmful) and feeds Table 5 (clustered CI by seed).

It deliberately REUSES the validated condition engine
(``stage5_unified_maintable_eval._run_condition`` + ``babysteps.stage5.maintable``
+ the conditions registry) so the 6 conditions are defined once. The PokeCube
specifics live only here:

* **Encoder-free failure sourcing** (the LOTO design): reach-filter the seeds
  (``_worst_waypoint_dist`` <= 0.785, the 3-way kill-gate filter), inject a goal
  direction, and ground a WRONG initial contact face — the natural
  mis-grounding stand-in. PokeCube has NO trained encoder BY DESIGN (that is what
  "held out" means), so there is no vision-decode here.
* **3 reachable candidates** ``{minus_x_face(+x), minus_y_face(+y),
  plus_y_face(-y)}`` (−x is reach-dead, excluded) via the spec's
  ``candidates_override``.
* **Frame capture** after the shared first failure (``render_frame`` +
  ``render_wrist_frame``), so the live VLM conditions (``vlm_free_replan``,
  ``vlm_diagnosis_local_edit``) see a real PokeCube failure image.
* **Per-task editor = the PushCube-family residual editor** (reused on the
  held-out task — itself a transfer baseline; the contact_region family's
  hand-built value producer vs the ONE shared scorer).

``PokeCubeEnvRunner.run`` self-resets to the last ``reset`` seed, so the engine's
retry ``runner.run(revised, scene)`` re-runs the same scene with the episode's
injection — set once per episode before the conditions loop.

GPU/Vulkan + InternVL for the real run; ``--fake --mock`` is a login-node smoke
(FakePokeEnvRunner reuses push physics; frames are skipped; the MockVLM ignores
images). Sim-free tests live in tests/test_stage5_pokecube_maintable_eval.py.

    python scripts/stage5_pokecube_maintable_eval.py \\
        --scorer models/stage5/shared_policy/pooled_gi_none.pt \\
        --pack-dir models/stage5/p1_vision_4way/PushCube-v1 \\
        --residual-head models/stage5/p1_vision_4way/PushCube-v1/revise_head_residual.pt \\
        --seeds 0-299 --target-n 20 --max-approach-dist 0.785 \\
        --out reports/stage5/pokecube_maintable/results.json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _ROOT / "scripts"
for _p in (str(_ROOT), str(_SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from babysteps.stage5.conditions import CONDITIONS  # noqa: E402
from babysteps.stage5.cluster_bootstrap import (  # noqa: E402
    clustered_bootstrap_ci, failing_clusters, paired_clustered_bootstrap_diff,
)
from babysteps.stage5.maintable import aggregate  # noqa: E402

# Reused condition engine + data contracts (pure; no sim import at module load).
from stage5_unified_maintable_eval import (  # noqa: E402
    EpisodeData, TaskSpec, _run_condition, _synthetic_pushcube_editor,
)
from stage5_pokecube_killgate import _DIR_TO_MOTION, _worst_waypoint_dist  # noqa: E402
from stage5_pokecube_loto_eval import _DIR_TO_FACE, LOTO_FACES, _intent  # noqa: E402

from babysteps.stage5.residual_reviser import _observed_residual  # noqa: E402
from babysteps.stage5.revision_policy import FailureEvidence  # noqa: E402

# Condition matrix: (display label, base condition, attributor_override).
# The two paired methods (shared scorer, per-task editor) are run under BOTH
# attributors so the table separates VALUE TRANSFER (oracle attribution isolates
# the value policy) from the ATTRIBUTION bottleneck (VLM attribution = the
# deployed-with-8B-VLM path). On held-out PokeCube the VLM mis-attributes
# contact_region from a single third-person frame (attribution_accuracy ~ 0), so
# the @oracle_attr rows are the honest "the shared policy transfers" claim and the
# VLM rows motivate build-order step 4 (distill a better attributor).
_CONDITION_MATRIX: list[tuple[str, str, str | None]] = [
    ("same_intent_retry", "same_intent_retry", None),
    ("random_factor_local_edit", "random_factor_local_edit", None),
    ("vlm_free_replan", "vlm_free_replan", None),
    ("vlm_diagnosis_local_edit", "vlm_diagnosis_local_edit", "vlm"),
    ("shared_revision_policy", "shared_revision_policy", "vlm"),
    ("vlm_diagnosis_local_edit@oracle_attr", "vlm_diagnosis_local_edit", "oracle"),
    ("shared_revision_policy@oracle_attr", "shared_revision_policy", "oracle"),
    ("oracle_single_slot", "oracle_single_slot", None),
]
_ALL_SIX = list(CONDITIONS)  # kept for back-compat (registry order)

_DIFF_PAIRS = [
    ("shared_revision_policy@oracle_attr", "oracle_single_slot"),
    ("shared_revision_policy@oracle_attr", "vlm_diagnosis_local_edit@oracle_attr"),
    ("shared_revision_policy", "vlm_free_replan"),
    ("shared_revision_policy", "same_intent_retry"),
]


def _parse_seeds(s: str) -> list[int]:
    if "-" in s and "," not in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",") if x]


def _save_png(path: Path, frame) -> None:
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(path)


def _reachable_seeds(runner, seeds, directions, max_approach_dist, target_n):
    """The 3-way kill-gate reach filter: keep a seed only if its worst
    prepoke/poke waypoint distance (over all directions) is within reach. Same
    selection across directions, so the conditions compare on identical seeds."""
    out: list[int] = []
    for s in seeds:
        if hasattr(runner, "set_injection"):
            runner.set_injection(None)
        scene0 = runner.reset(int(s))
        if (max_approach_dist is not None and _worst_waypoint_dist(
                scene0.cube_xy, scene0.goal_xy, list(directions)) > max_approach_dist):
            continue
        out.append(int(s))
        if target_n is not None and len(out) >= target_n:
            break
    return out


def _build_poke_spec(args, runner, adapter) -> TaskSpec:
    """contact_region spec for the held-out PokeCube family: 3 reachable
    candidates + the PushCube-family residual editor as the per-task value
    producer (reused on the held-out task)."""
    from babysteps.schemas import INTENT_FIELDS
    if args.fake:
        editor = _synthetic_pushcube_editor()
    else:
        from babysteps.stage5.residual_reviser import ResidualSlotEditor
        editor = ResidualSlotEditor.from_pack(
            args.pack_dir, args.residual_head, factor="contact_region")
    return TaskSpec(
        task="PokeCube-v1", implicated_factor="contact_region",
        factor_menu=INTENT_FIELDS,
        source_episode=lambda *_: None,  # not used; episodes built inline below
        editor_producers={
            "contact_region": lambda req: editor(
                req.current_value, req.e_fail.residual_xy, req.e_fail.predicate)},
        candidates_override={"contact_region": tuple(LOTO_FACES)})


def _source_episode(runner, adapter, seed, direction, wrong_face, *,
                    capture_frames, frames_dir):
    """Build ONE composite episode (seed, direction, wrong_face): inject the goal
    direction, ground the wrong initial face, run the SHARED first failure, and
    capture the failure frame for the VLM conditions."""
    motion = _DIR_TO_MOTION[direction]
    if hasattr(runner, "set_injection"):
        runner.set_injection(motion)
    scene = runner.reset(int(seed))
    initial = _intent(adapter, scene, wrong_face)   # oracle, contact=wrong_face
    attempt1 = runner.run(initial, scene)           # self-resets to seed+motion

    # Third-person frame only (PokeCube doesn't support panda_wristcam, which
    # would break the grasp). wrist_frame_path stays None -> the VLM prompts use
    # the single third-person view (wrist_view=False).
    frame_path = wrist_path = None
    if capture_frames:
        from babysteps.render.common import render_frame
        env = runner._env  # noqa: SLF001 — script-only access, mirrors P2
        tag = f"seed_{int(seed):04d}_{direction.replace('+', 'p').replace('-', 'm')}_{wrong_face}"
        fp_png = Path(frames_dir) / f"{tag}.png"
        _save_png(fp_png, render_frame(env))
        frame_path = str(fp_png)

    fp = adapter.build_failure_packet(initial, attempt1, scene)
    gt = adapter.oracle_correct_intent(scene)
    residual = _observed_residual(fp, scene)
    return EpisodeData(
        seed=int(seed), initial=initial, scene_exec=scene, fp=fp, gt=gt,
        initial_success=bool(attempt1.success),
        e_fail=FailureEvidence(predicate=fp.failure_predicate,
                               residual_xy=(float(residual[0]), float(residual[1]))),
        frame_path=frame_path, wrist_frame_path=wrist_path)


def run_maintable(runner, adapter, vlm, spec, *, seeds, directions,
                  condition_matrix, shared_policy, max_approach_dist, target_n,
                  capture_frames, frames_dir):
    """Per (reachable seed x direction x wrong-face): one shared first failure,
    then every (label, base_condition, attributor_override) variant's retry.
    Returns (rows_by_label, flat_rows, n_reachable)."""
    reachable = _reachable_seeds(runner, seeds, directions, max_approach_dist,
                                 target_n)
    labels = [lab for (lab, _b, _a) in condition_matrix]
    rows_by_label: dict[str, list[dict]] = {lab: [] for lab in labels}
    flat_rows: list[dict] = []
    for s in reachable:
        for d in directions:
            correct = _DIR_TO_FACE[d]
            for w in LOTO_FACES:
                if w == correct:
                    continue
                ep = _source_episode(
                    runner, adapter, s, d, w,
                    capture_frames=capture_frames, frames_dir=frames_dir)
                flat = {"seed": s, "direction": d, "wrong_face": w,
                        "correct_face": correct,
                        "initial_success": ep.initial_success}
                for label, base_cond, attr_override in condition_matrix:
                    row = _run_condition(
                        base_cond, ep, spec, runner, vlm, f"{s}:{d}:{w}",
                        shared_policy=shared_policy,
                        attributor_override=attr_override)
                    rows_by_label[label].append(row)
                    flat[f"{label}_success"] = bool(row["final_success"])
                flat_rows.append(flat)
    return rows_by_label, flat_rows, len(reachable)


def _ci_table(flat_rows, labels, *, n_boot, seed):
    """Table 5: clustered (by seed) recovery CI per label + paired diffs.

    Computed on the INITIAL-FAIL subset (the fair-recovery question is "given a
    failure, who recovers"), matching ``recovery_on_initial_fail``. In this
    LOTO-style design every episode is engineered to fail initially, so this is
    normally the full set, but the subset is taken explicitly for correctness."""
    fail_rows = [r for r in flat_rows if not r.get("initial_success")]
    ci = {lab: clustered_bootstrap_ci(fail_rows, f"{lab}_success",
                                      n_boot=n_boot, seed=seed)
          for lab in labels}
    diffs = {}
    for a, b in _DIFF_PAIRS:
        if a in labels and b in labels:
            diffs[f"{a}__minus__{b}"] = paired_clustered_bootstrap_diff(
                fail_rows, f"{a}_success", f"{b}_success",
                n_boot=n_boot, seed=seed)
    # Value-transfer failure attribution: where does the shared policy (under
    # ORACLE attribution) fail, and does the oracle value ALSO fail there?
    vt = "shared_revision_policy@oracle_attr"
    fail = (failing_clusters(
        [{**r, "scorer_success": r.get(f"{vt}_success"),
          "oracle_success": r.get("oracle_single_slot_success")}
         for r in fail_rows], "scorer_success")
        if vt in labels and "oracle_single_slot" in labels else {})
    return {"recovery_ci": ci, "paired_diffs": diffs,
            "n_initial_fail": len(fail_rows),
            "value_transfer_failure_attribution": fail}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scorer", type=Path, default=None,
                   help="Shared-scorer checkpoint (enables shared_revision_policy).")
    p.add_argument("--pack-dir", type=Path, default=None,
                   help="PushCube 4-way latent pack (per-task residual editor).")
    p.add_argument("--residual-head", type=Path, default=None)
    p.add_argument("--seeds", default="0-299")
    p.add_argument("--directions", default="+x,+y,-y")
    p.add_argument("--max-approach-dist", type=float, default=0.785)
    p.add_argument("--target-n", type=int, default=20)
    p.add_argument("--fake", action="store_true")
    p.add_argument("--mock", action="store_true")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--out", type=Path,
                   default=Path("reports/stage5/pokecube_maintable/results.json"))
    args = p.parse_args(argv)

    matrix = _CONDITION_MATRIX
    labels = [lab for (lab, _b, _a) in matrix]
    needs_scorer = any(b == "shared_revision_policy" for (_l, b, _a) in matrix)
    if needs_scorer and args.scorer is None:
        p.error("shared_revision_policy needs --scorer <checkpoint>")
    directions = [d.strip() for d in args.directions.split(",") if d.strip()]
    seeds = _parse_seeds(args.seeds)
    frames_dir = args.out.parent / "frames"

    # Runner + adapter (real runner renders for the VLM; fake skips frames).
    from babysteps.envs.pokecube_adapter import PokeCubeAdapter
    adapter = PokeCubeAdapter()
    if args.fake:
        from tests.conftest import FakePokeEnvRunner
        runner = FakePokeEnvRunner()
        capture_frames = False
    else:
        from babysteps.envs.pokecube_runner import PokeCubeEnvRunner
        # Third-person render only (default panda). PokeCube does NOT support the
        # panda_wristcam robot — swapping it in breaks the grasp+poke (oracle ->
        # 0%, observed on job 10979367). The third-person frame is the VLM's
        # primary view; the wrist view is dropped for this held-out family.
        runner = PokeCubeEnvRunner(render_mode="rgb_array")
        capture_frames = True

    if args.mock:
        from babysteps.stage5.vlm_attribute import MockVLMClient
        vlm = MockVLMClient(constrained_response="contact_region")
    else:
        from babysteps.stage5.vlm_attribute import InternVLClient
        vlm = InternVLClient()
        print("loading InternVL3.5-8B ...")
        vlm.load()

    shared_policy = None
    if needs_scorer:
        from babysteps.stage5.shared_revision_policy import SharedScorerPolicy
        shared_policy = SharedScorerPolicy.from_pack(args.scorer)
        print(f"loaded shared scorer: {args.scorer}")

    spec = _build_poke_spec(args, runner, adapter)

    print(f"=== PokeCube fair-recovery main table (fake={args.fake}, "
          f"mock={args.mock}) dirs={directions} target_n={args.target_n} ===")
    try:
        rows_by_label, flat_rows, n_reach = run_maintable(
            runner, adapter, vlm, spec, seeds=seeds, directions=directions,
            condition_matrix=matrix, shared_policy=shared_policy,
            max_approach_dist=args.max_approach_dist, target_n=args.target_n,
            capture_frames=capture_frames, frames_dir=frames_dir)
    finally:
        if hasattr(runner, "close"):
            try:
                runner.close()
            except Exception:
                pass

    summary = aggregate(rows_by_label)
    ci = _ci_table(flat_rows, labels, n_boot=args.n_boot, seed=0)
    result = {
        "summary": summary, "ci": ci,
        "n_reachable_seeds": n_reach, "n_episodes": len(flat_rows),
        "candidates": list(LOTO_FACES),
        "config": {"scorer": str(args.scorer), "directions": directions,
                   "fake": args.fake, "mock": args.mock,
                   "condition_matrix": [list(t) for t in matrix]},
        "flat_rows": flat_rows,
    }

    print(f"\n  n_reachable_seeds={n_reach}  n_episodes={len(flat_rows)}")
    for lab in labels:
        s = summary[lab]
        rc = ci["recovery_ci"][lab]
        print(f"  {lab:38s} recov_on_fail={_fmt(s['recovery_on_initial_fail'])} "
              f"[{rc['lo']:.2f},{rc['hi']:.2f}]  attr={_fmt(s.get('attribution_accuracy'))} "
              f"pres={_fmt(s['preservation_mean'])} "
              f"edits={_fmt(s['edit_cardinality_mean'])} "
              f"vlm_tok={_fmt(s['vlm_gen_tokens_mean'])} "
              f"dec_lat={_fmt(s['total_decision_latency_s_mean'])}")
    for k, d in ci["paired_diffs"].items():
        print(f"  {k}: {d['diff']:+.3f} [{d['lo']:+.3f},{d['hi']:+.3f}]")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    if hasattr(adapter, "close"):
        adapter.close()
    return 0


def _fmt(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"


if __name__ == "__main__":
    sys.exit(main())
