"""Stage-5 — NATURAL-failure, seed-DECOUPLED babystep loop (PushCube binary +x/-x).

The honest paradigm (redesign_failure_paradigm.md Layer 2, see
project_stage5_loop_failure_is_artificial): the demo encodes ONE instance and
the Franka executes a DIFFERENT instance, with the artificial `blocked_sides`
block dropped. The initial attempt fails NATURALLY when the demo-derived intent
does not transfer to the execution scene — not because of an injected block.

Why PushCube is binary +x/-x (job 10966085, project_pushcube_native_geometry_constraint):
native PushCube pins the goal at +x for every seed, so raw seed-change creates no
geometry mismatch; we vary the goal DIRECTION via the runner's injection
mechanism. The +x-tuned open-loop controller only pushes the x-axis reliably
(+x/-x 100%, +y 0%), so the natural loop is a binary +x vs -x problem.

What this measures — recovery under a wrong PUSH DIRECTION, across revisers:

  * same_intent     — retry the identical (stale) intent.            [open-loop]
  * rule_orthogonal — revision.revise_intent(contact_region): the Stage-0
                      rule picks the 90-deg ORTHOGONAL face, ignoring the
                      failure direction.                              [open-loop]
  * feedback_flip   — use the observed displacement-vector feedback to flip the
                      push to the OPPOSITE face (non-privileged heuristic).
  * oracle_value    — set contact_region to the exec-scene-correct face
                      (robot observes its own goal at exec time).   [upper bound]

The headline: under the honest seed-mismatch paradigm the OPEN-LOOP revisers
(same_intent, rule_orthogonal) do NOT recover — only revisers that consume the
EXECUTION FEEDBACK (feedback_flip / oracle_value) do. The block hid this because
any unblocked approach trivially worked. This motivates the learned,
feedback-conditioned ReviseHead (Step 3).

GPU/Vulkan for the real PushCube runner; the core loop is sim-free and unit-tested
against tests.conftest.FakeEnvRunner (whose reset varies direction by seed%4).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from babysteps.envs.scene import (  # noqa: E402
    direction_to_face, face_to_push_unit, motion_to_unit,
)
from babysteps.episode import generate_proxy_demo  # noqa: E402
from babysteps.failure import Attribution  # noqa: E402
from babysteps.revision import revise_intent as rule_revise_intent  # noqa: E402
from babysteps.schemas import INTENT_FIELDS, Intent  # noqa: E402

_X_MOTIONS = ("translate_+x", "translate_-x")
_MOTIONS_BY_SET = {
    "x": ("translate_+x", "translate_-x"),
    "xy": ("translate_+x", "translate_-x", "translate_+y", "translate_-y"),
}
_OPPOSITE_MOTION = {
    "translate_+x": "translate_-x", "translate_-x": "translate_+x",
    "translate_+y": "translate_-y", "translate_-y": "translate_+y",
}
_OPPOSITE_FACE = {
    "minus_x_face": "plus_x_face", "plus_x_face": "minus_x_face",
    "minus_y_face": "plus_y_face", "plus_y_face": "minus_y_face",
}


def _stable_choice(seed: int, salt: str, options: tuple[str, ...]) -> str:
    h = int(hashlib.sha256(f"{seed}:{salt}".encode()).hexdigest()[:8], 16)
    return options[h % len(options)]


def _scene_dir_face(scene) -> str:
    """The correct push face for a scene = face toward (goal - cube)."""
    return direction_to_face(np.asarray(scene.goal_xy) - np.asarray(scene.cube_xy))


# ---------- revisers (the comparison set) ----------------------------------- #

def _rev_same_intent(initial: Intent, fp, scene, adapter) -> Intent:
    return initial


def _rev_rule_orthogonal(initial: Intent, fp, scene, adapter) -> Intent:
    """Stage-0 rule-based contact_substitution (picks the orthogonal face)."""
    attr = Attribution(
        semantic_failure=True, wrong_factor="contact_region",
        freeze=tuple(f for f in INTENT_FIELDS if f != "contact_region"),
        revise=("contact_region",),
    )
    revised, _rec = rule_revise_intent(initial, attr, scene)
    return revised


def _rev_feedback_flip(initial: Intent, fp, scene, adapter) -> Intent:
    """Non-privileged: read the displacement-vector feedback (where the cube
    ACTUALLY went) and flip the contact to push the OPPOSITE way. Uses only the
    observed object motion, never the goal coordinate handed to the probe."""
    vec = fp.object_displacement_vec
    if vec is None or (abs(vec[0]) < 1e-9 and abs(vec[1]) < 1e-9):
        return initial
    # The cube drifted along `vec` (~the wrong push direction). The corrective
    # push is the opposite: choose the face whose push_unit points along -vec.
    target_unit = -np.asarray(vec, dtype=np.float64)
    faces = ("minus_x_face", "plus_x_face", "minus_y_face", "plus_y_face")
    best = max(faces, key=lambda f: float(np.dot(face_to_push_unit(f), target_unit)))
    return replace(initial, contact_region=best)


def _rev_feedback_residual(initial: Intent, fp, scene, adapter) -> Intent:
    """Non-privileged, 4-way: read the RESIDUAL (where the cube still needs to
    go) = goal - final_cube, both observable in the robot's exec view, and pick
    the face that pushes that way. Unlike feedback_flip (which only reverses, so
    it fails a perpendicular +x->+y mismatch), the residual recovers ANY
    cardinal mismatch."""
    vec = fp.object_displacement_vec or (0.0, 0.0)
    cube0 = np.asarray(scene.cube_xy, dtype=np.float64)
    final = cube0 + np.asarray(vec, dtype=np.float64)
    residual = np.asarray(scene.goal_xy, dtype=np.float64) - final
    if float(np.linalg.norm(residual)) < 1e-6:
        return initial
    return replace(initial, contact_region=direction_to_face(residual))


def _rev_oracle_value(initial: Intent, fp, scene, adapter) -> Intent:
    """Upper bound: set contact_region to the exec-scene-correct face. The
    executing robot observes its own goal, so this is execution-time
    information, not demo-path privilege."""
    correct = adapter.oracle_correct_intent(scene)
    return replace(initial, contact_region=correct.contact_region)


REVISERS = {
    "same_intent": _rev_same_intent,
    "rule_orthogonal": _rev_rule_orthogonal,
    "feedback_flip": _rev_feedback_flip,
    "feedback_residual": _rev_feedback_residual,
    "oracle_value": _rev_oracle_value,
}


def _observed_residual(fp, scene) -> np.ndarray:
    """The goal-relative residual = goal - final_cube, both observable in the
    robot's exec view (final = cube0 + observed displacement). Shared by the
    feedback_residual hand rule and the latent_learned head so they consume the
    identical non-privileged signal."""
    vec = fp.object_displacement_vec or (0.0, 0.0)
    final = np.asarray(scene.cube_xy, dtype=np.float64) + np.asarray(vec, dtype=np.float64)
    return np.asarray(scene.goal_xy, dtype=np.float64) - final


def make_latent_learned_reviser(pack_dir, residual_head_path,
                                *, factor: str = "contact_region"):
    """The LEARNED counterpart of `feedback_residual`: a residual-conditioned
    ReviseHead edits the implicated slot in the vision-grounded latent space and
    nearest-centroid-decodes the corrected face — no hand-coded direction_to_face.

    g_slot = centroid[demo_face] (the vision-grounded latent representative of the
    stale demo face); fp = factor+predicate one-hot + the observed residual unit
    vector; head -> revised g_slot -> decode_slot -> corrected face token. Uses
    the SAME observed residual as the feedback_residual hand rule, so a match to
    it (and to oracle) is evidence the learned edit is correct."""
    import torch

    from babysteps.stage4.latent_policy import load_latent_pack
    from babysteps.stage4.revise_head import (
        load_revise_head, vectorize_failure_packet_residual,
    )
    from babysteps.stage4.slot_decode import decode_slot

    pack = load_latent_pack(pack_dir)
    head = load_revise_head(residual_head_path)
    fi = INTENT_FIELDS.index(factor)
    if fi not in pack.centroids:
        raise ValueError(f"pack {pack_dir} has no centroids for {factor}")
    centroids = pack.centroids[fi]
    tokens = pack.label_tokens[fi]
    tok2cls = {t: i for i, t in enumerate(tokens)}

    def _rev(initial: Intent, fp, scene, adapter) -> Intent:
        demo_face = getattr(initial, factor)
        if demo_face not in tok2cls:
            return initial  # face outside the learned latent vocab
        g_slot = np.asarray(centroids[tok2cls[demo_face]], dtype=np.float32)
        residual = _observed_residual(fp, scene)
        rec = {"revision": {"factor": factor},
               "failure_packet": {
                   "failure_predicate": fp.failure_predicate or "direction_error"}}
        fp_vec = vectorize_failure_packet_residual(rec, residual)
        with torch.no_grad():
            g_rev = head(torch.tensor(g_slot).unsqueeze(0),
                         torch.tensor(fp_vec).unsqueeze(0)).numpy()[0]
        cls = decode_slot(g_rev, centroids)
        return replace(initial, **{factor: tokens[cls]})

    return _rev


# ---------- the natural-failure episode ------------------------------------- #

def run_natural_episode(adapter, runner, *, demo_seed: int, exec_seed: int,
                        demo_motion: str | None, exec_motion: str | None,
                        revisers: list[str],
                        reviser_fns: dict | None = None,
                        tuple_sink: list | None = None,
                        vision_extractor=None,
                        vision_features_dir=None,
                        vision_suffix: str = "dinov2") -> dict:
    """One demo(A) -> intent -> exec(B, no block) -> fail -> revise -> retry.

    Returns per-reviser final success plus the initial attempt and geometry.
    `demo_motion`/`exec_motion` are injected when the runner supports it (real
    PushCube, whose native goal is fixed at +x); for the FakeEnvRunner they are
    None and the direction comes from seed%4.

    `reviser_fns` overrides the module-level REVISERS map (e.g. to add the
    runtime-built latent_learned reviser). `tuple_sink`, when given, collects a
    residual-head TRAINING tuple {demo_face, correct_face, residual_xy,
    failure_predicate} for every mismatched episode whose initial attempt failed
    (`scripts/stage5_train_residual_revise_head.py` consumes these).
    """
    fns = reviser_fns or REVISERS
    has_inj = hasattr(runner, "set_injection")

    # ---- demo phase (instance A) ----
    # Vision mode: the initial intent is decoded from the cached demo CLIP feature
    # (full-vision closed loop) instead of the scripted sim-state evidence. The
    # demo's true direction comes from demo_motion; the vision-decoded contact may
    # differ (decode error), which the latent_learned reviser must then recover.
    vision_decoded_contact = None
    vision_decode_correct = None
    if vision_extractor is not None:
        demo_face = direction_to_face(motion_to_unit(demo_motion))  # true demo dir
        initial_intent = vision_extractor.decode_from_cache(
            vision_features_dir, demo_seed, demo_motion, vision_suffix)
        vision_decoded_contact = initial_intent.contact_region
        vision_decode_correct = bool(vision_decoded_contact == demo_face)
    else:
        if has_inj:
            runner.set_injection(demo_motion)
        scene_demo = runner.reset(demo_seed)
        demo_evidence = generate_proxy_demo(runner, scene_demo, adapter)
        initial_intent = adapter.scripted_demo_to_intent(demo_evidence)
        demo_face = _scene_dir_face(scene_demo)

    # ---- exec phase (instance B; NO artificial block) ----
    if has_inj:
        runner.set_injection(exec_motion)
    scene_exec = runner.reset(exec_seed)
    scene_executor = replace(scene_exec, blocked_sides=())  # natural mode
    exec_face = _scene_dir_face(scene_executor)

    attempt_1 = runner.run(initial_intent, scene_executor)
    fp = adapter.build_failure_packet(initial_intent, attempt_1, scene_executor)

    out = {
        "demo_seed": demo_seed, "exec_seed": exec_seed,
        "demo_face": demo_face, "exec_face": exec_face,
        "direction_mismatch": demo_face != exec_face,
        "initial_intent_contact": initial_intent.contact_region,
        "initial_success": bool(attempt_1.success),
        "failure_predicate": fp.failure_predicate,
        "displacement_vec": fp.object_displacement_vec,
        "vision_decoded_contact": vision_decoded_contact,
        "vision_decode_correct": vision_decode_correct,
        "revisers": {},
    }

    # Collect a residual-head training tuple for failed mismatched episodes.
    # correct_face is a sim-derived TRAINING label (allowed off the demo->intent
    # path, CLAUDE.md inv #4); residual_xy is the non-privileged exec feedback.
    if tuple_sink is not None and not attempt_1.success:
        residual = _observed_residual(fp, scene_executor)
        correct = adapter.oracle_correct_intent(scene_executor)
        tuple_sink.append({
            "demo_seed": demo_seed, "exec_seed": exec_seed,
            "demo_face": initial_intent.contact_region,
            "correct_face": correct.contact_region,
            "residual_xy": [float(residual[0]), float(residual[1])],
            "failure_predicate": fp.failure_predicate,
        })

    for name in revisers:
        if attempt_1.success:
            out["revisers"][name] = {"final_success": True, "revised": False,
                                     "new_contact": initial_intent.contact_region}
            continue
        revised = fns[name](initial_intent, fp, scene_executor, adapter)
        attempt_2 = runner.run(revised, scene_executor)
        out["revisers"][name] = {
            "final_success": bool(attempt_2.success),
            "revised": revised.contact_region != initial_intent.contact_region,
            "new_contact": revised.contact_region,
        }
    return out


def summarize(rows: list[dict], revisers: list[str]) -> dict:
    n = len(rows)
    mism = [r for r in rows if r["direction_mismatch"]]
    fails = [r for r in rows if not r["initial_success"]]  # initial-attempt failures
    summ = {
        "n": n, "n_mismatch": len(mism), "n_initial_fail": len(fails),
        "initial_success_rate": sum(r["initial_success"] for r in rows) / max(1, n),
        "final_success_rate": {},
        "final_success_rate_on_mismatch": {},
        "final_success_rate_on_initial_fail": {},
    }
    # Vision-decode accuracy (only populated in --vision-intent mode).
    vdc = [r["vision_decode_correct"] for r in rows
           if r.get("vision_decode_correct") is not None]
    if vdc:
        summ["vision_decode_acc"] = sum(vdc) / len(vdc)
    for name in revisers:
        summ["final_success_rate"][name] = (
            sum(r["revisers"][name]["final_success"] for r in rows) / max(1, n))
        summ["final_success_rate_on_mismatch"][name] = (
            sum(r["revisers"][name]["final_success"] for r in mism) / max(1, len(mism)))
        summ["final_success_rate_on_initial_fail"][name] = (
            sum(r["revisers"][name]["final_success"] for r in fails) / max(1, len(fails)))
    return summ


def _parse_seed_range(s: str) -> list[int]:
    if "-" in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(s)]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="PushCube-v1", choices=["PushCube-v1"])
    p.add_argument("--axes", choices=["x", "xy"], default="x",
                   help="x = binary +x/-x (controller default); xy = 4-way "
                        "(requires orient_control, the gripper-yaw fix).")
    p.add_argument("--eval-seeds", default="100-149",
                   help="Held-out EXEC seed range. The demo seed is exec_seed "
                        "offset by --demo-seed-offset (a DIFFERENT instance).")
    p.add_argument("--demo-seed-offset", type=int, default=500,
                   help="demo_seed = exec_seed + offset (decoupled instance).")
    p.add_argument("--mismatch", choices=["random", "always", "never"],
                   default="random",
                   help="random: per-seed independent +x/-x for demo & exec "
                        "(~50%% natural failure); always: exec is the opposite "
                        "of demo (guaranteed failure); never: matched (control).")
    p.add_argument("--revisers",
                   default="same_intent,rule_orthogonal,feedback_flip,"
                           "feedback_residual,oracle_value",
                   help="Comma-separated reviser names. Include 'latent_learned' "
                        "(needs --pack-dir + --residual-head) for the learned "
                        "residual-conditioned ReviseHead in latent space.")
    p.add_argument("--pack-dir", type=Path, default=None,
                   help="4-way LatentPack dir (stage5_train_4way_pack.py) for "
                        "the latent_learned reviser.")
    p.add_argument("--residual-head", type=Path, default=None,
                   help="revise_head_residual.pt (stage5_train_residual_revise_head.py).")
    p.add_argument("--dump-tuples", type=Path, default=None,
                   help="Write residual-head training tuples (one per failed "
                        "mismatched episode) to this jsonl instead of/alongside "
                        "the normal eval. Use on TRAINING seeds.")
    p.add_argument("--vision-intent", action="store_true",
                   help="FULL-VISION loop: decode the initial intent from the "
                        "cached demo CLIP feature (VisionIntentExtractor over "
                        "--pack-dir) instead of the scripted sim-state evidence. "
                        "Requires --pack-dir + --vision-features-dir.")
    p.add_argument("--vision-features-dir", type=Path, default=None,
                   help="Dir of cached per-(demo_seed,direction) demo features "
                        "(seed_NNNN_<tag>_<suffix>.npy).")
    p.add_argument("--vision-encoder", default="dinov2",
                   choices=["dinov2", "vjepa21"],
                   help="Feature-file suffix for --vision-intent (default dinov2).")
    p.add_argument("--out-dir", type=Path,
                   default=Path("reports/stage5/natural_loop/PushCube-v1"))
    p.add_argument("--fake", action="store_true",
                   help="Sim-free smoke via tests.conftest.FakeEnvRunner.")
    args = p.parse_args(argv)

    revisers = [r.strip() for r in args.revisers.split(",") if r.strip()]
    exec_seeds = _parse_seed_range(args.eval_seeds)

    # Build the reviser map; add the runtime latent_learned head when requested.
    reviser_fns = dict(REVISERS)
    if "latent_learned" in revisers:
        if args.pack_dir is None or args.residual_head is None:
            p.error("latent_learned requires --pack-dir and --residual-head")
        reviser_fns["latent_learned"] = make_latent_learned_reviser(
            args.pack_dir, args.residual_head)
    unknown = [r for r in revisers if r not in reviser_fns]
    if unknown:
        p.error(f"unknown reviser(s): {unknown}")
    tuple_sink: list | None = [] if args.dump_tuples is not None else None

    # Full-vision: build the VisionIntentExtractor (decodes the initial intent
    # from the cached demo CLIP feature). Built after the adapter exists (needs a
    # template intent for the task-constant factors).
    vision_extractor = None
    if args.vision_intent:
        if args.pack_dir is None or args.vision_features_dir is None:
            p.error("--vision-intent requires --pack-dir and --vision-features-dir")
        if args.fake:
            p.error("--vision-intent is not supported with --fake")

    if args.fake:
        from tests.conftest import FakeEnvRunner
        from babysteps.envs.pushcube_adapter import PushCubeAdapter
        adapter = PushCubeAdapter()
        runner = FakeEnvRunner()
    else:
        from babysteps.envs.pushcube_adapter import PushCubeAdapter
        from babysteps.envs.pushcube_runner import PushCubeEnvRunner
        adapter = PushCubeAdapter()
        # orient_control=True so y-axis pushes present a flat face (the 4-way
        # fix). For --axes x this is a no-op (push_yaw_deg=0). For --axes xy it
        # is required. The committed data path constructs the runner without it.
        runner = PushCubeEnvRunner(orient_control=True)

    if args.vision_intent:
        from babysteps.stage4.vision_intent import VisionIntentExtractor
        # Template = task-constant factors (goal_state/constraint_region/
        # embodiment_mapping/direction_grounding are scene-invariant for PushCube);
        # the grounded factors get overwritten by the vision decode.
        ref_scene = replace(runner.reset(args.demo_seed_offset), blocked_sides=())
        template = adapter.oracle_correct_intent(ref_scene)
        vision_extractor = VisionIntentExtractor.from_pack(args.pack_dir, template)
        print(f"vision-intent ON: pack={args.pack_dir} "
              f"features={args.vision_features_dir} suffix={args.vision_encoder}")

    rows = []
    for es in exec_seeds:
        demo_seed = es + args.demo_seed_offset
        if args.fake:
            # FakeEnvRunner has no injection; its direction is fixed by seed%4
            # (0:+x, 1:+y, 2:-x, 3:-y). Direction control comes from seed choice,
            # not motions. The careful x-axis pairing lives in the unit test;
            # here we just smoke the plumbing on whatever seed%4 yields.
            demo_motion = exec_motion = None
            es_eff = es
        else:
            motions = _MOTIONS_BY_SET[args.axes]
            demo_motion = _stable_choice(demo_seed, "demo", motions)
            if args.mismatch == "always":
                exec_motion = _OPPOSITE_MOTION[demo_motion]
            elif args.mismatch == "never":
                exec_motion = demo_motion
            else:
                exec_motion = _stable_choice(es, "exec", motions)
            es_eff = es

        row = run_natural_episode(
            adapter, runner, demo_seed=demo_seed, exec_seed=es_eff,
            demo_motion=demo_motion, exec_motion=exec_motion, revisers=revisers,
            reviser_fns=reviser_fns, tuple_sink=tuple_sink,
            vision_extractor=vision_extractor,
            vision_features_dir=args.vision_features_dir,
            vision_suffix=args.vision_encoder)
        rows.append(row)

    summ = summarize(rows, revisers)
    print(f"=== natural loop ({args.task}, mismatch={args.mismatch}, "
          f"fake={args.fake}, n={summ['n']}, n_mismatch={summ['n_mismatch']}, "
          f"n_initial_fail={summ['n_initial_fail']}) ===")
    if "vision_decode_acc" in summ:
        print(f"  vision-decode accuracy: {summ['vision_decode_acc']:.3f}")
    print(f"  initial success: {summ['initial_success_rate']:.3f}")
    for name in revisers:
        print(f"  final[{name:16s}] all={summ['final_success_rate'][name]:.3f}  "
              f"on_mismatch={summ['final_success_rate_on_mismatch'][name]:.3f}  "
              f"on_fail={summ['final_success_rate_on_initial_fail'][name]:.3f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "natural_loop_results.json").write_text(json.dumps(
        {"config": {"task": args.task, "eval_seeds": args.eval_seeds,
                    "demo_seed_offset": args.demo_seed_offset,
                    "mismatch": args.mismatch, "fake": args.fake,
                    "revisers": revisers,
                    "vision_intent": bool(args.vision_intent),
                    "vision_encoder": args.vision_encoder if args.vision_intent else None,
                    "axes": args.axes},
         "summary": summ, "rows": rows}, indent=2) + "\n")
    print(f"\nwrote {args.out_dir}/natural_loop_results.json")

    if tuple_sink is not None:
        args.dump_tuples.parent.mkdir(parents=True, exist_ok=True)
        with args.dump_tuples.open("w") as f:
            for t in tuple_sink:
                f.write(json.dumps(t) + "\n")
        print(f"wrote {len(tuple_sink)} residual-head tuples to {args.dump_tuples}")

    if hasattr(adapter, "close"):
        adapter.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
