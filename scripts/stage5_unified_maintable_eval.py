"""Stage-5 — UNIFIED main-table evaluator (PushCube + StackCube, one harness).

Build-order step 1 of the elevated target (a shared, task-general revision
policy; see ``redesign_failure_paradigm.md``). This merges the two separate
natural-loop evaluators (``stage5_natural_loop_eval.py`` for PushCube,
``stage5_goalstate_loop_eval.py`` for StackCube) into ONE harness that, on
held-out seeds with NATURALLY-sourced failures (no ``blocked_sides``), runs the
same conditions and emits the same metrics so the two tasks are directly
comparable.

Conditions (``babysteps.stage5.conditions``): ``same_intent_retry``,
``random_factor_local_edit``, ``vlm_free_replan``, ``vlm_diagnosis_local_edit``,
``oracle_single_slot`` (runnable now); ``shared_revision_policy`` (the proposed
method — build-order step 2 — reported as deferred).

Attribution and value-revision are kept separate behind the no-leakage
interface in ``babysteps.stage5.revision_policy``: a policy sees only the
diagnosed factor + observable failure evidence + typed candidates, never the
task id / scene / gt / full intent. The evaluator-side compiler applies the
typed decision and enforces exactly one changed slot.

Per-condition metrics (``babysteps.stage5.maintable``): recovery (overall +
on-initial-fail), preservation, unnecessary/harmful change rates, edit
cardinality, decision latency split (diagnosis / revision / joint-reasoning /
total) + rollout latency, and VLM call/token cost.

Sim-free smoke (login node, no GPU/Vulkan)::

    python scripts/stage5_unified_maintable_eval.py --task StackCube-v1 \\
        --fake --mock --eval-seeds 200-205

The real run (GPU; InternVL + ManiSkill, vision-decoded initial intents) is
build-order step 5 and reuses this same file.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Optional

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from babysteps.schemas import INTENT_FIELDS, Intent  # noqa: E402
from babysteps.stage5.conditions import (  # noqa: E402
    CONDITION_REGISTRY, CONDITIONS, available_conditions, deferred_conditions,
)
from babysteps.stage5.maintable import aggregate, per_condition_metrics  # noqa: E402
from babysteps.stage5.residual_reviser import _observed_residual  # noqa: E402
from babysteps.stage5.revision_policy import (  # noqa: E402
    AttributionObs, FailureEvidence, OracleAttributor, OracleValuePolicy,
    PerTaskEditorAdapter, RandomAttributor, RandomCandidatePolicy,
    RevisionRequest, TYPED_OPERATORS, VLMAttributor, candidates_for,
    compile_single_slot_edit,
)


# --------------------------------------------------------------------------- #
# Episode materials (evaluator-only state) + per-task sourcing
# --------------------------------------------------------------------------- #

@dataclass
class EpisodeData:
    seed: int
    initial: Intent          # vision-decoded (or fake) initial intent
    scene_exec: object       # evaluator-only
    fp: object               # evaluator-only failure packet
    gt: Intent               # evaluator-only oracle correct intent
    initial_success: bool
    e_fail: FailureEvidence  # MODEL-VISIBLE observable evidence
    frame_path: Optional[str] = None
    wrist_frame_path: Optional[str] = None


@dataclass
class TaskSpec:
    task: str
    implicated_factor: str
    factor_menu: tuple[str, ...]
    source_episode: Callable[[int], EpisodeData]
    editor_producers: dict  # {factor: (req) -> token}
    # Optional {factor: candidate tuple} override for the model-visible request,
    # used when a task exposes a restricted candidate set (e.g. held-out PokeCube
    # exposes only the 3 REACHABLE faces, not the full 4-face vocab). Defaults to
    # the schema/task vocab via candidates_for (backward compatible).
    candidates_override: Optional[dict] = None


# --------------------------------------------------------------------------- #
# Condition execution (uniform across tasks)
# --------------------------------------------------------------------------- #

def _run_condition(cond: str, ep: EpisodeData, spec: TaskSpec, runner,
                   vlm, seed: int, shared_policy=None,
                   attributor_override=None, distilled_attributor=None) -> dict:
    """Run one condition on one episode → a per-condition metrics row.

    ``attributor_override`` (``None`` | ``"oracle"`` | ``"distilled"``) forces
    the attributor for the paired conditions (shared_revision_policy /
    vlm_diagnosis_local_edit): ``"oracle"`` substitutes the privileged
    OracleAttributor so the row isolates the VALUE policy (held-out value
    transfer) from attribution quality; ``"distilled"`` substitutes the loaded
    DistilledAttributor (the proposed VLM-free diagnosis — the recovery gate).
    The default (``None``) uses each condition's native attributor (VLM)."""
    cspec = CONDITION_REGISTRY[cond]
    menu = spec.factor_menu
    implicated = spec.implicated_factor

    # Initial already succeeded → no revision; every condition trivially holds.
    if ep.initial_success:
        return per_condition_metrics(
            initial=ep.initial, revised=ep.initial, gt=ep.gt,
            implicated_factor=implicated, factor_menu=menu,
            initial_success=True, retry_success=None,
        )

    revised: Optional[Intent] = ep.initial
    diag = rev = joint = 0.0
    vlm_cost = None
    attribution_correct: Optional[bool] = None

    if cspec.kind == "identity":
        revised = ep.initial

    elif cspec.kind == "oracle":
        attr = OracleAttributor(implicated).attribute(_obs(spec, ep, seed))
        policy = OracleValuePolicy(getattr(ep.gt, implicated))
        decision = policy.decide(_request(spec, ep, attr.factor))
        revised = compile_single_slot_edit(ep.initial, decision, menu)
        attribution_correct = (attr.factor == implicated)

    elif cspec.kind == "free_replan":
        if hasattr(vlm, "reset_cost"):
            vlm.reset_cost()
        full, _raw = vlm.diagnose_free_form_verbose(
            task=spec.task, image_path=ep.frame_path,
            initial_intent=ep.initial, failure_predicate=ep.e_fail.predicate,
            wrist_image_path=ep.wrist_frame_path,
        )
        vlm_cost = vlm.cost_snapshot() if hasattr(vlm, "cost_snapshot") else None
        joint = float(vlm_cost["latency_s"]) if vlm_cost else 0.0
        revised = full if full is not None else None  # parse-fail → no change

    elif cspec.kind == "paired":
        attributor, policy = _paired_actors(
            cond, spec, ep, vlm, seed, shared_policy=shared_policy,
            attributor_override=attributor_override,
            distilled_attributor=distilled_attributor)
        attr = attributor.attribute(_obs(spec, ep, seed))
        diag = float(attr.latency_s)
        vlm_cost = dict(attr.cost) if attr.cost else None
        if attr.factor is None:
            revised = None  # diagnosis parse-fail → no edit
        else:
            t0 = time.perf_counter()
            decision = policy.decide(_request(spec, ep, attr.factor))
            rev = time.perf_counter() - t0
            revised = compile_single_slot_edit(ep.initial, decision, menu)
        attribution_correct = (attr.factor == implicated)
    else:
        raise ValueError(f"unknown condition kind {cspec.kind!r}")

    # Rollout the retry (evaluator-side; rollout latency measured here).
    retry_success: Optional[bool] = None
    if revised is not None:
        t0 = time.perf_counter()
        attempt = runner.run(revised, ep.scene_exec)
        rollout = time.perf_counter() - t0
        retry_success = bool(attempt.success)
    else:
        rollout = 0.0  # no revision produced → no retry

    return per_condition_metrics(
        initial=ep.initial, revised=revised, gt=ep.gt,
        implicated_factor=implicated, factor_menu=menu,
        initial_success=False, retry_success=retry_success,
        diagnosis_latency_s=diag, revision_latency_s=rev,
        joint_reasoning_latency_s=joint, rollout_latency_s=rollout,
        vlm_cost=vlm_cost, attribution_correct=attribution_correct,
    )


def _obs(spec: TaskSpec, ep: EpisodeData, seed: int) -> AttributionObs:
    return AttributionObs(
        task=spec.task, factor_menu=spec.factor_menu,
        failure_predicate=ep.e_fail.predicate, initial_intent=ep.initial,
        frame_path=ep.frame_path, wrist_frame_path=ep.wrist_frame_path,
        key=seed,
        # Additive multimodal evidence (consumed only by DistilledAttributor;
        # VLM/oracle/random ignore it). The residual is the observable
        # goal-relative displacement already on the failure packet.
        residual_xy=(ep.e_fail.residual_xy if ep.e_fail else None))


def _request(spec: TaskSpec, ep: EpisodeData, factor: str) -> RevisionRequest:
    """Build the MODEL-VISIBLE request: no task id / scene / gt / full intent."""
    override = (spec.candidates_override or {}).get(factor)
    candidates = (override if override is not None
                  else candidates_for(spec.task, factor))
    return RevisionRequest(
        factor=factor, current_value=getattr(ep.initial, factor),
        candidates=candidates, e_fail=ep.e_fail, g_i=None, z=None)


def _paired_actors(cond: str, spec: TaskSpec, ep: EpisodeData, vlm, seed: int,
                   shared_policy=None, attributor_override=None,
                   distilled_attributor=None):
    if cond == "random_factor_local_edit":
        return RandomAttributor(seed=0), RandomCandidatePolicy(seed=0)

    def _attributor():
        # "oracle" → privileged OracleAttributor (isolates the value policy);
        # "distilled" → the loaded VLM-free DistilledAttributor (recovery gate);
        # default → the VLM teacher/baseline (the deployed-with-VLM path).
        if attributor_override == "oracle":
            return OracleAttributor(spec.implicated_factor)
        if attributor_override == "distilled":
            if distilled_attributor is None:
                raise ValueError(
                    "attributor_override='distilled' needs a loaded "
                    "--distilled-head (DistilledAttributor)")
            return distilled_attributor
        return VLMAttributor(vlm)

    if cond == "vlm_diagnosis_local_edit":
        return _attributor(), PerTaskEditorAdapter(spec.editor_producers)
    if cond == "shared_revision_policy":
        # The proposed method: shared attributor with vlm_diagnosis_local_edit
        # (so the comparison isolates the VALUE policy — one shared scorer vs the
        # hand-built per-task editors), the ONE frozen shared scorer for the
        # value. Loaded once and threaded through (no per-episode load).
        if shared_policy is None:
            raise ValueError(
                "shared_revision_policy requires a loaded --scorer checkpoint")
        return _attributor(), shared_policy
    raise ValueError(f"no paired actors for condition {cond!r}")


# --------------------------------------------------------------------------- #
# Per-task specs
# --------------------------------------------------------------------------- #

def _build_contact_region_spec(task_id, args, runner, adapter) -> TaskSpec:
    """contact_region natural-loop spec, shared by PushCube and PokeCube (step 3).

    Both families revise contact_region with the SAME 4-face candidate vocab and
    the SAME 2D-residual->face rule (consumed by ResidualSlotEditor /
    SharedScorerPolicy). The poke-vs-push difference lives only in the GPU runner;
    sim-free, the two are structurally identical (deterministic opposite-face
    failure), which is exactly what makes them a valid leave-one-FAMILY-out pair
    for the shared contact_region rule."""
    from babysteps.stage5.residual_reviser import ResidualSlotEditor

    menu = INTENT_FIELDS
    _opposite_face = {
        "minus_x_face": "plus_x_face", "plus_x_face": "minus_x_face",
        "minus_y_face": "plus_y_face", "plus_y_face": "minus_y_face",
    }

    if args.fake:
        editor = _synthetic_pushcube_editor()
    else:
        editor = ResidualSlotEditor.from_pack(
            args.pack_dir, args.residual_head, factor="contact_region")

    def source_episode(seed: int) -> EpisodeData:
        if args.fake:
            # Deterministic natural-style failure: the demo grounded the OPPOSITE
            # push face, so the initial attempt drives the cube away from the
            # goal (no block injected). Mirrors the seed-decoupled mismatch of
            # the real loop without needing vision features on the login node.
            scene_exec = replace(runner.reset(seed), blocked_sides=())
            gt0 = adapter.oracle_correct_intent(scene_exec)
            initial = replace(
                gt0, contact_region=_opposite_face.get(
                    gt0.contact_region, gt0.contact_region))
        else:
            # Real path (step 5): vision-decode the initial contact face and use
            # the injection mechanism for the +x/-x exec direction.
            initial, scene_exec = _pushcube_real_initial(args, runner, adapter, seed)
        attempt = runner.run(initial, scene_exec)
        fp = adapter.build_failure_packet(initial, attempt, scene_exec)
        gt = adapter.oracle_correct_intent(scene_exec)
        residual = _observed_residual(fp, scene_exec)
        return EpisodeData(
            seed=seed, initial=initial, scene_exec=scene_exec, fp=fp, gt=gt,
            initial_success=bool(attempt.success),
            e_fail=FailureEvidence(
                predicate=fp.failure_predicate,
                residual_xy=(float(residual[0]), float(residual[1]))),
            frame_path=None, wrist_frame_path=None)

    return TaskSpec(
        task=task_id, implicated_factor="contact_region",
        factor_menu=menu, source_episode=source_episode,
        editor_producers={
            "contact_region": lambda req: editor(
                req.current_value, req.e_fail.residual_xy, req.e_fail.predicate)})


def _build_stackcube_spec(args, runner, adapter) -> TaskSpec:
    from babysteps.schemas import INTENT_FIELDS

    def source_episode(seed: int) -> EpisodeData:
        scene_exec = runner.reset(seed)
        gt = adapter.oracle_correct_intent(scene_exec)
        if args.fake:
            # Natural under-specification proxy: the demo grounds the ambiguous
            # `cube_at_target` (place-near) instead of `cubeA_on_cubeB` (stack).
            initial = replace(gt, goal_state="cube_at_target")
        else:
            initial = _stackcube_real_initial(args, adapter, scene_exec, seed)
        attempt = runner.run(initial, scene_exec)
        fp = adapter.build_failure_packet(initial, attempt, scene_exec)
        return EpisodeData(
            seed=seed, initial=initial, scene_exec=scene_exec, fp=fp, gt=gt,
            initial_success=bool(attempt.success),
            e_fail=FailureEvidence(predicate=fp.failure_predicate,
                                   residual_xy=None),
            frame_path=None, wrist_frame_path=None)

    return TaskSpec(
        task="StackCube-v1", implicated_factor="goal_state",
        factor_menu=INTENT_FIELDS, source_episode=source_episode,
        editor_producers={
            "goal_state": lambda req: TYPED_OPERATORS["goal_refinement"].get(
                req.current_value)})


def _synthetic_pushcube_editor():
    """Tiny sim-free ResidualSlotEditor for the --fake smoke (untrained head;
    structural plumbing only). The LEARNED-edit correctness is asserted in
    tests/test_stage5_revision_policy.py with a trained head."""
    from babysteps.stage4.revise_head import FP_VECTOR_DIM_RESIDUAL, ReviseHead
    from babysteps.stage5.residual_reviser import ResidualSlotEditor
    faces = ("minus_x_face", "plus_x_face", "minus_y_face", "plus_y_face")
    centroids = {i: np.eye(4, dtype=np.float32)[i] for i in range(4)}
    head = ReviseHead(d_slot=4, fp_dim=FP_VECTOR_DIM_RESIDUAL, hidden=8, seed=0)
    return ResidualSlotEditor(factor="contact_region", centroids=centroids,
                              tokens=faces, head=head)


def _pushcube_real_initial(args, runner, adapter, seed: int):
    """Real PushCube initial intent: vision-decode the contact face + inject the
    exec push direction (build-order step 5; not exercised by --fake tests)."""
    from babysteps.stage4.vision_intent import VisionIntentExtractor
    import hashlib

    def _choice(s, salt, opts):
        h = int(hashlib.sha256(f"{salt}:{s}".encode()).hexdigest()[:8], 16)
        return opts[h % len(opts)]

    motions = ("translate_+x", "translate_-x")
    demo_seed = seed + args.demo_seed_offset
    demo_motion = _choice(demo_seed, "demo", motions)
    exec_motion = (_choice(seed, "exec", motions) if args.mismatch == "random"
                   else {"always": {"translate_+x": "translate_-x",
                                    "translate_-x": "translate_+x"}[demo_motion],
                         "never": demo_motion}[args.mismatch])
    ref = replace(runner.reset(demo_seed), blocked_sides=())
    template = adapter.oracle_correct_intent(ref)
    extractor = VisionIntentExtractor.from_pack(args.pack_dir, template)
    initial = extractor.decode_from_cache(
        args.vision_features_dir, demo_seed, demo_motion, args.vision_encoder)
    runner.set_injection(exec_motion)
    scene_exec = replace(runner.reset(seed), blocked_sides=())
    return initial, scene_exec


def _stackcube_real_initial(args, adapter, scene_exec, seed: int):
    """Real StackCube initial: vision-decode goal_state from the demo clip."""
    from babysteps.schemas import INTENT_FIELDS
    from babysteps.stage4.vision_intent import VisionIntentExtractor
    gt = adapter.oracle_correct_intent(scene_exec)
    extractor = VisionIntentExtractor.from_pack(args.pack_dir, gt)
    feat = np.load(Path(args.vision_features_dir) /
                   f"seed_{seed:04d}_stack_{args.feature_suffix}.npy")
    decoded = extractor.decode_factor(feat, INTENT_FIELDS.index("goal_state"))
    return replace(gt, goal_state=decoded)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def _parse_seed_range(s: str) -> list[int]:
    if "-" in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(s)]


def _make_runner_adapter(task: str, fake: bool):
    if task == "PushCube-v1":
        from babysteps.envs.pushcube_adapter import PushCubeAdapter
        adapter = PushCubeAdapter()
        if fake:
            from tests.conftest import FakeEnvRunner
            return adapter, FakeEnvRunner()
        from babysteps.envs.pushcube_runner import PushCubeEnvRunner
        return adapter, PushCubeEnvRunner(orient_control=True)
    if task == "PokeCube-v1":
        # Second contact_region family (step 3). Sim-free fake reuses the
        # PushCube push physics; the real grasp+poke runner is GPU (build-order
        # step-3 GPU, behind the poke-feasibility kill-gate).
        from babysteps.envs.pokecube_adapter import PokeCubeAdapter
        adapter = PokeCubeAdapter()
        if fake:
            from tests.conftest import FakePokeEnvRunner
            return adapter, FakePokeEnvRunner()
        from babysteps.envs.pokecube_runner import PokeCubeEnvRunner
        return adapter, PokeCubeEnvRunner()
    if task == "StackCube-v1":
        from babysteps.envs.stackcube_adapter import StackCubeAdapter
        adapter = StackCubeAdapter()
        if fake:
            from tests.conftest import FakeStackCubeEnvRunner
            return adapter, FakeStackCubeEnvRunner()
        from babysteps.envs.stackcube_runner import StackCubeEnvRunner
        return adapter, StackCubeEnvRunner()
    raise ValueError(f"unsupported task {task!r}")


def run_eval(spec: TaskSpec, runner, vlm, seeds: list[int],
             conditions: list[str], shared_policy=None) -> dict:
    """Run the available conditions over all seeds; return the results dict."""
    rows_by_condition: dict[str, list[dict]] = {c: [] for c in conditions}
    per_seed: list[dict] = []
    for seed in seeds:
        ep = spec.source_episode(seed)
        seed_row = {"seed": seed, "initial_success": ep.initial_success,
                    "failure_predicate": ep.e_fail.predicate,
                    "initial_intent": ep.initial.to_dict(),
                    "conditions": {}}
        for cond in conditions:
            row = _run_condition(cond, ep, spec, runner, vlm, seed,
                                 shared_policy=shared_policy)
            rows_by_condition[cond].append(row)
            seed_row["conditions"][cond] = row
        per_seed.append(seed_row)
    # Registry-deferred conditions that were NOT explicitly enabled this run
    # (e.g. shared_revision_policy when no --scorer was supplied).
    deferred = [c for c in deferred_conditions() if c not in conditions]
    return {
        "summary": aggregate(rows_by_condition),
        "deferred_conditions": deferred,
        "per_seed": per_seed,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True,
                   choices=["PushCube-v1", "PokeCube-v1", "StackCube-v1"])
    p.add_argument("--eval-seeds", default="200-209",
                   help="Held-out exec seed range.")
    p.add_argument("--conditions", default=",".join(available_conditions()),
                   help="Comma list; default = all runnable conditions.")
    p.add_argument("--fake", action="store_true",
                   help="Sim-free FakeEnvRunner (login-node smoke).")
    p.add_argument("--mock", action="store_true",
                   help="MockVLMClient (no GPU / no transformers).")
    p.add_argument("--scorer", type=Path, default=None,
                   help="Frozen shared-scorer checkpoint. Supplying it ENABLES "
                        "the shared_revision_policy condition (registry default "
                        "is not-runnable = no committed checkpoint). The .pt is "
                        "gitignored, so sim-free tests pass a synthetic one.")
    p.add_argument("--max-episodes", type=int, default=None)
    # Real-run (step 5) inputs:
    p.add_argument("--pack-dir", type=Path, default=None)
    p.add_argument("--residual-head", type=Path, default=None)
    p.add_argument("--vision-features-dir", type=Path, default=None)
    p.add_argument("--vision-encoder", default="dinov2")
    p.add_argument("--feature-suffix", default="dinov2_fl")
    p.add_argument("--demo-seed-offset", type=int, default=500)
    p.add_argument("--mismatch", choices=["random", "always", "never"],
                   default="random")
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args(argv)

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    unknown = [c for c in conditions if c not in CONDITION_REGISTRY]
    if unknown:
        p.error(f"unknown condition(s): {unknown}")
    not_runnable = [c for c in conditions
                    if not CONDITION_REGISTRY[c].runnable]
    # shared_revision_policy is registry-deferred (no committed checkpoint) but
    # becomes runnable the moment a --scorer is supplied (the explicit opt-in).
    if "shared_revision_policy" in not_runnable and args.scorer is not None:
        not_runnable.remove("shared_revision_policy")
    if not_runnable:
        p.error(f"condition(s) not runnable yet (deferred): {not_runnable}. "
                f"shared_revision_policy needs --scorer <checkpoint>.")

    seeds = _parse_seed_range(args.eval_seeds)
    if args.max_episodes:
        seeds = seeds[: args.max_episodes]

    adapter, runner = _make_runner_adapter(args.task, args.fake)
    if args.task in ("PushCube-v1", "PokeCube-v1"):
        spec = _build_contact_region_spec(args.task, args, runner, adapter)
    else:
        spec = _build_stackcube_spec(args, runner, adapter)

    if args.mock:
        from babysteps.stage5.vlm_attribute import MockVLMClient
        vlm = MockVLMClient()
    else:
        from babysteps.stage5.vlm_attribute import InternVLClient
        vlm = InternVLClient()
        print("loading InternVL3.5-8B ...")
        vlm.load()

    shared_policy = None
    if "shared_revision_policy" in conditions:
        from babysteps.stage5.shared_revision_policy import SharedScorerPolicy
        shared_policy = SharedScorerPolicy.from_pack(args.scorer)
        print(f"loaded shared scorer: {args.scorer}")

    result = run_eval(spec, runner, vlm, seeds, conditions,
                      shared_policy=shared_policy)
    result["config"] = {
        "task": args.task, "eval_seeds": args.eval_seeds, "fake": args.fake,
        "mock": args.mock, "conditions": conditions}

    print(f"=== unified main table ({args.task}, fake={args.fake}, "
          f"mock={args.mock}, n={len(seeds)}) ===")
    for cond in conditions:
        s = result["summary"][cond]
        print(f"  {cond:26s} recovery_all={s['recovery_overall']:.3f}  "
              f"on_fail={_fmt(s['recovery_on_initial_fail'])}  "
              f"pres={_fmt(s['preservation_mean'])}  "
              f"edits={_fmt(s['edit_cardinality_mean'])}")
    print(f"  deferred: {result['deferred_conditions']}")

    out_dir = args.out_dir or Path(
        f"reports/stage5/unified_maintable/{args.task}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nwrote {out_dir}/results.json")

    if hasattr(adapter, "close"):
        adapter.close()
    return 0


def _fmt(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"


if __name__ == "__main__":
    sys.exit(main())
