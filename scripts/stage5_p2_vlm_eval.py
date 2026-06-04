"""Stage-5 P2 — VLM attribution + retry eval. Compares C1 (constrained
diagnosis + slot-local revision) against C2 (VLM free-form replan).

For each held-out failure episode (cached frame + failure_packet from
scripts/stage5_p2_render_failure_frames.py), runs both conditions through
the real env_runner retry mechanism and computes:

* attribution_accuracy (C1 only; C2 doesn't pick a factor)
* final_success_rate            (both)
* frozen_factor_preservation    (both; for C2, frozen = factors other than
                                  the oracle wrong factor)
* unnecessary_factor_change_rate (both)
* parse_failure_rate            (both; C1: factor name not in menu;
                                  C2: malformed JSON / invalid token)

Also computes the rule-table attribution accuracy on the same set for the
G_P2_acc gate (C1 acc >= rule-table acc).

Example::

    python scripts/stage5_p2_vlm_eval.py \\
        --task PushCube-v1 \\
        --episodes datasets/stage5/p2_vlm/PushCube-v1/episodes.jsonl \\
        --out-dir reports/stage5/p2_vlm_attribution/PushCube-v1/

Pass --mock for sim-free smoke (no GPU, no VLM call); --max-episodes N
to subset for debugging.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from babysteps.envs.task_registry import get_task_entry  # noqa: E402
from babysteps.failure import Attribution  # noqa: E402
from babysteps.schemas import INTENT_FIELDS, Intent  # noqa: E402
from babysteps.stage4.latent_policy import load_latent_pack  # noqa: E402
from babysteps.stage5.latent_intent import (  # noqa: E402
    build_latent_intent, latent_factor_names, latent_slot_edit,
)
from babysteps.stage5.selectivity import selectivity_metrics  # noqa: E402
from babysteps.stage5.vlm_attribute import (  # noqa: E402
    InternVLClient, MockVLMClient, get_factor_menu,
)


def _make_vision_provider(features_dir: Path):
    """Return ``provider(seed) -> Z`` loading cached DINOv2 features."""
    features_dir = Path(features_dir)

    def _provider(seed: int) -> np.ndarray:
        return np.load(
            features_dir / f"seed_{seed:04d}_dinov2.npy"
        ).astype(np.float32)

    return _provider


def _make_fake_adapter(task: str):
    """Stub adapter with a sim-free FakeEnvRunner (login-node smoke).

    Success bits from the fake runner are NOT meaningful — this exists to
    exercise the C1/C2 plumbing (including the latent path) without GPU.
    """
    from tests.conftest import (
        FakeEnvRunner, FakePickEnvRunner, FakeStackCubeEnvRunner,
    )
    fakes = {
        "PushCube-v1": FakeEnvRunner,
        "PickCube-v1": FakePickEnvRunner,
        "StackCube-v1": FakeStackCubeEnvRunner,
    }
    base_cls = get_task_entry(task).adapter_cls
    fake_runner = fakes[task]()

    class _StubAdapter(base_cls):
        def make_env_runner(self):
            return fake_runner

    return _StubAdapter()


def _base_intent_from_jsonl(path: Path) -> Intent:
    """Base intent for the constant (non-decoded) factors, from the TRAIN cut.

    The latent decode overwrites every factor that varies in the cut; only
    the trivially-constant factors of this base survive, and those are task
    constants sourced from SUPERVISION data — never from an eval episode.
    """
    for line in Path(path).read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            return Intent.from_dict(rec["execution"]["initial_intent"])
    raise ValueError(f"no records in {path}")


def _read_episodes(path: Path) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return [r for r in rows if r.get("is_failure", False)]


def _make_vlm_attribution(factor: str, factor_menu: tuple[str, ...]) -> Attribution:
    """Build an Attribution where the VLM's factor IS the wrong_factor.

    `factor_menu` is the per-task factor list (6 for the four 6-factor tasks,
    7 for CrossViewPush). Freeze = every other factor in the menu.
    """
    return Attribution(
        semantic_failure=True,
        wrong_factor=factor,
        freeze=tuple(f for f in factor_menu if f != factor),
        revise=(factor,),
    )


def _factors_changed(
    a: Intent, b: Intent, fields: tuple[str, ...] = INTENT_FIELDS,
) -> tuple[str, ...]:
    return tuple(f for f in fields if getattr(a, f) != getattr(b, f))


def _per_episode_c1(
    *, vlm_factor: Optional[str], oracle_factor: str,
    initial_intent: Intent, revised_intent: Optional[Intent],
    retry_success: Optional[bool], initial_success: bool,
    gt_intent: Intent,
    factor_menu: tuple[str, ...] = INTENT_FIELDS,
) -> dict:
    """Compute C1 metrics for one episode.

    Selectivity metrics (preservation / unnecessary / harmful) are measured
    against ``implicated_factor = oracle_factor`` for BOTH C1 and C2 so the two
    conditions are directly comparable (see babysteps.stage5.selectivity).
    """
    # Selectivity is measured against the ORACLE wrong factor for both
    # conditions (revised=None → preservation 0.0; see selectivity_metrics).
    sel = selectivity_metrics(
        initial=initial_intent, revised=revised_intent, gt=gt_intent,
        implicated_factor=oracle_factor, factor_menu=factor_menu,
    )
    if vlm_factor is None:
        return {
            "vlm_factor": None,
            "parse_failed": True,
            "attribution_correct": False,
            "factors_changed": [],
            "frozen_factor_preserved": None,
            "unnecessary_change": None,
            "final_success": bool(initial_success),
            "retry_success": None,
            **sel,
        }
    factors_changed = (_factors_changed(initial_intent, revised_intent, factor_menu)
                       if revised_intent is not None else ())
    # Frozen: no factor OTHER than the VLM-picked one changed.
    frozen_preserved = all(
        f == vlm_factor or f not in factors_changed for f in factor_menu
    )
    unnecessary = any(f != vlm_factor for f in factors_changed)
    final = (bool(retry_success) if retry_success is not None
             else bool(initial_success))
    return {
        "vlm_factor": vlm_factor,
        "parse_failed": False,
        "attribution_correct": vlm_factor == oracle_factor,
        "factors_changed": list(factors_changed),
        "frozen_factor_preserved": frozen_preserved,
        "unnecessary_change": unnecessary,
        "final_success": final,
        "retry_success": retry_success,
        **sel,
    }


def _per_episode_c2(
    *, revised_intent: Optional[Intent], oracle_factor: str,
    initial_intent: Intent, retry_success: Optional[bool],
    initial_success: bool, gt_intent: Intent,
    factor_menu: tuple[str, ...] = INTENT_FIELDS,
) -> dict:
    """Compute C2 metrics. For C2 there is no 'predicted factor' — instead
    we measure which factors changed vs the oracle-frozen set (all but the
    true wrong factor).

    Selectivity metrics use the SAME ``implicated_factor = oracle_factor`` as
    C1, so C1/C2 preservation / unnecessary / harmful are directly comparable.
    """
    sel = selectivity_metrics(
        initial=initial_intent, revised=revised_intent, gt=gt_intent,
        implicated_factor=oracle_factor, factor_menu=factor_menu,
    )
    if revised_intent is None:
        return {
            "parse_failed": True,
            "factors_changed": [],
            "frozen_factor_preserved": None,
            "unnecessary_change": None,
            "fixed_oracle_factor": None,
            "final_success": bool(initial_success),
            "retry_success": None,
            **sel,
        }
    factors_changed = _factors_changed(initial_intent, revised_intent, factor_menu)
    # Frozen-preserved (C2 sense): no factor OTHER than oracle_factor changed.
    frozen_preserved = all(
        f == oracle_factor or f not in factors_changed for f in factor_menu
    )
    # Unnecessary: any factor change OTHER than the oracle's wrong factor.
    unnecessary = any(f != oracle_factor for f in factors_changed)
    fixed_oracle = oracle_factor in factors_changed
    final = (bool(retry_success) if retry_success is not None
             else bool(initial_success))
    return {
        "parse_failed": False,
        "factors_changed": list(factors_changed),
        "frozen_factor_preserved": frozen_preserved,
        "unnecessary_change": unnecessary,
        "fixed_oracle_factor": fixed_oracle,
        "final_success": final,
        "retry_success": retry_success,
        **sel,
    }


# Numeric selectivity keys present on every row (C1 + C2). These are MEANS,
# not boolean rates — aggregated over ALL rows (parse-fail rows carry the
# revised=None values: preservation 0.0, unnecessary/harmful 0). They are
# directly comparable across C1 and C2 (same implicated_factor = oracle).
_SELECTIVITY_MEAN_KEYS: tuple[str, ...] = (
    "preservation",
    "unnecessary_changes_count", "unnecessary_changes_rate",
    "harmful_changes_count", "harmful_changes_rate",
)


def _aggregate(rows: list[dict], keys: list[str]) -> dict:
    """Rate of each boolean key, ignoring None entries; plus the mean of each
    numeric selectivity key (over ALL rows — None-free by construction)."""
    out: dict = {}
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None]
        out[k + "_rate"] = (sum(bool(v) for v in vals) / len(vals)
                            if vals else None)
        out["n_" + k] = len(vals)
    for k in _SELECTIVITY_MEAN_KEYS:
        vals = [r[k] for r in rows if r.get(k) is not None]
        out[k + "_mean"] = (sum(vals) / len(vals)) if vals else None
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True,
                   choices=["PushCube-v1", "PickCube-v1", "StackCube-v1",
                            "TurnFaucet-v1", "CrossViewPush-v1"])
    p.add_argument("--episodes", type=Path, required=True,
                   help="episodes.jsonl from stage5_p2_render_failure_frames.py")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--mock", action="store_true",
                   help="Use MockVLMClient (no GPU, no transformers).")
    p.add_argument("--max-episodes", type=int, default=None,
                   help="Subset for debugging.")
    p.add_argument("--conditions", default="c1,c2",
                   help="Comma list: c1,c2 or just one.")
    p.add_argument("--no-wrist", action="store_true",
                   help="Ignore wrist_frame_path even when present (single "
                        "third-person image, the original P2 setup). Use to "
                        "A/B single- vs multi-image on identical frames.")
    p.add_argument("--latent", action="store_true",
                   help="Latent-input mode: derive the initial intent from "
                        "vision (DINOv2->IntentHead->nearest-centroid) and "
                        "repair C1 via the learned slot-local ReviseHead, "
                        "instead of the JSON intent + discrete operator. The "
                        "JSON factors are then used only for supervision + "
                        "eval. Requires --pack-dir, --features-dir, "
                        "--train-jsonl.")
    p.add_argument("--pack-dir", type=Path, default=None,
                   help="LatentPack dir (required with --latent).")
    p.add_argument("--features-dir", type=Path, default=None,
                   help="Cached DINOv2 features dir (required with --latent).")
    p.add_argument("--train-jsonl", type=Path, default=None,
                   help="Training samples.jsonl; sources the constant (non-"
                        "decoded) factors of the base intent (required with "
                        "--latent).")
    p.add_argument("--fake", action="store_true",
                   help="Use a sim-free FakeEnvRunner (login-node smoke; "
                        "success bits are not meaningful).")
    args = p.parse_args(argv)

    episodes = _read_episodes(args.episodes)
    if args.max_episodes:
        episodes = episodes[: args.max_episodes]
    print(f"loaded {len(episodes)} failure episodes for {args.task}")

    entry = get_task_entry(args.task)
    adapter = _make_fake_adapter(args.task) if args.fake else entry.adapter_cls()
    factor_menu = get_factor_menu(args.task)

    # ---- Latent-input mode setup (Sever A + Sever B) ---- #
    pack = None
    vision_provider = None
    base_intent = None
    if args.latent:
        if not (args.pack_dir and args.features_dir and args.train_jsonl):
            p.error("--latent requires --pack-dir, --features-dir, --train-jsonl")
        pack = load_latent_pack(args.pack_dir)
        vision_provider = _make_vision_provider(args.features_dir)
        base_intent = _base_intent_from_jsonl(args.train_jsonl)
        print(f"LATENT mode: pack decodes {latent_factor_names(pack)} from "
              f"vision; constant factors from {args.train_jsonl}")

    vlm: MockVLMClient | InternVLClient
    if args.mock:
        vlm = MockVLMClient()
    else:
        vlm = InternVLClient()
        print("loading InternVL3.5-8B ...")
        vlm.load()
        print("loaded.")

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    c1_rows: list[dict] = []
    c2_rows: list[dict] = []
    rule_correct, rule_total = 0, 0

    for ep in episodes:
        seed = ep["seed"]
        # Sever A — the method input. In latent mode the initial intent is
        # decoded from vision (DINOv2->IntentHead->nearest-centroid); the
        # stored JSON is used only to (a) source constant factors via the
        # train cut and (b) audit faithfulness. Default mode reads the JSON.
        if args.latent:
            z = vision_provider(seed)
            initial = build_latent_intent(pack, z, base_intent)
            latent_matches_stored = (initial.to_dict() == ep["initial_intent"])
        else:
            z = None
            initial = Intent.from_dict(ep["initial_intent"])
            latent_matches_stored = None
        oracle_factor = ep["oracle_wrong_factor"]
        rule_factor = ep["rule_table_wrong_factor"]
        if rule_factor is not None:
            rule_total += 1
            if rule_factor == oracle_factor:
                rule_correct += 1

        # Rebuild executor scene for the retry rollout (deterministic seed).
        env_runner = adapter.env_runner()
        scene_initial = env_runner.reset(seed)
        scene_executor = replace(
            scene_initial,
            blocked_sides=adapter.default_blocked_factory(initial),
        )
        # Ground-truth correct intent for this scene — the selectivity
        # reference for BOTH C1 and C2 (harmful_changes is measured against it).
        gt_intent = adapter.oracle_correct_intent(scene_executor)
        # In latent mode the initial intent is vision-derived, so recompute
        # the oracle wrong factor from THAT intent for self-consistency
        # (PushCube: still "approach_direction" whenever the demonstrated
        # approach is blocked — matches the stored label for 49/50 seeds).
        if args.latent:
            oracle_factor = adapter.oracle_wrong_factor(initial, scene_executor)

        # First-person wrist frame (PushCube only; None elsewhere or when
        # --no-wrist). When present, both conditions diagnose from the
        # third-person + wrist pair via the multi-image VLM path.
        wrist_path = None if args.no_wrist else ep.get("wrist_frame_path")

        # ---------- C1: VLM constrained → discrete revision ---------- #
        if "c1" in conditions:
            vlm_factor = vlm.diagnose_constrained(
                task=args.task,
                image_path=ep["frame_path"],
                initial_intent=initial,
                failure_predicate=ep["failure_predicate"],
                wrist_image_path=wrist_path,
            )
            revised: Optional[Intent] = None
            retry_success: Optional[bool] = None
            if vlm_factor is not None:
                try:
                    # Sever B — the repair. Latent mode edits the implicated
                    # slot via the learned ReviseHead (decode back to a
                    # token); default mode applies the discrete operator.
                    if args.latent:
                        revised = latent_slot_edit(
                            pack, z, initial, vlm_factor,
                            ep["failure_predicate"],
                        )
                    else:
                        attribution = _make_vlm_attribution(vlm_factor, factor_menu)
                        revised, _rev = adapter.revise_intent(
                            initial, attribution, scene_executor,
                        )
                    env_runner.reset(seed)
                    attempt = env_runner.run(revised, scene_executor)
                    retry_success = bool(attempt.success)
                except Exception as exc:
                    print(f"WARN: C1 retry exception seed {seed}: {exc}",
                          file=sys.stderr)
                    revised, retry_success = None, None
            row = _per_episode_c1(
                vlm_factor=vlm_factor, oracle_factor=oracle_factor,
                initial_intent=initial, revised_intent=revised,
                retry_success=retry_success,
                initial_success=ep["initial_success"],
                gt_intent=gt_intent,
                factor_menu=factor_menu,
            )
            row.update({
                "seed": seed, "oracle_wrong_factor": oracle_factor,
                "initial_intent": initial.to_dict(),
                "revised_intent": revised.to_dict() if revised is not None else None,
                "gt_intent": gt_intent.to_dict(),
                "latent_matches_stored": latent_matches_stored,
            })
            c1_rows.append(row)
            print(f"  C1 seed={seed} vlm={vlm_factor!r:>22} "
                  f"oracle={oracle_factor!r:>22} "
                  f"retry_success={retry_success}")

        # ---------- C2: VLM free-form → verbatim retry ---------- #
        if "c2" in conditions:
            # diagnose_free_form_verbose returns (intent_or_None, raw_vlm_text)
            # and runs the ONE format-repair retry internally; we persist the
            # raw text so future parse debugging needs no re-run.
            revised2, raw_vlm_text = vlm.diagnose_free_form_verbose(
                task=args.task,
                image_path=ep["frame_path"],
                initial_intent=initial,
                failure_predicate=ep["failure_predicate"],
                wrist_image_path=wrist_path,
            )
            retry_success2: Optional[bool] = None
            if revised2 is not None:
                try:
                    env_runner.reset(seed)
                    attempt2 = env_runner.run(revised2, scene_executor)
                    retry_success2 = bool(attempt2.success)
                except Exception as exc:
                    print(f"WARN: C2 retry exception seed {seed}: {exc}",
                          file=sys.stderr)
                    retry_success2 = None
            row2 = _per_episode_c2(
                revised_intent=revised2, oracle_factor=oracle_factor,
                initial_intent=initial, retry_success=retry_success2,
                initial_success=ep["initial_success"],
                gt_intent=gt_intent,
                factor_menu=factor_menu,
            )
            row2.update({
                "seed": seed, "oracle_wrong_factor": oracle_factor,
                "raw_vlm_text": raw_vlm_text,
                "initial_intent": initial.to_dict(),
                "revised_intent": revised2.to_dict() if revised2 is not None else None,
                "gt_intent": gt_intent.to_dict(),
                "latent_matches_stored": latent_matches_stored,
            })
            c2_rows.append(row2)
            print(f"  C2 seed={seed} revised={revised2 is not None} "
                  f"retry_success={retry_success2}")

    adapter.close()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rule_acc = rule_correct / rule_total if rule_total else None

    if "c1" in conditions:
        c1_summary = _aggregate(c1_rows, [
            "attribution_correct", "frozen_factor_preserved",
            "unnecessary_change", "final_success", "parse_failed",
        ])
        (args.out_dir / "c1_results.json").write_text(json.dumps({
            "task": args.task,
            "latent_mode": bool(args.latent),
            "n_latent_mismatch": sum(
                1 for r in c1_rows if r.get("latent_matches_stored") is False),
            "rule_table_accuracy": rule_acc,
            "n_episodes": len(c1_rows),
            "summary": c1_summary,
            "per_episode": c1_rows,
        }, indent=2, sort_keys=True) + "\n")
        print(f"\nC1 summary: {c1_summary}")

    if "c2" in conditions:
        c2_summary = _aggregate(c2_rows, [
            "frozen_factor_preserved", "unnecessary_change",
            "fixed_oracle_factor", "final_success", "parse_failed",
        ])
        (args.out_dir / "c2_results.json").write_text(json.dumps({
            "task": args.task,
            "latent_mode": bool(args.latent),
            "n_latent_mismatch": sum(
                1 for r in c2_rows if r.get("latent_matches_stored") is False),
            "n_episodes": len(c2_rows),
            "summary": c2_summary,
            "per_episode": c2_rows,
        }, indent=2, sort_keys=True) + "\n")
        print(f"C2 summary: {c2_summary}")

    # If we only ran ONE condition this invocation, load the other's
    # results from disk (from a previous full run) so the merged report
    # carries both columns + the gates. This is the C2-only re-run path.
    report_c1 = c1_rows if "c1" in conditions else None
    report_c2 = c2_rows if "c2" in conditions else None
    report_rule_acc = rule_acc
    if report_c1 is None and (args.out_dir / "c1_results.json").exists():
        prev = json.loads((args.out_dir / "c1_results.json").read_text())
        report_c1 = prev["per_episode"]
        # Prefer the saved rule_table_accuracy from the original run (it
        # was computed across the full episode set, identical to this run).
        if report_rule_acc is None:
            report_rule_acc = prev.get("rule_table_accuracy")
    if report_c2 is None and (args.out_dir / "c2_results.json").exists():
        prev = json.loads((args.out_dir / "c2_results.json").read_text())
        report_c2 = prev["per_episode"]
    _write_report_md(
        args.out_dir / "report.md", args.task, report_rule_acc,
        report_c1, report_c2,
    )
    print(f"\nwrote {args.out_dir}/")
    return 0


def _write_report_md(
    out_path: Path, task: str, rule_acc: Optional[float],
    c1_rows: Optional[list[dict]], c2_rows: Optional[list[dict]],
) -> None:
    def rate(rows, key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return (sum(bool(v) for v in vals) / len(vals)
                if vals else float("nan"))

    def mean(rows, key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return (sum(vals) / len(vals)) if vals else float("nan")

    lines = [f"# Stage-5 P2 VLM attribution — {task}", ""]
    if rule_acc is not None:
        lines.append(
            f"- Rule-table attribution accuracy (baseline): **{rule_acc:.3f}**"
        )
        lines.append("")
    if c1_rows is not None:
        lines.extend([
            "## C1 — VLM-constrained diagnosis + slot-local revision (ours)",
            "",
            f"- n_episodes: {len(c1_rows)}",
            f"- attribution_accuracy: **{rate(c1_rows, 'attribution_correct'):.3f}**",
            f"- frozen_factor_preservation: **{rate(c1_rows, 'frozen_factor_preserved'):.3f}**",
            f"- unnecessary_factor_change: **{rate(c1_rows, 'unnecessary_change'):.3f}**",
            f"- final_success_rate: **{rate(c1_rows, 'final_success'):.3f}**",
            f"- parse_failure_rate: {rate(c1_rows, 'parse_failed'):.3f}",
            f"- preservation (mean over non-implicated factors): "
            f"**{mean(c1_rows, 'preservation'):.3f}**",
            f"- unnecessary_changes_rate (mean): "
            f"{mean(c1_rows, 'unnecessary_changes_rate'):.3f}",
            f"- harmful_changes_rate (mean): "
            f"{mean(c1_rows, 'harmful_changes_rate'):.3f}",
            "",
        ])
    if c2_rows is not None:
        lines.extend([
            "## C2 — VLM free-form replan (baseline)",
            "",
            f"- n_episodes: {len(c2_rows)}",
            f"- frozen_factor_preservation: **{rate(c2_rows, 'frozen_factor_preserved'):.3f}**",
            f"- unnecessary_factor_change: **{rate(c2_rows, 'unnecessary_change'):.3f}**",
            f"- fixed_oracle_factor_rate: {rate(c2_rows, 'fixed_oracle_factor'):.3f}",
            f"- final_success_rate: **{rate(c2_rows, 'final_success'):.3f}**",
            f"- parse_failure_rate: {rate(c2_rows, 'parse_failed'):.3f}",
            f"- preservation (mean over non-implicated factors): "
            f"**{mean(c2_rows, 'preservation'):.3f}**",
            f"- unnecessary_changes_rate (mean): "
            f"{mean(c2_rows, 'unnecessary_changes_rate'):.3f}",
            f"- harmful_changes_rate (mean): "
            f"{mean(c2_rows, 'harmful_changes_rate'):.3f}",
            "",
        ])
    if c1_rows is not None and c2_rows is not None:
        d_pres = (rate(c1_rows, "frozen_factor_preserved")
                   - rate(c2_rows, "frozen_factor_preserved")) * 100
        d_succ = (rate(c1_rows, "final_success")
                   - rate(c2_rows, "final_success")) * 100
        lines.extend([
            "## Gates",
            "",
            f"- **C1 attribution ≥ rule-table**: "
            f"{rate(c1_rows, 'attribution_correct'):.3f} vs "
            f"{rule_acc if rule_acc is not None else float('nan'):.3f} → "
            f"{'PASS' if (rule_acc is not None and rate(c1_rows, 'attribution_correct') >= rule_acc) else 'FAIL'}",
            f"- **C1 preservation ≥ C2 preservation** "
            f"(Δ = {d_pres:+.1f}pp; PASS if Δ ≥ 0)",
            f"- **C1 success ≥ C2 success within 5pp** "
            f"(Δ = {d_succ:+.1f}pp; PASS if Δ ≥ -5)",
            f"- **C1 selectivity-preservation ≥ C2** "
            f"(mean preservation Δ = "
            f"{(mean(c1_rows, 'preservation') - mean(c2_rows, 'preservation')) * 100:+.1f}pp; "
            f"PASS if Δ ≥ 0)",
            f"- **C1 harmful-changes ≤ C2** "
            f"(mean harmful_changes_rate Δ = "
            f"{(mean(c1_rows, 'harmful_changes_rate') - mean(c2_rows, 'harmful_changes_rate')) * 100:+.1f}pp; "
            f"PASS if Δ ≤ 0)",
            "",
        ])
    out_path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
