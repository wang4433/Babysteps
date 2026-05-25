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

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from babysteps.envs.task_registry import get_task_entry  # noqa: E402
from babysteps.failure import Attribution  # noqa: E402
from babysteps.schemas import INTENT_FIELDS, Intent  # noqa: E402
from babysteps.stage5.vlm_attribute import (  # noqa: E402
    InternVLClient, MockVLMClient,
)


def _read_episodes(path: Path) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return [r for r in rows if r.get("is_failure", False)]


def _make_vlm_attribution(factor: str) -> Attribution:
    """Build an Attribution where the VLM's factor IS the wrong_factor."""
    return Attribution(
        semantic_failure=True,
        wrong_factor=factor,
        freeze=tuple(f for f in INTENT_FIELDS if f != factor),
        revise=(factor,),
    )


def _factors_changed(a: Intent, b: Intent) -> tuple[str, ...]:
    return tuple(f for f in INTENT_FIELDS if getattr(a, f) != getattr(b, f))


def _per_episode_c1(
    *, vlm_factor: Optional[str], oracle_factor: str,
    initial_intent: Intent, revised_intent: Optional[Intent],
    retry_success: Optional[bool], initial_success: bool,
) -> dict:
    """Compute C1 metrics for one episode."""
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
        }
    factors_changed = (_factors_changed(initial_intent, revised_intent)
                       if revised_intent is not None else ())
    # Frozen: no factor OTHER than the VLM-picked one changed.
    frozen_preserved = all(
        f == vlm_factor or f not in factors_changed for f in INTENT_FIELDS
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
    }


def _per_episode_c2(
    *, revised_intent: Optional[Intent], oracle_factor: str,
    initial_intent: Intent, retry_success: Optional[bool],
    initial_success: bool,
) -> dict:
    """Compute C2 metrics. For C2 there is no 'predicted factor' — instead
    we measure which factors changed vs the oracle-frozen set (all but the
    true wrong factor)."""
    if revised_intent is None:
        return {
            "parse_failed": True,
            "factors_changed": [],
            "frozen_factor_preserved": None,
            "unnecessary_change": None,
            "fixed_oracle_factor": None,
            "final_success": bool(initial_success),
            "retry_success": None,
        }
    factors_changed = _factors_changed(initial_intent, revised_intent)
    # Frozen-preserved (C2 sense): no factor OTHER than oracle_factor changed.
    frozen_preserved = all(
        f == oracle_factor or f not in factors_changed for f in INTENT_FIELDS
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
    }


def _aggregate(rows: list[dict], keys: list[str]) -> dict:
    """Rate of each key, ignoring None entries."""
    out: dict = {}
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None]
        out[k + "_rate"] = (sum(bool(v) for v in vals) / len(vals)
                            if vals else None)
        out["n_" + k] = len(vals)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True,
                   choices=["PushCube-v1", "PickCube-v1", "StackCube-v1"])
    p.add_argument("--episodes", type=Path, required=True,
                   help="episodes.jsonl from stage5_p2_render_failure_frames.py")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--mock", action="store_true",
                   help="Use MockVLMClient (no GPU, no transformers).")
    p.add_argument("--max-episodes", type=int, default=None,
                   help="Subset for debugging.")
    p.add_argument("--conditions", default="c1,c2",
                   help="Comma list: c1,c2 or just one.")
    args = p.parse_args(argv)

    episodes = _read_episodes(args.episodes)
    if args.max_episodes:
        episodes = episodes[: args.max_episodes]
    print(f"loaded {len(episodes)} failure episodes for {args.task}")

    entry = get_task_entry(args.task)
    adapter = entry.adapter_cls()

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
        initial = Intent.from_dict(ep["initial_intent"])
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

        # ---------- C1: VLM constrained → discrete revision ---------- #
        if "c1" in conditions:
            vlm_factor = vlm.diagnose_constrained(
                image_path=ep["frame_path"],
                initial_intent=initial,
                failure_predicate=ep["failure_predicate"],
            )
            revised: Optional[Intent] = None
            retry_success: Optional[bool] = None
            if vlm_factor is not None:
                try:
                    attribution = _make_vlm_attribution(vlm_factor)
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
            )
            row.update({"seed": seed, "oracle_wrong_factor": oracle_factor})
            c1_rows.append(row)
            print(f"  C1 seed={seed} vlm={vlm_factor!r:>22} "
                  f"oracle={oracle_factor!r:>22} "
                  f"retry_success={retry_success}")

        # ---------- C2: VLM free-form → verbatim retry ---------- #
        if "c2" in conditions:
            revised2 = vlm.diagnose_free_form(
                image_path=ep["frame_path"],
                initial_intent=initial,
                failure_predicate=ep["failure_predicate"],
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
            )
            row2.update({"seed": seed, "oracle_wrong_factor": oracle_factor})
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
            f"- **C1 preservation ≫ C2 preservation** "
            f"(Δ = {d_pres:+.1f}pp; PASS if Δ > 0)",
            f"- **C1 success ≥ C2 success within 5pp** "
            f"(Δ = {d_succ:+.1f}pp; PASS if Δ ≥ -5)",
            "",
        ])
    out_path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
