"""Stage-5 — clustered-CI + selectivity report for the PokeCube LOTO eval.

Turns the ALREADY-COMMITTED LOTO results JSON into a defensible, reviewer-facing
summary WITHOUT any GPU:

* Table 5 (statistics): 95% clustered-bootstrap CIs for every condition's
  recovery + the shared-scorer face-pick accuracy, resampling by SCENE SEED
  (the ~20 independent clusters), plus PAIRED diff CIs (shared_scorer − oracle,
  − random, − open_loop). The shared_scorer−oracle diff CI is the honest
  "== oracle ceiling" statement.
* Failure attribution: which seed-clusters carry the failures and whether the
  privileged oracle ALSO fails there (→ oracle-coincident geometry, not a policy
  error).
* Selectivity disclosure (honest): the within-direction direction→face decision
  is a DETERMINISTIC 1:1 residual-sign→face map; wrong_face does not affect the
  scorer's choice. Reported plainly so the result is not over-claimed.

Run on the login node::

    python scripts/stage5_loto_ci_report.py \\
        --results reports/stage5/pokecube_loto/results_pooled_gi_none.json \\
                  reports/stage5/pokecube_loto/results_pushonly_gi_none.json \\
        --out reports/stage5/pokecube_loto/ci_report
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from babysteps.stage5.cluster_bootstrap import (  # noqa: E402
    clustered_bootstrap_ci, failing_clusters, paired_clustered_bootstrap_diff,
)

# row column -> human label for the recovery table
_RECOVERY_COLS = {
    "open_loop_success": "same_intent / open_loop",
    "random_success": "random_face",
    "scorer_success": "shared_scorer (frozen)",
    "oracle_success": "oracle (ceiling)",
}


def _ci_str(ci: dict) -> str:
    return f"{ci['mean']:.3f}  [{ci['lo']:.3f}, {ci['hi']:.3f}]"


def analyze(results: dict, *, n_boot: int, seed: int) -> dict:
    rows = results["rows"]
    out: dict = {
        "scorer": results.get("scorer"),
        "n_episodes": len(rows),
        "candidates": results.get("candidates"),
        "recovery_ci": {}, "paired_diffs": {},
    }
    for col, label in _RECOVERY_COLS.items():
        out["recovery_ci"][label] = clustered_bootstrap_ci(
            rows, col, n_boot=n_boot, seed=seed)
    out["face_acc_ci"] = clustered_bootstrap_ci(
        rows, "scorer_face_correct", n_boot=n_boot, seed=seed)
    out["paired_diffs"]["scorer_minus_oracle"] = paired_clustered_bootstrap_diff(
        rows, "scorer_success", "oracle_success", n_boot=n_boot, seed=seed)
    out["paired_diffs"]["scorer_minus_random"] = paired_clustered_bootstrap_diff(
        rows, "scorer_success", "random_success", n_boot=n_boot, seed=seed)
    out["paired_diffs"]["scorer_minus_open_loop"] = paired_clustered_bootstrap_diff(
        rows, "scorer_success", "open_loop_success", n_boot=n_boot, seed=seed)

    # Failure attribution (oracle-coincident?).
    out["failure_attribution"] = failing_clusters(rows, "scorer_success")

    out["selectivity"] = selectivity_disclosure(rows)
    return out


def selectivity_disclosure(rows) -> dict:
    """Honest "degenerate 1:1 rule" disclosure (pure, testable):

    * ``direction_to_face_is_deterministic`` — for each goal direction, does the
      scorer always pick the SAME face (regardless of wrong_face)?
    * ``wrong_face_changes_choice`` — for any FIXED direction, do different
      ``wrong_face`` values produce different scorer faces? (i.e. does the
      current value, not just the residual, drive the choice?)

    The second metric must compare ACROSS wrong_face values within a direction —
    not look for stochasticity within a single (direction, wrong_face) pair,
    which is a different question (review fix)."""
    by_dir: dict[str, set] = defaultdict(set)
    by_dir_wf: dict[tuple, set] = defaultdict(set)
    for r in rows:
        by_dir[r["direction"]].add(r["scorer_face"])
        by_dir_wf[(r["direction"], r["wrong_face"])].add(r["scorer_face"])

    # wrong_face matters iff, for some direction, two wrong_face values map to
    # different face-sets.
    directions = {d for (d, _wf) in by_dir_wf}
    wrong_face_changes_choice = any(
        len({frozenset(faces) for (d2, _wf), faces in by_dir_wf.items()
             if d2 == d}) > 1
        for d in directions)

    return {
        "direction_to_face": {d: sorted(v) for d, v in by_dir.items()},
        "direction_to_face_is_deterministic": all(
            len(v) == 1 for v in by_dir.values()),
        "wrong_face_changes_choice": wrong_face_changes_choice,
    }


def _markdown(name: str, a: dict) -> str:
    L = [f"## {name}", "",
         f"- scorer: `{a['scorer']}`",
         f"- episodes: {a['n_episodes']}  | clusters (seeds): "
         f"{a['face_acc_ci']['n_clusters']}  | candidates: {a['candidates']}",
         "",
         "### Table 5 — recovery, 95% clustered-bootstrap CI (resampled by scene seed)",
         "", "| condition | recovery  [95% CI] |", "|---|---|"]
    for label, ci in a["recovery_ci"].items():
        L.append(f"| {label} | {_ci_str(ci)} |")
    L += ["", f"shared-scorer face-pick accuracy: **{_ci_str(a['face_acc_ci'])}**",
          "", "### Paired difference CIs (same cluster resample → paired)", ""]
    for k, d in a["paired_diffs"].items():
        sig = "excludes 0" if (d["lo"] > 0 or d["hi"] < 0) else "INCLUDES 0"
        L.append(f"- {k}: **{d['diff']:+.3f}**  [{d['lo']:+.3f}, {d['hi']:+.3f}]  ({sig})")
    fa = a["failure_attribution"]
    L += ["", "### Failure attribution", ""]
    if not fa:
        L.append("- no shared-scorer failures.")
    for cl, info in fa.items():
        L.append(f"- seed {cl}: {info['n_fail']} fail, oracle-coincident="
                 f"{info['all_oracle_coincident']} "
                 f"({info['n_fail_oracle_also_fails']}/{info['n_fail']} also fail oracle)")
    sel = a["selectivity"]
    L += ["", "### Selectivity disclosure (honest)", "",
          f"- direction→face deterministic: **{sel['direction_to_face_is_deterministic']}** "
          f"(degenerate 1:1 residual-sign→face rule)",
          f"- wrong_face changes the choice: **{sel['wrong_face_changes_choice']}**",
          f"- map: {sel['direction_to_face']}", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", nargs="+", type=Path, required=True)
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path,
                   default=Path("reports/stage5/pokecube_loto/ci_report"))
    args = p.parse_args(argv)

    report: dict = {"analyses": {}}
    md_parts = ["# PokeCube LOTO — clustered-CI + selectivity report", ""]
    for path in args.results:
        results = json.loads(Path(path).read_text())
        a = analyze(results, n_boot=args.n_boot, seed=args.seed)
        report["analyses"][path.name] = a
        md_parts.append(_markdown(path.name, a))
        md_parts.append("")

    md = "\n".join(md_parts)
    print(md)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    (args.out.with_suffix(".json")).write_text(json.dumps(report, indent=2) + "\n")
    (args.out.with_suffix(".md")).write_text(md + "\n")
    print(f"\nwrote {args.out}.json and {args.out}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
