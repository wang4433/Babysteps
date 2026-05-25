#!/usr/bin/env python
"""Stage-5 P1 PushCube held-out eval — visualization report generator.

Consumes the existing eval output at
`reports/stage5/p1_vision_g4_g5/PushCube-v1/p1_results.json` (produced by
`scripts/stage5_p1_run_eval.py`, commit 7a8c2d7) and emits:

- `per_policy_summary.csv` — one row per policy (success/n, rate, n_failures)
- `per_seed_long.csv` — one row per (seed, policy) with success flags
- `summary.json` — machine-readable digest of the run (config + gates + per_policy)
- `bar_chart.{png,pdf}` — matplotlib bar chart with counts/rates annotated
- `gallery.html` — seed × policy table + "failures only" section, with
  per-cell pass/fail badges and links to MP4 renders when found

The HTML gallery discovers per-seed MP4s under `--renders-dir` and links them
when present. The Stage-5 eval rollout only emits success metrics — no MP4s —
so for seeds 100-149 the gallery will mostly show "(no video)". The HTML
output documents the exact render command needed to populate the missing
videos; it does NOT attempt to launch ManiSkill itself.

Sim-free: no GPU / Vulkan / ManiSkill imports. Reads JSON, writes CSV/JSON/PNG/PDF/HTML.

Example::

    python scripts/report_stage5_pushcube_heldout.py
    # Defaults expect the existing eval output and write under
    #   reports/stage5/p1_vision_g4_g5/PushCube-v1/report_gallery/

    python scripts/report_stage5_pushcube_heldout.py \\
        --results reports/stage5/p1_vision_g4_g5/PushCube-v1/p1_results.json \\
        --renders-dir renders/pushcube/videos_maniskill \\
        --out-dir reports/stage5/p1_vision_g4_g5/PushCube-v1/report_gallery
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from textwrap import dedent
from typing import Optional

# Policy order matches the user's requested table layout:
# latent (vision-grounded) | oracle | babysteps_selective | same_intent_retry.
POLICIES: tuple[str, ...] = (
    "latent",
    "oracle_factor_revision",
    "babysteps_selective",
    "same_intent_retry",
)
POLICY_DISPLAY: dict[str, str] = {
    "latent": "latent (vision-grounded)",
    "oracle_factor_revision": "oracle",
    "babysteps_selective": "babysteps_selective",
    "same_intent_retry": "same_intent_retry",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def build_per_seed_long(results: dict) -> list[dict]:
    """Long-form table: one row per (policy, seed) with success flags."""
    rows: list[dict] = []
    for policy in POLICIES:
        for r in results["per_policy"][policy]["rows"]:
            rows.append({
                "seed": r["seed"],
                "policy": policy,
                "initial_success": r["initial_success"],
                "retry_success": r["retry_success"],
                "final_success": r["final_success"],
            })
    return rows


def index_per_seed_policy(results: dict) -> dict[tuple[int, str], dict]:
    """(seed, policy) -> the row dict from per_policy."""
    idx: dict[tuple[int, str], dict] = {}
    for policy in POLICIES:
        for r in results["per_policy"][policy]["rows"]:
            idx[(r["seed"], policy)] = r
    return idx


# ---------------------------------------------------------------------------
# MP4 discovery
# ---------------------------------------------------------------------------

def find_retry_mp4(renders_dir: Path, seed: int, policy: str) -> Optional[Path]:
    """Try a few plausible patterns; return the first existing MP4 for (seed, policy).

    The current repo has render_stage0_maniskill.py which only emits the
    standard babysteps_selective 3-phase render (no --policy flag), so for
    most (seed, policy) pairs nothing is found.
    """
    candidates = [
        # Per-policy hypothetical naming (would require a render-pipeline extension).
        f"pushcube_{policy}_seed_{seed:04d}__3_retry.mp4",
        f"pushcube_seed_{seed:04d}_{policy}__3_retry.mp4",
        # Standard babysteps render (selective by default — link from selective column).
        f"pushcube_blocked_approach_seed_{seed:04d}__3_retry.mp4"
        if policy == "babysteps_selective" else None,
    ]
    for pat in candidates:
        if pat is None:
            continue
        p = renders_dir / pat
        if p.exists():
            return p
    return None


def find_demo_mp4(renders_dir: Path, seed: int) -> Optional[Path]:
    """The 1_demo phase is policy-independent; one MP4 per seed."""
    candidates = [
        f"pushcube_blocked_approach_seed_{seed:04d}__1_demo.mp4",
        f"pushcube_seed_{seed:04d}__1_demo.mp4",
    ]
    for pat in candidates:
        p = renders_dir / pat
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# CSV / JSON writers
# ---------------------------------------------------------------------------

def write_per_policy_csv(results: dict, out_path: Path) -> None:
    rows = []
    for policy in POLICIES:
        pp = results["per_policy"][policy]
        n = pp["n"]
        n_success = int(round(pp["final_success_rate"] * n))
        rows.append({
            "policy": policy,
            "display_name": POLICY_DISPLAY[policy],
            "n": n,
            "final_success_count": n_success,
            "final_success_rate": round(pp["final_success_rate"], 6),
            "initial_success_rate": round(pp["initial_success_rate"], 6),
            "n_failures": n - n_success,
        })
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def write_per_seed_csv(rows: list[dict], out_path: Path) -> None:
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def write_summary_json(results: dict, out_path: Path) -> None:
    seeds = results["config"]["eval_seeds"]
    per_policy_agg = {}
    for policy in POLICIES:
        pp = results["per_policy"][policy]
        n = pp["n"]
        n_success = int(round(pp["final_success_rate"] * n))
        failure_seeds = [
            r["seed"] for r in pp["rows"] if not r["final_success"]
        ]
        per_policy_agg[policy] = {
            "display_name": POLICY_DISPLAY[policy],
            "n": n,
            "final_success_count": n_success,
            "final_success_rate": pp["final_success_rate"],
            "n_failures": n - n_success,
            "failure_seeds": failure_seeds,
        }
    out = {
        "task": results["config"]["task"],
        "real_env": not results["config"]["fake"],
        "pack_dir": results["config"]["pack_dir"],
        "features_dir": results["config"]["features_dir"],
        "n_seeds": len(seeds),
        "seed_range": [min(seeds), max(seeds)],
        "per_policy": per_policy_agg,
        "gates": {
            "g4_delta_pp_vs_same_intent_retry": results["summary"]["delta_pp_vs_same_intent_retry"],
            "g4_pass": results["summary"]["delta_pp_vs_same_intent_retry"] >= 10,
            "g5_delta_pp_vs_oracle": results["summary"]["delta_pp_vs_oracle"],
            "g5_pass": results["summary"]["delta_pp_vs_oracle"] >= -5,
        },
    }
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# Bar chart
# ---------------------------------------------------------------------------

def make_bar_chart(results: dict, out_dir: Path) -> Optional[tuple[Path, Path]]:
    """Generate PNG + PDF; return paths or None if matplotlib unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    pols = list(POLICIES)
    display = [POLICY_DISPLAY[p] for p in pols]
    rates = [results["per_policy"][p]["final_success_rate"] for p in pols]
    ns = [results["per_policy"][p]["n"] for p in pols]
    counts = [int(round(r * n)) for r, n in zip(rates, ns)]

    # Color: green for above-chance, red for below — keeps the SIR-fails visual.
    colors = ["#2ecc71" if r >= 0.5 else "#e74c3c" for r in rates]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = list(range(len(pols)))
    bars = ax.bar(x, rates, color=colors, edgecolor="black", linewidth=0.6)
    for bar, count, n in zip(bars, counts, ns):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.02,
            f"{count}/{n}\n({100 * count / n:.0f}%)",
            ha="center", va="bottom", fontsize=10,
        )

    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Final success rate", fontsize=11)
    g4 = results["summary"]["delta_pp_vs_same_intent_retry"]
    g5 = results["summary"]["delta_pp_vs_oracle"]
    ax.set_title(
        "Stage-5 P1 PushCube held-out eval (seeds 100-149, n=50)\n"
        f"G4 = +{g4:.0f}pp (latent vs SIR)   "
        f"G5 = {g5:+.0f}pp (latent vs oracle)",
        fontsize=11,
    )
    ax.axhline(0.5, color="gray", linewidth=0.5, linestyle="--", alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(display, rotation=12, ha="right")
    plt.tight_layout()

    png_path = out_dir / "bar_chart.png"
    pdf_path = out_dir / "bar_chart.pdf"
    fig.savefig(png_path, dpi=150)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


# ---------------------------------------------------------------------------
# HTML gallery
# ---------------------------------------------------------------------------

def _cell_html(success: bool, video_path: Optional[Path], rel_to: Path) -> str:
    """One <td> for the main gallery table — pass/fail badge + optional video link."""
    if success:
        badge_text, bg, fg = "✓ PASS", "#d4edda", "#155724"
    else:
        badge_text, bg, fg = "✗ FAIL", "#f8d7da", "#721c24"
    style = (
        f"background:{bg};color:{fg};text-align:center;padding:6px 8px;"
        f"font-weight:600;"
    )
    if video_path is not None:
        try:
            rel = video_path.relative_to(rel_to)
        except ValueError:
            rel = video_path
        link = f'<br><a href="{rel}" style="color:inherit;text-decoration:underline;">video</a>'
    else:
        link = '<br><span style="color:#888;font-size:0.85em;font-weight:400;">(no video)</span>'
    return f'<td style="{style}">{badge_text}{link}</td>'


def build_gallery_html(
    results: dict,
    renders_dir: Path,
    out_path: Path,
    bar_chart_png: Optional[Path],
) -> dict:
    """Write gallery.html. Returns coverage metadata."""
    seeds = results["config"]["eval_seeds"]
    per_seed_policy = index_per_seed_policy(results)
    out_dir = out_path.parent

    # Discover per-cell MP4s
    found_count: dict[str, int] = {p: 0 for p in POLICIES}
    cell_mp4: dict[tuple[int, str], Optional[Path]] = {}
    for seed in seeds:
        for policy in POLICIES:
            mp4 = find_retry_mp4(renders_dir, seed, policy)
            if mp4 is not None:
                found_count[policy] += 1
            cell_mp4[(seed, policy)] = mp4
    demo_mp4_by_seed: dict[int, Optional[Path]] = {
        seed: find_demo_mp4(renders_dir, seed) for seed in seeds
    }

    # Identify "interesting" failures: any of the *revising* policies failed.
    # same_intent_retry fails on every seed by construction (no revision → no
    # recovery from the blocked initial attempt), so including it would make
    # the "failures-only" section equivalent to the full table. The
    # interesting cases are seeds where latent / oracle / babysteps_selective
    # disagree or where the env hits a hard ceiling.
    REVISING_POLICIES = (
        "latent", "oracle_factor_revision", "babysteps_selective",
    )
    failure_seeds: list[int] = []
    for seed in seeds:
        for policy in REVISING_POLICIES:
            r = per_seed_policy.get((seed, policy))
            if r and not r["final_success"]:
                failure_seeds.append(seed)
                break

    # Build the main gallery rows
    def _row_for(seed: int, include_demo_col: bool) -> str:
        cells = []
        for policy in POLICIES:
            r = per_seed_policy.get((seed, policy))
            success = bool(r["final_success"]) if r else False
            cells.append(_cell_html(success, cell_mp4[(seed, policy)], out_dir))
        demo_td = ""
        if include_demo_col:
            demo_path = demo_mp4_by_seed[seed]
            if demo_path is not None:
                try:
                    rel = demo_path.relative_to(out_dir)
                except ValueError:
                    rel = demo_path
                demo_td = (
                    f'<td style="padding:6px;text-align:center;">'
                    f'<a href="{rel}">demo</a></td>'
                )
            else:
                demo_td = (
                    '<td style="padding:6px;text-align:center;color:#888;'
                    'font-size:0.85em;">(no demo)</td>'
                )
        return (
            f'<tr><td style="padding:6px;text-align:center;font-weight:600;">'
            f"{seed}</td>" + demo_td + "".join(cells) + "</tr>"
        )

    rows_html = [_row_for(seed, include_demo_col=True) for seed in seeds]
    failure_rows_html = [_row_for(seed, include_demo_col=False) for seed in failure_seeds]

    # Per-policy summary table
    pol_summary_rows = []
    for policy in POLICIES:
        pp = results["per_policy"][policy]
        n = pp["n"]
        ns = int(round(pp["final_success_rate"] * n))
        pol_summary_rows.append(
            "<tr>"
            f'<td style="padding:6px;">{POLICY_DISPLAY[policy]}</td>'
            f'<td style="padding:6px;text-align:right;">{ns}/{n}</td>'
            f'<td style="padding:6px;text-align:right;">{pp["final_success_rate"]:.3f}</td>'
            f'<td style="padding:6px;text-align:right;color:#888;">'
            f"{found_count[policy]}/{n} videos</td>"
            "</tr>"
        )

    # Gate banner
    sm = results["summary"]
    g4 = sm["delta_pp_vs_same_intent_retry"]
    g5 = sm["delta_pp_vs_oracle"]
    g4_class = "gate-pass" if g4 >= 10 else "gate-fail"
    g5_class = "gate-pass" if g5 >= -5 else "gate-fail"

    # Bar chart embed
    bar_img_html = ""
    if bar_chart_png is not None and bar_chart_png.exists():
        try:
            rel = bar_chart_png.relative_to(out_dir)
        except ValueError:
            rel = bar_chart_png
        bar_img_html = (
            '<p style="margin:12px 0;"><img src='
            f'"{rel}" alt="bar chart" style="max-width:720px;width:100%;'
            'border:1px solid #ddd;border-radius:4px;"></p>'
        )

    # The render command surfaced for missing MP4s
    render_cmd_default = (
        "python scripts/render_stage0_maniskill.py \\\n"
        "    --task PushCube-v1 --n_episodes 50 --seed_start 100 \\\n"
        "    --out_dir renders/pushcube_heldout/"
    )

    # Build HTML — double-brace any literal `{`/`}` in CSS to survive .format-like ops.
    html = dedent(
        """\
        <!doctype html>
        <html lang="en">
        <head><meta charset="utf-8">
        <title>Stage-5 P1 PushCube held-out — eval gallery</title>
        <style>
          body { font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
                 margin: 24px; max-width: 1200px; color: #222; line-height: 1.5; }
          h1 { margin-top: 0; }
          h2 { border-bottom: 1px solid #ddd; padding-bottom: 6px; margin-top: 32px; }
          table { border-collapse: collapse; margin: 12px 0; }
          th, td { border: 1px solid #ddd; }
          th { background: #f4f4f4; padding: 8px 10px; text-align: left; }
          .gate-pass { color: #155724; font-weight: 600; }
          .gate-fail { color: #c0392b; font-weight: 600; }
          .summary-box { background: #f8f9fa; padding: 12px 18px;
                         border-left: 4px solid #007bff; margin: 12px 0;
                         border-radius: 0 4px 4px 0; }
          .cmd-box { background: #2d2d2d; color: #f8f8f2;
                     padding: 12px 16px; border-radius: 4px;
                     overflow-x: auto; font-family: ui-monospace, Menlo, monospace;
                     font-size: 0.9em; white-space: pre; }
          a { color: #007bff; text-decoration: none; }
          a:hover { text-decoration: underline; }
          .meta { color: #888; font-size: 0.85em; }
        </style>
        </head><body>
        """
    )
    html += f"""\
<h1>Stage-5 P1 PushCube held-out eval (seeds 100-149, n=50)</h1>

<div class="summary-box">
  <strong>Source data:</strong>
  <code>reports/stage5/p1_vision_g4_g5/PushCube-v1/p1_results.json</code><br>
  <strong>Task:</strong> {results['config']['task']}
  &nbsp;|&nbsp; <strong>Real env:</strong> {not results['config']['fake']}<br>
  <strong>Pack:</strong> <code>{results['config']['pack_dir']}</code><br>
  <strong>Features:</strong> <code>{results['config']['features_dir']}</code>
  (spatial_mean DINOv2 ViT-B/14, 768-dim)
</div>

<h2>G4 / G5 gates</h2>
<p>
  <span class="{g4_class}">G4 = {g4:+.0f}pp (latent vs same_intent_retry)</span>
  — threshold ≥ +10pp<br>
  <span class="{g5_class}">G5 = {g5:+.0f}pp (latent vs oracle)</span>
  — threshold ≥ −5pp
</p>

<h2>Per-policy success</h2>
{bar_img_html}
<table>
  <thead><tr>
    <th>Policy</th><th>Success / n</th><th>Rate</th><th>Videos found</th>
  </tr></thead>
  <tbody>
  {''.join(pol_summary_rows)}
  </tbody>
</table>

<h2>Failures-only ({len(failure_seeds)} seeds where at least one revising policy failed)</h2>
<p>Filter excludes <code>same_intent_retry</code>'s universal failure (it cannot recover by
construction — no revision → no recovery from the blocked initial attempt).
This leaves the seeds where the <em>revising</em> policies
(<code>latent</code>, <code>oracle</code>, <code>babysteps_selective</code>) disagree or where
the env hits a hard ceiling. Failed cells highlighted red.</p>
<table>
  <thead><tr>
    <th>seed</th>
    {''.join(f'<th>{POLICY_DISPLAY[p]}</th>' for p in POLICIES)}
  </tr></thead>
  <tbody>
  {''.join(failure_rows_html) if failure_rows_html else '<tr><td colspan="5" style="padding:8px;text-align:center;color:#888;">No failures.</td></tr>'}
  </tbody>
</table>

<h2>Full per-seed gallery</h2>
<p>One row per seed (50 total). Click "video" if available.</p>
<table>
  <thead><tr>
    <th>seed</th>
    <th>demo</th>
    {''.join(f'<th>{POLICY_DISPLAY[p]}</th>' for p in POLICIES)}
  </tr></thead>
  <tbody>
  {''.join(rows_html)}
  </tbody>
</table>

<h2>Missing video renders</h2>
<p>The eval rollout at <code>scripts/stage5_p1_run_eval.py</code> measures
success rates but does not emit MP4s. For seeds 100-149 the standard
3-phase MP4 set (<code>1_demo</code>, <code>2_attempt_blocked</code>,
<code>3_retry</code>) can be regenerated via the existing
<code>render_stage0_maniskill.py</code> pipeline. On a GPU node:</p>

<div class="cmd-box">{render_cmd_default}</div>

<p>This produces the standard babysteps_selective render for each seed —
written to <code>renders/pushcube_heldout/videos_maniskill/</code>.
Re-run this report with
<code>--renders-dir renders/pushcube_heldout/videos_maniskill</code>
to pick those up in the demo column and the babysteps_selective video links.</p>

<p><strong>Per-policy MP4s</strong> (separate latent / oracle / same_intent_retry
clips per seed) are NOT producible by the current repo:
<code>render_stage0_maniskill.py</code> hardcodes the babysteps_selective loop
and <code>render_baseline_contrast.py</code> only renders the
<code>selective vs full_replan_analogue</code> contrast. Adding per-policy
rendering would be a small extension to <code>render_stage0_maniskill.py</code>
(e.g. a <code>--policy</code> flag swapping the RetryPolicy in
<code>run_episode</code>) but is out of scope for this report.</p>

<p class="meta">Generated by <code>scripts/report_stage5_pushcube_heldout.py</code>.</p>
</body></html>
"""
    out_path.write_text(html)

    return {
        "n_failures": len(failure_seeds),
        "failure_seeds": failure_seeds,
        "failure_seeds_by_policy": {
            policy: [
                r["seed"]
                for r in results["per_policy"][policy]["rows"]
                if not r["final_success"]
            ]
            for policy in POLICIES
        },
        "videos_found": found_count,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--results", type=Path,
        default=Path("reports/stage5/p1_vision_g4_g5/PushCube-v1/p1_results.json"),
        help="Path to the S5 Half B eval results JSON.",
    )
    p.add_argument(
        "--renders-dir", type=Path,
        default=Path("renders/pushcube/videos_maniskill"),
        help="Directory to scan for existing per-seed MP4s.",
    )
    p.add_argument(
        "--out-dir", type=Path,
        default=Path("reports/stage5/p1_vision_g4_g5/PushCube-v1/report_gallery"),
        help="Output dir for CSV/JSON/PNG/PDF/HTML.",
    )
    args = p.parse_args(argv)

    if not args.results.exists():
        print(f"ERROR: --results {args.results} not found", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = load_results(args.results)

    # 1) Tabular outputs
    write_per_policy_csv(results, args.out_dir / "per_policy_summary.csv")
    print(f"wrote {args.out_dir / 'per_policy_summary.csv'}")

    per_seed_rows = build_per_seed_long(results)
    write_per_seed_csv(per_seed_rows, args.out_dir / "per_seed_long.csv")
    print(f"wrote {args.out_dir / 'per_seed_long.csv'}  ({len(per_seed_rows)} rows)")

    write_summary_json(results, args.out_dir / "summary.json")
    print(f"wrote {args.out_dir / 'summary.json'}")

    # 2) Bar chart (PNG + PDF). Matplotlib is optional.
    chart = make_bar_chart(results, args.out_dir)
    bar_png = None
    if chart is None:
        print("WARNING: matplotlib not available — skipped bar chart")
    else:
        bar_png = chart[0]
        for path in chart:
            print(f"wrote {path}")

    # 3) HTML gallery
    meta = build_gallery_html(
        results, args.renders_dir, args.out_dir / "gallery.html", bar_png,
    )
    print(f"wrote {args.out_dir / 'gallery.html'}  "
          f"({meta['n_failures']} seeds with at least one failure)")

    print()
    print(f"renders scanned: {args.renders_dir}"
          f"  (exists={args.renders_dir.exists()})")
    print("video coverage per policy:")
    for policy in POLICIES:
        print(f"  {POLICY_DISPLAY[policy]:32s} {meta['videos_found'][policy]:>3d}"
              f" / {results['per_policy'][policy]['n']}")

    print()
    print("Failure seeds per policy:")
    any_fail = False
    for policy in POLICIES:
        seeds_fail = meta["failure_seeds_by_policy"][policy]
        if seeds_fail:
            any_fail = True
            print(f"  {POLICY_DISPLAY[policy]:32s} {seeds_fail}")
    if not any_fail:
        print("  (no failures)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
