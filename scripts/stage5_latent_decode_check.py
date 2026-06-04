"""Stage-5 — latent-input faithfulness check (GPU-free).

Answers the load-bearing pre-flight question for the latent-intent pivot:

    On the exact held-out seeds the P2 selectivity harness uses, how well
    does the vision-decoded initial intent (DINOv2 -> IntentHead ->
    nearest-centroid) reproduce the hand-authored JSON intent it would
    replace?

If latent ≈ JSON here, swapping the P2 input from JSON to latent is clean
and honest (the cached failure frames stay valid, the oracle labels stay
consistent). If not, the gap is surfaced *before* any VLM/GPU spend.

Per-factor agreement is reported for the factors the pack decodes from
vision (those with a centroid bank); trivially-constant factors are
listed but not scored (they are task constants, filled from a base).

Sim-free: numpy + CPU torch. No simulator, no VLM.

Example::

    python scripts/stage5_latent_decode_check.py \\
        --task PushCube-v1 \\
        --pack-dir models/stage5/p1_vision/PushCube-v1 \\
        --features-dir datasets/stage5/varied_intent/PushCube-v1/features \\
        --episodes datasets/stage5/p2_vlm/PushCube-v1/episodes.jsonl \\
        --out-dir reports/stage5/latent_decode_check/PushCube-v1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from babysteps.schemas import INTENT_FIELDS, Intent  # noqa: E402
from babysteps.stage4.latent_policy import load_latent_pack  # noqa: E402
from babysteps.stage5.latent_intent import (  # noqa: E402
    decode_latent_factors, latent_factor_names,
)


def _read_episodes(path: Path) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    # P2 episodes flag failure rows; if the flag is absent keep everything.
    fails = [r for r in rows if r.get("is_failure", False)]
    return fails if fails else rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True)
    p.add_argument("--pack-dir", type=Path, required=True)
    p.add_argument("--features-dir", type=Path, required=True)
    p.add_argument("--episodes", type=Path, required=True,
                   help="P2 episodes.jsonl carrying seed + initial_intent.")
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args(argv)

    pack = load_latent_pack(args.pack_dir)
    decodable = latent_factor_names(pack)
    print(f"pack decodes from vision: {decodable}")

    episodes = _read_episodes(args.episodes)
    print(f"loaded {len(episodes)} episodes for {args.task}")

    # Per-factor agreement counters over the decodable factors.
    agree = {f: 0 for f in decodable}
    total = 0
    missing_feature = 0
    rows: list[dict] = []
    # Per-factor confusion (decoded vs stored token) for diagnostics.
    confusion: dict[str, dict[str, int]] = {f: {} for f in decodable}

    for ep in episodes:
        seed = ep["seed"]
        fpath = args.features_dir / f"seed_{seed:04d}_dinov2.npy"
        if not fpath.exists():
            missing_feature += 1
            continue
        z = np.load(fpath).astype(np.float32)
        decoded = decode_latent_factors(pack, z)
        stored = ep["initial_intent"]
        total += 1
        row = {"seed": seed, "per_factor": {}}
        for f in decodable:
            dv, sv = decoded.get(f), stored.get(f)
            ok = (dv == sv)
            agree[f] += int(ok)
            key = f"{dv}->{sv}"
            confusion[f][key] = confusion[f].get(key, 0) + 1
            row["per_factor"][f] = {"decoded": dv, "stored": sv, "agree": ok}
        # Whole-decodable-intent exact match.
        row["all_decodable_agree"] = all(
            row["per_factor"][f]["agree"] for f in decodable
        )
        rows.append(row)

    per_factor_rate = {
        f: (agree[f] / total if total else None) for f in decodable
    }
    all_agree_rate = (
        sum(r["all_decodable_agree"] for r in rows) / total if total else None
    )
    constant_factors = tuple(
        f for f in INTENT_FIELDS if f not in decodable
    )

    summary = {
        "task": args.task,
        "n_scored": total,
        "n_missing_feature": missing_feature,
        "decodable_factors": list(decodable),
        "constant_factors_filled_from_base": list(constant_factors),
        "per_factor_agreement": per_factor_rate,
        "all_decodable_agree_rate": all_agree_rate,
        "confusion_decoded_to_stored": confusion,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "decode_check.json").write_text(
        json.dumps({"summary": summary, "per_episode": rows},
                   indent=2, sort_keys=True) + "\n"
    )
    _write_md(args.out_dir / "report.md", args, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nwrote {args.out_dir}/decode_check.json + report.md")
    return 0


def _write_md(out_path: Path, args, summary: dict) -> None:
    lines = [
        f"# Stage-5 latent-input faithfulness check — {summary['task']}",
        "",
        "How well the vision-decoded initial intent (DINOv2 → IntentHead →",
        "nearest-centroid) reproduces the hand-authored JSON intent it would",
        "replace, on the P2 held-out seeds.",
        "",
        f"- Pack: `{args.pack_dir}`",
        f"- Features: `{args.features_dir}`",
        f"- Episodes: `{args.episodes}`",
        f"- Scored: **{summary['n_scored']}** "
        f"(missing feature files: {summary['n_missing_feature']})",
        f"- Decoded-from-vision factors: "
        f"`{', '.join(summary['decodable_factors']) or '(none)'}`",
        f"- Constant factors (filled from task base): "
        f"`{', '.join(summary['constant_factors_filled_from_base'])}`",
        "",
        "## Per-factor latent-vs-JSON agreement",
        "",
        "| factor | agreement |",
        "|---|---|",
    ]
    for f, r in summary["per_factor_agreement"].items():
        lines.append(f"| `{f}` | {r:.3f} |" if r is not None else f"| `{f}` | n/a |")
    aar = summary["all_decodable_agree_rate"]
    lines.extend([
        "",
        f"- **All decodable factors agree (exact match): "
        f"{aar:.3f}**" if aar is not None else "- all-agree: n/a",
        "",
        "## Confusion (decoded → stored), per factor",
        "",
    ])
    for f, conf in summary["confusion_decoded_to_stored"].items():
        lines.append(f"- `{f}`: " + ", ".join(
            f"`{k}`×{v}" for k, v in sorted(conf.items(), key=lambda kv: -kv[1])
        ))
    out_path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
