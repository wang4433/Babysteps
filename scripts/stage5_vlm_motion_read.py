"""Stage-5 option-3 probe (v2): VLM reads object_motion via BEFORE/AFTER frames.

Supersedes the single-strip approach (``stage5_vlm_intent_read.py``), which
the diagnostics showed makes the VLM read panel-layout as motion. Here START
and END frames are fed as two SEPARATE images; the VLM reports an
image-relative direction (left/right/up/down) which is mapped to the world
``object_motion`` token via a fixed, blob-calibrated lookup.

- PushCube: world axes separate cleanly in image space (+x↔left, -x↔right),
  so we score accuracy + confusion against the oracle label.
- StackCube: calibration shows all four directions project to "rightish" in
  the fixed oblique camera (entangled), so a clean image→world map does not
  exist. We emit the (oracle token × VLM direction) CROSS-TAB as evidence of
  (non-)separability rather than a misleading accuracy number.

Reuses the P2 InternVL3.5 loader. numpy/imageio import lazily (login-safe).

Run (GPU, handover env)::

    python scripts/stage5_vlm_motion_read.py \\
        --out-dir reports/stage5/p3demo_vlm_motion_read
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

# PushCube image-direction -> world object_motion token (blob-calibrated:
# world +x demos move the cube lower-LEFT; -x demos upper-RIGHT).
_PUSH_MAP = {
    "left": "translate_+x", "down": "translate_+x",
    "right": "translate_-x", "up": "translate_-x", "none": None,
}

# Per-factor object_motion accuracy references for the report comparison.
_REF = {
    "PushCube-v1": {"dinov2_probe": 0.95, "learned_slot": 0.95},
    "StackCube-v1": {"dinov2_probe": 0.42, "learned_slot": 0.95},
}
_GATE = 0.90

# How each task is scored: "mapped" (clean image→world map) or "crosstab".
_MODE = {"PushCube-v1": "mapped", "StackCube-v1": "crosstab"}


def _seed(rec: dict) -> int:
    return int(rec["episode_id"].split("_")[-1])


def load_labels(labels_root: Path, task: str) -> dict[int, str]:
    out: dict[int, str] = {}
    with (labels_root / task / "samples.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                out[_seed(rec)] = rec["execution"]["initial_intent"]["object_motion"]
    return out


def evaluate_task(
    *, task: str, client, frames_root: Path, labels_root: Path,
    frames_out: Path, limit: Optional[int],
) -> dict:
    import numpy as np
    import imageio.v2 as imageio

    labels = load_labels(labels_root, task)
    frames_dir = frames_root / task / "frames"
    out_dir = frames_out / task
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for seed in sorted(labels):
        npz = frames_dir / f"seed_{seed:04d}.npz"
        if not npz.exists():
            continue
        fr = np.load(npz)["frames"]
        start_p = out_dir / f"seed_{seed:04d}_start.png"
        end_p = out_dir / f"seed_{seed:04d}_end.png"
        imageio.imwrite(start_p, np.asarray(fr[0])[..., :3])
        imageio.imwrite(end_p, np.asarray(fr[-1])[..., :3])
        direction = client.read_motion_direction(
            task=task, start_path=str(start_p), end_path=str(end_p),
        )
        gt = labels[seed]
        row = {"seed": seed, "gt": gt, "vlm_direction": direction}
        if _MODE[task] == "mapped":
            pred = _PUSH_MAP.get(direction) if direction else None
            row["pred_world"] = pred
            row["correct"] = bool(pred == gt)
        rows.append(row)
        print(f"  {task} seed {seed:04d}  gt={gt:14s} dir={direction} "
              f"{'-> ' + str(row.get('pred_world')) if _MODE[task]=='mapped' else ''}")
        if limit is not None and len(rows) >= limit:
            break

    result = {"task": task, "n": len(rows), "mode": _MODE[task],
              "ref": _REF.get(task, {}), "gate": _GATE, "rows": rows}

    if _MODE[task] == "mapped":
        acc = sum(r["correct"] for r in rows) / len(rows) if rows else 0.0
        dist = Counter(r["gt"] for r in rows)
        result["accuracy"] = acc
        result["majority_baseline"] = (max(dist.values()) / len(rows)) if rows else 0.0
        result["passes_gate"] = acc >= _GATE
    else:
        # cross-tab: oracle token (rows) x VLM image direction (cols)
        ct: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for r in rows:
            ct[r["gt"]][r["vlm_direction"] or "none/parse_fail"] += 1
        # best-case accuracy if we mapped each VLM direction to its majority
        # oracle token (an OPTIMISTIC upper bound on any image→world map):
        by_dir: dict[str, Counter] = defaultdict(Counter)
        for r in rows:
            by_dir[r["vlm_direction"] or "none"][r["gt"]] += 1
        best = sum(c.most_common(1)[0][1] for c in by_dir.values())
        result["crosstab"] = {k: dict(v) for k, v in ct.items()}
        result["best_case_mapped_accuracy"] = best / len(rows) if rows else 0.0
        result["majority_baseline"] = (
            max(Counter(r["gt"] for r in rows).values()) / len(rows)
        ) if rows else 0.0
    return result


def _render_md(results: list[dict]) -> str:
    L = [
        "# Stage-5 option-3 probe v2 — VLM reads object_motion (before/after)",
        "",
        "START and END frames fed as two SEPARATE images; VLM reports an",
        "image-relative direction, mapped to the world token via a fixed",
        "blob-calibrated lookup. (Supersedes the panel-strip run, which the",
        "VLM read as layout, not motion.)",
        "",
    ]
    for r in results:
        L.append(f"## {r['task']}  (n={r['n']}, mode={r['mode']})")
        ref = r["ref"]
        L.append(
            f"- reference: DINOv2 probe {ref.get('dinov2_probe')}, "
            f"learned slot {ref.get('learned_slot')}, "
            f"majority {r.get('majority_baseline'):.2f}"
        )
        if r["mode"] == "mapped":
            verdict = "PASS" if r.get("passes_gate") else "BELOW-GATE"
            L.append(f"- **VLM read accuracy: {r['accuracy']:.3f}**  "
                     f"(gate {_GATE:.2f}: {verdict})")
            conf = defaultdict(lambda: defaultdict(int))
            for row in r["rows"]:
                conf[row["gt"]][row.get("pred_world") or "none"] += 1
            cols = sorted({c for d in conf.values() for c in d})
            L += ["", "| gt \\ pred | " + " | ".join(cols) + " |",
                  "| --- | " + " | ".join("---" for _ in cols) + " |"]
            for gt in sorted(conf):
                L.append(f"| {gt} | " + " | ".join(str(conf[gt].get(c, 0)) for c in cols) + " |")
        else:
            L.append(f"- **best-case image→world map accuracy (optimistic upper "
                     f"bound): {r['best_case_mapped_accuracy']:.3f}**  "
                     f"(majority {r['majority_baseline']:.2f}) — if this is near "
                     "majority, the directions are not separable in this view.")
            ct = r["crosstab"]
            cols = sorted({c for d in ct.values() for c in d})
            L += ["", "| oracle \\ VLM dir | " + " | ".join(cols) + " |",
                  "| --- | " + " | ".join("---" for _ in cols) + " |"]
            for gt in sorted(ct):
                L.append(f"| {gt} | " + " | ".join(str(ct[gt].get(c, 0)) for c in cols) + " |")
        L.append("")
    return "\n".join(L)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", action="append", default=None)
    p.add_argument("--frames-root", type=Path,
                   default=Path("datasets/stage5/varied_intent"))
    p.add_argument("--labels-root", type=Path,
                   default=Path("datasets/stage4/varied_intent"))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--frames-out", type=Path, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-tiles", type=int, default=12)
    p.add_argument("--mock", action="store_true")
    args = p.parse_args(argv)

    tasks = args.task or ["PushCube-v1", "StackCube-v1"]
    frames_out = args.frames_out or (args.out_dir / "frames")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.mock:
        from babysteps.stage5.vlm_attribute import MockVLMClient
        client = MockVLMClient()
    else:
        from babysteps.stage5.vlm_attribute import InternVLClient
        client = InternVLClient(max_num_tiles=args.max_tiles)
        client.load()

    results = []
    for task in tasks:
        print(f"=== {task} ===")
        results.append(evaluate_task(
            task=task, client=client, frames_root=args.frames_root,
            labels_root=args.labels_root, frames_out=frames_out, limit=args.limit,
        ))

    (args.out_dir / "report.json").write_text(json.dumps(results, indent=2))
    (args.out_dir / "report.md").write_text(_render_md(results))
    print(f"\nWrote {args.out_dir}/report.json + report.md")
    for r in results:
        if r["mode"] == "mapped":
            print(f"  {r['task']}: accuracy {r['accuracy']:.3f} "
                  f"({'PASS' if r.get('passes_gate') else 'BELOW-GATE'})")
        else:
            print(f"  {r['task']}: best-case map {r['best_case_mapped_accuracy']:.3f} "
                  f"vs majority {r['majority_baseline']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
