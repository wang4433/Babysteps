"""Stage-5 option-3 de-risking probe: can a VLM READ object_motion off a demo?

Pokes InternVL3.5-8B with a third-person DEMO strip (start/middle/end panels)
and asks for the ``object_motion`` token, then measures agreement with the
sim-oracle label. This is the cheapest decisive test for whether a VLM is a
viable *distillation supervision* source for the varied-intent latent slots
(it does NOT train anything).

Reuses the P2 InternVL3.5 loader + dynamic-tile pipeline. Demo frames come
from the P1 cache; labels from the Stage-4 varied-intent cut.

Pre-registered kill criterion (design workflow w0a3e4rpf, point 8):
demo-frame agreement < 0.90 ⇒ the distilled target is below the G1 gate by
construction ⇒ report as a relational-gap-reduction ablation, not a P1 pass.

Sim-free helpers (build_strip, scoring) are import-safe; numpy/imageio import
lazily inside function bodies so the module loads on the login node.

Run (GPU, handover env)::

    python scripts/stage5_vlm_intent_read.py \\
        --task PushCube-v1 --task StackCube-v1 \\
        --out-dir reports/stage5/p3demo_vlm_intent_read
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Optional

# Per-factor object_motion accuracy reference points, for the report's
# comparison column (P1 G1 spatial_mean DINOv2 linear probe; M2a learned slot).
_REF = {
    "PushCube-v1": {"dinov2_probe": 0.95, "learned_slot": 0.95},
    "StackCube-v1": {"dinov2_probe": 0.42, "learned_slot": 0.95},
}
_GATE = 0.90  # G1 absolute gate; demo-read agreement below this can't certify.


def _seed_from_episode_id(rec: dict) -> int:
    return int(rec["episode_id"].split("_")[-1])


def load_labels(labels_root: Path, task: str) -> dict[int, str]:
    """{seed: object_motion} from the varied-intent samples.jsonl."""
    jsonl = labels_root / task / "samples.jsonl"
    out: dict[int, str] = {}
    with jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            seed = _seed_from_episode_id(rec)
            out[seed] = rec["execution"]["initial_intent"]["object_motion"]
    return out


def build_strip(frames, *, n_panels: int = 3, downsample: int = 2):
    """Horizontal start..end strip from (T,H,W,3) frames. Lazy numpy."""
    import numpy as np

    T = len(frames)
    if T == 0:
        raise ValueError("empty frame stack")
    if n_panels == 3:
        idx = [0, T // 2, T - 1]
    else:
        idx = [int(round(i * (T - 1) / (n_panels - 1))) for i in range(n_panels)]
    s = downsample
    return np.hstack([np.asarray(frames[i])[::s, ::s, :3] for i in idx])


def _accuracy(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    return sum(r["correct"] for r in rows) / len(rows)


def _confusion(rows: list[dict], menu: tuple[str, ...]) -> dict[str, dict[str, int]]:
    conf = {gt: {pr: 0 for pr in list(menu) + ["parse_fail"]} for gt in menu}
    for r in rows:
        gt, pr = r["gt"], r["pred"]
        if gt not in conf:
            conf[gt] = {pr2: 0 for pr2 in list(menu) + ["parse_fail"]}
        key = pr if pr is not None else "parse_fail"
        conf[gt][key] = conf[gt].get(key, 0) + 1
    return conf


def evaluate_task(
    *, task: str, client, frames_root: Path, labels_root: Path,
    strips_dir: Path, n_panels: int, limit: Optional[int],
) -> dict:
    import numpy as np  # noqa: F401  (lazy; keeps login-node import clean)
    import imageio.v2 as imageio

    from babysteps.stage5.vlm_attribute import OBJECT_MOTION_MENU_4

    labels = load_labels(labels_root, task)
    frames_dir = frames_root / task / "frames"
    out_strips = strips_dir / task
    out_strips.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for seed in sorted(labels):
        npz = frames_dir / f"seed_{seed:04d}.npz"
        if not npz.exists():
            continue
        frames = np.load(npz)["frames"]
        strip = build_strip(frames, n_panels=n_panels)
        strip_path = out_strips / f"seed_{seed:04d}.png"
        imageio.imwrite(strip_path, strip)
        pred = client.read_object_motion(task=task, image_path=str(strip_path))
        gt = labels[seed]
        rows.append({
            "seed": seed, "gt": gt, "pred": pred, "correct": bool(pred == gt),
        })
        print(f"  {task} seed {seed:04d}  gt={gt:14s} pred={pred}")
        if limit is not None and len(rows) >= limit:
            break

    acc = _accuracy(rows)
    dist = Counter(r["gt"] for r in rows)
    majority = max(dist.values()) / len(rows) if rows else 0.0
    n_parse_fail = sum(1 for r in rows if r["pred"] is None)
    return {
        "task": task,
        "n": len(rows),
        "object_motion_agreement": acc,
        "majority_baseline": majority,
        "n_parse_fail": n_parse_fail,
        "label_dist": dict(dist),
        "confusion": _confusion(rows, OBJECT_MOTION_MENU_4),
        "ref": _REF.get(task, {}),
        "gate": _GATE,
        "passes_gate": acc >= _GATE,
        "rows": rows,
    }


def _render_md(results: list[dict]) -> str:
    lines = [
        "# Stage-5 option-3 probe — VLM reads object_motion off the demo",
        "",
        "VLM-vs-oracle agreement reading `object_motion` from a third-person",
        "demo strip (start/middle/end). No training. Compared against the P1",
        "G1 DINOv2 linear probe and the M2a learned slot on the same factor.",
        "",
        f"Pre-registered kill criterion: agreement < {_GATE:.2f} ⇒ below the G1",
        "gate by construction ⇒ relational-gap-reduction ablation, not a pass.",
        "",
        "| task | n | VLM read | DINOv2 probe | learned slot | majority | gate |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        ref = r["ref"]
        verdict = "PASS" if r["passes_gate"] else "BELOW-GATE"
        lines.append(
            f"| {r['task']} | {r['n']} | **{r['object_motion_agreement']:.3f}** | "
            f"{ref.get('dinov2_probe', float('nan')):.2f} | "
            f"{ref.get('learned_slot', float('nan')):.2f} | "
            f"{r['majority_baseline']:.2f} | {verdict} |"
        )
    lines.append("")
    for r in results:
        lines.append(f"## {r['task']} — confusion (rows=gt, cols=pred)")
        lines.append("")
        menu = sorted({k for k in r["confusion"]})
        cols = sorted({c for row in r["confusion"].values() for c in row})
        lines.append("| gt \\ pred | " + " | ".join(cols) + " |")
        lines.append("| --- | " + " | ".join("---" for _ in cols) + " |")
        for gt in menu:
            row = r["confusion"][gt]
            lines.append(f"| {gt} | " + " | ".join(str(row.get(c, 0)) for c in cols) + " |")
        lines.append("")
        if r["n_parse_fail"]:
            lines.append(f"_parse failures: {r['n_parse_fail']}_\n")
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", action="append", default=None,
                   help="repeat per task; default PushCube-v1 + StackCube-v1")
    p.add_argument("--frames-root", type=Path,
                   default=Path("datasets/stage5/varied_intent"))
    p.add_argument("--labels-root", type=Path,
                   default=Path("datasets/stage4/varied_intent"))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--strips-dir", type=Path, default=None,
                   help="where demo strips are written (default <out-dir>/strips)")
    p.add_argument("--n-panels", type=int, default=3)
    p.add_argument("--limit", type=int, default=None, help="cap seeds (smoke)")
    p.add_argument("--max-tiles", type=int, default=12)
    p.add_argument("--mock", action="store_true",
                   help="use MockVLMClient (plumbing smoke; no GPU)")
    args = p.parse_args(argv)

    tasks = args.task or ["PushCube-v1", "StackCube-v1"]
    strips_dir = args.strips_dir or (args.out_dir / "strips")
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
            labels_root=args.labels_root, strips_dir=strips_dir,
            n_panels=args.n_panels, limit=args.limit,
        ))

    (args.out_dir / "report.json").write_text(json.dumps(results, indent=2))
    (args.out_dir / "report.md").write_text(_render_md(results))
    print(f"\nWrote {args.out_dir}/report.json + report.md")
    for r in results:
        verdict = "PASS" if r["passes_gate"] else "BELOW-GATE"
        print(f"  {r['task']}: VLM read {r['object_motion_agreement']:.3f} "
              f"(DINOv2 {r['ref'].get('dinov2_probe')}, "
              f"learned {r['ref'].get('learned_slot')}) [{verdict}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
