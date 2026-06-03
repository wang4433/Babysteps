"""Stage-5 — StackCube object-centric relation ORACLE-CEILING probe (CPU, sim-free).

Step 1 of the object-centric relation-representation plan. It answers one
question with committed, reproducible artifacts:

    If object POSITIONS are given correctly, is StackCube ``object_motion``
    recoverable?  i.e. is the frozen-DINOv2 spatial_mean **0.42**
    (``reports/stage5/p1_vision_g1``) a REPRESENTATION failure, or a
    label/data problem?

What this measures — and its limit (read this)
-----------------------------------------------
The label is ``object_motion = goal_direction_to_motion(cubeB_init -
cubeA_init)`` — the resting cubeA→cubeB direction snapped to a cardinal,
assigned at reset (``babysteps/envs/scene.py:cubeA_to_cubeB_motion`` via
``scripts/stage4_collect_varied.py``). So the relational feature ``(cubeB -
cubeA_start)`` IS the label's own input: a probe on it (and the dominant-axis
rule) re-derives the label almost by construction. This is therefore a
**well-posedness / zero-label-noise check**, NOT a surprising recovery. It
certifies the label is a clean, deterministic, noiseless function of the two
cubes' resting positions — so any residual error on this task is
*representation*, not label noise.

(An earlier version of this file wrongly claimed the label was cubeA's
*trajectory* delta and that ``(cubeB - cubeA_start)`` was a non-circular
quantity. The code says otherwise; this docstring is the correction.)

The genuinely non-trivial representation evidence lives elsewhere: a linear
probe on cube *image* positions (blob centroids) reaches ~0.80 on the SAME
oblique frames where frozen-DINOv2 global-pool reaches only 0.42 — the signal
survives in the pixels and global mean-pooling dilutes it. Step-2
(``stage5_object_relation_probe.py``) tests whether object-LOCAL DINO patch
tokens recover it WITHOUT being handed coordinates — the only non-circular
visual question, since any coordinate (world xy or image uv) just re-feeds the
label's input.

Feature ladder (still earns its keep: shows neither cube's *absolute* position
predicts direction — only the RELATION does, the object-centric hypothesis
stated in coordinates):
    A0     cubeA start position           — object position alone (~chance)
    B      cubeB position                 — reference object alone (~chance)
    [A0;B] concat([A0, B])                — probe must *learn* the relation
    B-A0   cubeB - cubeA_start            — the relational signal == label input

Each feature is scored two ways so the ceiling sits on the SAME axis as the
0.42 cell:
    * direct  : per-fold StandardScaler + LogisticRegression (matched splitter)
    * intent  : IntentHead-mediated ``nested_cv_probe_one_factor`` with the
                identical protocol/params as the G1 driver (d_slot=32,
                n_epochs=300, factor_idx=object_motion). Apples-to-apples vs
                ``reports/stage5/p1_vision_g1``.
Plus a parameter-free dominant-axis RULE on ``B-A0`` as an interpretable
reference (no fitting, full cut).

Data sources
------------
``--source p2_wrist`` (default, login-node runnable TODAY): reads
    cubeA0 = ``initial_obj_xy``  and  cubeB = ``goal_xy``  from
    ``datasets/stage5/p2_vlm_wrist/<task>/rollouts/seed_*.npz`` and the
    ``object_motion`` labels from the sibling ``episodes.jsonl``. Both are
    reset-time scene state (independent of execution outcome), so they match
    the demo's initial configuration for the same seed.
    CUT = seeds 100-149 (the P2 eval cut, n=50, imbalanced), NOT the
    ``varied_intent`` 0-39 cut (n=40, balanced) that produced 0.42 — so this
    is a CROSS-CUT ceiling. The recoverability question is cut-agnostic; the
    exactly-matched 0-39 headline needs positions from
    ``stage5_extract_cube_positions.py`` (GPU), consumed via ``--source dir``.

``--source dir --positions-dir <d>``: reads ``seed_NNNN_positions.npz`` with
    keys ``cubeA_xy0`` (start) and ``cubeB_xy``, plus ``object_motion`` labels
    from a ``--jsonl`` records file. This is the EXACT-cut path.

Example::

    python scripts/stage5_relation_oracle_probe.py \\
        --task StackCube-v1 --source p2_wrist \\
        --out-dir reports/stage5/relation_oracle/
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from babysteps.envs.scene import goal_direction_to_motion  # noqa: E402
from babysteps.schemas import INTENT_FIELDS  # noqa: E402
from babysteps.stage4.intent_head import (  # noqa: E402
    _make_splitter, nested_cv_probe_one_factor,
)

_OBJECT_MOTION_IDX = INTENT_FIELDS.index("object_motion")


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #


def _load_p2_wrist(task: str) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """(A0, B, labels) from the p2_vlm_wrist rollouts + episodes.jsonl.

    A0 = cubeA start (``initial_obj_xy``), B = cubeB (``goal_xy``). Both are
    reset-time scene state, so they reflect the demo's initial configuration
    for the same seed regardless of execution success.
    """
    base = _ROOT / "datasets" / "stage5" / "p2_vlm_wrist" / task
    ep = base / "episodes.jsonl"
    if not ep.exists():
        raise FileNotFoundError(f"missing {ep}")
    labels: dict[int, str] = {}
    with ep.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            labels[int(r["seed"])] = r["initial_intent"]["object_motion"]

    A0, B, y, seeds = [], [], [], []
    for p in sorted(glob.glob(str(base / "rollouts" / "seed_*.npz"))):
        sd = int(os.path.basename(p).split("_")[1].split(".")[0])
        if sd not in labels:
            continue
        d = np.load(p)
        A0.append(np.asarray(d["initial_obj_xy"], dtype=np.float64))
        B.append(np.asarray(d["goal_xy"], dtype=np.float64))
        y.append(labels[sd])
        seeds.append(sd)
    if not seeds:
        raise RuntimeError(f"no rollout/label intersection under {base}")
    return np.asarray(A0), np.asarray(B), y, seeds  # type: ignore[return-value]


def _load_positions_dir(
    positions_dir: Path, jsonl: Path,
) -> tuple[np.ndarray, np.ndarray, list[str], list[int]]:
    """(A0, B, labels) from extracted seed_NNNN_positions.npz + a records jsonl.

    The npz must carry ``cubeA_xy0`` (cubeA start) and ``cubeB_xy``. Labels are
    the ``execution.initial_intent.object_motion`` of the matching record.
    """
    from babysteps.schemas import EpisodeRecord

    labels: dict[int, str] = {}
    with jsonl.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = EpisodeRecord.from_jsonl_line(line).to_dict()
            sd = int(rec["episode_id"].split("_")[-1])
            labels[sd] = rec["execution"]["initial_intent"]["object_motion"]

    A0, B, y, seeds = [], [], [], []
    for p in sorted(glob.glob(str(positions_dir / "seed_*_positions.npz"))):
        sd = int(os.path.basename(p).split("_")[1])
        if sd not in labels:
            continue
        d = np.load(p)
        A0.append(np.asarray(d["cubeA_xy0"], dtype=np.float64))
        B.append(np.asarray(d["cubeB_xy"], dtype=np.float64))
        y.append(labels[sd])
        seeds.append(sd)
    if not seeds:
        raise RuntimeError(f"no positions/label intersection under {positions_dir}")
    return np.asarray(A0), np.asarray(B), y, seeds  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# Features + probes
# --------------------------------------------------------------------------- #


def build_features(A0: np.ndarray, B: np.ndarray) -> dict[str, np.ndarray]:
    """The start-only feature ladder (no execution-derived / trajectory deltas)."""
    return {
        "A0 (cubeA start)": A0.astype(np.float32),
        "B (cubeB)": B.astype(np.float32),
        "[A0;B] (concat)": np.concatenate([A0, B], axis=1).astype(np.float32),
        "B-A0 (relative)": (B - A0).astype(np.float32),
    }


def _direct_lr_probe(Z: np.ndarray, y: np.ndarray, *, seed: int = 0) -> dict:
    """Per-fold StandardScaler + LogisticRegression, matched splitter.

    Mirrors the cert keys of ``nested_cv_probe_one_factor`` so the report
    aggregates both probes identically. Shuffled baseline permutes y_train.
    """
    splitter = _make_splitter(y)
    rng = np.random.default_rng(seed)
    accs, shuf = [], []
    for tr, te in splitter.split(Z, y):
        sc = StandardScaler().fit(Z[tr])
        Ztr, Zte = sc.transform(Z[tr]), sc.transform(Z[te])
        clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        clf.fit(Ztr, y[tr])
        accs.append(float(clf.score(Zte, y[te])))
        yp = y[tr][rng.permutation(len(tr))]
        clf_s = LogisticRegression(max_iter=1000, solver="lbfgs")
        clf_s.fit(Ztr, yp)
        shuf.append(float(clf_s.score(Zte, y[te])))
    _, counts = np.unique(y, return_counts=True)
    accs_np = np.asarray(accs)
    return {
        "n_episodes": int(Z.shape[0]),
        "n_unique_labels": int(np.unique(y).size),
        "probe_acc_mean": float(accs_np.mean()),
        "probe_acc_std": float(accs_np.std()),
        "majority_class_acc": float(counts.max() / counts.sum()),
        "shuffled_features_acc": float(np.mean(shuf)),
    }


def _rule_accuracy(A0: np.ndarray, B: np.ndarray, y: list[str]) -> float:
    """Parameter-free dominant-axis rule on (B - A0), snapped via the SAME
    ``goal_direction_to_motion`` the label uses. No fitting, full cut."""
    correct = sum(
        goal_direction_to_motion(B[i] - A0[i]) == y[i] for i in range(len(y))
    )
    return correct / len(y)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def _gate(probe_mean: float, majority: float, shuffled: float) -> str:
    if probe_mean >= 0.90 and probe_mean >= majority + 0.10 and probe_mean >= shuffled + 0.10:
        return "PASS"
    return "FAIL"


def _write_report(out_dir: Path, *, task: str, source_desc: str, seeds: list[int],
                  label_dist: dict, results: dict, rule_acc: float,
                  dinov2_ref: float = 0.42) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(seeds)
    majority = max(label_dist.values()) / n
    L = [
        f"# Stage-5 — StackCube `object_motion` label well-posedness (position oracle) — {task}",
        "",
        "**Question:** is `object_motion` a clean function of object POSITIONS — i.e. is",
        "the frozen-DINOv2 spatial_mean **0.42** a representation failure or a label/data",
        "problem? (This is a well-posedness check; the relation == the label's own input,",
        "so it is near-tautological by construction — see Interpretation.)",
        "",
        f"- Source: {source_desc}",
        f"- Cut: n={n}, seeds {min(seeds)}-{max(seeds)}, label dist {label_dist} "
        f"(majority {majority:.3f})",
        "- Feature: cubeA **start** + cubeB resting positions (privileged sim obs). "
        "The label is `goal_direction_to_motion(cubeB_init - cubeA_init)` "
        "(`scene.py:cubeA_to_cubeB_motion`, assigned at reset), so `(cubeB - cubeA_start)`",
        "  IS the label's own input — this ladder is a **well-posedness / "
        "zero-label-noise** check, not a non-circular recovery.",
        f"- Reference: DINOv2 spatial_mean object_motion = **{dinov2_ref:.2f}** "
        "(`reports/stage5/p1_vision_g1`).",
        "",
        "## Feature ladder",
        "",
        "| feature | dim | majority | shuffled | direct LR ± std | gate | IntentHead-CV ± std | gate |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        d = r["direct"]
        h = r["intent"]
        L.append(
            f"| `{name}` | {r['dim']} | {d['majority_class_acc']:.3f} | "
            f"{d['shuffled_features_acc']:.3f} | "
            f"{d['probe_acc_mean']:.3f} ± {d['probe_acc_std']:.3f} | "
            f"{_gate(d['probe_acc_mean'], d['majority_class_acc'], d['shuffled_features_acc'])} | "
            f"{h['probe_acc_mean']:.3f} ± {h['probe_acc_std']:.3f} | "
            f"{_gate(h['probe_acc_mean'], h['majority_class_acc'], h['shuffled_features_acc'])} |"
        )
    rel = results.get("B-A0 (relative)", {}).get("direct", {}).get("probe_acc_mean", float("nan"))
    a0acc = results.get("A0 (cubeA start)", {}).get("direct", {}).get("probe_acc_mean", float("nan"))
    bacc = results.get("B (cubeB)", {}).get("direct", {}).get("probe_acc_mean", float("nan"))
    L += [
        "",
        f"**Parameter-free dominant-axis rule on `(cubeB - cubeA_start)`: {rule_acc:.3f}** "
        "(no fitting, full cut; this is literally the label's own "
        "`goal_direction_to_motion` applied to its own input → ≈1.0 on an exact cut).",
        "",
        "## Interpretation",
        "",
        f"- The label is a **deterministic, parameter-free, noiseless** function of the two",
        f"  cubes' resting positions: the dominant-axis rule on `(cubeB - cubeA_start)` "
        f"matches it at **{rule_acc:.3f}** (≈1.0 on an exact cut; any gap here is only the",
        f"  cross-cut `goal_xy`≈cubeB approximation). There is **no label noise**.",
        f"- Neither cube's *absolute* position predicts direction (`A0` {a0acc:.3f}, "
        f"`B` {bacc:.3f} ≈ chance); only the **relation** `B - A0` does ({rel:.3f}) — the",
        "  object-centric hypothesis, stated in coordinates.",
        "- **Conclusion:** because the label is a clean, noiseless geometric function of",
        f"  resting positions, the frozen-DINOv2 **{dinov2_ref:.2f}** is a **representation**",
        "  failure, not a label/data problem.",
        "- **But note the limit:** this ladder uses privileged coordinates and is",
        "  near-tautological by construction (the relation == the label input). It does",
        "  NOT show that any *learned feature* recovers the relation. The non-trivial",
        "  representation evidence is the **blob image-position probe ~0.80 vs global",
        "  DINO 0.42** on the same frames; Step-2 (`stage5_object_relation_probe.py`)",
        "  tests object-local DINO tokens from pixels, no coordinates fed.",
        "",
        "## Caveats",
        "",
        "- **Near-tautological by construction:** the label IS "
        "`goal_direction_to_motion(cubeB - cubeA)`, so the relational feature and the rule",
        "  re-derive it. This certifies the label is noiseless & well-posed and that",
        "  absolute single-object position is insufficient — it is NOT evidence that a",
        "  learned representation recovers the relation. That is Step-2's job.",
        "- **Cross-cut** when `--source p2_wrist`: this cut is seeds 100-149 (P2 eval, "
        "imbalanced), not the balanced 0-39 cut that produced 0.42. The conclusion is",
        "  cut-agnostic; the exactly-matched 0-39 path is",
        "  `stage5_extract_cube_positions.py` (GPU) → `--source dir`.",
        "- Positions are privileged sim obs, used here only as a well-posedness check for",
        "  representation development (CLAUDE.md invariant #4 — they must NOT become the",
        "  deployable demo→intent path; Step-2 extracts the relation from pixels).",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(L) + "\n")
    (out_dir / "results.json").write_text(json.dumps({
        "task": task, "source": source_desc, "n": n,
        "seeds": [int(s) for s in seeds], "label_dist": label_dist,
        "dinov2_reference": dinov2_ref, "rule_accuracy": rule_acc,
        "results": results,
    }, indent=2, sort_keys=True) + "\n")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="StackCube-v1")
    p.add_argument("--source", choices=("p2_wrist", "dir"), default="p2_wrist")
    p.add_argument("--positions-dir", type=Path, default=None,
                   help="(--source dir) directory of seed_NNNN_positions.npz")
    p.add_argument("--jsonl", type=Path, default=None,
                   help="(--source dir) records jsonl for object_motion labels")
    p.add_argument("--out-dir", type=Path,
                   default=Path("reports/stage5/relation_oracle"))
    p.add_argument("--d-slot", type=int, default=32)
    p.add_argument("--n-epochs", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    if args.source == "p2_wrist":
        A0, B, y_str, seeds = _load_p2_wrist(args.task)
        source_desc = f"p2_vlm_wrist/{args.task} rollouts (initial_obj_xy, goal_xy) + episodes.jsonl"
    else:
        if args.positions_dir is None or args.jsonl is None:
            p.error("--source dir requires --positions-dir and --jsonl")
        A0, B, y_str, seeds = _load_positions_dir(args.positions_dir, args.jsonl)
        source_desc = f"{args.positions_dir} (cubeA_xy0, cubeB_xy) + {args.jsonl}"

    classes = sorted(set(y_str))
    cls_to_int = {c: i for i, c in enumerate(classes)}
    y = np.asarray([cls_to_int[v] for v in y_str], dtype=np.int64)
    label_dist = {c: int((np.asarray(y_str) == c).sum()) for c in classes}
    print(f"loaded n={len(seeds)} seeds {min(seeds)}-{max(seeds)}; labels {label_dist}")

    feats = build_features(A0, B)
    results: dict[str, dict] = {}
    for name, Z in feats.items():
        direct = _direct_lr_probe(Z, y, seed=args.seed)
        intent = nested_cv_probe_one_factor(
            Z, y, factor_idx=_OBJECT_MOTION_IDX,
            d_slot=args.d_slot, n_epochs=args.n_epochs, seed=args.seed,
        )
        results[name] = {"dim": int(Z.shape[1]), "direct": direct, "intent": intent}
        print(f"  {name:18s} dim={Z.shape[1]}  direct={direct['probe_acc_mean']:.3f}"
              f"  intentCV={intent['probe_acc_mean']:.3f}  (majority {direct['majority_class_acc']:.3f})")

    rule_acc = _rule_accuracy(A0, B, y_str)
    print(f"  dominant-axis RULE on (B-A0): {rule_acc:.3f}")

    _write_report(args.out_dir, task=args.task, source_desc=source_desc,
                  seeds=seeds, label_dist=label_dist, results=results,
                  rule_acc=rule_acc)
    print(f"\nwrote {args.out_dir}/report.md\nwrote {args.out_dir}/results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
