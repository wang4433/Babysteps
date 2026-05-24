"""Stage-4 M2a Stage A2 — joint train + ReviseHead + G2 cert + decode acc.

Per task, per outer fold:
  1. Joint-train IntentHead on TRAIN fold across all non-trivial factors.
  2. Build per-factor centroids on TRAIN G.
  3. Train ReviseHead on TRAIN counterfactual pairs
     (g_pre = G[wrong_factor], fp_vec, target = centroid[revised_class]).
  4. Held-out eval on TEST fold:
     a. Decode ReviseHead output via nearest centroid → check it matches
        revision.new_value (revised-slot decode accuracy).
     b. Verify other slots bit-identical after apply_revision (G2-mechanical).

Cert thresholds reported (not yet enforced as M2a entrance gates):
  G2 ℓ2 drift: must be 0.0 by construction (deterministic encoder).
  Revised-slot decode acc: report mean ± std across folds; the
  threshold can be calibrated against Stage-0 attribution accuracy
  on the same cut once user reviews this report.

Sim-free, CPU-only. ~30s wall-clock end-to-end.

Example::

    python scripts/stage4_m2a_a2_eval.py \\
        --jsonl datasets/stage4/varied_intent/PushCube-v1/samples.jsonl \\
        --jsonl datasets/stage4/varied_intent/StackCube-v1/samples.jsonl \\
        --out-dir reports/stage4/m2a_a2/ --seed 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.preprocessing import LabelEncoder

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from babysteps.schemas import INTENT_FIELDS, EpisodeRecord  # noqa: E402
from babysteps.stage4.features import extract_episode_features  # noqa: E402
from babysteps.stage4.intent_head import (  # noqa: E402
    IntentHead, train_intent_head_joint,
)
from babysteps.stage4.revise_head import (  # noqa: E402
    FP_VECTOR_DIM, ReviseHead, apply_revision,
    train_revise_head_l2, vectorize_failure_packet,
)
from babysteps.stage4.slot_decode import (  # noqa: E402
    build_factor_centroids, decode_slot,
)

_PRIMARY = [
    _ROOT / "datasets/stage4/varied_intent" / task / "samples.jsonl"
    for task in ("PushCube-v1", "StackCube-v1")
]


def _load_records(paths: list[Path]) -> list[dict]:
    records: list[dict] = []
    for path in paths:
        with Path(path).open() as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(EpisodeRecord.from_jsonl_line(line).to_dict())
    return records


def _make_splitter(n_episodes: int):
    """5-fold KFold; LOO when n is small enough that 5-fold leaves <2 train."""
    if n_episodes < 10:
        return LeaveOneOut()
    return KFold(n_splits=5, shuffle=True, random_state=0)


def _prepare_task(records: list[dict]):
    """Returns:
        Z (B, 20) float32
        labels_per_factor {factor_idx: (y, n_classes)} for non-trivial factors
        encoders {factor_idx: LabelEncoder}
        revisions list of dicts {factor_idx, new_class_int, fp_vec, factor_name}
    """
    Z = np.stack([extract_episode_features(r) for r in records]).astype(np.float32)
    # Initial-intent labels per factor for IntentHead supervision
    encoders: dict[int, LabelEncoder] = {}
    labels_per_factor: dict[int, tuple[np.ndarray, int]] = {}
    for fi, factor in enumerate(INTENT_FIELDS):
        labels_str = [r["execution"]["initial_intent"][factor] for r in records]
        if len(set(labels_str)) < 2:
            continue
        enc = LabelEncoder().fit(labels_str)
        labels_per_factor[fi] = (enc.transform(labels_str).astype(np.int64),
                                  len(enc.classes_))
        encoders[fi] = enc
    # Revision pairs: one per episode (varied cut always has a revision)
    revisions: list[dict] = []
    for i, r in enumerate(records):
        rv = r.get("revision")
        if rv is None:
            continue
        factor = rv["factor"]
        if factor not in INTENT_FIELDS:
            continue
        fi = INTENT_FIELDS.index(factor)
        # Make sure the revised factor was supervised; otherwise we have no
        # centroid space to target (would be skipped during eval).
        if fi not in encoders:
            # If the revision factor is "trivially-constant" in initial-intent
            # but appears as a revision target, register it now from the
            # revision values themselves (rare; happens when the initial
            # value is one constant and revisions all flip to another).
            extra_vals = sorted({rv["old_value"], rv["new_value"]})
            enc = LabelEncoder().fit(extra_vals)
            encoders[fi] = enc
            # Add a constant-old-value label vector so joint-train will
            # also supervise this slot.
            y = enc.transform([rv["old_value"]] * len(records)).astype(np.int64)
            labels_per_factor[fi] = (y, len(enc.classes_))
        enc = encoders[fi]
        try:
            new_class = int(enc.transform([rv["new_value"]])[0])
        except ValueError:
            # revised value not in encoder — extend (additive-schema rule)
            old_classes = list(enc.classes_)
            new_classes = sorted(set(old_classes) | {rv["new_value"]})
            enc = LabelEncoder().fit(new_classes)
            encoders[fi] = enc
            # Rebuild the labels for this factor in the new encoder space
            labels_str = [
                records[j]["execution"]["initial_intent"][INTENT_FIELDS[fi]]
                for j in range(len(records))
            ]
            labels_per_factor[fi] = (enc.transform(labels_str).astype(np.int64),
                                      len(enc.classes_))
            new_class = int(enc.transform([rv["new_value"]])[0])
        revisions.append({
            "episode_idx": i,
            "factor_idx": fi,
            "new_class_int": new_class,
            "fp_vec": vectorize_failure_packet(r),
            "factor_name": factor,
        })
    return Z, labels_per_factor, encoders, revisions


def _train_fold(Z_tr, labels_per_factor_tr, revisions_tr, *,
                d_slot, hidden, n_epochs_intent, n_epochs_revise, lr, seed):
    """Train IntentHead jointly, build centroids, train ReviseHead. Returns
    (head, centroids, revise_head)."""
    head = IntentHead(
        z_dim=Z_tr.shape[1], n_factors=len(INTENT_FIELDS),
        d_slot=d_slot, hidden=hidden, seed=seed,
    )
    train_intent_head_joint(
        head, Z_tr, labels_per_factor_tr,
        n_epochs=n_epochs_intent, lr=lr, seed=seed,
    )
    head.eval()
    with torch.no_grad():
        G_tr = head(torch.from_numpy(Z_tr)).numpy()
    centroids = build_factor_centroids(
        G_tr, {fi: y for fi, (y, _) in labels_per_factor_tr.items()},
    )
    # Build ReviseHead training pairs (use local-in-train index).
    # Drop "uncertable" revisions whose target class has no centroid: that
    # happens when the initial-intent space never contained that class
    # (e.g. StackCube approach_direction is always "from_above" in initial
    # intents but revisions flip to "from_minus_x" — the encoder never
    # produced a "from_minus_x" slot region, so the centroid is absent).
    # Skipping is the honest accounting; the limitation is surfaced in the
    # report's "uncertable" count.
    certable_tr = [
        rv for rv in revisions_tr
        if rv["factor_idx"] in centroids
        and rv["new_class_int"] in centroids[rv["factor_idx"]]
    ]
    if certable_tr:
        g_pre = np.stack([G_tr[rv["tr_local_idx"], rv["factor_idx"]]
                          for rv in certable_tr]).astype(np.float32)
        fp = np.stack([rv["fp_vec"] for rv in certable_tr]).astype(np.float32)
        g_tgt = np.stack([
            centroids[rv["factor_idx"]][rv["new_class_int"]]
            for rv in certable_tr
        ]).astype(np.float32)
        revise = ReviseHead(d_slot=d_slot, hidden=hidden, seed=seed)
        train_revise_head_l2(
            revise, g_pre, fp, g_tgt,
            n_epochs=n_epochs_revise, lr=lr, seed=seed,
        )
    else:
        revise = ReviseHead(d_slot=d_slot, hidden=hidden, seed=seed)
    return head, centroids, revise


def _eval_fold(Z_te, revisions_te, head, centroids, revise):
    """For each test episode with a revision: apply_revision → decode →
    compare to revision.new_value. Also verify G2 (other slots unchanged).
    Returns (decode_correct, decode_total, max_other_slot_drift)."""
    if not revisions_te:
        return {"decode_correct": 0, "decode_certable_total": 0,
                "n_uncertable": 0, "max_other_slot_drift_l2": 0.0}
    head.eval()
    with torch.no_grad():
        G_te = head(torch.from_numpy(Z_te))  # (B, F, d_slot)
    correct = 0
    certable_total = 0
    uncertable = 0
    max_drift = 0.0
    for rv in revisions_te:
        fi = rv["factor_idx"]
        # Uncertable: no centroid for the target class (see _train_fold's
        # comment). Count it for transparency; do not include in decode_acc.
        if fi not in centroids or rv["new_class_int"] not in centroids[fi]:
            uncertable += 1
            continue
        idx_in_te = rv["te_local_idx"]
        g_single = G_te[idx_in_te:idx_in_te + 1]
        fp_t = torch.from_numpy(rv["fp_vec"]).unsqueeze(0)
        G_revised = apply_revision(g_single, fi, fp_t, revise)
        # G2 mechanical: drift on other slots
        for j in range(g_single.shape[1]):
            if j == fi:
                continue
            drift = float(torch.norm(G_revised[0, j] - g_single[0, j]).item())
            if drift > max_drift:
                max_drift = drift
        revised_slot_np = G_revised[0, fi].numpy()
        decoded = decode_slot(revised_slot_np, centroids[fi])
        if decoded == rv["new_class_int"]:
            correct += 1
        certable_total += 1
    return {
        "decode_correct": correct,
        "decode_certable_total": certable_total,
        "n_uncertable": uncertable,
        "max_other_slot_drift_l2": max_drift,
    }


def _eval_task(records, *, d_slot, hidden, n_epochs_intent, n_epochs_revise,
               lr, seed):
    """Full per-task A2 eval: per-fold train + eval, aggregate."""
    Z, labels_per_factor, encoders, revisions = _prepare_task(records)
    n = len(records)
    splitter = _make_splitter(n)
    fold_results = []
    for fold_i, (tr_idx, te_idx) in enumerate(splitter.split(range(n))):
        # Split labels_per_factor by indices
        labels_tr = {fi: (y[tr_idx], n_cls)
                     for fi, (y, n_cls) in labels_per_factor.items()}
        # Revisions for train vs test (with local indices into G_tr/G_te)
        tr_local = {int(g): i for i, g in enumerate(tr_idx)}
        te_local = {int(g): i for i, g in enumerate(te_idx)}
        revisions_tr = []
        for rv in revisions:
            if rv["episode_idx"] in tr_local:
                rv_tr = dict(rv)
                rv_tr["tr_local_idx"] = tr_local[rv["episode_idx"]]
                revisions_tr.append(rv_tr)
        revisions_te = []
        for rv in revisions:
            if rv["episode_idx"] in te_local:
                rv_te = dict(rv)
                rv_te["te_local_idx"] = te_local[rv["episode_idx"]]
                revisions_te.append(rv_te)
        head, centroids, revise = _train_fold(
            Z[tr_idx], labels_tr, revisions_tr,
            d_slot=d_slot, hidden=hidden,
            n_epochs_intent=n_epochs_intent, n_epochs_revise=n_epochs_revise,
            lr=lr, seed=seed + fold_i,
        )
        fold_eval = _eval_fold(
            Z[te_idx], revisions_te, head, centroids, revise,
        )
        fold_results.append({
            "fold": fold_i,
            "n_train": int(len(tr_idx)),
            "n_test": int(len(te_idx)),
            **fold_eval,
        })
    return {
        "n_episodes": n,
        "n_revisions": len(revisions),
        "factor_counts": dict({INTENT_FIELDS[fi]: int(n_cls)
                               for fi, (_, n_cls) in labels_per_factor.items()}),
        "revision_factor_counts": {
            k: sum(1 for rv in revisions if rv["factor_name"] == k)
            for k in set(rv["factor_name"] for rv in revisions)
        },
        "folds": fold_results,
    }


def _aggregate(task_results: dict) -> dict:
    folds = task_results["folds"]
    correct = sum(f["decode_correct"] for f in folds)
    certable_total = sum(f["decode_certable_total"] for f in folds)
    uncertable_total = sum(f["n_uncertable"] for f in folds)
    decode_acc = (correct / certable_total) if certable_total else float("nan")
    max_drift = max(f["max_other_slot_drift_l2"] for f in folds) if folds else 0.0
    return {
        "decode_accuracy_certable": decode_acc,
        "n_revisions_certable": certable_total,
        "n_revisions_uncertable": uncertable_total,
        "max_other_slot_drift_l2": max_drift,
        "g2_passes": max_drift == 0.0,
    }


def _render_markdown(report: dict, *, d_slot, n_epochs_intent,
                     n_epochs_revise, lr) -> str:
    lines = [
        "# Stage-4 M2a Stage A2 — Joint IntentHead + ReviseHead",
        "",
        ("Per task: joint IntentHead training across all non-trivial "
         "intent factors; per-factor centroid `slot_decode`; "
         "single-slot ReviseHead trained on counterfactual "
         "(g_pre, fp) → centroid[revised_class] pairs."),
        "",
        (f"IntentHead: F=6, d_slot={d_slot}, hidden=64, "
         f"n_epochs={n_epochs_intent}, lr={lr}."),
        (f"ReviseHead: d_slot={d_slot}, fp_dim={FP_VECTOR_DIM}, "
         f"hidden=64, n_epochs={n_epochs_revise}, L2 loss."),
        "",
        ("Cert metrics:"),
        ("- **G2 (frozen-slot preservation)**: max ℓ2 drift of "
         "unedited slots after `apply_revision`. Spec gate: ≤ ε. "
         "Deterministic encoder → ε = 0 by construction (`apply_revision` "
         "only writes the implicated slot)."),
        ("- **Revised-slot decode acc**: on held-out test folds, does "
         "`decode_slot(apply_revision(G, factor_idx, fp)[slot])` match "
         "the ground-truth `revision.new_value`? This is the more "
         "interesting number — measures whether ReviseHead actually "
         "moves the slot to the right centroid, not just that the type "
         "signature works."),
        "",
        "## Per-task headline",
        "",
        ("| task | n_episodes | n_revisions | certable | uncertable "
         "| revised-slot decode acc | G2 max drift | G2 |"),
        ("| --- | --- | --- | --- | --- | --- | --- | --- |"),
    ]
    for task in sorted(report["per_task"]):
        agg = report["per_task"][task]["aggregate"]
        cert = agg["n_revisions_certable"]
        unc = agg["n_revisions_uncertable"]
        decode_str = (
            f"{agg['decode_accuracy_certable']:.3f}" if cert else "n/a (0 certable)"
        )
        lines.append(
            f"| {task} | {report['per_task'][task]['n_episodes']} "
            f"| {report['per_task'][task]['n_revisions']} "
            f"| {cert} | {unc} "
            f"| {decode_str} "
            f"| {agg['max_other_slot_drift_l2']:.2e} "
            f"| {'PASS' if agg['g2_passes'] else 'FAIL'} |"
        )
    lines.append("")
    lines.append("## Revision factor distribution (per task)")
    lines.append("")
    for task in sorted(report["per_task"]):
        rfc = report["per_task"][task]["revision_factor_counts"]
        lines.append(f"- **{task}**: {rfc}")
    lines.append("")
    lines.append("## Per-fold detail (machine-readable in JSON)")
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Stage-4 M2a A2 eval.")
    p.add_argument("--jsonl", action="append", type=Path,
                   help="EpisodeRecord JSONL path (repeatable). "
                        "Default = the two primary varied-intent cuts.")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--d-slot", type=int, default=16)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--n-epochs-intent", type=int, default=200)
    p.add_argument("--n-epochs-revise", type=int, default=400)
    p.add_argument("--lr", type=float, default=1e-2)
    args = p.parse_args(argv)

    paths = args.jsonl if args.jsonl else _PRIMARY
    records = _load_records(paths)
    by_task: dict[str, list[dict]] = {}
    for r in records:
        by_task.setdefault(r["task"], []).append(r)

    per_task: dict[str, dict] = {}
    for task in sorted(by_task):
        print(f"\n=== Evaluating {task} (n={len(by_task[task])}) ===")
        tr = _eval_task(
            by_task[task],
            d_slot=args.d_slot, hidden=args.hidden,
            n_epochs_intent=args.n_epochs_intent,
            n_epochs_revise=args.n_epochs_revise,
            lr=args.lr, seed=args.seed,
        )
        agg = _aggregate(tr)
        tr["aggregate"] = agg
        per_task[task] = tr
        cert = agg["n_revisions_certable"]
        unc = agg["n_revisions_uncertable"]
        if cert:
            print(f"  decode_acc: {agg['decode_accuracy_certable']:.3f} "
                  f"on {cert} certable revisions ({unc} uncertable)")
        else:
            print(f"  decode_acc: n/a — 0 certable revisions "
                  f"({unc} uncertable); see notes for diagnosis")
        print(f"  G2 max drift: {agg['max_other_slot_drift_l2']:.2e} "
              f"({'PASS' if agg['g2_passes'] else 'FAIL'})")

    report = {
        "config": {
            "seed": args.seed, "d_slot": args.d_slot, "hidden": args.hidden,
            "n_epochs_intent": args.n_epochs_intent,
            "n_epochs_revise": args.n_epochs_revise, "lr": args.lr,
        },
        "per_task": per_task,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "a2_results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (args.out_dir / "a2_results.md").write_text(
        _render_markdown(
            report, d_slot=args.d_slot,
            n_epochs_intent=args.n_epochs_intent,
            n_epochs_revise=args.n_epochs_revise, lr=args.lr,
        ) + "\n"
    )
    print(f"\nwrote {args.out_dir}/a2_results.{{json,md}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
