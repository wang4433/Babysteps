"""Stage-5 — train the shared, task-general RevisionPolicy (build-order step 2,
trained at the step-5 GPU/data stage).

ONE SharedScorer checkpoint over POOLED ``--dump-tuples`` from both natural
loops (PushCube residual-choice tuples + StackCube goal_state coverage tuples).
Candidate-wise cross-entropy: the target is the index of the oracle-correct
value within ``candidates_for(factor)``. The candidate set is driven from the
SAME source the evaluator uses, so train/eval candidates never desync.

g_i source (open decision; see redesign_failure_paradigm.md): default is the
deterministic per-class CENTROID proxy from the relevant per-task pack (a pure
function of the token — login-node runnable, no features) via ``--gi centroid``;
``--gi none`` trains a g_i-free scorer. The real vision-decoded-slot source is a
step-5 add-on. Honest scope: StackCube goal_state is included as pipeline
COVERAGE but excluded from any learned-choice metric (its single valid
transition stays the deterministic ``goal_refinement`` operator).

The pure helpers (``normalize_tuple``, ``build_training_rows``,
``train_shared_scorer``) are sim-free and unit-tested; pack/feature loading is
deferred to ``main`` call time so this module imports on the login node.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from babysteps.schemas import INTENT_FIELDS  # noqa: E402
from babysteps.stage5.revision_policy import candidates_for  # noqa: E402
from babysteps.stage5.shared_revision_policy import (  # noqa: E402
    FACTOR_ORDER, SharedScorer, _efail_vec, _gi_vec, build_value_vocab,
    save_shared_scorer,
)

_FIDX = {f: i for i, f in enumerate(FACTOR_ORDER)}


def normalize_tuple(t: dict) -> dict:
    """Normalize a dumped tuple (PushCube residual schema OR StackCube goal_state
    schema) into a common training record. Raises on an unrecognized schema."""
    if "demo_face" in t:  # PushCube natural-loop residual tuple
        return {
            "task": "PushCube-v1", "factor": "contact_region",
            "current": t["demo_face"], "correct": t["correct_face"],
            "predicate": t.get("failure_predicate"),
            "residual_xy": t.get("residual_xy"),
            "candidates": list(candidates_for("PushCube-v1", "contact_region")),
        }
    if "current_value" in t and "factor" in t:  # generic (StackCube) tuple
        task = t.get("task", "")
        return {
            "task": task, "factor": t["factor"],
            "current": t["current_value"], "correct": t["correct_value"],
            "predicate": t.get("failure_predicate"),
            "residual_xy": t.get("residual_xy"),
            "candidates": list(t.get("candidates")
                               or candidates_for(task, t["factor"])),
        }
    raise ValueError(f"unrecognized tuple schema: keys={sorted(t)}")


def build_training_rows(
    tuples: list[dict],
    gi_lookup: Optional[Callable[[str, str, str], Optional[np.ndarray]]] = None,
) -> list[dict]:
    """Normalize tuples → training rows, attaching g_i via ``gi_lookup(task,
    factor, current)`` (None → g_i-free). Drops rows whose correct value is not
    among the candidates (cannot be a CE target)."""
    rows = []
    for t in tuples:
        r = normalize_tuple(t)
        if r["correct"] not in r["candidates"]:
            continue
        r["gi"] = (gi_lookup(r["task"], r["factor"], r["current"])
                   if gi_lookup is not None else None)
        rows.append(r)
    return rows


def _fit_scaler(rows: list[dict]) -> Optional[dict]:
    gis = [np.asarray(r["gi"], dtype=np.float32) for r in rows
           if r.get("gi") is not None]
    if not gis:
        return None
    X = np.stack(gis)
    return {"mean": X.mean(0), "scale": X.std(0) + 1e-6}


def train_shared_scorer(rows: list[dict], value_vocab: dict, *, d_gi: int,
                        epochs: int = 400, lr: float = 1e-2, wd: float = 1e-4,
                        seed: int = 0, scaler: Optional[dict] = None):
    """Candidate-wise CE training of a SharedScorer over pooled rows."""
    torch.manual_seed(seed)
    scorer = SharedScorer(vocab_size=len(value_vocab) + 1, d_gi=d_gi, seed=seed)
    opt = torch.optim.Adam(scorer.parameters(), lr=lr, weight_decay=wd)
    scorer.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.zeros(())
        for r in rows:
            cands = r["candidates"]
            in_vocab = [c for c in cands if (r["factor"], c) in value_vocab]
            if r["correct"] not in in_vocab:
                continue
            b = len(in_vocab)
            gi = torch.tensor(_gi_vec(r.get("gi"), scaler, dim=d_gi),
                              dtype=torch.float32).expand(b, -1)
            ef = torch.tensor(_efail_vec(r["predicate"], r["residual_xy"]),
                              dtype=torch.float32).expand(b, -1)
            fidx = torch.full((b,), _FIDX[r["factor"]], dtype=torch.long)
            cur = torch.full((b,), value_vocab.get((r["factor"], r["current"]),
                                                   0), dtype=torch.long)
            cand = torch.tensor([value_vocab[(r["factor"], c)] for c in in_vocab],
                                dtype=torch.long)
            logits = scorer(gi, ef, fidx, cur, cand)
            target = torch.tensor(in_vocab.index(r["correct"]), dtype=torch.long)
            loss = loss + F.cross_entropy(logits.unsqueeze(0),
                                          target.unsqueeze(0))
        if loss.requires_grad:  # at least one CE term was added this epoch
            loss.backward()
            opt.step()
    scorer.eval()
    return scorer


def _centroid_gi_lookup(pack_by_task: dict):
    """g_i = per-class centroid (deterministic proxy) from the relevant pack."""
    def _lookup(task, factor, current):
        pack = pack_by_task.get(task)
        if pack is None:
            return None
        fi = INTENT_FIELDS.index(factor)
        if fi not in pack.centroids:
            return None
        toks = pack.label_tokens[fi]
        if current not in toks:
            return None
        return np.asarray(pack.centroids[fi][toks.index(current)],
                          dtype=np.float32)
    return _lookup


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tuples", type=Path, nargs="+", required=True,
                   help="One or more dumped --dump-tuples jsonl files (pooled).")
    p.add_argument("--out", type=Path,
                   default=Path("models/stage5/shared/scorer.pt"))
    p.add_argument("--gi", choices=["centroid", "none"], default="centroid",
                   help="g_i source: deterministic pack centroid proxy, or none.")
    p.add_argument("--pack", action="append", default=[],
                   help="TASK=PACK_DIR for the centroid g_i lookup (repeatable).")
    p.add_argument("--d-gi", type=int, default=33,
                   help="g_i input dim (d_slot+1); 33 for the d_slot=32 packs.")
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    args = p.parse_args(argv)

    tuples: list[dict] = []
    for f in args.tuples:
        tuples.extend(_read_jsonl(f))
    print(f"loaded {len(tuples)} pooled tuples from {len(args.tuples)} file(s)")

    gi_lookup = None
    if args.gi == "centroid":
        from babysteps.stage4.latent_policy import load_latent_pack
        pack_by_task = {}
        for spec in args.pack:
            task, pd = spec.split("=", 1)
            pack_by_task[task] = load_latent_pack(pd)
        if not pack_by_task:
            p.error("--gi centroid requires at least one --pack TASK=DIR")
        gi_lookup = _centroid_gi_lookup(pack_by_task)

    vocab = build_value_vocab()
    rows = build_training_rows(tuples, gi_lookup)
    print(f"built {len(rows)} training rows "
          f"({sum(r['task'] == 'PushCube-v1' for r in rows)} PushCube, "
          f"{sum(r['task'] == 'StackCube-v1' for r in rows)} StackCube)")
    scaler = _fit_scaler(rows) if args.gi == "centroid" else None
    scorer = train_shared_scorer(rows, vocab, d_gi=args.d_gi, epochs=args.epochs,
                                 lr=args.lr, wd=args.weight_decay, scaler=scaler)

    # PushCube learned-choice train accuracy (StackCube excluded — no choice).
    push = [r for r in rows if r["task"] == "PushCube-v1"]
    if push:
        from babysteps.stage5.shared_revision_policy import SharedScorerPolicy
        from babysteps.stage5.revision_policy import RevisionRequest, FailureEvidence
        pol = SharedScorerPolicy(scorer, vocab, scaler=scaler)
        ok = 0
        for r in push:
            dec = pol.decide(RevisionRequest(
                factor=r["factor"], current_value=r["current"],
                candidates=tuple(r["candidates"]),
                e_fail=FailureEvidence(r["predicate"], r["residual_xy"]),
                g_i=r.get("gi")))
            if dec.new_value == r["correct"]:
                ok += 1
        print(f"PushCube learned-choice train acc = {ok/len(push):.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_shared_scorer(scorer, vocab, args.out, scaler=scaler)
    print(f"saved shared scorer to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
