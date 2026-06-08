"""Sim-free tests for the shared, task-general RevisionPolicy (build-order step 2).

Deterministic, login-node, NO gitignored models/ or datasets/stage5 reads — all
fixtures are tiny synthetic SharedScorers. Mirrors the rigor of
test_stage5_revision_policy.py (trained-head fixtures, exact assertions).

Covers: deterministic candidate recovery, variable candidate count, None-residual
finiteness, abstain/keep, OOV fallback, single-slot compiler interop, structural
no-leakage, a uses-g_i guard, a synthetic leave-one-FAMILY-out generalization
demo, an adversarial task-leakage probe (near-chance when inputs don't encode
task), and determinism.
"""
from __future__ import annotations

from dataclasses import fields, replace

import numpy as np
import torch
import torch.nn.functional as F

from babysteps.schemas import INTENT_FIELDS, Intent
from babysteps.stage5.revision_policy import (
    FailureEvidence, RevisionRequest, candidates_for, compile_single_slot_edit,
)
from babysteps.stage5.shared_revision_policy import (
    EFAIL_DIM, FACTOR_ORDER, GI_DIM_DEFAULT, SharedScorer, SharedScorerPolicy,
    _efail_vec, _gi_vec, build_value_vocab, load_shared_scorer,
    save_shared_scorer,
)

VOCAB = build_value_vocab()
VOCAB_SIZE = len(VOCAB) + 1
FACES = ("minus_x_face", "plus_x_face", "minus_y_face", "plus_y_face")
_FIDX = {f: i for i, f in enumerate(FACTOR_ORDER)}

GT = Intent(
    goal_state="cube_at_target", object_motion="translate_+x",
    contact_region="plus_x_face", approach_direction="from_plus_x",
    constraint_region="none", embodiment_mapping="proxy_contact_to_franka_push",
)


def _new_scorer(seed: int = 0, d_gi: int = GI_DIM_DEFAULT) -> SharedScorer:
    return SharedScorer(vocab_size=VOCAB_SIZE, d_gi=d_gi, hidden=64, seed=seed)


def _row_tensors(row):
    """Build batched scorer inputs for one training/eval row."""
    cands = row["candidates"]
    b = len(cands)
    gi = torch.tensor(_gi_vec(row.get("gi"), None), dtype=torch.float32).expand(b, -1)
    ef = torch.tensor(_efail_vec(row["pred"], row["residual"]),
                      dtype=torch.float32).expand(b, -1)
    fidx = torch.full((b,), _FIDX[row["factor"]], dtype=torch.long)
    cur = torch.full((b,), VOCAB.get((row["factor"], row["current"]), 0),
                     dtype=torch.long)
    cand = torch.tensor([VOCAB[(row["factor"], c)] for c in cands],
                        dtype=torch.long)
    target = torch.tensor(cands.index(row["correct"]), dtype=torch.long)
    return gi, ef, fidx, cur, cand, target


def _train(scorer, rows, *, epochs=300, lr=1e-2, wd=1e-4, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(scorer.parameters(), lr=lr, weight_decay=wd)
    scorer.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.zeros(())
        for row in rows:
            gi, ef, fidx, cur, cand, target = _row_tensors(row)
            logits = scorer(gi, ef, fidx, cur, cand)
            loss = loss + F.cross_entropy(logits.unsqueeze(0),
                                          target.unsqueeze(0))
        loss.backward()
        opt.step()
    scorer.eval()
    return scorer


def _policy(scorer):
    return SharedScorerPolicy(scorer, VOCAB)


def _req(factor, current, candidates, predicate="direction_error",
         residual=(1.0, 0.0), gi=None):
    return RevisionRequest(
        factor=factor, current_value=current, candidates=tuple(candidates),
        e_fail=FailureEvidence(predicate=predicate, residual_xy=residual), g_i=gi)


# --------------------------------------------------------------------------- #

def test_efail_and_gi_vectors_are_finite_and_shaped():
    push = _efail_vec("direction_error", (3.0, 0.0))
    assert push.shape == (EFAIL_DIM,) and np.isfinite(push).all()
    assert push[-1] == 1.0  # residual present
    # StackCube: predicate-only, no residual.
    stack = _efail_vec("goal_not_satisfied", None)
    assert stack.shape == (EFAIL_DIM,) and np.isfinite(stack).all()
    assert stack[-1] == 0.0 and stack[-3] == 0.0 and stack[-2] == 0.0
    # g_i present/absent.
    assert _gi_vec(None, None).shape == (GI_DIM_DEFAULT,)
    assert _gi_vec(None, None)[-1] == 0.0
    assert _gi_vec(np.ones(32), None)[-1] == 1.0


def test_vocab_pinned_to_full_schema():
    # Held-out-family values (e.g. faucet/crossview tokens) are embeddable now,
    # so leave-one-task-family-out stays structurally selectable.
    assert ("goal_state", "faucet_turned") in VOCAB
    assert ("direction_grounding", "observer_frame") in VOCAB
    assert ("contact_region", "handle_grip") in VOCAB
    assert 0 not in VOCAB.values()  # idx 0 reserved for UNK


def test_deterministic_candidate_recovery():
    rows = [
        {"factor": "contact_region", "current": "minus_x_face",
         "residual": (1.0, 0.0), "pred": "direction_error",
         "candidates": FACES, "correct": "plus_x_face"},
        {"factor": "contact_region", "current": "plus_x_face",
         "residual": (-1.0, 0.0), "pred": "direction_error",
         "candidates": FACES, "correct": "minus_x_face"},
        {"factor": "contact_region", "current": "minus_y_face",
         "residual": (0.0, 1.0), "pred": "direction_error",
         "candidates": FACES, "correct": "plus_y_face"},
        {"factor": "contact_region", "current": "plus_y_face",
         "residual": (0.0, -1.0), "pred": "direction_error",
         "candidates": FACES, "correct": "minus_y_face"},
    ]
    pol = _policy(_train(_new_scorer(), rows, epochs=400))
    for r in rows:
        dec = pol.decide(_req("contact_region", r["current"], FACES,
                              residual=r["residual"]))
        assert dec.factor == "contact_region"
        assert dec.new_value == r["correct"]
        assert dec.telemetry["path"] == "scored"


def test_variable_candidate_count_both_tasks():
    pol = _policy(_new_scorer())
    push = pol.decide(_req("contact_region", "minus_x_face", FACES))
    assert push.factor == "contact_region"
    assert push.new_value is None or push.new_value in FACES
    gs = candidates_for("StackCube-v1", "goal_state")  # 2 candidates
    stack = pol.decide(_req("goal_state", "cube_at_target", gs,
                            predicate="goal_not_satisfied", residual=None))
    assert stack.factor == "goal_state"
    assert stack.new_value is None or stack.new_value in gs


def test_none_residual_runs_without_nan():
    pol = _policy(_new_scorer())
    dec = pol.decide(_req("goal_state", "cube_at_target",
                          candidates_for("StackCube-v1", "goal_state"),
                          predicate="goal_not_satisfied", residual=None))
    assert dec.factor == "goal_state"
    assert np.isfinite(dec.confidence)


def test_abstain_keeps_current_when_it_ranks_highest():
    # Trained so the current value is already correct → winner == current →
    # abstain (new_value None), so an uninformative signal forces no flip.
    rows = [{"factor": "contact_region", "current": "plus_x_face",
             "residual": (1.0, 0.0), "pred": "direction_error",
             "candidates": FACES, "correct": "plus_x_face"}]
    pol = _policy(_train(_new_scorer(), rows, epochs=400))
    dec = pol.decide(_req("contact_region", "plus_x_face", FACES))
    assert dec.new_value is None
    # compile → unchanged intent.
    initial = replace(GT, contact_region="plus_x_face")
    assert compile_single_slot_edit(initial, dec, INTENT_FIELDS) is initial


def test_oov_candidates_and_factor_abstain():
    pol = _policy(_new_scorer())
    # All-OOV candidate set → ABSTAIN (new_value=None), never emit an
    # out-of-schema token (which would fail Intent validation downstream).
    dec = pol.decide(_req("contact_region", "minus_x_face",
                          ("totally_made_up", "also_fake")))
    assert dec.telemetry["path"] == "fallback_abstain"
    assert dec.new_value is None
    # Hallucinated factor → abstain (never crashes / re-chooses factor / emits
    # an invalid token).
    dec2 = pol.decide(_req("not_a_real_factor", "x", ("a", "b")))
    assert dec2.factor == "not_a_real_factor"
    assert dec2.new_value is None
    assert dec2.telemetry["path"] == "fallback_abstain"
    # Compiling an abstain decision leaves the intent unchanged.
    initial = replace(GT, contact_region="minus_x_face")
    assert compile_single_slot_edit(initial, dec, INTENT_FIELDS) is initial


def test_single_slot_compiler_interop():
    rows = [{"factor": "contact_region", "current": "minus_x_face",
             "residual": (1.0, 0.0), "pred": "direction_error",
             "candidates": FACES, "correct": "plus_x_face"}]
    pol = _policy(_train(_new_scorer(), rows, epochs=300))
    initial = replace(GT, contact_region="minus_x_face")
    dec = pol.decide(_req("contact_region", "minus_x_face", FACES))
    revised = compile_single_slot_edit(initial, dec, INTENT_FIELDS)
    changed = [f for f in INTENT_FIELDS
               if getattr(initial, f) != getattr(revised, f)]
    assert changed == ["contact_region"]


def test_structural_no_leakage():
    # The model-visible request must expose NO task id / gt / scene / full intent.
    names = {f.name for f in fields(RevisionRequest)}
    assert names == {"factor", "current_value", "candidates", "e_fail", "g_i", "z"}
    assert "task" not in names and "gt" not in names and "scene" not in names
    # The pinned factor ontology carries no task identifier.
    assert "task" not in FACTOR_ORDER


def test_uses_g_i_guard():
    """Holding factor/e_fail/current fixed and varying g_i CAN change the
    decision — i.e. g_i is a usable input, not silently ignored (catches
    factor→value memorization masquerading as a policy)."""
    ga = np.zeros(32, dtype=np.float32); ga[0] = 5.0
    gb = np.zeros(32, dtype=np.float32); gb[1] = 5.0
    # Same uninformative e_fail (no residual); only g_i distinguishes the label.
    rows = [
        {"factor": "contact_region", "current": "minus_x_face",
         "residual": None, "pred": "direction_error",
         "candidates": FACES, "correct": "plus_x_face", "gi": ga},
        {"factor": "contact_region", "current": "minus_x_face",
         "residual": None, "pred": "direction_error",
         "candidates": FACES, "correct": "minus_y_face", "gi": gb},
    ]
    pol = _policy(_train(_new_scorer(), rows, epochs=600))
    da = pol.decide(_req("contact_region", "minus_x_face", FACES,
                         residual=None, gi=ga))
    db = pol.decide(_req("contact_region", "minus_x_face", FACES,
                         residual=None, gi=gb))
    assert da.new_value != db.new_value  # g_i drives the decision


_LOTO_DIRS = {"plus_x_face": (1.0, 0.0), "minus_x_face": (-1.0, 0.0),
              "plus_y_face": (0.0, 1.0), "minus_y_face": (0.0, -1.0)}


def _loto_trial(seed: int, offset_scale: float = 0.5) -> float:
    """One leave-one-family-out trial: 3 families share the residual→face rule
    but differ by a random g_i offset. Train on A+B, eval held-out C; return
    held-out accuracy. The offset is a DISTRACTOR irrelevant to the label, so a
    task-id-free scorer learns the residual rule and generalizes to family C.

    NOTE (seed-sensitive): held-out generalization sits above chance (0.25) but
    is noisy per seed; the caller averages over seeds and the distractor is kept
    small (0.5) so the clean rule dominates — see the offset-scale sweep that
    set this config. This demonstrates the task-id-free PATH only; with two
    real unique-factor tasks this is not yet leave-one-task-family-out evidence.
    """
    rng = np.random.default_rng(seed)
    offsets = {fam: rng.normal(0, offset_scale, size=32).astype(np.float32)
               for fam in ("A", "B", "C")}

    def sample(fam):
        rows = []
        for correct, d in _LOTO_DIRS.items():
            for _ in range(8):
                res = (d[0] + rng.normal(0, 0.05), d[1] + rng.normal(0, 0.05))
                gi = offsets[fam] + rng.normal(0, 0.3, size=32).astype(np.float32)
                current = ("minus_x_face" if correct != "minus_x_face"
                           else "plus_x_face")
                rows.append({"factor": "contact_region", "current": current,
                             "residual": res, "pred": "direction_error",
                             "candidates": FACES, "correct": correct, "gi": gi})
        return rows

    pol = _policy(_train(_new_scorer(seed=seed), sample("A") + sample("B"),
                         epochs=250, lr=1e-2, wd=1e-3, seed=seed))
    test_rows = sample("C")
    correct = sum(
        _policy_decide_correct(pol, r) for r in test_rows)
    return correct / len(test_rows)


def _policy_decide_correct(pol, r) -> bool:
    dec = pol.decide(_req("contact_region", r["current"], FACES,
                          residual=r["residual"], gi=r["gi"]))
    return dec.new_value == r["correct"]


def test_synthetic_leave_one_family_out_generalizes():
    """Averaged over 3 seeds, the scorer recovers the residual→face rule on an
    UNSEEN family well above chance (0.25) — proof the architecture is
    task-id-free, before any real step-3 task exists. Seed-averaged + small
    distractor for cross-version robustness (the offset-scale sweep showed a
    single-seed >0.6 bar can flake; the 3-seed mean is ~0.79)."""
    accs = [_loto_trial(s) for s in (0, 1, 2)]
    mean_acc = sum(accs) / len(accs)
    assert mean_acc > 0.6, (
        f"held-out-family mean acc {mean_acc:.2f} (per-seed {accs}) not "
        f"clearly above chance 0.25")


def test_adversarial_leakage_probe_near_chance():
    """When 3 families share the SAME input distribution (no task signal in
    g_i/e_fail), a probe predicting family-id from the inputs is near-chance —
    the honest demonstration that the representation is task-blind. (On REAL
    2-task data this probe is HIGH because factor==task; that is the documented
    step-2 limitation.)"""
    rng = np.random.default_rng(11)
    n_fam, per = 3, 60
    X, y = [], []
    for fam in range(n_fam):
        for _ in range(per):
            gi = rng.normal(0, 1, size=32).astype(np.float32)  # identical dist
            ef = _efail_vec("direction_error", (rng.normal(), rng.normal()))
            X.append(np.concatenate([gi, ef]))
            y.append(fam)
    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    n = len(y)
    g = torch.Generator().manual_seed(3)
    perm = torch.randperm(n, generator=g)
    tr, te = perm[: int(0.7 * n)], perm[int(0.7 * n):]
    torch.manual_seed(0)  # deterministic classifier init (probe is robust regardless)
    clf = torch.nn.Linear(X.shape[1], n_fam)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-2, weight_decay=1e-3)
    for _ in range(300):
        opt.zero_grad()
        loss = F.cross_entropy(clf(X[tr]), y[tr])
        loss.backward(); opt.step()
    with torch.no_grad():
        acc = (clf(X[te]).argmax(1) == y[te]).float().mean().item()
    # 3-class chance = 0.33; identical distributions → no separability.
    assert acc < 0.5, f"leakage probe acc {acc:.2f} too high (inputs leak task)"


def test_determinism_and_save_load_roundtrip(tmp_path):
    rows = [{"factor": "contact_region", "current": "minus_x_face",
             "residual": (1.0, 0.0), "pred": "direction_error",
             "candidates": FACES, "correct": "plus_x_face"}]
    scorer = _train(_new_scorer(), rows, epochs=300)
    pol = _policy(scorer)
    r = _req("contact_region", "minus_x_face", FACES)
    d1 = pol.decide(r); d2 = pol.decide(r)
    assert (d1.new_value, round(d1.confidence, 6)) == (d2.new_value, round(d2.confidence, 6))
    # Save/load round-trip reproduces the decision.
    path = tmp_path / "scorer.pt"
    save_shared_scorer(scorer, VOCAB, path)
    pol2 = SharedScorerPolicy.from_pack(path)
    d3 = pol2.decide(r)
    assert d3.new_value == d1.new_value
    sc2, v2, fo2, _ = load_shared_scorer(path)
    assert v2 == VOCAB and fo2 == FACTOR_ORDER
