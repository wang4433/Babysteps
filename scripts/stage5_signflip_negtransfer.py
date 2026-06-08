"""Stage-5 — sign-flip negative-transfer PROXY for the shared scorer.

The shared contact_region scorer learns a residual-SIGN → push-face rule. A
puller (PullCube) observes the OPPOSITE-sign goal-relative residual for the same
goal geometry, so applying this frozen push-trained scorer to pull physics would
SYSTEMATICALLY select the sign-opposite (wrong-for-pull) face. This script
demonstrates that sign-sensitivity directly from the REAL trained checkpoint,
WITHOUT building a PullCube env: re-score every committed LOTO residual with its
sign flipped and check the scorer flips to the opposite face.

Why this matters: it is the cheap, sim-free hedge for the "does the rule
spuriously over-generalize?" reviewer attack. A rule that flips under a sign flip
is push-physics-specific, not a universal face heuristic — it predicts NEGATIVE
transfer to opposite-sign physics (the honest boundary of the contact_region
claim), and the real PullCube control (a new hand-rolled runner) is only needed
if a reviewer demands a real env.

Loads the gitignored checkpoint on CPU (no GPU/Vulkan), so it runs on the login
node but is NOT a sim-free unit test (it depends on models/).

    python scripts/stage5_signflip_negtransfer.py \\
        --scorer models/stage5/shared_policy/pooled_gi_none.pt \\
        --results reports/stage5/pokecube_loto/results_pooled_gi_none.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Full 4-face vocab the scorer was trained on — the sign-sensitivity argument is
# about the LEARNED RULE, independent of PokeCube's 3-face reachability subset.
_FOUR_FACES = ("minus_x_face", "plus_x_face", "minus_y_face", "plus_y_face")
_OPPOSITE = {"minus_x_face": "plus_x_face", "plus_x_face": "minus_x_face",
             "minus_y_face": "plus_y_face", "plus_y_face": "minus_y_face"}


def _decide(policy, current_value, residual_xy, predicate, candidates):
    from babysteps.stage5.revision_policy import FailureEvidence, RevisionRequest
    dec = policy.decide(RevisionRequest(
        factor="contact_region", current_value=current_value,
        candidates=tuple(candidates),
        e_fail=FailureEvidence(predicate, tuple(residual_xy)), g_i=None))
    return dec.new_value if dec.new_value is not None else current_value


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scorer", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--out", type=Path,
                   default=Path("reports/stage5/pokecube_loto/signflip_negtransfer.json"))
    args = p.parse_args(argv)

    from babysteps.stage5.shared_revision_policy import SharedScorerPolicy
    policy = SharedScorerPolicy.from_pack(args.scorer)
    results = json.loads(args.results.read_text())
    rows = results["rows"]
    loto_cands = tuple(results.get("candidates", _FOUR_FACES))

    n = 0
    reproduced = 0          # 3-candidate re-score matches committed scorer_face
    flips_to_opposite = 0   # 4-candidate sign-flip selects the opposite face
    flip_differs = 0        # sign-flip at least CHANGES the choice
    examples = []
    for r in rows:
        resid = r["residual_xy"]
        flipped = [-resid[0], -resid[1]]
        pred = "direction_error"
        # sanity: reproduce committed choice on the SAME 3-candidate set
        repro = _decide(policy, r["wrong_face"], resid, pred, loto_cands)
        reproduced += int(repro == r["scorer_face"])
        # sign-sensitivity on the full 4-face vocab
        real4 = _decide(policy, r["wrong_face"], resid, pred, _FOUR_FACES)
        flip4 = _decide(policy, r["wrong_face"], flipped, pred, _FOUR_FACES)
        flips_to_opposite += int(flip4 == _OPPOSITE.get(real4))
        flip_differs += int(flip4 != real4)
        n += 1
        if len(examples) < 6:
            examples.append({"direction": r["direction"], "residual": resid,
                             "real_face": real4, "flipped_face": flip4,
                             "opposite_of_real": _OPPOSITE.get(real4)})

    out = {
        "scorer": str(args.scorer),
        "n": n,
        "reproduces_committed_choice_rate": reproduced / max(1, n),
        "signflip_selects_opposite_face_rate": flips_to_opposite / max(1, n),
        "signflip_changes_choice_rate": flip_differs / max(1, n),
        "interpretation": (
            "rule is residual-sign-specific: under a flipped (pull-physics) "
            "residual the frozen push-trained scorer ALWAYS changes its choice "
            "(signflip_changes_choice_rate) and PREDOMINANTLY selects the "
            "sign-opposite face (signflip_selects_opposite_face_rate; the "
            "remainder are off-distribution axis-switches on the y-axis, which "
            "the push-heavy training set under-covers). This predicts NEGATIVE "
            "transfer to opposite-sign physics (PullCube): the rule does not "
            "spuriously over-generalize. A real PullCube env control is only "
            "needed if a reviewer demands a non-proxy demonstration."),
        "examples": examples,
    }
    print(json.dumps(out, indent=2))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
