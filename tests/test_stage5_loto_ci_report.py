"""Sim-free tests for scripts/stage5_loto_ci_report.py.

Guards the reviewer-facing selectivity disclosure (the `wrong_face_changes_choice`
metric had a semantic bug: it must compare ACROSS wrong_face values within a
direction, not look for within-pair stochasticity) and the analyze() plumbing.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _ROOT / "scripts"
for p in (str(_ROOT), str(_SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

import stage5_loto_ci_report as R


def _row(seed, direction, wrong_face, scorer_face, *, scorer_ok=True):
    return {
        "seed": seed, "direction": direction, "wrong_face": wrong_face,
        "scorer_face": scorer_face,
        "scorer_face_correct": scorer_ok,
        "open_loop_success": False,
        "scorer_success": scorer_ok,
        "random_success": False,
        "oracle_success": True,
    }


def test_selectivity_pokecube_like_deterministic_wrongface_irrelevant():
    # +x always -> minus_x_face regardless of which wrong face we started from.
    rows = [_row(s, "+x", wf, "minus_x_face")
            for s in range(3)
            for wf in ("minus_y_face", "plus_y_face")]
    sel = R.selectivity_disclosure(rows)
    assert sel["direction_to_face_is_deterministic"] is True
    assert sel["wrong_face_changes_choice"] is False
    assert sel["direction_to_face"]["+x"] == ["minus_x_face"]


def test_selectivity_detects_wrong_face_dependence():
    # Same direction, but the chosen face DEPENDS on wrong_face -> must be True.
    # (The old buggy metric returned False here: a false negative.)
    rows = [
        _row(0, "+x", "minus_y_face", "minus_x_face"),
        _row(0, "+x", "plus_y_face", "plus_y_face"),
        _row(1, "+x", "minus_y_face", "minus_x_face"),
        _row(1, "+x", "plus_y_face", "plus_y_face"),
    ]
    sel = R.selectivity_disclosure(rows)
    assert sel["wrong_face_changes_choice"] is True


def test_selectivity_within_pair_stochasticity_is_not_wrong_face_dependence():
    # A single (direction, wrong_face) pair maps to two faces (stochastic), but
    # there is only ONE wrong_face -> wrong_face does NOT change the choice.
    # (The old buggy metric returned True here: a false positive.)
    rows = [
        _row(0, "+x", "minus_y_face", "minus_x_face"),
        _row(1, "+x", "minus_y_face", "plus_y_face"),  # same pair, diff face
    ]
    sel = R.selectivity_disclosure(rows)
    assert sel["wrong_face_changes_choice"] is False


def test_analyze_plumbing_smoke():
    rows = [_row(s, "+x", "minus_y_face", "minus_x_face") for s in range(5)]
    a = R.analyze({"rows": rows, "scorer": "x", "candidates": ["a"]},
                  n_boot=100, seed=0)
    assert "shared_scorer (frozen)" in a["recovery_ci"]
    assert a["face_acc_ci"]["n_clusters"] == 5
    assert a["paired_diffs"]["scorer_minus_oracle"]["n_clusters"] == 5
