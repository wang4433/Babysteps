"""Sim-free tests for babysteps.stage5.cluster_bootstrap.

Guards the clustered (block) bootstrap used to put a defensible CI on the LOTO
recovery: resample by SCENE SEED (the independent unit), not by episode.
"""
from __future__ import annotations

from babysteps.stage5.cluster_bootstrap import (
    clustered_bootstrap_ci,
    failing_clusters,
    paired_clustered_bootstrap_diff,
)


def _rows():
    """20 seed-clusters x 3 rows; one cluster (seed 124) fails 'scorer_success'
    AND 'oracle_success' (oracle-coincident); 'random_success' fails ~1/3."""
    rows = []
    for s in range(20):
        for j in range(3):
            fail = (s == 124 or s == 19)  # seed 19 stands in for the bad cluster
            rows.append({
                "seed": s,
                "scorer_success": not fail,
                "oracle_success": not fail,
                "open_loop_success": False,
                "random_success": (j == 0),  # 1 of 3 per cluster
            })
    return rows


def test_counts_clusters_not_rows():
    ci = clustered_bootstrap_ci(_rows(), "scorer_success", n_boot=200, seed=0)
    assert ci["n_clusters"] == 20
    assert ci["n_rows"] == 60


def test_constant_column_has_degenerate_ci():
    rows = [{"seed": s, "v": True} for s in range(10) for _ in range(3)]
    ci = clustered_bootstrap_ci(rows, "v", n_boot=200, seed=0)
    assert ci["mean"] == 1.0 and ci["lo"] == 1.0 and ci["hi"] == 1.0


def test_point_estimate_matches_naive_rate():
    rows = _rows()
    ci = clustered_bootstrap_ci(rows, "scorer_success", n_boot=200, seed=0)
    # 19 of 20 clusters pass -> 57/60 rows
    assert abs(ci["mean"] - 57 / 60) < 1e-9
    assert ci["lo"] <= ci["mean"] <= ci["hi"]


def test_paired_diff_identical_columns_is_zero():
    # scorer == oracle row-for-row -> the paired diff is EXACTLY zero, CI [0,0].
    d = paired_clustered_bootstrap_diff(
        _rows(), "scorer_success", "oracle_success", n_boot=300, seed=0)
    assert d["diff"] == 0.0 and d["lo"] == 0.0 and d["hi"] == 0.0


def test_paired_diff_scorer_beats_random_is_positive():
    d = paired_clustered_bootstrap_diff(
        _rows(), "scorer_success", "random_success", n_boot=500, seed=0)
    assert d["diff"] > 0.4
    assert d["lo"] > 0.0  # CI excludes zero -> significant


def test_deterministic_with_fixed_seed():
    a = clustered_bootstrap_ci(_rows(), "scorer_success", n_boot=300, seed=7)
    b = clustered_bootstrap_ci(_rows(), "scorer_success", n_boot=300, seed=7)
    assert a == b


def test_failing_clusters_flags_oracle_coincident():
    fc = failing_clusters(_rows(), "scorer_success")
    # only the bad cluster appears, and every failure is oracle-coincident
    assert set(fc) == {19}
    assert fc[19]["all_oracle_coincident"] is True
    assert fc[19]["n_fail"] == 3


def test_empty_rows():
    assert clustered_bootstrap_ci([], "v")["n_clusters"] == 0
    assert paired_clustered_bootstrap_diff([], "a", "b")["n_clusters"] == 0
