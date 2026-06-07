"""Stage-5 — clustered (block) bootstrap CIs for the LOTO eval (PURE, sim-free).

The PokeCube LOTO eval emits one row per (scene seed × goal direction × wrong
initial face). Those rows are NOT independent: every row sharing a ``seed`` comes
from the SAME sampled scene geometry, so the independent statistical unit is the
SCENE SEED (~20 clusters), not the ~120 episodes. A naive per-episode CI would
overstate precision by ~sqrt(rows / clusters). This module resamples whole seed
CLUSTERS with replacement (the standard cluster / block bootstrap) so the CI
reflects the true cluster sample size.

This is the user's locked step 2 ("clustered bootstrap CI by independent scene
seed") and the pasted-analysis Table 5 ("Poke's 120 episodes = only 20
independent scene clusters → CI over 20 seeds"). No env / GPU / torch — operates
on the committed results-JSON rows, so it runs on the login node and delivers a
defensible interval with zero compute.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Mapping, Optional, Sequence


def _group_by_cluster(rows: Sequence[Mapping], cluster_key: str):
    groups: dict[object, list] = defaultdict(list)
    for r in rows:
        groups[r[cluster_key]].append(r)
    return groups


def _rate(rows: Sequence[Mapping], value_key: str) -> float:
    vals = [bool(r[value_key]) for r in rows if r.get(value_key) is not None]
    return (sum(vals) / len(vals)) if vals else 0.0


def _percentile(sorted_vals: Sequence[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = int(q * (len(sorted_vals) - 1))
    return sorted_vals[idx]


def clustered_bootstrap_ci(
    rows: Sequence[Mapping], value_key: str, *, cluster_key: str = "seed",
    n_boot: int = 10000, alpha: float = 0.05, seed: int = 0,
) -> dict:
    """Percentile CI for the mean of a boolean ``value_key``, resampling whole
    ``cluster_key`` groups with replacement.

    Returns ``{mean, lo, hi, n_clusters, n_rows}``. ``mean`` is the point
    estimate over the real rows; ``lo``/``hi`` are the ``alpha/2`` and
    ``1-alpha/2`` percentiles of the cluster-bootstrap distribution."""
    groups = _group_by_cluster(rows, cluster_key)
    clusters = list(groups.values())
    n = len(clusters)
    point = _rate(rows, value_key)
    if n == 0:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "n_clusters": 0, "n_rows": 0}
    rng = random.Random(seed)
    boots: list[float] = []
    for _ in range(n_boot):
        sample_rows: list = []
        for _ in range(n):
            sample_rows.extend(clusters[rng.randrange(n)])
        boots.append(_rate(sample_rows, value_key))
    boots.sort()
    return {
        "mean": point,
        "lo": _percentile(boots, alpha / 2),
        "hi": _percentile(boots, 1 - alpha / 2),
        "n_clusters": n,
        "n_rows": len(rows),
    }


def paired_clustered_bootstrap_diff(
    rows: Sequence[Mapping], key_a: str, key_b: str, *, cluster_key: str = "seed",
    n_boot: int = 10000, alpha: float = 0.05, seed: int = 0,
) -> dict:
    """CI for the PAIRED difference ``mean(key_a) - mean(key_b)``.

    The SAME cluster resample feeds both arms each iteration (so the difference
    is paired and the between-condition correlation is preserved — exactly what
    is needed to show ``shared_scorer == oracle`` with a tight diff CI). Returns
    ``{diff, lo, hi, n_clusters}``."""
    groups = _group_by_cluster(rows, cluster_key)
    clusters = list(groups.values())
    n = len(clusters)
    point = _rate(rows, key_a) - _rate(rows, key_b)
    if n == 0:
        return {"diff": 0.0, "lo": 0.0, "hi": 0.0, "n_clusters": 0}
    rng = random.Random(seed)
    boots: list[float] = []
    for _ in range(n_boot):
        sample_rows: list = []
        for _ in range(n):
            sample_rows.extend(clusters[rng.randrange(n)])
        boots.append(_rate(sample_rows, key_a) - _rate(sample_rows, key_b))
    boots.sort()
    return {
        "diff": point,
        "lo": _percentile(boots, alpha / 2),
        "hi": _percentile(boots, 1 - alpha / 2),
        "n_clusters": n,
    }


def failing_clusters(rows: Sequence[Mapping], value_key: str, *,
                     cluster_key: str = "seed") -> dict:
    """Diagnose which clusters carry the failures of ``value_key``, and whether
    the privileged ``oracle_success`` ALSO fails there (→ oracle-coincident
    geometry, not a policy error). Used for the honest failure-attribution note
    the synthesis requires."""
    groups = _group_by_cluster(rows, cluster_key)
    out: dict = {}
    for cl, grp in groups.items():
        fails = [r for r in grp if not r.get(value_key)]
        if fails:
            oracle_fails = sum(1 for r in fails if not r.get("oracle_success"))
            out[cl] = {
                "n_fail": len(fails),
                "n_fail_oracle_also_fails": oracle_fails,
                "all_oracle_coincident": oracle_fails == len(fails),
            }
    return out
