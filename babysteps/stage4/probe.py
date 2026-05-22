"""Stage-4 linear probe with chance + shuffled-features baselines."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_score


def _make_splitter(y: np.ndarray) -> Any:
    _, counts = np.unique(y, return_counts=True)
    if counts.min() < 5:
        return LeaveOneOut()
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=0)


def train_probe(X: np.ndarray, y: np.ndarray, *, seed: int = 0) -> dict:
    n_episodes = int(X.shape[0])
    n_unique = int(np.unique(y).size)

    if n_unique <= 1:
        return {
            "n_episodes": n_episodes,
            "n_unique_labels": n_unique,
            "probe_acc_mean": 1.0,
            "probe_acc_std": 0.0,
            "majority_class_acc": 1.0,
            "shuffled_features_acc": 1.0,
            "trivially_constant": True,
        }

    splitter = _make_splitter(y)
    # NB: sklearn 1.7 still accepts LogisticRegression(multi_class=...) but emits
    # a FutureWarning (removed in 1.8; default becomes 'multinomial'). The lbfgs
    # default is already multinomial, so we omit the kwarg — behaviourally
    # identical, warning-free.
    clf = LogisticRegression(max_iter=1000, solver="lbfgs")

    probe_scores = cross_val_score(clf, X, y, cv=splitter, scoring="accuracy")

    rng = np.random.default_rng(seed)
    X_shuf = X.copy()
    rng.shuffle(X_shuf)
    shuf_scores = cross_val_score(clf, X_shuf, y, cv=splitter, scoring="accuracy")

    _, counts = np.unique(y, return_counts=True)
    majority = float(counts.max() / counts.sum())

    return {
        "n_episodes": n_episodes,
        "n_unique_labels": n_unique,
        "probe_acc_mean": float(probe_scores.mean()),
        "probe_acc_std": float(probe_scores.std()),
        "majority_class_acc": majority,
        "shuffled_features_acc": float(shuf_scores.mean()),
        "trivially_constant": False,
    }
