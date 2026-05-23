"""Isolate the pre-fix StackCube/object_motion 0.72 cause: angle wraparound.

Reproduces the three feature variants reported in the parent notes.md
("Update (2026-05-23)" section):

  A: current 19-dim (start, end, disp, |disp|, angle, path_len, oh) → 0.72
  B: angle → [sin, cos]                                  (20-dim) → 0.95
  C: angle removed                                       (18-dim) → 0.72

Run from the repo root after the 20-dim features.py is committed; the
"current" 19-dim variant is inlined here as a control. Output is appended
to diag_angle_variants.out next to this file.

    cd <repo root>
    python reports/stage4/schema_recoverability_varied/diagnostics/diag_angle_variants.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.preprocessing import LabelEncoder

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from babysteps.schemas import CONTACT_REGIONS, GOAL_STATES  # noqa: E402

_CO = tuple(sorted(CONTACT_REGIONS))
_GO = tuple(sorted(GOAL_STATES))


def _feats(rec: dict, variant: str) -> np.ndarray:
    demo = rec["demo"]
    traj = np.asarray(demo["object_trajectory"], dtype=np.float64)
    start, end = traj[0], traj[-1]
    disp = end - start
    disp_norm = float(np.linalg.norm(disp))
    angle = float(np.arctan2(disp[1], disp[0]))
    path_len = (
        float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)))
        if traj.shape[0] >= 2 else 0.0
    )
    contact_oh = np.zeros(len(_CO)); contact_oh[_CO.index(demo["contact_region_label"])] = 1.0
    goal_oh = np.zeros(len(_GO)); goal_oh[_GO.index(demo["final_state"])] = 1.0
    base = [start, end, disp, [disp_norm]]
    if variant == "A":      # raw angle (pre-fix)
        base += [[angle], [path_len]]
    elif variant == "B":    # [sin, cos] (post-fix)
        base += [[np.sin(angle), np.cos(angle)], [path_len]]
    elif variant == "C":    # angle removed
        base += [[path_len]]
    return np.concatenate([*base, contact_oh, goal_oh])


def main() -> None:
    p = _ROOT / "datasets/stage4/varied_intent/StackCube-v1/samples.jsonl"
    records = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    labels = [r["execution"]["initial_intent"]["object_motion"] for r in records]
    enc = LabelEncoder().fit(labels)
    y = enc.transform(labels)
    print(f"n={len(records)}, classes={list(enc.classes_)}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for variant in ("A", "B", "C"):
        X = np.stack([_feats(r, variant) for r in records])
        clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
        pred = cross_val_predict(clf, X, y, cv=cv)
        cm = np.zeros((4, 4), dtype=int)
        for t, pp in zip(y, pred):
            cm[t, pp] += 1
        print(
            f"\n--- variant {variant}  feat_dim={X.shape[1]}  "
            f"acc={scores.mean():.3f} ± {scores.std():.3f} ---"
        )
        print("confusion (rows=true, cols=pred), labels:", list(enc.classes_))
        for i, row in enumerate(cm):
            print(f"  {enc.classes_[i]:14s}  {row.tolist()}")


if __name__ == "__main__":
    main()
