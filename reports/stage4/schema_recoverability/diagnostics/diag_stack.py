"""Diagnostic: why StackCube/object_motion only reaches 0.75 (Task-6 aid)."""
import collections
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/scratch/gilbreth/wang4433/babysteps")
sys.path.insert(0, str(ROOT))
from babysteps.schemas import EpisodeRecord  # noqa: E402
from babysteps.stage4.features import extract_episode_features  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import LeaveOneOut, cross_val_predict  # noqa: E402
from sklearn.preprocessing import LabelEncoder  # noqa: E402

p = ROOT / "datasets/stage0_baselines/babysteps_selective/StackCube-v1/samples.jsonl"
recs = [EpisodeRecord.from_jsonl_line(line).to_dict() for line in open(p) if line.strip()]
X = np.stack([extract_episode_features(r) for r in recs])
labels = [r["execution"]["initial_intent"]["object_motion"] for r in recs]
le = LabelEncoder()
y = le.fit_transform(labels)
classes = list(le.classes_)
print("classes:", classes)
# feature idx: 0,1 start; 2,3 end; 4,5 disp; 6 disp_norm; 7 angle; 8 path_len
print("\nper-class demo-trajectory direction:")
for ci, cname in enumerate(classes):
    m = X[y == ci]
    print(f"  {cname:14s} n={len(m):2d}  "
          f"disp=({m[:, 4].mean():+.4f},{m[:, 5].mean():+.4f})  "
          f"angle_deg={np.degrees(m[:, 7]).mean():+7.1f}  "
          f"disp_norm={m[:, 6].mean():.4f}")

clf = LogisticRegression(max_iter=1000, solver="lbfgs")
pred = cross_val_predict(clf, X, y, cv=LeaveOneOut())
print(f"\nLOO accuracy: {(pred == y).mean():.4f}  ({(pred == y).sum()}/{len(y)})")
print("confusion (true -> pred):")
conf = collections.Counter((classes[t], classes[pr]) for t, pr in zip(y, pred))
for (t, pr), n in sorted(conf.items()):
    print(f"  {t:12s} -> {pr:12s} : {n}{'' if t == pr else '   <-- error'}")
