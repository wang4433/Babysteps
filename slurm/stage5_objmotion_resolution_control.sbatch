#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --requeue
#SBATCH --job-name=s5-objmotion-resctrl
#SBATCH --output=slurm/logs/objmotion-resctrl-%j.out
#SBATCH --error=slurm/logs/objmotion-resctrl-%j.err
# Stage-5 — decompose the V-JEPA object_motion lift (0.856 LR vs DINOv2-B@224
# 0.682). V-JEPA conflates model + 384px + clip-tokens. These FRAME-encoder
# controls on the SAME 200 StackCube frames isolate the cause:
#   DINOv2-B @224  (baseline, already cached: 0.682)
#   DINOv2-B @392  -> resolution effect at fixed (small) model
#   DINOv3-L @224  -> model scale/pretraining at fixed (low) res
#   DINOv3-L @384  -> scale + resolution, a hi-res strong FRAME encoder
#   V-JEPA-L @384  (already cached: 0.856) -> + video architecture/temporal
# If DINOv3-L@384 ≈ 0.85 -> the lift is "hi-res ViT-L frame encoder", not video.
# If it stays ≈0.68    -> the lift is V-JEPA-specific.
set -euo pipefail
cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export TORCH_HOME=/scratch/gilbreth/wang4433/torch_hub_cache

ROOT=datasets/stage5/object_relation_n200/StackCube-v1
FR=$ROOT/frames
FE=$ROOT/features

echo "=== cache DINOv2-B @392 (spatial_mean) ==="
python scripts/stage5_cache_dinov2.py --frames-dir "$FR" --out-dir "$FE" \
  --encoder dinov2_vitb14 --pool spatial_mean --resolution 392 --feature-suffix dinov2_392 --device cuda
echo "=== cache DINOv3-L @224 (spatial_mean) ==="
python scripts/stage5_cache_dinov2.py --frames-dir "$FR" --out-dir "$FE" \
  --encoder dinov3_vitl16 --pool spatial_mean --resolution 224 --feature-suffix dinov3l224 --device cuda
echo "=== cache DINOv3-L @384 (spatial_mean) ==="
python scripts/stage5_cache_dinov2.py --frames-dir "$FR" --out-dir "$FE" \
  --encoder dinov3_vitl16 --pool spatial_mean --resolution 384 --feature-suffix dinov3l384 --device cuda

echo "=== leak-free multi-seed linear-probe decomposition (object_motion) ==="
python - <<'PY'
import json, numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
root=Path("datasets/stage5/object_relation_n200/StackCube-v1")
recs=[json.loads(l) for l in open(root/"samples.jsonl") if l.strip()]
seeds=[int(r["episode_id"].split("_")[-1]) for r in recs]
y=LabelEncoder().fit_transform([r["execution"]["initial_intent"]["object_motion"] for r in recs])
def load(s): return np.stack([np.load(root/"features"/f"seed_{x:04d}_{s}.npy") for x in seeds]).astype(np.float64)
print(f"{'features':22s} {'dim':>5s}  direct-LR(leak-free, 10 seeds)   gate=0.90 chance=0.27")
for s in ["dinov2","dinov2_392","dinov3l224","dinov3l384","vjepa21","vjepa21_shuf"]:
    try: Z=load(s)
    except FileNotFoundError: print(f"{s:22s}  (missing)"); continue
    a=[cross_val_score(make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000)),Z,y,
        cv=StratifiedKFold(5,shuffle=True,random_state=r)).mean() for r in range(10)]
    a=np.array(a); print(f"{s:22s} {Z.shape[1]:5d}  {a.mean():.3f} ± {a.std():.3f}")
PY
echo "=== DONE ==="
