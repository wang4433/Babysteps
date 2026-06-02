# Stage-5 P1 — Vision-grounded G1 (DINOv2 ViT-B/14)

Input Z: 768-dim DINOv2 ViT-B/14 features (spatial_mean pool over demo frames).
IntentHead: F=6, d_slot=32, hidden=256, n_epochs=300, lr=0.01.
Outer CV: per-fold IntentHead training; frozen LogisticRegression on G_train, evaluated on G_test.

### StackCube-v1

| factor | class | n_unique | n_episodes | majority | shuffled | probe ± std | gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| approach_direction | trivially_constant | 1 | 40 | 1.00 | 1.00 | 1.00 ± 0.00 | trivial |
| constraint_region | trivially_constant | 1 | 40 | 1.00 | 1.00 | 1.00 ± 0.00 | trivial |
| contact_region | trivially_constant | 1 | 40 | 1.00 | 1.00 | 1.00 ± 0.00 | trivial |
| embodiment_mapping | trivially_constant | 1 | 40 | 1.00 | 1.00 | 1.00 ± 0.00 | trivial |
| goal_state | trivially_constant | 1 | 40 | 1.00 | 1.00 | 1.00 ± 0.00 | trivial |
| object_motion | geometric | 4 | 40 | 0.25 | 0.33 | 0.30 ± 0.19 | FAIL |


**Gate:** all geometric cells >= 90% (margin 10% over majority & shuffled) -> **FAIL**

Cells: 6 total | 1 geometric (0 pass / 1 fail) | 0 label-identity | 5 trivially constant.

## Falsification log

Three pooling ablations were pre-registered against `_pool_cls` (`babysteps/stage4/vision_features.py`). Only one (spatial_mean) is cached on disk in `datasets/stage5/varied_intent/<task>/features/`; the others were tried in sequence and the cache was overwritten between runs. Numbers below are from those runs (PushCube and StackCube `object_motion` rows of the gate table).

| pool | dim | PushCube object_motion | StackCube object_motion |
| --- | --- | --- | --- |
| cls_mean | 768 | 0.95 ± 0.22 (weak — near-binary labels) | 0.30 ± 0.22 (FAIL) |
| cls_first_last | 1536 | 0.95 ± 0.22 (clean PASS) | 0.23 ± 0.09 (below chance — d ≫ n at d=1536, n=40) |
| **spatial_mean** | **768** | **0.95 ± 0.22** | **0.42 ± 0.10** (best of failed; no d-vs-n pathology) |

## Interpretation

PushCube `object_motion` recovers cleanly under any pool, including `cls_mean` — but PushCube's 3-class label set collapses to a near-binary +x/-x split (one outlier seed produces the third class), so the 0.95 is a weaker pass than the headline suggests.

StackCube `object_motion` is the cleaner test: 4 balanced classes (10/10/10/10), labels defined by the **relative** direction from cubeA to cubeB. Both cubes are visible at the first and last frames (cubeA just translates on top of cubeB), so `cls_first_last`'s between-frame delta does not isolate a single moving object. Increasing dim to 1536 with n=40 pushes the linear probe below chance.

`spatial_mean` is the best of the three ablations: 768-dim avoids the d ≫ n pathology, mean-pooled patch tokens retain more local structure than CLS alone. It still falls to 0.42 — well below the 0.90 gate.

**Falsifiable finding:** Frozen DINOv2 with a linear nested-CV probe supports single-object motion direction when temporal endpoints are exposed (PushCube `cls_first_last` 0.95 PASS), but fails on two-object relational direction under n=40 (StackCube under all three pools, best 0.42). The bottleneck is not only temporal pooling; it is object-centric relational representation, possibly compounded by data scale.

## Implications for S5

S5 scope narrows to PushCube end-to-end only. StackCube is logged as an open relational-factor failure of frozen-feature G1, not papered over with mixed per-task pooling. The remaining ablations spec §6 listed (R3M, encoder swap) are not pursued — they would not address the relational-representation bottleneck diagnosed here.

**Failing geometric cells:** StackCube-v1/object_motion

