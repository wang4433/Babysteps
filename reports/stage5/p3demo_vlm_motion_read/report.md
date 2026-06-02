# Stage-5 option-3 probe v2 — VLM reads object_motion (before/after)

START and END frames fed as two SEPARATE images; VLM reports an
image-relative direction, mapped to the world token via a fixed
blob-calibrated lookup. (Supersedes the panel-strip run, which the
VLM read as layout, not motion.)

## PushCube-v1  (n=20, mode=mapped)
- reference: DINOv2 probe 0.95, learned slot 0.95, majority 0.50
- **VLM read accuracy: 0.550**  (gate 0.90: BELOW-GATE)

| gt \ pred | none | translate_+x | translate_-x |
| --- | --- | --- | --- |
| translate_+x | 0 | 2 | 8 |
| translate_-x | 0 | 0 | 9 |
| translate_-y | 1 | 0 | 0 |

## StackCube-v1  (n=40, mode=crosstab)
- reference: DINOv2 probe 0.42, learned slot 0.95, majority 0.25
- **best-case image→world map accuracy (optimistic upper bound): 0.325**  (majority 0.25) — if this is near majority, the directions are not separable in this view.

| oracle \ VLM dir | left | none | right | up |
| --- | --- | --- | --- | --- |
| translate_+x | 1 | 2 | 3 | 4 |
| translate_+y | 1 | 0 | 5 | 4 |
| translate_-x | 1 | 0 | 5 | 4 |
| translate_-y | 2 | 0 | 4 | 4 |
