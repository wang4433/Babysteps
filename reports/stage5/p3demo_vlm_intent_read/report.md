# Stage-5 option-3 probe — VLM reads object_motion off the demo

VLM-vs-oracle agreement reading `object_motion` from a third-person
demo strip (start/middle/end). No training. Compared against the P1
G1 DINOv2 linear probe and the M2a learned slot on the same factor.

Pre-registered kill criterion: agreement < 0.90 ⇒ below the G1
gate by construction ⇒ relational-gap-reduction ablation, not a pass.

| task | n | VLM read | DINOv2 probe | learned slot | majority | gate |
| --- | --- | --- | --- | --- | --- | --- |
| PushCube-v1 | 20 | **0.000** | 0.95 | 0.95 | 0.50 | BELOW-GATE |
| StackCube-v1 | 40 | **0.250** | 0.42 | 0.95 | 0.25 | BELOW-GATE |

## PushCube-v1 — confusion (rows=gt, cols=pred)

| gt \ pred | parse_fail | translate_+x | translate_+y | translate_-x | translate_-y |
| --- | --- | --- | --- | --- | --- |
| translate_+x | 0 | 0 | 10 | 0 | 0 |
| translate_+y | 0 | 0 | 0 | 0 | 0 |
| translate_-x | 0 | 0 | 9 | 0 | 0 |
| translate_-y | 0 | 0 | 1 | 0 | 0 |

## StackCube-v1 — confusion (rows=gt, cols=pred)

| gt \ pred | parse_fail | translate_+x | translate_+y | translate_-x | translate_-y |
| --- | --- | --- | --- | --- | --- |
| translate_+x | 0 | 0 | 10 | 0 | 0 |
| translate_+y | 0 | 0 | 10 | 0 | 0 |
| translate_-x | 0 | 0 | 10 | 0 | 0 |
| translate_-y | 0 | 0 | 10 | 0 | 0 |
