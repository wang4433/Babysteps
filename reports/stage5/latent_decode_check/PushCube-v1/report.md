# Stage-5 latent-input faithfulness check — PushCube-v1

How well the vision-decoded initial intent (DINOv2 → IntentHead →
nearest-centroid) reproduces the hand-authored JSON intent it would
replace, on the P2 held-out seeds.

- Pack: `models/stage5/p1_vision/PushCube-v1`
- Features: `datasets/stage5/varied_intent/PushCube-v1/features`
- Episodes: `datasets/stage5/p2_vlm/PushCube-v1/episodes.jsonl`
- Scored: **50** (missing feature files: 0)
- Decoded-from-vision factors: `object_motion, contact_region, approach_direction`
- Constant factors (filled from task base): `goal_state, constraint_region, embodiment_mapping`

## Per-factor latent-vs-JSON agreement

| factor | agreement |
|---|---|
| `object_motion` | 0.980 |
| `contact_region` | 0.980 |
| `approach_direction` | 0.980 |

- **All decodable factors agree (exact match): 0.980**

## Confusion (decoded → stored), per factor

- `object_motion`: `translate_+x->translate_+x`×49, `translate_-x->translate_+x`×1
- `contact_region`: `minus_x_face->minus_x_face`×49, `plus_x_face->minus_x_face`×1
- `approach_direction`: `from_minus_x->from_minus_x`×49, `from_plus_x->from_minus_x`×1
