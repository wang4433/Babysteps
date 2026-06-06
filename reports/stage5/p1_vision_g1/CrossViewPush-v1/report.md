# Stage-5 P1 — CrossViewPush · `direction_grounding` (latent-groundability boundary point)

**Verdict: ❌ INVISIBLE / METADATA factor — not a clean latent task.** This is a
*characterized boundary point*, the cross-view twin of PickCube·`contact_region`
and StackCube·`goal_state` (clip): the factor is structurally absent from the
demo pixels, so the G1 latent probe cannot recover it. No GPU probe was run
because the cell is **vacuous by construction** (see below); a naive
"mirror PushCube" run would silently emit a `trivially_constant` row that looks
like a test but tests nothing.

> Scope choice (Stage-5): **Path A — characterize it.** The intent is to log
> this as a boundary point that *strengthens* the factor-observability spine of
> the paper, not to claim a 2nd clean latent task. The full on-pixels empirical
> probe (Path B) is deferred — it would require a non-canonical mixed
> `actor`/`observer` data cut (a validity red flag) and is expected to be
> negative anyway.

## Why the cell is non-probeable (three independent, code-verified reasons)

1. **The initial-intent label is constant.** The demo→intent path hard-codes
   `direction_grounding="actor_frame"` for every seed and observer yaw —
   `babysteps/envs/crossview_adapter.py:65` (`oracle_correct_intent`) and `:100`
   (`scripted_demo_to_intent`, commented "the egocentric bug"). `observer_frame`
   appears **only** in the *revised* (retry) intent, never in the initial one
   the probe reads. Verified empirically on the committed snapshot
   (`tests/snapshots/crossview_samples_seeds_0_4.jsonl`): all 5 seeds
   `initial_intent.direction_grounding == actor_frame`. A single-valued label →
   `nested_cv_probe_one_factor` returns `trivially_constant` → the gate
   excludes it.

2. **The factor is outside the probe's loop.** `direction_grounding` is the
   additive 7th field; `INTENT_FIELDS` (`babysteps/schemas.py`) is the 6-tuple
   and excludes it, and `scripts/stage5_p1_g1_cert.py` iterates only
   `INTENT_FIELDS`. The default probe never even reads `direction_grounding`.

3. **No pixel signature.** All demo phases render from PushCube's **world**
   camera; the observer yaw is applied to the *grounding math*
   (`observe_demo` / resolution), **not** to a physically rotated SAPIEN camera
   (`babysteps/render/crossview.py:108-116`, verbatim NOTE). And `actor_frame`
   resolves to identity (`babysteps/envs/scene.py`), so the demo executes the
   **same world-correct push from the same camera for every yaw**. The
   observer frame lives in `scene.extra["observer_yaw_deg"]` (metadata), not in
   the clip — there isn't even a background-rotation channel to leak it (unlike
   the PlugCharger `charger_yaw` false positive,
   `reports/stage5/plugcharger_probe`).

## Place in the factor-observability boundary

| task · factor | groundable from 3rd-person demo? | why |
|---|---|---|
| PushCube · object_motion | ✅ PASS (0.95) | single-object motion plainly visible |
| StackCube · object_motion | ❌ 0.685 | two-object *relational* direction (rep-blocked) |
| StackCube · goal_state (clip) | ❌ 0.82 | demo hides vertical stack motion (info-loss) |
| PickCube · contact_region | ❌ invisible | gripper occludes contact site |
| **CrossViewPush · direction_grounding** | **❌ invisible (metadata)** | **world camera + identity push → factor not in pixels** |

This is consistent with the locked guidance (`milestones.md`): a genuine 2nd
latent task needs a factor *plainly visible in a third-person RGB clip* — which
`direction_grounding`, as built, is not.

## Guard

`tests/test_crossview.py::test_direction_grounding_is_non_probeable_constant_initial_label`
asserts the constant-label invariant (single-valued `actor_frame` across yaws/goals,
and exclusion from `INTENT_FIELDS`) so this boundary characterization stays true.

## Optional empirical confirmation (not required)

The structural proof is complete (reason 1 is an unconditional hard-code, not a
sampling fact). If an empirical artifact is wanted, regenerate a 24-seed cut on
GPU (`scripts/stage0_collect.py --task CrossViewPush-v1`) and confirm
`initial_intent.direction_grounding` is uniformly `actor_frame` — but this adds
no information beyond the adapter source.
