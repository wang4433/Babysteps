# Stage-4 Milestone 2 — Learned Slot-Intent Latent — Design Spec

> **Status:** design draft 2026-05-23, **pending user review before
> implementation.** This is the "M2 is sufficiently specified but not
> yet safe to implement" deliverable from the 2026-05-23 goal block.
> **Why this exists:** the M1.5 varied cut + the 2026-05-23
> circular-angle fix together make the Stage-4 cert scaffold
> evaluable, non-trivial, and now-passing on hand-built features
> (see `reports/stage4/schema_recoverability_varied/notes.md`). M2 is
> unblocked; this spec pins down the smallest version that can claim
> "factorized latent intent" without also claiming
> "perception-from-pixels".
>
> Authority: `goal.md` §"Stage 4 / Architecture" + §"Stage 4 /
> Certification Interface" + §"Stage 4 / Success Criteria". Code source
> of truth: `babysteps/stage4/{features,probe,report}.py`,
> `babysteps/{schemas,episode,revision,failure}.py`,
> `babysteps/skills/*`.

---

## 1. Scope & Non-Goals

**In scope:**

- A **factorized latent intent** `G = {g^1, …, g^F}` indexed by the
  **six Stage-0 intent factors** (`goal_state`, `object_motion`,
  `contact_region`, `approach_direction`, `constraint_region`,
  `embodiment_mapping`) — one slot per factor.
- An **`IntentHead`** that consumes a fixed input encoding `Z` of demo
  evidence and emits `G`. The 20-dim handcrafted feature vector from
  `babysteps/stage4/features.py` is the **default `Z`** for M2; a
  learned `SlotEncoder(rgbd_video) → Z` is **deferred to M2b** (see
  §6).
- A **`ReviseHead(g^i, failure_packet_vector) → g̃^i`** that consumes
  exactly one slot intent and the vectorized failure packet and emits
  exactly one revised slot intent. Type signature enforces the
  single-factor revision invariant.
- A **frozen-decoder path** `G → Intent JSON` that maps slot vectors
  back to the existing discrete schema so the existing skill compilers
  (`babysteps/skills/*`) can execute the revised intent **unchanged**.
- The three Stage-4 certification gates (`goal.md` §"Certification
  Interface"): (1) probe recoverability per factor, (2) frozen-slot
  preservation, (3) selectivity cert. Numeric thresholds = the
  `goal.md` values, calibrated against Stage-0 oracle variance on the
  varied cut.
- A sim-free training and probe pipeline (M2a) reusing the existing
  varied cut as supervision.

**Explicitly NOT in scope (YAGNI):**

- **No pixel-input SlotEncoder.** M2a uses the 20-dim hand features as
  the encoder output `Z`; the "learn-from-pixels" claim is M2b and is
  spec'd separately if pursued.
- **No new intent factors, failure predicates, or revision operators.**
  M2 wraps the existing Stage-0 schema; it does not extend it.
- **No new tasks.** M2 runs on PushCube-v1 + StackCube-v1 varied cut.
  PickCube / TurnFaucet / CrossView fold in after M2a passes.
- **No learned attribution.** `failure.py`'s rule-based attribution
  remains the teacher (`goal.md` §"Stage 4 / Data Dependencies"); the
  learned attribution head is a later milestone.
- **No VLM/LLM** anywhere in `ReviseHead`. Whole-`G` regeneration is
  forbidden by interface (`goal.md` §"Stage 4 / Architecture"
  invariant 1).

---

## 2. Background — where we are after the 2026-05-23 fix

The M1.5 varied cut + the circular-angle fix together produce:

```
Cells: 12 total | 2 geometric (2 pass / 0 fail) | 2 label-identity | 8 trivially constant.
PushCube object_motion:  0.95 PASS
StackCube object_motion: 0.95 PASS
```

The pre-fix "0.72 FAIL is the M2 target" framing is **dead**: hand
features clear 0.95 on both geometric cells. The cert scaffold works
as designed, but it no longer demands a learned latent — it only
demands one that is **as good as** the hand features. So M2 has to
earn its existence one of two ways:

1. **Add structure that hand features lack** (M2a, this spec) — a
   factorized latent indexed by factor, with a slot-local revision
   interface that the existing `revision.py` cannot natively express
   in continuous space.
2. **Match the hand-feature ceiling from a harder input** (M2b, future
   spec) — encoder consumes `demo.rgbd_video` directly; same 0.95
   probe gate must hold without the privileged 2-D trajectory.

This spec covers M2a. M2b is in §6 only as a follow-up gate.

---

## 3. Architecture (concrete)

```
                       (sim-free; no Vulkan)
        +-----------+   Z (20-dim)     +------------+   G (F slots)
demo -->| features  |----------------->| IntentHead |--------+
record  |  (today)  |                  |  (new)     |        |
        +-----------+                  +------------+        |
                                                             v
                                          +---------------------+
                                          | ReviseHead(g^i, fp) |
                                          |  -> g̃^i  (new)     |
                                          +---------------------+
                                                             |
                                          +---------------------+
                                          | slot_decode(G)      |
                                          |  -> Intent JSON     |
                                          |  (table lookup)     |
                                          +---------------------+
                                                             |
                                                             v
                                          existing skills compile unchanged
```

**Shapes (M2a defaults; revisit at review):**

- `F = 6` — one slot per Stage-0 intent factor (matches
  `babysteps.schemas.INTENT_FIELDS`).
- `d_slot = 16` — each slot vector `g^i ∈ R^16`. Small on purpose;
  the cert pressure is on probe recoverability not capacity.
- `Z ∈ R^20` — current `extract_episode_features` output, frozen.
- `failure_packet_vector ∈ R^{F + |predicates|}` — a one-hot of
  the implicated factor (length `F = 6`) concatenated with a
  one-hot of the failure predicate (length = `|FAILURE_PREDICATES|`
  in `babysteps.schemas`; verify count at impl).
- `IntentHead`: 2-layer MLP `R^20 → R^{F · d_slot}` reshaped to
  `(F, d_slot)`. **GELU activation, no dropout** in M2a (small data).
- `ReviseHead`: 2-layer MLP `R^{d_slot + |fp|} → R^{d_slot}`. **Same
  ReviseHead is applied to every slot** — the slot index never enters
  the MLP except via `g^i`; this is what makes the interface
  slot-local (`goal.md` §"Stage 4 / Architecture" invariant 1).
- `slot_decode`: per-factor nearest-neighbor lookup against a table
  of training-time `(g^i, factor_value)` centroids. **Deterministic,
  no learned decoder weights** — this keeps the "learned latent /
  rule-based schema" boundary clean and makes the discrete intent
  always recoverable.

---

## 4. Acceptance Gates (matches `goal.md` Success Criteria)

The numeric thresholds are `goal.md`'s; ε and α calibrate against
Stage-0 oracle variance on the varied cut at impl time.

| Gate | Spec value | M2a Status target |
| --- | --- | --- |
| **G1 — Probe recoverability** per geometric factor | acc ≥ 0.90 | PushCube `object_motion` ≥ 0.90 **and** StackCube `object_motion` ≥ 0.90, both probed from `G` with a frozen `LogisticRegression` (same `babysteps.stage4.probe.train_probe`). |
| **G2 — Frozen-slot preservation** | ℓ2 drift of unedited slots ≤ ε | After `g^i ← g̃^i` for the implicated `i`, `‖g^j_revised − g^j_pre‖_2 ≤ ε ∀ j ≠ i`. ε = 99th percentile of natural per-episode drift of unedited slots across paired re-runs of the same seed. |
| **G3 — Selectivity cert** | paired t-test on counterfactual cross-slot drift, `p > α` | Pairs: (same `Z`, edit slot `i` vs. no edit). Measure: predicted future-slot drift on slots `j ≠ i`. Test: paired t against ε_sim, `α = 0.05`. |
| **G4 — Δpp vs failure-agnostic retry** | ≥ 10 | Δpp(M2a revised vs. `same_intent_retry` baseline from `project_baselines_m3` table) ≥ 10 on PushCube + StackCube. |
| **G5 — Δpp vs oracle discrete revision** | ≤ 5 (i.e. not worse than oracle by more than 5pp) | Δpp(M2a revised vs. `babysteps_selective` oracle) within −5pp on PushCube + StackCube. |

**M2a entrance gate:** G1 only (probe recoverability). G2–G5 are M2a
exit gates, evaluated only after G1 passes.

---

## 5. Three-stage rollout (single sequential plan, not parallel)

Each stage produces one PR-sized commit cluster with a focused
sim-free test suite (M2a is sim-free top to bottom — the input is the
existing `samples.jsonl`).

### Stage A1 — `IntentHead` + probe gate (G1)

- New: `babysteps/stage4/intent_head.py` (PyTorch nn.Module).
- New: `scripts/stage4_train_intent_head.py` (train + dump
  `G_episode.npz`).
- Reuse: `scripts/stage4_probe_schema_recoverability.py` with a new
  `--from-G G_episode.npz` flag that bypasses `extract_episode_features`.
- Acceptance: G1 ≥ 0.90 on PushCube **and** StackCube
  `object_motion`. Headline goal: meet 0.95 hand-feature parity.
- Tests: shape + determinism + a "label-identity" guard (probe must
  fail on shuffled-labels training).

### Stage A2 — `ReviseHead` + frozen-slot preservation (G2)

- New: `babysteps/stage4/revise_head.py`.
- New: `babysteps/stage4/slot_decode.py` (centroid lookup table built
  at A1 end-of-training).
- New: `scripts/stage4_eval_frozen_slot.py` (computes G2 metric).
- Acceptance: G2 ε calibrated; passes on PushCube + StackCube varied
  cut.
- Tests: type-signature test (ReviseHead consumes one slot, returns
  one slot — enforced by `typing` + assertions), monotonicity test
  (revising the same `(g^i, fp)` twice is idempotent within ε).

### Stage A3 — End-to-end gate (G3, G4, G5)

- Reuse: `babysteps/episode.py` with a new `latent_revision=True` mode
  that calls `slot_decode(G_revised)` instead of `revision.py`.
- New: `scripts/stage4_eval_e2e.py` reporting all five gates.
- Acceptance: G3 + G4 + G5 all pass.
- Tests: paired-rollout fixture (same seed, with/without revision)
  asserting G3 is computable.

---

## 6. Open decisions to settle BEFORE Stage A1

These are real design choices that change Stage A1's shape; the spec
recommends a default but flags the choice for the user.

1. **`F = 6` (one slot per factor) vs. `F = K` (one slot per
   object).** `goal.md` §"Architecture" specifies object-centric
   slots (`Z_t = {z_t^1, …, z_t^K}`); this spec deviates to
   factor-indexed slots because (a) M2a uses hand-features as `Z`
   which has no object grounding, and (b) the cert gate is per-factor.
   Object-centric slots return in M2b when `SlotEncoder` consumes
   pixels. **Risk:** "factor slot" reads as a category error to
   reviewers familiar with object-centric papers. **Mitigation:**
   document the M2a → M2b transition explicitly in the paper.
2. **`d_slot = 16`.** Picked for "small enough that a linear probe
   has limited room to memorize, large enough to carry direction +
   class info". Alternatives: 8 (tighter cert pressure) or 32
   (training stability). **Recommend 16** as the M2a default; revisit
   if A1 fails G1.
3. **`failure_packet_vector` encoding.** One-hot factor + one-hot
   predicate (proposed §3). Alternatives: include the
   `wrong_factor_value` string as a learnable embedding (would
   leak label structure into the revision input). **Recommend the
   one-hot-only version**; the value comes through `g^i`.
4. **Loss for `IntentHead` training.** Options: (a) supervise each
   slot with cross-entropy over its discrete factor value via a
   per-slot linear decoder (multi-head classification); (b)
   reconstruction-like contrastive loss between `(IntentHead(Z))` and
   a frozen target embedding of the Intent JSON. (a) is simpler and
   directly aligns slots with factors; (b) is more "latent". **Recommend
   (a) for M2a**; (b) is an A1 fallback if (a) collapses slots.
5. **`slot_decode` mechanism.** Centroid lookup (§3, the
   recommendation) is simplest. Alternative: a per-slot k-NN with k=3
   majority-vote. **Recommend centroid lookup**; flip to k-NN only if
   the centroid has < 1 sample per class on any factor.
6. **Whether to seed the `IntentHead` weights from `Z`'s hand-feature
   columns.** Zero-init MLP is fine for d=20→F·16=96 with N=60
   episodes. **Recommend random init with seed=0**, document the
   seed.

---

## 7. Risks & open questions

- **Data scarcity for ReviseHead.** N = 60 episodes (20 PushCube + 40
  StackCube) is small for a learned revision head. Most revisions are
  same-factor across the cut. M2a may need to **synthesize counter-
  factual training pairs** via the existing `revision.py` (apply each
  factor revision to each episode → labelled `(g^i, fp, g̃^i)` pair).
  Mark as Stage A2 task 0.
- **The factor-indexed slot deviation from `goal.md`.** Surfaced in
  §6 #1. If the user prefers object-indexed slots even in M2a,
  swap §3 `F = 6` → `F = K = 4` (max objects across both tasks) and
  treat each factor as a sum/concat over the K slots — the cert and
  revision interfaces still work but the design loses its
  "single slot per factor" cleanness.
- **G1 might be trivially passable** with a 20-dim → 96-dim MLP +
  per-slot linear decoder (the decoder is the cert probe). If
  `LogisticRegression` on `G` clears 1.00 ± 0.00, the cert is not
  measuring what we think. **Mitigation:** train the decoder and the
  probe on the **same** loss space but with **different random
  splits** — if probe acc collapses to majority on a held-out fold,
  G1 is not actually being met.
- **PushCube `object_motion` n=3 unique (incl. the singleton -y)**
  means LOO triggers; M2a numbers will inherit the noisy ± 0.22 std.
  Not a blocker but reviewers may ask. Optional cleanup: drop the
  PushCube -y singleton episode before training (n=19, 2 unique,
  cleaner number).
- **`embodiment_mapping` is trivially constant on PushCube +
  StackCube** — its slot has nothing to learn on this cut. **Plan:**
  initialise its slot to a per-task constant and skip its row in the
  cert table until a varied-embodiment cut exists (TurnFaucet-D track).

---

## 8. Test plan (sim-free, additive)

A1 minimum tests:
- `test_intent_head_shape`: input (B, 20) → output (B, F, d_slot).
- `test_intent_head_determinism`: same input + same seed → same output.
- `test_g1_probe_runs_on_synthetic`: synthetic `G_episode.npz` with
  perfectly separable per-slot labels → probe = 1.00.

A2 minimum tests:
- `test_revise_head_signature`: type check; single-slot in, single-
  slot out; refuses tensor of shape `(F, d_slot)`.
- `test_slot_decode_round_trip`: encode all training intents → build
  centroids → decode every training `G` → matches the source intent
  on every factor (training accuracy = 1.00 by construction).
- `test_g2_eps_calibrates`: ε ≥ 0 on the natural drift baseline.

A3 minimum tests:
- `test_latent_revision_episode_runs`: `episode.py` with
  `latent_revision=True` on a fake env runs to completion without
  touching `revision.py`'s discrete operators.

No new tests for the existing probe / report / features modules.
All M2a tests stay sim-free.

---

## 9. Out of scope (so reviewers know what NOT to expect)

- A PR diff > ~600 LOC (the spec is intentionally minimal).
- Any change to `babysteps/skills/*`, `babysteps/envs/*`, the
  Stage-0 episode JSONL format, or `goal.md`.
- A learned image encoder or any GPU dependency.
- Any new failure predicate or revision operator.
- The TurnFaucet, CrossView, or PickCube tasks (M2a is two-task).
- The Stage-1 human-demo bridge (M5; separate ICLR-track).

---

## 10. What "M2 is sufficiently specified but not yet safe to implement" means here

Sufficient: §1 (scope), §3 (architecture shapes), §4 (acceptance
gates), §5 (rollout) are concrete enough to start typing tests
without re-asking the user every step.

Not yet safe: §6 has 6 open decisions; §7 has 4 risks. **The user
should review and lock §6's recommendations** (or flip them) before
Stage A1 begins, because each one changes a shape Stage A1 freezes.
