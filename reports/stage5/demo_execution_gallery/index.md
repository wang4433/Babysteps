# BABYSTEPS — Demo & Execution Inspection Gallery

106 videos · 59 rows · 13 sections · key frames: first / mid / late(n-5) / final · every success/failure label is read back from the clip's own burned-in caption (the rendered outcome), not assumed.

> Open `index.html` for the full side-by-side visual gallery. Frames live under `frames/`+`strips/` (gitignored); regenerate with `python scripts/stage5_build_demo_execution_gallery.py`.

**Reading outcomes.** Every SUCCESS/FAILURE label is read back from the clip's own burned-in caption (font-exact template match), never assumed. Trustworthy for the method's real performance: the **Stage-0 PushCube** section (standard-panda, third-person) and the **measured June-3 intent decode** table. Sections marked **wristcam** use the `panda_wristcam` variant whose retry success flag is a documented artifact (reads False even when the cube reaches the target — see `renders/results/README.md`); those are flagged ⚠ and must not be read as method failures.

**Coverage.** This gallery extracts every unique-content MP4 under `renders/` and `datasets/`. Timestamped-subdir re-renders (identical seed sets) and the `official_demo_verify/` re-verification clips are skipped as duplicates of the parent-directory clips shown here.

## Stage-0 PushCube — blocked approach (standard-panda, third-person; trustworthy)

### PushCube-v1 · seed 0000 — revised `approach_direction` (stage0_pushcube_blocked) — **retry SUCCESS**

- **status:** attempt-1: blocked/failed (designed failure) · revised retry: SUCCESS  (read from burned-in caption)
- _Corroborated by aggregate report: initial 0% success, retry 100% success over n=5._
- **demo** (`pushcube_blocked_approach_seed_0000__1_demo.mp4`, n=13) — [▶ play mp4](../../../datasets/stage0_pushcube_blocked/videos_maniskill/pushcube_blocked_approach_seed_0000__1_demo.mp4)
  ![demo](strips/stage0_pushcube_blocked_0000_demo__strip.jpg)
- **attempt-1** (`pushcube_blocked_approach_seed_0000__2_attempt_blocked.mp4`, n=40) — [▶ play mp4](../../../datasets/stage0_pushcube_blocked/videos_maniskill/pushcube_blocked_approach_seed_0000__2_attempt_blocked.mp4)
  ![attempt-1](strips/stage0_pushcube_blocked_0000_attempt1__strip.jpg)
- **retry** (`pushcube_blocked_approach_seed_0000__3_retry.mp4`, n=13) — [▶ play mp4](../../../datasets/stage0_pushcube_blocked/videos_maniskill/pushcube_blocked_approach_seed_0000__3_retry.mp4)
  ![retry](strips/stage0_pushcube_blocked_0000_retry__strip.jpg)

### PushCube-v1 · seed 0001 — revised `approach_direction` (stage0_pushcube_blocked) — **retry SUCCESS**

- **status:** attempt-1: blocked/failed (designed failure) · revised retry: SUCCESS  (read from burned-in caption)
- _Corroborated by aggregate report: initial 0% success, retry 100% success over n=5._
- **demo** (`pushcube_blocked_approach_seed_0001__1_demo.mp4`, n=13) — [▶ play mp4](../../../datasets/stage0_pushcube_blocked/videos_maniskill/pushcube_blocked_approach_seed_0001__1_demo.mp4)
  ![demo](strips/stage0_pushcube_blocked_0001_demo__strip.jpg)
- **attempt-1** (`pushcube_blocked_approach_seed_0001__2_attempt_blocked.mp4`, n=40) — [▶ play mp4](../../../datasets/stage0_pushcube_blocked/videos_maniskill/pushcube_blocked_approach_seed_0001__2_attempt_blocked.mp4)
  ![attempt-1](strips/stage0_pushcube_blocked_0001_attempt1__strip.jpg)
- **retry** (`pushcube_blocked_approach_seed_0001__3_retry.mp4`, n=13) — [▶ play mp4](../../../datasets/stage0_pushcube_blocked/videos_maniskill/pushcube_blocked_approach_seed_0001__3_retry.mp4)
  ![retry](strips/stage0_pushcube_blocked_0001_retry__strip.jpg)

### PushCube-v1 · seed 0002 — revised `approach_direction` (stage0_pushcube_blocked) — **retry SUCCESS**

- **status:** attempt-1: blocked/failed (designed failure) · revised retry: SUCCESS  (read from burned-in caption)
- _Corroborated by aggregate report: initial 0% success, retry 100% success over n=5._
- **demo** (`pushcube_blocked_approach_seed_0002__1_demo.mp4`, n=11) — [▶ play mp4](../../../datasets/stage0_pushcube_blocked/videos_maniskill/pushcube_blocked_approach_seed_0002__1_demo.mp4)
  ![demo](strips/stage0_pushcube_blocked_0002_demo__strip.jpg)
- **attempt-1** (`pushcube_blocked_approach_seed_0002__2_attempt_blocked.mp4`, n=40) — [▶ play mp4](../../../datasets/stage0_pushcube_blocked/videos_maniskill/pushcube_blocked_approach_seed_0002__2_attempt_blocked.mp4)
  ![attempt-1](strips/stage0_pushcube_blocked_0002_attempt1__strip.jpg)
- **retry** (`pushcube_blocked_approach_seed_0002__3_retry.mp4`, n=11) — [▶ play mp4](../../../datasets/stage0_pushcube_blocked/videos_maniskill/pushcube_blocked_approach_seed_0002__3_retry.mp4)
  ![retry](strips/stage0_pushcube_blocked_0002_retry__strip.jpg)

### PushCube-v1 · seed 0003 — revised `approach_direction` (stage0_pushcube_blocked) — **retry SUCCESS**

- **status:** attempt-1: blocked/failed (designed failure) · revised retry: SUCCESS  (read from burned-in caption)
- _Corroborated by aggregate report: initial 0% success, retry 100% success over n=5._
- **demo** (`pushcube_blocked_approach_seed_0003__1_demo.mp4`, n=26) — [▶ play mp4](../../../datasets/stage0_pushcube_blocked/videos_maniskill/pushcube_blocked_approach_seed_0003__1_demo.mp4)
  ![demo](strips/stage0_pushcube_blocked_0003_demo__strip.jpg)
- **attempt-1** (`pushcube_blocked_approach_seed_0003__2_attempt_blocked.mp4`, n=40) — [▶ play mp4](../../../datasets/stage0_pushcube_blocked/videos_maniskill/pushcube_blocked_approach_seed_0003__2_attempt_blocked.mp4)
  ![attempt-1](strips/stage0_pushcube_blocked_0003_attempt1__strip.jpg)
- **retry** (`pushcube_blocked_approach_seed_0003__3_retry.mp4`, n=28) — [▶ play mp4](../../../datasets/stage0_pushcube_blocked/videos_maniskill/pushcube_blocked_approach_seed_0003__3_retry.mp4)
  ![retry](strips/stage0_pushcube_blocked_0003_retry__strip.jpg)

### PushCube-v1 · seed 0004 — revised `approach_direction` (stage0_pushcube_blocked) — **retry SUCCESS**

- **status:** attempt-1: blocked/failed (designed failure) · revised retry: SUCCESS  (read from burned-in caption)
- _Corroborated by aggregate report: initial 0% success, retry 100% success over n=5._
- **demo** (`pushcube_blocked_approach_seed_0004__1_demo.mp4`, n=12) — [▶ play mp4](../../../datasets/stage0_pushcube_blocked/videos_maniskill/pushcube_blocked_approach_seed_0004__1_demo.mp4)
  ![demo](strips/stage0_pushcube_blocked_0004_demo__strip.jpg)
- **attempt-1** (`pushcube_blocked_approach_seed_0004__2_attempt_blocked.mp4`, n=40) — [▶ play mp4](../../../datasets/stage0_pushcube_blocked/videos_maniskill/pushcube_blocked_approach_seed_0004__2_attempt_blocked.mp4)
  ![attempt-1](strips/stage0_pushcube_blocked_0004_attempt1__strip.jpg)
- **retry** (`pushcube_blocked_approach_seed_0004__3_retry.mp4`, n=12) — [▶ play mp4](../../../datasets/stage0_pushcube_blocked/videos_maniskill/pushcube_blocked_approach_seed_0004__3_retry.mp4)
  ![retry](strips/stage0_pushcube_blocked_0004_retry__strip.jpg)

## PushCube — curated three-phase clips (wristcam)

### PushCube-v1 · seed 0000 — revised `approach_direction` (pushcube_render) — **⚠ wristcam (flag unreliable)**

- **status:** attempt-1: blocked/failed (designed failure) · revised retry: FAILURE  (read from burned-in caption)
- **⚠ First-person panda_wristcam render: the burned-in retry success flag is UNRELIABLE on this variant (reads False even when the cube reaches the target — a documented controller×wristcam artifact, see renders/results/README.md). Trust the standard-panda Stage-0 PushCube section and the measured June-3 eval (latent retry success 0.96) for the real outcome, not these flags.**
- **demo** (`pushcube_blocked_approach_seed_0000__1_demo.mp4`, n=51) — [▶ play mp4](../../../renders/pushcube/videos_maniskill/pushcube_blocked_approach_seed_0000__1_demo.mp4)
  ![demo](strips/pushcube_render_0000_demo__strip.jpg)
- **attempt-1** (`pushcube_blocked_approach_seed_0000__2_attempt_blocked.mp4`, n=51) — [▶ play mp4](../../../renders/pushcube/videos_maniskill/pushcube_blocked_approach_seed_0000__2_attempt_blocked.mp4)
  ![attempt-1](strips/pushcube_render_0000_attempt1__strip.jpg)
- **retry** (`pushcube_blocked_approach_seed_0000__3_retry.mp4`, n=51) — [▶ play mp4](../../../renders/pushcube/videos_maniskill/pushcube_blocked_approach_seed_0000__3_retry.mp4)
  ![retry](strips/pushcube_render_0000_retry__strip.jpg)

### PushCube-v1 · seed 0001 — revised `approach_direction` (pushcube_render) — **⚠ wristcam (flag unreliable)**

- **status:** attempt-1: blocked/failed (designed failure) · revised retry: FAILURE  (read from burned-in caption)
- **⚠ First-person panda_wristcam render: the burned-in retry success flag is UNRELIABLE on this variant (reads False even when the cube reaches the target — a documented controller×wristcam artifact, see renders/results/README.md). Trust the standard-panda Stage-0 PushCube section and the measured June-3 eval (latent retry success 0.96) for the real outcome, not these flags.**
- **demo** (`pushcube_blocked_approach_seed_0001__1_demo.mp4`, n=51) — [▶ play mp4](../../../renders/pushcube/videos_maniskill/pushcube_blocked_approach_seed_0001__1_demo.mp4)
  ![demo](strips/pushcube_render_0001_demo__strip.jpg)
- **attempt-1** (`pushcube_blocked_approach_seed_0001__2_attempt_blocked.mp4`, n=51) — [▶ play mp4](../../../renders/pushcube/videos_maniskill/pushcube_blocked_approach_seed_0001__2_attempt_blocked.mp4)
  ![attempt-1](strips/pushcube_render_0001_attempt1__strip.jpg)
- **retry** (`pushcube_blocked_approach_seed_0001__3_retry.mp4`, n=51) — [▶ play mp4](../../../renders/pushcube/videos_maniskill/pushcube_blocked_approach_seed_0001__3_retry.mp4)
  ![retry](strips/pushcube_render_0001_retry__strip.jpg)

## PushCube — clutter ablation clips (wristcam)

### PushCube-v1 · seed 0000 — revised `approach_direction` (pushcube_clutter) — **⚠ wristcam (flag unreliable)**

- **status:** attempt-1: blocked/failed (designed failure) · revised retry: FAILURE  (read from burned-in caption)
- **⚠ First-person panda_wristcam render: the burned-in retry success flag is UNRELIABLE on this variant (reads False even when the cube reaches the target — a documented controller×wristcam artifact, see renders/results/README.md). Trust the standard-panda Stage-0 PushCube section and the measured June-3 eval (latent retry success 0.96) for the real outcome, not these flags.**
- **demo** (`pushcube_blocked_approach_seed_0000__1_demo.mp4`, n=51) — [▶ play mp4](../../../renders/pushcube_clutter/videos_maniskill/pushcube_blocked_approach_seed_0000__1_demo.mp4)
  ![demo](strips/pushcube_clutter_0000_demo__strip.jpg)
- **attempt-1** (`pushcube_blocked_approach_seed_0000__2_attempt_blocked.mp4`, n=51) — [▶ play mp4](../../../renders/pushcube_clutter/videos_maniskill/pushcube_blocked_approach_seed_0000__2_attempt_blocked.mp4)
  ![attempt-1](strips/pushcube_clutter_0000_attempt1__strip.jpg)
- **retry** (`pushcube_blocked_approach_seed_0000__3_retry.mp4`, n=51) — [▶ play mp4](../../../renders/pushcube_clutter/videos_maniskill/pushcube_blocked_approach_seed_0000__3_retry.mp4)
  ![retry](strips/pushcube_clutter_0000_retry__strip.jpg)

### PushCube-v1 · seed 0001 — revised `approach_direction` (pushcube_clutter) — **⚠ wristcam (flag unreliable)**

- **status:** attempt-1: blocked/failed (designed failure) · revised retry: FAILURE  (read from burned-in caption)
- **⚠ First-person panda_wristcam render: the burned-in retry success flag is UNRELIABLE on this variant (reads False even when the cube reaches the target — a documented controller×wristcam artifact, see renders/results/README.md). Trust the standard-panda Stage-0 PushCube section and the measured June-3 eval (latent retry success 0.96) for the real outcome, not these flags.**
- **demo** (`pushcube_blocked_approach_seed_0001__1_demo.mp4`, n=51) — [▶ play mp4](../../../renders/pushcube_clutter/videos_maniskill/pushcube_blocked_approach_seed_0001__1_demo.mp4)
  ![demo](strips/pushcube_clutter_0001_demo__strip.jpg)
- **attempt-1** (`pushcube_blocked_approach_seed_0001__2_attempt_blocked.mp4`, n=51) — [▶ play mp4](../../../renders/pushcube_clutter/videos_maniskill/pushcube_blocked_approach_seed_0001__2_attempt_blocked.mp4)
  ![attempt-1](strips/pushcube_clutter_0001_attempt1__strip.jpg)
- **retry** (`pushcube_blocked_approach_seed_0001__3_retry.mp4`, n=51) — [▶ play mp4](../../../renders/pushcube_clutter/videos_maniskill/pushcube_blocked_approach_seed_0001__3_retry.mp4)
  ![retry](strips/pushcube_clutter_0001_retry__strip.jpg)

### PushCube-v1 · seed 0002 — revised `approach_direction` (pushcube_clutter) — **⚠ wristcam (flag unreliable)**

- **status:** attempt-1: blocked/failed (designed failure) · revised retry: FAILURE  (read from burned-in caption)
- **⚠ First-person panda_wristcam render: the burned-in retry success flag is UNRELIABLE on this variant (reads False even when the cube reaches the target — a documented controller×wristcam artifact, see renders/results/README.md). Trust the standard-panda Stage-0 PushCube section and the measured June-3 eval (latent retry success 0.96) for the real outcome, not these flags.**
- **demo** (`pushcube_blocked_approach_seed_0002__1_demo.mp4`, n=51) — [▶ play mp4](../../../renders/pushcube_clutter/videos_maniskill/pushcube_blocked_approach_seed_0002__1_demo.mp4)
  ![demo](strips/pushcube_clutter_0002_demo__strip.jpg)
- **attempt-1** (`pushcube_blocked_approach_seed_0002__2_attempt_blocked.mp4`, n=51) — [▶ play mp4](../../../renders/pushcube_clutter/videos_maniskill/pushcube_blocked_approach_seed_0002__2_attempt_blocked.mp4)
  ![attempt-1](strips/pushcube_clutter_0002_attempt1__strip.jpg)
- **retry** (`pushcube_blocked_approach_seed_0002__3_retry.mp4`, n=51) — [▶ play mp4](../../../renders/pushcube_clutter/videos_maniskill/pushcube_blocked_approach_seed_0002__3_retry.mp4)
  ![retry](strips/pushcube_clutter_0002_retry__strip.jpg)

## PushCube — paper figure (contact_region, wrong-intent attempt, wristcam)

### PushCube-v1 · seed 0000 — revised `contact_region` (pushcube_paper_figure) — **⚠ wristcam (flag unreliable)**

- **status:** attempt-1: blocked/failed (designed failure) · revised retry: FAILURE  (read from burned-in caption)
- **⚠ First-person panda_wristcam render: the burned-in retry success flag is UNRELIABLE on this variant (reads False even when the cube reaches the target — a documented controller×wristcam artifact, see renders/results/README.md). Trust the standard-panda Stage-0 PushCube section and the measured June-3 eval (latent retry success 0.96) for the real outcome, not these flags.**
- **demo** (`pushcube_blocked_approach_seed_0000__1_demo.mp4`, n=51) — [▶ play mp4](../../../renders/pushcube/videos_paper_figure/2026-05-28_164447/pushcube_blocked_approach_seed_0000__1_demo.mp4)
  ![demo](strips/pushcube_paper_figure_0000_demo__strip.jpg)
- **attempt-1** (`pushcube_blocked_approach_seed_0000__2_attempt_wrong_intent.mp4`, n=51) — [▶ play mp4](../../../renders/pushcube/videos_paper_figure/2026-05-28_164447/pushcube_blocked_approach_seed_0000__2_attempt_wrong_intent.mp4)
  ![attempt-1](strips/pushcube_paper_figure_0000_attempt1__strip.jpg)
- **retry** (`pushcube_blocked_approach_seed_0000__3_retry.mp4`, n=51) — [▶ play mp4](../../../renders/pushcube/videos_paper_figure/2026-05-28_164447/pushcube_blocked_approach_seed_0000__3_retry.mp4)
  ![retry](strips/pushcube_paper_figure_0000_retry__strip.jpg)

### PushCube-v1 · seed 0001 — revised `contact_region` (pushcube_paper_figure) — **⚠ wristcam (flag unreliable)**

- **status:** attempt-1: blocked/failed (designed failure) · revised retry: FAILURE  (read from burned-in caption)
- **⚠ First-person panda_wristcam render: the burned-in retry success flag is UNRELIABLE on this variant (reads False even when the cube reaches the target — a documented controller×wristcam artifact, see renders/results/README.md). Trust the standard-panda Stage-0 PushCube section and the measured June-3 eval (latent retry success 0.96) for the real outcome, not these flags.**
- **demo** (`pushcube_blocked_approach_seed_0001__1_demo.mp4`, n=51) — [▶ play mp4](../../../renders/pushcube/videos_paper_figure/2026-05-28_164447/pushcube_blocked_approach_seed_0001__1_demo.mp4)
  ![demo](strips/pushcube_paper_figure_0001_demo__strip.jpg)
- **attempt-1** (`pushcube_blocked_approach_seed_0001__2_attempt_wrong_intent.mp4`, n=51) — [▶ play mp4](../../../renders/pushcube/videos_paper_figure/2026-05-28_164447/pushcube_blocked_approach_seed_0001__2_attempt_wrong_intent.mp4)
  ![attempt-1](strips/pushcube_paper_figure_0001_attempt1__strip.jpg)
- **retry** (`pushcube_blocked_approach_seed_0001__3_retry.mp4`, n=51) — [▶ play mp4](../../../renders/pushcube/videos_paper_figure/2026-05-28_164447/pushcube_blocked_approach_seed_0001__3_retry.mp4)
  ![retry](strips/pushcube_paper_figure_0001_retry__strip.jpg)

### PushCube-v1 · seed 0002 — revised `contact_region` (pushcube_paper_figure) — **⚠ wristcam (flag unreliable)**

- **status:** attempt-1: blocked/failed (designed failure) · revised retry: FAILURE  (read from burned-in caption)
- **⚠ First-person panda_wristcam render: the burned-in retry success flag is UNRELIABLE on this variant (reads False even when the cube reaches the target — a documented controller×wristcam artifact, see renders/results/README.md). Trust the standard-panda Stage-0 PushCube section and the measured June-3 eval (latent retry success 0.96) for the real outcome, not these flags.**
- **demo** (`pushcube_blocked_approach_seed_0002__1_demo.mp4`, n=51) — [▶ play mp4](../../../renders/pushcube/videos_paper_figure/2026-05-28_164447/pushcube_blocked_approach_seed_0002__1_demo.mp4)
  ![demo](strips/pushcube_paper_figure_0002_demo__strip.jpg)
- **attempt-1** (`pushcube_blocked_approach_seed_0002__2_attempt_wrong_intent.mp4`, n=51) — [▶ play mp4](../../../renders/pushcube/videos_paper_figure/2026-05-28_164447/pushcube_blocked_approach_seed_0002__2_attempt_wrong_intent.mp4)
  ![attempt-1](strips/pushcube_paper_figure_0002_attempt1__strip.jpg)
- **retry** (`pushcube_blocked_approach_seed_0002__3_retry.mp4`, n=51) — [▶ play mp4](../../../renders/pushcube/videos_paper_figure/2026-05-28_164447/pushcube_blocked_approach_seed_0002__3_retry.mp4)
  ![retry](strips/pushcube_paper_figure_0002_retry__strip.jpg)

### PushCube-v1 · seed 0003 — revised `contact_region` (pushcube_paper_figure) — **⚠ wristcam (flag unreliable)**

- **status:** attempt-1: blocked/failed (designed failure) · revised retry: FAILURE  (read from burned-in caption)
- **⚠ First-person panda_wristcam render: the burned-in retry success flag is UNRELIABLE on this variant (reads False even when the cube reaches the target — a documented controller×wristcam artifact, see renders/results/README.md). Trust the standard-panda Stage-0 PushCube section and the measured June-3 eval (latent retry success 0.96) for the real outcome, not these flags.**
- **demo** (`pushcube_blocked_approach_seed_0003__1_demo.mp4`, n=51) — [▶ play mp4](../../../renders/pushcube/videos_paper_figure/2026-05-28_164447/pushcube_blocked_approach_seed_0003__1_demo.mp4)
  ![demo](strips/pushcube_paper_figure_0003_demo__strip.jpg)
- **attempt-1** (`pushcube_blocked_approach_seed_0003__2_attempt_wrong_intent.mp4`, n=51) — [▶ play mp4](../../../renders/pushcube/videos_paper_figure/2026-05-28_164447/pushcube_blocked_approach_seed_0003__2_attempt_wrong_intent.mp4)
  ![attempt-1](strips/pushcube_paper_figure_0003_attempt1__strip.jpg)
- **retry** (`pushcube_blocked_approach_seed_0003__3_retry.mp4`, n=51) — [▶ play mp4](../../../renders/pushcube/videos_paper_figure/2026-05-28_164447/pushcube_blocked_approach_seed_0003__3_retry.mp4)
  ![retry](strips/pushcube_paper_figure_0003_retry__strip.jpg)

### PushCube-v1 · seed 0004 — revised `contact_region` (pushcube_paper_figure) — **⚠ wristcam (flag unreliable)**

- **status:** attempt-1: blocked/failed (designed failure) · revised retry: FAILURE  (read from burned-in caption)
- **⚠ First-person panda_wristcam render: the burned-in retry success flag is UNRELIABLE on this variant (reads False even when the cube reaches the target — a documented controller×wristcam artifact, see renders/results/README.md). Trust the standard-panda Stage-0 PushCube section and the measured June-3 eval (latent retry success 0.96) for the real outcome, not these flags.**
- **demo** (`pushcube_blocked_approach_seed_0004__1_demo.mp4`, n=51) — [▶ play mp4](../../../renders/pushcube/videos_paper_figure/2026-05-28_164447/pushcube_blocked_approach_seed_0004__1_demo.mp4)
  ![demo](strips/pushcube_paper_figure_0004_demo__strip.jpg)
- **attempt-1** (`pushcube_blocked_approach_seed_0004__2_attempt_wrong_intent.mp4`, n=51) — [▶ play mp4](../../../renders/pushcube/videos_paper_figure/2026-05-28_164447/pushcube_blocked_approach_seed_0004__2_attempt_wrong_intent.mp4)
  ![attempt-1](strips/pushcube_paper_figure_0004_attempt1__strip.jpg)
- **retry** (`pushcube_blocked_approach_seed_0004__3_retry.mp4`, n=51) — [▶ play mp4](../../../renders/pushcube/videos_paper_figure/2026-05-28_164447/pushcube_blocked_approach_seed_0004__3_retry.mp4)
  ![retry](strips/pushcube_paper_figure_0004_retry__strip.jpg)

## PickCube — grasp slip (contact_region)

### PickCube-v1 · seed 0000 — revised `contact_region` (pickcube_render) — **retry FAILURE**

- **status:** attempt-1: blocked/failed (designed failure) · revised retry: FAILURE  (read from burned-in caption)
- **demo** (`pickcube_grasp_slip_seed_0000__1_demo.mp4`, n=15) — [▶ play mp4](../../../renders/pickcube/videos_maniskill/pickcube_grasp_slip_seed_0000__1_demo.mp4)
  ![demo](strips/pickcube_render_0000_demo__strip.jpg)
- **attempt-1** (`pickcube_grasp_slip_seed_0000__2_attempt_blocked.mp4`, n=35) — [▶ play mp4](../../../renders/pickcube/videos_maniskill/pickcube_grasp_slip_seed_0000__2_attempt_blocked.mp4)
  ![attempt-1](strips/pickcube_render_0000_attempt1__strip.jpg)
- **retry** (`pickcube_grasp_slip_seed_0000__3_retry.mp4`, n=15) — [▶ play mp4](../../../renders/pickcube/videos_maniskill/pickcube_grasp_slip_seed_0000__3_retry.mp4)
  ![retry](strips/pickcube_render_0000_retry__strip.jpg)

### PickCube-v1 · seed 0001 — revised `contact_region` (pickcube_render) — **retry FAILURE**

- **status:** attempt-1: blocked/failed (designed failure) · revised retry: FAILURE  (read from burned-in caption)
- **demo** (`pickcube_grasp_slip_seed_0001__1_demo.mp4`, n=16) — [▶ play mp4](../../../renders/pickcube/videos_maniskill/pickcube_grasp_slip_seed_0001__1_demo.mp4)
  ![demo](strips/pickcube_render_0001_demo__strip.jpg)
- **attempt-1** (`pickcube_grasp_slip_seed_0001__2_attempt_blocked.mp4`, n=36) — [▶ play mp4](../../../renders/pickcube/videos_maniskill/pickcube_grasp_slip_seed_0001__2_attempt_blocked.mp4)
  ![attempt-1](strips/pickcube_render_0001_attempt1__strip.jpg)
- **retry** (`pickcube_grasp_slip_seed_0001__3_retry.mp4`, n=16) — [▶ play mp4](../../../renders/pickcube/videos_maniskill/pickcube_grasp_slip_seed_0001__3_retry.mp4)
  ![retry](strips/pickcube_render_0001_retry__strip.jpg)

## StackCube — underspecified goal (goal_state)

### StackCube-v1 · seed 0000 — revised `goal_state` (stackcube_render) — **retry SUCCESS**

- **status:** attempt-1: blocked/failed (designed failure) · revised retry: SUCCESS  (read from burned-in caption)
- **demo** (`stackcube_underspec_goal_seed_0000__1_demo.mp4`, n=30) — [▶ play mp4](../../../renders/stackcube/videos_maniskill/stackcube_underspec_goal_seed_0000__1_demo.mp4)
  ![demo](strips/stackcube_render_0000_demo__strip.jpg)
- **attempt-1** (`stackcube_underspec_goal_seed_0000__2_attempt_blocked.mp4`, n=68) — [▶ play mp4](../../../renders/stackcube/videos_maniskill/stackcube_underspec_goal_seed_0000__2_attempt_blocked.mp4)
  ![attempt-1](strips/stackcube_render_0000_attempt1__strip.jpg)
- **retry** (`stackcube_underspec_goal_seed_0000__3_retry.mp4`, n=30) — [▶ play mp4](../../../renders/stackcube/videos_maniskill/stackcube_underspec_goal_seed_0000__3_retry.mp4)
  ![retry](strips/stackcube_render_0000_retry__strip.jpg)

### StackCube-v1 · seed 0001 — revised `goal_state` (stackcube_render) — **retry SUCCESS**

- **status:** attempt-1: blocked/failed (designed failure) · revised retry: SUCCESS  (read from burned-in caption)
- **demo** (`stackcube_underspec_goal_seed_0001__1_demo.mp4`, n=24) — [▶ play mp4](../../../renders/stackcube/videos_maniskill/stackcube_underspec_goal_seed_0001__1_demo.mp4)
  ![demo](strips/stackcube_render_0001_demo__strip.jpg)
- **attempt-1** (`stackcube_underspec_goal_seed_0001__2_attempt_blocked.mp4`, n=63) — [▶ play mp4](../../../renders/stackcube/videos_maniskill/stackcube_underspec_goal_seed_0001__2_attempt_blocked.mp4)
  ![attempt-1](strips/stackcube_render_0001_attempt1__strip.jpg)
- **retry** (`stackcube_underspec_goal_seed_0001__3_retry.mp4`, n=24) — [▶ play mp4](../../../renders/stackcube/videos_maniskill/stackcube_underspec_goal_seed_0001__3_retry.mp4)
  ![retry](strips/stackcube_render_0001_retry__strip.jpg)

## TurnFaucet — wrong contact (grasp-turn lineage)

### TurnFaucet-v1 · seed 0000 — revised `embodiment_mapping` (turnfaucet_render) — **retry FAILURE**

- **status:** attempt-1: blocked/failed (designed failure) · revised retry: FAILURE  (read from burned-in caption)
- **demo** (`turnfaucet_wrong_contact_seed_0000__1_demo.mp4`, n=28) — [▶ play mp4](../../../renders/turnfaucet/videos_maniskill/turnfaucet_wrong_contact_seed_0000__1_demo.mp4)
  ![demo](strips/turnfaucet_render_0000_demo__strip.jpg)
- **attempt-1** (`turnfaucet_wrong_contact_seed_0000__2_attempt_blocked.mp4`, n=139) — [▶ play mp4](../../../renders/turnfaucet/videos_maniskill/turnfaucet_wrong_contact_seed_0000__2_attempt_blocked.mp4)
  ![attempt-1](strips/turnfaucet_render_0000_attempt1__strip.jpg)
- **retry** (`turnfaucet_wrong_contact_seed_0000__3_retry.mp4`, n=28) — [▶ play mp4](../../../renders/turnfaucet/videos_maniskill/turnfaucet_wrong_contact_seed_0000__3_retry.mp4)
  ![retry](strips/turnfaucet_render_0000_retry__strip.jpg)

### TurnFaucet-v1 · seed 0001 — revised `embodiment_mapping` (turnfaucet_render) — **retry FAILURE**

- **status:** attempt-1: blocked/failed (designed failure) · revised retry: FAILURE  (read from burned-in caption)
- **demo** (`turnfaucet_wrong_contact_seed_0001__1_demo.mp4`, n=61) — [▶ play mp4](../../../renders/turnfaucet/videos_maniskill/turnfaucet_wrong_contact_seed_0001__1_demo.mp4)
  ![demo](strips/turnfaucet_render_0001_demo__strip.jpg)
- **attempt-1** (`turnfaucet_wrong_contact_seed_0001__2_attempt_blocked.mp4`, n=221) — [▶ play mp4](../../../renders/turnfaucet/videos_maniskill/turnfaucet_wrong_contact_seed_0001__2_attempt_blocked.mp4)
  ![attempt-1](strips/turnfaucet_render_0001_attempt1__strip.jpg)
- **retry** (`turnfaucet_wrong_contact_seed_0001__3_retry.mp4`, n=61) — [▶ play mp4](../../../renders/turnfaucet/videos_maniskill/turnfaucet_wrong_contact_seed_0001__3_retry.mp4)
  ![retry](strips/turnfaucet_render_0001_retry__strip.jpg)

## TurnFaucet — embodiment substitution

### TurnFaucet-v1 · seed 0000 — revised `embodiment_mapping` (turnfaucet_embodiment) — **retry FAILURE**

- **status:** attempt-1: blocked/failed (designed failure) · revised retry: FAILURE  (read from burned-in caption)
- **demo** (`turnfaucet_wrong_contact_seed_0000__1_demo.mp4`, n=40) — [▶ play mp4](../../../renders/turnfaucet_embodiment/videos_maniskill/turnfaucet_wrong_contact_seed_0000__1_demo.mp4)
  ![demo](strips/turnfaucet_embodiment_0000_demo__strip.jpg)
- **attempt-1** (`turnfaucet_wrong_contact_seed_0000__2_attempt_blocked.mp4`, n=221) — [▶ play mp4](../../../renders/turnfaucet_embodiment/videos_maniskill/turnfaucet_wrong_contact_seed_0000__2_attempt_blocked.mp4)
  ![attempt-1](strips/turnfaucet_embodiment_0000_attempt1__strip.jpg)
- **retry** (`turnfaucet_wrong_contact_seed_0000__3_retry.mp4`, n=201) — [▶ play mp4](../../../renders/turnfaucet_embodiment/videos_maniskill/turnfaucet_wrong_contact_seed_0000__3_retry.mp4)
  ![retry](strips/turnfaucet_embodiment_0000_retry__strip.jpg)

### TurnFaucet-v1 · seed 0001 — revised `embodiment_mapping` (turnfaucet_embodiment) — **retry FAILURE**

- **status:** attempt-1: blocked/failed (designed failure) · revised retry: FAILURE  (read from burned-in caption)
- **demo** (`turnfaucet_wrong_contact_seed_0001__1_demo.mp4`, n=40) — [▶ play mp4](../../../renders/turnfaucet_embodiment/videos_maniskill/turnfaucet_wrong_contact_seed_0001__1_demo.mp4)
  ![demo](strips/turnfaucet_embodiment_0001_demo__strip.jpg)
- **attempt-1** (`turnfaucet_wrong_contact_seed_0001__2_attempt_blocked.mp4`, n=90) — [▶ play mp4](../../../renders/turnfaucet_embodiment/videos_maniskill/turnfaucet_wrong_contact_seed_0001__2_attempt_blocked.mp4)
  ![attempt-1](strips/turnfaucet_embodiment_0001_attempt1__strip.jpg)
- **retry** (`turnfaucet_wrong_contact_seed_0001__3_retry.mp4`, n=19) — [▶ play mp4](../../../renders/turnfaucet_embodiment/videos_maniskill/turnfaucet_wrong_contact_seed_0001__3_retry.mp4)
  ![retry](strips/turnfaucet_embodiment_0001_retry__strip.jpg)

### TurnFaucet-v1 · seed 0002 — revised `embodiment_mapping` (turnfaucet_embodiment) — **retry SUCCESS**

- **status:** attempt-1: blocked/failed (designed failure) · revised retry: SUCCESS  (read from burned-in caption)
- **demo** (`turnfaucet_wrong_contact_seed_0002__1_demo.mp4`, n=40) — [▶ play mp4](../../../renders/turnfaucet_embodiment/videos_maniskill/turnfaucet_wrong_contact_seed_0002__1_demo.mp4)
  ![demo](strips/turnfaucet_embodiment_0002_demo__strip.jpg)
- **attempt-1** (`turnfaucet_wrong_contact_seed_0002__2_attempt_blocked.mp4`, n=23) — [▶ play mp4](../../../renders/turnfaucet_embodiment/videos_maniskill/turnfaucet_wrong_contact_seed_0002__2_attempt_blocked.mp4)
  ![attempt-1](strips/turnfaucet_embodiment_0002_attempt1__strip.jpg)
- **retry** (`turnfaucet_wrong_contact_seed_0002__3_retry.mp4`, n=3) — [▶ play mp4](../../../renders/turnfaucet_embodiment/videos_maniskill/turnfaucet_wrong_contact_seed_0002__3_retry.mp4)
  ![retry](strips/turnfaucet_embodiment_0002_retry__strip.jpg)

### TurnFaucet-v1 · seed 0003 — revised `embodiment_mapping` (turnfaucet_embodiment) — **retry FAILURE**

- **status:** attempt-1: blocked/failed (designed failure) · revised retry: FAILURE  (read from burned-in caption)
- **demo** (`turnfaucet_wrong_contact_seed_0003__1_demo.mp4`, n=40) — [▶ play mp4](../../../renders/turnfaucet_embodiment/videos_maniskill/turnfaucet_wrong_contact_seed_0003__1_demo.mp4)
  ![demo](strips/turnfaucet_embodiment_0003_demo__strip.jpg)
- **attempt-1** (`turnfaucet_wrong_contact_seed_0003__2_attempt_blocked.mp4`, n=221) — [▶ play mp4](../../../renders/turnfaucet_embodiment/videos_maniskill/turnfaucet_wrong_contact_seed_0003__2_attempt_blocked.mp4)
  ![attempt-1](strips/turnfaucet_embodiment_0003_attempt1__strip.jpg)
- **retry** (`turnfaucet_wrong_contact_seed_0003__3_retry.mp4`, n=25) — [▶ play mp4](../../../renders/turnfaucet_embodiment/videos_maniskill/turnfaucet_wrong_contact_seed_0003__3_retry.mp4)
  ![retry](strips/turnfaucet_embodiment_0003_retry__strip.jpg)

### TurnFaucet-v1 · seed 0004 — revised `embodiment_mapping` (turnfaucet_embodiment) — **retry FAILURE**

- **status:** attempt-1: blocked/failed (designed failure) · revised retry: FAILURE  (read from burned-in caption)
- **demo** (`turnfaucet_wrong_contact_seed_0004__1_demo.mp4`, n=40) — [▶ play mp4](../../../renders/turnfaucet_embodiment/videos_maniskill/turnfaucet_wrong_contact_seed_0004__1_demo.mp4)
  ![demo](strips/turnfaucet_embodiment_0004_demo__strip.jpg)
- **attempt-1** (`turnfaucet_wrong_contact_seed_0004__2_attempt_blocked.mp4`, n=221) — [▶ play mp4](../../../renders/turnfaucet_embodiment/videos_maniskill/turnfaucet_wrong_contact_seed_0004__2_attempt_blocked.mp4)
  ![attempt-1](strips/turnfaucet_embodiment_0004_attempt1__strip.jpg)
- **retry** (`turnfaucet_wrong_contact_seed_0004__3_retry.mp4`, n=201) — [▶ play mp4](../../../renders/turnfaucet_embodiment/videos_maniskill/turnfaucet_wrong_contact_seed_0004__3_retry.mp4)
  ![retry](strips/turnfaucet_embodiment_0004_retry__strip.jpg)

## PushCube latent loop — iconic composites (wristcam; flags unreliable)

### seed 0100 — latent — **⚠ wristcam (flag unreliable)**

- Latent input + learned slot-local edit (BabySteps, the method)
- **status:** FAILURE (read from burned-in caption; this is the May-24 iconic render)
- **⚠ First-person panda_wristcam render: the burned-in retry success flag is UNRELIABLE on this variant (reads False even when the cube reaches the target — a documented controller×wristcam artifact, see renders/results/README.md). Trust the standard-panda Stage-0 PushCube section and the measured June-3 eval (latent retry success 0.96) for the real outcome, not these flags.**
- [▶ play full clip mp4](../../../renders/stage5_p1_iconic/pushcube/pushcube_seed_0100__latent_full.mp4)
  ![strip](strips/iconic_0100_latent__strip.jpg)

### seed 0100 — babysteps_selective — **⚠ wristcam (flag unreliable)**

- Oracle input + learned selective edit
- **status:** FAILURE (read from burned-in caption; this is the May-24 iconic render)
- **⚠ First-person panda_wristcam render: the burned-in retry success flag is UNRELIABLE on this variant (reads False even when the cube reaches the target — a documented controller×wristcam artifact, see renders/results/README.md). Trust the standard-panda Stage-0 PushCube section and the measured June-3 eval (latent retry success 0.96) for the real outcome, not these flags.**
- [▶ play full clip mp4](../../../renders/stage5_p1_iconic/pushcube/pushcube_seed_0100__babysteps_selective_full.mp4)
  ![strip](strips/iconic_0100_babysteps_selective__strip.jpg)

### seed 0100 — oracle_factor_revision — **⚠ wristcam (flag unreliable)**

- Oracle single-factor revision (skyline)
- **status:** FAILURE (read from burned-in caption; this is the May-24 iconic render)
- **⚠ First-person panda_wristcam render: the burned-in retry success flag is UNRELIABLE on this variant (reads False even when the cube reaches the target — a documented controller×wristcam artifact, see renders/results/README.md). Trust the standard-panda Stage-0 PushCube section and the measured June-3 eval (latent retry success 0.96) for the real outcome, not these flags.**
- [▶ play full clip mp4](../../../renders/stage5_p1_iconic/pushcube/pushcube_seed_0100__oracle_factor_revision_full.mp4)
  ![strip](strips/iconic_0100_oracle_factor_revision__strip.jpg)

### seed 0100 — same_intent_retry — **⚠ wristcam (flag unreliable)**

- Retry with unchanged intent (control — must fail)
- **status:** FAILURE (read from burned-in caption; this is the May-24 iconic render)
- **⚠ First-person panda_wristcam render: the burned-in retry success flag is UNRELIABLE on this variant (reads False even when the cube reaches the target — a documented controller×wristcam artifact, see renders/results/README.md). Trust the standard-panda Stage-0 PushCube section and the measured June-3 eval (latent retry success 0.96) for the real outcome, not these flags.**
- [▶ play full clip mp4](../../../renders/stage5_p1_iconic/pushcube/pushcube_seed_0100__same_intent_retry_full.mp4)
  ![strip](strips/iconic_0100_same_intent_retry__strip.jpg)

### seed 0110 — latent — **⚠ wristcam (flag unreliable)**

- Latent input + learned slot-local edit (BabySteps, the method)
- **status:** FAILURE (read from burned-in caption; this is the May-24 iconic render)
- **⚠ First-person panda_wristcam render: the burned-in retry success flag is UNRELIABLE on this variant (reads False even when the cube reaches the target — a documented controller×wristcam artifact, see renders/results/README.md). Trust the standard-panda Stage-0 PushCube section and the measured June-3 eval (latent retry success 0.96) for the real outcome, not these flags.**
- [▶ play full clip mp4](../../../renders/stage5_p1_iconic/pushcube/pushcube_seed_0110__latent_full.mp4)
  ![strip](strips/iconic_0110_latent__strip.jpg)

### seed 0110 — babysteps_selective — **⚠ wristcam (flag unreliable)**

- Oracle input + learned selective edit
- **status:** FAILURE (read from burned-in caption; this is the May-24 iconic render)
- **⚠ First-person panda_wristcam render: the burned-in retry success flag is UNRELIABLE on this variant (reads False even when the cube reaches the target — a documented controller×wristcam artifact, see renders/results/README.md). Trust the standard-panda Stage-0 PushCube section and the measured June-3 eval (latent retry success 0.96) for the real outcome, not these flags.**
- [▶ play full clip mp4](../../../renders/stage5_p1_iconic/pushcube/pushcube_seed_0110__babysteps_selective_full.mp4)
  ![strip](strips/iconic_0110_babysteps_selective__strip.jpg)

### seed 0110 — oracle_factor_revision — **⚠ wristcam (flag unreliable)**

- Oracle single-factor revision (skyline)
- **status:** FAILURE (read from burned-in caption; this is the May-24 iconic render)
- **⚠ First-person panda_wristcam render: the burned-in retry success flag is UNRELIABLE on this variant (reads False even when the cube reaches the target — a documented controller×wristcam artifact, see renders/results/README.md). Trust the standard-panda Stage-0 PushCube section and the measured June-3 eval (latent retry success 0.96) for the real outcome, not these flags.**
- [▶ play full clip mp4](../../../renders/stage5_p1_iconic/pushcube/pushcube_seed_0110__oracle_factor_revision_full.mp4)
  ![strip](strips/iconic_0110_oracle_factor_revision__strip.jpg)

### seed 0110 — same_intent_retry — **⚠ wristcam (flag unreliable)**

- Retry with unchanged intent (control — must fail)
- **status:** FAILURE (read from burned-in caption; this is the May-24 iconic render)
- **⚠ First-person panda_wristcam render: the burned-in retry success flag is UNRELIABLE on this variant (reads False even when the cube reaches the target — a documented controller×wristcam artifact, see renders/results/README.md). Trust the standard-panda Stage-0 PushCube section and the measured June-3 eval (latent retry success 0.96) for the real outcome, not these flags.**
- [▶ play full clip mp4](../../../renders/stage5_p1_iconic/pushcube/pushcube_seed_0110__same_intent_retry_full.mp4)
  ![strip](strips/iconic_0110_same_intent_retry__strip.jpg)

### seed 0120 — latent — **⚠ wristcam (flag unreliable)**

- Latent input + learned slot-local edit (BabySteps, the method)
- **status:** FAILURE (read from burned-in caption; this is the May-24 iconic render)
- **⚠ First-person panda_wristcam render: the burned-in retry success flag is UNRELIABLE on this variant (reads False even when the cube reaches the target — a documented controller×wristcam artifact, see renders/results/README.md). Trust the standard-panda Stage-0 PushCube section and the measured June-3 eval (latent retry success 0.96) for the real outcome, not these flags.**
- [▶ play full clip mp4](../../../renders/stage5_p1_iconic/pushcube/pushcube_seed_0120__latent_full.mp4)
  ![strip](strips/iconic_0120_latent__strip.jpg)

### seed 0120 — babysteps_selective — **⚠ wristcam (flag unreliable)**

- Oracle input + learned selective edit
- **status:** FAILURE (read from burned-in caption; this is the May-24 iconic render)
- **⚠ First-person panda_wristcam render: the burned-in retry success flag is UNRELIABLE on this variant (reads False even when the cube reaches the target — a documented controller×wristcam artifact, see renders/results/README.md). Trust the standard-panda Stage-0 PushCube section and the measured June-3 eval (latent retry success 0.96) for the real outcome, not these flags.**
- [▶ play full clip mp4](../../../renders/stage5_p1_iconic/pushcube/pushcube_seed_0120__babysteps_selective_full.mp4)
  ![strip](strips/iconic_0120_babysteps_selective__strip.jpg)

### seed 0120 — oracle_factor_revision — **⚠ wristcam (flag unreliable)**

- Oracle single-factor revision (skyline)
- **status:** FAILURE (read from burned-in caption; this is the May-24 iconic render)
- **⚠ First-person panda_wristcam render: the burned-in retry success flag is UNRELIABLE on this variant (reads False even when the cube reaches the target — a documented controller×wristcam artifact, see renders/results/README.md). Trust the standard-panda Stage-0 PushCube section and the measured June-3 eval (latent retry success 0.96) for the real outcome, not these flags.**
- [▶ play full clip mp4](../../../renders/stage5_p1_iconic/pushcube/pushcube_seed_0120__oracle_factor_revision_full.mp4)
  ![strip](strips/iconic_0120_oracle_factor_revision__strip.jpg)

### seed 0120 — same_intent_retry — **⚠ wristcam (flag unreliable)**

- Retry with unchanged intent (control — must fail)
- **status:** FAILURE (read from burned-in caption; this is the May-24 iconic render)
- **⚠ First-person panda_wristcam render: the burned-in retry success flag is UNRELIABLE on this variant (reads False even when the cube reaches the target — a documented controller×wristcam artifact, see renders/results/README.md). Trust the standard-panda Stage-0 PushCube section and the measured June-3 eval (latent retry success 0.96) for the real outcome, not these flags.**
- [▶ play full clip mp4](../../../renders/stage5_p1_iconic/pushcube/pushcube_seed_0120__same_intent_retry_full.mp4)
  ![strip](strips/iconic_0120_same_intent_retry__strip.jpg)

### seed 0130 — latent — **⚠ wristcam (flag unreliable)**

- Latent input + learned slot-local edit (BabySteps, the method)
- **status:** FAILURE (read from burned-in caption; this is the May-24 iconic render)
- **⚠ First-person panda_wristcam render: the burned-in retry success flag is UNRELIABLE on this variant (reads False even when the cube reaches the target — a documented controller×wristcam artifact, see renders/results/README.md). Trust the standard-panda Stage-0 PushCube section and the measured June-3 eval (latent retry success 0.96) for the real outcome, not these flags.**
- [▶ play full clip mp4](../../../renders/stage5_p1_iconic/pushcube/pushcube_seed_0130__latent_full.mp4)
  ![strip](strips/iconic_0130_latent__strip.jpg)

### seed 0130 — babysteps_selective — **⚠ wristcam (flag unreliable)**

- Oracle input + learned selective edit
- **status:** FAILURE (read from burned-in caption; this is the May-24 iconic render)
- **⚠ First-person panda_wristcam render: the burned-in retry success flag is UNRELIABLE on this variant (reads False even when the cube reaches the target — a documented controller×wristcam artifact, see renders/results/README.md). Trust the standard-panda Stage-0 PushCube section and the measured June-3 eval (latent retry success 0.96) for the real outcome, not these flags.**
- [▶ play full clip mp4](../../../renders/stage5_p1_iconic/pushcube/pushcube_seed_0130__babysteps_selective_full.mp4)
  ![strip](strips/iconic_0130_babysteps_selective__strip.jpg)

### seed 0130 — oracle_factor_revision — **⚠ wristcam (flag unreliable)**

- Oracle single-factor revision (skyline)
- **status:** FAILURE (read from burned-in caption; this is the May-24 iconic render)
- **⚠ First-person panda_wristcam render: the burned-in retry success flag is UNRELIABLE on this variant (reads False even when the cube reaches the target — a documented controller×wristcam artifact, see renders/results/README.md). Trust the standard-panda Stage-0 PushCube section and the measured June-3 eval (latent retry success 0.96) for the real outcome, not these flags.**
- [▶ play full clip mp4](../../../renders/stage5_p1_iconic/pushcube/pushcube_seed_0130__oracle_factor_revision_full.mp4)
  ![strip](strips/iconic_0130_oracle_factor_revision__strip.jpg)

### seed 0130 — same_intent_retry — **⚠ wristcam (flag unreliable)**

- Retry with unchanged intent (control — must fail)
- **status:** FAILURE (read from burned-in caption; this is the May-24 iconic render)
- **⚠ First-person panda_wristcam render: the burned-in retry success flag is UNRELIABLE on this variant (reads False even when the cube reaches the target — a documented controller×wristcam artifact, see renders/results/README.md). Trust the standard-panda Stage-0 PushCube section and the measured June-3 eval (latent retry success 0.96) for the real outcome, not these flags.**
- [▶ play full clip mp4](../../../renders/stage5_p1_iconic/pushcube/pushcube_seed_0130__same_intent_retry_full.mp4)
  ![strip](strips/iconic_0130_same_intent_retry__strip.jpg)

### seed 0143 — latent — **⚠ wristcam (flag unreliable)**

- Latent input + learned slot-local edit (BabySteps, the method)
- **status:** FAILURE (read from burned-in caption; this is the May-24 iconic render)
- **⚠ First-person panda_wristcam render: the burned-in retry success flag is UNRELIABLE on this variant (reads False even when the cube reaches the target — a documented controller×wristcam artifact, see renders/results/README.md). Trust the standard-panda Stage-0 PushCube section and the measured June-3 eval (latent retry success 0.96) for the real outcome, not these flags.**
- [▶ play full clip mp4](../../../renders/stage5_p1_iconic/pushcube/pushcube_seed_0143__latent_full.mp4)
  ![strip](strips/iconic_0143_latent__strip.jpg)

### seed 0143 — babysteps_selective — **⚠ wristcam (flag unreliable)**

- Oracle input + learned selective edit
- **status:** FAILURE (read from burned-in caption; this is the May-24 iconic render)
- **⚠ First-person panda_wristcam render: the burned-in retry success flag is UNRELIABLE on this variant (reads False even when the cube reaches the target — a documented controller×wristcam artifact, see renders/results/README.md). Trust the standard-panda Stage-0 PushCube section and the measured June-3 eval (latent retry success 0.96) for the real outcome, not these flags.**
- [▶ play full clip mp4](../../../renders/stage5_p1_iconic/pushcube/pushcube_seed_0143__babysteps_selective_full.mp4)
  ![strip](strips/iconic_0143_babysteps_selective__strip.jpg)

### seed 0143 — oracle_factor_revision — **⚠ wristcam (flag unreliable)**

- Oracle single-factor revision (skyline)
- **status:** FAILURE (read from burned-in caption; this is the May-24 iconic render)
- **⚠ First-person panda_wristcam render: the burned-in retry success flag is UNRELIABLE on this variant (reads False even when the cube reaches the target — a documented controller×wristcam artifact, see renders/results/README.md). Trust the standard-panda Stage-0 PushCube section and the measured June-3 eval (latent retry success 0.96) for the real outcome, not these flags.**
- [▶ play full clip mp4](../../../renders/stage5_p1_iconic/pushcube/pushcube_seed_0143__oracle_factor_revision_full.mp4)
  ![strip](strips/iconic_0143_oracle_factor_revision__strip.jpg)

### seed 0143 — same_intent_retry — **⚠ wristcam (flag unreliable)**

- Retry with unchanged intent (control — must fail)
- **status:** FAILURE (read from burned-in caption; this is the May-24 iconic render)
- **⚠ First-person panda_wristcam render: the burned-in retry success flag is UNRELIABLE on this variant (reads False even when the cube reaches the target — a documented controller×wristcam artifact, see renders/results/README.md). Trust the standard-panda Stage-0 PushCube section and the measured June-3 eval (latent retry success 0.96) for the real outcome, not these flags.**
- [▶ play full clip mp4](../../../renders/stage5_p1_iconic/pushcube/pushcube_seed_0143__same_intent_retry_full.mp4)
  ![strip](strips/iconic_0143_same_intent_retry__strip.jpg)

## Measured latent-intent decode — current P1/P2 run (June-3 eval; tabular, newer than the iconic videos above)

### seed 0100 — decoded intent (SUCCESS) — VLM factor `approach_direction`, changed `['approach_direction']`, attribution_correct=True

| factor | ground-truth | decoded (latent) | revised | flag |
|---|---|---|---|---|
| goal_state | cube_at_target | cube_at_target | cube_at_target | match |
| object_motion | translate_+x | translate_+x | translate_+x | match |
| contact_region | minus_x_face | minus_x_face | minus_x_face | match |
| approach_direction | from_minus_x | from_minus_x | from_plus_x | match · REVISED |
| constraint_region | none | none | none | match |
| embodiment_mapping | proxy_contact_to_franka_push | proxy_contact_to_franka_push | proxy_contact_to_franka_push | match |

### seed 0110 — decoded intent (SUCCESS) — VLM factor `approach_direction`, changed `['approach_direction']`, attribution_correct=True

| factor | ground-truth | decoded (latent) | revised | flag |
|---|---|---|---|---|
| goal_state | cube_at_target | cube_at_target | cube_at_target | match |
| object_motion | translate_+x | translate_+x | translate_+x | match |
| contact_region | minus_x_face | minus_x_face | minus_x_face | match |
| approach_direction | from_minus_x | from_minus_x | from_plus_x | match · REVISED |
| constraint_region | none | none | none | match |
| embodiment_mapping | proxy_contact_to_franka_push | proxy_contact_to_franka_push | proxy_contact_to_franka_push | match |

### seed 0120 — decoded intent (SUCCESS) — VLM factor `approach_direction`, changed `['approach_direction']`, attribution_correct=True

| factor | ground-truth | decoded (latent) | revised | flag |
|---|---|---|---|---|
| goal_state | cube_at_target | cube_at_target | cube_at_target | match |
| object_motion | translate_+x | translate_+x | translate_+x | match |
| contact_region | minus_x_face | minus_x_face | minus_x_face | match |
| approach_direction | from_minus_x | from_minus_x | from_plus_x | match · REVISED |
| constraint_region | none | none | none | match |
| embodiment_mapping | proxy_contact_to_franka_push | proxy_contact_to_franka_push | proxy_contact_to_franka_push | match |

### seed 0130 — decoded intent (SUCCESS) — VLM factor `approach_direction`, changed `['approach_direction']`, attribution_correct=True

| factor | ground-truth | decoded (latent) | revised | flag |
|---|---|---|---|---|
| goal_state | cube_at_target | cube_at_target | cube_at_target | match |
| object_motion | translate_+x | translate_+x | translate_+x | match |
| contact_region | minus_x_face | minus_x_face | minus_x_face | match |
| approach_direction | from_minus_x | from_minus_x | from_plus_x | match · REVISED |
| constraint_region | none | none | none | match |
| embodiment_mapping | proxy_contact_to_franka_push | proxy_contact_to_franka_push | proxy_contact_to_franka_push | match |

### seed 0143 — decoded intent (FAILURE) — VLM factor `approach_direction`, changed `['approach_direction']`, attribution_correct=True

| factor | ground-truth | decoded (latent) | revised | flag |
|---|---|---|---|---|
| goal_state | cube_at_target | cube_at_target | cube_at_target | match |
| object_motion | translate_+x | translate_-x | translate_-x | MISMATCH |
| contact_region | minus_x_face | plus_x_face | plus_x_face | MISMATCH |
| approach_direction | from_minus_x | from_plus_x | from_minus_x | MISMATCH · REVISED |
| constraint_region | none | none | none | match |
| embodiment_mapping | proxy_contact_to_franka_push | proxy_contact_to_franka_push | proxy_contact_to_franka_push | match |

## Official ManiSkill MP demo replays (third-person demo source)

### PushCube-v1 official demo replay

- Official ManiSkill motion-planning demos replayed third-person — the demo source for Scope-A 1_demo clips.
- [▶ play mp4](../../../renders/official_demo_smoke/PushCube-v1_seed_0000__official_replay.mp4)
  ![strip](strips/official_demo_pushcube_v1_official_demo_replay__strip.jpg)

### StackCube-v1 official demo replay

- Official ManiSkill motion-planning demos replayed third-person — the demo source for Scope-A 1_demo clips.
- [▶ play mp4](../../../renders/official_demo_smoke/StackCube-v1_seed_0000__official_replay.mp4)
  ![strip](strips/official_demo_stackcube_v1_official_demo_replay__strip.jpg)

### PickCube-v1 official demo replay

- Official ManiSkill motion-planning demos replayed third-person — the demo source for Scope-A 1_demo clips.
- [▶ play mp4](../../../renders/official_demo_smoke/PickCube-v1_seed_0000__official_replay.mp4)
  ![strip](strips/official_demo_pickcube_v1_official_demo_replay__strip.jpg)

## Task comparison montages

### PushCube comparison

- Side-by-side montages (demo | attempt | retry composited into one frame).
- [▶ play mp4](../../../renders/comparison/PushCube_comparison.mp4)
  ![strip](strips/comparison_pushcube_comparison__strip.jpg)

### PickCube comparison

- Side-by-side montages (demo | attempt | retry composited into one frame).
- [▶ play mp4](../../../renders/comparison/PickCube_comparison.mp4)
  ![strip](strips/comparison_pickcube_comparison__strip.jpg)

### StackCube comparison

- Side-by-side montages (demo | attempt | retry composited into one frame).
- [▶ play mp4](../../../renders/comparison/StackCube_comparison.mp4)
  ![strip](strips/comparison_stackcube_comparison__strip.jpg)

### TurnFaucet comparison

- Side-by-side montages (demo | attempt | retry composited into one frame).
- [▶ play mp4](../../../renders/comparison/TurnFaucet_comparison.mp4)
  ![strip](strips/comparison_turnfaucet_comparison__strip.jpg)

## Annotated latent-intent result (paper deliverable)

### Annotated demo→blocked→revised retry (seed 0000)

- Burned-in captions: inferred 6-slot intent, failure predicate, learned attribution, single-factor edit.
- [▶ play mp4](../../../renders/results/stage4_latent_intent_pushcube_seed_0000.mp4)
  ![strip](strips/annotated_result_annotated_demo_blocked_revised_retry_seed_0000__strip.jpg)
