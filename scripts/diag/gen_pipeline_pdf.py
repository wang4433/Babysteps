#!/usr/bin/env python3
"""Generate a code-grounded BABYSTEPS pipeline overview PDF for verification.

Facts reconciled from goal.md / CLAUDE.md / schemas.py / episode.py /
failure.py / revision.py / vlm_attribute.py / vision_features.py and the
Stage-5 slurm status (workflow wf_0cc0ae84-e36, 2026-06-01).

Pipeline: graphviz `dot` -> high-DPI PNG -> embedded in a matplotlib
multi-page PDF (no LaTeX / pandoc / pdf-merge tool required).
"""
import subprocess, textwrap, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

OUT_DIR = "/scratch/gilbreth/wang4433/babysteps/reports"
TMP = "/tmp/_babysteps_pipeline"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TMP, exist_ok=True)
DOT_PATH = os.path.join(TMP, "pipeline.dot")
PNG_PATH = os.path.join(TMP, "pipeline.png")
PDF_PATH = os.path.join(OUT_DIR, "babysteps_pipeline_overview.pdf")

# Palette
H = "#1f3a5f"      # header navy
S0 = "#d6e4f0"     # stage-0 (discrete) light blue
S5 = "#fce5cd"     # stage-5 (vision/VLM) light orange
OPEN = "#e9e9e9"   # open/deferred grey
SHARED = "#eef2f4" # shared/neutral

def swap_node(nid, num, title, s0_label, s0_text, s5_label, s5_text, badge=""):
    b = f'&nbsp;<FONT COLOR="#b30000"><B>[{badge}]</B></FONT>' if badge else ""
    return f'''  {nid} [label=<
    <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">
      <TR><TD COLSPAN="2" BGCOLOR="{H}"><FONT COLOR="white"><B>{num}&nbsp;&nbsp;{title}</B></FONT>{b}</TD></TR>
      <TR><TD BGCOLOR="{S0}" ALIGN="LEFT"><B>{s0_label}</B></TD><TD BGCOLOR="{S0}" ALIGN="LEFT">{s0_text}</TD></TR>
      <TR><TD BGCOLOR="{S5}" ALIGN="LEFT"><B>{s5_label}</B></TD><TD BGCOLOR="{S5}" ALIGN="LEFT">{s5_text}</TD></TR>
    </TABLE>>];'''

def shared_node(nid, num, title, text, bg=SHARED):
    return f'''  {nid} [label=<
    <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">
      <TR><TD BGCOLOR="{H}"><FONT COLOR="white"><B>{num}&nbsp;&nbsp;{title}</B></FONT></TD></TR>
      <TR><TD BGCOLOR="{bg}" ALIGN="LEFT">{text}</TD></TR>
    </TABLE>>];'''

nodes = []
nodes.append(shared_node("n1", "①", "Demo evidence  (input)",
    "DemoEvidence{object_trajectory, contact_region_label, final_state}<BR/>"
    "camera = <B>third_person</B> &nbsp;·&nbsp; demonstrator = <B>proxy_oracle</B> (Franka, never human)"))

nodes.append(swap_node("n2", "②", "Demo &#8594; Intent",
    "STAGE-0", "adapter.scripted_demo_to_intent()<BR/>object-trajectory rules",
    "STAGE-5", "frozen <B>DINOv2 ViT-B/14</B> (768-d) &#8594; <B>IntentHead</B> (d_slot=32)<BR/>&#8594; slot intents G &#8712; R[6&#215;32]",
    badge="P1"))

nodes.append(shared_node("n3", "③", "Initial Intent  (the 6+1 factors)",
    "<B>6 core</B> INTENT_FIELDS: goal_state · object_motion · contact_region ·<BR/>"
    "approach_direction · constraint_region · embodiment_mapping<BR/>"
    "<B>+ direction_grounding</B> (7th, additive, default world_frame — Sub-project E)"))

nodes.append(swap_node("n4", "④", "Failure condition",
    "STAGE-0", "<B>injected</B> controlled error via blocked_sides<BR/>(StackCube: deliberately-wrong goal label)",
    "STAGE-5", "<B>arises naturally</B> from imperfect<BR/>vision-grounded intent inference"))

nodes.append(shared_node("n5", "⑤", "Execute attempt 1  (Franka, first-person)",
    "env_runner.run(): compile_intent_to_{push,pick,stack,turn}_skill<BR/>"
    "&#8594; ManiSkill proportional-control waypoint loop<BR/>"
    "<I>(only *_runner.py touch GPU / Vulkan — adapters stay sim-free)</I>"))

nodes.append(shared_node("n6", "⑥", "Structured failure packet",
    "adapter.build_failure_packet() &#8594; FailurePacket{failure_predicate}<BR/>"
    "strict precedence: success &gt; approach_blocked &gt; constraint_violation &gt;<BR/>"
    "grasp_infeasible &gt; grasp_slip &gt; contact_failure &gt; no_motion &gt; direction_error &gt; goal_not_satisfied"))

nodes.append(swap_node("n7", "⑦", "Attribute &#8594; which factor failed?",
    "STAGE-0", "rule table <B>FAILURE_TO_FACTOR</B><BR/>&#8594; Attribution{wrong_factor, freeze, revise}",
    "STAGE-5", "<B>VLM C1</B> — InternVL3.5-8B reads failure frame +<BR/>intent + predicate &#8594; <B>exactly ONE</B> factor name<BR/><I>(diagnosis only — never free-form replan)</I>",
    badge="P2"))

nodes.append(swap_node("n8", "⑧", "Revise EXACTLY ONE factor  (freeze the rest)",
    "STAGE-0", "rule operators: approach/contact_substitution,<BR/>goal_refinement, embodiment/grounding_substitution",
    "STAGE-5", "learned <B>ReviseHead</B> (latent_revision)<BR/>or VLM-constrained edit (vlm_constrained_revision)"))

nodes.append(shared_node("n9", "⑨", "Retry attempt 2",
    "env_runner.run(revised_intent) &#8594; attempt_2<BR/>(same compile + execute path; stable rollout seed)"))

nodes.append(shared_node("n10", "⑩", "Record + acceptance gate",
    "EpisodeRecord{episode_id, stage, task, <B>claim_boundary</B>, demo,<BR/>"
    "execution, failure_packet, revision, retry, metrics}<BR/>"
    "<B>GATE: delta_pp = retry_success &#8722; initial_success &#8805; 10pp</B>"))

nodes.append(swap_node("n11", "⑪", "Selectivity certification  (G3)",
    "STAGE-0/4", "mechanical <B>bit-identity</B> of frozen factors",
    "STAGE-5", "learned world model f(z,a)&#8594;z' counterfactual<BR/><B>[P3 — OPEN: not trained; probes falsified]</B>",
    badge="P3"))

edges = "\n".join(f"  n{i} -> n{i+1};" for i in range(1, 11))

dot = f'''digraph pipeline {{
  rankdir=TB;
  bgcolor="white";
  node [shape=plaintext, fontname="Helvetica", fontsize=11];
  edge [color="#34495e", penwidth=1.6, arrowsize=0.9];
  labelloc="t";
  fontname="Helvetica-Bold"; fontsize=20;
  label=<BABYSTEPS — failure-guided single-factor intent-revision loop<BR/><FONT POINT-SIZE="12" COLOR="#555555">blue = Stage-0 discrete/scripted path &#160;&#160;|&#160;&#160; orange = Stage-5 vision-grounded / VLM overlay &#160;&#160;|&#160;&#160; P4 = learned action decoder (replaces skill compiler, DEFERRED)</FONT>>;

{chr(10).join(nodes)}

{edges}
  n10 -> n11;
  n10 -> n1 [label=<&#160;next episode / retry loop&#160;>, style=dashed, color="#7f8c8d", fontcolor="#7f8c8d", constraint=false];
}}
'''

with open(DOT_PATH, "w") as f:
    f.write(dot)

subprocess.run(["dot", "-Tpng", "-Gdpi=220", DOT_PATH, "-o", PNG_PATH], check=True)
print("rendered PNG:", PNG_PATH)

# ----------------------------------------------------------------------------
# Assemble multi-page PDF
# ----------------------------------------------------------------------------
def text_page(pdf, title, blocks, subtitle=None, dense=False):
    """blocks: list of (kind, payload). kind in {'h','bul','tbl','note'}.
    dense=True scales spacing/fonts down so a content-heavy page fits one sheet."""
    fs_body, fs_label, fs_h, fs_note = (8.9, 9.2, 11.5, 8.9) if dense else (9.3, 9.6, 12.5, 9.3)
    lh, lh_label, lh_note = (0.0140, 0.0150, 0.0150) if dense else (0.0158, 0.0168, 0.0175)
    h_pre, h_post, bul_gap, note_gap = (0.004, 0.020, 0.004, 0.005) if dense else (0.006, 0.026, 0.007, 0.006)
    wrap_w, note_w = (116, 120) if dense else (108, 112)
    fig = plt.figure(figsize=(8.5, 11))
    fig.subplots_adjust(left=0.06, right=0.96, top=0.93, bottom=0.04)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_autoscale_on(False)
    y = 0.955
    ax.text(0.06, y, title, fontsize=18, fontweight="bold", color=H, va="top")
    y -= 0.030
    if subtitle:
        ax.text(0.06, y, subtitle, fontsize=9.5, color="#666666", va="top", style="italic")
        y -= 0.022
    ax.plot([0.06, 0.94], [y, y], color=H, lw=1.2); y -= 0.022
    for kind, payload in blocks:
        if y < 0.06:
            break
        if kind == "h":
            y -= h_pre
            ax.text(0.06, y, payload, fontsize=fs_h, fontweight="bold", color="#0b2545", va="top")
            y -= h_post
        elif kind == "bul":
            label, body = payload
            ax.text(0.065, y, "•", fontsize=fs_label + 1.4, color=H, va="top", fontweight="bold")
            if label:
                ax.text(0.085, y, label, fontsize=fs_label, fontweight="bold", color="#0b2545", va="top")
                y -= lh_label
            for ln in textwrap.wrap(body, width=wrap_w):
                ax.text(0.085, y, ln, fontsize=fs_body, color="#222222", va="top")
                y -= lh
            y -= bul_gap
        elif kind == "note":
            for ln in textwrap.wrap(payload, width=note_w):
                ax.text(0.06, y, ln, fontsize=fs_note, color="#444444", va="top")
                y -= lh_note
            y -= note_gap
        elif kind == "tbl":
            headers, rows, colw = payload
            t = ax.table(cellText=rows, colLabels=headers, colWidths=colw,
                         cellLoc="left", loc="upper left",
                         bbox=[0.06, max(0.05, y - 0.030 * (len(rows) + 1)), 0.88, 0.030 * (len(rows) + 1)])
            t.auto_set_font_size(False); t.set_fontsize(8.4)
            for (r, c), cell in t.get_celld().items():
                cell.set_edgecolor("#cccccc")
                if r == 0:
                    cell.set_facecolor(H); cell.set_text_props(color="white", fontweight="bold")
                elif r % 2 == 0:
                    cell.set_facecolor("#f4f7fa")
            y -= 0.030 * (len(rows) + 1) + 0.02
    pdf.savefig(fig)
    if os.environ.get("DUMP_PNG"):
        text_page._n = getattr(text_page, "_n", 1) + 1
        fig.savefig(os.path.join(TMP, f"page{text_page._n}.png"), dpi=130)
    plt.close(fig)

with PdfPages(PDF_PATH) as pdf:
    # ---- Page 1: the flow diagram ----
    img = plt.imread(PNG_PATH)
    h, w = img.shape[0], img.shape[1]
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96]); ax.axis("off")
    ax.imshow(img)
    pdf.savefig(fig, dpi=220); plt.close(fig)

    # ---- Page 2: data contract + invariants ----
    factor_rows = [
        ["goal_state", "desired final object relation/pose", "cube_at_target · cubeA_on_cubeB · faucet_turned"],
        ["object_motion", "observed/intended object movement", "translate_+x · lift_up · place_on · turn"],
        ["contact_region", "demonstrated/inferred contact site", "minus_x_face · handle_grip · faucet_base"],
        ["approach_direction", "route/side used to reach contact", "from_minus_x · from_above"],
        ["constraint_region", "scene/state to preserve", "none · faucet_base_static"],
        ["embodiment_mapping", "proxy contact -> Franka action", "..._push · ..._grasp · ..._poke_turn"],
        ["direction_grounding", "(7th, additive) demo-direction frame", "actor_frame · observer_frame · world_frame"],
    ]
    invariants = [
        ("Single-factor revision.", "every revision changes exactly one INTENT_FIELDS factor; Revision.frozen_factors lists the rest; enforced via dataclasses.replace(). (One legacy 2-factor exception: constraint_introduction — see verification page.)"),
        ("Additive schema.", "add new tokens + stop emitting old ones; defer removal to a cleanup pass. direction_grounding is the 7th field, outside INTENT_FIELDS, serialized only when non-default."),
        ("Demo = object evidence.", "demo describes object motion, never an executable Franka motor program; camera=third_person, demonstrator=proxy_oracle."),
        ("Sim privilege boundary.", "only *_runner.py import ManiSkill/Vulkan; the demo->intent path and tests/ stay sim-free (login-node runnable)."),
        ("One orchestrator.", "Stage-0 and Stage-5 share run_episode() and the same EpisodeRecord JSON; claim_boundary stamped on every record."),
        ("Universal gate.", "delta_pp = retry_success - initial_success >= 10pp, across Stage-0 discrete and Stage-4/5 latent revision alike."),
        ("VLM = diagnosis only.", "the VLM picks one factor name (which?), never free-form replanning (what value?). C2 free-form is a baseline only."),
    ]
    text_page(pdf, "Data contract & working invariants",
              [("h", "The six core + one additive intent factors (object-centric, task-agnostic)"),
               ("tbl", (["factor", "meaning", "example tokens (schemas.py whitelist)"], factor_rows, [0.20, 0.39, 0.41])),
               ("h", "Confirmed invariants (verified against code)"),
               *[("bul", inv) for inv in invariants]],
              subtitle="Source of truth = babysteps/schemas.py whitelists; goal.md authoritative over docs where they disagree.")

    # ---- Page 3: sub-projects + status ----
    sub_rows = [
        ["A", "PushCube-v1", "contact_region / approach_direction", "demo viewpoint conflates which face was contacted"],
        ["B", "PickCube-v1", "contact_region", "grasp face occluded in demo; grasp_slip at runtime"],
        ["C", "StackCube-v1", "goal_state", "demo goal-ambiguous (place-near vs stack-on); failure in label, not blocked_sides"],
        ["D", "TurnFaucet-v1", "embodiment_mapping", "demo grasp-turn infeasible on Franka wrist range (poke_turn unreliable)"],
        ["E", "CrossViewPush-v1", "direction_grounding", "demo camera != robot camera -> direction-frame mismatch (7-factor menu)"],
    ]
    status = [
        ("P1 vision swap — PARTIAL.", "PushCube end-to-end PASSES: held-out seeds 100-149 latent 48/50 (0.96) vs oracle 49/50 (0.98); G4 +96pp, G5 -2pp, object_motion probe 0.95. StackCube G1 FAILS: relational object_motion 0.42 (<0.90) across all pooling — scope narrowed to PushCube-only."),
        ("P2 VLM attribution — DONE.", "5-task main table passes all 3 gates (C1 attr >= rule-table; C1 pres >= C2; C1 succ within 5pp of C2). C1 attr: Push 1.0 · Pick 1.0 · Stack 0.86 · Turn 1.0 · CrossView 1.0; +92pp success on PickCube. StackCube 0.86 = residual open issue (7 misses on object_motion)."),
        ("P3 world model — OPEN.", "no learned world model trained; G3 selectivity still mechanical bit-identity. Only de-risking probes run (VLM-reads-from-demo, top-down vision) — ALL falsified."),
        ("P4 action decoder — DEFERRED.", "optional; paper stands on P1-P3 + skill compilers."),
        ("M2b pixel encoder — OPEN.", "not a named milestone; most likely the P1 vision_features.py pixel path, itself blocked on the StackCube relational bottleneck."),
        ("Done supporting:", "M1 locked claim · M2 5-task Stage-0 · M3 7 procedural baselines (Push 0.98/Pick 0.92/Stack 0.70) · M6 related-work. M4 main table + M7 paper PENDING P3."),
    ]
    text_page(pdf, "Sub-projects (A–E) & Stage-5 status",
              [("h", "Five Stage-0 task families — each revises a different single factor"),
               ("tbl", (["ID", "task", "revised factor", "inference failure mode"], sub_rows, [0.05, 0.18, 0.27, 0.50])),
               ("h", "Stage-5 priorities (status as of 2026-06-01)"),
               *[("bul", s) for s in status]],
              subtitle="All five pass the delta_pp >= 10 acceptance gate in Stage-0.")

    # ---- Page 4: misalignments to verify ----
    resolved = [
        ("VLM model identity.", "docs now state InternVL3.5-8B (method VLM-agnostic) across goal.md, CLAUDE.md, milestones.md, update.md and the locked-claim doc."),
        ("goal.md P2 status refreshed.", "stale v1 (3 tasks, StackCube 0%) replaced with the 5-task main table — all gates pass, StackCube attr 0.86, PickCube +92pp."),
        ("'six' -> '6 core + 1 additive'.", "CLAUDE.md prose now matches its own 7-row table (direction_grounding = additive 7th, Sub-project E)."),
        ("goal.md example tokens.", "shorthand translate_right / left_face / from_left replaced with real schema tokens translate_+x / minus_x_face / from_minus_x."),
        ("latent-intent framing.", "new 'Framing: latent intent vs. the swappable diagnoser' subsection added to goal.md Stage-5 (addresses the headline-scope risk below)."),
    ]
    real = [
        ("[LOW] claim_boundary phrasing.", "docs paraphrase it; the actual token is CLAIM_BOUNDARY='third_person_demo_proxy_not_human_demo' (schemas.py). Cosmetic — left as-is."),
    ]
    risk = [
        ("[HIGH] '5-task vision-grounded' headline.", "P1 latent G is end-to-end ONLY on PushCube (StackCube probe 0.42, scope narrowed). P2 VLM attribution runs on failure FRAMES + predicate strings, not on latent G — so 4/5 tasks use the Stage-4 discrete scaffold. Now framed honestly in goal.md ('latent' = representation + slot-local revision, demonstrated end-to-end on PushCube); hold this line in the paper draft."),
        ("[MED] single-factor invariant vs constraint_introduction.", "constraint_introduction (TurnFaucet, now deprecated in favour of embodiment_substitution) revises constraint_region AND contact_region — a 2-factor operator whose docstring still says single-factor. frozen_factors accounts for it, but legacy _compute_metrics could mis-handle len==2. The headline invariant has one live exception."),
    ]
    gaps = [
        ("[MED] continuous_rotation_infeasible has no rule.", "predicate is in FAILURE_PREDICATES (schemas.py) but absent from FAILURE_TO_FACTOR (failure.py) — attribute_failure() would raise ValueError if it is ever emitted (TurnFaucet redesign in flight)."),
        ("[LOW] silent learned-policy degradation.", "episode.py comment says provider=None makes latent_revision 'degrade to a no-op'; in fact the default policy just ignores demo_features with no warning/metric. Add an explicit guard/log."),
        ("[LOW] deprecated tokens linger.", "proxy_contact_to_franka_turn, constraint_violation, constraint_introduction kept in whitelists per the additive rule; schedule the cleanup pass once tests prove no references."),
    ]
    artifacts = [
        ("Revision field names.", "an extractor listed Revision={wrong_factor, operator_name}; the real dataclass is {operator, factor, old_value, new_value, frozen_factors} — code is correct, wrong_factor lives on Attribution."),
        ("Operator names.", "an extractor wrote goal_substitution / constraint_substitution; the real operators are goal_refinement / constraint_introduction — code is correct."),
        ("Enum membership.", "an extractor gave 3 direction_groundings & omitted faucet_base; code has 4 groundings (+object_frame, reserved) and 6 contact_regions — code is correct."),
    ]
    text_page(pdf, "Misalignments to double-check",
              [("note", "Items resolved 2026-06-01 are listed first; then still-open repo items, then extractor noise where the CODE IS ALREADY CORRECT (listed so you don't chase false alarms)."),
               ("h", "Resolved 2026-06-01 (docs updated this session)"),
               *[("bul", x) for x in resolved],
               ("h", "A. Doc <-> code drift (remaining)"),
               *[("bul", x) for x in real],
               ("h", "B. Claim / scope risks (paper-facing)"),
               *[("bul", x) for x in risk],
               ("h", "C. Code gaps / cruft"),
               *[("bul", x) for x in gaps],
               ("h", "D. Extractor artifacts — CODE IS FINE (ignore)"),
               *[("bul", x) for x in artifacts]],
              subtitle="Verified against schemas.py / revision.py / failure.py / episode.py / vlm_attribute.py on 2026-06-01.",
              dense=True)

print("wrote PDF:", PDF_PATH)
