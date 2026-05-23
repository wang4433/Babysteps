"""Diagnostic: per-task per-factor label diversity + demo-evidence diversity.

Not part of the tested codebase — a Task-6 understanding aid.
"""
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("/scratch/gilbreth/wang4433/babysteps")
sys.path.insert(0, str(ROOT))
from babysteps.schemas import EpisodeRecord, INTENT_FIELDS  # noqa: E402


def load(p):
    return [EpisodeRecord.from_jsonl_line(line).to_dict()
            for line in open(p) if line.strip()]


base = ROOT / "datasets/stage0_baselines"
for task in ("PushCube-v1", "PickCube-v1", "StackCube-v1"):
    for bl in ("babysteps_selective", "oracle_factor_revision"):
        recs = load(base / bl / task / "samples.jsonl")
        print(f"\n=== {task} [{bl}] n={len(recs)} ===")
        for f in INTENT_FIELDS:
            c = Counter(r["execution"]["initial_intent"][f] for r in recs)
            flag = "  <-- VARIES" if len(c) > 1 else ""
            print(f"  intent.{f:20s} uniq={len(c)} {dict(c)}{flag}")
        cc = Counter(r["demo"]["contact_region_label"] for r in recs)
        fc = Counter(r["demo"]["final_state"] for r in recs)
        print(f"  demo.contact_region_label  uniq={len(cc)} {dict(cc)}")
        print(f"  demo.final_state           uniq={len(fc)} {dict(fc)}")
