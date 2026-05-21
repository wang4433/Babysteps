# datasets/ — collected Stage-0 episode data

Output of `scripts/stage0_collect.py`. One subdirectory per data cut.

## Layout

```text
datasets/
  stage0_pushcube_blocked/
    samples.jsonl          one Stage-0 episode record per line
    report.{json,md}       metrics + acceptance gate (from stage0_summarize.py)
    videos/                top-down (sim-free) renders
    videos_maniskill/      real-sim renders
```

Large real-sim data cuts (e.g. the CrossViewPush 24-seed gate) live **outside
the repo** under `/scratch/gilbreth/wang4433/data_<task>/` to keep the tree
light. This directory holds the in-repo reference cut(s).

## Episode record

Each `samples.jsonl` line is one episode with `demo / execution /
failure_packet / revision / retry` sections plus the `oracle_wrong_factor`
label (schema in `goal.md` and `babysteps/schemas.py`). The demo→intent path
must not contain privileged robot state — sim privilege is for labels and
success checks only.

## Rules

- Treat these as regenerable artifacts, not source. The generators
  (`stage0_collect.py`, `stage0_summarize.py`) are authoritative.
- Don't hand-edit `samples.jsonl` / `report.json`; re-run the summarizer.
