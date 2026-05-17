"""Stage-0 multi-task summarizer.

Reads a `samples.jsonl` of EpisodeRecords and writes `report.{md,json}` in
`--out_dir`. The summarizer is decoupled from collection so reports can be
regenerated after metric-schema changes without re-running the simulator.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the project root importable without `pip install -e .`.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from babysteps.eval import compute_metrics, write_report  # noqa: E402
from babysteps.schemas import EpisodeRecord  # noqa: E402


def _iter_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield EpisodeRecord.from_jsonl_line(line)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    args = p.parse_args(argv)

    records = list(_iter_jsonl(args.samples))
    metrics = compute_metrics(records)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_report(metrics, args.out_dir)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
