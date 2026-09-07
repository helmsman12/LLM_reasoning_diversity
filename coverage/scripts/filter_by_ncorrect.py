#!/usr/bin/env python3
"""Filter problems by minimum n_correct across all verification files.

Reads verification JSONL files (output of ``evaluate.py --phase verify``),
finds the intersection of problem IDs, and keeps only those where every
checkpoint has ``n_correct >= --min-correct``.

Usage:
    python scripts/filter_by_ncorrect.py \
        --verification-dir data/verification/ \
        --min-correct 16 \
        --output data/filtered_problem_ids.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List


def load_verification(path: Path) -> Dict[str, int]:
    """Return {problem_id: n_correct} from a verification JSONL."""
    result: Dict[str, int] = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            result[rec["problem_id"]] = rec["n_correct"]
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--verification-dir",
        required=True,
        help="Directory containing verification JSONL files.",
    )
    p.add_argument(
        "--glob",
        default="*.jsonl",
        help="Glob pattern to match verification files (default: *.jsonl).",
    )
    p.add_argument(
        "--min-correct",
        type=int,
        default=16,
        help="Minimum n_correct required in ALL checkpoints (default: 16).",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output file: one problem_id per line.",
    )
    args = p.parse_args()

    vdir = Path(args.verification_dir)
    files = sorted(vdir.glob(args.glob))
    if not files:
        print(f"ERROR: no files matching {args.glob} in {vdir}", file=sys.stderr)
        sys.exit(1)

    # Load all verification data
    all_data: Dict[str, Dict[str, int]] = {}  # tag -> {pid -> n_correct}
    for f in files:
        tag = f.stem
        all_data[tag] = load_verification(f)

    print(f"Loaded {len(all_data)} verification files:")
    for tag, data in all_data.items():
        print(f"  {tag}: {len(data)} problems")

    # Intersection of problem IDs
    common_pids = set.intersection(*(set(d.keys()) for d in all_data.values()))
    print(f"\nCommon problems across all files: {len(common_pids)}")

    # Filter by min_correct
    feasible: List[str] = []
    for pid in sorted(common_pids):
        min_nc = min(all_data[tag][pid] for tag in all_data)
        if min_nc >= args.min_correct:
            feasible.append(pid)

    print(f"Feasible (n_correct >= {args.min_correct} in all): {len(feasible)}")

    # Per-checkpoint stats on feasible set
    print(f"\nPer-checkpoint n_correct stats on feasible set:")
    for tag in sorted(all_data.keys()):
        vals = [all_data[tag][pid] for pid in feasible]
        if vals:
            avg = sum(vals) / len(vals)
            print(f"  {tag}: mean={avg:.1f}, min={min(vals)}, max={max(vals)}")

    # Write output
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(feasible) + "\n")
    print(f"\nWrote {len(feasible)} problem IDs to {out}")


if __name__ == "__main__":
    main()
