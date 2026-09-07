"""Aggregate per-run result files into a single analysis-ready dataset.

Each run produces one JSONL file under ``results/runs/``.
This script collects all of them (or a filtered subset), computes
dataset-level metrics (AUC), and writes a combined JSONL plus a
human-readable summary to stdout.

Usage
-----
# Aggregate everything into results/all_results.jsonl
python scripts/aggregate_results.py

# Filter by model or eval_set
python scripts/aggregate_results.py --model-id Qwen2.5-3B
python scripts/aggregate_results.py --eval-set dry-run-10
python scripts/aggregate_results.py --model-id Qwen2.5-3B --eval-set dry-run-10

# Custom input / output paths
python scripts/aggregate_results.py \
    --runs-dir results/runs \
    --output results/all_results.jsonl

# Print summary without writing output file
python scripts/aggregate_results.py --summary-only
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_runs(
    runs_dir: str,
    model_id: Optional[str] = None,
    checkpoint: Optional[str] = None,
    eval_set: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load all per-problem records from every run file, with optional filters."""
    records: List[Dict[str, Any]] = []
    run_files = sorted(Path(runs_dir).glob("*.jsonl"))
    if not run_files:
        print(f"[aggregate] no run files found in {runs_dir}", file=sys.stderr)
        return records

    for path in run_files:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if model_id and rec.get("model_id") != model_id:
                    continue
                if checkpoint and rec.get("checkpoint") != checkpoint:
                    continue
                if eval_set and rec.get("eval_set") != eval_set:
                    continue
                rec["_source_file"] = str(path)
                records.append(rec)

    return records


# ---------------------------------------------------------------------------
# Dataset-level AUC (CLAUDE.md §3)
# ---------------------------------------------------------------------------

N_VALUES = (1, 2, 4, 8, 16, 32, 64)


def compute_auc(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute mean Coverage@N across problems (NaN-safe) and sum for AUC.

    Groups by (model_id, checkpoint, eval_set) before averaging so that
    mixing runs doesn't conflate different experimental conditions.
    """
    # group records
    groups: Dict[tuple, List[Dict]] = defaultdict(list)
    for rec in records:
        key = (rec.get("model_id"), rec.get("checkpoint"), rec.get("eval_set"))
        groups[key].append(rec)

    results = []
    for (mid, ckpt, eset), grp in sorted(groups.items()):
        cov_by_n: Dict[int, List[float]] = {n: [] for n in N_VALUES}
        for rec in grp:
            cov_at_n = (rec.get("metrics") or {}).get("cov_at_n") or {}
            for n in N_VALUES:
                v = cov_at_n.get(n) or cov_at_n.get(str(n))
                if v is not None and not (isinstance(v, float) and math.isnan(v)):
                    cov_by_n[n].append(v)

        mean_cov = {}
        for n in N_VALUES:
            vals = cov_by_n[n]
            mean_cov[n] = sum(vals) / len(vals) if vals else None

        auc = sum(v for v in mean_cov.values() if v is not None)
        n_ok = sum(1 for r in grp if r.get("status") == "ok")
        n_total = len(grp)

        results.append({
            "model_id": mid,
            "checkpoint": ckpt,
            "eval_set": eset,
            "n_problems": n_total,
            "n_ok": n_ok,
            "n_no_correct": sum(1 for r in grp if r.get("status") == "no_correct_solutions"),
            "n_clustering_failed": sum(1 for r in grp if r.get("status") == "clustering_failed"),
            "mean_cov_at_n": mean_cov,
            "auc": auc,
        })
    return results


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(auc_results: List[Dict[str, Any]]) -> None:
    print()
    print("=" * 70)
    print("AGGREGATE SUMMARY")
    print("=" * 70)
    for r in auc_results:
        print(f"\nmodel={r['model_id']}  ckpt={r['checkpoint']}  eval_set={r['eval_set']}")
        print(f"  problems : {r['n_problems']}  (ok={r['n_ok']}  no_correct={r['n_no_correct']}  clustering_failed={r['n_clustering_failed']})")
        print(f"  AUC (sum mean Cov@N)    : {r['auc']:.4f}")
        print(f"  mean Cov@N breakdown:")
        for n, v in r["mean_cov_at_n"].items():
            bar = "#" * int((v or 0) * 4)
            print(f"    N={n:2d}: {_fmt(v):>6s}  {bar}")
    print()


def _fmt(v) -> str:
    if v is None:
        return "  None"
    return f"{v:.4f}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate per-run coverage results.")
    p.add_argument("--runs-dir", default="results/runs")
    p.add_argument("--output", default="results/all_results.jsonl",
                   help="Combined output JSONL (pass '' to skip writing)")
    p.add_argument("--model-id", default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--eval-set", default=None)
    p.add_argument("--summary-only", action="store_true",
                   help="Print summary without writing output file")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    records = load_runs(
        args.runs_dir,
        model_id=args.model_id,
        checkpoint=args.checkpoint,
        eval_set=args.eval_set,
    )
    print(f"[aggregate] loaded {len(records)} problem records from {args.runs_dir}")

    if not records:
        print("[aggregate] nothing to aggregate", file=sys.stderr)
        sys.exit(0)

    # Write combined JSONL
    if not args.summary_only and args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[aggregate] wrote {len(records)} records → {out_path}")

    # Compute and print dataset-level AUC summary
    auc_results = compute_auc(records)
    print_summary(auc_results)

    # Write AUC summary as JSON
    if not args.summary_only and args.output:
        summary_path = Path(args.output).with_suffix(".summary.json")
        with open(summary_path, "w") as f:
            json.dump(auc_results, f, indent=2, ensure_ascii=False)
        print(f"[aggregate] wrote AUC summary → {summary_path}")


if __name__ == "__main__":
    main()
