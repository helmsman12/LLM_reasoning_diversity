"""Group-wise filtered aggregation of coverage results.

Addresses the selection-bias issue in ``aggregate_results.py``: mean
Cov@N there is computed over *different* problem subsets for each N
(problems with ``n_correct >= N``), so small, easy-problem subsets at
large N can produce non-monotonic curves.

This script instead fixes a single problem set per *group* of
checkpoints: the intersection of problems where **every** checkpoint in
the group has ``status == "ok"`` and a non-null ``cov_at_n`` entry for
every N up to ``--max-n``. All per-checkpoint means are then computed
on that common set, so every number within a group is comparable on the
same denominator.

Two groups by default:
  - ``diver`` : checkpoints whose name starts with ``diver-``
  - ``sft``   : checkpoints whose name starts with ``sft-``

Checkpoints that fail the group-wide filter (e.g. runs with
``n_ok == 0`` like ``sft-yes-diverse-ckpt160``) can be dropped with
``--exclude`` so they don't collapse the intersection to zero.

Usage
-----
python scripts/aggregate_filtered.py \
    --eval-set eval-file-qwen25-3b \
    --max-n 32 \
    --exclude sft-yes-diverse-ckpt160 sft-yes-diverse-ckpt200
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


N_VALUES = (1, 2, 4, 8, 16, 32, 64)


def build_pid_map(group_file: str, base_file: str) -> Dict[str, str]:
    """Map a group's problem_id (line-index in group_file) -> base's
    problem_id (line-index in base_file) via shared ``problem`` text.

    Returns {} if the files can't be loaded."""
    try:
        with open(group_file) as f:
            group_lines = [json.loads(l) for l in f if l.strip()]
        with open(base_file) as f:
            base_lines = [json.loads(l) for l in f if l.strip()]
    except FileNotFoundError:
        return {}
    base_index = {rec.get("problem"): i for i, rec in enumerate(base_lines)}
    out: Dict[str, str] = {}
    for i, rec in enumerate(group_lines):
        j = base_index.get(rec.get("problem"))
        if j is None:
            continue
        out[f"problem_{i:04d}"] = f"problem_{j:04d}"
    return out


def load_runs(runs_dir: str, eval_set: Optional[str]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in sorted(Path(runs_dir).glob("*.jsonl")):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if eval_set and rec.get("eval_set") != eval_set:
                    continue
                records.append(rec)
    return records


def _cov_value(rec: Dict[str, Any], n: int) -> Optional[float]:
    cov = (rec.get("metrics") or {}).get("cov_at_n") or {}
    v = cov.get(n)
    if v is None:
        v = cov.get(str(n))
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    return float(v)


def classify(checkpoint: str) -> Optional[str]:
    if checkpoint.startswith("diver-"):
        return "diver"
    if checkpoint.startswith("sft-"):
        return "sft"
    return None


def _valid_pids(prob_map: Dict[str, Dict[str, Any]], max_n: int) -> Set[str]:
    required_ns = [n for n in N_VALUES if n <= max_n]
    ok: Set[str] = set()
    for pid, rec in prob_map.items():
        if rec.get("status") != "ok":
            continue
        if any(_cov_value(rec, n) is None for n in required_ns):
            continue
        ok.add(pid)
    return ok


def compute_common_problem_set(
    by_ckpt: Dict[str, Dict[str, Dict[str, Any]]],
    max_n: int,
    base_prob_map: Optional[Dict[str, Dict[str, Any]]] = None,
    pid_map: Optional[Dict[str, str]] = None,
) -> Set[str]:
    """Problems (in group-pid space) where *every* checkpoint has
    status=ok and valid cov_at_n for all N <= max_n. If ``base_prob_map``
    and ``pid_map`` are given, also require the remapped base-model
    record to be valid at max_n."""
    common: Optional[Set[str]] = None
    for prob_map in by_ckpt.values():
        ok = _valid_pids(prob_map, max_n)
        common = ok if common is None else (common & ok)
    common = common or set()

    if base_prob_map is not None and pid_map is not None:
        base_ok = _valid_pids(base_prob_map, max_n)
        common = {
            g_pid for g_pid in common
            if pid_map.get(g_pid) in base_ok
        }
    return common


def summarize_one(
    label: str,
    prob_map: Dict[str, Dict[str, Any]],
    pids: Set[str],
    max_n: int,
) -> Dict[str, Any]:
    """Compute mean Cov@N for a single checkpoint
    restricted to ``pids``. Records missing from ``prob_map`` are ignored."""
    mean_cov: Dict[int, Optional[float]] = {}
    for n in N_VALUES:
        if n > max_n:
            mean_cov[n] = None
            continue
        vals: List[float] = []
        for pid in pids:
            rec = prob_map.get(pid)
            if rec is None:
                continue
            v = _cov_value(rec, n)
            if v is not None:
                vals.append(v)
        mean_cov[n] = sum(vals) / len(vals) if vals else None

    matched = sum(1 for pid in pids if prob_map.get(pid) is not None)

    return {
        "checkpoint": label,
        "n_matched": matched,
        "mean_cov_at_n": mean_cov,
        "auc_up_to_max_n": sum(v for v in mean_cov.values() if v is not None),
    }


def summarize_group(
    group_name: str,
    by_ckpt: Dict[str, Dict[str, Dict[str, Any]]],
    common: Set[str],
    max_n: int,
) -> Dict[str, Any]:
    rows = []
    for ckpt in sorted(by_ckpt.keys()):
        row = summarize_one(ckpt, by_ckpt[ckpt], common, max_n)
        row["n_common_problems"] = len(common)
        rows.append(row)
    return {
        "group": group_name,
        "max_n": max_n,
        "n_common_problems": len(common),
        "checkpoints": sorted(by_ckpt.keys()),
        "rows": rows,
    }


def _fmt(v) -> str:
    return "  None" if v is None else f"{v:.4f}"


def print_group(summary: Dict[str, Any]) -> None:
    print()
    print("=" * 72)
    base_tag = " +base" if summary.get("base_in_filter") else ""
    print(f"GROUP: {summary['group']}  (max_n={summary['max_n']}{base_tag})")
    print(f"common problem set size: {summary['n_common_problems']}")
    print(f"members: {', '.join(summary['checkpoints'])}")
    print("=" * 72)
    if summary["n_common_problems"] == 0:
        print("  (empty intersection — nothing to report)")
        return
    ns_used = [n for n in N_VALUES if n <= summary["max_n"]]
    header = "  ckpt".ljust(40) + "   AUC   " + "".join(
        f"  N={n:<4d}" for n in ns_used
    ) + "      n"
    print(header)
    for row in summary["rows"]:
        line = f"  {row['checkpoint']:<38s}"
        line += f" {row['auc_up_to_max_n']:6.3f} "
        for n in ns_used:
            line += f" {_fmt(row['mean_cov_at_n'][n])}"
        line += f"  {row.get('n_matched', summary['n_common_problems'])}"
        print(line)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-dir", default="results/runs")
    p.add_argument("--eval-set", default="eval-file-qwen25-3b")
    p.add_argument("--max-n", type=int, nargs="+", default=[32],
                   help="One or more max_n values. Each produces its own "
                        "common-problem set and per-checkpoint summary. "
                        "Default [32]. 64 is usually empty.")
    p.add_argument("--exclude", nargs="*", default=[],
                   help="Checkpoints to drop from their group before "
                        "computing the common problem set.")
    p.add_argument("--only-group", choices=["diver", "sft"], default=None)
    p.add_argument("--base-checkpoint", default="Qwen/Qwen2.5-3B",
                   help="Checkpoint name of base model to compare against. "
                        "Pass '' to disable.")
    p.add_argument("--base-eval-set", default="eval-file",
                   help="eval_set the base model was evaluated on.")
    p.add_argument("--group-eval-file",
                   default="data/eval_file_qwen25_3b.jsonl",
                   help="Data file whose line indices define group "
                        "checkpoints' problem_ids.")
    p.add_argument("--base-eval-file",
                   default="data/eval_file.jsonl",
                   help="Data file whose line indices define the base "
                        "model's problem_ids. Used to remap problem_ids "
                        "across the two eval_sets.")
    p.add_argument("--no-base-in-filter", action="store_true",
                   help="Do NOT require the base model to also have valid "
                        "cov_at_n on the common set. Default is to include "
                        "base feasibility in the filter.")
    p.add_argument("--output", default="results/filtered_summary.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    records = load_runs(args.runs_dir, args.eval_set)
    if not records:
        print(f"[filtered] no records for eval_set={args.eval_set}", file=sys.stderr)
        sys.exit(1)

    # Partition: group -> ckpt -> problem_id -> record
    groups: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {
        "diver": defaultdict(dict), "sft": defaultdict(dict),
    }
    excluded = set(args.exclude)
    for rec in records:
        ckpt = rec.get("checkpoint") or ""
        if ckpt in excluded:
            continue
        g = classify(ckpt)
        if g is None:
            continue
        pid = rec.get("problem_id")
        if pid is None:
            continue
        groups[g][ckpt][pid] = rec

    # Base model: separate load (different eval_set), plus pid remap.
    base_prob_map: Dict[str, Dict[str, Any]] = {}
    pid_map: Dict[str, str] = {}  # group_pid -> base_pid
    if args.base_checkpoint:
        base_records = load_runs(args.runs_dir, args.base_eval_set)
        for rec in base_records:
            if rec.get("checkpoint") != args.base_checkpoint:
                continue
            pid = rec.get("problem_id")
            if pid:
                base_prob_map[pid] = rec
        pid_map = build_pid_map(args.group_eval_file, args.base_eval_file)
        if not pid_map and base_prob_map:
            print(
                f"[filtered] WARNING: could not build pid remap from "
                f"{args.group_eval_file} → {args.base_eval_file}; "
                "base model row will be skipped.",
                file=sys.stderr,
            )

    summaries: List[Dict[str, Any]] = []
    for max_n in args.max_n:
        for group_name in ("diver", "sft"):
            if args.only_group and args.only_group != group_name:
                continue
            by_ckpt = groups[group_name]
            if not by_ckpt:
                continue
            use_base_filter = (
                bool(base_prob_map) and bool(pid_map) and not args.no_base_in_filter
            )
            common = compute_common_problem_set(
                by_ckpt,
                max_n,
                base_prob_map=base_prob_map if use_base_filter else None,
                pid_map=pid_map if use_base_filter else None,
            )
            summary = summarize_group(group_name, by_ckpt, common, max_n)
            summary["base_in_filter"] = use_base_filter

            # Append a base-model row computed on the remapped common set.
            if base_prob_map and pid_map:
                base_pids = {
                    pid_map[g_pid] for g_pid in common if g_pid in pid_map
                }
                base_row = summarize_one(
                    f"{args.base_checkpoint} (base)",
                    base_prob_map,
                    base_pids,
                    max_n,
                )
                base_row["n_common_problems"] = len(common)
                summary["rows"].append(base_row)
                summary["base_checkpoint"] = args.base_checkpoint
                summary["base_n_mapped"] = len(base_pids)

            summaries.append(summary)

    for s in summaries:
        print_group(s)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(summaries, f, indent=2, ensure_ascii=False)
        print(f"\n[filtered] wrote {out}")


if __name__ == "__main__":
    main()
