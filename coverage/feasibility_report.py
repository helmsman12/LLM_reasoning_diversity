from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from statistics import median
from typing import Dict, List


def load(path: str) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out[r["problem_id"]] = r
    return out


def is_feasible(rec: dict) -> bool:
    return (
        rec.get("status") == "ok"
        and (rec.get("n_correct") or 0) >= 2
        and (rec.get("n_approaches") or 0) >= 2
    )


def failure_reason(rec: dict) -> str:
    if rec.get("status") != "ok":
        return f"status={rec.get('status')}"
    if (rec.get("n_correct") or 0) < 2:
        return f"n_correct<2"
    if (rec.get("n_approaches") or 0) < 2:
        return "n_approaches<2"
    return "feasible"


def summarize(label: str, recs: Dict[str, dict]) -> Dict:
    n = len(recs)
    feasible_ids = [p for p, r in recs.items() if is_feasible(r)]
    reasons = Counter(failure_reason(r) for r in recs.values())
    n_correct_vals = [r.get("n_correct") or 0 for r in recs.values() if is_feasible(r)]
    n_app_vals = [r.get("n_approaches") or 0 for r in recs.values() if is_feasible(r)]
    return {
        "label": label,
        "n_problems": n,
        "n_feasible": len(feasible_ids),
        "reason_breakdown": dict(reasons),
        "feasible_n_correct": {
            "median": median(n_correct_vals) if n_correct_vals else None,
            "min": min(n_correct_vals) if n_correct_vals else None,
            "max": max(n_correct_vals) if n_correct_vals else None,
        },
        "feasible_n_approaches": {
            "median": median(n_app_vals) if n_app_vals else None,
            "min": min(n_app_vals) if n_app_vals else None,
            "max": max(n_app_vals) if n_app_vals else None,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", required=True, help="cluster-run jsonl for pre (baseline) checkpoint")
    ap.add_argument("--post", required=True, help="cluster-run jsonl for post (trained) checkpoint")
    ap.add_argument("--pre-label", default="pre")
    ap.add_argument("--post-label", default="post")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    pre = load(args.pre)
    post = load(args.post)

    common = set(pre) & set(post)
    only_pre = set(pre) - set(post)
    only_post = set(post) - set(pre)

    def collapsed(rec: dict) -> bool:
        return not is_feasible(rec)

    pre_col = {p for p in common if collapsed(pre[p])}
    post_col = {p for p in common if collapsed(post[p])}

    only_pre_col = sorted(pre_col - post_col)
    only_post_col = sorted(post_col - pre_col)
    both_col = sorted(pre_col & post_col)

    feasible_both = sorted(common - (pre_col | post_col))

    report = {
        "inputs": {
            args.pre_label: os.path.abspath(args.pre),
            args.post_label: os.path.abspath(args.post),
        },
        "per_file_summary": {
            args.pre_label: summarize(args.pre_label, pre),
            args.post_label: summarize(args.post_label, post),
        },
        "paired_summary": {
            "n_common_problems": len(common),
            "n_only_in_pre_file": len(only_pre),
            "n_only_in_post_file": len(only_post),
            "n_feasible_in_both": len(feasible_both),
            "n_at_least_one_collapse": len(pre_col | post_col),
        },
        "collapse_asymmetry": {
            f"collapse_only_in_{args.pre_label}__diverse_in_{args.post_label}": {
                "count": len(only_pre_col),
                "problem_ids": only_pre_col,
            },
            f"collapse_only_in_{args.post_label}__diverse_in_{args.pre_label}": {
                "count": len(only_post_col),
                "problem_ids": only_post_col,
            },
            "collapse_in_both": {
                "count": len(both_col),
                "problem_ids": both_col,
            },
        },
        "feasible_problem_ids": feasible_both,
    }

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    # short stderr summary
    import sys
    print(f"wrote {args.output}", file=sys.stderr)
    print(f"feasible in both: {len(feasible_both)}", file=sys.stderr)
    print(
        f"collapse asymmetry: only-{args.pre_label}={len(only_pre_col)} "
        f"only-{args.post_label}={len(only_post_col)} both={len(both_col)}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
