#!/usr/bin/env python3
"""Approach-uniqueness judge (filtering stage 4).

For every problem, an LLM judge reads the candidate approach plans and
reports how many genuinely distinct approaches they contain. Problems
with ``num_unique_approaches >= 3`` are "positive" and are written to
``<output>_positive.json``.

Input (``.json`` list or ``.jsonl``): one record per problem with
``problem`` and ``response.approaches`` (the output of
``generate_approaches_batch.py`` / ``check_feasible_plans.py``).

Modes
-----
* default        : OpenAI Batch API (50% discount), blocks until done
* --batch-id     : resume polling an already-submitted batch
* --raw-results-file : re-process saved raw results (no API calls)
* --realtime     : synchronous chat.completions with a thread pool
* --eval-only    : re-evaluate / re-extract positives from an existing output
* --eval         : compare predictions to a ``label`` field
                   ("positive"/"negative") and print P/R/F1

Usage
-----
    python uniqueness_judge.py \\
        --input-file feasible_plans.jsonl \\
        --output-file outputs/uniqueness.json \\
        --model gpt-5.1 --reasoning-effort low
"""

import argparse
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

# Load OPENAI_API_KEY from the package-level .env, if present.
# Variables already set in the environment are never overridden.
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
except ImportError:  # python-dotenv is optional; fall back to `source .env`
    pass

from uniqueness_judge_utils import (
    build_message,
    calculate_cost,
    load_results,
    parse_num_unique,
    pred_positive,
)

_openai_client = None


def get_openai_client():
    """Lazily construct the OpenAI client."""
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set (see .env.example)")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


# ---------------------------------------------------------------------------
# OpenAI Batch API helpers
# ---------------------------------------------------------------------------

def upload_batch_file(client, batch_requests: List[dict]) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for request in batch_requests:
            f.write(json.dumps(request) + "\n")
        temp_file_path = f.name
    try:
        with open(temp_file_path, "rb") as f:
            file_response = client.files.create(file=f, purpose="batch")
        return file_response.id
    finally:
        os.unlink(temp_file_path)


def create_batch_job(client, file_id: str) -> str:
    batch_response = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    return batch_response.id


def wait_for_batch_completion(client, batch_id: str, wait_time: int = 30) -> List[dict]:
    print(f"Batch job {batch_id} created. Waiting for completion...")
    while True:
        batch_status = client.batches.retrieve(batch_id)
        print(f"Batch status: {batch_status.status}")
        if batch_status.status == "completed":
            break
        if batch_status.status in ("failed", "expired", "cancelled"):
            raise RuntimeError(f"Batch job {batch_status.status}: {getattr(batch_status, 'errors', '')}")
        time.sleep(wait_time)

    results_file_id = batch_status.output_file_id
    if results_file_id is None:
        raise RuntimeError(
            "Batch completed but output_file_id is None. "
            f"Request counts: {batch_status.request_counts}"
        )
    results_file = client.files.content(results_file_id)
    return [json.loads(line) for line in results_file.text.split("\n") if line.strip()]


# ---------------------------------------------------------------------------
# Request construction
# ---------------------------------------------------------------------------

def _iter_tasks(data: List[dict], num_votes: int):
    """Yield (custom_id, messages) for every judgeable item and vote."""
    for idx, item in enumerate(data):
        problem = item.get("problem")
        response = item.get("response")
        if response is None:
            print(f"Warning: Skipping item {idx} - missing response")
            continue
        approaches = response.get("approaches")
        if approaches is None:
            print(f"Warning: Skipping item {idx} - missing approaches")
            continue
        messages = build_message(problem, approaches)
        for vote_idx in range(num_votes):
            custom_id = f"request_{idx}_vote_{vote_idx}" if num_votes > 1 else f"request_{idx}"
            yield custom_id, messages


def _request_body(model: str, messages: List[dict], reasoning_effort: Optional[str]) -> dict:
    body = {"model": model, "messages": messages}
    if reasoning_effort is not None and reasoning_effort != "none":
        body["reasoning_effort"] = reasoning_effort
    return body


def create_batch_requests(data: List[dict], model: str, num_votes: int,
                          reasoning_effort: Optional[str]) -> List[dict]:
    return [
        {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": _request_body(model, messages, reasoning_effort),
        }
        for custom_id, messages in _iter_tasks(data, num_votes)
    ]


def run_realtime_requests(data: List[dict], model: str, num_votes: int,
                          reasoning_effort: Optional[str], num_workers: int) -> List[dict]:
    """Query the judge synchronously. Returns results in Batch-API format."""
    tasks = list(_iter_tasks(data, num_votes))

    def _query(task):
        custom_id, messages = task
        try:
            resp = get_openai_client().chat.completions.create(
                **_request_body(model, messages, reasoning_effort)
            )
            return {
                "custom_id": custom_id,
                "response": {
                    "status_code": 200,
                    "body": {
                        "choices": [{"message": {"content": resp.choices[0].message.content}}],
                        "usage": {
                            "prompt_tokens": resp.usage.prompt_tokens,
                            "completion_tokens": resp.usage.completion_tokens,
                        },
                    },
                },
            }
        except Exception as e:
            print(f"  [realtime] Request {custom_id} failed: {e}")
            return {"custom_id": custom_id, "response": {"status_code": 500, "body": {"error": str(e)}}}

    print(f"Running {len(tasks)} realtime requests with {num_workers} workers (model={model})")
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_query, t) for t in tasks]
        for i, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if i % 10 == 0 or i == len(tasks):
                print(f"  [realtime] {i}/{len(tasks)} completed")
    return results


# ---------------------------------------------------------------------------
# Result processing
# ---------------------------------------------------------------------------

def _set_empty(result: dict) -> None:
    result["judge_result"] = None
    result["num_unique_approaches"] = None
    result["explanation"] = None


def process_batch_results(batch_results: List[dict], original_data: List[dict],
                          num_votes: int = 1) -> List[dict]:
    """Merge judge outputs into the input records (majority vote if num_votes > 1)."""
    results = original_data.copy()

    if num_votes == 1:
        for batch_result in batch_results:
            custom_id = batch_result.get("custom_id", "")
            idx = int(custom_id.split("_")[1]) if custom_id else -1
            if batch_result.get("response", {}).get("status_code") == 200:
                content = batch_result["response"]["body"]["choices"][0]["message"]["content"]
                num_unique = parse_num_unique(content)
                if num_unique is None:
                    print(f"Warning: Unparseable \\boxed answer for request {idx}")
                results[idx]["judge_result"] = content
                results[idx]["num_unique_approaches"] = num_unique
                results[idx]["explanation"] = content
            else:
                print(f"Warning: Failed result for request {idx}")
                if idx >= 0:
                    _set_empty(results[idx])
        return results

    # Majority voting over num_votes independent judgements per item.
    votes_by_item = {}
    for batch_result in batch_results:
        if batch_result.get("response", {}).get("status_code") != 200:
            continue
        parts = batch_result["custom_id"].split("_")
        idx = int(parts[1])
        vote_idx = int(parts[3]) if len(parts) >= 4 else 0
        content = batch_result["response"]["body"]["choices"][0]["message"]["content"]
        votes_by_item.setdefault(idx, []).append({
            "vote_idx": vote_idx,
            "judge_result": content,
            "num_unique_approaches": parse_num_unique(content),
            "explanation": content,
        })

    for idx, votes in votes_by_item.items():
        positive_votes = [v for v in votes if v["num_unique_approaches"] is not None and pred_positive(v)]
        negative_votes = [v for v in votes if v["num_unique_approaches"] is not None and not pred_positive(v)]
        total_valid = len(positive_votes) + len(negative_votes)

        if total_valid == 0:
            _set_empty(results[idx])
            results[idx]["explanation"] = "No valid votes"
            results[idx]["voting_details"] = {
                "total_votes": len(votes), "valid_votes": 0,
                "positive_votes": 0, "negative_votes": 0, "all_votes": votes,
            }
            continue

        is_majority_positive = len(positive_votes) > len(negative_votes)
        representative = positive_votes[0] if is_majority_positive else negative_votes[0]
        results[idx]["judge_result"] = representative["judge_result"]
        results[idx]["num_unique_approaches"] = representative["num_unique_approaches"]
        results[idx]["explanation"] = representative["explanation"]
        results[idx]["voting_details"] = {
            "total_votes": len(votes),
            "valid_votes": total_valid,
            "positive_votes": len(positive_votes),
            "negative_votes": len(negative_votes),
            "majority_decision": "positive" if is_majority_positive else "negative",
            "all_votes": votes,
        }

    return results


# ---------------------------------------------------------------------------
# Evaluation against labels
# ---------------------------------------------------------------------------

def evaluate_judge_result(results: List[dict]) -> Tuple[List[int], int, List[dict]]:
    """Return ([TP, FP, TN, FN], total, results) using the ``label`` field."""
    evals = [0, 0, 0, 0]
    total = 0
    for result in results:
        if result.get("judge_result") is None or result.get("num_unique_approaches") is None:
            continue
        if "label" not in result:
            print("Warning: 'label' field missing in result, skipping evaluation")
            continue
        total += 1
        is_pred_positive = pred_positive(result)
        is_label_positive = result["label"].lower() == "positive"
        if is_pred_positive and is_label_positive:
            result["pred"] = "Correct"; evals[0] += 1
        elif is_pred_positive and not is_label_positive:
            result["pred"] = "Incorrect"; evals[1] += 1
        elif not is_pred_positive and not is_label_positive:
            result["pred"] = "Correct"; evals[2] += 1
        else:
            result["pred"] = "Incorrect"; evals[3] += 1
    return evals, total, results


def _evaluation_dict(evals: List[int], total: int) -> dict:
    tp, fp, tn, fn = evals
    out = {"total": total, "true_positive": tp, "false_positive": fp,
           "true_negative": tn, "false_negative": fn}
    if total > 0:
        out["accuracy"] = (tp + tn) / total
        out["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        out["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        out["f1_score"] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    return out


def _print_evaluation(ev: dict) -> None:
    if ev["total"] == 0:
        print("No valid results to evaluate.")
        return
    print(f"Raw counts: TP: {ev['true_positive']}, FP: {ev['false_positive']},  "
          f"TN: {ev['true_negative']}, FN: {ev['false_negative']}")
    print(f"Accuracy: {ev['accuracy']:.3f}")
    print(f"Precision: {ev['precision']:.3f}")
    print(f"Recall: {ev['recall']:.3f}")
    print(f"F1 Score: {ev['f1_score']:.3f}")


def print_fail_cases(results: List[dict], case_type: str = "ALL", max_display: int = 5) -> None:
    fail_cases = {"FP": [], "FN": []}
    for idx, result in enumerate(results):
        if result.get("pred") != "Incorrect":
            continue
        is_pred_positive = pred_positive(result)
        is_label_positive = result.get("label", "").lower() == "positive"
        if is_pred_positive and not is_label_positive:
            fail_cases["FP"].append((idx, result))
        elif not is_pred_positive and is_label_positive:
            fail_cases["FN"].append((idx, result))

    for ctype in (["FP", "FN"] if case_type == "ALL" else [case_type]):
        cases = fail_cases[ctype]
        case_name = "False Positive" if ctype == "FP" else "False Negative"
        print("\n" + "=" * 80)
        print(f"{case_name.upper()} CASES ({len(cases)} total)")
        print("=" * 80)
        for i, (idx, result) in enumerate(cases[:max_display]):
            print(f"\n{'─' * 80}\nCASE #{i + 1} (Index: {idx})\n{'─' * 80}")
            print(f"Label:      {result.get('label', 'N/A')}")
            print(f"Prediction: {'Positive' if pred_positive(result) else 'Negative'} "
                  f"(num_unique={result.get('num_unique_approaches', 'N/A')})\n")
            problem = result.get("problem", "N/A")
            print(f"[PROBLEM]\n{problem[:300]}{'...' if len(problem) > 300 else ''}\n")
            approaches = result.get("response", {}).get("approaches", [])
            print(f"[APPROACHES] ({len(approaches)} approaches)")
            for j, approach in enumerate(approaches, 1):
                text = approach.get("name", "N/A")
                print(f"  {j}. {text[:200]}{'...' if len(text) > 200 else ''}")
            print(f"\n[EXPLANATION]\n{result.get('explanation', 'N/A')}\n")
        if len(cases) > max_display:
            print(f"\n... and {len(cases) - max_display} more {case_name} cases not displayed.")
        print("=" * 80)


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

def _save_positive(results: List[dict], output_file: str, metadata: dict) -> None:
    positive_samples = [r for r in results if pred_positive(r)]
    base_name, ext = os.path.splitext(output_file)
    positive_output_file = f"{base_name}_positive{ext}"
    print(f"Found {len(positive_samples)} positive samples.")
    with open(positive_output_file, "w") as f:
        json.dump({"results": positive_samples,
                   "metadata": {**metadata, "total_positive": len(positive_samples)}},
                  f, indent=2)
    print(f"Positive samples saved to: {positive_output_file}")


def _save_raw(batch_results: List[dict], output_file: str, tag: str) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.dirname(os.path.abspath(output_file))
    raw_path = os.path.join(out_dir, f"{tag}_results_raw_{timestamp}.jsonl")
    with open(raw_path, "w") as f:
        for result in batch_results:
            f.write(json.dumps(result) + "\n")
    print(f"Raw {tag} results saved to: {raw_path}")
    return raw_path


def _print_cost(cost_info: dict, header: str) -> None:
    print("\n" + "=" * 60)
    print(header)
    print("=" * 60)
    print(f"Total Input Tokens:  {cost_info['total_input_tokens']:>12,}")
    print(f"Total Output Tokens: {cost_info['total_output_tokens']:>12,}")
    print(f"Total Tokens:        {cost_info['total_tokens']:>12,}")
    print("-" * 60)
    print(f"Input Cost:          ${cost_info['input_cost']:>11.4f}")
    print(f"Output Cost:         ${cost_info['output_cost']:>11.4f}")
    print(f"Total Cost:          ${cost_info['total_cost']:>11.4f}")
    print("=" * 60 + "\n")


def run_eval_only(output_file: str, do_eval: bool, show_fail_cases: bool,
                  fail_case_type: str, max_fail_cases: int) -> None:
    print("=" * 60 + "\nEVALUATION ONLY MODE\n" + "=" * 60)
    print(f"Loading existing results from: {output_file}")
    with open(output_file, "r") as f:
        output_data = json.load(f)
    results = output_data.get("results", [])
    print(f"Loaded {len(results)} results")
    if output_data.get("cost_info"):
        _print_cost(output_data["cost_info"], "API COST REPORT (from previous run)")

    if do_eval:
        evals, total, results = evaluate_judge_result(results)
        ev = _evaluation_dict(evals, total)
        _print_evaluation(ev)
        if show_fail_cases and total > 0:
            print_fail_cases(results, fail_case_type, max_fail_cases)
        output_data["results"] = results
        output_data["evaluation"] = ev
    else:
        _save_positive(results, output_file, output_data.get("metadata", {}))
        output_data.pop("evaluation", None)

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Updated results saved to: {output_file}")


def run_batch_judgment(input_file: str, output_file: str, model: str,
                       dry_run: bool, batch_id: Optional[str],
                       show_fail_cases: bool, fail_case_type: str, max_fail_cases: int,
                       num_votes: int, raw_results_file: Optional[str],
                       reasoning_effort: Optional[str], do_eval: bool,
                       realtime: bool, num_workers: int) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    print(f"Loading input data from: {input_file}")
    data = load_results(input_file)
    print(f"Loaded {len(data)} items")

    if realtime:
        batch_results = run_realtime_requests(data, model, num_votes, reasoning_effort, num_workers)
        _save_raw(batch_results, output_file, "realtime")
    elif raw_results_file:
        print(f"Loading raw batch results from: {raw_results_file}")
        with open(raw_results_file, "r") as f:
            batch_results = [json.loads(line) for line in f if line.strip()]
        print(f"Loaded {len(batch_results)} raw results")
    elif batch_id:
        print(f"Tracking existing batch with ID: {batch_id}")
        batch_results = wait_for_batch_completion(get_openai_client(), batch_id)
        _save_raw(batch_results, output_file, "batch")
    else:
        batch_requests = create_batch_requests(data, model, num_votes, reasoning_effort)
        print(f"Created {len(batch_requests)} batch requests "
              f"({len(data)} items x {num_votes} vote(s))")
        if not batch_requests:
            print("No valid requests to process.")
            return
        if dry_run:
            print("Dry run mode. Exiting...")
            return
        file_id = upload_batch_file(get_openai_client(), batch_requests)
        print(f"File uploaded: {file_id}")
        batch_id = create_batch_job(get_openai_client(), file_id)
        print(f"Batch job created with ID: {batch_id}")
        batch_results = wait_for_batch_completion(get_openai_client(), batch_id)
        _save_raw(batch_results, output_file, "batch")

    results = process_batch_results(batch_results, data, num_votes=num_votes)

    cost_info = calculate_cost(batch_results, model=model)
    _print_cost(cost_info, f"API COST REPORT ({model}, {'realtime' if realtime else 'batch (50% discount)'})")

    if num_votes > 1:
        unanimous = majority = tied = 0
        for result in results:
            vd = result.get("voting_details")
            if vd and vd.get("valid_votes", 0) > 0:
                if vd["positive_votes"] == vd["valid_votes"] or vd["negative_votes"] == vd["valid_votes"]:
                    unanimous += 1
                elif vd["positive_votes"] == vd["negative_votes"]:
                    tied += 1
                else:
                    majority += 1
        print(f"MAJORITY VOTING SUMMARY: items={len(results)} unanimous={unanimous} "
              f"majority={majority} tied={tied}\n")

    metadata = {"model": model, "reasoning_effort": reasoning_effort,
                "num_votes": num_votes, "total_items": len(data)}
    output_data = {"results": results, "cost_info": cost_info, "metadata": metadata}

    if do_eval:
        evals, total, results = evaluate_judge_result(results)
        ev = _evaluation_dict(evals, total)
        _print_evaluation(ev)
        if show_fail_cases and total > 0:
            print_fail_cases(results, fail_case_type, max_fail_cases)
        output_data["results"] = results
        output_data["evaluation"] = ev
    else:
        _save_positive(results, output_file, metadata)

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Results saved to: {output_file}")
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Approach-uniqueness judge (LLM)")
    parser.add_argument("--input-file", type=str,
                        help="Input file (json or jsonl) with problem and response.approaches")
    parser.add_argument("--output-file", type=str, required=True, help="Output JSON file")
    parser.add_argument("--model", type=str, default="gpt-5.1", help="Judge model (default: gpt-5.1)")
    parser.add_argument("--reasoning-effort", type=str, default="low",
                        choices=["none", "low", "medium", "high"],
                        help="Reasoning effort for the judge model (default: low)")
    parser.add_argument("--num-votes", type=int, default=1,
                        help="Independent votes per item for majority voting (default: 1)")
    parser.add_argument("--dry-run", action="store_true", help="Build requests but do not submit")
    parser.add_argument("--batch-id", type=str, default=None, help="Resume an existing OpenAI batch")
    parser.add_argument("--raw-results-file", type=str, default=None,
                        help="Process saved raw results instead of calling the API")
    parser.add_argument("--realtime", action="store_true",
                        help="Query synchronously instead of using the Batch API")
    parser.add_argument("--num-workers", type=int, default=8, help="Workers for --realtime (default: 8)")
    parser.add_argument("--eval", action="store_true",
                        help="Evaluate predictions against a 'label' field (positive/negative)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only re-process an existing output file (no API calls)")
    parser.add_argument("--no-fail-cases", dest="show_fail_cases", action="store_false",
                        help="Do not print FP/FN cases after --eval")
    parser.add_argument("--fail-case-type", type=str, default="ALL", choices=["FP", "FN", "ALL"])
    parser.add_argument("--max-fail-cases", type=int, default=5)
    args = parser.parse_args()

    if args.eval_only:
        if not os.path.exists(args.output_file):
            print(f"Error: --eval-only requires existing output file at {args.output_file}")
            sys.exit(1)
        run_eval_only(args.output_file, args.eval, args.show_fail_cases,
                      args.fail_case_type, args.max_fail_cases)
        return

    if not args.input_file:
        print("Error: --input-file is required unless using --eval-only")
        sys.exit(1)

    run_batch_judgment(
        input_file=args.input_file,
        output_file=args.output_file,
        model=args.model,
        dry_run=args.dry_run,
        batch_id=args.batch_id,
        show_fail_cases=args.show_fail_cases,
        fail_case_type=args.fail_case_type,
        max_fail_cases=args.max_fail_cases,
        num_votes=args.num_votes,
        raw_results_file=args.raw_results_file,
        reasoning_effort=args.reasoning_effort,
        do_eval=args.eval,
        realtime=args.realtime,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
