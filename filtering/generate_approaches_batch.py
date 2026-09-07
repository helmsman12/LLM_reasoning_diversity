import argparse
import json
import os
import tempfile
import time
from typing import Dict, List
from pathlib import Path

# Load OPENAI_API_KEY (and friends) from the package-level .env, if present.
# Variables already set in the environment are never overridden.
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
except ImportError:  # python-dotenv is optional; fall back to `source .env`
    pass

from openai import OpenAI
from tqdm import tqdm

APPROACH_SYSTEM_PROMPT = """
You are an expert strategy enumerator for math and algorithmic problems.

TASK
- For ONE given problem and an integer K, list up to K GENUINELY DISTINCT solving approaches.
- Return strategic plans only (high-level steps). Do NOT compute the final answer or show hidden internal reasoning.

WHAT COUNTS AS “DISTINCT”
- Same: share the same core mechanism, only differ in stylistic/verbal manner.
- Different: use a different paradigm/reduction/decomposition, feasibility oracle, proof style (direct/induction/contradiction), or key transformation.

SCOPE & CONTENT
- Each approach: short title + 3–8 bullet steps describing the plan.
- Assume that the problem is only solvable by hand: one cannot access other tools such as writing a computer program
- Keep concise, technical, and non-redundant. No paraphrase-only variants.

QUALITY BAR
- Before writing, brainstorm several candidate families.
- Merge/drop near-duplicates; output only truly distinct approaches. If fewer than K exist, return fewer.
"""

def build_message(problem: str, k: int = 4):
    usr_msg = f"K: {k}\nProblem: {problem}"
    return [
        {"role": "system", "content": APPROACH_SYSTEM_PROMPT},
        {"role": "user", "content": usr_msg}
    ]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Filtered problems JSONL",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="",
        help="Output JSONL with approaches (default: input_file with _with_plans.jsonl)",
    )
    parser.add_argument("--model", type=str, default="gpt-5.1")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--max_completion_tokens", type=int, default=32768)
    parser.add_argument("--poll_interval", type=int, default=120)
    parser.add_argument(
        "--raw_results_dir",
        type=str,
        default="raw_batch_results",
    )
    parser.add_argument(
        "--batch_id",
        type=str,
        default="",
        help="Existing batch ID to retrieve results from (skip submission)",
    )
    return parser.parse_args()


def json_schema() -> Dict[str, object]:
    return {
        "name": "unique_approaches",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "problem_brief": {
                    "type": "string",
                    "description": "one-sentence restatement of the task",
                },
                "approaches": {
                    "type": "array",
                    "description": "suggested approaches for the given problem",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "short, specific title capturing the core idea",
                            },
                            "core_idea": {
                                "type": "string",
                                "description": "2-3 sentences describing the key mechanism or reduction",
                            },
                            "plan": {
                                "type": "array",
                                "description": "3-8 high-level steps. Each item is a short imperative action.",
                                "items": {"type": "string"},
                            },
                            "uniqueness_signature": {
                                "type": "string",
                                "description": "one line that makes this approach different from others",
                            },
                        },
                        "required": ["name", "core_idea", "plan", "uniqueness_signature"],
                    },
                },
                "num_approaches": {
                    "type": "integer",
                    "description": "total number of suggested approaches",
                },
            },
            "required": ["problem_brief", "approaches", "num_approaches"],
        },
    }


def upload_batch_file(client: OpenAI, batch_requests: List[dict]) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for request in batch_requests:
            f.write(json.dumps(request) + "\n")
        temp_file_path = f.name

    try:
        with open(temp_file_path, "rb") as f:
            file_response = client.files.create(
                file=f,
                purpose="batch",
            )
        return file_response.id
    finally:
        os.unlink(temp_file_path)


def create_batch_job(client: OpenAI, file_id: str) -> str:
    batch_response = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    return batch_response.id


def wait_for_batch_completion(
    client: OpenAI,
    batch_id: str,
    poll_interval: int,
) -> List[dict]:
    print(f"Batch job {batch_id} created. Waiting for completion...")

    while True:
        batch_status = client.batches.retrieve(batch_id)
        print(f"Batch status: {batch_status.status}")

        if batch_status.status == "completed":
            break
        if batch_status.status == "failed":
            raise RuntimeError(f"Batch job failed: {batch_status.errors}")
        if batch_status.status == "expired":
            raise RuntimeError("Batch job expired")
        if batch_status.status == "cancelled":
            raise RuntimeError("Batch job was cancelled")

        time.sleep(poll_interval)

    results_file_id = batch_status.output_file_id
    if results_file_id is None:
        raise RuntimeError(
            "Batch completed but output_file_id is None. "
            f"Request counts: {batch_status.request_counts}"
        )

    results_file = client.files.content(results_file_id)
    results = []
    for line in results_file.text.split("\n"):
        if line.strip():
            results.append(json.loads(line))
    return results


def create_batch_requests(
    problem_data: List[dict],
    model: str,
    k: int,
    max_completion_tokens: int,
) -> List[dict]:
    batch_requests = []
    schema = json_schema()

    for idx, data in enumerate(tqdm(problem_data, desc="Preparing requests")):
        problem_key = "problem" if "problem" in data else "question"
        problem = data[problem_key]
        msg = build_message(problem, k=k)

        request = {
            "custom_id": f"problem_{idx}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": msg,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": schema,
                },
                "max_completion_tokens": max_completion_tokens,
            },
        }
        batch_requests.append(request)

    return batch_requests


def process_batch_results(
    batch_results: List[dict],
    problem_data: List[dict],
) -> List[dict]:
    problem_results: Dict[int, dict] = {}
    for result in batch_results:
        if result.get("response", {}).get("status_code") != 200:
            print(f"Error in result: {result}")
            continue

        custom_id = result["custom_id"]
        problem_idx = int(custom_id.split("_")[1])
        response_content = result["response"]["body"]["choices"][0]["message"]["content"]
        try:
            parsed_response = json.loads(response_content)
        except json.JSONDecodeError:
            print(f"Failed to parse JSON for problem {problem_idx}")
            continue
        problem_results[problem_idx] = parsed_response

    output_records = []
    for idx, data in enumerate(problem_data):
        if idx not in problem_results:
            print(f"Warning: no response for problem {idx}")
            continue
        problem_key = "problem" if "problem" in data else "question"
        record = {
            "index": idx,
            "problem": data[problem_key],
            "answer": data["answer"],
            "response": problem_results[idx],
        }
        output_records.append(record)

    return output_records


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    output_file = args.output_file or args.input_file.replace(".jsonl", "_with_plans.jsonl")

    with open(args.input_file, "r") as f:
        problem_data = [json.loads(line) for line in f if line.strip()]

    client = OpenAI(api_key=api_key)

    if args.batch_id:
        batch_id = args.batch_id
        print(f"Resuming existing batch: {batch_id}")
    else:
        batch_requests = create_batch_requests(
            problem_data=problem_data,
            model=args.model,
            k=args.k,
            max_completion_tokens=args.max_completion_tokens,
        )

        file_id = upload_batch_file(client, batch_requests)
        batch_id = create_batch_job(client, file_id)
        print(f"Batch job created: {batch_id}")

    batch_results = wait_for_batch_completion(
        client,
        batch_id,
        poll_interval=args.poll_interval,
    )
    print(f"Batch completed. Processing {len(batch_results)} results...")

    os.makedirs(args.raw_results_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    raw_results_file = os.path.join(
        args.raw_results_dir,
        f"approaches_batch_{timestamp}.jsonl",
    )
    with open(raw_results_file, "w") as f:
        for result in batch_results:
            f.write(json.dumps(result) + "\n")
    print(f"Saved raw batch results to {raw_results_file}")

    output_records = process_batch_results(batch_results, problem_data)
    with open(output_file, "w") as f:
        for record in output_records:
            f.write(json.dumps(record) + "\n")
    print(f"Wrote {len(output_records)} records to {output_file}")


if __name__ == "__main__":
    main()
