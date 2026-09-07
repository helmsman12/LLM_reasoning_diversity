#!/usr/bin/env python3
"""
LLM-judge correctness scoring of generated solutions (Qwen3-4B judge).

Reads a JSONL file with one problem per line (`question`, `answer`,
`solutions`, optional `pred`), asks a vLLM-served judge model whether each
solution's final answer is equivalent to the golden answer, and writes the
same records with a `scores` field (1.0 correct / 0.0 incorrect). The scored
file is the input format expected by `sol_diversity_judge.py`.

Usage:
    python verify_solutions.py generations.jsonl \
        --model-name Qwen/Qwen3-4B --api-base http://localhost:9000/v1
"""

import json
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

BATCH_SIZE = 1024


def extract_boxed_answer(response: str) -> Optional[str]:
    """Extract answer from \\boxed{...} notation with proper nested brace handling."""
    idx = response.find("\\boxed{")
    if idx == -1:
        return None
    end_idx = idx + 7
    depth = 1
    while end_idx < len(response):
        if response[end_idx] == "{":
            depth += 1
        elif response[end_idx] == "}":
            depth -= 1
        if depth == 0:
            break
        end_idx += 1
    ret = response[idx + 7:end_idx]
    return ret


SYSTEM_PROMPT = """
        You are a math expert acting as a strict judge.
        You will be given a "Golden Answer" and a "Predicted Answer".
        Your task is to verify if the predicted answer is mathematically equivalent to the golden answer.

        Please follow these steps:
        1. Identify the final value or expression in the "Predicted Answer".
        2. Compare it against the "Golden Answer".
        - Consider mathematical equivalence (e.g., 0.5 is equal to 1/2).
        - Ignore minor formatting differences (e.g., "x = 5" matches "5").
        3. Provide a brief explanation of your reasoning.
        4. Conclude with the final result in this exact format: "Verification: [correct]" or "Verification: [incorrect]".
    """

FEW_SHOT_PAIRS = [
    (
        "Golden answer: 540\nPredicted answer: The total number of ways the cars can stack up so that all three lanes are occupied is calculated to be 750.",
        "Reasoning: The golden answer is 540. The predicted answer explicitly states the calculated value is 750. Since 750 is not equal to 540, the prediction is wrong.\nVerification: [incorrect]",
    ),
    (
        "Golden answer: 3\nPredicted answer: The ratio \\\\frac{A C}{A E} = 3.",
        "Reasoning: The golden answer is 3. The predicted answer identifies the ratio as 3. Although it includes the variable name, the numerical value matches exactly.\nVerification: [correct]",
    ),
    (
        "Golden answer: \\\\frac{1}{2}\nPredicted answer: The probability is 0.5.",
        "Reasoning: The golden answer is 1/2. The predicted answer is 0.5. Since 0.5 is mathematically equivalent to the fraction 1/2, the answer is correct.\nVerification: [correct]",
    ),
]


def build_verification_prompt(tokenizer, problem: str, answer: str, pred: str) -> str:
    """Apply chat template to build a raw text prompt for the completions endpoint."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_msg, asst_msg in FEW_SHOT_PAIRS:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": asst_msg})
    messages.append({"role": "user", "content": f"Problem: {problem}, Golden answer: {answer}\nPredicted answer: {pred}"})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def verify_batch(
    client: OpenAI,
    model_name: str,
    tokenizer,
    batch: List[Tuple[int, int, str, str, str, str]],
    max_retries: int = 3
) -> List[Tuple[int, int, bool]]:
    """Send a batch of verification tasks to the /v1/completions endpoint."""
    prompts, task_ids = [], []
    for prob_idx, sol_idx, question, answer, pred, solution in batch:
        final_pred = pred.strip() if pred and pred.strip() else (extract_boxed_answer(solution) or solution)
        prompts.append(build_verification_prompt(tokenizer, question, answer, final_pred))
        task_ids.append((prob_idx, sol_idx))

    for attempt in range(max_retries):
        try:
            response = client.completions.create(
                model=model_name,
                prompt=prompts,
                temperature=0.7,
                top_p=0.8,
                presence_penalty=0,
                max_tokens=512,
                extra_body={"top_k": 20}
            )
            choices = sorted(response.choices, key=lambda c: c.index)
            return [(*task_ids[i], "Verification: [correct]" in c.text)
                    for i, c in enumerate(choices)]
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                logger.warning(f"Batch failed (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                logger.error(f"Batch failed after {max_retries} attempts: {e}")
                return [(*tid, False) for tid in task_ids]

    return [(*tid, False) for tid in task_ids]


def score_jsonl_file(
    input_path: str,
    output_path: str,
    client: OpenAI,
    model_name: str,
) -> Dict[str, Any]:
    """
    Score a JSONL file by verifying each solution against the ground truth.
    Uses parallel processing for verification calls.

    Returns:
        Dict of summary statistics
    """
    stats = {
        "problems_total": 0,
        "solutions_total": 0,
        "solutions_correct": 0,
        "solutions_incorrect": 0,
    }

    logger.info(f"Reading input file: {input_path}")
    logger.info(f"Output will be written to: {output_path}")

    # Step 1: Read all problems into memory
    problems = []
    with open(input_path, 'r') as infile:
        for line_num, line in enumerate(infile, 1):
            try:
                problem = json.loads(line.strip())
                problems.append(problem)
            except json.JSONDecodeError as e:
                logger.error(f"Line {line_num}: Invalid JSON - {e}")

    stats["problems_total"] = len(problems)
    logger.info(f"Loaded {len(problems)} problems")

    # Step 2: Build flat list of verification tasks
    tasks: List[Tuple[int, int, str, str, str, str]] = []
    for prob_idx, problem in enumerate(problems):
        question = problem.get("question", "")
        answer = problem.get("answer", "")
        solutions = problem.get("solutions", [])
        preds = problem.get("pred", [])

        stats["solutions_total"] += len(solutions)

        for sol_idx, solution in enumerate(solutions):
            pred = preds[sol_idx] if sol_idx < len(preds) else ""
            tasks.append((prob_idx, sol_idx, question, answer, pred, solution))

    logger.info(f"Built {len(tasks)} verification tasks across {len(problems)} problems")

    # Load tokenizer once (transformers uses HF cache; model already downloaded for vLLM)
    logger.info(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Step 3: Submit tasks in batches via completions API
    results: List[Tuple[int, int, bool]] = []
    total_tasks = len(tasks)
    num_batches = (total_tasks + BATCH_SIZE - 1) // BATCH_SIZE
    logger.info(f"Starting batch verification: {total_tasks} tasks, batch size {BATCH_SIZE}, {num_batches} batches...")
    start_time = time.time()

    with tqdm(total=total_tasks, desc="Verifying", unit="task") as pbar:
        for batch_idx in range(num_batches):
            batch = tasks[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]
            batch_results = verify_batch(client, model_name, tokenizer, batch)
            results.extend(batch_results)
            pbar.update(len(batch))

    elapsed = time.time() - start_time
    logger.info(f"All {total_tasks} verification tasks completed in {elapsed:.1f}s")

    # Step 4: Reassemble scores per problem
    # Initialize scores lists
    scores_by_problem: Dict[int, Dict[int, float]] = {}
    for prob_idx, sol_idx, is_correct in results:
        if prob_idx not in scores_by_problem:
            scores_by_problem[prob_idx] = {}
        scores_by_problem[prob_idx][sol_idx] = 1.0 if is_correct else 0.0
        if is_correct:
            stats["solutions_correct"] += 1
        else:
            stats["solutions_incorrect"] += 1

    # Step 5: Write output JSONL
    with open(output_path, 'w') as outfile:
        for prob_idx, problem in enumerate(problems):
            solutions = problem.get("solutions", [])
            sol_scores = scores_by_problem.get(prob_idx, {})
            # Build scores list ordered by sol_idx
            scores = [sol_scores.get(i, 0.0) for i in range(len(solutions))]

            scored_problem = problem.copy()
            scored_problem["scores"] = scores
            outfile.write(json.dumps(scored_problem) + '\n')

            correct_count = sum(1 for s in scores if s == 1.0)
            logger.debug(
                f"Problem {prob_idx}: {correct_count}/{len(solutions)} correct"
            )

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Score JSONL evaluation files by verifying solutions against ground truth"
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to input JSONL file"
    )
    parser.add_argument(
        "-o", "--output-file",
        type=str,
        default=None,
        help="Path to output JSONL file (default: input with '_scored' suffix)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3-4B",
        help="vLLM-served judge model"
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default="http://localhost:8000/v1",
        help="vLLM API server base URL"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity"
    )

    args = parser.parse_args()

    # Configure logging level
    logger.setLevel(getattr(logging, args.log_level))

    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error(f"Input file does not exist: {input_path}")
        return 1

    # Determine output path
    if args.output_file:
        output_path = Path(args.output_file)
    else:
        output_path = input_path.parent / f"{input_path.stem}_scored{input_path.suffix}"

    # Initialize OpenAI client
    logger.info(f"Connecting to vLLM API at {args.api_base}")
    logger.info(f"Using model: {args.model_name}")
    client = OpenAI(base_url=args.api_base, api_key="EMPTY")

    # Process file
    logger.info("Starting scoring process...")
    stats = score_jsonl_file(str(input_path), str(output_path), client, args.model_name)

    # Print summary statistics
    logger.info("\n" + "="*60)
    logger.info("SCORING SUMMARY")
    logger.info("="*60)
    logger.info(f"Problems processed:      {stats['problems_total']}")
    logger.info(f"Solutions total:         {stats['solutions_total']}")
    logger.info(f"Solutions correct:       {stats['solutions_correct']}")
    logger.info(f"Solutions incorrect:     {stats['solutions_incorrect']}")

    if stats['solutions_total'] > 0:
        accuracy = stats['solutions_correct'] / stats['solutions_total'] * 100
        logger.info(f"Overall accuracy:        {accuracy:.2f}%")

    logger.info("="*60)
    logger.info(f"Output written to: {output_path}")

    return 0


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        exit(1)
