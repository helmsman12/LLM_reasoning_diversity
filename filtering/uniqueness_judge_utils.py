"""Prompt, parsing and cost helpers for the approach-uniqueness judge.

Used by ``uniqueness_judge.py``. The judge receives one problem together
with its candidate approach plans and reports how many *genuinely
distinct* approaches the plans represent.
"""

import json
from typing import List, Optional

# ---------------------------------------------------------------------------
# Pricing (USD per token). Batch API prices already include the 50% discount.
# Update these if you use a different judge model.
# ---------------------------------------------------------------------------
GPT_5_2_INPUT_PRICE = 1.75 / 1_000_000 * 0.5
GPT_5_2_OUTPUT_PRICE = 14.0 / 1_000_000 * 0.5

MODEL_PRICING = {
    "gpt-5.2": (GPT_5_2_INPUT_PRICE, GPT_5_2_OUTPUT_PRICE),
}

# A problem is "positive" (kept) when the judge finds at least this many
# distinct approaches.
POSITIVE_THRESHOLD = 3


def calculate_cost(batch_results: List[dict], model: str = "gpt-5.2") -> dict:
    """Sum token usage over batch-format results and convert to USD."""
    total_input_tokens = 0
    total_output_tokens = 0
    for batch_result in batch_results:
        if batch_result.get("response", {}).get("status_code") == 200:
            usage = batch_result.get("response", {}).get("body", {}).get("usage", {})
            total_input_tokens += usage.get("prompt_tokens", 0)
            total_output_tokens += usage.get("completion_tokens", 0)

    input_price, output_price = GPT_5_2_INPUT_PRICE, GPT_5_2_OUTPUT_PRICE
    for prefix, (in_p, out_p) in MODEL_PRICING.items():
        if model.startswith(prefix):
            input_price, output_price = in_p, out_p
            break

    input_cost = total_input_tokens * input_price
    output_cost = total_output_tokens * output_price
    return {
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
    }


def pred_positive(result: dict) -> bool:
    n = result.get("num_unique_approaches")
    return n is not None and n >= POSITIVE_THRESHOLD


def extract_boxed_answer(text: str) -> Optional[str]:
    """Return the content of the first ``\\boxed{...}`` (nested braces handled)."""
    idx = text.find("\\boxed{")
    if idx == -1:
        return None
    end_idx = idx + 7
    depth = 1
    while end_idx < len(text):
        if text[end_idx] == "{":
            depth += 1
        elif text[end_idx] == "}":
            depth -= 1
        if depth == 0:
            break
        end_idx += 1
    return text[idx + 7:end_idx]


def parse_num_unique(response_content: str) -> Optional[int]:
    """Parse the judge's ``\\boxed{n}`` answer into an int (None if unparseable)."""
    boxed = extract_boxed_answer(response_content)
    try:
        return int(boxed)
    except (TypeError, ValueError):
        return None


def plans_to_string(plan: List[str]) -> str:
    return "".join(f"Step {i + 1}: {step}\n" for i, step in enumerate(plan))


def build_user_prompt(problem: str, approaches: List[dict]) -> str:
    usr_msg = f"Problem: {problem}\n\n"
    for i, approach in enumerate(approaches):
        usr_msg += f"<Approach {i + 1}>:\n"
        usr_msg += f"*Name*: {approach['name']}\n"
        usr_msg += f"*Core Principle*: {approach['core_idea']}\n"
        usr_msg += f"*Plan Steps*:\n {plans_to_string(approach['plan'])}\n"
        usr_msg += "\n"
    return usr_msg


def build_message(problem: str, approaches: List[dict]) -> List[dict]:
    """Build the chat messages for one judge call."""
    return [
        {"role": "system", "content": UNIQUENESS_JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(problem, approaches)},
    ]


def load_results(file_path: str) -> List[dict]:
    """Load a ``.json`` (list) or ``.jsonl`` file."""
    if file_path.endswith(".json"):
        with open(file_path, "r") as f:
            return json.load(f)
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


UNIQUENESS_JUDGE_SYSTEM_PROMPT = """
<identity>
You are an analytical judge for mathematical solution plans.
Your purpose is to determine whether two or more solution plans
for the same math problem use the same underlying approach
or fundamentally different mechanisms or interpretations.
You communicate in a clear, direct, and structured way.
Your goal is to make reliable, mechanism-level judgments.

<task_overview>
You will be given a math problem and a list of solution approaches. You must examine all provided solution approaches, identify which approaches share
the same core mechanism and interpretation, group them into clusters, and report the total number of unique approaches.

*Definition of different approaches*:
When determining whether two plans represent the same or different approaches, focus on the underlying mathematical mechanism **and** the conceptual interpretation used in the reasoning.
Two plans must be classified as different approaches if they rely on different mathematical tools,
different definitions or theoretical structures, or different representational viewpoints (e.g., vector-based, geometric, algebraic, functional, or symmetry-based interpretations).

Examples of distinct interpretations include:
- Viewing lines as geometric objects and constructing a triangle vs treating lines as vectors in linear algebra.
- Using slope as a trigonometric tangent quantity vs using direction angles with the x-axis.
- Treating a sequence via its explicit formula vs analyzing it through recurrence, symmetry, or linear-function view.

Do not merge plans just because they belong to the same domain.
Different interpretations or representational viewpoints count as different approaches.

<instructions>
Before analyzing any of the plans, first evaluate the problem itself.
Decide whether the problem is sufficiently complex to allow multiple, genuinely distinct solution approaches.
If the problem does not support meaningful approach diversity, apply the simplicity rule.

*simplicity_rule*:
Before evaluating any plan, first judge the inherent structure of the problem itself.
If the problem can be solved through a single dominant method that is standard,
forced, or mechanically determined, then treat all plans as the same approach.

A problem should be treated as "simple" (and unique approaches = 1) if:
- it reduces directly to writing one standard equation and solving it (linear, quadratic, or simple rational equation),
- the solution follows automatically from a basic definition or identity,
- only one inequality or one standard condition (triangle inequality, discriminant condition,
  distance formula, midpoint formula, Vieta, etc.) is required,

If the problem fits one of these conditions, set num_unique_approaches = 1 and return the decision.

Otherwise, follow the instructions below to compare different approaches.

1. Recognize the core mechanism of each approach.
   - Recognize the key idea, transformation, lemma, or strategy.

2. Compare the mechanisms.
   - Treat two plans as different approaches if they use different mathematical tools, structures, or representational viewpoints—even when they compute the same quantity.
   - If two plans rely on the same mathematical essence, categorize them as the same approach.

3. Count the number of unique approaches.
   - The number of unique approaches is the number of different mechanisms used.

4. Provide a brief explanation.
   - Explain the reasoning behind the decision, focusing only on the mechanisms use

5. Do NOT do the following:
   - Do not solve the problem.
   - Do not judge correctness beyond feasibility of the steps.
   - Do not consider stylistic differences.
   - Do not introduce new plans or speculate about hidden steps.
   - Do not combine or rewrite the plans.

<context>
You will be given:
- the original math problem
- a list of candidate approaches (each with title, core idea, stepwise plan steps)
Use only the provided context to make judgments.

<output format>
Provide your final answer in the following structure:

EXPLANATION:
A brief explanation of how you grouped the plans and why.
You may discuss similarities, differences, and mechanism-level reasoning.

NUMBER OF UNIQUE APPROACHES:
\\boxed{{n}}

Where n is the number of unique approaches you identified.
"""
