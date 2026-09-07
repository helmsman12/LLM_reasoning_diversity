#!/usr/bin/env python3
"""
GPT Solution Diversity Judge

Evaluates GPT-based solution clustering using:
- Input: JSONL files with problems, solutions, and golden groupings
- Model: Configurable (default: gpt-5.2)
- Prompt: ACTIVE_SYSTEM_PROMPT (conservative clustering prompt) from utils.py
- Metrics: ARI, Homogeneity, Completeness (sklearn)
- Modes: Realtime API calls OR OpenAI Batch API
"""

import argparse
import ast
import concurrent.futures
import json
import logging
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Load OPENAI_API_KEY (and friends) from the package-level .env, if present.
# Variables already set in the environment are never overridden.
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
except ImportError:  # python-dotenv is optional; fall back to `source .env`
    pass

import numpy as np
from openai import OpenAI
from sklearn.metrics import (
    adjusted_rand_score,
    homogeneity_score,
    completeness_score
)
from tqdm import tqdm

# Import prompt templates from utils.py
from utils import ACTIVE_SYSTEM_PROMPT

REPAIR_SYSTEM_PROMPT = """
You repair malformed clustering outputs for math-solution clustering.

Return exactly one JSON object with this schema:
{
  "reasoning_trace": "...",
  "groups": [
    {
      "group_name": "...",
      "core_idea": "...",
      "solution_ids": [1, 2]
    }
  ]
}

Requirements:
- Use each solution id from 1..N exactly once.
- No duplicates and no missing ids.
- Output plain JSON only, with no markdown fences or extra text.
- Keep strings short and plain-text.
- Preserve the original intended grouping as much as possible.
"""


@dataclass
class ClusteringConfig:
    """Configuration for clustering evaluation."""
    # API Configuration
    api_key: str
    api_base: Optional[str] = None
    model: str = "gpt-5.2"
    mode: str = "realtime"  # "realtime" or "batch"
    reasoning_effort: str = "none"  # "low", "medium", "high"
    thinking_token_budget: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    presence_penalty: Optional[float] = None
    repetition_penalty: Optional[float] = None
    seed: Optional[int] = None

    # Realtime settings
    rate_limit_rpm: int = 60
    max_retries: int = 5
    timeout: int = 120

    # Batch settings
    poll_interval: int = 60

    # General settings
    output_dir: str = "outputs/clustering_results"
    output_prefix: str = "results"
    log_level: str = "INFO"
    limit: Optional[int] = None
    resume_file: Optional[str] = None
    resume_stage1: Optional[str] = None  # Path to raw Stage 1 batch results for resuming
    resume_stage2_input_file_id: Optional[str] = None  # OpenAI file ID for an already-submitted Stage 2 batch input
    quiet: bool = False
    eval: bool = True
    chunk_size: Optional[int] = None  # None = single-stage (default)
    num_workers: int = 1  # Number of parallel workers for realtime mode
    min_correct: int = 0  # Skip problems with fewer than this many correct solutions


@dataclass
class CostTracker:
    """Track API usage and costs (thread-safe).

    Update PRICING dict with actual model pricing from OpenAI.
    Prices are per 1M tokens in USD (realtime pricing).
    Batch API pricing is automatically halved (50% discount).
    """
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_requests: int = 0

    # Model pricing (per 1M tokens) - REALTIME pricing
    # Batch API is automatically 50% cheaper
    # TODO: Update with actual pricing for your models
    # See: https://openai.com/pricing
    PRICING = {
        "gpt-5.2": {"input": 1.75, "output": 14.00},  # Realtime pricing - batch is 50% off
    }

    def __post_init__(self):
        self._lock = threading.Lock()

    def add_usage(self, usage_dict: Dict[str, int]):
        """Add usage from API response (thread-safe)."""
        with self._lock:
            self.input_tokens += usage_dict.get("prompt_tokens", 0)
            self.output_tokens += usage_dict.get("completion_tokens", 0)
            self.reasoning_tokens += usage_dict.get("reasoning_tokens", 0)
            self.total_requests += 1

    def get_cost(self, model: str, is_batch: bool = False) -> float:
        """Calculate total cost in USD.

        Args:
            model: Model name
            is_batch: If True, apply 50% discount for batch API pricing
        """
        # Get pricing for model (default to gpt-5.2 pricing if unknown)
        pricing = self.PRICING.get(model, self.PRICING.get("gpt-5.2", {"input": 1.75, "output": 14.00}))

        input_cost = (self.input_tokens / 1_000_000) * pricing["input"]
        output_cost = (self.output_tokens / 1_000_000) * pricing["output"]

        total_cost = input_cost + output_cost

        # Batch API is 50% cheaper
        if is_batch:
            total_cost *= 0.5

        return total_cost

    def get_summary(self, model: str, is_batch: bool = False) -> Dict[str, Any]:
        """Get detailed cost summary.

        Args:
            model: Model name
            is_batch: If True, apply 50% discount for batch API pricing
        """
        return {
            "total_requests": self.total_requests,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "total_cost_usd": self.get_cost(model, is_batch),
            "pricing_mode": "batch (50% discount)" if is_batch else "realtime"
        }


class GPTClient:
    """Client for interacting with OpenAI API in realtime or batch mode."""

    def __init__(self, config: ClusteringConfig, cost_tracker: Optional[CostTracker] = None):
        self.config = config
        client_kwargs = {"api_key": config.api_key}
        if config.api_base:
            client_kwargs["base_url"] = config.api_base
        self.client = OpenAI(**client_kwargs)
        self.logger = logging.getLogger(__name__)
        self.cost_tracker = cost_tracker or CostTracker()

    def call_realtime(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """Make a single API call with retry logic."""
        max_retries = self.config.max_retries
        wait_time = 1  # Start with 1 second

        for attempt in range(max_retries + 1):
            try:
                kwargs = {
                    "model": self.config.model,
                    "messages": messages,
                    "timeout": self.config.timeout
                }
                if self.config.temperature is not None:
                    kwargs["temperature"] = self.config.temperature
                if self.config.top_p is not None:
                    kwargs["top_p"] = self.config.top_p
                if self.config.presence_penalty is not None:
                    kwargs["presence_penalty"] = self.config.presence_penalty
                if self.config.seed is not None:
                    kwargs["seed"] = self.config.seed
                extra_body = {}
                if self.config.thinking_token_budget is not None:
                    extra_body["thinking_token_budget"] = self.config.thinking_token_budget
                if self.config.top_k is not None:
                    extra_body["top_k"] = self.config.top_k
                if self.config.min_p is not None:
                    extra_body["min_p"] = self.config.min_p
                if self.config.repetition_penalty is not None:
                    extra_body["repetition_penalty"] = self.config.repetition_penalty
                if extra_body:
                    kwargs["extra_body"] = extra_body
                if self.config.reasoning_effort and self.config.reasoning_effort != "none":
                    kwargs["reasoning_effort"] = self.config.reasoning_effort
                response = self.client.chat.completions.create(**kwargs)

                # Track usage
                if hasattr(response, 'usage') and response.usage:
                    usage_dict = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "reasoning_tokens": getattr(response.usage, 'reasoning_tokens', 0)
                    }
                    self.cost_tracker.add_usage(usage_dict)
                # print(sanitize_response_content(response.choices[0].message.content))
                return sanitize_response_content(response.choices[0].message.content)

            except Exception as e:
                error_str = str(e).lower()

                if attempt >= max_retries:
                    self.logger.error(f"Max retries reached. Failed to get response: {e}")
                    return None

                retry_count = attempt + 1

                # Handle rate limits (429)
                if "429" in error_str or "rate limit" in error_str:
                    wait_time = min(60, wait_time * 2)  # Exponential backoff, max 60s
                    self.logger.warning(f"Rate limit hit. Waiting {wait_time}s before retry {retry_count}/{max_retries}")
                    time.sleep(wait_time)

                # Handle timeouts
                elif "timeout" in error_str:
                    wait_time = 5 * retry_count  # 5s, 10s, 15s
                    self.logger.warning(f"Timeout. Waiting {wait_time}s before retry {retry_count}/{max_retries}")
                    time.sleep(wait_time)

                # Handle general API errors
                else:
                    wait_time = min(16, 2 ** retry_count)  # Exponential: 1, 2, 4, 8, 16
                    self.logger.warning(f"API error: {e}. Waiting {wait_time}s before retry {retry_count}/{max_retries}")
                    time.sleep(wait_time)

        return None

    def create_batch_file(self, problems_data: List[Dict], output_path: str) -> str:
        """Create JSONL batch file for OpenAI Batch API."""
        with open(output_path, 'w') as f:
            for idx, problem_data in enumerate(problems_data):
                messages = problem_data['messages']
                body = {
                    "model": self.config.model,
                    "messages": messages,
                }
                if self.config.temperature is not None:
                    body["temperature"] = self.config.temperature
                if self.config.top_p is not None:
                    body["top_p"] = self.config.top_p
                if self.config.presence_penalty is not None:
                    body["presence_penalty"] = self.config.presence_penalty
                if self.config.seed is not None:
                    body["seed"] = self.config.seed
                if self.config.top_k is not None:
                    body["top_k"] = self.config.top_k
                if self.config.min_p is not None:
                    body["min_p"] = self.config.min_p
                if self.config.repetition_penalty is not None:
                    body["repetition_penalty"] = self.config.repetition_penalty
                if self.config.reasoning_effort and self.config.reasoning_effort != "none":
                    body["reasoning_effort"] = self.config.reasoning_effort
                batch_request = {
                    "custom_id": f"problem_{problem_data['problem_idx']}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body
                }
                f.write(json.dumps(batch_request) + '\n')

        self.logger.info(f"Created batch file: {output_path}")
        return output_path

    def submit_batch(self, batch_file_path: str) -> str:
        """Upload and submit batch job."""
        with open(batch_file_path, 'rb') as f:
            batch_input_file = self.client.files.create(
                file=f,
                purpose="batch"
            )

        self.logger.info(f"Uploaded batch file: {batch_input_file.id}")

        batch = self.client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )

        self.logger.info(f"Created batch job: {batch.id}")
        return batch.id

    def poll_batch(self, batch_id: str) -> bool:
        """Poll batch until completion. Always waits for completion."""
        self.logger.info(f"Polling batch {batch_id} every {self.config.poll_interval}s...")

        while True:
            batch = self.client.batches.retrieve(batch_id)
            status = batch.status

            self.logger.info(f"Batch status: {status}")

            if status == "completed":
                self.logger.info("Batch completed successfully")
                return True
            elif status in ["failed", "expired", "cancelled"]:
                self.logger.error(f"Batch failed with status: {status}")
                return False

            time.sleep(self.config.poll_interval)

    def find_batch_by_input_file_id(self, file_id: str) -> Optional[str]:
        """Search recent batches for one whose input_file_id matches file_id. Returns batch ID or None."""
        self.logger.info(f"Searching for batch with input_file_id={file_id} ...")
        try:
            page = self.client.batches.list(limit=100)
            while True:
                for batch in page.data:
                    if batch.input_file_id == file_id:
                        self.logger.info(f"Found batch {batch.id} (status={batch.status})")
                        return batch.id
                if not page.has_more:
                    break
                page = self.client.batches.list(limit=100, after=page.data[-1].id)
        except Exception as e:
            self.logger.error(f"Failed to list batches: {e}")
        self.logger.error(f"No batch found with input_file_id={file_id}")
        return None

    def retrieve_batch_results(self, batch_id: str, output_path: str) -> Optional[str]:
        """Download and save raw batch results to file."""
        try:
            batch = self.client.batches.retrieve(batch_id)

            if batch.output_file_id is None:
                if batch.error_file_id:
                    try:
                        error_response = self.client.files.content(batch.error_file_id)
                        error_lines = error_response.content.decode("utf-8").strip().splitlines()
                        sample = error_lines[0] if error_lines else "(empty)"
                        self.logger.error(
                            f"No output file ID in batch. Batch had {len(error_lines)} failed request(s). "
                            f"First error: {sample}"
                        )
                    except Exception as err_e:
                        self.logger.error(f"No output file ID in batch (could not read error file: {err_e})")
                else:
                    self.logger.error("No output file ID in batch (and no error file ID)")
                return None

            file_response = self.client.files.content(batch.output_file_id)

            with open(output_path, 'wb') as f:
                f.write(file_response.content)

            self.logger.info(f"Saved raw batch results to: {output_path}")
            return output_path

        except Exception as e:
            self.logger.error(f"Failed to retrieve batch results: {e}")
            return None


def sanitize_response_content(content: Any) -> str:
    """Normalize API message content and strip any <think>...</think> blocks."""
    if content is None:
        return ""

    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif item.get("type") == "text" and isinstance(item.get("content"), str):
                    parts.append(item["content"])
        text = "\n".join(parts)
    else:
        text = str(content)

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def build_prompt(
    problem: str,
    solutions: List[str],
    system_prompt: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Build messages for OpenAI API."""
    if system_prompt is None:
        system_prompt = ACTIVE_SYSTEM_PROMPT

    # Build user message with problem and solutions
    user_content_parts = [f"Problem: {problem}\n"]

    for idx, solution in enumerate(solutions, 1):
        user_content_parts.append(f"\nSolution #{idx}: {solution}")

    user_content = "\n".join(user_content_parts)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    return messages


def build_repair_prompt(
    problem: str,
    solutions: List[str],
    bad_response: str,
    issue: str
) -> List[Dict[str, str]]:
    """Build a repair prompt for malformed clustering outputs."""
    user_content_parts = [
        f"Problem: {problem}",
        f"Number of solutions: {len(solutions)}",
        f"Issue: {issue}",
        "Original numbered solutions:"
    ]

    for idx, solution in enumerate(solutions, 1):
        user_content_parts.append(f"\nSolution #{idx}: {solution}")

    user_content_parts.append("\nMalformed draft to repair:")
    user_content_parts.append(bad_response)

    return [
        {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(user_content_parts)}
    ]


def divide_into_chunks(solutions: List[str], chunk_size: int) -> List[List[str]]:
    """Divide solutions into chunks of at most chunk_size."""
    return [solutions[i:i + chunk_size] for i in range(0, len(solutions), chunk_size)]


def aggregate_stage1_results(
    stage1_results: List[Optional[Dict]],
    chunk_offsets: List[int],
    solutions: List[str]
) -> List[Dict]:
    """
    From each successful Stage 1 chunk, pick the first solution in each group as a representative.

    Args:
        stage1_results: List of parsed responses (None for failed chunks), one per chunk.
        chunk_offsets: List of starting indices (0-based) for each chunk in the original solutions list.
        solutions: Full list of original solutions.

    Returns:
        List of representative dicts with keys:
            - original_idx: 0-based index in the original solutions list
            - solution_text: the solution text
            - source_chunk_idx: which chunk this came from
            - source_group_name: group name from Stage 1
            - source_group_members_original: list of original 1-indexed IDs for all members of this Stage 1 group
    """
    representatives = []
    num_solutions = len(solutions)
    for chunk_idx, parsed in enumerate(stage1_results):
        if parsed is None:
            continue
        offset = chunk_offsets[chunk_idx]
        # Compute max valid local_id for this chunk
        next_offset = chunk_offsets[chunk_idx + 1] if chunk_idx + 1 < len(chunk_offsets) else num_solutions
        chunk_len = next_offset - offset
        for group in parsed["groups"]:
            local_ids = group["solution_ids"]  # 1-indexed within chunk
            if not local_ids:
                logging.warning(f"Skipping group '{group.get('group_name', '?')}' with empty solution_ids in chunk {chunk_idx}")
                continue
            # Filter out-of-range local IDs
            valid_local_ids = [lid for lid in local_ids if 1 <= lid <= chunk_len]
            if not valid_local_ids:
                logging.warning(f"Skipping group '{group.get('group_name', '?')}' in chunk {chunk_idx}: all solution_ids out of range (got {local_ids}, chunk has {chunk_len} solutions)")
                continue
            if len(valid_local_ids) < len(local_ids):
                logging.warning(f"Filtered {len(local_ids) - len(valid_local_ids)} out-of-range IDs in group '{group.get('group_name', '?')}' chunk {chunk_idx}")
            # Convert to original 1-indexed IDs
            original_ids = [offset + local_id for local_id in valid_local_ids]
            # Pick first solution as representative
            rep_original_0idx = offset + valid_local_ids[0] - 1  # 0-indexed
            representatives.append({
                "original_idx": rep_original_0idx,
                "original_id": original_ids[0],  # 1-indexed
                "solution_text": solutions[rep_original_0idx],
                "source_chunk_idx": chunk_idx,
                "source_group_name": group["group_name"],
                "source_group_members_original": original_ids  # 1-indexed
            })
    return representatives


def map_stage2_to_final(
    stage2_parsed: Dict,
    representatives: List[Dict],
    num_solutions: int,
    stage1_results: List[Optional[Dict]],
    chunk_offsets: List[int],
    chunk_size: int
) -> Dict[str, List[int]]:
    """
    Map Stage 2 grouping of representatives back to final grouping over all original solutions.

    For each Stage 2 group, expand each representative back to its full Stage 1 group members.

    Returns:
        final_grouping: dict mapping group_name -> list of original 1-indexed solution IDs
    """
    final_grouping = {}
    assigned = set()

    for group in stage2_parsed["groups"]:
        group_name = group["group_name"]
        stage2_ids = group["solution_ids"]  # 1-indexed among representatives
        all_original_ids = []
        for s2_id in stage2_ids:
            if s2_id < 1 or s2_id > len(representatives):
                logging.warning(f"Stage 2 solution_id {s2_id} out of range (1-{len(representatives)}), skipping")
                continue
            rep = representatives[s2_id - 1]  # Convert 1-indexed to 0-indexed
            all_original_ids.extend(rep["source_group_members_original"])
        final_grouping[group_name] = sorted(all_original_ids)
        assigned.update(all_original_ids)

    # Handle unassigned solutions (from failed chunks)
    all_ids = set(range(1, num_solutions + 1))
    unassigned = sorted(all_ids - assigned)
    if unassigned:
        final_grouping["_unassigned"] = unassigned

    return final_grouping


def fallback_stage1_only(
    stage1_results: List[Optional[Dict]],
    chunk_offsets: List[int],
    num_solutions: int
) -> Dict[str, List[int]]:
    """
    Fallback when Stage 2 fails: each Stage 1 group becomes a final group
    with name prefixed by chunk index.
    """
    final_grouping = {}
    assigned = set()

    for chunk_idx, parsed in enumerate(stage1_results):
        if parsed is None:
            continue
        offset = chunk_offsets[chunk_idx]
        next_offset = chunk_offsets[chunk_idx + 1] if chunk_idx + 1 < len(chunk_offsets) else num_solutions
        chunk_len = next_offset - offset
        for group in parsed["groups"]:
            local_ids = group["solution_ids"]
            valid_local_ids = [lid for lid in local_ids if 1 <= lid <= chunk_len]
            if not valid_local_ids:
                continue
            original_ids = [offset + lid for lid in valid_local_ids]
            final_name = f"[chunk{chunk_idx}] {group['group_name']}"
            final_grouping[final_name] = sorted(original_ids)
            assigned.update(original_ids)

    # Handle unassigned solutions (from failed chunks)
    all_ids = set(range(1, num_solutions + 1))
    unassigned = sorted(all_ids - assigned)
    if unassigned:
        final_grouping["_unassigned"] = unassigned

    return final_grouping


def _balanced_json_candidates(text: str) -> List[str]:
    """Extract balanced top-level JSON object substrings from text."""
    candidates = []
    start = None
    depth = 0
    in_string = False
    escape = False

    for idx, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(text[start:idx + 1])
                    start = None
    return candidates


def _repair_invalid_backslashes(text: str) -> str:
    """Escape stray backslashes that are invalid in JSON strings."""
    return re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r"\\\\", text)


def _extract_json_candidates(response_text: str) -> List[str]:
    """Return likely JSON payload candidates in priority order."""
    text = response_text.strip()
    text = (text
            .replace("\u201c", '"')
            .replace("\u201d", '"')
            .replace("\u2018", "'")
            .replace("\u2019", "'"))

    candidates = []

    for pattern in (r"```json\s*(.*?)\s*```", r"```\s*(.*?)\s*```"):
        for match in re.finditer(pattern, text, re.DOTALL | re.IGNORECASE):
            candidates.append(match.group(1).strip())

    candidates.extend(_balanced_json_candidates(text))
    candidates.append(text)

    deduped = []
    seen = set()
    for candidate in candidates:
        normalized = candidate.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _parse_json_candidate(candidate: str) -> Tuple[Optional[Dict], Optional[Exception]]:
    """Try several increasingly permissive parsing strategies for one candidate."""
    parse_error: Optional[Exception] = None

    for variant in (candidate, _repair_invalid_backslashes(candidate)):
        try:
            data = json.loads(variant)
            if isinstance(data, dict):
                return data, None
        except json.JSONDecodeError as e:
            parse_error = e

    try:
        data = ast.literal_eval(candidate)
        if isinstance(data, dict):
            return data, None
    except (SyntaxError, ValueError) as e:
        parse_error = e

    return None, parse_error


def _normalize_solution_ids(raw_ids: Any) -> List[int]:
    """Coerce a solution-id payload into a best-effort list of integers."""
    if isinstance(raw_ids, list):
        items = raw_ids
    else:
        items = [raw_ids]

    normalized = []
    for item in items:
        if isinstance(item, bool):
            continue
        if isinstance(item, int):
            normalized.append(item)
            continue
        if isinstance(item, str):
            for match in re.finditer(r"\d+", item):
                normalized.append(int(match.group(0)))
    return normalized


def parse_response(response_text: str) -> Optional[Dict]:
    """Parse and validate JSON response from model."""
    if not response_text:
        return None

    data = None
    last_error: Optional[Exception] = None

    for candidate in _extract_json_candidates(response_text):
        data, last_error = _parse_json_candidate(candidate)
        if data is not None:
            break

    if data is None:
        if last_error is not None:
            logging.error(f"Failed to parse JSON: {last_error}")
        else:
            logging.error("Failed to parse JSON: no valid JSON candidate found")
        return None

    # Validate structure
    if "groups" not in data:
        logging.error("Missing required field: groups")
        return None
    if "reasoning_trace" not in data:
        data["reasoning_trace"] = ""
    if not isinstance(data["groups"], list):
        logging.error("groups must be a list")
        return None

    # Validate groups
    for group in data["groups"]:
        # Guard against non-dict entries (e.g. ast.literal_eval returning
        # Ellipsis for a `...` placeholder in the judge response — see issue
        # surfaced in the judge-cluster-n64 Qwen retry run).
        if not isinstance(group, dict):
            logging.error(f"Invalid group entry (not a dict): {group!r}")
            return None
        if "group_name" not in group or "solution_ids" not in group:
            logging.error(f"Invalid group structure: {group}")
            return None
        if "core_idea" not in group:
            group["core_idea"] = ""

        group["solution_ids"] = _normalize_solution_ids(group["solution_ids"])

    # Filter out groups with empty solution_ids, and sanitize non-integer IDs
    cleaned_groups = []
    for g in data["groups"]:
        valid_ids = [sid for sid in g["solution_ids"] if isinstance(sid, int)]
        if not valid_ids:
            continue
        g["solution_ids"] = valid_ids
        cleaned_groups.append(g)
    data["groups"] = cleaned_groups

    return data


def repair_grouping_dict(
    grouping_dict: Dict[str, List[int]],
    num_solutions: int
) -> Tuple[Dict[str, List[int]], Dict[str, Any]]:
    """
    Make a best-effort valid partition from a predicted grouping.

    Drops invalid/out-of-range IDs, removes duplicates, and assigns any missing
    solution IDs to singleton recovery groups.
    """
    repaired: Dict[str, List[int]] = {}
    assigned = set()
    dropped_duplicates = 0
    dropped_invalid = 0
    renamed_groups = 0

    for raw_name, raw_ids in grouping_dict.items():
        base_name = str(raw_name).strip() or "group"
        group_name = base_name
        suffix = 2
        while group_name in repaired:
            renamed_groups += 1
            group_name = f"{base_name}_{suffix}"
            suffix += 1

        ids = _normalize_solution_ids(raw_ids)
        cleaned_ids = []
        local_seen = set()

        for sol_id in ids:
            if sol_id < 1 or sol_id > num_solutions:
                dropped_invalid += 1
                continue
            if sol_id in local_seen or sol_id in assigned:
                dropped_duplicates += 1
                continue
            local_seen.add(sol_id)
            assigned.add(sol_id)
            cleaned_ids.append(sol_id)

        if cleaned_ids:
            repaired[group_name] = cleaned_ids

    recovered_missing = []
    for sol_id in range(1, num_solutions + 1):
        if sol_id not in assigned:
            repaired[f"_recovered_{sol_id}"] = [sol_id]
            recovered_missing.append(sol_id)

    changed = (
        repaired != grouping_dict
        or bool(recovered_missing)
        or dropped_duplicates > 0
        or dropped_invalid > 0
        or renamed_groups > 0
    )
    metadata = {
        "changed": changed,
        "dropped_duplicates": dropped_duplicates,
        "dropped_invalid": dropped_invalid,
        "recovered_missing": recovered_missing,
        "renamed_groups": renamed_groups,
    }
    return repaired, metadata


def parse_or_repair_response(
    client: GPTClient,
    problem_text: str,
    solutions: List[str],
    response_text: str,
    logger: logging.Logger,
    issue: str
) -> Optional[Dict]:
    """Parse a model response and, on failure, make one repair call."""
    parsed = parse_response(response_text)
    if parsed is not None:
        return parsed

    logger.warning(f"Parse failed; attempting repair call ({issue})")
    repair_messages = build_repair_prompt(problem_text, solutions, response_text[:4000], issue)
    repaired_text = client.call_realtime(repair_messages)
    if repaired_text is None:
        return None
    return parse_response(repaired_text)


def grouping_to_labels(grouping_dict: Dict[str, List[int]], num_solutions: int) -> Optional[np.ndarray]:
    """
    Convert grouping dictionary to label array.

    Args:
        grouping_dict: Dictionary with group names as keys and solution IDs as values (1-indexed)
        num_solutions: Total number of solutions

    Returns:
        0-indexed label array, or None if validation fails
    """
    labels = np.full(num_solutions, -1, dtype=int)

    # Assign integer labels to groups
    for group_idx, (group_name, solution_ids) in enumerate(grouping_dict.items()):
        for sol_id in solution_ids:
            # Convert 1-indexed to 0-indexed
            array_idx = sol_id - 1

            # Validate solution ID
            if array_idx < 0 or array_idx >= num_solutions:
                logging.error(f"Invalid solution ID: {sol_id} (must be 1-{num_solutions})")
                return None

            # Check for duplicates
            if labels[array_idx] != -1:
                logging.error(f"Duplicate solution ID: {sol_id}")
                return None

            labels[array_idx] = group_idx

    # Check all solutions assigned
    if -1 in labels:
        unassigned = [i + 1 for i, label in enumerate(labels) if label == -1]
        logging.error(f"Unassigned solutions: {unassigned}")
        return None

    return labels


def compute_metrics(true_labels: np.ndarray, pred_labels: np.ndarray) -> Dict[str, float]:
    """Calculate clustering metrics."""
    metrics = {
        "adjusted_rand_index": float(adjusted_rand_score(true_labels, pred_labels)),
        "homogeneity": float(homogeneity_score(true_labels, pred_labels)),
        "completeness": float(completeness_score(true_labels, pred_labels)),
        "num_true_clusters": int(len(np.unique(true_labels))),
        "num_pred_clusters": int(len(np.unique(pred_labels)))
    }
    return metrics


def load_jsonl_data(file_path: str, limit: Optional[int] = None) -> List[Dict]:
    """Load JSONL file and optionally limit number of problems."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if limit and len(data) >= limit:
                break
            data.append(json.loads(line.strip()))
            
    # if problem_idx not in data, add it now
    if "problem_idx" not in data[0]:
        for idx, problem_data in enumerate(data):
            problem_data["problem_idx"] = idx
    
    return data


def two_stage_cluster(
    client: GPTClient,
    problem_text: str,
    solutions: List[str],
    chunk_size: int,
    logger: logging.Logger,
    system_prompt: str,
) -> Optional[Dict]:
    """
    Two-stage clustering for realtime mode.

    Stage 1: Divide solutions into chunks and cluster each independently.
    Stage 2: Collect representatives from Stage 1, re-cluster them, then map back.

    Returns dict with final_grouping, stage1_results, stage2_result, representatives, reasoning_trace,
    or None if all Stage 1 chunks fail.
    """
    num_solutions = len(solutions)
    chunks = divide_into_chunks(solutions, chunk_size)
    # chunk_offsets[i] is the offset to convert chunk-local 1-indexed IDs to original 1-indexed IDs:
    # original_1indexed = chunk_offsets[i] + local_1indexed_id
    # So chunk_offsets[i] = i * chunk_size (since local ID 1 in chunk i -> original ID i*chunk_size + 1)
    chunk_offsets = [i * chunk_size for i in range(len(chunks))]

    logger.info(f"Two-stage clustering: {num_solutions} solutions -> {len(chunks)} chunks of up to {chunk_size}")

    # --- Stage 1: cluster each chunk ---
    stage1_results = []
    stage1_reasoning = []
    for chunk_idx, chunk in enumerate(chunks):
        logger.info(f"  Stage 1: processing chunk {chunk_idx + 1}/{len(chunks)} ({len(chunk)} solutions)")
        messages = build_prompt(problem_text, chunk, system_prompt=system_prompt)
        response_text = client.call_realtime(messages)

        if response_text is None:
            logger.warning(f"  Stage 1: chunk {chunk_idx} failed (no response)")
            stage1_results.append(None)
            stage1_reasoning.append(None)
            continue

        parsed = parse_or_repair_response(
            client, problem_text, chunk, response_text, logger,
            issue=f"stage1 chunk {chunk_idx}"
        )
        if parsed is None:
            logger.warning(f"  Stage 1: chunk {chunk_idx} failed (parse error)")
            stage1_results.append(None)
            stage1_reasoning.append(response_text[:500])
            continue

        stage1_results.append(parsed)
        stage1_reasoning.append(parsed.get("reasoning_trace", ""))

    # Check if any chunk succeeded
    successful_chunks = [r for r in stage1_results if r is not None]
    if not successful_chunks:
        logger.error("Two-stage clustering: all Stage 1 chunks failed")
        return None

    # --- Aggregate representatives ---
    representatives = aggregate_stage1_results(stage1_results, chunk_offsets, solutions)
    logger.info(f"  Aggregated {len(representatives)} representatives from {len(successful_chunks)} successful chunks")

    # If only 1 representative, skip Stage 2
    if len(representatives) <= 1:
        logger.info("  Only 1 representative, skipping Stage 2")
        final_grouping = fallback_stage1_only(stage1_results, chunk_offsets, num_solutions)
        return {
            "final_grouping": final_grouping,
            "stage1_results": [_summarize_parsed(p) for p in stage1_results],
            "stage2_result": None,
            "representatives": [_summarize_rep(r) for r in representatives],
            "reasoning_trace": stage1_reasoning
        }

    # --- Stage 2: re-cluster representatives ---
    rep_solutions = [r["solution_text"] for r in representatives]
    logger.info(f"  Stage 2: clustering {len(rep_solutions)} representatives")
    messages = build_prompt(problem_text, rep_solutions, system_prompt=system_prompt)
    response_text = client.call_realtime(messages)

    if response_text is None:
        logger.warning("  Stage 2 failed (no response), falling back to Stage 1 only")
        final_grouping = fallback_stage1_only(stage1_results, chunk_offsets, num_solutions)
        return {
            "final_grouping": final_grouping,
            "stage1_results": [_summarize_parsed(p) for p in stage1_results],
            "stage2_result": {"error": "no response"},
            "representatives": [_summarize_rep(r) for r in representatives],
            "reasoning_trace": stage1_reasoning
        }

    stage2_parsed = parse_or_repair_response(
        client, problem_text, rep_solutions, response_text, logger,
        issue="stage2 representatives"
    )
    if stage2_parsed is None:
        logger.warning("  Stage 2 failed (parse error), falling back to Stage 1 only")
        final_grouping = fallback_stage1_only(stage1_results, chunk_offsets, num_solutions)
        return {
            "final_grouping": final_grouping,
            "stage1_results": [_summarize_parsed(p) for p in stage1_results],
            "stage2_result": {"error": "parse failure", "raw": response_text[:500]},
            "representatives": [_summarize_rep(r) for r in representatives],
            "reasoning_trace": stage1_reasoning
        }

    # --- Map back to original solution IDs ---
    final_grouping = map_stage2_to_final(
        stage2_parsed, representatives, num_solutions,
        stage1_results, chunk_offsets, chunk_size
    )

    combined_reasoning = stage1_reasoning + [stage2_parsed.get("reasoning_trace", "")]

    return {
        "final_grouping": final_grouping,
        "stage1_results": [_summarize_parsed(p) for p in stage1_results],
        "stage2_result": _summarize_parsed(stage2_parsed),
        "representatives": [_summarize_rep(r) for r in representatives],
        "reasoning_trace": combined_reasoning
    }


def _summarize_parsed(parsed: Optional[Dict]) -> Optional[Dict]:
    """Summarize a parsed response for storage in two_stage_detail."""
    if parsed is None:
        return None
    return {
        "groups": [
            {"group_name": g["group_name"], "solution_ids": g["solution_ids"]}
            for g in parsed.get("groups", [])
        ],
        "reasoning_trace": parsed.get("reasoning_trace", "")
    }


def _summarize_rep(rep: Dict) -> Dict:
    """Summarize a representative for storage."""
    return {
        "original_id": rep["original_id"],
        "source_chunk_idx": rep["source_chunk_idx"],
        "source_group_name": rep["source_group_name"],
        "source_group_members_original": rep["source_group_members_original"]
    }


class RealtimeProcessor:
    """Process problems using realtime API calls with optional parallelism."""

    def __init__(self, config: ClusteringConfig, client: GPTClient, cost_tracker: CostTracker):
        self.config = config
        self.client = client
        self.cost_tracker = cost_tracker
        self.logger = logging.getLogger(__name__)
        self._results_lock = threading.Lock()
        self.system_prompt = ACTIVE_SYSTEM_PROMPT

    def _process_one_problem(self, problem_data: Dict) -> Optional[Dict]:
        """Process a single problem. Returns a result dict or None if skipped."""
        start_time = time.perf_counter()
        problem_idx = problem_data["problem_idx"]

        # Determine keys
        problem_key = "problem" if "problem" in problem_data else "question"
        solutions_key = "solutions_to_eval" if "solutions_to_eval" in problem_data else "solutions"
        problem_text = problem_data[problem_key]
        all_solutions = problem_data[solutions_key]

        # Filter to correct solutions only if scores are available
        scores = problem_data.get("scores")
        if scores is not None:
            solutions = [sol for sol, score in zip(all_solutions, scores) if score == 1.0]
        else:
            solutions = all_solutions
        num_solutions = len(solutions)

        if num_solutions == 0:
            self.logger.info(f"Skipping problem {problem_idx}: no correct solutions")
            return None

        if num_solutions <= self.config.min_correct:
            self.logger.info(f"Skipping problem {problem_idx}: only {num_solutions} correct solution(s) (min_correct={self.config.min_correct})")
            return None

        use_two_stage = (self.config.chunk_size is not None
                         and num_solutions > self.config.chunk_size)

        if use_two_stage:
        # --- Two-stage clustering ---
            ts_result = two_stage_cluster(
                self.client, problem_text, solutions,
                self.config.chunk_size, self.logger, self.system_prompt
            )

            if ts_result is None:
                self.logger.error(f"Two-stage clustering failed for problem {problem_idx}")
                return {
                    "problem_idx": problem_idx,
                    "error": "Two-stage clustering failed (all chunks failed)",
                    "clustering_mode": "two_stage",
                    "latency_seconds": time.perf_counter() - start_time,
                    "timestamp": time.time()
                }

            pred_grouping = ts_result["final_grouping"]
            pred_labels = grouping_to_labels(pred_grouping, num_solutions)
            if pred_labels is None:
                repaired_grouping, repair_meta = repair_grouping_dict(pred_grouping, num_solutions)
                if repair_meta["changed"]:
                    self.logger.warning(
                        f"Repaired two-stage grouping for problem {problem_idx}: "
                        f"missing={len(repair_meta['recovered_missing'])}, "
                        f"dup={repair_meta['dropped_duplicates']}, "
                        f"invalid={repair_meta['dropped_invalid']}"
                    )
                    pred_grouping = repaired_grouping
                    pred_labels = grouping_to_labels(pred_grouping, num_solutions)

            if pred_labels is None:
                self.logger.error(f"Failed to convert two-stage grouping to labels for problem {problem_idx}")
                return {
                    "problem_idx": problem_idx,
                    "error": "Invalid two-stage grouping structure",
                    "predicted_grouping": pred_grouping,
                    "clustering_mode": "two_stage",
                    "latency_seconds": time.perf_counter() - start_time,
                    "timestamp": time.time()
                }

            result = {
                "problem_idx": problem_idx,
                "problem": problem_data[problem_key],
                "num_solutions": num_solutions,
                "predicted_grouping": pred_grouping,
                "reasoning_trace": ts_result["reasoning_trace"],
                "clustering_mode": "two_stage",
                "latency_seconds": time.perf_counter() - start_time,
                "two_stage_detail": {
                    "stage1_results": ts_result["stage1_results"],
                    "stage2_result": ts_result["stage2_result"],
                    "representatives": ts_result["representatives"]
                },
                "timestamp": time.time()
            }

        else:
            # --- Single-stage clustering ---
            messages = build_prompt(problem_text, solutions, system_prompt=self.system_prompt)
            response_text = self.client.call_realtime(messages)

            if response_text is None:
                self.logger.error(f"Failed to get response for problem {problem_idx}")
                return {
                    "problem_idx": problem_idx,
                    "error": "Failed to get response from API",
                    "clustering_mode": "single_stage",
                    "latency_seconds": time.perf_counter() - start_time,
                    "timestamp": time.time()
                }

            parsed_response = parse_or_repair_response(
                self.client, problem_text, solutions, response_text, self.logger,
                issue=f"single-stage problem {problem_idx}"
            )

            if parsed_response is None:
                self.logger.error(f"Failed to parse response for problem {problem_idx}")
                return {
                    "problem_idx": problem_idx,
                    "error": "Failed to parse response",
                    "raw_response": response_text[:500],
                    "clustering_mode": "single_stage",
                    "latency_seconds": time.perf_counter() - start_time,
                    "timestamp": time.time()
                }

            pred_grouping = {
                group["group_name"]: group["solution_ids"]
                for group in parsed_response["groups"]
            }

            pred_labels = grouping_to_labels(pred_grouping, num_solutions)
            if pred_labels is None:
                repaired_grouping, repair_meta = repair_grouping_dict(pred_grouping, num_solutions)
                if repair_meta["changed"]:
                    self.logger.warning(
                        f"Repaired grouping for problem {problem_idx}: "
                        f"missing={len(repair_meta['recovered_missing'])}, "
                        f"dup={repair_meta['dropped_duplicates']}, "
                        f"invalid={repair_meta['dropped_invalid']}"
                    )
                    pred_grouping = repaired_grouping
                    pred_labels = grouping_to_labels(pred_grouping, num_solutions)

            if pred_labels is None:
                self.logger.error(f"Failed to convert predicted grouping to labels for problem {problem_idx}")
                return {
                    "problem_idx": problem_idx,
                    "error": "Invalid grouping structure",
                    "predicted_grouping": pred_grouping,
                    "clustering_mode": "single_stage",
                    "latency_seconds": time.perf_counter() - start_time,
                    "timestamp": time.time()
                }

            result = {
                "problem_idx": problem_idx,
                "problem": problem_data[problem_key],
                "num_solutions": num_solutions,
                "predicted_grouping": pred_grouping,
                "reasoning_trace": parsed_response["reasoning_trace"],
                "clustering_mode": "single_stage",
                "latency_seconds": time.perf_counter() - start_time,
                "timestamp": time.time()
            }

        # Evaluate against golden labels if eval mode is on
        if self.config.eval:
            golden_grouping = problem_data["group_data"]
            true_labels = grouping_to_labels(golden_grouping, num_solutions)

            if true_labels is None:
                self.logger.error(f"Failed to convert golden grouping to labels for problem {problem_idx}")
                return {
                    "problem_idx": problem_idx,
                    "error": "Invalid golden grouping",
                    "golden_grouping": golden_grouping,
                    "latency_seconds": time.perf_counter() - start_time,
                    "timestamp": time.time()
                }

            metrics = compute_metrics(true_labels, pred_labels)
            result["golden_grouping"] = golden_grouping
            result["metrics"] = metrics

        return result

    def process(self, input_file: str, output_file: str) -> Dict:
        """Process all problems with optional parallelism."""
        run_start = time.perf_counter()
        # Load data
        problems = load_jsonl_data(input_file, self.config.limit)
        self.logger.info(f"Loaded {len(problems)} problems from {input_file}")

        # Check for resume
        processed_indices = set()
        results = []

        if self.config.resume_file and os.path.exists(self.config.resume_file):
            with open(self.config.resume_file, 'r') as f:
                resume_data = json.load(f)
                results = resume_data.get("results", [])
                processed_indices = {r["problem_idx"] for r in results}
            self.logger.info(f"Resuming from {self.config.resume_file}. Already processed: {len(processed_indices)} problems")

        # Filter problems to process
        problems_to_process = [
            p for p in problems if p["problem_idx"] not in processed_indices
        ]

        successful = sum(1 for r in results if "error" not in r)
        failed = sum(1 for r in results if "error" in r)

        num_workers = self.config.num_workers

        if num_workers <= 1:
            # Sequential processing (original behavior with rate limiting)
            self._process_sequential(problems_to_process, results, successful, failed, output_file)
        else:
            # Parallel processing
            self.logger.info(f"Parallel processing with {num_workers} workers")
            self._process_parallel(problems_to_process, results, successful, failed, output_file, num_workers)

        successful = sum(1 for r in results if "error" not in r)
        failed = sum(1 for r in results if "error" in r)

        # Final metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "model": self.config.model,
            "api_base": self.config.api_base,
            "mode": "realtime",
            "thinking_token_budget": self.config.thinking_token_budget,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "min_p": self.config.min_p,
            "presence_penalty": self.config.presence_penalty,
            "repetition_penalty": self.config.repetition_penalty,
            "seed": self.config.seed,
            "input_file": input_file,
            "num_problems": len(problems),
            "num_successful": successful,
            "num_failed": failed,
            "chunk_size": self.config.chunk_size,
            "num_workers": num_workers,
            "run_wall_time_seconds": time.perf_counter() - run_start
        }

        return {"metadata": metadata, "results": results}

    def _process_sequential(self, problems: List[Dict], results: List[Dict],
                            successful: int, failed: int, output_file: str):
        """Process problems sequentially with rate limiting."""
        min_interval = 60.0 / self.config.rate_limit_rpm
        last_call_time = 0

        progress_bar = tqdm(problems, disable=self.config.quiet, desc="Processing problems")

        for problem_data in progress_bar:
            # Rate limiting
            elapsed = time.time() - last_call_time
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            last_call_time = time.time()

            result = self._process_one_problem(problem_data)
            if result is None:
                continue

            results.append(result)
            if "error" in result:
                failed += 1
            else:
                successful += 1

            self._save_intermediate_results(output_file, results, successful, failed)
            progress_bar.set_postfix({"success": successful, "failed": failed})

    def _process_parallel(self, problems: List[Dict], results: List[Dict],
                          successful: int, failed: int, output_file: str, num_workers: int):
        """Process problems in parallel using ThreadPoolExecutor."""
        progress_bar = tqdm(total=len(problems), disable=self.config.quiet, desc=f"Processing ({num_workers} workers)")

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_idx = {
                executor.submit(self._process_one_problem, p): p["problem_idx"]
                for p in problems
            }

            for future in concurrent.futures.as_completed(future_to_idx):
                problem_idx = future_to_idx[future]
                try:
                    result = future.result()
                except Exception as e:
                    self.logger.error(f"Unexpected error for problem {problem_idx}: {e}")
                    result = {
                        "problem_idx": problem_idx,
                        "error": f"Unexpected error: {e}",
                        "timestamp": time.time()
                    }

                if result is not None:
                    with self._results_lock:
                        results.append(result)
                        if "error" in result:
                            failed += 1
                        else:
                            successful += 1
                        self._save_intermediate_results(output_file, results, successful, failed)

                progress_bar.update(1)
                progress_bar.set_postfix({"success": successful, "failed": failed})

        progress_bar.close()

    def _save_intermediate_results(self, output_file: str, results: List[Dict], successful: int, failed: int):
        """Save intermediate results for resume capability (caller must hold _results_lock if parallel)."""
        intermediate_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "model": self.config.model,
                "mode": "realtime",
                    "num_successful": successful,
                "num_failed": failed,
                "status": "in_progress"
            },
            "results": results
        }

        with open(output_file, 'w') as f:
            json.dump(intermediate_data, f, indent=2)


class BatchProcessor:
    """Process problems using OpenAI Batch API with optional two-stage clustering."""

    def __init__(self, config: ClusteringConfig, client: GPTClient, cost_tracker: CostTracker):
        self.config = config
        self.client = client
        self.cost_tracker = cost_tracker
        self.logger = logging.getLogger(__name__)
        self.system_prompt = ACTIVE_SYSTEM_PROMPT

    def process(self, input_file: str, output_dir: str) -> Dict:
        """Process all problems using batch API, with two-stage support."""

        problems = load_jsonl_data(input_file, self.config.limit)
        self.logger.info(f"Loaded {len(problems)} problems from {input_file}")

        input_basename = Path(input_file).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Categorize problems and prepare Stage 1 batch
        problems_info = self._categorize_problems(problems)

        batch1_id = None
        if self.config.resume_stage1 and os.path.exists(self.config.resume_stage1):
            # Resume from saved Stage 1 raw results
            self.logger.info(f"Resuming from Stage 1 results: {self.config.resume_stage1}")
            raw_results_1 = self._load_raw_results(self.config.resume_stage1)
        else:
            stage1_requests = self._prepare_stage1_batch(problems_info)

            if not stage1_requests:
                return {"metadata": {"error": "No requests to process"}, "results": []}

            # Submit Stage 1 batch
            batch1_file = os.path.join(output_dir, f"batch_input_stage1_{input_basename}_{timestamp}.jsonl")
            self._write_batch_file(stage1_requests, batch1_file)
            batch1_id = self.client.submit_batch(batch1_file)
            self.logger.info(f"Submitted Stage 1 batch: {batch1_id}")

            if not self.client.poll_batch(batch1_id):
                self.logger.error("Stage 1 batch processing failed")
                return {"metadata": {"error": "Stage 1 batch processing failed"}, "results": []}

            raw1_path = os.path.join(output_dir, f"batch_raw_stage1_{input_basename}_{timestamp}.jsonl")
            results_file = self.client.retrieve_batch_results(batch1_id, raw1_path)
            if results_file is None:
                return {"metadata": {"error": "Failed to retrieve Stage 1 results"}, "results": []}

            raw_results_1 = self._load_raw_results(results_file)

        # Process Stage 1 results
        single_stage_results, two_stage_intermediates = self._process_stage1_batch_results(
            raw_results_1, problems_info
        )

        # Stage 2: prepare and submit if there are two-stage problems needing it
        # Populate representatives on each intermediate (needed for result processing)
        stage2_requests = self._prepare_stage2_batch(two_stage_intermediates, problems_info)
        batch2_id = None

        resume_file_id = self.config.resume_stage2_input_file_id

        if resume_file_id:
            # Attach to an already-submitted Stage 2 batch instead of creating a new one
            batch2_id = self.client.find_batch_by_input_file_id(resume_file_id)
            if batch2_id is None:
                self.logger.warning("Could not locate Stage 2 batch; falling back to Stage 1 only")
                two_stage_results = self._fallback_all_stage1(two_stage_intermediates, problems_info)
            elif not self.client.poll_batch(batch2_id):
                self.logger.warning("Resumed Stage 2 batch failed; falling back to Stage 1 only")
                two_stage_results = self._fallback_all_stage1(two_stage_intermediates, problems_info)
            else:
                raw2_path = os.path.join(output_dir, f"batch_raw_stage2_{input_basename}_{timestamp}.jsonl")
                results_file_2 = self.client.retrieve_batch_results(batch2_id, raw2_path)
                if results_file_2 is None:
                    self.logger.warning("Failed to retrieve resumed Stage 2 results; falling back")
                    two_stage_results = self._fallback_all_stage1(two_stage_intermediates, problems_info)
                else:
                    raw_results_2 = self._load_raw_results(results_file_2)
                    two_stage_results = self._process_stage2_batch_results(
                        raw_results_2, two_stage_intermediates, problems_info
                    )
        elif stage2_requests:
            batch2_file = os.path.join(output_dir, f"batch_input_stage2_{input_basename}_{timestamp}.jsonl")
            self._write_batch_file(stage2_requests, batch2_file)
            batch2_id = self.client.submit_batch(batch2_file)
            self.logger.info(f"Submitted Stage 2 batch: {batch2_id}")

            if not self.client.poll_batch(batch2_id):
                self.logger.warning("Stage 2 batch failed, falling back to Stage 1 only for two-stage problems")
                two_stage_results = self._fallback_all_stage1(two_stage_intermediates, problems_info)
            else:
                raw2_path = os.path.join(output_dir, f"batch_raw_stage2_{input_basename}_{timestamp}.jsonl")
                results_file_2 = self.client.retrieve_batch_results(batch2_id, raw2_path)
                if results_file_2 is None:
                    self.logger.warning("Failed to retrieve Stage 2 results, falling back")
                    two_stage_results = self._fallback_all_stage1(two_stage_intermediates, problems_info)
                else:
                    raw_results_2 = self._load_raw_results(results_file_2)
                    two_stage_results = self._process_stage2_batch_results(
                        raw_results_2, two_stage_intermediates, problems_info
                    )
        else:
            # No two-stage problems or all intermediates failed
            two_stage_results = self._fallback_all_stage1(two_stage_intermediates, problems_info)

        # Merge results
        all_results = single_stage_results + two_stage_results
        all_results.sort(key=lambda r: r["problem_idx"])

        # Evaluate against golden labels
        if self.config.eval:
            for result in all_results:
                if "error" in result:
                    continue
                pinfo = problems_info[result["problem_idx"]]
                original_data = pinfo["original_data"]
                if "group_data" not in original_data:
                    continue
                num_solutions = pinfo["num_solutions"]
                golden_grouping = original_data["group_data"]
                true_labels = grouping_to_labels(golden_grouping, num_solutions)
                pred_labels = grouping_to_labels(result["predicted_grouping"], num_solutions)
                if true_labels is not None and pred_labels is not None:
                    metrics = compute_metrics(true_labels, pred_labels)
                    result["golden_grouping"] = golden_grouping
                    result["metrics"] = metrics

        successful = sum(1 for r in all_results if "error" not in r)
        failed = len(all_results) - successful

        metadata = {
            "timestamp": datetime.now().isoformat(),
            "model": self.config.model,
            "api_base": self.config.api_base,
            "mode": "batch",
            "thinking_token_budget": self.config.thinking_token_budget,
            "input_file": input_file,
            "batch_id_stage1": batch1_id,
            "batch_id_stage2": batch2_id,
            "num_problems": len(problems),
            "num_successful": successful,
            "num_failed": failed,
            "chunk_size": self.config.chunk_size
        }

        return {"metadata": metadata, "results": all_results}

    def _categorize_problems(self, problems: List[Dict]) -> Dict[int, Dict]:
        """Categorize each problem as single-stage, two-stage, or skipped (no correct solutions)."""
        problems_info = {}
        skipped_no_correct = 0
        skipped_min_correct = 0
        for problem_data in problems:
            problem_idx = problem_data["problem_idx"]
            problem_key = "problem" if "problem" in problem_data else "question"
            solutions_key = "solutions_to_eval" if "solutions_to_eval" in problem_data else "solutions"
            all_solutions = problem_data[solutions_key]

            # Filter to correct solutions only if scores are available
            scores = problem_data.get("scores")
            if scores is not None:
                solutions = [sol for sol, score in zip(all_solutions, scores) if score == 1.0]
            else:
                solutions = all_solutions

            num_solutions = len(solutions)

            if num_solutions == 0:
                skipped_no_correct += 1
                continue

            if num_solutions <= self.config.min_correct:
                skipped_min_correct += 1
                continue

            use_two_stage = (self.config.chunk_size is not None
                             and num_solutions > self.config.chunk_size)

            problems_info[problem_idx] = {
                "problem_idx": problem_idx,
                "problem_key": problem_key,
                "solutions_key": solutions_key,
                "original_data": problem_data,
                "solutions": solutions,
                "num_solutions": num_solutions,
                "use_two_stage": use_two_stage
            }

        if skipped_no_correct > 0:
            self.logger.info(f"Skipped {skipped_no_correct} problems with no correct solutions")
        if skipped_min_correct > 0:
            self.logger.info(f"Skipped {skipped_min_correct} problems with <= {self.config.min_correct} correct solutions")

        return problems_info

    def _prepare_stage1_batch(self, problems_info: Dict[int, Dict]) -> List[Dict]:
        """Create batch requests for Stage 1 (single-stage full requests + two-stage chunk requests)."""
        requests = []
        for problem_idx, pinfo in sorted(problems_info.items()):
            original_data = pinfo["original_data"]
            problem_text = original_data[pinfo["problem_key"]]
            solutions = pinfo["solutions"]

            if pinfo["use_two_stage"]:
                chunks = divide_into_chunks(solutions, self.config.chunk_size)
                for chunk_idx, chunk in enumerate(chunks):
                    messages = build_prompt(problem_text, chunk, system_prompt=self.system_prompt)
                    requests.append({
                        "custom_id": f"problem_{problem_idx}_chunk_{chunk_idx}",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": self._make_batch_body(messages)
                    })
            else:
                messages = build_prompt(problem_text, solutions, system_prompt=self.system_prompt)
                requests.append({
                    "custom_id": f"problem_{problem_idx}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": self._make_batch_body(messages)
                })
        return requests

    def _make_batch_body(self, messages: List[Dict]) -> Dict:
        """Build a batch request body, omitting reasoning_effort for non-reasoning models."""
        body = {"model": self.config.model, "messages": messages}
        if self.config.temperature is not None:
            body["temperature"] = self.config.temperature
        if self.config.top_p is not None:
            body["top_p"] = self.config.top_p
        if self.config.presence_penalty is not None:
            body["presence_penalty"] = self.config.presence_penalty
        if self.config.seed is not None:
            body["seed"] = self.config.seed
        if self.config.top_k is not None:
            body["top_k"] = self.config.top_k
        if self.config.min_p is not None:
            body["min_p"] = self.config.min_p
        if self.config.repetition_penalty is not None:
            body["repetition_penalty"] = self.config.repetition_penalty
        if self.config.reasoning_effort and self.config.reasoning_effort != "none":
            body["reasoning_effort"] = self.config.reasoning_effort
        return body

    def _write_batch_file(self, requests: List[Dict], path: str):
        """Write batch requests to JSONL file."""
        with open(path, 'w') as f:
            for req in requests:
                f.write(json.dumps(req) + '\n')
        self.logger.info(f"Created batch file: {path} ({len(requests)} requests)")

    def _load_raw_results(self, results_file: str) -> Dict[str, Dict]:
        """Load raw batch results into a dict keyed by custom_id."""
        raw_results = {}
        with open(results_file, 'r') as f:
            for line in f:
                result = json.loads(line.strip())
                raw_results[result["custom_id"]] = result
        return raw_results

    def _extract_response_text(self, raw_result: Dict) -> Optional[str]:
        """Extract response text from a batch result, tracking usage."""
        if raw_result["response"]["status_code"] != 200:
            return None

        response_text = sanitize_response_content(
            raw_result["response"]["body"]["choices"][0]["message"]["content"]
        )

        if "usage" in raw_result["response"]["body"]:
            usage = raw_result["response"]["body"]["usage"]
            usage_dict = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "reasoning_tokens": usage.get("reasoning_tokens", 0)
            }
            self.cost_tracker.add_usage(usage_dict)

        return response_text

    def _process_stage1_batch_results(
        self,
        raw_results: Dict[str, Dict],
        problems_info: Dict[int, Dict]
    ) -> Tuple[List[Dict], Dict[int, Dict]]:
        """
        Process Stage 1 batch results.

        Returns:
            single_stage_results: Final results for single-stage problems
            two_stage_intermediates: Dict mapping problem_idx -> intermediate data for two-stage problems
        """
        single_stage_results = []
        two_stage_intermediates = {}

        for problem_idx, pinfo in sorted(problems_info.items()):
            original_data = pinfo["original_data"]
            problem_key = pinfo["problem_key"]
            num_solutions = pinfo["num_solutions"]

            if not pinfo["use_two_stage"]:
                # Single-stage: process directly
                custom_id = f"problem_{problem_idx}"
                if custom_id not in raw_results:
                    self.logger.error(f"Missing result for problem {problem_idx}")
                    single_stage_results.append({
                        "problem_idx": problem_idx,
                        "error": "Missing result in batch output",
                        "clustering_mode": "single_stage",
                        "timestamp": time.time()
                    })
                    continue

                response_text = self._extract_response_text(raw_results[custom_id])
                if response_text is None:
                    self.logger.error(f"API error for problem {problem_idx}")
                    single_stage_results.append({
                        "problem_idx": problem_idx,
                        "error": f"API error: {raw_results[custom_id]['response']['status_code']}",
                        "clustering_mode": "single_stage",
                        "timestamp": time.time()
                    })
                    continue

                parsed = parse_response(response_text)
                if parsed is None:
                    single_stage_results.append({
                        "problem_idx": problem_idx,
                        "error": "Failed to parse response",
                        "raw_response": response_text[:500],
                        "clustering_mode": "single_stage",
                        "timestamp": time.time()
                    })
                    continue

                pred_grouping = {g["group_name"]: g["solution_ids"] for g in parsed["groups"]}
                pred_labels = grouping_to_labels(pred_grouping, num_solutions)
                if pred_labels is None:
                    pred_grouping, _ = repair_grouping_dict(pred_grouping, num_solutions)
                    pred_labels = grouping_to_labels(pred_grouping, num_solutions)

                if pred_labels is None:
                    single_stage_results.append({
                        "problem_idx": problem_idx,
                        "error": "Invalid grouping structure",
                        "predicted_grouping": pred_grouping,
                        "clustering_mode": "single_stage",
                        "timestamp": time.time()
                    })
                    continue

                single_stage_results.append({
                    "problem_idx": problem_idx,
                    "problem": original_data[problem_key],
                    "num_solutions": num_solutions,
                    "predicted_grouping": pred_grouping,
                    "reasoning_trace": parsed["reasoning_trace"],
                    "clustering_mode": "single_stage",
                    "timestamp": time.time()
                })

            else:
                # Two-stage: collect chunk results
                solutions = pinfo["solutions"]
                chunks = divide_into_chunks(solutions, self.config.chunk_size)
                chunk_offsets = [i * self.config.chunk_size for i in range(len(chunks))]

                stage1_parsed = []
                stage1_reasoning = []
                for chunk_idx in range(len(chunks)):
                    custom_id = f"problem_{problem_idx}_chunk_{chunk_idx}"
                    if custom_id not in raw_results:
                        self.logger.warning(f"Missing chunk {chunk_idx} for problem {problem_idx}")
                        stage1_parsed.append(None)
                        stage1_reasoning.append(None)
                        continue

                    response_text = self._extract_response_text(raw_results[custom_id])
                    if response_text is None:
                        self.logger.warning(f"API error for problem {problem_idx} chunk {chunk_idx}")
                        stage1_parsed.append(None)
                        stage1_reasoning.append(None)
                        continue

                    parsed = parse_response(response_text)
                    if parsed is None:
                        self.logger.warning(f"Parse error for problem {problem_idx} chunk {chunk_idx}")
                        stage1_parsed.append(None)
                        stage1_reasoning.append(response_text[:500])
                        continue

                    stage1_parsed.append(parsed)
                    stage1_reasoning.append(parsed.get("reasoning_trace", ""))

                two_stage_intermediates[problem_idx] = {
                    "stage1_parsed": stage1_parsed,
                    "stage1_reasoning": stage1_reasoning,
                    "chunk_offsets": chunk_offsets,
                    "solutions": solutions
                }

        return single_stage_results, two_stage_intermediates

    def _prepare_stage2_batch(
        self,
        two_stage_intermediates: Dict[int, Dict],
        problems_info: Dict[int, Dict]
    ) -> List[Dict]:
        """Create batch requests for Stage 2 of two-stage problems."""
        requests = []

        for problem_idx, intermediate in sorted(two_stage_intermediates.items()):
            pinfo = problems_info[problem_idx]
            original_data = pinfo["original_data"]
            problem_text = original_data[pinfo["problem_key"]]
            solutions = intermediate["solutions"]

            # Aggregate representatives
            representatives = aggregate_stage1_results(
                intermediate["stage1_parsed"],
                intermediate["chunk_offsets"],
                solutions
            )
            intermediate["representatives"] = representatives

            if len(representatives) <= 1:
                # Skip Stage 2 for this problem, will use fallback
                self.logger.info(f"Problem {problem_idx}: {len(representatives)} representatives, skipping Stage 2")
                continue

            # Check if all chunks failed
            if not representatives:
                self.logger.warning(f"Problem {problem_idx}: all chunks failed, no representatives")
                continue

            rep_solutions = [r["solution_text"] for r in representatives]
            messages = build_prompt(problem_text, rep_solutions, system_prompt=self.system_prompt)
            requests.append({
                "custom_id": f"problem_{problem_idx}_stage2",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": self._make_batch_body(messages)
            })

        return requests

    def _process_stage2_batch_results(
        self,
        raw_results: Dict[str, Dict],
        two_stage_intermediates: Dict[int, Dict],
        problems_info: Dict[int, Dict]
    ) -> List[Dict]:
        """Process Stage 2 batch results and produce final groupings for two-stage problems."""
        results = []

        for problem_idx, intermediate in sorted(two_stage_intermediates.items()):
            pinfo = problems_info[problem_idx]
            original_data = pinfo["original_data"]
            problem_key = pinfo["problem_key"]
            num_solutions = pinfo["num_solutions"]
            representatives = intermediate.get("representatives", [])
            chunk_offsets = intermediate["chunk_offsets"]
            stage1_parsed = intermediate["stage1_parsed"]
            stage1_reasoning = intermediate["stage1_reasoning"]

            # Check if all chunks failed
            if not any(p is not None for p in stage1_parsed):
                results.append({
                    "problem_idx": problem_idx,
                    "error": "Two-stage clustering failed (all chunks failed)",
                    "clustering_mode": "two_stage",
                    "timestamp": time.time()
                })
                continue

            # If only 0-1 representatives, use fallback (no Stage 2 was submitted)
            if len(representatives) <= 1:
                final_grouping = fallback_stage1_only(stage1_parsed, chunk_offsets, num_solutions)
                pred_labels = grouping_to_labels(final_grouping, num_solutions)
                if pred_labels is None:
                    results.append({
                        "problem_idx": problem_idx,
                        "error": "Invalid fallback grouping",
                        "clustering_mode": "two_stage",
                        "timestamp": time.time()
                    })
                    continue
                results.append({
                    "problem_idx": problem_idx,
                    "problem": original_data[problem_key],
                    "num_solutions": num_solutions,
                    "predicted_grouping": final_grouping,
                    "reasoning_trace": stage1_reasoning,
                    "clustering_mode": "two_stage",
                    "two_stage_detail": {
                        "stage1_results": [_summarize_parsed(p) for p in stage1_parsed],
                        "stage2_result": None,
                        "representatives": [_summarize_rep(r) for r in representatives]
                    },
                    "timestamp": time.time()
                })
                continue

            # Look for Stage 2 result
            custom_id = f"problem_{problem_idx}_stage2"
            if custom_id not in raw_results:
                self.logger.warning(f"Missing Stage 2 result for problem {problem_idx}, falling back")
                final_grouping = fallback_stage1_only(stage1_parsed, chunk_offsets, num_solutions)
                stage2_detail = {"error": "missing result"}
            else:
                response_text = self._extract_response_text(raw_results[custom_id])
                if response_text is None:
                    self.logger.warning(f"Stage 2 API error for problem {problem_idx}, falling back")
                    final_grouping = fallback_stage1_only(stage1_parsed, chunk_offsets, num_solutions)
                    stage2_detail = {"error": f"API error: {raw_results[custom_id]['response']['status_code']}"}
                else:
                    stage2_parsed = parse_response(response_text)
                    if stage2_parsed is None:
                        self.logger.warning(f"Stage 2 parse error for problem {problem_idx}, falling back")
                        final_grouping = fallback_stage1_only(stage1_parsed, chunk_offsets, num_solutions)
                        stage2_detail = {"error": "parse failure", "raw": response_text[:500]}
                    else:
                        final_grouping = map_stage2_to_final(
                            stage2_parsed, representatives, num_solutions,
                            stage1_parsed, chunk_offsets, self.config.chunk_size
                        )
                        stage2_detail = _summarize_parsed(stage2_parsed)

            pred_labels = grouping_to_labels(final_grouping, num_solutions)
            if pred_labels is None:
                final_grouping, _ = repair_grouping_dict(final_grouping, num_solutions)
                pred_labels = grouping_to_labels(final_grouping, num_solutions)
            if pred_labels is None:
                results.append({
                    "problem_idx": problem_idx,
                    "error": "Invalid two-stage grouping structure",
                    "predicted_grouping": final_grouping,
                    "clustering_mode": "two_stage",
                    "timestamp": time.time()
                })
                continue

            combined_reasoning = stage1_reasoning + (
                [stage2_detail.get("reasoning_trace", "")] if isinstance(stage2_detail, dict) and "groups" in stage2_detail else []
            )

            results.append({
                "problem_idx": problem_idx,
                "problem": original_data[problem_key],
                "num_solutions": num_solutions,
                "predicted_grouping": final_grouping,
                "reasoning_trace": combined_reasoning if combined_reasoning else stage1_reasoning,
                "clustering_mode": "two_stage",
                "two_stage_detail": {
                    "stage1_results": [_summarize_parsed(p) for p in stage1_parsed],
                    "stage2_result": stage2_detail,
                    "representatives": [_summarize_rep(r) for r in representatives]
                },
                "timestamp": time.time()
            })

        return results

    def _fallback_all_stage1(
        self,
        two_stage_intermediates: Dict[int, Dict],
        problems_info: Dict[int, Dict]
    ) -> List[Dict]:
        """Fallback: produce results for all two-stage problems using Stage 1 only."""
        results = []

        for problem_idx, intermediate in sorted(two_stage_intermediates.items()):
            pinfo = problems_info[problem_idx]
            original_data = pinfo["original_data"]
            problem_key = pinfo["problem_key"]
            num_solutions = pinfo["num_solutions"]
            stage1_parsed = intermediate["stage1_parsed"]
            chunk_offsets = intermediate["chunk_offsets"]
            representatives = intermediate.get("representatives", [])

            if not any(p is not None for p in stage1_parsed):
                results.append({
                    "problem_idx": problem_idx,
                    "error": "Two-stage clustering failed (all chunks failed)",
                    "clustering_mode": "two_stage",
                    "timestamp": time.time()
                })
                continue

            final_grouping = fallback_stage1_only(stage1_parsed, chunk_offsets, num_solutions)
            pred_labels = grouping_to_labels(final_grouping, num_solutions)

            if pred_labels is None:
                results.append({
                    "problem_idx": problem_idx,
                    "error": "Invalid fallback grouping",
                    "clustering_mode": "two_stage",
                    "timestamp": time.time()
                })
                continue

            results.append({
                "problem_idx": problem_idx,
                "problem": original_data[problem_key],
                "num_solutions": num_solutions,
                "predicted_grouping": final_grouping,
                "reasoning_trace": intermediate.get("stage1_reasoning", []),
                "clustering_mode": "two_stage",
                "two_stage_detail": {
                    "stage1_results": [_summarize_parsed(p) for p in stage1_parsed],
                    "stage2_result": {"error": "Stage 2 batch failed"},
                    "representatives": [_summarize_rep(r) for r in representatives]
                },
                "timestamp": time.time()
            })

        return results


class ClusteringEvaluator:
    """Main orchestrator for clustering evaluation."""

    def __init__(self, config: ClusteringConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.cost_tracker = CostTracker()
        self.client = GPTClient(config, self.cost_tracker)

        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)

    def _setup_logging(self) -> logging.Logger:
        """Setup logging with file and console handlers."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)

        # File handler - always DEBUG level
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.config.output_dir, f"clustering_eval_{timestamp}.log")
        os.makedirs(self.config.output_dir, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Console handler - respects config
        if not self.config.quiet:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, self.config.log_level.upper()))
            console_formatter = logging.Formatter('%(levelname)s - %(message)s')
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

        return logger

    def run(self, input_file: str) -> Dict:
        """Main execution flow."""
        self.logger.info(f"Starting clustering evaluation in {self.config.mode} mode")
        self.logger.info(f"Model: {self.config.model}")
        self.logger.info(f"Input: {input_file}")

        # Extract input filename for output naming
        input_basename = Path(input_file).stem  # Get filename without extension
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(
            self.config.output_dir,
            f"{self.config.output_prefix}_{input_basename}_{timestamp}_reasoning_{self.config.reasoning_effort}.json"
        )

        # Process based on mode
        if self.config.mode == "realtime":
            processor = RealtimeProcessor(self.config, self.client, self.cost_tracker)
            results = processor.process(input_file, output_file)
        elif self.config.mode == "batch":
            processor = BatchProcessor(self.config, self.client, self.cost_tracker)
            results = processor.process(input_file, self.config.output_dir)
        else:
            raise ValueError(f"Unknown mode: {self.config.mode}")

        # Add cost information to results
        is_batch = self.config.mode == "batch"
        results["cost_summary"] = self.cost_tracker.get_summary(self.config.model, is_batch)

        # Generate summary and save results
        if self.config.eval:
            summary = self.generate_summary(results)
            self.save_results(results, summary, output_file)
        else:
            self.save_results_predictions_only(results, output_file)

        return results

    def generate_summary(self, results: Dict) -> Dict:
        """Aggregate metrics across all results."""
        successful_results = [r for r in results["results"] if "metrics" in r]

        if not successful_results:
            return {
                "error": "No successful results to summarize"
            }

        # Extract metrics
        ari_scores = [r["metrics"]["adjusted_rand_index"] for r in successful_results]
        homogeneity_scores = [r["metrics"]["homogeneity"] for r in successful_results]
        completeness_scores = [r["metrics"]["completeness"] for r in successful_results]
        latency_scores = [r["latency_seconds"] for r in successful_results if "latency_seconds" in r]

        summary = {
            "mean_ari": float(np.mean(ari_scores)),
            "median_ari": float(np.median(ari_scores)),
            "std_ari": float(np.std(ari_scores)),
            "min_ari": float(np.min(ari_scores)),
            "max_ari": float(np.max(ari_scores)),
            "mean_homogeneity": float(np.mean(homogeneity_scores)),
            "std_homogeneity": float(np.std(homogeneity_scores)),
            "mean_completeness": float(np.mean(completeness_scores)),
            "std_completeness": float(np.std(completeness_scores)),
            "mean_latency_seconds": float(np.mean(latency_scores)) if latency_scores else None,
            "median_latency_seconds": float(np.median(latency_scores)) if latency_scores else None,
            "p90_latency_seconds": float(np.percentile(latency_scores, 90)) if latency_scores else None,
            "max_latency_seconds": float(np.max(latency_scores)) if latency_scores else None,
        }

        return summary

    def save_results(self, results: Dict, summary: Dict, output_file: str):
        """Save results and summary to files."""
        # Save results JSON
        results["summary_metrics"] = summary

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        self.logger.info(f"Saved results to: {output_file}")

        # Save human-readable summary
        # Extract input filename from metadata
        input_basename = Path(results['metadata']['input_file']).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = os.path.join(self.config.output_dir, f"metrics_summary_{input_basename}_{timestamp}.txt")

        with open(summary_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("CLUSTERING EVALUATION SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Timestamp: {results['metadata']['timestamp']}\n")
            f.write(f"Model: {results['metadata']['model']}\n")
            f.write(f"Mode: {results['metadata']['mode']}\n")
            f.write(f"Input File: {results['metadata']['input_file']}\n\n")

            f.write(f"Total Problems: {results['metadata']['num_problems']}\n")
            f.write(f"Successful: {results['metadata']['num_successful']}\n")
            f.write(f"Failed: {results['metadata']['num_failed']}\n\n")

            f.write("-" * 80 + "\n")
            f.write("METRICS\n")
            f.write("-" * 80 + "\n\n")

            if "error" not in summary:
                f.write(f"Adjusted Rand Index (ARI):\n")
                f.write(f"  Mean:   {summary['mean_ari']:.4f}\n")
                f.write(f"  Median: {summary['median_ari']:.4f}\n")
                f.write(f"  Std:    {summary['std_ari']:.4f}\n")
                f.write(f"  Min:    {summary['min_ari']:.4f}\n")
                f.write(f"  Max:    {summary['max_ari']:.4f}\n\n")

                f.write(f"Homogeneity:\n")
                f.write(f"  Mean: {summary['mean_homogeneity']:.4f}\n")
                f.write(f"  Std:  {summary['std_homogeneity']:.4f}\n\n")

                f.write(f"Completeness:\n")
                f.write(f"  Mean: {summary['mean_completeness']:.4f}\n")
                f.write(f"  Std:  {summary['std_completeness']:.4f}\n\n")

                if summary.get("mean_latency_seconds") is not None:
                    f.write(f"Latency (seconds per problem):\n")
                    f.write(f"  Mean:   {summary['mean_latency_seconds']:.2f}\n")
                    f.write(f"  Median: {summary['median_latency_seconds']:.2f}\n")
                    f.write(f"  P90:    {summary['p90_latency_seconds']:.2f}\n")
                    f.write(f"  Max:    {summary['max_latency_seconds']:.2f}\n\n")
            else:
                f.write(f"ERROR: {summary['error']}\n\n")

            # Add cost summary to text file
            if "cost_summary" in results:
                cost = results["cost_summary"]
                f.write("-" * 80 + "\n")
                f.write("API COST SUMMARY\n")
                f.write("-" * 80 + "\n\n")
                f.write(f"Pricing Mode:      {cost['pricing_mode']}\n")
                f.write(f"Total Requests:    {cost['total_requests']:,}\n")
                f.write(f"Input Tokens:      {cost['input_tokens']:,}\n")
                f.write(f"Output Tokens:     {cost['output_tokens']:,}\n")
                if cost['reasoning_tokens'] > 0:
                    f.write(f"Reasoning Tokens:  {cost['reasoning_tokens']:,}\n")
                f.write(f"Total Tokens:      {cost['total_tokens']:,}\n")
                f.write(f"Total Cost:        ${cost['total_cost_usd']:.4f}\n\n")

        self.logger.info(f"Saved summary to: {summary_file}")

    def save_results_predictions_only(self, results: Dict, output_file: str):
        """Save prediction results without evaluation metrics."""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        self.logger.info(f"Saved prediction results to: {output_file}")


def main():
    """Entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Evaluate GPT-based solution clustering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Realtime mode
  python sol_diversity_judge.py input.jsonl --model gpt-5.2

  # Batch mode (always waits for completion)
  python sol_diversity_judge.py input.jsonl --mode batch

  # Resume interrupted run
  python sol_diversity_judge.py input.jsonl --resume results_*.json

  # Test on subset
  python sol_diversity_judge.py input.jsonl --limit 5
        """
    )

    # Required arguments
    parser.add_argument("input_file", help="Path to JSONL input file")

    # API configuration
    parser.add_argument("--api-key", help="OpenAI API key (or use OPENAI_API_KEY env var)")
    parser.add_argument("--api-base", help="OpenAI-compatible API base URL (or use OPENAI_BASE_URL env var)")
    parser.add_argument("--model", default="gpt-5.2", help="Model name (default: gpt-5.2)")
    parser.add_argument("--reasoning-effort", choices=["none", "low", "medium"], default="none",
                        help="Reasoning effort level (default: none)")
    parser.add_argument("--thinking-token-budget", type=int,
                        help="Per-request reasoning token budget for OpenAI-compatible servers that support it")
    parser.add_argument("--temperature", type=float,
                        help="Sampling temperature. Set 0 for maximally deterministic decoding when supported.")
    parser.add_argument("--top-p", type=float,
                        help="Nucleus sampling parameter. For deterministic decoding, pair with --temperature 0.")
    parser.add_argument("--top-k", type=int,
                        help="Top-k sampling cutoff for OpenAI-compatible servers such as vLLM.")
    parser.add_argument("--min-p", type=float,
                        help="Min-p sampling threshold for OpenAI-compatible servers such as vLLM.")
    parser.add_argument("--presence-penalty", type=float,
                        help="Presence penalty for generated text.")
    parser.add_argument("--repetition-penalty", type=float,
                        help="Repetition penalty for OpenAI-compatible servers such as vLLM.")
    parser.add_argument("--seed", type=int,
                        help="Sampling seed for OpenAI-compatible servers that support deterministic seeding.")

    # Mode configuration
    parser.add_argument("--mode", choices=["realtime", "batch"], default="realtime",
                        help="Processing mode (default: realtime)")

    # Realtime settings
    parser.add_argument("--resume", help="Resume from results file (realtime only)")
    parser.add_argument("--resume-stage1", help="Resume batch mode from saved Stage 1 raw results file")
    parser.add_argument("--resume-stage2-input-file-id",
                        help="OpenAI file ID (file-*) of an already-submitted Stage 2 batch input; "
                             "poll & retrieve it instead of submitting a new batch (requires --resume-stage1)")
    parser.add_argument("--rate-limit", type=int, default=60,
                        help="Requests per minute for realtime mode (default: 60)")
    parser.add_argument("--max-retries", type=int, default=5,
                        help="Max retries for failures (default: 5)")
    parser.add_argument("--timeout", type=int, default=120,
                        help="Per-request timeout in seconds for realtime mode (default: 120)")

    # Batch settings
    parser.add_argument("--poll-interval", type=int, default=60,
                        help="Batch polling interval in seconds (default: 60)")

    # General settings
    parser.add_argument("--output-dir", default="outputs/clustering_results",
                        help="Output directory (default: outputs/clustering_results)")
    parser.add_argument("--output-prefix", default="results",
                        help="Prefix for output files (default: results)")
    parser.add_argument("--limit", type=int, help="Process only first N problems")
    parser.add_argument("--chunk-size", type=int, default=8,
                        help="Max solutions per chunk for two-stage clustering")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level (default: INFO)")
    parser.add_argument("--num-workers", type=int, default=16,
                        help="Number of parallel workers for realtime mode (default: 16)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress bars and console output")
    parser.add_argument("--eval", action="store_true", default=False,
                        help="Evaluate predictions against golden labels (default: off)")
    parser.add_argument("--min-correct", type=int, default=0,
                        help="Skip problems with this many or fewer correct solutions (default: 0, i.e. skip only if 0)")

    args = parser.parse_args()

    # Validate chunk_size
    if args.chunk_size is not None and args.chunk_size < 2:
        print("Error: --chunk-size must be at least 2", file=sys.stderr)
        sys.exit(1)

    # Get API key
    api_base = args.api_base or os.environ.get("OPENAI_BASE_URL")
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key and api_base:
        # Many OpenAI-compatible local servers (for example vLLM) accept any non-empty API key.
        api_key = "EMPTY"
    if not api_key:
        print("Error: OpenAI API key required. Set OPENAI_API_KEY env var or use --api-key", file=sys.stderr)
        sys.exit(1)

    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)

    # Create config
    config = ClusteringConfig(
        api_key=api_key,
        api_base=api_base,
        model=args.model,
        mode=args.mode,
        reasoning_effort=args.reasoning_effort,
        thinking_token_budget=args.thinking_token_budget,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        presence_penalty=args.presence_penalty,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
        rate_limit_rpm=args.rate_limit,
        max_retries=args.max_retries,
        timeout=args.timeout,
        poll_interval=args.poll_interval,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        log_level=args.log_level,
        limit=args.limit,
        resume_file=args.resume,
        resume_stage1=args.resume_stage1,
        resume_stage2_input_file_id=args.resume_stage2_input_file_id,
        quiet=args.quiet,
        eval=args.eval,
        chunk_size=args.chunk_size,
        num_workers=args.num_workers,
        min_correct=args.min_correct,
    )

    # Run evaluation
    evaluator = ClusteringEvaluator(config)

    try:
        results = evaluator.run(args.input_file)
        print(f"\n{'='*80}")
        print("COMPLETE")
        print(f"{'='*80}")
        print(f"Results saved to: {config.output_dir}")

        if config.eval and "summary_metrics" in results and "error" not in results["summary_metrics"]:
            summary = results["summary_metrics"]
            print(f"\nSummary Metrics:")
            print(f"  Mean ARI: {summary['mean_ari']:.4f}")
            print(f"  Mean Homogeneity: {summary['mean_homogeneity']:.4f}")
            print(f"  Mean Completeness: {summary['mean_completeness']:.4f}")

        # Print cost summary
        if "cost_summary" in results:
            cost = results["cost_summary"]
            print(f"\n{'='*80}")
            print("API COST SUMMARY")
            print(f"{'='*80}")
            print(f"  Pricing Mode:      {cost['pricing_mode']}")
            print(f"  Total Requests:    {cost['total_requests']:,}")
            print(f"  Input Tokens:      {cost['input_tokens']:,}")
            print(f"  Output Tokens:     {cost['output_tokens']:,}")
            if cost['reasoning_tokens'] > 0:
                print(f"  Reasoning Tokens:  {cost['reasoning_tokens']:,}")
            print(f"  Total Tokens:      {cost['total_tokens']:,}")
            print(f"  Total Cost:        ${cost['total_cost_usd']:.4f}")
            print(f"{'='*80}")

    except KeyboardInterrupt:
        # Print cost summary even on interrupt
        if evaluator and evaluator.cost_tracker:
            is_batch = config.mode == "batch"
            cost = evaluator.cost_tracker.get_summary(config.model, is_batch)
            print(f"\n\n{'='*80}")
            print("INTERRUPTED - API COST SUMMARY (Partial)")
            print(f"{'='*80}")
            print(f"  Pricing Mode:      {cost['pricing_mode']}")
            print(f"  Total Requests:    {cost['total_requests']:,}")
            print(f"  Input Tokens:      {cost['input_tokens']:,}")
            print(f"  Output Tokens:     {cost['output_tokens']:,}")
            if cost['reasoning_tokens'] > 0:
                print(f"  Reasoning Tokens:  {cost['reasoning_tokens']:,}")
            print(f"  Total Tokens:      {cost['total_tokens']:,}")
            print(f"  Total Cost:        ${cost['total_cost_usd']:.4f}")
            print(f"{'='*80}\n")
        print("Interrupted by user. Partial results may be saved.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        # Print cost summary even on error
        if evaluator and evaluator.cost_tracker:
            is_batch = config.mode == "batch"
            cost = evaluator.cost_tracker.get_summary(config.model, is_batch)
            print(f"\n\n{'='*80}")
            print("ERROR - API COST SUMMARY (Partial)")
            print(f"{'='*80}")
            print(f"  Pricing Mode:      {cost['pricing_mode']}")
            print(f"  Total Requests:    {cost['total_requests']:,}")
            print(f"  Input Tokens:      {cost['input_tokens']:,}")
            print(f"  Output Tokens:     {cost['output_tokens']:,}")
            if cost['reasoning_tokens'] > 0:
                print(f"  Reasoning Tokens:  {cost['reasoning_tokens']:,}")
            print(f"  Total Tokens:      {cost['total_tokens']:,}")
            print(f"  Total Cost:        ${cost['total_cost_usd']:.4f}")
            print(f"{'='*80}\n")
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
        
if __name__ == "__main__":
    main()
