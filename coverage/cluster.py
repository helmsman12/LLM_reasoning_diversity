"""Batch-oriented LLM judge wrapper.

Thin wrapper around
``evaluate_solutions/sol_diversity_judge.py``
that hands an entire list of clustering jobs to ``BatchProcessor`` in a
single OpenAI Batch API submission. This matches
``cluster_generations.sh`` (the reference protocol) and unlocks the
50% batch-pricing discount instead of paying per-problem realtime
calls.

See ``docs/plan.md`` § 3 for the interface contract with ``evaluate.py``.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Load OPENAI_API_KEY (and friends) from the package-level .env, if present.
# Variables already set in the environment are never overridden.
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
except ImportError:  # python-dotenv is optional; fall back to `source .env`
    pass

# ---------------------------------------------------------------------
# sol_diversity_judge import shim
# ---------------------------------------------------------------------

DEFAULT_JUDGE_DIR = os.path.join(os.path.dirname(__file__), "..", "evaluate_solutions")
JUDGE_DIR = os.environ.get("DIV_BENCH_JUDGE_DIR", DEFAULT_JUDGE_DIR)
if JUDGE_DIR not in sys.path:
    sys.path.insert(0, JUDGE_DIR)

# Imported lazily-but-eagerly so import errors surface immediately.
from sol_diversity_judge import (  # noqa: E402
    BatchProcessor,
    ClusteringConfig as JudgeClusteringConfig,
    CostTracker,
    GPTClient,
    RealtimeProcessor,
    grouping_to_labels,
)


# ---------------------------------------------------------------------
# Public dataclasses (contract with evaluate.py)
# ---------------------------------------------------------------------


@dataclass
class ClusterJob:
    """One problem's clustering job.

    ``correct_solutions`` must already contain *only* the verified-
    correct solution texts. ``problem_id`` is opaque to cluster.py and
    is round-tripped back to evaluate.py via ``BatchClusterResult``.
    """

    problem_id: str
    problem_text: str
    correct_solutions: List[str]


@dataclass
class BatchClusterResult:
    labels_by_problem: Dict[str, Optional[List[int]]]
    raw_response_paths: Dict[str, Optional[str]]
    batch_metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------


def _build_judge_config(
    *,
    model: str,
    mode: str,
    reasoning_effort: str,
    chunk_size: int,
    min_correct: int,
    output_dir: str,
    output_prefix: str,
    num_workers: int = 1,
) -> JudgeClusteringConfig:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set; cluster.py requires it to call "
            "the OpenAI Batch API"
        )
    return JudgeClusteringConfig(
        api_key=api_key,
        model=model,
        mode=mode,
        reasoning_effort=reasoning_effort,
        chunk_size=chunk_size,
        min_correct=min_correct,
        output_dir=output_dir,
        output_prefix=output_prefix,
        num_workers=num_workers,
        eval=False,  # we have no golden grouping
    )


def _write_temp_input_jsonl(jobs: List[ClusterJob], path: str) -> None:
    """Emit a jsonl in the format BatchProcessor expects.

    ``_categorize_problems`` looks for ``problem`` (or ``question``)
    and ``solutions`` (or ``solutions_to_eval``). If ``scores`` is
    absent it keeps all solutions. We pass only correct solutions and
    omit ``scores`` to keep the file simple.
    """
    with open(path, "w") as f:
        for idx, job in enumerate(jobs):
            rec = {
                "problem_idx": idx,
                "problem": job.problem_text,
                "solutions": list(job.correct_solutions),
            }
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")


def _extract_labels(result_entry: Dict[str, Any]) -> Optional[List[int]]:
    """Convert a BatchProcessor result entry to a problem-local label list.

    Returns ``None`` if the entry is an error / missing / unparseable
    grouping.
    """
    if "error" in result_entry:
        return None
    grouping = result_entry.get("predicted_grouping")
    num_solutions = result_entry.get("num_solutions")
    if grouping is None or num_solutions is None:
        return None
    labels = grouping_to_labels(grouping, num_solutions)
    if labels is None:
        return None
    return [int(x) for x in labels.tolist()]


# ---------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------


def cluster_batch(
    jobs: List[ClusterJob],
    *,
    model: str = "gpt-5.2",
    mode: str = "batch",
    reasoning_effort: str = "none",
    chunk_size: int = 8,
    min_correct: int = 2,
    output_dir: str = "clustering_responses",
    run_tag: str = "",
    resume_batch_id: Optional[str] = None,
    num_workers: int = 1,
    resume_realtime_file: Optional[str] = None,
) -> BatchClusterResult:
    """Cluster a batch of problems via the LLM judge.

    Submits a single OpenAI Batch API job (Stage 1 + optional Stage 2,
    handled inside ``BatchProcessor``) and returns a problem-id-keyed
    mapping back to evaluate.py.

    Returned ``labels_by_problem[pid]`` is ``None`` for any problem
    that:
      - was filtered out by ``min_correct`` inside the judge,
      - returned an error from the batch API,
      - or whose response failed to parse into a valid grouping.

    ``raw_response_paths[pid]`` points to the per-run Stage 1 raw
    jsonl (shared across all problems in the same batch — the judge
    does not split per-problem files).
    """
    logger = logging.getLogger(__name__)

    if not jobs:
        return BatchClusterResult(
            labels_by_problem={},
            raw_response_paths={},
            batch_metadata={"num_jobs": 0},
        )

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    safe_tag = (run_tag or "run").replace("/", "_")
    output_prefix = f"{safe_tag}_{timestamp}"

    judge_config = _build_judge_config(
        model=model,
        mode=mode,
        reasoning_effort=reasoning_effort,
        chunk_size=chunk_size,
        min_correct=min_correct,
        output_dir=str(output_dir_path),
        output_prefix=output_prefix,
        num_workers=num_workers,
    )

    # idx → problem_id (so we can round-trip after BatchProcessor sorts
    # results by problem_idx).
    idx_to_pid: Dict[int, str] = {i: job.problem_id for i, job in enumerate(jobs)}

    # Write temp input jsonl into the same output dir so all batch
    # artefacts live together. The stem deliberately omits a
    # ``batch_input_`` prefix — BatchProcessor reuses this stem when
    # naming its own ``batch_input_stage{1,2}_<stem>_<ts>.jsonl`` and
    # ``batch_raw_stage{1,2}_<stem>_<ts>.jsonl`` files, so a clean stem
    # keeps model_id/checkpoint/eval_set immediately visible without
    # prefix duplication.
    input_jsonl = output_dir_path / f"{output_prefix}.jsonl"
    _write_temp_input_jsonl(jobs, str(input_jsonl))
    logger.info(
        "cluster_batch: %d jobs → %s (model=%s mode=%s chunk_size=%d min_correct=%d)",
        len(jobs),
        input_jsonl,
        model,
        mode,
        chunk_size,
        min_correct,
    )

    cost_tracker = CostTracker()
    client = GPTClient(judge_config, cost_tracker=cost_tracker)

    # If resuming from an existing batch, poll it and retrieve results
    # so BatchProcessor can skip Stage 1 submission via resume_stage1.
    if resume_batch_id:
        logger.info("resuming from existing batch job: %s", resume_batch_id)
        if not client.poll_batch(resume_batch_id):
            raise RuntimeError(
                f"Resumed batch {resume_batch_id} failed/expired/cancelled"
            )
        raw1_path = str(
            output_dir_path
            / f"batch_raw_stage1_{input_jsonl.stem}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jsonl"
        )
        retrieved = client.retrieve_batch_results(resume_batch_id, raw1_path)
        if retrieved is None:
            raise RuntimeError(
                f"Failed to retrieve results for batch {resume_batch_id}"
            )
        judge_config.resume_stage1 = retrieved
        logger.info("batch results saved to %s; passing as resume_stage1", retrieved)

    if mode == "realtime":
        realtime_output_file = output_dir_path / f"realtime_results_{output_prefix}.json"
        if resume_realtime_file:
            judge_config.resume_file = resume_realtime_file
            logger.info(
                "cluster_batch: realtime resume from %s",
                resume_realtime_file,
            )
        processor = RealtimeProcessor(judge_config, client, cost_tracker)
        logger.info(
            "cluster_batch: dispatching to RealtimeProcessor (num_workers=%d)",
            num_workers,
        )
        judge_output = processor.process(str(input_jsonl), str(realtime_output_file))
        raw_path_str: Optional[str] = str(realtime_output_file)
    else:
        processor = BatchProcessor(judge_config, client, cost_tracker)
        judge_output = processor.process(str(input_jsonl), str(output_dir_path))
        # The judge writes batch_raw_stage1_<input_basename>_<ts>.jsonl into
        # output_dir. Surface that path for every problem (it's a shared
        # file — per-problem isolation is not provided upstream).
        input_basename = input_jsonl.stem
        candidate_raw_files = sorted(
            output_dir_path.glob(f"batch_raw_stage1_{input_basename}_*.jsonl")
        )
        raw_path_str = (
            str(candidate_raw_files[-1]) if candidate_raw_files else None
        )

    metadata = judge_output.get("metadata", {})
    results_list = judge_output.get("results", [])

    # Default every problem to None; fill in successful ones.
    labels_by_problem: Dict[str, Optional[List[int]]] = {
        pid: None for pid in idx_to_pid.values()
    }
    raw_response_paths: Dict[str, Optional[str]] = {
        pid: None for pid in idx_to_pid.values()
    }

    n_ok = 0
    n_failed = 0
    for entry in results_list:
        pidx = entry.get("problem_idx")
        if pidx is None or pidx not in idx_to_pid:
            continue
        pid = idx_to_pid[pidx]
        labels = _extract_labels(entry)
        labels_by_problem[pid] = labels
        raw_response_paths[pid] = raw_path_str
        if labels is None:
            n_failed += 1
            logger.warning(
                "cluster_batch: problem_id=%s (idx=%d) → no labels (entry keys=%s)",
                pid,
                pidx,
                sorted(entry.keys()),
            )
        else:
            n_ok += 1

    n_skipped_min_correct = sum(
        1 for pid, labels in labels_by_problem.items() if labels is None
    ) - n_failed
    cost = cost_tracker.get_cost(model, is_batch=(mode == "batch"))

    batch_metadata = {
        "num_jobs": len(jobs),
        "num_ok": n_ok,
        "num_failed": n_failed,
        "num_skipped_min_correct": n_skipped_min_correct,
        "input_jsonl": str(input_jsonl),
        "judge_metadata": metadata,
        "cost_usd": cost,
    }
    logger.info(
        "cluster_batch done: ok=%d failed=%d skipped_min_correct=%d cost=$%.4f",
        n_ok,
        n_failed,
        n_skipped_min_correct,
        cost,
    )

    return BatchClusterResult(
        labels_by_problem=labels_by_problem,
        raw_response_paths=raw_response_paths,
        batch_metadata=batch_metadata,
    )


def cluster_one(job: ClusterJob, **kwargs: Any) -> BatchClusterResult:
    """Convenience wrapper that batches a single job. Debug use only."""
    return cluster_batch([job], **kwargs)
