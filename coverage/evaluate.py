"""Approach coverage evaluation entry point (phase-based pipeline).

Pipeline (per ``(model_checkpoint, eval_set)``):

    Phase 1 — Verify all
        Verify every prediction across every problem with a vLLM-served
        verifier model (no rule-based fallback). Per-problem booleans.

    Phase 2 — Build cluster jobs
        Collect ``(problem_id, problem_text, correct_solutions)`` for
        every problem with at least one verified-correct solution.

    Phase 3 — Cluster (one batch)
        Hand the entire job list to ``cluster.cluster_batch`` which
        submits a single OpenAI Batch API job (Stage 1 + optional
        Stage 2). This matches ``cluster_generations.sh`` and unlocks
        the 50% batch-pricing discount instead of paying per-problem
        realtime calls.

    Phase 4 — Finalize
        For each problem: assemble the record, compute metrics, append
        one line to ``results/results.jsonl`` (append-only).

Sampling itself is delegated to the existing div-bench generation
pipeline; this script consumes the resulting jsonl. Re-running the
same ``(model_id, checkpoint, eval_set, problem_id)`` simply appends a
new line with a fresh timestamp — existing lines are never modified.

See ``CLAUDE.md`` for the record schema and ``docs/plan.md`` for the
full task list.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml
from openai import OpenAI

import metrics
from cluster import ClusterJob, BatchClusterResult, cluster_batch


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------


@dataclass
class GenerationConfig:
    temperature: float
    top_p: float
    max_tokens: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }


@dataclass
class VerifierConfig:
    base_url: str
    model_name: str
    max_inflight: int = 8
    max_retries: int = 3


@dataclass
class ClusteringConfig:
    model: str = "gpt-5.2"
    mode: str = "batch"
    reasoning_effort: str = "none"
    chunk_size: int = 8
    min_correct: int = 4
    output_dir: str = "clustering_responses"
    num_workers: int = 1


@dataclass
class EvalConfig:
    generation: GenerationConfig
    verifier: VerifierConfig
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    eval_sets: Dict[str, str] = field(default_factory=dict)
    n_sampled: int = 64

    @classmethod
    def from_yaml(cls, path: str) -> "EvalConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls(
            generation=GenerationConfig(**raw["generation"]),
            verifier=VerifierConfig(**raw["verifier"]),
            clustering=ClusteringConfig(**raw.get("clustering", {})),
            eval_sets=raw.get("eval_sets", {}),
            n_sampled=int(raw.get("n_sampled", 64)),
        )


# ----------------------------------------------------------------------
# Phase 1 — Verifier (vLLM-served, LLM judge — no rule-based fallback)
# ----------------------------------------------------------------------


_thread_local = threading.local()


def _get_verify_client(base_url: str) -> OpenAI:
    if not hasattr(_thread_local, "client"):
        _thread_local.client = OpenAI(base_url=base_url, api_key="EMPTY")
    return _thread_local.client


def query_verifier(
    cfg: VerifierConfig,
    golden_answer: str,
    predicted_answer: str,
) -> bool:
    """Ask the verifier model whether ``predicted_answer`` matches.

    Mirrors ``RLVR_expr/re_evaluate_generations.py::query_openai_for_verification``.
    Returns ``True`` only if the model outputs ``"correct"`` (case-
    insensitive). Failures after retries return ``False``.
    """
    system_prompt = (
        "You are a math expert.\n"
        "You are given a golden answer and a predicted answer from a solver.\n"
        "You need to verify if the predicted answer is correct.\n"
        'Only output "correct" or "incorrect".'
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "Golden answer: 540, Predicted answer: The total number of "
                "ways the cars can stack up so that all three lanes are "
                "occupied is calculated to be 750."
            ),
        },
        {"role": "assistant", "content": "incorrect"},
        {
            "role": "user",
            "content": "Golden answer: 3, Predicted answer: The ratio \\frac{A C}{A E} = 3.",
        },
        {"role": "assistant", "content": "correct"},
        {
            "role": "user",
            "content": f"Golden answer: {golden_answer}, Predicted answer: {predicted_answer}",
        },
    ]

    client = _get_verify_client(cfg.base_url)
    last_err: Optional[Exception] = None
    for _ in range(cfg.max_retries):
        try:
            resp = client.chat.completions.create(
                model=cfg.model_name,
                messages=messages,
                temperature=0.7,
                top_p=0.8,
                presence_penalty=1.5,
                max_completion_tokens=16,
                extra_body={
                    "top_k": 20,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            text = (resp.choices[0].message.content or "").strip().lower()
            return text == "correct"
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            continue
    logging.warning("verifier failed after retries: %s", last_err)
    return False


FALLBACK_TAIL_CHARS = 300


def _pred_or_fallback(pred: Optional[str], solution: Optional[str]) -> Optional[str]:
    """Return ``pred`` if non-empty, else the last 300 chars of ``solution``.

    Upstream extracts ``pred`` from ``\\boxed{...}``. When that fails the
    field is None/empty/"None"; rather than auto-failing, hand the
    verifier the tail of the raw solution so it can still judge.
    """
    if pred not in (None, "", "None"):
        return pred
    if solution:
        tail = solution[-FALLBACK_TAIL_CHARS:].strip()
        return tail or None
    return None


def verify_predictions_for_problem(
    cfg: VerifierConfig,
    golden_answer: str,
    preds: Sequence[Optional[str]],
    semaphore: threading.BoundedSemaphore,
    solutions: Optional[Sequence[Optional[str]]] = None,
) -> List[bool]:
    """Verify all predictions for one problem (parallel).

    When ``preds[i]`` is missing/empty (boxed extraction failed) and a
    matching ``solutions[i]`` is available, the last 300 chars of the
    raw solution are passed to the verifier as a fallback.
    """
    results: List[bool] = [False] * len(preds)
    sols: Sequence[Optional[str]] = solutions if solutions is not None else [None] * len(preds)

    def _one(i: int, pred: Optional[str], solution: Optional[str]) -> Tuple[int, bool]:
        effective = _pred_or_fallback(pred, solution)
        if effective is None:
            return i, False
        with semaphore:
            return i, query_verifier(cfg, golden_answer, effective)

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(cfg.max_inflight, 1)
    ) as executor:
        futures = [
            executor.submit(_one, i, p, sols[i] if i < len(sols) else None)
            for i, p in enumerate(preds)
        ]
        for fut in concurrent.futures.as_completed(futures):
            i, ok = fut.result()
            results[i] = ok
    return results


# ----------------------------------------------------------------------
# Generations loader
# ----------------------------------------------------------------------


def load_generations(path: str) -> List[Dict[str, Any]]:
    """Load a div-bench generations jsonl.

    Each line is expected to have at least ``problem``, ``answer``,
    ``solutions``, and ``pred``. ``id`` (or ``problem_id``) is used as
    the problem identifier; falls back to the line index.
    """
    records: List[Dict[str, Any]] = []
    with open(path, "r") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rec.setdefault("_line_index", idx)
            records.append(rec)
    return records


def _problem_id(rec: Dict[str, Any]) -> str:
    for key in ("problem_id", "id", "index", "item_index"):
        if key in rec and rec[key] is not None:
            return str(rec[key])
    return f"problem_{rec['_line_index']:04d}"


# ----------------------------------------------------------------------
# Per-problem intermediate state (between phases)
# ----------------------------------------------------------------------


@dataclass
class ProblemState:
    problem_id: str
    record: Dict[str, Any]
    n_sampled: int
    scores: List[bool] = field(default_factory=list)

    @property
    def n_correct(self) -> int:
        return sum(self.scores)

    @property
    def correct_solutions(self) -> List[str]:
        sols = list(self.record.get("solutions", []))
        return [sols[i] for i, ok in enumerate(self.scores) if ok and i < len(sols)]


# ----------------------------------------------------------------------
# JSON serialization (NaN → null)
# ----------------------------------------------------------------------


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _scrub_nans(obj: Any) -> Any:
    """Replace float NaN with None recursively (JSON spec compliance)."""
    if isinstance(obj, float):
        if obj != obj:
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _scrub_nans(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_scrub_nans(v) for v in obj]
    return obj


def append_record(results_path: str, record: Dict[str, Any]) -> None:
    """Append one JSON record as a single line. Append-only."""
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    serializable = _scrub_nans(record)
    with open(results_path, "a") as f:
        f.write(json.dumps(serializable, default=_json_default, ensure_ascii=False))
        f.write("\n")


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat()


# ----------------------------------------------------------------------
# Verification persistence (verify-only / cluster-only split)
# ----------------------------------------------------------------------


def save_verification(states: List[ProblemState], path: str) -> None:
    """Persist Phase 1 scores to a JSONL file for later cluster-only runs."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for st in states:
            line = {
                "problem_id": st.problem_id,
                "n_sampled": st.n_sampled,
                "n_correct": st.n_correct,
                "scores": st.scores,
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    logging.info("saved verification for %d problems to %s", len(states), path)


def load_verification(states: List[ProblemState], path: str) -> None:
    """Restore Phase 1 scores from a previously saved verification JSONL."""
    lookup: Dict[str, List[bool]] = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            lookup[rec["problem_id"]] = rec["scores"]
    matched = 0
    for st in states:
        scores = lookup.get(st.problem_id)
        if scores is not None:
            st.scores = scores
            matched += 1
        else:
            logging.warning(
                "no verification entry for problem %s — scores left empty",
                st.problem_id,
            )
    logging.info(
        "loaded verification from %s: %d/%d problems matched",
        path, matched, len(states),
    )


# ----------------------------------------------------------------------
# Phase orchestration
# ----------------------------------------------------------------------


def phase1_verify_all(
    states: List[ProblemState],
    cfg: VerifierConfig,
    failures_logger: logging.Logger,
) -> None:
    """Populate ``state.scores`` for every problem (in place)."""
    semaphore = threading.BoundedSemaphore(cfg.max_inflight)
    for i, st in enumerate(states):
        preds = list(st.record.get("pred", []))
        solutions = list(st.record.get("solutions", []))
        if not preds:
            failures_logger.warning("no predictions for problem %s", st.problem_id)
            st.scores = []
            continue
        try:
            st.scores = verify_predictions_for_problem(
                cfg, st.record.get("answer", ""), preds, semaphore, solutions=solutions
            )
        except Exception as exc:  # noqa: BLE001
            failures_logger.exception(
                "verify raised for problem %s: %s", st.problem_id, exc
            )
            st.scores = [False] * len(preds)
        logging.info(
            "[verify %d/%d] %s  n_correct=%d/%d",
            i + 1,
            len(states),
            st.problem_id,
            st.n_correct,
            len(preds),
        )


def phase2_build_jobs(states: List[ProblemState]) -> List[ClusterJob]:
    """Collect cluster jobs for every problem with ``n_correct > 0``.

    The cluster.py-side ``min_correct`` threshold is enforced inside
    the batch processor; we still hand it everything > 0 so that
    threshold decisions live in one place.
    """
    jobs: List[ClusterJob] = []
    for st in states:
        if st.n_correct == 0:
            continue
        jobs.append(
            ClusterJob(
                problem_id=st.problem_id,
                problem_text=st.record.get("problem", ""),
                correct_solutions=st.correct_solutions,
            )
        )
    return jobs


def phase3_cluster(
    jobs: List[ClusterJob],
    cfg: ClusteringConfig,
    run_tag: str,
    resume_batch_id: Optional[str] = None,
    resume_realtime_file: Optional[str] = None,
) -> BatchClusterResult:
    """Run the single batch clustering call."""
    if not jobs:
        logging.info("phase 3: no cluster jobs (no problems with correct solutions)")
        return BatchClusterResult(
            labels_by_problem={},
            raw_response_paths={},
            batch_metadata={"num_jobs": 0},
        )
    if resume_batch_id:
        logging.info("phase 3: resuming batch %s for %d cluster jobs", resume_batch_id, len(jobs))
    else:
        logging.info("phase 3: submitting %d cluster jobs as one batch", len(jobs))
    return cluster_batch(
        jobs,
        model=cfg.model,
        mode=cfg.mode,
        reasoning_effort=cfg.reasoning_effort,
        chunk_size=cfg.chunk_size,
        min_correct=cfg.min_correct,
        output_dir=cfg.output_dir,
        run_tag=run_tag,
        resume_batch_id=resume_batch_id,
        num_workers=cfg.num_workers,
        resume_realtime_file=resume_realtime_file,
    )


def phase4_finalize(
    states: List[ProblemState],
    cluster_result: BatchClusterResult,
    common: Dict[str, Any],
    cfg: EvalConfig,
    results_file: str,
) -> Tuple[int, int, int]:
    """Build a record per problem and append to results.jsonl."""
    n_ok = n_no_correct = n_failed = 0
    timestamp = _now_iso()

    for st in states:
        base: Dict[str, Any] = {
            **common,
            "problem_id": st.problem_id,
            "generation_config": cfg.generation.to_dict(),
            "n_sampled": st.n_sampled,
            "timestamp": timestamp,
        }

        if st.n_correct == 0:
            record = {
                **base,
                "status": "no_correct_solutions",
                "n_correct": 0,
                "n_approaches": 0,
                "approach_labels": [],
                "clustering_response_path": None,
                "metrics": {
                    **metrics.compute_all([]),
                },
            }
            n_no_correct += 1
        else:
            labels = cluster_result.labels_by_problem.get(st.problem_id)
            raw_path = cluster_result.raw_response_paths.get(st.problem_id)

            if labels is None:
                # Either batch failure or below min_correct threshold.
                record = {
                    **base,
                    "status": "clustering_failed",
                    "n_correct": st.n_correct,
                    "n_approaches": 0,
                    "approach_labels": [],
                    "clustering_response_path": raw_path,
                    "metrics": {
                        **metrics.compute_all([]),
                    },
                }
                n_failed += 1
            else:
                n_approaches = len(set(labels)) if labels else 0
                record = {
                    **base,
                    "status": "ok",
                    "n_correct": st.n_correct,
                    "n_approaches": n_approaches,
                    "approach_labels": list(labels),
                    "clustering_response_path": raw_path,
                    "metrics": {
                        **metrics.compute_all(labels),
                    },
                }
                n_ok += 1

        append_record(results_file, record)

    return n_ok, n_no_correct, n_failed


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Approach coverage evaluation entry point.")
    p.add_argument("--model-id", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--training-method", required=True)
    p.add_argument("--eval-set", required=True)
    p.add_argument("--eval-set-version", required=True)
    p.add_argument(
        "--generations-file",
        default=None,
        help=(
            "Path to a div-bench generations jsonl. If omitted, the path "
            "is looked up in `eval_sets[<eval-set>]` from the config."
        ),
    )
    p.add_argument("--config", default="configs/eval_config.yaml")
    p.add_argument("--results-file", default="results/results.jsonl")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of problems (debug runs).",
    )
    p.add_argument(
        "--phase",
        choices=["all", "verify", "cluster"],
        default="all",
        help=(
            "Pipeline phase to run. 'verify' runs Phase 1 only and saves "
            "scores. 'cluster' loads saved scores and runs Phases 2-4. "
            "'all' (default) runs the full pipeline."
        ),
    )
    p.add_argument(
        "--verification-file",
        default=None,
        help=(
            "Path to write (verify phase) or read (cluster phase) "
            "verification JSONL. Required for verify/cluster phases."
        ),
    )
    p.add_argument(
        "--problem-ids-file",
        default=None,
        help=(
            "Text file with one problem_id per line. Only these problems "
            "are sent to clustering; others are skipped."
        ),
    )
    p.add_argument(
        "--resume-batch-id",
        default=None,
        help=(
            "OpenAI Batch API job ID to resume (e.g. batch_abc123). "
            "Polls the existing batch instead of submitting a new one."
        ),
    )
    p.add_argument(
        "--resume-realtime-file",
        default=None,
        help=(
            "Path to an intermediate realtime clustering results json "
            "(clustering_responses/realtime_results_*.json). Problems "
            "already present are skipped; remaining ones are processed."
        ),
    )
    return p.parse_args(argv)


def setup_logging(model_id: str, checkpoint: str, eval_set: str) -> logging.Logger:
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    safe_tag = f"{ts}_{model_id}_{checkpoint}_{eval_set}".replace("/", "_")
    log_path = logs_dir / f"{safe_tag}.log"

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.handlers = [fh, sh]

    for noisy in ("httpx", "httpcore", "openai", "openai._base_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    failures_logger = logging.getLogger("evaluate.failures")
    fail_fh = logging.FileHandler(logs_dir / f"{safe_tag}_failures.log")
    fail_fh.setFormatter(fmt)
    failures_logger.handlers = [fail_fh]
    failures_logger.setLevel(logging.WARNING)
    failures_logger.propagate = False
    return failures_logger


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    cfg = EvalConfig.from_yaml(args.config)
    failures_logger = setup_logging(args.model_id, args.checkpoint, args.eval_set)

    gen_path = args.generations_file or cfg.eval_sets.get(args.eval_set)
    if not gen_path:
        raise SystemExit(
            f"No generations file for eval_set={args.eval_set!r}; "
            "pass --generations-file or add it to config.eval_sets"
        )
    if not os.path.exists(gen_path):
        raise SystemExit(f"Generations file not found: {gen_path}")

    logging.info("loading generations from %s", gen_path)
    records = load_generations(gen_path)
    if args.limit is not None:
        records = records[: args.limit]
    logging.info("loaded %d problems", len(records))

    # Build per-problem state objects.
    states: List[ProblemState] = []
    for rec in records:
        n_sampled = max(
            len(rec.get("solutions", [])),
            len(rec.get("pred", [])),
            cfg.n_sampled,
        )
        states.append(
            ProblemState(
                problem_id=_problem_id(rec),
                record=rec,
                n_sampled=n_sampled,
            )
        )

    common = {
        "model_id": args.model_id,
        "checkpoint": args.checkpoint,
        "training_method": args.training_method,
        "eval_set": args.eval_set,
        "eval_set_version": args.eval_set_version,
    }
    run_tag = f"{args.model_id}_{args.checkpoint}_{args.eval_set}".replace("/", "_")

    phase = args.phase

    # -- Phase 1: Verify --
    if phase in ("all", "verify"):
        logging.info("phase 1: verifying %d problems", len(states))
        phase1_verify_all(states, cfg.verifier, failures_logger)
        if args.verification_file:
            save_verification(states, args.verification_file)
        if phase == "verify":
            n_correct_total = sum(st.n_correct for st in states)
            logging.info(
                "verify-only done: %d problems, %d total correct",
                len(states),
                n_correct_total,
            )
            return 0

    # -- Load pre-computed verification for cluster-only mode --
    if phase == "cluster":
        if not args.verification_file:
            raise SystemExit("--verification-file is required for --phase=cluster")
        load_verification(states, args.verification_file)

    # -- Apply problem-ID filter if given --
    if args.problem_ids_file:
        allowed = set(Path(args.problem_ids_file).read_text().strip().splitlines())
        filtered_states = [s for s in states if s.problem_id in allowed]
        excluded_states = [s for s in states if s.problem_id not in allowed]
        logging.info(
            "problem filter: %d/%d pass, %d excluded",
            len(filtered_states),
            len(states),
            len(excluded_states),
        )
    else:
        filtered_states = states
        excluded_states = []

    # Phase 2
    jobs = phase2_build_jobs(filtered_states)
    logging.info(
        "phase 2: %d/%d problems have correct solutions → cluster jobs",
        len(jobs),
        len(filtered_states),
    )

    # Phase 3
    try:
        cluster_result = phase3_cluster(
            jobs,
            cfg.clustering,
            run_tag=run_tag,
            resume_batch_id=args.resume_batch_id,
            resume_realtime_file=args.resume_realtime_file,
        )
    except Exception as exc:  # noqa: BLE001
        failures_logger.exception("phase 3 (batch cluster) raised: %s", exc)
        cluster_result = BatchClusterResult(
            labels_by_problem={},
            raw_response_paths={},
            batch_metadata={"error": str(exc)},
        )
    logging.info("phase 3: batch metadata = %s", cluster_result.batch_metadata)

    # Phase 4
    n_ok, n_no_correct, n_failed = phase4_finalize(
        filtered_states, cluster_result, common, cfg, args.results_file
    )

    logging.info(
        "done: ok=%d  no_correct=%d  clustering_failed=%d  total=%d",
        n_ok,
        n_no_correct,
        n_failed,
        len(filtered_states),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
