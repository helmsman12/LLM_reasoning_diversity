# Coverage & Multi-Approach Problem Filtering

Self-contained package for multi-approach problem filtering (approach generation → Qwen3-4B feasibility check → uniqueness judge), LLM-judge-based solution clustering, and Coverage@N measurement.

## Directory Structure

```
coverage-and-filtering/
├── README.md
├── requirements.txt
├── .env.example                     # Copy to .env and set OPENAI_API_KEY (single place for all scripts)
├── .gitignore
├── coverage/                        # Approach coverage analysis pipeline
│   ├── CLAUDE.md                    # Metric definitions and pipeline documentation
│   ├── metrics.py                   # Coverage@N estimator (pure functions)
│   ├── evaluate.py                  # 4-phase pipeline entry point (verify -> cluster -> finalize)
│   ├── cluster.py                   # LLM judge clustering wrapper (OpenAI Batch API)
│   ├── feasibility_report.py        # Pre/post checkpoint feasibility comparison report
│   ├── configs/
│   │   └── eval_config.yaml         # Sampling / verifier / clustering configuration
│   └── scripts/
│       ├── aggregate_results.py     # Aggregate all result JSONL files and compute AUC
│       ├── aggregate_filtered.py    # Group-wise aggregation on common problem sets
│       └── filter_by_ncorrect.py    # Filter problems by min n_correct across checkpoints
├── filtering/                       # Multi-approach problem filtering pipeline
│   ├── filter_by_avg.py             # Stage 1: difficulty classification (pass@1) + approach count filter
│   ├── generate_approaches_batch.py # Stage 2: generate K distinct approach plans per problem (OpenAI Batch API)
│   ├── check_feasible_plans.py      # Stage 3: plan feasibility via Qwen3-4B solver + Qwen3-4B LLM judge (vLLM)
│   ├── uniqueness_judge.py          # Stage 4: LLM uniqueness judge (>= 3 distinct approaches -> keep)
│   └── uniqueness_judge_utils.py    # Uniqueness judge prompt, \boxed{n} parsing, cost helpers
└── evaluate_solutions/              # LLM judge solution clustering
    ├── verify_solutions.py          # Qwen3-4B LLM-judge correctness scoring -> adds `scores` field
    ├── sol_diversity_judge.py       # BatchProcessor, RealtimeProcessor, core clustering
    ├── utils.py                     # Clustering judge system prompt (conservative merging policy)
    └── scripts/
        ├── cluster_generations.sh   # OpenAI Batch API clustering launch script
        ├── llm_judge_eval.sh        # OpenAI API judge evaluation script (batch mode)
        └── llm_judge_eval_vllm.sh   # vLLM local serving judge evaluation script
```

## Key Components

### 1. Coverage Metrics (`coverage/metrics.py`)

Pure function module (no I/O). Computes per-problem Coverage@N from approach label lists:
- **Coverage@N**: Expected number of distinct approaches in N sampled solutions (analytic unbiased estimator, sampling without replacement from the correct solutions)
- **Coverage Curve**: Coverage@N at N in {1, 2, 4, 8, 16, 32, 64}; NaN whenever N > n_correct
- **AUC** (dataset-level, computed by `scripts/aggregate_results.py`): sum over N of the mean Coverage@N across problems

### 2. Coverage Pipeline (`coverage/evaluate.py`)

4-phase pipeline for each `(model_checkpoint, eval_set)` pair:
1. **Verify**: Check all solutions for correctness via vLLM-served verifier
2. **Build Jobs**: Collect clustering jobs for problems with correct solutions
3. **Cluster**: Run LLM judge clustering via OpenAI Batch API
4. **Finalize**: Compute Coverage@N and append to results.jsonl

### 3. Multi-Approach Problem Filtering (`filtering/`)

Four stages select problems that admit several genuinely different, feasible solution approaches.

| Stage | Script | Model | Input → Output |
|---|---|---|---|
| 1. Difficulty filter | `filter_by_avg.py` | (pre-computed pass@1) | scored solutions → easy / medium / hard split; `filter_by_analysis()` keeps problems with > 2 approaches |
| 2. Approach generation | `generate_approaches_batch.py` | OpenAI Batch API | `{problem, answer}` → adds `response.approaches` (K plans: `name`, `core_idea`, `plan[]`) |
| 3. Feasibility check | `check_feasible_plans.py` | Qwen3-4B solver + Qwen3-4B judge (vLLM) | plans → `feasible_plans/<name>.jsonl` (only approaches with ≥ 1 verified rollout; problems with ≥ 3 feasible approaches) |
| 4. Uniqueness judge | `uniqueness_judge.py` | OpenAI (default `gpt-5.1`) | feasible plans → `<out>.json` + `<out>_positive.json` (problems judged to have ≥ 3 distinct approaches) |

**Stage 3 (Qwen3-4B LLM evaluator).** For each (problem, plan) pair the solver is prompted to follow that plan only. Rollouts are progressive (1, 2, 4, … up to `--n_rollouts`) and stop as soon as one rollout is verified. Verification first tries `math-verify` on the `\boxed{}` answer and falls back to the Qwen3-4B judge (few-shot "correct"/"incorrect" prompt, thinking disabled). Requires a running vLLM server:

```bash
vllm serve Qwen/Qwen3-4B --port 9000 --dtype bfloat16 --max-model-len 12288

python filtering/check_feasible_plans.py \
    --input_file plans_with_approaches.jsonl \
    --output_file outputs/stage3_feasibility.jsonl \
    --base_url http://localhost:9000/v1 --verifier-base-url http://localhost:9000/v1 \
    --model_name Qwen/Qwen3-4B --verifier-model-name Qwen/Qwen3-4B \
    --n_rollouts 8 --solver-batch-size 8 --n-solver-workers 16 --n-verifier-workers 4
```

**Stage 4 (uniqueness judge).** The judge sees the problem and its plans and answers with `\boxed{n}`, the number of mechanism-level distinct approaches (prompt in `uniqueness_judge_utils.py`). A problem is *positive* when `n >= 3`.

```bash
python filtering/uniqueness_judge.py \
    --input-file outputs/feasible_plans/plans_with_approaches.jsonl \
    --output-file outputs/stage4_uniqueness.json \
    --model gpt-5.1 --reasoning-effort low          # Batch API; add --realtime for sync calls
# options: --num-votes 3 (majority vote), --batch-id / --raw-results-file (resume),
#          --eval (needs a "label" field: positive/negative), --eval-only
```

### 4. Solution Clustering Judge (`evaluate_solutions/`)

LLM judge pipeline that clusters given solutions by approach.

**Correctness scoring: `verify_solutions.py`** (Qwen3-4B LLM judge)
Adds a `scores` field to a generations JSONL (`question`, `answer`, `solutions[]`, optional `pred[]`) by asking a vLLM-served judge whether each solution's final answer is equivalent to the golden answer (few-shot prompt, `Verification: [correct]` / `[incorrect]`).
```bash
python evaluate_solutions/verify_solutions.py generations.jsonl \
    --model-name Qwen/Qwen3-4B --api-base http://localhost:9000/v1   # writes generations_scored.jsonl
```

**Core file: `sol_diversity_judge.py`**
- `ClusteringConfig`: Full configuration (model, API settings, chunk size, min_correct, etc.)
- `CostTracker`: API cost tracking (automatic 50% batch discount)
- `GPTClient`: OpenAI API calls (realtime + batch mode)
- `BatchProcessor`: Large-scale clustering via OpenAI Batch API
  - Stage 1: Split solutions into chunks and cluster each
  - Stage 2: Merge Stage 1 results into final grouping
- `RealtimeProcessor`: Realtime API clustering (multi-worker)
- `grouping_to_labels()`: Convert JSON grouping to per-solution label array

**Prompt (`utils.py`):**
- `CONSERVATIVE_JSON_SYSTEM_PROMPT` (exported as `ACTIVE_SYSTEM_PROMPT`): the approach-clustering judge prompt with a conservative merging policy. Used for both the per-chunk Stage 1 calls and the Stage 2 merge call.

**Usage:**
```bash
cd evaluate_solutions

# Cluster via OpenAI Batch API (most common; key is read from ../.env)
python sol_diversity_judge.py input.jsonl \
    --model gpt-5.2 \
    --mode batch \
    --chunk-size 8 \
    --min-correct 4 \
    --output-dir outputs/clustering_results

# Cluster via local vLLM model (cost-effective)
python sol_diversity_judge.py input.jsonl \
    --api-base http://localhost:8000/v1 \
    --api-key EMPTY \
    --model Qwen/Qwen3-30B-A3B \
    --mode realtime \
    --num-workers 12 \
    --chunk-size 10
```

**Input JSONL format** (one line = one problem):
```json
{
    "problem": "problem text",
    "solutions": ["solution1", "solution2", ...],
    "scores": [1, 0, 1, ...]
}
```
- `scores`: Correctness of each solution (1=correct, 0=incorrect). If absent, all solutions are used.
- Problems with fewer correct solutions than `min-correct` are skipped.

**Output**: Per-problem approach group JSON (group_name, core_idea, solution_ids)

## Usage

### Setup
```bash
pip install -r requirements.txt

# Put your OpenAI API key in the package-level .env (used by every script)
cp .env.example .env
# then edit .env:  export OPENAI_API_KEY="your-key-here"
```

All Python entry points (`coverage/evaluate.py`, `coverage/cluster.py`, `filtering/generate_approaches_batch.py`, `evaluate_solutions/sol_diversity_judge.py`) load `.env` automatically via `python-dotenv`, and the shell scripts under `evaluate_solutions/scripts/` `source` it. A key already exported in your shell takes precedence over `.env`. `.env` is git-ignored; only `.env.example` is committed.

### Run Coverage Evaluation
```bash
cd coverage
python evaluate.py \
    --model-id Qwen2.5-3B \
    --checkpoint step-500 \
    --training-method GRPO \
    --eval-set eval-file \
    --eval-set-version v1.0 \
    --generations-file path/to/generations.jsonl \
    --config configs/eval_config.yaml
```

### Multi-Approach Filtering
```bash
# Stage 1: difficulty split / > 2 approaches (see section 3 for stages 2-4)
python filtering/filter_by_avg.py --output-path output/

# Filter by min n_correct >= 16 across all checkpoints
python coverage/scripts/filter_by_ncorrect.py \
    --verification-dir data/verification/ \
    --min-correct 16 \
    --output data/filtered_problem_ids.txt
```

### Aggregate Results
```bash
cd coverage
python scripts/aggregate_results.py --runs-dir results/runs
python scripts/aggregate_filtered.py --eval-set eval-file-qwen25-3b --max-n 32
```

## Dependencies

Install with `pip install -r requirements.txt`.

- Python 3.8+
- `openai` (OpenAI API client)
- `python-dotenv` (loads `.env`; optional — `source .env` works without it)
- `pyyaml`
- `numpy`
- `scikit-learn` (ARI, homogeneity, completeness for judge evaluation)
- `tqdm`
- `transformers` (chat template for the Qwen3-4B judge prompts)
- `math-verify` (symbolic answer check before the LLM judge in `check_feasible_plans.py`)
- `vllm` (not in requirements.txt; needed only to serve the Qwen3-4B solver/judge locally)
