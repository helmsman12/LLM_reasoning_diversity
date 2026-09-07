# Approach Coverage Analysis
 
## Project Goal
 
Analyze **approach coverage rate** across different training methods and RLVR algorithms (e.g., GRPO, SFT, DAPO variants).
 
**Approach coverage** measures how well a model samples *distinct solution approaches* — not just lexical or token-level diversity — over a set of correct solutions.
 
---
 
## Definitions
 
- **Approach**: A qualitatively distinct method for solving a problem, as judged by an LLM judge or embedding-based clustering.
- **Correct solution**: A generated solution whose final answer matches the ground-truth answer.
- **Approach group**: A cluster of correct solutions sharing the same approach, as determined by the clustering step.
 
---
 
## Pipeline Overview
 
For each `(model_checkpoint, eval_set)` pair:
 
1. **Sample**: Generate `N=64` solutions per problem using the target model checkpoint.
2. **Filter**: Evaluate each solution for correctness (answer match against ground truth).
3. **Cluster**: Query an LLM judge to partition correct solutions into distinct approach groups.
4. **Measure**: Compute Coverage@N (see `metrics.py`).
5. **Store**: Append results to the unified results file.
 
### Reference Implementation
 
Correctness evaluation and LLM-based approach clustering follow the procedure in:
```
evaluate_solutions/
```
Main execution script:
```
evaluate_solutions/scripts/cluster_generations.sh
```
Always refer to these files for the exact prompting strategy, API call format, and clustering logic. Do not deviate from the established LLM judge protocol without explicit instruction.
 
---
 
## Approach Coverage Metrics
 
All metrics are defined in `metrics.py` as explicit, importable functions. Metrics are computed **over correct solutions only** — i.e., over `S_c ⊆ S`, the subset of generated solutions that match the ground-truth answer. Let `n_correct = |S_c|`.
 
### 1. Coverage@N
Expected number of distinct approaches observed when sampling N solutions from `S_c`:
```
Cov@N = E[|J(S_c^(N))|]
```
where `J(·)` returns the set of distinct approach labels after clustering.
 
**Validity constraint**: Coverage@N is defined only when `N ≤ n_correct`. If `N > n_correct`, record `NaN`. This prevents overestimation of coverage for low-quality models with few correct solutions.
 
**Estimation method**: Do NOT use repeated random subsampling. Instead, use an **analytic unbiased estimator** based on the empirical approach frequency distribution from `S_c`. Given approach frequencies, compute the expected number of distinct approaches in N draws analytically.
 
### 2. Coverage Curve
Coverage@N evaluated at `N ∈ {1, 2, 4, 8, 16, 32, 64}`, with NaN for any N > n_correct:
```
N → Cov@N
```
 
### 3. Area Under Coverage Curve (AUC)
Defined at **dataset level**, not per-problem. Procedure:
1. For each N, compute the mean Coverage@N across all problems where Coverage@N is not NaN → `Cov_dataset(N)`
2. Sum across N values:
```
AUC = Σ_N Cov_dataset(N)
```
No normalization is applied.
 
---
 
## File Structure
 
```
project-root/
├── CLAUDE.md                    ← This file
├── metrics.py                   ← All metric functions (importable)
├── evaluate.py                  ← Main evaluation entry point
├── cluster.py                   ← Wraps LLM judge clustering logic
├── results/
│   └── results.jsonl            ← Unified results file (all checkpoints, models, metrics, eval sets)
├── configs/
│   └── eval_config.yaml         ← Sampling params, model paths, eval set paths
└── scripts/
    └── run_eval.sh              ← Batch evaluation launcher
```
 
---
 
## Results File Format
 
All results are stored in `results/results.jsonl`. Each line is a JSON record:
 
```json
{
  "model_id": "Qwen3-4B-Base",
  "checkpoint": "step-500",
  "training_method": "GRPO",
  "eval_set": "div-bench-indomain",
  "eval_set_version": "v1.0",
  "problem_id": "problem_042",
  "status": "ok",
  "generation_config": {
    "temperature": 0.9,
    "top_p": 0.95,
    "max_tokens": 4096
  },
  "n_sampled": 64,
  "n_correct": 21,
  "n_approaches": 4,
  "approach_labels": [0, 0, 1, 2, 1, 3],
  "clustering_response_path": "/path/to/raw/clustering/response.json",
  "metrics": {
    "cov_at_n": {"1": 1.0, "2": 1.4, "4": 2.1, "8": 2.9, "16": 3.4, "32": 3.8, "64": 4.0}
  },
  "timestamp": "2025-04-08T12:00:00"
}
```
 
**Status field values**:
- `"ok"` — normal record
- `"no_correct_solutions"` — `n_correct = 0`; all metrics are `NaN`
- `"clustering_failed"` — LLM judge clustering did not complete; record is retained with metrics as `NaN`
 
**Edge cases**:
- `n_correct = 0`: all metrics set to `NaN`, `status = "no_correct_solutions"`
- `n_correct = 1`: Coverage@N = 1 for N = 1, NaN for N > 1
- `N > n_correct`: Coverage@N = `NaN` for that N value
 
**Approach labels are problem-local IDs.** The same label value across different problems does not imply the same approach.
 
**AUC is not stored per-record** — it is a dataset-level aggregate computed at analysis time.
 
Never overwrite existing records. Always append. If re-running an existing `(model_id, checkpoint, eval_set, problem_id)` combo, add a new record with updated timestamp.
 
---
 
## Key Conventions
 
- **Default sample size**: N=64 solutions per problem unless specified otherwise.
- **Correctness check**: Exact or normalized answer match — follow the div-bench evaluation protocol exactly.
- **LLM judge**: Use the clustering procedure from `evaluate_solutions/scripts/cluster_generations.sh`. Do not substitute with embedding-only clustering unless explicitly instructed.
- **Coverage@N estimator**: Use the analytic unbiased estimator based on empirical approach frequency distribution. Do NOT use repeated random subsampling.
- **NaN discipline**: Coverage@N must be NaN when N > n_correct. Do not impute, interpolate, or substitute zero.
- **Approach labels are problem-local**: Labels from different problems are not comparable. Never aggregate label IDs across problems.
- **Metrics are per-problem**: All metrics in `metrics.py` operate on a single problem's data. Aggregation (macro average, weighted average, bootstrap CI) happens only in analysis scripts.
- **AUC is dataset-level**: Compute mean Coverage@N across non-NaN problems per N, then sum. Never compute AUC per-problem.
- **`metrics.py` is pure**: No I/O, no model calls. Only takes lists/arrays as input, returns scalars or dicts.
- **Reproducibility fields**: Every record must include `generation_config` (temperature, top_p, max_tokens) and `eval_set_version`. Store raw clustering response path instead of full content.
- **Aggregation default**: Problem-level macro average. Optionally use `n_correct`-weighted average or bootstrap confidence intervals when noted in analysis scripts.