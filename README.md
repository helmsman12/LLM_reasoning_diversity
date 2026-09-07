# Are We Measuring Strategy or Phrasing? The Gap Between Surface- and Approach-Level Diversity in LLM Math Reasoning

Official code and data of the paper [Are We Measuring Strategy or Phrasing? The Gap Between Surface- and Approach-Level Diversity in LLM Math Reasoning](https://arxiv.org/abs/2606.29985).

### News

[2026.06.28] Our paper has been accepted at the ICML 2026 AI4Math Workshop as a **Spotlight** paper!!

[2026.08.21] Our paper will be presented at **EMNLP 2026** as a main conference paper!!

## Overview

![main_figure](./img/figure.png)

What do we mean by 'diversity' in LLM math reasoning? Motivated by recent findings on mode collapse of RLVR and the effectiveness of diversity during test-time scaling, a number of works are focusing on diversity-aware training algorithms.
However, we find that these methods typically operationalize diversity by surface-level diversity—such as lexical overlap, embedding distance, or symbolic representations like the ratio of unique equations.
This trend leaves open a more fundamental question - *are models producing surface-level variants of the same strategy, or exploring genuinely different ways to solve the problem?*

To answer this question, we introduce **approach-level diversity**: variation in the underlying solution strategies used to arrive at the correct answer, beyond differences in wording, notation, or exposition.
Our analysis reveals that conventional, widely used diversity metrics are poor proxies for approach-level diversity, and optimizing such measures does not improve approach-level diversity: rather, policies tend to generate surface-diverse solutions within a narrower set of approaches. Refer to [our paper](https://arxiv.org/abs/2606.29985) if you are interested in our findings!

This repository contains the code and data used in our experiments. It contains three main components:
1. **Multiple-approach feasible problem set**: ~2500 multi-approach feasible math problems filtered from the MATH training dataset.
2. **Problem filtering pipeline**: Four-stage filtering pipeline for collecting multiple-approach feasible problems.
3. **Coverage evaluation script**: Python script for approach-coverage analysis of generated solutions.

## Dataset

The multi-approach feasible problem set (`data/`) contains 2,467 problems filtered from the MATH training set by the pipeline in `filtering/`.

| File | Problems | Description |
|---|---|---|
| `data/train.jsonl` | 2,000 | Training split, used for the RLVR / SFT experiments in the paper |
| `data/eval.jsonl` | 467 | Held-out evaluation split, used for coverage evaluation |

Each line is one problem together with its feasible approach plans and the uniqueness-judge verdict:

```json
{
  "problem": "Points $A,B,C,D,E$ and $F$ lie, in that order, on ...",
  "answer": "\\frac{5}{3}",
  "response": {
    "problem_brief": "...",
    "approaches": [
      {"name": "...", "core_idea": "...", "plan": ["step 1", "step 2", "..."]},
      "..."
    ]
  },
  "num_unique_approaches": 4,
  "judge_result": "EXPLANATION: ..."
}
```

Only `problem` and `answer` are needed for coverage evaluation.

## Usage

`pip install -r requirements.txt`, then put your OpenAI key in `.env` (`cp .env.example .env`). Correctness scoring and the feasibility check use a locally served Qwen3-4B (`vllm serve Qwen/Qwen3-4B --port 9000`).

### 1. Coverage@N

**Coverage@N** is the expected number of distinct solution approaches among N solutions sampled from a model's correct solutions. Given the approach label of each correct solution (from the clustering judge below), `coverage/metrics.py` computes it in closed form:

```python
from coverage.metrics import coverage_curve

labels = [0, 0, 1, 2, 1, 3, 0, 2]          # approach id of each correct solution (problem-local)
coverage_curve(labels)                      # {1: 1.0, 2: 1.82, 4: 3.0, 8: 4.0, 16: nan, 32: nan, 64: nan}
```

Coverage@N is `nan` whenever N exceeds the number of correct solutions, so it is never inflated for weak models. Dataset-level numbers are the mean Coverage@N over problems and the area under that curve (`coverage/scripts/aggregate_results.py`).

`coverage/evaluate.py` runs the whole procedure for one checkpoint (verify sampled solutions → cluster with the judge → Coverage@N per problem); see `coverage/configs/eval_config.yaml` for the settings used in the paper.

### 2. Clustering solutions by approach

The LLM judge assigns approach labels to correct solutions. Input is one problem per line with `problem`, `solutions` and `scores` (1 = correct, 0 = incorrect). If your generations are not scored yet:

```bash
python evaluate_solutions/verify_solutions.py generations.jsonl \
    --model-name Qwen/Qwen3-4B --api-base http://localhost:9000/v1     # writes generations_scored.jsonl
```

Then cluster the correct solutions:

```bash
cd evaluate_solutions

# OpenAI Batch API (the setting used in the paper)
python sol_diversity_judge.py generations_scored.jsonl \
    --model gpt-5.2 --mode batch --chunk-size 8 --min-correct 4 \
    --output-dir outputs/clustering_results

# Or a locally served open model, realtime
python sol_diversity_judge.py generations_scored.jsonl \
    --api-base http://localhost:8000/v1 --model Qwen/Qwen3-30B-A3B \
    --mode realtime --num-workers 12 --chunk-size 8
```

Solutions are clustered in chunks of `--chunk-size` and the chunk results are merged in a second call. The output JSON holds, per problem, the approach groups (`group_name`, `core_idea`, `solution_ids`). `scripts/cluster_generations.sh` is the exact command used in the paper.

### 3. Building a multi-approach problem set

Four stages select problems that admit several genuinely different, feasible solution approaches. Each stage reads the previous stage's output.

| Stage | Command | Model | Keeps |
|---|---|---|---|
| 1. Difficulty filter | `filtering/filter_by_avg.py` | pass@1 of a reference model | problems of medium difficulty |
| 2. Approach generation | `filtering/generate_approaches_batch.py` | OpenAI Batch API | K candidate plans per problem |
| 3. Feasibility check | `filtering/check_feasible_plans.py` | Qwen3-4B solver + judge (vLLM) | plans that Qwen3-4B can execute to a correct answer; problems with ≥ 3 such plans |
| 4. Uniqueness judge | `filtering/uniqueness_judge.py` | OpenAI (`gpt-5.1`) | problems whose feasible plans contain ≥ 3 mechanism-level distinct approaches |

```bash
# Stage 1: split problems by pass@1 (edit input_files inside the script), keep medium ones
python filtering/filter_by_avg.py --output-path outputs/stage1/

# Stage 2: generate K=4 approach plans per problem -> <input>_with_plans.jsonl
python filtering/generate_approaches_batch.py --input_file outputs/stage1/medium.jsonl --k 4 --model gpt-5.1

# Stage 3: solve each plan with Qwen3-4B, verify, keep feasible plans -> outputs/stage3/feasible_plans/
python filtering/check_feasible_plans.py \
    --input_file outputs/stage1/medium_with_plans.jsonl \
    --output_file outputs/stage3/feasibility.jsonl \
    --base_url http://localhost:9000/v1 --verifier-base-url http://localhost:9000/v1 \
    --n_rollouts 8

# Stage 4: judge approach uniqueness -> outputs/stage4/uniqueness_positive.json
python filtering/uniqueness_judge.py \
    --input-file outputs/stage3/feasible_plans/medium_with_plans.jsonl \
    --output-file outputs/stage4/uniqueness.json \
    --model gpt-5.1 --reasoning-effort low
```

Useful options for stage 4: `--realtime` (synchronous calls instead of the Batch API), `--num-votes 3` (majority vote), `--batch-id` / `--raw-results-file` (resume without re-querying), `--eval` (score the judge against a `label` field of `positive` / `negative`).

## Repository Structure

```
coverage-and-filtering/
├── data/                            # Multi-approach feasible problem set (train / eval)
├── coverage/                        # Coverage@N evaluation
│   ├── evaluate.py                  # Entry point: verify -> cluster -> Coverage@N
│   ├── metrics.py                   # Coverage@N estimator
│   ├── configs/eval_config.yaml     # Verifier / judge / eval-set settings
│   └── scripts/                     # Aggregation (mean Cov@N, AUC) and problem filtering
├── evaluate_solutions/              # LLM judge for approach clustering
│   ├── sol_diversity_judge.py       # Clustering judge (batch / realtime)
│   ├── verify_solutions.py          # Correctness scoring with Qwen3-4B
│   ├── utils.py                     # Judge prompt
│   └── scripts/                     # Launch scripts used in the paper
├── filtering/                       # Four-stage problem filtering pipeline
│   ├── filter_by_avg.py             # Stage 1
│   ├── generate_approaches_batch.py # Stage 2
│   ├── check_feasible_plans.py      # Stage 3
│   └── uniqueness_judge.py          # Stage 4 (prompt in uniqueness_judge_utils.py)
├── .env.example                     # OPENAI_API_KEY template
└── requirements.txt
```

## Citation

```bibtex
@misc{lee2026measuringstrategyphrasinggap,
      title={Are We Measuring Strategy or Phrasing? The Gap Between Surface- and Approach-Level Diversity in LLM Math Reasoning}, 
      author={Sangmook Lee and Minbeom Kim and Jeonghye Kim and Dohyung Kim and Sojeong Rhee and Kyomin Jung},
      year={2026},
      eprint={2606.29985},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2606.29985}, 
}
```
