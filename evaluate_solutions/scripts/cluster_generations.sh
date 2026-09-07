#!/bin/bash

# Load API key from the package-level .env (see .env.example)
_PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[ -f "${_PKG_ROOT}/.env" ] && source "${_PKG_ROOT}/.env"

echo "running evaluation"

eval_file="outputs/base_model/train_solutions_scored.jsonl"

echo "=========================================="
echo "Running evaluation:"
echo "  File: $eval_file"
echo "=========================================="

python -u sol_diversity_judge.py "$eval_file" \
    --model gpt-5.2 \
    --mode batch \
    --reasoning-effort "none" \
    --output-dir "outputs/clustering_results" \
    --output-prefix "train" \
    --chunk-size 8 \
    --min-correct 4 \

echo ""
echo "Completed: $solution_model with reasoning effort $effort"
echo ""
