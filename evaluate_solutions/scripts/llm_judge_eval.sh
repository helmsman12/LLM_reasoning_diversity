#!/bin/bash

# Load API key from the package-level .env (see .env.example)
_PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[ -f "${_PKG_ROOT}/.env" ] && source "${_PKG_ROOT}/.env"


# Configuration
JUDGE_MODEL="gpt-5.2"
OUTPUT_DIR="outputs/clustering_tests"
MODE="batch"

# Resume from completed Stage 1 batch results (set to "" to disable)
RESUME_STAGE1="outputs/clustering_tests/batch_raw_stage1_combined_eval_qwen2.5_7b_solutions_only_scored_20260219_175104.jsonl"

# Eval files with their corresponding solution model names
declare -A EVAL_FILES=(
    # ["judge_eval_files/test_3b_eval_file_filtered.jsonl"]="3b"
    # ["judge_eval_files/test_math_1.5b_eval_file_filtered.jsonl"]="math_1.5b"
    ["outputs/base_model/combined_eval_qwen2.5_7b_solutions_only_scored.jsonl"]="7b"
)

# Reasoning efforts
REASONING_EFFORTS=("none")

# Run evaluations
for eval_file in "${!EVAL_FILES[@]}"; do
    solution_model="${EVAL_FILES[$eval_file]}"

    for effort in "${REASONING_EFFORTS[@]}"; do
        echo "=========================================="
        echo "Running evaluation:"
        echo "  File: $eval_file"
        echo "  Solution Model: $solution_model"
        echo "  Judge Model: $JUDGE_MODEL"
        echo "  Reasoning Effort: $effort"
        if [ -n "$RESUME_STAGE1" ]; then
            echo "  Resume Stage 1: $RESUME_STAGE1"
        fi
        echo "=========================================="

        # Create output prefix with solution model and reasoning effort
        PREFIX="${solution_model}_reasoning_${effort}"

        RESUME_ARGS=""
        if [ -n "$RESUME_STAGE1" ]; then
            RESUME_ARGS="--resume-stage1 $RESUME_STAGE1"
        fi

        python -u sol_diversity_judge.py "$eval_file" \
            --model "$JUDGE_MODEL" \
            --mode "$MODE" \
            --reasoning-effort "$effort" \
            --output-dir "$OUTPUT_DIR" \
            --output-prefix "$PREFIX" \
            --resume-stage2-input-file-id file-X1GsVmaqyEj3xCDVzmHLdo \
            $RESUME_ARGS

        echo ""
        echo "Completed: $solution_model with reasoning effort $effort"
        echo ""
    done
done

echo "All evaluations completed!"
