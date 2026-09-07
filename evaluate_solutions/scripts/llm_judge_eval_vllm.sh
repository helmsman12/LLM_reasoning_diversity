#!/bin/bash
set -euo pipefail

# Sequential vLLM judge evaluation on a single 2-GPU node.
# Each judge server uses tensor parallelism across both GPUs, then shuts down
# before the next model starts.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_DIR"

VLLM_ENV="${VLLM_ENV:-vllm}"
VLLM_ENV_DIR="${VLLM_ENV_DIR:-$HOME/.conda/envs/${VLLM_ENV}}"
VLLM_BIN="${VLLM_BIN:-${VLLM_ENV_DIR}/bin/vllm}"
PYTHON_BIN="${PYTHON_BIN:-${VLLM_ENV_DIR}/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
TP_SIZE="${TP_SIZE:-2}"
VLLM_PORT="${VLLM_PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
API_BASE="http://${HOST}:${VLLM_PORT}/v1"
JUDGE_FILTER="${JUDGE_FILTER:-}"
BUDGET_FILTER="${BUDGET_FILTER:-}"
LIMIT="${LIMIT:-}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/clustering_tests_vllm}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/server_logs}"
DEFAULT_EVAL_FILE="${PROJECT_DIR}/judge_eval_files/test_3b_eval_file_filtered.jsonl"
EVAL_FILE="${EVAL_FILE:-${DEFAULT_EVAL_FILE}}"
EVAL_TAG="${EVAL_TAG:-test_3b_human_annotated}"

RATE_LIMIT="${RATE_LIMIT:-256}"
NUM_WORKERS="${NUM_WORKERS:-12}"
CHUNK_SIZE="${CHUNK_SIZE:-10}"
MAX_RETRIES="${MAX_RETRIES:-8}"

# Throughput-oriented vLLM settings. Override from the shell if needed.
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.95}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-1200}"
REASONING_CONFIG="${REASONING_CONFIG:-{\"reasoning_start_str\":\"<think>\",\"reasoning_end_str\":\"</think>\"}}"
VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
VLLM_USE_V1="${VLLM_USE_V1:-1}"
TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

declare -A EVAL_FILES=(
    ["${EVAL_FILE}"]="${EVAL_TAG}"
)

JUDGE_MODELS=(
    "Qwen/Qwen3-30B-A3B|qwen3_30b_a3b"
    "Qwen/Qwen3.5-35B-A3B|qwen35_35b_a3b"
)

THINKING_TOKEN_BUDGETS=(0 2048 4096)

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

if [[ ! -x "$VLLM_BIN" ]]; then
    echo "vLLM executable not found: $VLLM_BIN" >&2
    exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python executable not found: $PYTHON_BIN" >&2
    exit 1
fi

SERVER_PID=""

cleanup_server() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
    SERVER_PID=""
}

trap cleanup_server EXIT

wait_for_server() {
    local timeout_s="$1"
    local start_ts
    start_ts=$(date +%s)

    while true; do
        if curl -fsS "http://${HOST}:${VLLM_PORT}/health" >/dev/null 2>&1; then
            return 0
        fi

        if [[ -n "${SERVER_PID}" ]] && ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            return 1
        fi

        if (( "$(date +%s)" - start_ts > timeout_s )); then
            return 1
        fi

        sleep 5
    done
}

start_server() {
    local judge_model="$1"
    local judge_tag="$2"
    local log_file="${LOG_DIR}/${judge_tag}.log"

    echo "=========================================="
    echo "Starting vLLM judge server"
    echo "  Model: $judge_model"
    echo "  Env: $VLLM_ENV"
    echo "  vLLM Bin: $VLLM_BIN"
    echo "  Python Bin: $PYTHON_BIN"
    echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    echo "  Tensor Parallel Size: $TP_SIZE"
    echo "  API Base: $API_BASE"
    echo "  Reasoning Config: $REASONING_CONFIG"
    echo "  Log: $log_file"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    VLLM_ATTENTION_BACKEND="$VLLM_ATTENTION_BACKEND" \
    VLLM_USE_V1="$VLLM_USE_V1" \
    TOKENIZERS_PARALLELISM="$TOKENIZERS_PARALLELISM" \
    OMP_NUM_THREADS=1 \
    "$VLLM_BIN" serve "$judge_model" \
        --host "$HOST" \
        --port "$VLLM_PORT" \
        --api-key EMPTY \
        --served-model-name "$judge_model" \
        --reasoning-parser qwen3 \
        --reasoning-config "$REASONING_CONFIG" \
        --tensor-parallel-size "$TP_SIZE" \
        --dtype bfloat16 \
        --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
        --enable-prefix-caching \
        --no-enable-log-requests \
        --uvicorn-log-level warning \
        >"$log_file" 2>&1 &

    SERVER_PID=$!

    if ! wait_for_server "$SERVER_TIMEOUT"; then
        echo "vLLM server failed to become healthy for $judge_model" >&2
        echo "Last log lines:" >&2
        tail -n 80 "$log_file" >&2 || true
        return 1
    fi
}

run_eval() {
    local judge_model="$1"
    local judge_tag="$2"

    for eval_file in "${!EVAL_FILES[@]}"; do
        local solution_model="${EVAL_FILES[$eval_file]}"

        for budget in "${THINKING_TOKEN_BUDGETS[@]}"; do
            if [[ -n "$BUDGET_FILTER" ]] && [[ "$budget" != "$BUDGET_FILTER" ]]; then
                continue
            fi

            local prefix="${judge_tag}_${solution_model}_budget_${budget}"
            local limit_args=()
            if [[ -n "$LIMIT" ]]; then
                limit_args=(--limit "$LIMIT")
            fi

            echo "=========================================="
            echo "Running evaluation"
            echo "  File: $eval_file"
            echo "  Solution Model: $solution_model"
            echo "  Judge Model: $judge_model"
            echo "  Judge URL: $API_BASE"
            echo "  Thinking Budget: $budget"
            echo "  Num Workers: $NUM_WORKERS"
            echo "  Rate Limit: $RATE_LIMIT"
            echo "  Chunk Size: $CHUNK_SIZE"
            if [[ -n "$LIMIT" ]]; then
                echo "  Limit: $LIMIT"
            fi
            echo "=========================================="

            OPENAI_API_KEY=EMPTY \
            OPENAI_BASE_URL="$API_BASE" \
            "$PYTHON_BIN" -u sol_diversity_judge.py "$eval_file" \
                --api-base "$API_BASE" \
                --api-key EMPTY \
                --model "$judge_model" \
                --mode realtime \
                --reasoning-effort none \
                --thinking-token-budget "$budget" \
                --output-dir "$OUTPUT_DIR" \
                --output-prefix "$prefix" \
                --num-workers "$NUM_WORKERS" \
                --rate-limit "$RATE_LIMIT" \
                --max-retries "$MAX_RETRIES" \
                --chunk-size "$CHUNK_SIZE" \
                --eval \
                "${limit_args[@]}"

            echo ""
            echo "Completed: $solution_model with judge $judge_model at thinking budget $budget"
            echo ""
        done
    done
}

for judge_spec in "${JUDGE_MODELS[@]}"; do
    IFS='|' read -r judge_model judge_tag <<< "$judge_spec"
    if [[ -n "$JUDGE_FILTER" ]] && [[ "$judge_model" != "$JUDGE_FILTER" ]] && [[ "$judge_tag" != "$JUDGE_FILTER" ]]; then
        continue
    fi
    start_server "$judge_model" "$judge_tag"
    run_eval "$judge_model" "$judge_tag"
    cleanup_server
done

echo "All vLLM judge evaluations completed!"
