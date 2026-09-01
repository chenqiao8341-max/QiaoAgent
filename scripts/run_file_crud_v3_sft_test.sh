#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="Qwen3.5-4B-File-v3-SFT"
PORT="${PORT:-8002}"
DATASET="$ROOT/.agent_state/rl/datasets/file-crud-rl-v3/test.jsonl"
RUN_ROOT="$ROOT/.agent_state/rl/pipelines/file-crud-v3-production-repair-v1"
EVAL_DIR="$RUN_ROOT/evals"
LOG_DIR="$RUN_ROOT/logs"
AGENT_FILE_RL="$ROOT/.venv/bin/agent-file-rl"

mkdir -p "$EVAL_DIR" "$LOG_DIR"

run_eval() {
  local name="$1"
  shift
  env \
    MODEL_PROVIDER=local-vllm \
    LOCAL_VLLM_BASE_URL="http://127.0.0.1:$PORT/v1" \
    LOCAL_VLLM_MODEL="$MODEL" \
    LOCAL_VLLM_API_KEY=local \
    MODEL_TEMPERATURE=0 \
    "$AGENT_FILE_RL" \
      --dataset "$DATASET" \
      --policy local-model \
      --split test \
      --concurrency 16 \
      --progress-every 25 \
      --report "$EVAL_DIR/$name-report.json" \
      --trajectories "$EVAL_DIR/$name-trajectories.jsonl" \
      "$@" 2>&1 | tee "$LOG_DIR/$name.log"
}

curl -fsS --max-time 3 "http://127.0.0.1:$PORT/health" >/dev/null
curl -fsS "http://127.0.0.1:$PORT/v1/models" | grep -q "$MODEL"

run_eval sft-test-ordinary --safety-cases exclude
run_eval sft-test-safety --operation mixed --safety-cases only
