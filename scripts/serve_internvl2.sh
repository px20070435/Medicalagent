#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-${PROJECT_DIR}/data/models/InternVL2-8B}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
exec "${LMDEPLOY_BIN:-lmdeploy}" serve api_server "${MODEL_PATH}" \
  --model-name InternVL2-8B --backend turbomind --dtype bfloat16 --tp 1 \
  --server-name 127.0.0.1 --server-port "${PORT:-23333}" \
  --session-len "${SESSION_LEN:-8192}" --max-batch-size 1 \
  --vision-max-batch-size 1 --cache-max-entry-count 0.2 --log-level WARNING
