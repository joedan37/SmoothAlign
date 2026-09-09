#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATASET_DIR="${IEMOCAP_DATASET_DIR:-${SCRIPT_DIR}/data}"
PRETRAIN_DIR="${GRAM_PRETRAIN_DIR:-${SCRIPT_DIR}/../weights/gram_pretrained}"
CACHE_DIR="${IEMOCAP_CACHE_DIR:-${SCRIPT_DIR}/.cache/control_v3}"
OUTPUT_DIR="${IEMOCAP_OUTPUT_DIR:-${SCRIPT_DIR}/outputs/control_v3}"

if [[ ! -f "${CACHE_DIR}/train.pt" || ! -f "${CACHE_DIR}/test.pt" ]]; then
  printf 'Missing feature caches. Run iemocap_train_portable.sh or set IEMOCAP_CACHE_DIR.\n' >&2
  exit 2
fi
if [[ ! -f "${CACHE_DIR}/class_text_anchors.pt" ]]; then
  printf 'Missing class-anchor cache. Run iemocap_train_portable.sh first.\n' >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}/comparison"
"${PYTHON_BIN}" "${SCRIPT_DIR}/visualize_iemocap_comparison_v2.py" \
  --train-cache "${CACHE_DIR}/train.pt" \
  --test-cache "${CACHE_DIR}/test.pt" \
  --class-anchor-cache "${CACHE_DIR}/class_text_anchors.pt" \
  --pretrain-dir "${PRETRAIN_DIR}" \
  --v1-checkpoint "${OUTPUT_DIR}/v1/checkpoints/last.pt" \
  --v5-checkpoint "${OUTPUT_DIR}/v5/checkpoints/last.pt" \
  --output-dir "${OUTPUT_DIR}/comparison" \
  --class-count 3 --max-per-class 40 --perplexity 25 \
  --seeds 7 21 42 84 168 --primary-seed 42 --sample-seed 2025 \
  --gaussian-tau 0.5 --confidence-floor 0.35 --no-confidence-random-baseline

printf 'IEMOCAP controlled diagnostic visualization completed.\n'
