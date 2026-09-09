#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
GRAM_ROOT="${REPO_ROOT}/GRAM"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Supply licensed data and model files outside this archive, or override these
# defaults with environment variables.  No private machine paths are assumed.
DATASET_DIR="${IEMOCAP_DATASET_DIR:-${SCRIPT_DIR}/data}"
PRETRAIN_DIR="${GRAM_PRETRAIN_DIR:-${REPO_ROOT}/weights/gram_pretrained}"
CACHE_DIR="${IEMOCAP_CACHE_DIR:-${SCRIPT_DIR}/.cache/control_v3}"
OUTPUT_DIR="${IEMOCAP_OUTPUT_DIR:-${SCRIPT_DIR}/outputs/control_v3}"

if [[ ! -d "${DATASET_DIR}" ]]; then
  printf 'Missing IEMOCAP dataset. Set IEMOCAP_DATASET_DIR to a local licensed copy.\n' >&2
  exit 2
fi
if [[ ! -d "${PRETRAIN_DIR}" ]]; then
  printf 'Missing GRAM checkpoint directory. Set GRAM_PRETRAIN_DIR to a local copy.\n' >&2
  exit 2
fi

mkdir -p "${CACHE_DIR}" "${OUTPUT_DIR}/v1" "${OUTPUT_DIR}/v5"
"${PYTHON_BIN}" "${SCRIPT_DIR}/cache_iemocap_features_v2.py" \
  --dataset-dir "${DATASET_DIR}" \
  --pretrain-dir "${PRETRAIN_DIR}" \
  --cache-dir "${CACHE_DIR}" \
  --batch-size "${IEMOCAP_CACHE_BATCH_SIZE:-16}" \
  --num-workers "${IEMOCAP_NUM_WORKERS:-4}" \
  --num-video-frames 2 --image-size 224

COMMON_ARGS=(
  --train-cache "${CACHE_DIR}/train.pt"
  --class-anchor-cache "${CACHE_DIR}/class_text_anchors.pt"
  --pretrain-dir "${PRETRAIN_DIR}"
  --epochs 30 --batch-size 128 --lr 1e-4 --log-steps 10
)
"${PYTHON_BIN}" "${SCRIPT_DIR}/train_iemocap_cached_v2.py" --version v1 \
  "${COMMON_ARGS[@]}" --output-dir "${OUTPUT_DIR}/v1"
"${PYTHON_BIN}" "${SCRIPT_DIR}/train_iemocap_cached_v2.py" --version v5 \
  "${COMMON_ARGS[@]}" --output-dir "${OUTPUT_DIR}/v5" \
  --gaussian-tau 0.5 --prior-kl-reduction mean \
  --confidence-floor 0.35 --no-confidence-random-baseline

printf 'IEMOCAP controlled diagnostic training completed.\n'
