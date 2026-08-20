#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:?usage: run_efficiency_ntv3.sh runtime|memory|flops [physical_gpu_id]}"
GPU_ID="${2:-0}"
OUTPUT_DIR="${PROJECT_ROOT}/results/efficiency/ntv3"
mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_ROOT}"

case "${MODE}" in
  runtime|memory)
    for SPECIES_ID in $(seq 1 23); do
      CUDA_VISIBLE_DEVICES="${GPU_ID}" python -u tools/benchmark_efficiency_ntv3.py \
        "${MODE}" --species-id "${SPECIES_ID}" --device cuda:0 --output-dir "${OUTPUT_DIR}"
    done
    ;;
  flops)
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python -u tools/benchmark_efficiency_ntv3.py \
      flops --species-id 1 --device cuda:0 --output-dir "${OUTPUT_DIR}"
    ;;
  *)
    echo "unknown mode: ${MODE}" >&2
    exit 2
    ;;
esac
