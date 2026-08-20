#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:?usage: run_efficiency_ipro_mp.sh smoke|runtime|memory|flops [physical_gpu_id]}"
GPU_ID="${2:-0}"
OUTPUT_DIR="${PROJECT_ROOT}/results/efficiency/ipro_mp"
mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_ROOT}"

run_smoke() {
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python -u tools/benchmark_efficiency_ipro_mp.py \
    smoke --species-id 1 --device cuda:0 --output-dir "${OUTPUT_DIR}"
}

case "${MODE}" in
  smoke)
    run_smoke
    ;;
  runtime|memory)
    run_smoke
    for SPECIES_ID in $(seq 1 23); do
      CUDA_VISIBLE_DEVICES="${GPU_ID}" python -u tools/benchmark_efficiency_ipro_mp.py \
        "${MODE}" --species-id "${SPECIES_ID}" --device cuda:0 --output-dir "${OUTPUT_DIR}"
    done
    ;;
  flops)
    run_smoke
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python -u tools/benchmark_efficiency_ipro_mp.py \
      flops --species-id 1 --device cuda:0 --output-dir "${OUTPUT_DIR}"
    ;;
  *)
    echo "unknown mode: ${MODE}" >&2
    exit 2
    ;;
esac
