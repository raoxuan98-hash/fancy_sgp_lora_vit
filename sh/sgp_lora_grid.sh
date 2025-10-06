#!/usr/bin/env bash
set -euo pipefail

# Usage: bash sh/sgp_lora_grid.sh <dataset> [gpus]
# - dataset: one of datasets supported by main.py (e.g., cifar100_224)
# - gpus: comma-separated GPU IDs to round-robin assign per run (default: 0,1,2,3)

DATASET=${1:-cifar100_224}
GPU_LIST=${2:-0,1,2,3}
IFS=',' read -r -a GPUS <<< "$GPU_LIST"

# Grid settings for sgp_lora
PROJ_TEMPS=(1.0 4.0)
GAMMA_KDS=(0.0 0.5)
USE_AUXS=(true false)

run_idx=0
for T in "${PROJ_TEMPS[@]}"; do
  for G in "${GAMMA_KDS[@]}"; do
    for AUX in "${USE_AUXS[@]}"; do
      GPU=${GPUS[$((run_idx % ${#GPUS[@]}))]}
      export CUDA_VISIBLE_DEVICES=$GPU
      echo "[RUN $run_idx | GPU $GPU] dataset=$DATASET lora=sgp_lora proj_temp=$T gamma_kd=$G use_aux_for_kd=$AUX"
      python -u main.py \
        --dataset "$DATASET" \
        --smart_defaults \
        --lora_type sgp_lora \
        --weight_temp "$T" \
        --gamma_kd "$G" \
        --use_aux_for_kd "$AUX" \
        --test
      run_idx=$((run_idx+1))
    done
  done
done

echo "All runs completed."

