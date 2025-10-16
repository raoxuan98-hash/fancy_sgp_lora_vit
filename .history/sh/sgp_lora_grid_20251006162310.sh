#!/usr/bin/env bash
set -euo pipefail

DATASET=${1:-imagenet-r}
GPU_LIST=${2:-1,2,4}
IFS=',' read -r -a GPUS <<< "$GPU_LIST"

PROJ_TEMPS=(1.0 4.0)
GAMMA_KDS=(0.0 0.5)
USE_AUXS=(true false)

MAX_PARALLEL=3

run_idx=0
jobs_running=0

for T in "${PROJ_TEMPS[@]}"; do
  for G in "${GAMMA_KDS[@]}"; do
    for AUX in "${USE_AUXS[@]}"; do
      GPU=${GPUS[$((run_idx % ${#GPUS[@]}))]}
      export CUDA_VISIBLE_DEVICES=$GPU
      echo "[RUN $run_idx | GPU $GPU] dataset=$DATASET lora=sgp_lora proj_temp=$T gamma_kd=$G use_aux_for_kd=$AUX"

      # 后台运行，加上 &
      python -u main.py \
        --dataset "$DATASET" \
        --smart_defaults \
        --lora_type sgp_lora \
        --weight_temp "$T" \
        --gamma_kd "$G" \
        --use_aux_for_kd "$AUX" &

      run_idx=$((run_idx+1))
      jobs_running=$((jobs_running+1))

      # 控制并行数量
      if (( jobs_running % MAX_PARALLEL == 0 )); then
        wait    # 等这批跑完
        jobs_running=0
      fi
    done
  done
done

wait
echo "All runs completed."
