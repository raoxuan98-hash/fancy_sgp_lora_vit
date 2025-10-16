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
    # 如果 gamma_kd==0，只跑一次（use_aux=false）
    if (( $(echo "$G > 0" | bc -l) )); then
      aux_list=("${USE_AUXS[@]}")   # 两个值都跑
    else
      aux_list=(false)              # 只跑 false
    fi

    for AUX in "${aux_list[@]}"; do
      GPU=${GPUS[$((run_idx % ${#GPUS[@]}))]}
      echo "[RUN $run_idx | GPU $GPU] dataset=$DATASET lora=sgp_lora proj_temp=$T gamma_kd=$G use_aux_for_kd=$AUX"

      CUDA_VISIBLE_DEVICES=$GPU \
      python -u main.py \
        --dataset "$DATASET" \
        --smart_defaults \
        --lora_type sgp_lora \
        --weight_temp "$T" \
        --gamma_kd "$G" \
        --use_aux_for_kd "$AUX" &

      run_idx=$((run_idx+1))
      jobs_running=$((jobs_running+1))

      if (( jobs_running % MAX_PARALLEL == 0 )); then
        wait
        jobs_running=0
      fi
    done
  done
done

wait
echo "All runs completed."
