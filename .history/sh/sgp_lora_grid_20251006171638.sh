#!/usr/bin/env bash
set -euo pipefail

DATASET=${1:-imagenet-r}
GPU_LIST=${2:-1,2,4}
IFS=',' read -r -a GPUS <<< "$GPU_LIST"

PROJ_TEMPS=(1.0 4.0)
GAMMA_KDS=(0.0 0.5)

MAX_PARALLEL=3

run_idx=0
jobs_running=0

for T in "${PROJ_TEMPS[@]}"; do
  for G in "${GAMMA_KDS[@]}"; do
    # 决定是否需要测试 use_aux_for_kd=True 和 False
    if [[ "$G" == "0.5" ]]; then
      test_aux_true=true
      test_aux_false=true
    else  # G == "0.0"
      test_aux_true=false
      test_aux_false=true   # 只跑 False（即不加参数）
    fi

    # 先跑 use_aux_for_kd=False（即不加参数）
    if [[ "$test_aux_false" == "true" ]]; then
      GPU=${GPUS[$((run_idx % ${#GPUS[@]}))]}
      echo "[RUN $run_idx | GPU $GPU] dataset=$DATASET lora=sgp_lora proj_temp=$T gamma_kd=$G use_aux_for_kd=False"

      CUDA_VISIBLE_DEVICES=$GPU \
      python -u main.py \
        --dataset "$DATASET" \
        --smart_defaults \
        --lora_type sgp_lora \
        --weight_temp "$T" \
        --gamma_kd "$G" &

      run_idx=$((run_idx + 1))
      jobs_running=$((jobs_running + 1))

      if (( jobs_running % MAX_PARALLEL == 0 )); then
        wait
        jobs_running=0
      fi
    fi

    # 再跑 use_aux_for_kd=True（加上 --use_aux_for_kd）
    if [[ "$test_aux_true" == "true" ]]; then
      GPU=${GPUS[$((run_idx % ${#GPUS[@]}))]}
      echo "[RUN $run_idx | GPU $GPU] dataset=$DATASET lora=sgp_lora proj_temp=$T gamma_kd=$G use_aux_for_kd=True"

      CUDA_VISIBLE_DEVICES=$GPU \
      python -u main.py \
        --dataset "$DATASET" \
        --smart_defaults \
        --lora_type sgp_lora \
        --weight_temp "$T" \
        --gamma_kd "$G" \
        --use_aux_for_kd &

      run_idx=$((run_idx + 1))
      jobs_running=$((jobs_running + 1))

      if (( jobs_running % MAX_PARALLEL == 0 )); then
        wait
        jobs_running=0
      fi
    fi
  done
done

wait
echo "All runs completed."