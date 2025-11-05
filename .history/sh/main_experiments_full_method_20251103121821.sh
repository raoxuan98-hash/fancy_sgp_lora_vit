#!/usr/bin/env bash
set -euo pipefail

# 数据集列表
DATASETS=("cifar100_224" "imagenet-r" "cub200_224" "cars196_224")
SEEDS=(1993 1996 1997)

# 完整方法参数
LORA_TYPE="sgp_lora"
VIT_TYPE="vit-b-p16-mocov3"
GAMMA_KD=0.0  # 不使用蒸馏
WEIGHT_TEMP=1.0
WEIGHT_KIND="log1p"

# 顺序运行所有实验
for DATASET in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "Running Full Method experiment: dataset=$DATASET, seed=$SEED"
        
        CUDA_VISIBLE_DEVICES=0 python -u main.py \
            --dataset "$DATASET" \
            --smart_defaults \
            --lora_type "$LORA_TYPE" \
            --vit_type "$VIT_TYPE" \
            --gamma_kd "$GAMMA_KD" \
            --weight_temp "$WEIGHT_TEMP" \
            --weight_kind "$WEIGHT_KIND" \
            --seed_list "$SEED"
    done
done

echo "Full Method experiments completed."