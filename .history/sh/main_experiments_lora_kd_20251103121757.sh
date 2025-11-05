#!/usr/bin/env bash
set -euo pipefail

# 数据集列表
DATASETS=("cifar100_224" "imagenet-r" "cub200_224" "cars196_224")
SEEDS=(1993 1996 1997)

# LoRA + 蒸馏参数
LORA_TYPE="basic_lora"
VIT_TYPE="vit-b-p16-mocov3"
GAMMA_KD=1.0  # 使用蒸馏
UPDATE_TEACHER_EACH_TASK=True
DISTILLATION_TRANSFORM="identity"
KD_TYPE="feat"

# 顺序运行所有实验
for DATASET in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "Running LoRA + KD experiment: dataset=$DATASET, seed=$SEED"
        
        CUDA_VISIBLE_DEVICES=0 python -u main.py \
            --dataset "$DATASET" \
            --smart_defaults \
            --lora_type "$LORA_TYPE" \
            --vit_type "$VIT_TYPE" \
            --gamma_kd "$GAMMA_KD" \
            --update_teacher_each_task "$UPDATE_TEACHER_EACH_TASK" \
            --distillation_transform "$DISTILLATION_TRANSFORM" \
            --kd_type "$KD_TYPE" \
            --seed_list "$SEED"
    done
done

echo "LoRA + KD experiments completed."