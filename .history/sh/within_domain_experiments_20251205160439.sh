#!/bin/bash

# Configuration
DATASETS=("imagenet-r" "cifar100_224" "cub200_224" "cars196_224")
SEEDS=(1993 1996 1997)
GPUS=(0 1 2 4)
GAMMA_KDS=(0.0)
VIT_TYPES=("vit-b-p16" "vit-b-p16-mocov3")

# Ensure script is executable: chmod +x run_within_domain.sh

run_dataset() {
    local DATASET=$1
    local GPU=$2
    local GAMMA_KD=$3
    local VIT_TYPE=$4
    local SEEDS=("${@:5}")

    echo "============================================"
    echo "Starting dataset: $DATASET on GPU $GPU | Gamma_KD: $GAMMA_KD | Vit_Type: $VIT_TYPE | Seeds: ${SEEDS[*]}"
    echo "============================================"

    echo "[$(date)] Running $DATASET | Seed: $SEED | GPU: $GPU | Gamma_KD: $GAMMA_KD | Vit_Type: $VIT_TYPE"
    CUDA_VISIBLE_DEVICES=$GPU python main.py \
        --dataset "$DATASET" \
        --vit_type "$VIT_TYPE" \
        --lora_type "full_nsp" \
        --smart_defaults \
        --seed_list "$SEEDS" \
        --gamma_kd "$GAMMA_KD"

    echo "[$(date)] Completed: $DATASET seed $SEED (Gamma_KD: $GAMMA_KD | Vit_Type: $VIT_TYPE)"

    echo "[$(date)] All seeds done for dataset: $DATASET (Gamma_KD: $GAMMA_KD | Vit_Type: $VIT_TYPE)"
}

# 按顺序运行不同的vit_type和gamma_kd值
for VIT_TYPE in "${VIT_TYPES[@]}"; do
    for GAMMA_KD in "${GAMMA_KDS[@]}"; do
        echo "################################################################"
        echo "Starting experiments with Vit_Type = $VIT_TYPE | Gamma_KD = $GAMMA_KD"
        echo "################################################################"
        
        # 为当前vit_type和gamma_kd值启动所有数据集的并行运行
        for i in "${!DATASETS[@]}"; do
            DATASET=${DATASETS[$i]}
            GPU=${GPUS[$i]}
            run_dataset "$DATASET" "$GPU" "$GAMMA_KD" "$VIT_TYPE" "${SEEDS[@]}" &
        done

        # 等待当前vit_type和gamma_kd的所有作业完成
        wait
        
        echo "################################################################"
        echo "Completed all experiments with Vit_Type = $VIT_TYPE | Gamma_KD = $GAMMA_KD"
        echo "################################################################"
        echo ""
    done
done

echo "All within-domain evaluations completed for all Vit_Type and Gamma_KD values."