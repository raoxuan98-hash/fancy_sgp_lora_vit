#!/bin/bash

# Configuration
DATASETS=("imagenet-r" "cifar100_224" "cub200_224" "cars196_224")
SEEDS=(1993 1996 1997)
GPUS=(0 1 2 4)  # Assign one GPU per dataset

# Ensure script is executable: chmod +x run_within_domain.sh

for i in "${!DATASETS[@]}"; do
    DATASET=${DATASETS[$i]}
    GPU=${GPUS[$i]}

    for SEED in "${SEEDS[@]}"; do
        echo "============================================"
        echo "Running dataset: $DATASET | Seed: $SEED | GPU: $GPU"
        echo "============================================"

        CUDA_VISIBLE_DEVICES=$GPU python main.py \
            --dataset "$DATASET" \
            --vit_type "vit-b-p16" \
            --lora_type "basic_lora" \
            --smart_defaults \
            --seed_list "$SEED" \
            --cross_domain False

        # Optional: add small delay or log timestamp
        echo "Completed: $DATASET seed $SEED"
    done
done

echo "All within-domain evaluations completed."