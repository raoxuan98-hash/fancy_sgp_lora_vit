#!/bin/bash

# Configuration
DATASETS=("imagenet-r" "cifar100_224" "cub200_224" "cars196_224")
SEEDS=(1993 1996 1997)
GPUS=(0 1 2 4)

# Ensure script is executable: chmod +x run_within_domain.sh

run_dataset() {
    local DATASET=$1
    local GPU=$2
    local SEEDS=("${@:3}")  # Remaining args are seeds

    echo "============================================"
    echo "Starting dataset: $DATASET on GPU $GPU with seeds: ${SEEDS[*]}"
    echo "============================================"

    for SEED in "${SEEDS[@]}"; do
        echo "[$(date)] Running $DATASET | Seed: $SEED | GPU: $GPU"
        CUDA_VISIBLE_DEVICES=$GPU python main.py \
            --dataset "$DATASET" \
            --vit_type "vit-b-p16" \
            --lora_type "basic_lora" \
            --smart_defaults \
            --seed_list "$SEED" \

        echo "[$(date)] Completed: $DATASET seed $SEED"
    done

    echo "[$(date)] All seeds done for dataset: $DATASET"
}

# Launch each dataset on its assigned GPU in parallel
for i in "${!DATASETS[@]}"; do
    DATASET=${DATASETS[$i]}
    GPU=${GPUS[$i]}
    run_dataset "$DATASET" "$GPU" "${SEEDS[@]}" &
done

# Wait for all background jobs to finish
wait

echo "All within-domain evaluations completed."