#!/bin/bash

# GPU 0: nsp_weight=0.05, nsp_eps=0.05
export CUDA_VISIBLE_DEVICES=0

python main.py \
    --cross_domain \
    --enable_incremental_split False \
    --vit_type vit-b-p16-clip \
    --lora_type nsp_lora \
    --nsp_weight 0.05 \
    --nsp_eps 0.05 \
    --num_shots 64 \
    --lrate 0.0001 \
    --batch_size 16 \
    --iterations 1500 \
    --seed_list 1993 \
    --optimizer adamw \
    --weight_decay 3e-5 \
    --evaluate_final_only True \
    --feature_combination_type combined \
    --auxiliary_data_size 2048 \
    --smart_defaults True