#!/bin/bash

# GPU 4: nsp_weight=0.00, nsp_eps=0.10
export CUDA_VISIBLE_DEVICES=4

python main.py \
    --cross_domain True \
    --enable_incremental_split False \
    --vit_type vit-b-p16-clip \
    --lora_type nsp_lora \
    --nsp_weight 0.00 \
    --nsp_eps 0.10 \
    --num_shots 64 \
    --lrate 0.0001 \
    --batch_size 16 \
    --iterations 1500 \
    --seed_list 1993 1996 \
    --optimizer adamw \
    --weight_decay 3e-5 \
    --evaluate_final_only True \
    --feature_combination_type combined \
    --auxiliary_data_size 2048 \
    --smart_defaults True