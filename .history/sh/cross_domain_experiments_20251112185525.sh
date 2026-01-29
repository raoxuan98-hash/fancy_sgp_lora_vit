#!/bin/bash

# Cross-domain class-incremental learning experiments
# 10 datasets as 10 tasks

# Exit on error
set -e

# Create logs directory
mkdir -p sldc_logs_cross_domain

# Common parameters
COMMON_PARAMS="--model_name sldc --vit_type vit-b-p16-mocov3 --smart_defaults --batch_size 16 --lrate 0.0001 --optimizer adamw --iterations 2000 --seed_list 1993 1996 1997"

# Cross-domain dataset
CROSS_DOMAIN_DATASET="cross_domain_elevater"

echo "Starting cross-domain experiments with dataset: $CROSS_DOMAIN_DATASET"
echo "Datasets: caltech-101, dtd, eurosat_clip, fgvc-aircraft-2013b-variants102, food-101, mnist, oxford-flower-102, oxford-iiit-pets, stanford-cars, imagenet-r"
echo "Total tasks: 10 (one per dataset)"
echo "============================================================="

# ==========================================
# 1. Basic LoRA (baseline)
# ==========================================
echo "Running Basic LoRA..."
python main.py \
    --dataset $CROSS_DOMAIN_DATASET \
    --cross_domain \
    --lora_type basic_lora \
    --lora_rank 4 \
    $COMMON_PARAMS

# ==========================================
# 2. Basic LoRA + Knowledge Distillation
# ==========================================
echo "Running Basic LoRA + Knowledge Distillation..."
python main.py \
    --dataset $CROSS_DOMAIN_DATASET \
    --cross_domain \
    --lora_type basic_lora \
    --lora_rank 4 \
    --gamma_kd 1.0 \
    --kd_type feat \
    --distillation_transform identity \
    --update_teacher_each_task True \
    $COMMON_PARAMS

# ==========================================
# 3. NSP LoRA
# ==========================================
echo "Running NSP LoRA..."
python main.py \
    --dataset $CROSS_DOMAIN_DATASET \
    --cross_domain \
    --lora_type nsp_lora \
    --lora_rank 4 \
    --nsp_eps 0.05 \
    --nsp_weight 0.05 \
    $COMMON_PARAMS

# ==========================================
# 4. SGP LoRA (temperature=1.0)
# ==========================================
echo "Running SGP LoRA (temperature=1.0)..."
python main.py \
    --dataset $CROSS_DOMAIN_DATASET \
    --cross_domain \
    --lora_type sgp_lora \
    --lora_rank 4 \
    --weight_temp 1.0 \
    --weight_kind log1p \
    --weight_p 1.0 \
    $COMMON_PARAMS

# ==========================================
# 5. SGP LoRA (temperature=2.0)
# ==========================================
echo "Running SGP LoRA (temperature=2.0)..."
python main.py \
    --dataset $CROSS_DOMAIN_DATASET \
    --cross_domain \
    --lora_type sgp_lora \
    --lora_rank 4 \
    --weight_temp 2.0 \
    --weight_kind log1p \
    --weight_p 1.0 \
    $COMMON_PARAMS

# ==========================================
# 6. SGP LoRA (temperature=2.0, p=2.0)
# ==========================================
echo "Running SGP LoRA (temperature=2.0, p=2.0)..."
python main.py \
    --dataset $CROSS_DOMAIN_DATASET \
    --cross_domain \
    --lora_type sgp_lora \
    --lora_rank 4 \
    --weight_temp 2.0 \
    --weight_kind log1p \
    --weight_p 2.0 \
    $COMMON_PARAMS

echo "============================================================="
echo "All cross-domain experiments completed!"
echo "Results are saved in sldc_logs_cross_domain/"
echo "============================================================="