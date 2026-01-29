#!/bin/bash

# Configuration
DATASET="cross_domain_elevater"
GPUS=(0 1 2 4 5)
VIT_TYPES=("vit-b-p16")
SEEDS=(1993 1996 1997)

# Ensure script is executable: chmod +x cross_domain_experiments_methods.sh

# 运行单个实验的函数
run_experiment() {
    local METHOD_NAME=$1
    local LORA_TYPE=$2
    local GPU=$3
    local GAMMA_KD=$4
    local VIT_TYPE=$5
    local ADDITIONAL_PARAMS=$6
    
    echo "============================================"
    echo "Starting $METHOD_NAME on GPU $GPU | Gamma_KD: $GAMMA_KD | Vit_Type: $VIT_TYPE | Seeds: ${SEEDS[*]}"
    echo "============================================"

    echo "[$(date)] Running $METHOD_NAME | GPU: $GPU | Gamma_KD: $GAMMA_KD | Vit_Type: $VIT_TYPE"
    CUDA_VISIBLE_DEVICES=$GPU python main.py \
        --dataset "$DATASET" \
        --vit_type "$VIT_TYPE" \
        --lora_type "$LORA_TYPE" \
        --cross_domain \
        --seed_list "${SEEDS[@]}" \
        --gamma_kd "$GAMMA_KD" \
        $ADDITIONAL_PARAMS

    echo "[$(date)] Completed: $METHOD_NAME (Gamma_KD: $GAMMA_KD | Vit_Type: $VIT_TYPE)"
}

# 并行运行所有方法
echo "################################################################"
echo "Starting cross-domain experiments with all 5 methods"
echo "################################################################"

# 1. LoRA方法 (GPU 0)
{
    run_experiment "LoRA" "basic_lora" "0" "0.0" "${VIT_TYPES[0]}" ""
} &
PID_LORA=$!

# 2. LoRA + KD方法 (GPU 1)
{
    run_experiment "LoRA_KD" "basic_lora" "1" "1.0" "${VIT_TYPES[0]}" "--update_teacher_each_task --kd_type feat"
} &
PID_LORA_KD=$!

# 3. Full方法 (GPU 2)
{
    run_experiment "Full" "full" "2" "0.0" "${VIT_TYPES[0]}" "--lrate 5e-6"
} &
PID_FULL=$!

# 4. Full_NSP方法 (GPU 4)
{
    run_experiment "Full_NSP" "full_nsp" "4" "0.0" "${VIT_TYPES[0]}" "--lrate 5e-6 --nsp_eps 0.05 --nsp_weight 0.0"
} &
PID_FULL_NSP=$!

# 5. Full + KD方法 (GPU 5)
{
    run_experiment "Full_KD" "full" "5" "1.0" "${VIT_TYPES[0]}" "--lrate 5e-6 --update_teacher_each_task --kd_type cos"
} &
PID_FULL_KD=$!

# 等待所有实验完成
echo "Waiting for all experiments to complete..."
wait $PID_LORA
echo "LoRA experiments completed"
wait $PID_LORA_KD
echo "LoRA + KD experiments completed"
wait $PID_FULL
echo "Full experiments completed"
wait $PID_FULL_NSP
echo "Full_NSP experiments completed"
wait $PID_FULL_KD
echo "Full + KD experiments completed"

echo "################################################################"
echo "All cross-domain experiments completed for all methods"
echo "################################################################"