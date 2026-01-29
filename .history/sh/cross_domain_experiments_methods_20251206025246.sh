#!/bin/bash

# Configuration
DATASET="cross_domain_elevater"
GPUS=(0 1 2 4 5)
VIT_TYPES=("vit-b-p16" "vit-b-p16-clip")
SEEDS=(1993 1996)

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
        --iterations 2500 \
        $ADDITIONAL_PARAMS

    echo "[$(date)] Completed: $METHOD_NAME (Gamma_KD: $GAMMA_KD | Vit_Type: $VIT_TYPE)"
}

# 并行运行所有方法和架构
echo "################################################################"
echo "Starting cross-domain experiments with all 5 methods and ${#VIT_TYPES[@]} architectures"
echo "################################################################"

# 存储所有进程ID
PIDS=()

# 为每个架构运行所有方法
for VIT_TYPE in "${VIT_TYPES[@]}"; do
    echo "========================================================"
    echo "Starting experiments for architecture: $VIT_TYPE"
    echo "========================================================"

    # 1. LoRA方法 (GPU 0)
    {
        run_experiment "LoRA_${VIT_TYPE}" "basic_lora" "0" "0.0" "$VIT_TYPE" ""
    } &
    PIDS+=($!)
    
    # 2. LoRA + KD方法 (GPU 1)
    {
        run_experiment "LoRA_KD_${VIT_TYPE}" "basic_lora" "1" "1.0" "$VIT_TYPE" "--update_teacher_each_task --kd_type feat"
    } &
    PIDS+=($!)
    
    # 3. Full方法 (GPU 2)
    {
        run_experiment "Full_${VIT_TYPE}" "full" "2" "0.0" "$VIT_TYPE" "--lrate 5e-6"
    } &
    PIDS+=($!)
    
    # 4. Full_NSP方法 (GPU 4)
    {
        run_experiment "Full_NSP_${VIT_TYPE}" "full_nsp" "4" "0.5" "$VIT_TYPE" "--lrate 5e-6 --update_teacher_each_task --nsp_eps 0.05 --kd_type feat --nsp_weight 0.0"
    } &
    PIDS+=($!)
    
    # 5. Full + KD方法 (GPU 5)
    {
        run_experiment "Full_KD_${VIT_TYPE}" "full" "5" "1.0" "$VIT_TYPE" "--lrate 5e-6 --update_teacher_each_task --kd_type feat"
    } &
    PIDS+=($!)
done

# 等待所有实验完成
echo "Waiting for all experiments to complete..."
for PID in "${PIDS[@]}"; do
    wait $PID
    echo "Experiment with PID $PID completed"
done

echo "################################################################"
echo "All cross-domain experiments completed for all methods"
echo "################################################################"