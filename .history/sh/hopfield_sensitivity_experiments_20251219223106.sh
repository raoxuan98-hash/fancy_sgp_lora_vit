#!/bin/bash

# Configuration for Hopfield Network Sensitivity Analysis
DATASETS=("imagenet-r" "cifar100_224" "cub200_224" "cars196_224")
GPUS=(0 1 2 4)
VIT_TYPES=("vit-b-p16-mocov3")
LORA_TYPES=("full")

# Hopfield temperature values to test
HOPFIELD_TEMPS=(0.01 0.05 0.1 0.2 0.5)

# Hopfield topk values to test
HOPFIELD_TOPKS=(100 200 400 800 1000)

# Number of samples per class to test
NUM_SHOTS_LIST=(16 32 64 128 256)

# Ensure script is executable: chmod +x hopfield_sensitivity_experiments.sh

run_experiment() {
    local DATASET=$1
    local GPU=$2
    local VIT_TYPE=$3
    local LORA_TYPE=$4
    local HOPFIELD_TEMP=$5
    local HOPFIELD_TOPK=$6
    local NUM_SHOTS=$7
    
    echo "============================================"
    echo "Starting dataset: $DATASET on GPU $GPU | Vit_Type: $VIT_TYPE | Lora_Type: $LORA_TYPE"
    echo "Hopfield_Temp: $HOPFIELD_TEMP | Hopfield_Topk: $HOPFIELD_TOPK | Num_Shots: $NUM_SHOTS"
    echo "============================================"
    echo "[$(date)] Running $DATASET | GPU: $GPU | Vit_Type: $VIT_TYPE | Lora_Type: $LORA_TYPE | Hopfield_Temp: $HOPFIELD_TEMP | Hopfield_Topk: $HOPFIELD_TOPK | Num_Shots: $NUM_SHOTS"
    
    CUDA_VISIBLE_DEVICES=$GPU python main.py \
        --dataset "$DATASET" \
        --vit_type "$VIT_TYPE" \
        --lora_type "$LORA_TYPE" \
        --smart_defaults \
        --seed_list 1990 \
        --gamma_kd 0.0 \
        --compensator_types "SeqFT + Hopfield" \
        --hopfield_temp "$HOPFIELD_TEMP" \
        --hopfield_topk "$HOPFIELD_TOPK" \
        --num_shots "$NUM_SHOTS" \
        --cross_domain false

    echo "[$(date)] Completed: $DATASET (Vit_Type: $VIT_TYPE | Lora_Type: $LORA_TYPE | Hopfield_Temp: $HOPFIELD_TEMP | Hopfield_Topk: $HOPFIELD_TOPK | Num_Shots: $NUM_SHOTS)"
}

# Function to run temperature sensitivity experiments
run_temperature_sensitivity() {
    echo "################################################################"
    echo "Starting Hopfield Temperature Sensitivity Experiments"
    echo "################################################################"
    
    for HOPFIELD_TEMP in "${HOPFIELD_TEMPS[@]}"; do
        echo "------------------------------------------------------------"
        echo "Testing with Hopfield Temperature: $HOPFIELD_TEMP"
        echo "------------------------------------------------------------"
        
        # 使用固定的 hopfield_topk=400 和 num_shots=64
        for i in "${!DATASETS[@]}"; do
            DATASET=${DATASETS[$i]}
            GPU=${GPUS[$i]}
            run_experiment "$DATASET" "$GPU" "${VIT_TYPES[0]}" "${LORA_TYPES[0]}" "$HOPFIELD_TEMP" 400 64 &
        done
        
        # 等待当前温度的所有作业完成
        wait
        
        echo "------------------------------------------------------------"
        echo "Completed experiments with Hopfield Temperature: $HOPFIELD_TEMP"
        echo "------------------------------------------------------------"
        echo ""
    done
}

# Function to run topk sensitivity experiments
run_topk_sensitivity() {
    echo "################################################################"
    echo "Starting Hopfield TopK Sensitivity Experiments"
    echo "################################################################"
    
    for HOPFIELD_TOPK in "${HOPFIELD_TOPKS[@]}"; do
        echo "------------------------------------------------------------"
        echo "Testing with Hopfield TopK: $HOPFIELD_TOPK"
        echo "------------------------------------------------------------"
        
        # 使用固定的 hopfield_temp=0.05 和 num_shots=64
        for i in "${!DATASETS[@]}"; do
            DATASET=${DATASETS[$i]}
            GPU=${GPUS[$i]}
            run_experiment "$DATASET" "$GPU" "${VIT_TYPES[0]}" "${LORA_TYPES[0]}" 0.05 "$HOPFIELD_TOPK" 64 &
        done
        
        # 等待当前 topk 的所有作业完成
        wait
        
        echo "------------------------------------------------------------"
        echo "Completed experiments with Hopfield TopK: $HOPFIELD_TOPK"
        echo "------------------------------------------------------------"
        echo ""
    done
}

# Function to run sample size sensitivity experiments
run_sample_size_sensitivity() {
    echo "################################################################"
    echo "Starting Sample Size Sensitivity Experiments"
    echo "################################################################"
    
    for NUM_SHOTS in "${NUM_SHOTS_LIST[@]}"; do
        echo "------------------------------------------------------------"
        echo "Testing with Num Shots: $NUM_SHOTS"
        echo "------------------------------------------------------------"
        
        # 使用固定的 hopfield_temp=0.05 和 hopfield_topk=400
        for i in "${!DATASETS[@]}"; do
            DATASET=${DATASETS[$i]}
            GPU=${GPUS[$i]}
            run_experiment "$DATASET" "$GPU" "${VIT_TYPES[0]}" "${LORA_TYPES[0]}" 0.05 400 "$NUM_SHOTS" &
        done
        
        # 等待当前样本数量的所有作业完成
        wait
        
        echo "------------------------------------------------------------"
        echo "Completed experiments with Num Shots: $NUM_SHOTS"
        echo "------------------------------------------------------------"
        echo ""
    done
}

# Function to run combined sensitivity experiments (selected combinations)
run_combined_sensitivity() {
    echo "################################################################"
    echo "Starting Combined Sensitivity Experiments (Selected Combinations)"
    echo "################################################################"
    
    # 选择一些代表性的组合进行测试
    declare -a COMBINATIONS=(
        "0.01 200 32"  # 低温度，中等topk，小样本
        "0.01 800 128" # 低温度，高topk，中等样本
        "0.05 400 64"  # 默认值
        "0.1 200 128"  # 中等温度，中等topk，中等样本
        "0.2 800 256"  # 高温度，高topk，大样本
        "0.5 100 16"   # 很高温度，低topk，很小样本
    )
    
    for combo in "${COMBINATIONS[@]}"; do
        # 解析组合参数
        read -r HOPFIELD_TEMP HOPFIELD_TOPK NUM_SHOTS <<< "$combo"
        
        echo "------------------------------------------------------------"
        echo "Testing Combination: Temp=$HOPFIELD_TEMP, TopK=$HOPFIELD_TOPK, Shots=$NUM_SHOTS"
        echo "------------------------------------------------------------"
        
        for i in "${!DATASETS[@]}"; do
            DATASET=${DATASETS[$i]}
            GPU=${GPUS[$i]}
            run_experiment "$DATASET" "$GPU" "${VIT_TYPES[0]}" "${LORA_TYPES[0]}" "$HOPFIELD_TEMP" "$HOPFIELD_TOPK" "$NUM_SHOTS" &
        done
        
        # 等待当前组合的所有作业完成
        wait
        
        echo "------------------------------------------------------------"
        echo "Completed combination: Temp=$HOPFIELD_TEMP, TopK=$HOPFIELD_TOPK, Shots=$NUM_SHOTS"
        echo "------------------------------------------------------------"
        echo ""
    done
}

# 主执行流程
echo "Starting Hopfield Network Sensitivity Analysis"
echo "Date: $(date)"
echo ""

# 运行各项敏感性分析
run_temperature_sensitivity
run_topk_sensitivity
run_sample_size_sensitivity
run_combined_sensitivity

echo "All Hopfield Network sensitivity experiments completed."
echo "Date: $(date)"