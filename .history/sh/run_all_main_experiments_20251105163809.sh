#!/usr/bin/env bash
set -euo pipefail

echo "Starting all main experiments with parallel GPU execution..."

# 创建总日志目录
MASTER_LOG_DIR="logs/all_experiments_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$MASTER_LOG_DIR"

# 数据集列表
DATASETS=("cifar100_224" "imagenet-r" "cub200_224" "cars196_224")
SEEDS=(1993 1996 1997)

# GPU分配 - 每个数据集使用一个GPU
GPUS=(0 1 2 4)

# 运行单个实验类型的函数
run_experiment_type() {
    local experiment_name="$1"
    local lora_type="$2"
    shift 2
    local additional_params=("$@")
    
    echo "=========================================="
    echo "Running $experiment_name Experiments"
    echo "=========================================="
    
    # 创建实验类型特定的日志目录
    LOG_DIR="$MASTER_LOG_DIR/${experiment_name}_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$LOG_DIR"
    
    # 并行运行所有数据集
    PIDS=()
    
    for i in "${!DATASETS[@]}"; do
        DATASET="${DATASETS[$i]}"
        GPU="${GPUS[$i]}"
        
        # 为每个数据集创建子脚本
        cat > "$LOG_DIR/run_${DATASET}.sh" << EOF
#!/usr/bin/env bash
set -euo pipefail

DATASET="$DATASET"
GPU="$GPU"
LOG_DIR="$LOG_DIR"
LORA_TYPE="$lora_type"
VIT_TYPE="vit-b-p16-mocov3"
SEEDS=(${SEEDS[*]})
ADDITIONAL_PARAMS=(${additional_params[*]})

echo "Starting $experiment_name experiments for $DATASET on GPU \$GPU"

for SEED in "\${SEEDS[@]}"; do
    echo "Running $experiment_name: dataset=\$DATASET, seed=\$SEED, GPU=\$GPU"
    
    CUDA_VISIBLE_DEVICES=\$GPU python -u main.py \\
        --dataset "\$DATASET" \\
        --smart_defaults \\
        --lora_type "\$LORA_TYPE" \\
        --vit_type "\$VIT_TYPE" \\
        \${ADDITIONAL_PARAMS[@]} \\
        --seed_list "\$SEED" \\
        2>&1 | tee "\$LOG_DIR/\${DATASET}_seed\${SEED}.log"
done

echo "All $experiment_name experiments completed for $DATASET on GPU \$GPU"
EOF
        
        chmod +x "$LOG_DIR/run_${DATASET}.sh"
        
        # 在后台运行每个数据集的实验
        echo "Starting $experiment_name experiments for $DATASET on GPU $GPU"
        "$LOG_DIR/run_${DATASET}.sh" &
        PIDS+=($!)
    done
    
    # 等待所有实验完成
    echo "Waiting for all $experiment_name experiments to complete..."
    for PID in "${PIDS[@]}"; do
        wait $PID
    done
    
    echo "$experiment_name experiments completed. Logs saved to $LOG_DIR"
}

# 1. 运行基础LoRA实验
run_experiment_type "basic_lora" "basic_lora" "--gamma_kd" "0.0"

# 2. 运行LoRA + 蒸馏实验
run_experiment_type "lora_kd" "basic_lora" \
    "--gamma_kd" "1.0" \
    "--update_teacher_each_task" "True" \
    "--distillation_transform" "identity" \
    "--kd_type" "feat"

# 3. 运行LoRA-NSP实验 (nsp_weight=0.05)
run_experiment_type "nsp_lora_0.05" "nsp_lora" \
    "--gamma_kd" "0.0" \
    "--nsp_weight" "0.05" \
    "--nsp_eps" "0.05"

# 4. 运行LoRA-NSP实验 (nsp_weight=0.00)
run_experiment_type "nsp_lora_0.00" "nsp_lora" \
    "--gamma_kd" "0.0" \
    "--nsp_weight" "0.00" \
    "--nsp_eps" "0.05"

# 5. 运行完整方法实验
run_experiment_type "sgp_lora" "sgp_lora" \
    "--gamma_kd" "0.0" \
    "--weight_temp" "1.0" \
    "--weight_p" "1.0" \
    "--weight_kind" "log1p"

echo "=========================================="
echo "All main experiments completed!"
echo "Logs saved to: $MASTER_LOG_DIR"
echo "=========================================="

# 计算总实验数量
TOTAL_EXPERIMENTS=$((${#DATASETS[@]} * ${#SEEDS[@]} * 5))  # 5种实验类型
echo "Total experiments run: $TOTAL_EXPERIMENTS"
echo "Experiments per type: ${#DATASETS[@]} datasets × ${#SEEDS[@]} seeds = $((${#DATASETS[@]} * ${#SEEDS[@]}))"