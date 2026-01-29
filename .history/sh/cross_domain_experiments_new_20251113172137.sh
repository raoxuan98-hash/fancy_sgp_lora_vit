#!/usr/bin/env bash
set -euo pipefail

echo "Starting Cross-Domain Experiments..."

# 创建总日志目录
MASTER_LOG_DIR="logs/cross_domain_experiments_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$MASTER_LOG_DIR"

# 跨域实验数据集
DATASET="cross_domain_elevater"
SEEDS=(1993 1996 1997)

# GPU分配 - 使用所有5个可用GPU，并行运行不同方法
GPUS=(0 1 2 4 5)

# 运行单个实验类型的函数
run_experiment_type() {
    local experiment_name="$1"
    local lora_type="$2"
    shift 2
    local additional_params=("$@")
    
    echo "=========================================="
    echo "Running $experiment_name Cross-Domain Experiments"
    echo "=========================================="
    
    # 创建实验类型特定的日志目录
    LOG_DIR="$MASTER_LOG_DIR/${experiment_name}_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$LOG_DIR"
    
    # 并行运行所有种子
    PIDS=()
    
    for i in "${!SEEDS[@]}"; do
        SEED="${SEEDS[$i]}"
        GPU="${GPUS[$i]}"
        
        # 为每个种子创建子脚本
        cat > "$LOG_DIR/run_seed_${SEED}.sh" << EOF
#!/usr/bin/env bash
set -euo pipefail

DATASET="$DATASET"
GPU="$GPU"
LOG_DIR="$LOG_DIR"
SEED="$SEED"
LORA_TYPE="$lora_type"
VIT_TYPE="vit-b-p16-mocov3"
ADDITIONAL_PARAMS=(${additional_params[*]})

echo "Starting $experiment_name cross-domain experiment for seed \$SEED on GPU \$GPU"

CUDA_VISIBLE_DEVICES=\$GPU python -u main.py \\
    --dataset "\$DATASET" \\
    --smart_defaults \\
    --lora_type "\$LORA_TYPE" \\
    --vit_type "\$VIT_TYPE" \\
    --cross_domain "True" \\
    --num_shots "16" \\
    \${ADDITIONAL_PARAMS[@]} \\
    --seed_list "\$SEED" \\
    2>&1 | tee "\$LOG_DIR/seed\${SEED}.log"

echo "$experiment_name cross-domain experiment completed for seed \$SEED on GPU \$GPU"
EOF
        
        chmod +x "$LOG_DIR/run_seed_${SEED}.sh"
        
        # 在后台运行每个种子的实验
        echo "Starting $experiment_name cross-domain experiment for seed $SEED on GPU $GPU"
        "$LOG_DIR/run_seed_${SEED}.sh" &
        PIDS+=($!)
    done
    
    # 等待所有实验完成
    echo "Waiting for all $experiment_name cross-domain experiments to complete..."
    for PID in "${PIDS[@]}"; do
        wait $PID
    done
    
    echo "$experiment_name cross-domain experiments completed. Logs saved to $LOG_DIR"
}

# 1. 运行基础LoRA跨域实验
run_experiment_type "basic_lora" "basic_lora" "--gamma_kd" "0.0"

# 2. 运行LoRA + 蒸馏跨域实验 (gamma_kd=1.0)
run_experiment_type "lora_kd_1.0" "basic_lora" \
    "--gamma_kd" "1.0" \
    "--update_teacher_each_task" "True" \
    "--distillation_transform" "identity" \
    "--kd_type" "feat"

# 2b. 运行LoRA + 蒸馏跨域实验 (gamma_kd=0.5)
run_experiment_type "lora_kd_0.5" "basic_lora" \
    "--gamma_kd" "0.5" \
    "--update_teacher_each_task" "True" \
    "--distillation_transform" "identity" \
    "--kd_type" "feat"

# 3. 运行LoRA-NSP跨域实验 (nsp_weight=0.05)
run_experiment_type "nsp_lora_0.05" "nsp_lora" \
    "--gamma_kd" "0.0" \
    "--nsp_weight" "0.05" \
    "--nsp_eps" "0.05"

# 3b. 运行LoRA-NSP跨域实验 (nsp_weight=0.00)
run_experiment_type "nsp_lora_0.00" "nsp_lora" \
    "--gamma_kd" "0.0" \
    "--nsp_weight" "0.00" \
    "--nsp_eps" "0.05"

# 4. 运行LoRA-SGP跨域实验 (weight_temp=2.0, weight_p=1.0)
run_experiment_type "sgp_lora_t2.0_p1.0" "sgp_lora" \
    "--gamma_kd" "0.0" \
    "--weight_temp" "1.0" \
    "--weight_p" "1.0" \
    "--weight_kind" "log1p"

# 4b. 运行LoRA-SGP跨域实验 (weight_temp=2.0, weight_p=2.0)
run_experiment_type "sgp_lora_t2.0_p2.0" "sgp_lora" \
    "--gamma_kd" "0.0" \
    "--weight_temp" "2.0" \
    "--weight_p" "2.0" \
    "--weight_kind" "log1p"

echo "=========================================="
echo "All Cross-Domain experiments completed!"
echo "Logs saved to: $MASTER_LOG_DIR"
echo "=========================================="

# 计算总实验数量
TOTAL_EXPERIMENTS=$((${#SEEDS[@]} * 7))  # 7种实验变体
echo "Total experiments run: $TOTAL_EXPERIMENTS"
echo "Experiments per type: ${#SEEDS[@]} seeds"
echo "Experiment variants: 7 (basic_lora, lora_kd_1.0, lora_kd_0.5, nsp_lora_0.05, nsp_lora_0.00, sgp_lora_t1.0_p1.0, sgp_lora_t2.0_p2.0)"