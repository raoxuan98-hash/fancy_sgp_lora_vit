#!/usr/bin/env bash
set -euo pipefail

# 数据集列表
DATASETS=("cifar100_224" "imagenet-r" "cub200_224" "cars196_224")
# SEEDS=(1993 1996 1997)
SEEDS=(1993)

# 可用的GPU设备
GPUS=(0 1 2 4)

# 完整方法参数
LORA_TYPE="sgp_lora"
VIT_TYPE="vit-b-p16-mocov3"
GAMMA_KD=0.0  # 不使用蒸馏
WEIGHT_KIND="log1p"

# 参数网格测试
WEIGHT_TEMP_VALUES=(2.0)
WEIGHT_P_VALUES=(2.0)

# 创建日志目录
LOG_DIR="logs/full_method_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# 并行运行实验 - 每个GPU一次运行一个数据集的所有参数组合
PIDS=()

# 为每个数据集分配一个GPU，并行运行
for i in "${!DATASETS[@]}"; do
    DATASET="${DATASETS[$i]}"
    GPU="${GPUS[$i]}"
    
    # 为每个数据集创建一个子脚本，包含所有参数组合
    cat > "$LOG_DIR/run_${DATASET}_all_params.sh" << EOF
#!/usr/bin/env bash
set -euo pipefail

DATASET="$DATASET"
GPU="$GPU"
LOG_DIR="$LOG_DIR"
LORA_TYPE="$LORA_TYPE"
VIT_TYPE="$VIT_TYPE"
GAMMA_KD="$GAMMA_KD"
WEIGHT_KIND="$WEIGHT_KIND"
SEEDS=(${SEEDS[*]})
WEIGHT_TEMP_VALUES=(${WEIGHT_TEMP_VALUES[*]})
WEIGHT_P_VALUES=(${WEIGHT_P_VALUES[*]})

echo "Starting all experiments for $DATASET on GPU $GPU"
echo "Parameter combinations: ${#WEIGHT_TEMP_VALUES[@]} weight_temp × ${#WEIGHT_P_VALUES[@]} weight_p × ${#SEEDS[@]} seeds"

# 顺序运行所有参数组合
for WEIGHT_TEMP in "\${WEIGHT_TEMP_VALUES[@]}"; do
    for WEIGHT_P in "\${WEIGHT_P_VALUES[@]}"; do
        for SEED in "\${SEEDS[@]}"; do
            echo "Running Full Method experiment: dataset=\$DATASET, seed=\$SEED, weight_temp=\$WEIGHT_TEMP, weight_p=\$WEIGHT_P, GPU=\$GPU"
            
            CUDA_VISIBLE_DEVICES=\$GPU python -u main.py \\
                --dataset "\$DATASET" \\
                --smart_defaults \\
                --lora_type "\$LORA_TYPE" \\
                --vit_type "\$VIT_TYPE" \\
                --gamma_kd "\$GAMMA_KD" \\
                --weight_temp "\$WEIGHT_TEMP" \\
                --weight_p "\$WEIGHT_P" \\
                --weight_kind "\$WEIGHT_KIND" \\
                --seed_list "\$SEED" \\
                2>&1 | tee "\$LOG_DIR/\${DATASET}_temp\${WEIGHT_TEMP}_p\${WEIGHT_P}_seed\${SEED}.log"
        done
    done
done

echo "All experiments completed for $DATASET on GPU $GPU"
EOF
    
    chmod +x "$LOG_DIR/run_${DATASET}_all_params.sh"
    
    # 在后台运行每个数据集的所有实验
    echo "Starting experiments for $DATASET on GPU $GPU"
    "$LOG_DIR/run_${DATASET}_all_params.sh" &
    PIDS+=($!)
done

# 等待所有数据集的实验完成
echo "Waiting for all dataset experiments to complete..."
for PID in "${PIDS[@]}"; do
    wait $PID
done

# 计算总实验数量
TOTAL_COMBINATIONS=$((${#WEIGHT_TEMP_VALUES[@]} * ${#WEIGHT_P_VALUES[@]} * ${#DATASETS[@]} * ${#SEEDS[@]}))
echo "Full Method experiments completed."
echo "Total experiments run: $TOTAL_COMBINATIONS"
echo "Parameter combinations: ${#WEIGHT_TEMP_VALUES[@]} weight_temp × ${#WEIGHT_P_VALUES[@]} weight_p × ${#DATASETS[@]} datasets × ${#SEEDS[@]} seeds"
echo "Logs saved to: $LOG_DIR"
