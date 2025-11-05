#!/usr/bin/env bash
set -euo pipefail

# 数据集列表
DATASETS=("cifar100_224" "imagenet-r" "cub200_224" "cars196_224")
SEEDS=(1993 1996 1997)

# 可用的GPU设备
GPUS=(0 1 2 4)

# LoRA + 蒸馏参数
LORA_TYPE="basic_lora"
VIT_TYPE="vit-b-p16-mocov3"
GAMMA_KD=1.0  # 使用蒸馏
UPDATE_TEACHER_EACH_TASK=True
DISTILLATION_TRANSFORM="identity"
KD_TYPE="feat"

# 创建日志目录
LOG_DIR="logs/lora_kd_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# 并行运行所有实验
PIDS=()
for i in "${!DATASETS[@]}"; do
    DATASET="${DATASETS[$i]}"
    GPU="${GPUS[$((i % ${#GPUS[@]}))]}"
    
    # 为每个数据集创建一个子脚本来处理所有种子
    cat > "$LOG_DIR/run_${DATASET}.sh" << EOF
#!/usr/bin/env bash
set -euo pipefail

for SEED in "${SEEDS[@]}"; do
    echo "Running LoRA + KD experiment: dataset=$DATASET, seed=$SEED, GPU=$GPU"
    
    CUDA_VISIBLE_DEVICES=$GPU python -u main.py \\
        --dataset "$DATASET" \\
        --smart_defaults \\
        --lora_type "$LORA_TYPE" \\
        --vit_type "$VIT_TYPE" \\
        --gamma_kd "$GAMMA_KD" \\
        --update_teacher_each_task "$UPDATE_TEACHER_EACH_TASK" \\
        --distillation_transform "$DISTILLATION_TRANSFORM" \\
        --kd_type "$KD_TYPE" \\
        --seed_list "$SEED" \\
        2>&1 | tee "$LOG_DIR/${DATASET}_seed${SEED}.log"
done
EOF
    
    chmod +x "$LOG_DIR/run_${DATASET}.sh"
    
    # 在后台运行每个数据集的实验
    echo "Starting experiments for $DATASET on GPU $GPU"
    "$LOG_DIR/run_${DATASET}.sh" &
    PIDS+=($!)
done

# 等待所有实验完成
echo "Waiting for all experiments to complete..."
for PID in "${PIDS[@]}"; do
    wait $PID
done

echo "LoRA + KD experiments completed. Logs saved to $LOG_DIR"