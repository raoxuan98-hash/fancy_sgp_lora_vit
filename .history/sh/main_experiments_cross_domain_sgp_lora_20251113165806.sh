#!/usr/bin/env bash
set -euo pipefail

# 跨域实验数据集列表
DATASETS=("cross_domain_elevater")
SEEDS=(1993 1996 1997)

# 可用的GPU设备
GPUS=(0 1 2 4)

# LoRA-SGP参数
LORA_TYPE="sgp_lora"
VIT_TYPE="vit-b-p16-mocov3"
GAMMA_KD=0.0  # 不使用蒸馏
WEIGHT_TEMP=2.0
WEIGHT_KIND="log1p"
WEIGHT_P=1.0

# 跨域实验特定参数
CROSS_DOMAIN=True
NUM_SHOTS=16

# 创建日志目录
LOG_DIR="logs/cross_domain_sgp_lora_$(date +%Y%m%d_%H%M%S)"
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

DATASET="$DATASET"
GPU="$GPU"
LOG_DIR="$LOG_DIR"
LORA_TYPE="$LORA_TYPE"
VIT_TYPE="$VIT_TYPE"
GAMMA_KD="$GAMMA_KD"
WEIGHT_TEMP="$WEIGHT_TEMP"
WEIGHT_KIND="$WEIGHT_KIND"
WEIGHT_P="$WEIGHT_P"
CROSS_DOMAIN="$CROSS_DOMAIN"
NUM_SHOTS="$NUM_SHOTS"
SEEDS=(${SEEDS[*]})

for SEED in "\${SEEDS[@]}"; do
    echo "Running Cross-Domain LoRA-SGP experiment: dataset=\$DATASET, seed=\$SEED, GPU=\$GPU"
    
    CUDA_VISIBLE_DEVICES=\$GPU python -u main.py \\
        --dataset "\$DATASET" \\
        --smart_defaults \\
        --lora_type "\$LORA_TYPE" \\
        --vit_type "\$VIT_TYPE" \\
        --gamma_kd "\$GAMMA_KD" \\
        --weight_temp "\$WEIGHT_TEMP" \\
        --weight_kind "\$WEIGHT_KIND" \\
        --weight_p "\$WEIGHT_P" \\
        --cross_domain "\$CROSS_DOMAIN" \\
        --num_shots "\$NUM_SHOTS" \\
        --seed_list "\$SEED" \\
        2>&1 | tee "\$LOG_DIR/\${DATASET}_seed\${SEED}.log"
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

echo "Cross-Domain LoRA-SGP experiments completed. Logs saved to $LOG_DIR"