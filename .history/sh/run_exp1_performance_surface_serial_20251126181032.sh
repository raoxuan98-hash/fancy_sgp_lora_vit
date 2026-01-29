#!/bin/bash

# 性能曲面等高线图串行训练脚本
# 支持不同的模型和rank配置，串行运行不同秩的实验

# 设置日志目录
LOG_DIR="./logs/exp1_performance_surface"
mkdir -p $LOG_DIR

# 定义实验配置：模型名称、rank值、GPU编号
declare -a EXPERIMENTS=(
    # 格式: "模型名称:rank值:GPU编号:实验名称"
    # 每个架构使用不同的GPU
    # vit-b-p16 使用 GPU 0
    "vit-b-p16:1:0:vit-b-p16-rank1"
    "vit-b-p16:8:0:vit-b-p16-rank8"
    "vit-b-p16:32:0:vit-b-p16-rank32"
    # vit-b-p16-clip 使用 GPU 1
    "vit-b-p16-clip:1:1:vit-b-p16-clip-rank1"
    "vit-b-p16-clip:8:1:vit-b-p16-clip-rank8"
    "vit-b-p16-clip:32:1:vit-b-p16-clip-rank32"
    # vit-b-p16-dino 使用 GPU 2
    "vit-b-p16-dino:1:2:vit-b-p16-dino-rank1"
    "vit-b-p16-dino:8:2:vit-b-p16-dino-rank8"
    "vit-b-p16-dino:32:2:vit-b-p16-dino-rank32"
    # vit-b-p16-mocov3 使用 GPU 4
    "vit-b-p16-mocov3:1:4:vit-b-p16-mocov3-rank1"
    "vit-b-p16-mocov3:8:4:vit-b-p16-mocov3-rank8"
    "vit-b-p16-mocov3:32:4:vit-b-p16-mocov3-rank32"
)

echo "启动性能曲面等高线图实验（串行运行）..."
echo "总实验数量: ${#EXPERIMENTS[@]}"
echo "使用GPU: 0, 1, 2, 4"
echo "架构分配:"
echo "  vit-b-p16: GPU 0"
echo "  vit-b-p16-clip: GPU 1"
echo "  vit-b-p16-dino: GPU 2"
echo "  vit-b-p16-mocov3: GPU 4"
echo "=================================="

# 记录开始时间
start_time=$(date)
echo "实验开始时间: $start_time"
echo ""

# 串行运行所有实验
total_experiments=${#EXPERIMENTS[@]}
completed_experiments=0

for exp in "${EXPERIMENTS[@]}"; do
    # 解析实验配置
    IFS=':' read -ra CONFIG <<< "$exp"
    MODEL_NAME="${CONFIG[0]}"
    RANK="${CONFIG[1]}"
    GPU_ID="${CONFIG[2]}"
    EXP_NAME="${CONFIG[3]}"
    
    completed_experiments=$((completed_experiments + 1))
    
    echo "=================================="
    echo "启动实验 [$completed_experiments/$total_experiments]: $EXP_NAME"
    echo "  模型: $MODEL_NAME"
    echo "  Rank: $RANK"
    echo "  GPU: $GPU_ID"
    echo "  开始时间: $(date)"
    echo "=================================="
    
    # 运行实验（串行执行，不使用后台进程）
    LOG_FILE="$LOG_DIR/${EXP_NAME}.log"
    CUDA_VISIBLE_DEVICES=$GPU_ID python classifier_ablation/experiments/exp1_performance_surface.py \
        --model "$MODEL_NAME" \
        --rank "$RANK" \
        --gpu "$GPU_ID" \
        --iterations 0 \
        --num_shots 128 \
        > "$LOG_FILE" 2>&1
    
    # 检查实验是否成功完成
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "✅ 实验 $EXP_NAME 成功完成"
        # 提取最佳性能
        if grep -q "最佳参数:" "$LOG_FILE"; then
            echo "最佳性能: $(grep "最佳参数:" "$LOG_FILE" | tail -1)"
        fi
    else
        echo "❌ 实验 $EXP_NAME 失败，退出码: $exit_code"
        echo "查看日志文件: $LOG_FILE"
    fi
    
    echo "  结束时间: $(date)"
    echo ""
done

# 记录结束时间
end_time=$(date)
echo "=================================="
echo "所有实验已完成!"
echo "实验开始时间: $start_time"
echo "实验结束时间: $end_time"
echo "=================================="

# 汇总结果
echo "实验结果汇总:"
for exp in "${EXPERIMENTS[@]}"; do
    IFS=':' read -ra CONFIG <<< "$exp"
    EXP_NAME="${CONFIG[3]}"
    LOG_FILE="$LOG_DIR/${EXP_NAME}.log"
    
    if [ -f "$LOG_FILE" ]; then
        echo "--- $EXP_NAME ---"
        # 检查是否包含实验完成的标记
        if grep -q "实验完成" "$LOG_FILE"; then
            echo "状态: ✅ 完成"
            # 提取最佳性能
            if grep -q "最佳参数:" "$LOG_FILE"; then
                grep "最佳参数:" "$LOG_FILE" | tail -1
            fi
        else
            echo "状态: ❓ 未知"
        fi
        echo ""
    fi
done

echo "所有性能曲面等高线图实验完成!"