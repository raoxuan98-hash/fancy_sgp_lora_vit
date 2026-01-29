#!/bin/bash

# 性能曲面等高线图多GPU并行训练脚本
# 支持不同的模型和rank配置

# 设置日志目录
LOG_DIR="./logs/exp1_performance_surface"
mkdir -p $LOG_DIR

# 定义实验配置：模型名称、rank值、GPU编号
declare -a EXPERIMENTS=(
    # 格式: "模型名称:rank值:GPU编号:实验名称"
    "vit-b-p16:1:0:vit-b-p16-rank1"
    "vit-b-p16:8:1:vit-b-p16-rank8"
    "vit-b-p16:32:2:vit-b-p16-rank32"
    "vit-b-p16-clip:1:0:vit-b-p16-clip-rank1"
    "vit-b-p16-clip:8:1:vit-b-p16-clip-rank8"
    "vit-b-p16-clip:32:2:vit-b-p16-clip-rank32"
    "vit-b-p16-dino:1:0:vit-b-p16-dino-rank1"
    "vit-b-p16-dino:8:1:vit-b-p16-dino-rank8"
    "vit-b-p16-dino:32:2:vit-b-p16-dino-rank32"
    "vit-b-p16-mocov3:1:0:vit-b-p16-mocov3-rank1"
    "vit-b-p16-mocov3:8:1:vit-b-p16-mocov3-rank8"
    "vit-b-p16-mocov3:32:2:vit-b-p16-mocov3-rank32"
)

# 创建后台进程数组
declare -a PIDS=()

echo "启动性能曲面等高线图实验..."
echo "总实验数量: ${#EXPERIMENTS[@]}"
echo "使用GPU: 0, 1, 2, 3"
echo "=================================="

# 启动所有实验
for exp in "${EXPERIMENTS[@]}"; do
    # 解析实验配置
    IFS=':' read -ra CONFIG <<< "$exp"
    MODEL_NAME="${CONFIG[0]}"
    RANK="${CONFIG[1]}"
    GPU_ID="${CONFIG[2]}"
    EXP_NAME="${CONFIG[3]}"
    
    echo "启动实验: $EXP_NAME"
    echo "  模型: $MODEL_NAME"
    echo "  Rank: $RANK"
    echo "  GPU: $GPU_ID"
    
    # 启动实验进程
    LOG_FILE="$LOG_DIR/${EXP_NAME}.log"
    CUDA_VISIBLE_DEVICES=$GPU_ID nohup python classifier_ablation/experiments/exp1_performance_surface.py \
        --model "$MODEL_NAME" \
        --rank "$RANK" \
        --gpu "$GPU_ID" \
        --iterations 0 \
        --num_shots 128 \
        > "$LOG_FILE" 2>&1 &
    
    # 记录进程ID
    PID=$!
    PIDS+=($PID)
    
    echo "  进程ID: $PID"
    echo "  日志文件: $LOG_FILE"
    echo ""
    
    # 给系统一点时间启动进程
    sleep 2
done

echo "所有实验进程已启动..."
echo "进程ID列表: ${PIDS[@]}"
echo "=================================="

# 监控进程状态
monitor_experiments() {
    local active_pids=()
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            active_pids+=($pid)
        else
            echo "进程 $pid 已完成"
        fi
    done
    
    if [ ${#active_pids[@]} -eq 0 ]; then
        echo "所有实验已完成!"
        return 1
    else
        echo "运行中的实验: ${#active_pids[@]}/${#PIDS[@]}"
        PIDS=("${active_pids[@]}")
        return 0
    fi
}

# 定期检查进程状态
check_interval=30  # 30秒检查一次
while true; do
    if ! monitor_experiments; then
        break
    fi
    echo "等待 ${check_interval} 秒后再次检查..."
    sleep $check_interval
done

echo "=================================="
echo "所有实验已完成!"
echo "查看日志文件: $LOG_DIR"
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