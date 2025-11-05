#!/usr/bin/env bash
set -euo pipefail

echo "Starting all main experiments with parallel execution..."

# 创建总日志目录
MASTER_LOG_DIR="logs/all_experiments_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$MASTER_LOG_DIR"

# 并行运行所有实验类型
PIDS=()

# 运行基础LoRA实验
echo "=========================================="
echo "Starting Basic LoRA Experiments in background"
echo "=========================================="
bash sh/main_experiments_basic_lora.sh > "$MASTER_LOG_DIR/basic_lora.log" 2>&1 &
PIDS+=($!)
echo "Basic LoRA experiments started with PID $!"

# 运行LoRA + 蒸馏实验
echo "=========================================="
echo "Starting LoRA + KD Experiments in background"
echo "=========================================="
bash sh/main_experiments_lora_kd.sh > "$MASTER_LOG_DIR/lora_kd.log" 2>&1 &
PIDS+=($!)
echo "LoRA + KD experiments started with PID $!"

# 运行LoRA-NSP实验
echo "=========================================="
echo "Starting LoRA-NSP Experiments in background"
echo "=========================================="
bash sh/main_experiments_nsp_lora.sh > "$MASTER_LOG_DIR/nsp_lora.log" 2>&1 &
PIDS+=($!)
echo "LoRA-NSP experiments started with PID $!"

# 运行完整方法实验
echo "=========================================="
echo "Starting Full Method Experiments in background"
echo "=========================================="
bash sh/main_experiments_full_method.sh > "$MASTER_LOG_DIR/full_method.log" 2>&1 &
PIDS+=($!)
echo "Full Method experiments started with PID $!"

echo "=========================================="
echo "All experiment types started in parallel."
echo "Waiting for all experiments to complete..."
echo "=========================================="

# 等待所有实验完成
for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    echo "Waiting for experiment with PID $PID to complete..."
    wait $PID
    echo "Experiment with PID $PID completed."
done

echo "=========================================="
echo "All main experiments completed!"
echo "Logs saved to: $MASTER_LOG_DIR"
echo "=========================================="