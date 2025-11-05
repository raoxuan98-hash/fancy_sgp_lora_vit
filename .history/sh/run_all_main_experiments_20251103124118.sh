#!/usr/bin/env bash
set -euo pipefail

echo "Starting all main experiments with sequential execution..."

# 创建总日志目录
MASTER_LOG_DIR="logs/all_experiments_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$MASTER_LOG_DIR"

# 顺序运行所有实验类型

# 运行基础LoRA实验
echo "=========================================="
echo "Running Basic LoRA Experiments"
echo "=========================================="
bash sh/main_experiments_basic_lora.sh 2>&1 | tee "$MASTER_LOG_DIR/basic_lora.log"
echo "Basic LoRA experiments completed."

# 运行LoRA + 蒸馏实验
echo "=========================================="
echo "Running LoRA + KD Experiments"
echo "=========================================="
bash sh/main_experiments_lora_kd.sh 2>&1 | tee "$MASTER_LOG_DIR/lora_kd.log"
echo "LoRA + KD experiments completed."

# 运行LoRA-NSP实验
echo "=========================================="
echo "Running LoRA-NSP Experiments"
echo "=========================================="
bash sh/main_experiments_nsp_lora.sh 2>&1 | tee "$MASTER_LOG_DIR/nsp_lora.log"
echo "LoRA-NSP experiments completed."

echo "=========================================="
echo "All main experiments completed!"
echo "Logs saved to: $MASTER_LOG_DIR"
echo "=========================================="