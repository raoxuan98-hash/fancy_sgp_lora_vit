#!/usr/bin/env bash
set -euo pipefail

echo "Starting Cross-Domain Experiments with Incremental Split..."

# 创建总日志目录
MASTER_LOG_DIR="logs/cross_domain_experiments_incremental_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$MASTER_LOG_DIR"

# 跨域实验数据集
DATASET="cross_domain_elevater"
SEEDS=(1993)

# GPU分配 - 每个方法使用一个GPU，并行运行
GPUS=(0 1 2 4 5)

# 运行单个实验类型的函数
run_experiment_type() {
    local experiment_name="$1"
    local lora_type="$2"
    local gpu_id="$3"
    shift 3
    local additional_params=("$@")
    
    echo "=========================================="
    echo "Running $experiment_name Cross-Domain Experiments on GPU $gpu_id"
    echo "=========================================="
