#!/usr/bin/env bash
set -euo pipefail

echo "Starting all main experiments..."

# 运行基础LoRA实验
echo "=========================================="
echo "Running Basic LoRA Experiments"
echo "=========================================="
bash sh/main_experiments_basic_lora.sh

# 运行LoRA + 蒸馏实验
echo "=========================================="
echo "Running LoRA + KD Experiments"
echo "=========================================="
bash sh/main_experiments_lora_kd.sh

# 运行LoRA-NSP实验
echo "=========================================="
echo "Running LoRA-NSP Experiments"
echo "=========================================="
bash sh/main_experiments_nsp_lora.sh

# 运行完整方法实验
echo "=========================================="
echo "Running Full Method Experiments"
echo "=========================================="
bash sh/main_experiments_full_method.sh

echo "=========================================="
echo "All main experiments completed!"
echo "=========================================="