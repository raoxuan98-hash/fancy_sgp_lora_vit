# Run All Main Experiments 脚本设计文档

## 概述

本文档描述了`run_all_main_experiments.sh`脚本的详细设计，该脚本用于并行执行所有主要实验，确保每个数据集占用一个GPU，每个GPU只运行一个进程。

## 设计要求

1. **并行执行策略**：
   - 每个数据集分配一个专用GPU
   - 每个GPU上只运行一个进程（避免资源竞争）
   - 不同实验类型（basic_lora, lora_kd, nsp_lora, sgp_lora）按顺序执行

2. **GPU分配方案**：
   - 使用4个GPU：0, 1, 2, 4
   - 数据集分配：
     - GPU 0: cifar100_224
     - GPU 1: imagenet-r
     - GPU 2: cub200_224
     - GPU 4: cars196_224

3. **实验参数配置**：
   - 根据实验方案文档中的参数设置
   - SGP-LoRA使用weight_temp=1.0, weight_p=1.0（根据网格搜索结果）
   - 每个实验运行3次（种子：1993, 1996, 1997）

## 脚本结构

### 1. 初始化部分
```bash
#!/usr/bin/env bash
set -euo pipefail

# 创建总日志目录
MASTER_LOG_DIR="logs/all_experiments_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$MASTER_LOG_DIR"

# 数据集列表
DATASETS=("cifar100_224" "imagenet-r" "cub200_224" "cars196_224")
SEEDS=(1993 1996 1997)

# GPU分配
GPUS=(0 1 2 4)
```

### 2. 通用函数
```bash
# 运行单个实验类型的函数
run_experiment_type() {
    local experiment_name="$1"
    local lora_type="$2"
    local additional_params=("${@:3}")
    
    echo "=========================================="
    echo "Running $experiment_name Experiments"
    echo "=========================================="
    
    # 创建实验类型特定的日志目录
    LOG_DIR="$MASTER_LOG_DIR/${experiment_name}_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$LOG_DIR"
    
    # 并行运行所有数据集
    PIDS=()
    
    for i in "${!DATASETS[@]}"; do
        DATASET="${DATASETS[$i]}"
        GPU="${GPUS[$i]}"
        
        # 为每个数据集创建子脚本
        cat > "$LOG_DIR/run_${DATASET}.sh" << EOF
#!/usr/bin/env bash
set -euo pipefail

DATASET="$DATASET"
GPU="$GPU"
LOG_DIR="$LOG_DIR"
LORA_TYPE="$lora_type"
VIT_TYPE="vit-b-p16-mocov3"
SEEDS=(${SEEDS[*]})
ADDITIONAL_PARAMS=(${additional_params[*]})

echo "Starting $experiment_name experiments for $DATASET on GPU $GPU"

for SEED in "\${SEEDS[@]}"; do
    echo "Running $experiment_name: dataset=\$DATASET, seed=\$SEED, GPU=\$GPU"
    
    CUDA_VISIBLE_DEVICES=\$GPU python -u main.py \\
        --dataset "\$DATASET" \\
        --smart_defaults \\
        --lora_type "\$LORA_TYPE" \\
        --vit_type "\$VIT_TYPE" \\
        \${ADDITIONAL_PARAMS[@]} \\
        --seed_list "\$SEED" \\
        2>&1 | tee "\$LOG_DIR/\${DATASET}_seed\${SEED}.log"
done

echo "All $experiment_name experiments completed for $DATASET on GPU $GPU"
EOF
        
        chmod +x "$LOG_DIR/run_${DATASET}.sh"
        
        # 在后台运行每个数据集的实验
        echo "Starting $experiment_name experiments for $DATASET on GPU $GPU"
        "$LOG_DIR/run_${DATASET}.sh" &
        PIDS+=($!)
    done
    
    # 等待所有实验完成
    echo "Waiting for all $experiment_name experiments to complete..."
    for PID in "${PIDS[@]}"; do
        wait $PID
    done
    
    echo "$experiment_name experiments completed. Logs saved to $LOG_DIR"
}
```

### 3. 实验执行部分
```bash
# 1. 运行基础LoRA实验
run_experiment_type "basic_lora" "basic_lora" "--gamma_kd" "0.0"

# 2. 运行LoRA + 蒸馏实验
run_experiment_type "lora_kd" "basic_lora" \
    "--gamma_kd" "1.0" \
    "--update_teacher_each_task" "True" \
    "--distillation_transform" "identity" \
    "--kd_type" "feat"

# 3. 运行LoRA-NSP实验
run_experiment_type "nsp_lora" "nsp_lora" \
    "--gamma_kd" "0.0" \
    "--nsp_weight" "0.05" \
    "--nsp_eps" "0.05"

# 4. 运行完整方法实验
run_experiment_type "sgp_lora" "sgp_lora" \
    "--gamma_kd" "0.0" \
    "--weight_temp" "1.0" \
    "--weight_p" "1.0" \
    "--weight_kind" "log1p"
```

### 4. 完成部分
```bash
echo "=========================================="
echo "All main experiments completed!"
echo "Logs saved to: $MASTER_LOG_DIR"
echo "=========================================="

# 计算总实验数量
TOTAL_EXPERIMENTS=$((${#DATASETS[@]} * ${#SEEDS[@]} * 4))  # 4种实验类型
echo "Total experiments run: $TOTAL_EXPERIMENTS"
echo "Experiments per type: ${#DATASETS[@]} datasets × ${#SEEDS[@]} seeds = $((${#DATASETS[@]} * ${#SEEDS[@]}))"
```

## 关键特性

1. **GPU隔离**：每个数据集使用独立的GPU，避免资源竞争
2. **顺序执行**：不同实验类型按顺序执行，确保GPU资源不被同时占用
3. **并行处理**：同一实验类型内的不同数据集并行运行
4. **完整日志**：每个实验类型、数据集和种子组合都有独立的日志文件
5. **错误处理**：使用`set -euo pipefail`确保脚本在出错时立即停止
6. **进度跟踪**：实时显示实验进度和完成状态

## 使用方法

```bash
bash sh/run_all_main_experiments.sh
```

## 预期输出

脚本将创建以下目录结构：
```
logs/all_experiments_YYYYMMDD_HHMMSS/
├── basic_lora_YYYYMMDD_HHMMSS/
│   ├── cifar100_224_seed1993.log
│   ├── cifar100_224_seed1996.log
│   ├── cifar100_224_seed1997.log
│   ├── imagenet-r_seed1993.log
│   └── ...
├── lora_kd_YYYYMMDD_HHMMSS/
│   └── ...
├── nsp_lora_YYYYMMDD_HHMMSS/
│   └── ...
└── sgp_lora_YYYYMMDD_HHMMSS/
    └── ...
```

## 注意事项

1. 确保所有4个GPU（0, 1, 2, 4）都可用
2. 确保有足够的磁盘空间存储日志和实验结果
3. 脚本运行时间较长，建议在screen或tmux中运行
4. 如果某个实验失败，整个脚本将停止，需要检查日志并修复问题后重新运行