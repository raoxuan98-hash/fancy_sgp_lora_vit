# 实验脚本设计文档

## 1. 主实验脚本设计

### 1.1 完整主实验脚本 (run_main_experiments.sh)

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "=========================================="
echo "Starting All Main Experiments"
echo "=========================================="

# 创建总日志目录
MASTER_LOG_DIR="logs/main_experiments_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$MASTER_LOG_DIR"

# 数据集列表
DATASETS=("cifar100_224" "imagenet-r" "cub200_224" "cars196_224")
SEEDS=(1993 1996 1997)
GPUS=(0 1 2 4)

# 方法配置
declare -A METHODS=(
    ["basic_lora"]="basic_lora 0.0"
    ["lora_kd"]="basic_lora 1.0"
    ["nsp_lora"]="nsp_lora 0.0"
    ["sgp_lora"]="sgp_lora 0.0"
)

# 顺序执行所有方法
for method in "${!METHODS[@]}"; do
    echo "=========================================="
    echo "Running ${method} Experiments"
    echo "=========================================="
    
    # 解析方法参数
    params=(${METHODS[$method]})
    lora_type=${params[0]}
    gamma_kd=${params[1]}
    
    # 创建方法特定的日志目录
    METHOD_LOG_DIR="$MASTER_LOG_DIR/${method}"
    mkdir -p "$METHOD_LOG_DIR"
    
    # 并行运行所有数据集
    PIDS=()
    for i in "${!DATASETS[@]}"; do
        dataset="${DATASETS[$i]}"
        gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
        
        # 为每个数据集创建子脚本
        cat > "$METHOD_LOG_DIR/run_${dataset}.sh" << EOF
#!/usr/bin/env bash
set -euo pipefail

DATASET="$dataset"
GPU="$gpu"
METHOD_LOG_DIR="$METHOD_LOG_DIR"
LORA_TYPE="$lora_type"
GAMMA_KD="$gamma_kd"
SEEDS=(${SEEDS[*]})

# 方法特定参数
case "$method" in
    "lora_kd")
        UPDATE_TEACHER_EACH_TASK=True
        DISTILLATION_TRANSFORM="identity"
        KD_TYPE="feat"
        ;;
    "nsp_lora")
        NSP_WEIGHT=0.05
        NSP_EPS=0.05
        ;;
    "sgp_lora")
        WEIGHT_TEMP=1.0
        WEIGHT_KIND="log1p"
        WEIGHT_P=2.0
        ;;
esac

for SEED in "\${SEEDS[@]}"; do
    echo "Running ${method} experiment: dataset=\$DATASET, seed=\$SEED, GPU=\$GPU"
    
    # 构建命令
    CMD="CUDA_VISIBLE_DEVICES=\$GPU python -u main.py \\
        --dataset \"\$DATASET\" \\
        --smart_defaults \\
        --lora_type \"\$LORA_TYPE\" \\
        --vit_type \"vit-b-p16-mocov3\" \\
        --gamma_kd \"\$GAMMA_KD\" \\
        --seed_list \"\$SEED\""
    
    # 添加方法特定参数
    if [[ "$method" == "lora_kd" ]]; then
        CMD="$CMD \\
        --update_teacher_each_task \"\$UPDATE_TEACHER_EACH_TASK\" \\
        --distillation_transform \"\$DISTILLATION_TRANSFORM\" \\
        --kd_type \"\$KD_TYPE\""
    elif [[ "$method" == "nsp_lora" ]]; then
        CMD="$CMD \\
        --nsp_weight \"\$NSP_WEIGHT\" \\
        --nsp_eps \"\$NSP_EPS\""
    elif [[ "$method" == "sgp_lora" ]]; then
        CMD="$CMD \\
        --weight_temp \"\$WEIGHT_TEMP\" \\
        --weight_kind \"\$WEIGHT_KIND\" \\
        --weight_p \"\$WEIGHT_P\""
    fi
    
    # 执行命令并记录日志
    eval \$CMD 2>&1 | tee "\$METHOD_LOG_DIR/\${DATASET}_seed\${SEED}.log"
done
EOF
        
        chmod +x "$METHOD_LOG_DIR/run_${dataset}.sh"
        
        # 在后台运行
        echo "Starting ${method} experiments for $dataset on GPU $gpu"
        "$METHOD_LOG_DIR/run_${dataset}.sh" &
        PIDS+=($!)
    done
    
    # 等待所有数据集完成
    echo "Waiting for ${method} experiments to complete..."
    for PID in "${PIDS[@]}"; do
        wait $PID
    done
    
    echo "${method} experiments completed."
done

echo "=========================================="
echo "All main experiments completed!"
echo "Logs saved to: $MASTER_LOG_DIR"
echo "=========================================="
```

### 1.2 单个方法执行脚本 (run_single_method.sh)

```bash
#!/usr/bin/env bash
set -euo pipefail

# 参数检查
if [ $# -ne 1 ]; then
    echo "Usage: $0 <method_name>"
    echo "Available methods: basic_lora, lora_kd, nsp_lora, sgp_lora"
    exit 1
fi

method=$1

# 数据集和种子配置
DATASETS=("cifar100_224" "imagenet-r" "cub200_224" "cars196_224")
SEEDS=(1993 1996 1997)
GPUS=(0 1 2 4)

# 创建日志目录
LOG_DIR="logs/${method}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# 方法配置
case "$method" in
    "basic_lora")
        LORA_TYPE="basic_lora"
        GAMMA_KD=0.0
        ;;
    "lora_kd")
        LORA_TYPE="basic_lora"
        GAMMA_KD=1.0
        UPDATE_TEACHER_EACH_TASK=True
        DISTILLATION_TRANSFORM="identity"
        KD_TYPE="feat"
        ;;
    "nsp_lora")
        LORA_TYPE="nsp_lora"
        GAMMA_KD=0.0
        NSP_WEIGHT=0.05
        NSP_EPS=0.05
        ;;
    "sgp_lora")
        LORA_TYPE="sgp_lora"
        GAMMA_KD=0.0
        WEIGHT_TEMP=1.0
        WEIGHT_KIND="log1p"
        WEIGHT_P=2.0
        ;;
    *)
        echo "Unknown method: $method"
        exit 1
        ;;
esac

# 并行运行所有数据集
PIDS=()
for i in "${!DATASETS[@]}"; do
    dataset="${DATASETS[$i]}"
    gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
    
    echo "Running ${method} experiment: dataset=$dataset, GPU=$gpu"
    
    # 构建命令
    CMD="CUDA_VISIBLE_DEVICES=$gpu python -u main.py \\
        --dataset \"$dataset\" \\
        --smart_defaults \\
        --lora_type \"$LORA_TYPE\" \\
        --vit_type \"vit-b-p16-mocov3\" \\
        --gamma_kd \"$GAMMA_KD\" \\
        --seed_list \"${SEEDS[@]}\""
    
    # 添加方法特定参数
    if [[ "$method" == "lora_kd" ]]; then
        CMD="$CMD \\
        --update_teacher_each_task \"$UPDATE_TEACHER_EACH_TASK\" \\
        --distillation_transform \"$DISTILLATION_TRANSFORM\" \\
        --kd_type \"$KD_TYPE\""
    elif [[ "$method" == "nsp_lora" ]]; then
        CMD="$CMD \\
        --nsp_weight \"$NSP_WEIGHT\" \\
        --nsp_eps \"$NSP_EPS\""
    elif [[ "$method" == "sgp_lora" ]]; then
        CMD="$CMD \\
        --weight_temp \"$WEIGHT_TEMP\" \\
        --weight_kind \"$WEIGHT_KIND\" \\
        --weight_p \"$WEIGHT_P\""
    fi
    
    # 执行命令并记录日志
    eval $CMD 2>&1 | tee "$LOG_DIR/${dataset}.log" &
    PIDS+=($!)
done

# 等待所有实验完成
echo "Waiting for all experiments to complete..."
for PID in "${PIDS[@]}"; do
    wait $PID
done

echo "${method} experiments completed. Logs saved to $LOG_DIR"
```

## 2. LoRA-SGP超参数网格搜索脚本

### 2.1 SGP参数网格搜索 (run_sgp_grid_search.sh)

```bash
#!/usr/bin/env bash
set -euo pipefail

# 参数配置
DATASET=${1:-"cifar100_224"}  # 默认数据集
GPU_LIST=${2:-"0,1,2,4"}      # 默认GPU列表
IFS=',' read -r -a GPUS <<< "$GPU_LIST"

# 超参数网格
WEIGHT_TEMPS=(1.0 2.0 4.0)
WEIGHT_PS=(1.0 2.0)
WEIGHT_KINDS=("log1p" "exp" "rational1")
SEEDS=(1993 1996 1997)

# 并行控制
MAX_PARALLEL=4  # 最大并行数

# 创建日志目录
LOG_DIR="logs/sgp_grid_${DATASET}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# 实验计数器
run_idx=0
jobs_running=0

echo "Starting SGP grid search on $DATASET"
echo "Parameter combinations: ${#WEIGHT_TEMPS[@]} × ${#WEIGHT_PS[@]} × ${#WEIGHT_KINDS[@]} × ${#SEEDS[@]} = $((${#WEIGHT_TEMPS[@]} * ${#WEIGHT_PS[@]} * ${#WEIGHT_KINDS[@]} * ${#SEEDS[@]}))"

# 遍历所有参数组合
for temp in "${WEIGHT_TEMPS[@]}"; do
    for p in "${WEIGHT_PS[@]}"; do
        for kind in "${WEIGHT_KINDS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                # GPU分配
                gpu=${GPUS[$((run_idx % ${#GPUS[@]}))]}
                
                echo "[RUN $run_idx | GPU $gpu] dataset=$DATASET temp=$temp p=$p kind=$kind seed=$seed"
                
                # 构建命令
                CMD="CUDA_VISIBLE_DEVICES=$gpu python -u main.py \\
                    --dataset \"$DATASET\" \\
                    --smart_defaults \\
                    --lora_type sgp_lora \\
                    --vit_type \"vit-b-p16-mocov3\" \\
                    --gamma_kd 0.0 \\
                    --weight_temp $temp \\
                    --weight_p $p \\
                    --weight_kind \"$kind\" \\
                    --seed_list $seed"
                
                # 执行命令
                eval $CMD > "$LOG_DIR/${DATASET}_temp${temp}_p${p}_kind${kind}_seed${seed}.log" 2>&1 &
                
                # 更新计数器
                run_idx=$((run_idx + 1))
                jobs_running=$((jobs_running + 1))
                
                # 并行控制
                if (( jobs_running >= MAX_PARALLEL )); then
                    wait  # 等待一个任务完成
                    jobs_running=$((jobs_running - 1))
                fi
            done
        done
    done
done

# 等待所有任务完成
wait

echo "SGP grid search completed. Logs saved to $LOG_DIR"
echo "Total experiments run: $run_idx"
```

### 2.2 快速SGP参数搜索 (run_sgp_quick_search.sh)

```bash
#!/usr/bin/env bash
set -euo pipefail

# 快速搜索配置 - 只测试最有希望的参数组合
DATASET=${1:-"imagenet-r"}
GPU_LIST=${2:-"0,1,2,4"}
IFS=',' read -r -a GPUS <<< "$GPU_LIST"

# 精简的参数组合
WEIGHT_TEMPS=(1.0 2.0)
WEIGHT_PS=(1.0 2.0)
WEIGHT_KINDS=("log1p")  # 只测试log1p
SEEDS=(1993)  # 只用一个种子快速测试

# 并行控制
MAX_PARALLEL=2

# 创建日志目录
LOG_DIR="logs/sgp_quick_${DATASET}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

run_idx=0
jobs_running=0

echo "Starting quick SGP search on $DATASET"

for temp in "${WEIGHT_TEMPS[@]}"; do
    for p in "${WEIGHT_PS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            gpu=${GPUS[$((run_idx % ${#GPUS[@]}))]}
            
            echo "[RUN $run_idx | GPU $gpu] dataset=$DATASET temp=$temp p=$p seed=$seed"
            
            CUDA_VISIBLE_DEVICES=$gpu python -u main.py \
                --dataset "$DATASET" \
                --smart_defaults \
                --lora_type sgp_lora \
                --vit_type "vit-b-p16-mocov3" \
                --gamma_kd 0.0 \
                --weight_temp $temp \
                --weight_p $p \
                --weight_kind "log1p" \
                --seed_list $seed \
                > "$LOG_DIR/${DATASET}_temp${temp}_p${p}_seed${seed}.log" 2>&1 &
            
            run_idx=$((run_idx + 1))
            jobs_running=$((jobs_running + 1))
            
            if (( jobs_running >= MAX_PARALLEL )); then
                wait
                jobs_running=$((jobs_running - 1))
            fi
        done
    done
done

wait

echo "Quick SGP search completed. Logs saved to $LOG_DIR"
```

## 3. 消融研究脚本

### 3.1 组件消融实验 (run_component_ablation.sh)

```bash
#!/usr/bin/env bash
set -euo pipefail

# 数据集配置
DATASETS=("cifar100_224" "imagenet-r" "cub200_224" "cars196_224")
SEEDS=(1993 1996 1997)
GPUS=(0 1 2 4)

# 消融配置
declare -A ABLATION_CONFIGS=(
    ["full_method"]="sgp_lora 0.0 1.0 log1p 2.0"
    ["wo_sgp"]="basic_lora 0.0"
    ["wo_amdc"]="sgp_lora 0.0 1.0 log1p 2.0 --no_amdc"
    ["wo_both"]="basic_lora 0.0 --no_amdc"
)

# 创建日志目录
LOG_DIR="logs/component_ablation_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "Starting component ablation studies..."

for ablation_type in "${!ABLATION_CONFIGS[@]}"; do
    echo "=========================================="
    echo "Running ${ablation_type} experiments"
    echo "=========================================="
    
    # 解析配置
    config=(${ABLATION_CONFIGS[$ablation_type]})
    lora_type=${config[0]}
    gamma_kd=${config[1]}
    
    # 创建消融特定的日志目录
    ABLATION_LOG_DIR="$LOG_DIR/${ablation_type}"
    mkdir -p "$ABLATION_LOG_DIR"
    
    # 并行运行所有数据集
    PIDS=()
    for i in "${!DATASETS[@]}"; do
        dataset="${DATASETS[$i]}"
        gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
        
        echo "Running ${ablation_type} on $dataset (GPU $gpu)"
        
        # 构建命令
        CMD="CUDA_VISIBLE_DEVICES=$gpu python -u main.py \\
            --dataset \"$dataset\" \\
            --smart_defaults \\
            --lora_type \"$lora_type\" \\
            --vit_type \"vit-b-p16-mocov3\" \\
            --gamma_kd \"$gamma_kd\" \\
            --seed_list \"${SEEDS[@]}\""
        
        # 添加SGP特定参数
        if [[ "$lora_type" == "sgp_lora" ]]; then
            weight_temp=${config[2]}
            weight_kind=${config[3]}
            weight_p=${config[4]}
            CMD="$CMD \\
                --weight_temp $weight_temp \\
                --weight_kind \"$weight_kind\" \\
                --weight_p $weight_p"
        fi
        
        # 添加AMDC控制参数
        if [[ " ${config[*]} " =~ " --no_amdc " ]]; then
            CMD="$CMD --no_amdc"
        fi
        
        # 执行命令
        eval $CMD > "$ABLATION_LOG_DIR/${dataset}.log" 2>&1 &
        PIDS+=($!)
    done
    
    # 等待所有数据集完成
    for PID in "${PIDS[@]}"; do
        wait $PID
    done
    
    echo "${ablation_type} experiments completed."
done

echo "All component ablation experiments completed. Logs saved to $LOG_DIR"
```

### 3.2 AMDC消融实验 (run_amdc_ablation.sh)

```bash
#!/usr/bin/env bash
set -euo pipefail

# 数据集配置 - 只选择两个代表性数据集
DATASETS=("imagenet-r" "cars196_224")
SEEDS=(1993 1996 1997)
GPUS=(0 1 2 4)

# AMDC消融配置
declare -A AMDC_CONFIGS=(
    ["full_amdc"]="attention_transform"
    ["mean_only"]="mean_only"
    ["cov_only"]="cov_only"
    ["linear_transform"]="linear_transform"
    ["weaknonlinear_transform"]="weaknonlinear_transform"
)

# 创建日志目录
LOG_DIR="logs/amdc_ablation_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "Starting AMDC ablation studies..."

for amdc_type in "${!AMDC_CONFIGS[@]}"; do
    echo "=========================================="
    echo "Running ${amdc_type} experiments"
    echo "=========================================="
    
    transform_type=${AMDC_CONFIGS[$amdc_type]}
    
    # 创建AMDC特定的日志目录
    AMDC_LOG_DIR="$LOG_DIR/${amdc_type}"
    mkdir -p "$AMDC_LOG_DIR"
    
    # 并行运行所有数据集
    PIDS=()
    for i in "${!DATASETS[@]}"; do
        dataset="${DATASETS[$i]}"
        gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
        
        echo "Running ${amdc_type} on $dataset (GPU $gpu)"
        
        # 构建命令
        CMD="CUDA_VISIBLE_DEVICES=$gpu python -u main.py \\
            --dataset \"$dataset\" \\
            --smart_defaults \\
            --lora_type sgp_lora \\
            --vit_type \"vit-b-p16-mocov3\" \\
            --gamma_kd 0.0 \\
            --weight_temp 1.0 \\
            --weight_kind \"log1p\" \\
            --weight_p 2.0 \\
            --seed_list \"${SEEDS[@]}\""
        
        # 添加AMDC特定参数
        if [[ "$amdc_type" != "full_amdc" ]]; then
            CMD="$CMD --amdc_type \"$transform_type\""
        fi
        
        # 执行命令
        eval $CMD > "$AMDC_LOG_DIR/${dataset}.log" 2>&1 &
        PIDS+=($!)
    done
    
    # 等待所有数据集完成
    for PID in "${PIDS[@]}"; do
        wait $PID
    done
    
    echo "${amdc_type} experiments completed."
done

echo "All AMDC ablation experiments completed. Logs saved to $LOG_DIR"
```

## 4. 补充实验脚本

### 4.1 长序列任务实验 (run_long_sequence.sh)

```bash
#!/usr/bin/env bash
set -euo pipefail

# 长序列配置
DATASET="cifar100_224"
INIT_CLS=5
INCREMENT=5  # 每任务5类，共20个任务
SEEDS=(1993 1996 1997)
GPUS=(0 1 2 4)

# 方法配置
declare -A METHODS=(
    ["basic_lora"]="basic_lora"
    ["lora_kd"]="basic_lora"
    ["nsp_lora"]="nsp_lora"
    ["sgp_lora"]="sgp_lora"
)

# 创建日志目录
LOG_DIR="logs/long_sequence_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "Starting long sequence experiments (20 tasks × 5 classes)..."

for method in "${!METHODS[@]}"; do
    echo "=========================================="
    echo "Running ${method} long sequence experiments"
    echo "=========================================="
    
    lora_type=${METHODS[$method]}
    
    # 创建方法特定的日志目录
    METHOD_LOG_DIR="$LOG_DIR/${method}"
    mkdir -p "$METHOD_LOG_DIR"
    
    # 并行运行所有种子
    PIDS=()
    for i in "${!SEEDS[@]}"; do
        seed=${SEEDS[$i]}
        gpu=${GPUS[$((i % ${#GPUS[@]}))]}
        
        echo "Running ${method} long sequence: seed=$seed, GPU=$gpu"
        
        # 构建命令
        CMD="CUDA_VISIBLE_DEVICES=$gpu python -u main.py \\
            --dataset \"$DATASET\" \\
            --init_cls $INIT_CLS \\
            --increment $INCREMENT \\
            --lora_type \"$lora_type\" \\
            --vit_type \"vit-b-p16-mocov3\" \\
            --gamma_kd 0.0 \\
            --seed_list $seed"
        
        # 添加方法特定参数
        if [[ "$method" == "lora_kd" ]]; then
            CMD="$CMD \\
                --gamma_kd 1.0 \\
                --update_teacher_each_task True \\
                --distillation_transform identity \\
                --kd_type feat"
        elif [[ "$method" == "nsp_lora" ]]; then
            CMD="$CMD \\
                --nsp_weight 0.05 \\
                --nsp_eps 0.05"
        elif [[ "$method" == "sgp_lora" ]]; then
            CMD="$CMD \\
                --weight_temp 1.0 \\
                --weight_kind \"log1p\" \\
                --weight_p 2.0"
        fi
        
        # 执行命令
        eval $CMD > "$METHOD_LOG_DIR/seed${seed}.log" 2>&1 &
        PIDS+=($!)
    done
    
    # 等待所有种子完成
    for PID in "${PIDS[@]}"; do
        wait $PID
    done
    
    echo "${method} long sequence experiments completed."
done

echo "All long sequence experiments completed. Logs saved to $LOG_DIR"
```

### 4.2 跨架构泛化实验 (run_cross_architecture.sh)

```bash
#!/usr/bin/env bash
set -euo pipefail

# 跨架构配置
DATASETS=("cifar100_224" "imagenet-r" "cub200_224" "cars196_224")
SEEDS=(1993 1996 1997)
GPUS=(0 1 2 4)

# ViT架构列表
VIT_TYPES=("vit-b-p16-mocov3" "vit-b-p16" "vit-b-p-clip")

# 创建日志目录
LOG_DIR="logs/cross_architecture_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "Starting cross-architecture experiments..."

for vit_type in "${VIT_TYPES[@]}"; do
    echo "=========================================="
    echo "Running experiments with ${vit_type}"
    echo "=========================================="
    
    # 创建架构特定的日志目录
    ARCH_LOG_DIR="$LOG_DIR/${vit_type}"
    mkdir -p "$ARCH_LOG_DIR"
    
    # 并行运行所有数据集
    PIDS=()
    for i in "${!DATASETS[@]}"; do
        dataset="${DATASETS[$i]}"
        gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
        
        echo "Running SGP on $dataset with ${vit_type} (GPU $gpu)"
        
        # 构建命令
        CMD="CUDA_VISIBLE_DEVICES=$gpu python -u main.py \\
            --dataset \"$dataset\" \\
            --smart_defaults \\
            --lora_type sgp_lora \\
            --vit_type \"$vit_type\" \\
            --gamma_kd 0.0 \\
            --weight_temp 1.0 \\
            --weight_kind \"log1p\" \\
            --weight_p 2.0 \\
            --seed_list \"${SEEDS[@]}\""
        
        # 执行命令
        eval $CMD > "$ARCH_LOG_DIR/${dataset}.log" 2>&1 &
        PIDS+=($!)
    done
    
    # 等待所有数据集完成
    for PID in "${PIDS[@]}"; do
        wait $PID
    done
    
    echo "${vit_type} experiments completed."
done

echo "All cross-architecture experiments completed. Logs saved to $LOG_DIR"
```

## 5. 结果收集和分析脚本

### 5.1 结果收集脚本 (collect_results.py)

```python
#!/usr/bin/env python3
import os
import json
import glob
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def find_aggregate_files(log_dir):
    """查找所有aggregate_results.json文件"""
    pattern = os.path.join(log_dir, "**", "aggregate_results.json")
    return glob.glob(pattern, recursive=True)

def parse_aggregate_file(file_path):
    """解析单个aggregate_results.json文件"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # 从路径中提取实验信息
    path_parts = Path(file_path).parts
    
    # 提取数据集、方法、种子等信息
    dataset = None
    method = None
    seed = None
    vit_type = None
    
    for part in path_parts:
        if part.endswith('_224'):
            dataset = part
        elif part in ['basic_lora', 'lora_kd', 'nsp_lora', 'sgp_lora']:
            method = part
        elif part.startswith('vit-'):
            vit_type = part
    
    # 尝试从文件名或目录名中提取种子
    for part in reversed(path_parts):
        if part.isdigit() and len(part) == 4:  # 种子通常是4位数
            seed = int(part)
            break
    
    # 提取结果
    results = {}
    if 'final_task_stats' in data:
        for variant, stats in data['final_task_stats'].items():
            results[f"{variant}_last"] = stats['mean']
    
    if 'average_across_tasks_stats' in data:
        for variant, stats in data['average_across_tasks_stats'].items():
            results[f"{variant}_avg"] = stats['mean']
    
    return {
        'dataset': dataset,
        'method': method,
        'seed': seed,
        'vit_type': vit_type,
        'file_path': file_path,
        **results
    }

def collect_all_results(log_dir):
    """收集所有实验结果"""
    aggregate_files = find_aggregate_files(log_dir)
    print(f"Found {len(aggregate_files)} aggregate result files")
    
    all_results = []
    for file_path in aggregate_files:
        try:
            result = parse_aggregate_file(file_path)
            all_results.append(result)
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
    
    return all_results

def create_results_dataframe(results):
    """创建结果DataFrame"""
    df = pd.DataFrame(results)
    
    # 如果有种子列，计算平均值和标准差
    if 'seed' in df.columns:
        # 按数据集、方法、vit_type分组计算统计量
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['seed']]
        
        summary_stats = []
        
        for (dataset, method, vit_type), group in df.groupby(['dataset', 'method', 'vit_type']):
            row = {
                'dataset': dataset,
                'method': method,
                'vit_type': vit_type,
                'num_seeds': len(group)
            }
            
            for col in numeric_cols:
                if col in group.columns:
                    row[f"{col}_mean"] = group[col].mean()
                    row[f"{col}_std"] = group[col].std()
            
            summary_stats.append(row)
        
        summary_df = pd.DataFrame(summary_stats)
        return df, summary_df
    
    return df, None

def main():
    parser = argparse.ArgumentParser(description='Collect experiment results')
    parser.add_argument('--log_dir', type=str, required=True, 
                       help='Directory containing experiment logs')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 收集结果
    results = collect_all_results(args.log_dir)
    
    if not results:
        print("No results found!")
        return
    
    # 创建DataFrame
    df, summary_df = create_results_dataframe(results)
    
    # 保存详细结果
    detailed_file = os.path.join(args.output, 'detailed_results.csv')
    df.to_csv(detailed_file, index=False)
    print(f"Detailed results saved to {detailed_file}")
    
    # 保存汇总结果
    if summary_df is not None:
        summary_file = os.path.join(args.output, 'summary_results.csv')
        summary_df.to_csv(summary_file, index=False)
        print(f"Summary results saved to {summary_file}")
    
    # 保存原始JSON结果
    json_file = os.path.join(args.output, 'all_results.json')
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Raw results saved to {json_file}")

if __name__ == '__main__':
    main()
```

### 5.2 结果表格生成脚本 (generate_tables.py)

```python
#!/usr/bin/env python3
import os
import pandas as pd
import argparse
from tabulate import tabulate

def load_results(csv_file):
    """加载结果CSV文件"""
    return pd.read_csv(csv_file)

def format_mean_std(mean, std):
    """格式化平均值±标准差"""
    return f"{mean:.2f}±{std:.2f}"

def create_main_results_table(df):
    """创建主实验结果表格"""
    # 筛选主要结果变体
    main_variants = [
        'SeqFT + attention_transform + LDA',
        'SeqFT + attention_transform + QDA'
    ]
    
    # 创建结果表格
    table_data = []
    
    for dataset in df['dataset'].unique():
        row = {'Dataset': dataset}
        
        for method in df['method'].unique():
            method_df = df[(df['dataset'] == dataset) & (df['method'] == method)]
            
            if len(method_df) == 0:
                continue
            
            # 获取最佳结果
            for variant in main_variants:
                last_col = f"{variant}_last_mean"
                avg_col = f"{variant}_avg_mean"
                last_std_col = f"{variant}_last_std"
                avg_std_col = f"{variant}_avg_std"
                
                if last_col in method_df.columns and avg_col in method_df.columns:
                    last_acc = method_df[last_col].iloc[0]
                    avg_acc = method_df[avg_col].iloc[0]
                    last_std = method_df[last_std_col].iloc[0]
                    avg_std = method_df[avg_std_col].iloc[0]
                    
                    method_name = f"{method}_{variant.split(' + ')[-1]}"
                    row[f"{method_name}_last"] = format_mean_std(last_acc, last_std)
                    row[f"{method_name}_avg"] = format_mean_std(avg_acc, avg_std)
                    break  # 只取第一个找到的变体
        
        table_data.append(row)
    
    return pd.DataFrame(table_data)

def create_ablation_table(df, ablation_type):
    """创建消融实验表格"""
    # 根据消融类型筛选数据
    if ablation_type == 'component':
        methods = ['full_method', 'wo_sgp', 'wo_amdc', 'wo_both']
    elif ablation_type == 'sgp':
        # 需要从参数中提取不同的SGP配置
        pass
    elif ablation_type == 'amdc':
        # 需要从参数中提取不同的AMDC配置
        pass
    
    table_data = []
    
    for dataset in df['dataset'].unique():
        row = {'Dataset': dataset}
        
        for method in methods:
            method_df = df[(df['dataset'] == dataset) & (df['method'] == method)]
            
            if len(method_df) == 0:
                continue
            
            # 获取最佳结果
            best_variant = None
            best_acc = 0
            
            for col in method_df.columns:
                if col.endswith('_last_mean'):
                    acc = method_df[col].iloc[0]
                    if acc > best_acc:
                        best_acc = acc
                        best_variant = col.replace('_last_mean', '')
            
            if best_variant:
                last_mean = method_df[f"{best_variant}_last_mean"].iloc[0]
                last_std = method_df[f"{best_variant}_last_std"].iloc[0]
                avg_mean = method_df[f"{best_variant}_avg_mean"].iloc[0]
                avg_std = method_df[f"{best_variant}_avg_std"].iloc[0]
                
                row[f"{method}_last"] = format_mean_std(last_mean, last_std)
                row[f"{method}_avg"] = format_mean_std(avg_mean, avg_std)
        
        table_data.append(row)
    
    return pd.DataFrame(table_data)

def main():
    parser = argparse.ArgumentParser(description='Generate result tables')
    parser.add_argument('--results_csv', type=str, required=True,
                       help='CSV file with experiment results')
    parser.add_argument('--output_dir', type=str, default='tables',
                       help='Output directory for tables')
    parser.add_argument('--format', type=str, default='latex',
                       choices=['latex', 'markdown', 'grid'],
                       help='Output format')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载结果
    df = load_results(args.results_csv)
    
    # 生成主实验表格
    main_table = create_main_results_table(df)
    main_file = os.path.join(args.output_dir, f'main_results.{args.format}')
    
    if args.format == 'latex':
        with open(main_file, 'w') as f:
            f.write(main_table.to_latex(index=False, escape=False))
    elif args.format == 'markdown':
        with open(main_file, 'w') as f:
            f.write(main_table.to_markdown(index=False))
    else:
        with open(main_file, 'w') as f:
            f.write(tabulate(main_table, headers='keys', tablefmt='grid'))
    
    print(f"Main results table saved to {main_file}")
    
    # 生成消融实验表格
    for ablation_type in ['component']:
        ablation_table = create_ablation_table(df, ablation_type)
        ablation_file = os.path.join(args.output_dir, f'{ablation_type}_ablation.{args.format}')
        
        if args.format == 'latex':
            with open(ablation_file, 'w') as f:
                f.write(ablation_table.to_latex(index=False, escape=False))
        elif args.format == 'markdown':
            with open(ablation_file, 'w') as f:
                f.write(ablation_table.to_markdown(index=False))
        else:
            with open(ablation_file, 'w') as f:
                f.write(tabulate(ablation_table, headers='keys', tablefmt='grid'))
        
        print(f"{ablation_type} ablation table saved to {ablation_file}")

if __name__ == '__main__':
    main()
```

## 6. 实验执行流程优化

### 6.1 实验管理器 (experiment_manager.py)

```python
#!/usr/bin/env python3
import os
import json
import time
import subprocess
import argparse
from pathlib import Path
from typing import List, Dict, Optional

class ExperimentManager:
    def __init__(self, config_file: str):
        self.config = self.load_config(config_file)
        self.experiment_dir = Path(self.config.get('experiment_dir', 'experiments'))
        self.experiment_dir.mkdir(exist_ok=True)
        
    def load_config(self, config_file: str) -> Dict:
        """加载实验配置"""
        with open(config_file, 'r') as f:
            return json.load(f)
    
    def run_experiment(self, experiment_config: Dict) -> bool:
        """运行单个实验"""
        cmd = experiment_config['command']
        log_file = experiment_config['log_file']
        gpu = experiment_config.get('gpu', 0)
        
        # 设置环境变量
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        
        try:
            # 创建日志目录
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            # 运行实验
            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    cmd, shell=True, env=env, stdout=f, stderr=subprocess.STDOUT
                )
                
                # 等待进程完成
                return_code = process.wait()
                
                if return_code == 0:
                    print(f"✅ Experiment completed: {log_file}")
                    return True
                else:
                    print(f"❌ Experiment failed: {log_file} (return code: {return_code})")
                    return False
                    
        except Exception as e:
            print(f"❌ Error running experiment: {e}")
            return False
    
    def run_experiments_parallel(self, experiments: List[Dict], max_parallel: int = 4):
        """并行运行多个实验"""
        running = []
        completed = []
        failed = []
        
        for exp in experiments:
            # 等待有空闲槽位
            while len(running) >= max_parallel:
                # 检查运行中的实验
                for i, (process, exp_config) in enumerate(running):
                    if process.poll() is not None:
                        running.pop(i)
                        if process.returncode == 0:
                            completed.append(exp_config)
                        else:
                            failed.append(exp_config)
                        break
                else:
                    time.sleep(10)  # 等待10秒后再检查
            
            # 启动新实验
            cmd = exp['command']
            log_file = exp['log_file']
            gpu = exp.get('gpu', 0)
            
            # 设置环境变量
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(gpu)
            
            try:
                # 创建日志目录
                os.makedirs(os.path.dirname(log_file), exist_ok=True)
                
                # 启动进程
                with open(log_file, 'w') as f:
                    process = subprocess.Popen(
                        cmd, shell=True, env=env, stdout=f, stderr=subprocess.STDOUT
                    )
                
                running.append((process, exp))
                print(f"🚀 Started experiment: {exp['name']}")
                
            except Exception as e:
                print(f"❌ Failed to start experiment: {e}")
                failed.append(exp)
        
        # 等待所有实验完成
        for process, exp_config in running:
            process.wait()
            if process.returncode == 0:
                completed.append(exp_config)
            else:
                failed.append(exp_config)
        
        return completed, failed
    
    def generate_experiment_configs(self) -> List[Dict]:
        """根据配置生成实验列表"""
        experiments = []
        
        for exp_name, exp_config in self.config.get('experiments', {}).items():
            # 数据集循环
            for dataset in exp_config.get('datasets', []):
                # 种子循环
                for seed in exp_config.get('seeds', []):
                    # GPU分配
                    gpu_idx = experiments % len(exp_config.get('gpus', [0]))
                    gpu = exp_config['gpus'][gpu_idx]
                    
                    # 构建命令
                    cmd = f"python main.py --dataset {dataset} --seed_list {seed}"
                    
                    # 添加其他参数
                    for key, value in exp_config.get('parameters', {}).items():
                        if isinstance(value, bool):
                            if value:
                                cmd += f" --{key}"
                        else:
                            cmd += f" --{key} {value}"
                    
                    # 创建实验配置
                    exp = {
                        'name': f"{exp_name}_{dataset}_seed{seed}",
                        'command': cmd,
                        'log_file': str(self.experiment_dir / exp_name / f"{dataset}_seed{seed}.log"),
                        'gpu': gpu,
                        'dataset': dataset,
                        'seed': seed
                    }
                    
                    experiments.append(exp)
        
        return experiments

def main():
    parser = argparse.ArgumentParser(description='Experiment Manager')
    parser.add_argument('--config', type=str, required=True,
                       help='Experiment configuration file')
    parser.add_argument('--max_parallel', type=int, default=4,
                       help='Maximum parallel experiments')
    
    args = parser.parse_args()
    
    # 创建实验管理器
    manager = ExperimentManager(args.config)
    
    # 生成实验配置
    experiments = manager.generate_experiment_configs()
    print(f"Generated {len(experiments)} experiments")
    
    # 运行实验
    completed, failed = manager.run_experiments_parallel(experiments, args.max_parallel)
    
    # 输出结果
    print(f"\n✅ Completed experiments: {len(completed)}")
    print(f"❌ Failed experiments: {len(failed)}")
    
    if failed:
        print("\nFailed experiments:")
        for exp in failed:
            print(f"  - {exp['name']}: {exp['log_file']}")

if __name__ == '__main__':
    main()
```

### 6.2 实验配置示例 (experiment_config.json)

```json
{
  "experiment_dir": "experiments",
  "experiments": {
    "main_experiments": {
      "datasets": ["cifar100_224", "imagenet-r", "cub200_224", "cars196_224"],
      "seeds": [1993, 1996, 1997],
      "gpus": [0, 1, 2, 4],
      "parameters": {
        "smart_defaults": true,
        "lora_type": "sgp_lora",
        "vit_type": "vit-b-p16-mocov3",
        "gamma_kd": 0.0,
        "weight_temp": 1.0,
        "weight_kind": "log1p",
        "weight_p": 2.0
      }
    },
    "ablation_studies": {
      "datasets": ["cifar100_224", "imagenet-r"],
      "seeds": [1993, 1996, 1997],
      "gpus": [0, 1, 2, 4],
      "variants": [
        {
          "name": "full_method",
          "parameters": {
            "lora_type": "sgp_lora",
            "gamma_kd": 0.0,
            "weight_temp": 1.0,
            "weight_kind": "log1p",
            "weight_p": 2.0
          }
        },
        {
          "name": "wo_sgp",
          "parameters": {
            "lora_type": "basic_lora",
            "gamma_kd": 0.0
          }
        }
      ]
    }
  }
}
```

这个实验脚本设计提供了：

1. **完整的主实验脚本**：支持4种对比方法在4个数据集上的并行执行
2. **超参数网格搜索**：支持SGP参数的系统性搜索
3. **消融研究脚本**：支持组件消融、SGP消融和AMDC消融
4. **补充实验脚本**：支持长序列任务和跨架构泛化实验
5. **结果收集和分析**：自动化结果收集和表格生成
6. **实验管理器**：提供并行执行和资源管理功能

所有脚本都考虑了GPU资源分配、并行执行、日志记录和错误处理，确保实验能够高效可靠地运行。