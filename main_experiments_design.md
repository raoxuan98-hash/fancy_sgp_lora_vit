# 主实验脚本设计文档

## 概述

本文档描述了用于实施主实验的shell脚本设计，包括四种对比方法的实验设置。

## 实验方法

1. **基础LoRA**：使用`basic_lora`类型
2. **LoRA + Distillation**：设置`gamma_kd=1.0`，使用知识蒸馏
3. **LoRA-NSP**：使用`nsp_lora`类型，设置`nsp_weight=0.05`
4. **完整方法**：`sgp_lora` + `RGDA` + `AMDC`

## 数据集设置

| 数据集     | 任务设置              | 初始类别数 | 每任务新增类别数 | 总任务数           |
| ---------- | --------------------- | ---------- | ---------------- | ------------------ |
| CIFAR-100  | 10 tasks × 10 classes | 10         | 10               | 10                 |
| ImageNet-R | 10 tasks × 20 classes | 20         | 20               | 10                 |
| CUB-200    | 10 tasks × 20 classes | 20         | 20               | 10                 |
| Cars-196   | 10 tasks × 20 classes | 20         | 20               | 10 (最后一任务6类) |

## 脚本结构

### 1. 基础LoRA实验脚本 (`sh/main_experiments_basic_lora.sh`)

```bash
#!/usr/bin/env bash
set -euo pipefail

# 数据集列表
DATASETS=("cifar100_224" "imagenet-r" "cub200_224" "cars196_224")
SEEDS=(1993 1996 1997)

# 基础LoRA参数
LORA_TYPE="basic_lora"
VIT_TYPE="vit-b-p16-mocov3"
GAMMA_KD=0.0  # 不使用蒸馏

# 顺序运行所有实验
for DATASET in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "Running basic LoRA experiment: dataset=$DATASET, seed=$SEED"
        
        CUDA_VISIBLE_DEVICES=0 python -u main.py \
            --dataset "$DATASET" \
            --smart_defaults \
            --lora_type "$LORA_TYPE" \
            --vit_type "$VIT_TYPE" \
            --gamma_kd "$GAMMA_KD" \
            --seed_list "$SEED"
    done
done

echo "Basic LoRA experiments completed."
```

### 2. LoRA + 蒸馏实验脚本 (`sh/main_experiments_lora_kd.sh`)

```bash
#!/usr/bin/env bash
set -euo pipefail

# 数据集列表
DATASETS=("cifar100_224" "imagenet-r" "cub200_224" "cars196_224")
SEEDS=(1993 1996 1997)

# LoRA + 蒸馏参数
LORA_TYPE="basic_lora"
VIT_TYPE="vit-b-p16-mocov3"
GAMMA_KD=1.0  # 使用蒸馏
UPDATE_TEACHER_EACH_TASK=True
DISTILLATION_TRANSFORM="identity"
KD_TYPE="feat"

# 顺序运行所有实验
for DATASET in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "Running LoRA + KD experiment: dataset=$DATASET, seed=$SEED"
        
        CUDA_VISIBLE_DEVICES=0 python -u main.py \
            --dataset "$DATASET" \
            --smart_defaults \
            --lora_type "$LORA_TYPE" \
            --vit_type "$VIT_TYPE" \
            --gamma_kd "$GAMMA_KD" \
            --update_teacher_each_task "$UPDATE_TEACHER_EACH_TASK" \
            --distillation_transform "$DISTILLATION_TRANSFORM" \
            --kd_type "$KD_TYPE" \
            --seed_list "$SEED"
    done
done

echo "LoRA + KD experiments completed."
```

### 3. LoRA-NSP实验脚本 (`sh/main_experiments_nsp_lora.sh`)

```bash
#!/usr/bin/env bash
set -euo pipefail

# 数据集列表
DATASETS=("cifar100_224" "imagenet-r" "cub200_224" "cars196_224")
SEEDS=(1993 1996 1997)

# LoRA-NSP参数
LORA_TYPE="nsp_lora"
VIT_TYPE="vit-b-p16-mocov3"
GAMMA_KD=0.0  # 不使用蒸馏
NSP_WEIGHT=0.05
NSP_EPS=0.05

# 顺序运行所有实验
for DATASET in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "Running LoRA-NSP experiment: dataset=$DATASET, seed=$SEED"
        
        CUDA_VISIBLE_DEVICES=0 python -u main.py \
            --dataset "$DATASET" \
            --smart_defaults \
            --lora_type "$LORA_TYPE" \
            --vit_type "$VIT_TYPE" \
            --gamma_kd "$GAMMA_KD" \
            --nsp_weight "$NSP_WEIGHT" \
            --nsp_eps "$NSP_EPS" \
            --seed_list "$SEED"
    done
done

echo "LoRA-NSP experiments completed."
```

### 4. 完整方法实验脚本 (`sh/main_experiments_full_method.sh`)

```bash
#!/usr/bin/env bash
set -euo pipefail

# 数据集列表
DATASETS=("cifar100_224" "imagenet-r" "cub200_224" "cars196_224")
SEEDS=(1993 1996 1997)

# 完整方法参数
LORA_TYPE="sgp_lora"
VIT_TYPE="vit-b-p16-mocov3"
GAMMA_KD=0.0  # 不使用蒸馏
WEIGHT_TEMP=1.0
WEIGHT_KIND="log1p"

# 顺序运行所有实验
for DATASET in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "Running Full Method experiment: dataset=$DATASET, seed=$SEED"
        
        CUDA_VISIBLE_DEVICES=0 python -u main.py \
            --dataset "$DATASET" \
            --smart_defaults \
            --lora_type "$LORA_TYPE" \
            --vit_type "$VIT_TYPE" \
            --gamma_kd "$GAMMA_KD" \
            --weight_temp "$WEIGHT_TEMP" \
            --weight_kind "$WEIGHT_KIND" \
            --seed_list "$SEED"
    done
done

echo "Full Method experiments completed."
```

### 5. 主实验汇总脚本 (`sh/run_all_main_experiments.sh`)

```bash
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
```

## 参数说明

### 通用参数
- `--dataset`: 数据集名称
- `--smart_defaults`: 使用智能默认参数
- `--vit_type`: ViT架构类型
- `--seed_list`: 随机种子列表

### LoRA类型特定参数
- `--lora_type`: LoRA类型 (basic_lora/sgp_lora/nsp_lora)
- `--gamma_kd`: 知识蒸馏权重
- `--update_teacher_each_task`: 是否每个任务更新教师网络
- `--distillation_transform`: 蒸馏变换类型
- `--kd_type`: 知识蒸馏类型

### SGP-LoRA特定参数
- `--weight_temp`: 投影温度参数
- `--weight_kind`: 权重函数类型

### NSP-LoRA特定参数
- `--nsp_weight`: NSP权重
- `--nsp_eps`: NSP epsilon参数

## 实验结果存储

实验结果将根据`trainer.py`中的`build_log_dirs`函数自动存储在以下结构中：
```
sldc_logs_{user}/{dataset}_{vit_type}/init-{init_cls}_inc-{increment}/lrank-{lora_rank}_ltype-{lora_type}/...
```

## 使用说明

1. 运行所有实验：
   ```bash
   bash sh/run_all_main_experiments.sh
   ```

2. 运行特定方法实验：
   ```bash
   bash sh/main_experiments_basic_lora.sh
   bash sh/main_experiments_lora_kd.sh
   bash sh/main_experiments_nsp_lora.sh
   bash sh/main_experiments_full_method.sh
   ```

## 注意事项

1. 确保有足够的GPU资源运行实验
2. 实验是顺序运行的，总时间会较长
3. 可以根据需要调整GPU设置
4. 结果会自动保存在相应的日志目录中