# Within-Domain数据集评估

## 概述

这个脚本用于执行within-domain数据集评估，使用vit-b-p16架构评估四个数据集，每个数据集使用一个GPU，每个数据集运行三个随机种子。

## 使用方法

### 方法1：使用Shell脚本

```bash
cd /home/raoxuan/projects/low_rank_rda
./sh/run_within_domain_evaluation.sh
```

### 方法2：直接执行Python命令

如果您想手动控制每个实验，可以直接使用以下命令格式：

```bash
# 设置GPU
export CUDA_VISIBLE_DEVICES=0

# 运行单个数据集的单个种子
python main.py \
    --dataset imagenet-r \
    --vit_type vit-b-p16 \
    --lora_type basic_lora \
    --seed_list 1993 \
    --smart_defaults \
    --cross_domain False
```

## 配置说明

### 固定配置
- **架构**: vit-b-p16
- **LoRA类型**: basic_lora
- **智能默认参数**: 启用 (--smart_defaults)

### 数据集配置
- imagenet-r (GPU 0)
- cifar100_224 (GPU 1) 
- cub200_224 (GPU 2)
- cars196_224 (GPU 3)

### 随机种子
每个数据集按顺序执行以下随机种子：
- 1993
- 1996
- 1997

### Smart Defaults参数
根据数据集自动设置以下参数：
- **imagenet-r**: init_cls=20, increment=20, iterations=1500
- **cifar100_224**: init_cls=10, increment=10, iterations=1500
