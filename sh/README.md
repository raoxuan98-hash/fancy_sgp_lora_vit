# 主实验脚本使用说明

## 概述

本目录包含了用于实施主实验的shell脚本，包括四种对比方法的实验设置。

## 脚本文件

1. **`main_experiments_basic_lora.sh`** - 基础LoRA实验
2. **`main_experiments_lora_kd.sh`** - LoRA + 蒸馏实验
3. **`main_experiments_nsp_lora.sh`** - LoRA-NSP实验
4. **`main_experiments_full_method.sh`** - 完整方法实验
5. **`run_all_main_experiments.sh`** - 主实验汇总脚本

## 实验方法

### 1. 基础LoRA
- LoRA类型：`basic_lora`
- 不使用知识蒸馏（`gamma_kd=0.0`）
- 使用vit-b-p16-mocov3架构

### 2. LoRA + 蒸馏
- LoRA类型：`basic_lora`
- 使用知识蒸馏（`gamma_kd=1.0`）
- 蒸馏类型：特征蒸馏（`kd_type=feat`）
- 蒸馏变换：恒等变换（`distillation_transform=identity`）
- 每个任务更新教师网络（`update_teacher_each_task=True`）

### 3. LoRA-NSP
- LoRA类型：`nsp_lora`
- NSP权重：0.05
- NSP epsilon：0.05
- 不使用知识蒸馏

### 4. 完整方法
- LoRA类型：`sgp_lora`
- 投影温度：1.0
- 权重函数：log1p
- 不使用知识蒸馏

## 数据集

实验将在以下四个数据集上进行：
- CIFAR-100 (10 tasks × 10 classes)
- ImageNet-R (10 tasks × 20 classes)
- CUB-200 (10 tasks × 20 classes)
- Cars-196 (10 tasks × 20 classes, 最后一任务6类)

## 使用方法

### 运行所有实验
```bash
bash sh/run_all_main_experiments.sh
```

### 运行特定方法实验
```bash
# 基础LoRA
bash sh/main_experiments_basic_lora.sh

# LoRA + 蒸馏
bash sh/main_experiments_lora_kd.sh

# LoRA-NSP
bash sh/main_experiments_nsp_lora.sh

# 完整方法
bash sh/main_experiments_full_method.sh
```

## 实验设置

- 每个实验运行3次（随机种子：1993, 1996, 1997）
- 顺序运行（非并行）
- 使用智能默认参数（`--smart_defaults`）
- 结果自动保存在相应的日志目录中

## 结果存储

实验结果将根据`trainer.py`中的`build_log_dirs`函数自动存储在以下结构中：
```
sldc_logs_{user}/{dataset}_{vit_type}/init-{init_cls}_inc-{increment}/lrank-{lora_rank}_ltype-{lora_type}/...
```

## 注意事项

1. 确保有足够的GPU资源运行实验
2. 实验是顺序运行的，总时间会较长
3. 可以根据需要调整GPU设置（修改脚本中的`CUDA_VISIBLE_DEVICES`）
4. 结果会自动保存在相应的日志目录中
5. 每个数据集的实验参数会根据`smart_defaults`自动调整

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

## 故障排除

如果遇到问题，请检查：
1. GPU资源是否足够
2. 数据集是否正确下载和配置
3. 依赖库是否正确安装
4. 路径设置是否正确