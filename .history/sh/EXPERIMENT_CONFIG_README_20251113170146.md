# 实验配置详细说明

## 概述

本项目包含四种不同的LoRA实验方法的脚本，每种方法都针对类增量学习场景进行了优化。所有脚本都支持并行GPU执行，并且每个实验运行3个随机种子以确保结果的可重现性。

## 实验方法详解

### 1. 基础LoRA (basic_lora)

**脚本**: [`sh/main_experiments_basic_lora.sh`](sh/main_experiments_basic_lora.sh)

**核心参数配置**:
```bash
LORA_TYPE="basic_lora"
VIT_TYPE="vit-b-p16-mocov3"
GAMMA_KD=0.0  # 不使用知识蒸馏
```

**特点**:
- 使用标准的LoRA适配器进行参数高效微调
- 不使用任何知识蒸馏或正则化技术
- 作为其他方法的基线对比

**适用场景**: 评估基本的参数高效微调在类增量学习中的表现

### 2. LoRA + 知识蒸馏 (lora_kd)

**脚本**: [`sh/main_experiments_lora_kd.sh`](sh/main_experiments_lora_kd.sh)

**核心参数配置**:
```bash
LORA_TYPE="basic_lora"
VIT_TYPE="vit-b-p16-mocov3"
GAMMA_KD=1.0  # 使用知识蒸馏
UPDATE_TEACHER_EACH_TASK=True
DISTILLATION_TRANSFORM="identity"
KD_TYPE="feat"
```

**特点**:
- 在基础LoRA上添加特征知识蒸馏
- 每个任务后更新教师网络
- 使用恒等变换进行特征蒸馏
- 蒸馏权重为1.0，表示与原始损失同等重要

**适用场景**: 减少灾难性遗忘，保持对先前任务的知识

### 3. LoRA-NSP (nsp_lora)

**脚本**: [`sh/main_experiments_nsp_lora.sh`](sh/main_experiments_nsp_lora.sh)

**核心参数配置**:
```bash
LORA_TYPE="nsp_lora"
VIT_TYPE="vit-b-p16-mocov3"
GAMMA_KD=0.0  # 不使用蒸馏
NSP_WEIGHT=0.05
NSP_EPS=0.05
```

**特点**:
- 使用NSP (Negative Semantic Preservation) 正则化
- NSP权重为0.05，控制正则化强度
- NSP epsilon为0.05，用于数值稳定性
- 不使用知识蒸馏

**适用场景**: 通过负语义保持防止模型遗忘先前学到的知识

### 4. LoRA-SGP (sgp_lora)

**脚本**: 
- [`sh/main_experiments_within_domain_sgp_lora.sh`](sh/main_experiments_within_domain_sgp_lora.sh) - 域内实验
- [`sh/main_experiments_cross_domain_sgp_lora.sh`](sh/main_experiments_cross_domain_sgp_lora.sh) - 跨域实验

**核心参数配置**:
```bash
# 域内实验
LORA_TYPE="sgp_lora"
VIT_TYPE="vit-b-p16-mocov3"
GAMMA_KD=0.0
WEIGHT_TEMP=1.0
WEIGHT_KIND="log1p"
WEIGHT_P=1.0

# 跨域实验
LORA_TYPE="sgp_lora"
VIT_TYPE="vit-b-p16-mocov3"
GAMMA_KD=0.0
WEIGHT_TEMP=2.0  # 跨域实验使用更高的温度
WEIGHT_KIND="log1p"
WEIGHT_P=1.0
CROSS_DOMAIN=True
NUM_SHOTS=16
```

**特点**:
- 使用SGP (Semantic-Guided Projection) 技术
- 投影温度控制特征变换的平滑程度
- 权重函数使用log1p，平衡新旧任务的重要性
- 跨域实验使用更高的温度(2.0)以适应更大的域差异
- 跨域实验使用16-shot学习设置

**适用场景**: 通过语义引导投影平衡新旧任务的学习

## 统一实验脚本

**脚本**: [`sh/run_all_four_methods_experiments.sh`](sh/run_all_four_methods_experiments.sh)

这个脚本整合了所有四种实验方法，可以一次性运行所有对比实验。

**运行逻辑**:
1. 为每种实验类型创建独立的日志目录
2. 将每个数据集分配到不同的GPU上并行执行
3. 每个数据集上依次运行3个随机种子的实验
4. 等待所有实验完成后汇总结果

## 数据集配置

所有脚本使用相同的数据集列表:
```bash
DATASETS=("cifar100_224" "imagenet-r" "cub200_224" "cars196_224")
```

**数据集特点**:
- **cifar100_224**: 100类图像分类，每任务10类，共10个任务
- **imagenet-r**: ImageNet的渲染版本，20类每任务，共10个任务
- **cub200_224**: 200种鸟类分类，20类每任务，共10个任务
- **cars196_224**: 196种汽车分类，20类每任务，最后一任务6类

## GPU分配策略

```bash
GPUS=(0 1 2 4)
```

每个数据集分配到一个独立的GPU，实现并行执行:
- cifar100_224 → GPU 0
- imagenet-r → GPU 1
- cub200_224 → GPU 2
- cars196_224 → GPU 4

## 随机种子配置

所有实验使用相同的3个随机种子:
```bash
SEEDS=(1993 1996 1997)
```

这确保了结果的可重现性，并允许计算平均性能和标准差。

## 智能默认参数

所有脚本都使用 `--smart_defaults` 参数，这会根据数据集自动调整以下参数:

```python
if ns.dataset == 'cars196_224':
    ns.init_cls, ns.increment, ns.iterations = 20, 20, 1500
elif ns.dataset == 'imagenet-r':
    ns.init_cls, ns.increment, ns.iterations = 20, 20, 2000
elif ns.dataset == 'cifar100_224':
    ns.init_cls, ns.increment, ns.iterations = 10, 10, 2000
elif ns.dataset == 'cub200_224':
    ns.init_cls, ns.increment, ns.iterations = 20, 20, 1500
```

## 运行命令示例

### 运行单个方法实验
```bash
# 基础LoRA
bash sh/main_experiments_basic_lora.sh

# LoRA + 知识蒸馏
bash sh/main_experiments_lora_kd.sh

# LoRA-NSP
bash sh/main_experiments_nsp_lora.sh

# LoRA-SGP (域内)
bash sh/main_experiments_within_domain_sgp_lora.sh

# LoRA-SGP (跨域)
bash sh/main_experiments_cross_domain_sgp_lora.sh
```

### 运行所有四种方法
```bash
bash sh/run_all_four_methods_experiments.sh
```

## 实验结果存储

实验结果根据 `trainer.py` 中的 `build_log_dirs` 函数自动存储在以下结构中:
```
sldc_logs_{user}/{dataset}_{vit_type}/init-{init_cls}_inc-{increment}/lrank-{lora_rank}_ltype-{lora_type}/...
```

每个实验都会生成详细的日志文件和结果统计，包括:
- 训练过程日志
- 最终性能指标
- 多种子统计结果
- 参数配置文件

## 注意事项

1. **GPU资源**: 确保有足够的GPU资源(至少4个GPU)来并行执行实验
2. **存储空间**: 实验会产生大量日志文件，确保有足够的存储空间
3. **运行时间**: 完整实验可能需要数小时到数天，取决于GPU性能
4. **依赖检查**: 运行前确保所有依赖库已正确安装
5. **数据集**: 确保所有数据集已正确下载和配置

## 故障排除

如果遇到问题，请检查:
1. GPU是否可用且未被其他进程占用
2. 数据集路径是否正确配置
3. 依赖库是否完整安装
4. 脚本权限是否正确设置
5. 日志目录是否有写入权限