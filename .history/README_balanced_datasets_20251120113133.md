# Cross-Domain平衡数据集使用指南

## 概述

本项目提供了一个完整的解决方案，用于重新划分cross-domain数据集，解决原始数据集中类别样本数量不平衡的问题。通过将每个类别的样本数量限制到128，并确保测试集每个类别至少有128个样本，我们可以获得更加平衡和公平的实验环境。

## 主要特性

- ✅ **自动平衡采样**：将每个类别的样本数量限制到128
- ✅ **智能补充策略**：当测试样本不足时，从训练集中采样补充
- ✅ **完整元数据记录**：记录原始分布和采样过程的详细信息
- ✅ **向后兼容**：与现有的CrossDomainDataManagerCore完全兼容
- ✅ **可重现性**：使用固定随机种子确保结果可重现

## 文件结构

```
├── dataset_resplitter.py              # 数据集重新划分核心实现
├── utils/
│   └── balanced_cross_domain_data_manager.py  # 平衡数据管理器
├── test_balanced_datasets.py          # 测试脚本
├── cross_domain_dataset_resplit_plan.md  # 详细设计方案
└── balanced_datasets/                # 平衡后数据集目录
    ├── metadata/                     # 元数据目录
    │   ├── original_distribution.json
    │   ├── balanced_distribution.json
    │   ├── sampling_config.json
    │   └── dataset_statistics.json
    ├── cifar100_224/               # 平衡后的数据集
    │   ├── train/
    │   │   ├── 0/
    │   │   ├── 1/
    │   │   └── ...
    │   ├── test/
    │   │   ├── 0/
    │   │   ├── 1/
    │   │   └── ...
    │   └── label.txt
    └── ...
```

## 快速开始

### 1. 重新划分数据集

```python
from dataset_resplitter import DatasetResplitter

# 创建重新划分器
resplitter = DatasetResplitter(
    max_samples_per_class=128,  # 每个类别的最大样本数
    seed=42,                    # 随机种子
    output_dir="balanced_datasets"  # 输出目录
)

# 处理所有数据集
default_datasets = [
    'cifar100_224', 'cub200_224', 'resisc45', 'imagenet-r', 'caltech-101', 
    'dtd', 'fgvc-aircraft-2013b-variants102', 'food-101', 'mnist', 
    'oxford-flower-102', 'oxford-iiit-pets', 'cars196_224'
]

results = resplitter.resplit_all_datasets(default_datasets)
```

### 2. 使用平衡后的数据管理器

```python
from utils.balanced_cross_domain_data_manager import create_balanced_data_manager

# 创建平衡数据管理器
manager = create_balanced_data_manager(
    dataset_names=default_datasets,
    balanced_datasets_root="balanced_datasets",
    use_balanced_datasets=True  # 使用平衡后的数据集
)

# 获取数据集
train_dataset = manager.get_subset(task_id=0, source="train", mode="train")
test_dataset = manager.get_subset(task_id=0, source="test", mode="test")

print(f"训练集样本数: {len(train_dataset)}")
print(f"测试集样本数: {len(test_dataset)}")
```

### 3. 与原始数据集比较

```python
# 获取平衡后的统计信息
stats = manager.get_balanced_statistics()
for dataset_name, stat in stats.items():
    print(f"{dataset_name}:")
    print(f"  训练每类: min={stat['train_per_class']['min']}, max={stat['train_per_class']['max']}")
    print(f"  测试每类: min={stat['test_per_class']['min']}, max={stat['test_per_class']['max']}")

# 与原始数据集比较
comparison = manager.compare_with_original()
for dataset_name, comp in comparison.items():
    print(f"{dataset_name}:")
    print(f"  原始测试样本: {comp['original']['total_test_samples']}")
    print(f"  平衡测试样本: {comp['balanced']['total_test_samples']}")
```

## 详细使用说明

### 数据集重新划分算法

平衡算法遵循以下原则：

1. **上限约束**：每个类别最多128个样本（训练集和测试集）
2. **下限保证**：每个类别至少128个测试样本
3. **平衡补充**：测试样本不足时从训练样本中随机采样
4. **随机性控制**：使用固定种子确保可重现性

#### 具体流程

对于每个类别的处理：

```python
if test_count >= 128:
    # 测试样本足够，直接采样128个
    # 训练样本也采样128个（如果足够）
else:
    # 测试样本不足，需要从训练集补充
    needed = 128 - test_count
    if train_count >= needed:
        # 从训练集中移动needed个样本到测试集
        # 剩余训练样本采样128个（如果足够）
    else:
        # 即使所有训练样本移动也不够
        # 保留所有样本，标记为低样本类别
```

### 配置参数

#### DatasetResplitter参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_samples_per_class` | int | 128 | 每个类别的最大样本数 |
| `seed` | int | 42 | 随机种子 |
| `output_dir` | str | "balanced_datasets" | 输出目录 |

#### BalancedCrossDomainDataManagerCore参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `dataset_names` | List[str] | - | 数据集名称列表 |
| `balanced_datasets_root` | str | "balanced_datasets" | 平衡数据集根目录 |
| `use_balanced_datasets` | bool | True | 是否使用平衡数据集 |
| 其他参数 | - | - | 与CrossDomainDataManagerCore相同 |

### 元数据说明

平衡过程会生成详细的元数据：

#### original_distribution.json
记录原始数据集的分布信息：
```json
{
  "dataset_name": {
    "train_counts": {"0": 500, "1": 500, ...},
    "test_counts": {"0": 100, "1": 100, ...},
    "class_names": ["class_0", "class_1", ...],
    "total_train_samples": 50000,
    "total_test_samples": 10000,
    "num_classes": 100
  }
}
```

#### balanced_distribution.json
记录平衡后的分布信息：
```json
{
  "dataset_name": {
    "train_counts": {"0": 128, "1": 128, ...},
    "test_counts": {"0": 128, "1": 128, ...},
    "total_train_samples": 12800,
    "total_test_samples": 12800,
    "sampling_info": {
      "classes_processed": 100,
      "classes_with_insufficient_samples": [],
      "samples_moved_from_train_to_test": {"0": 28, "1": 28, ...}
    }
  }
}
```

#### sampling_config.json
记录采样配置：
```json
{
  "max_samples_per_class": 128,
  "seed": 42,
  "output_dir": "balanced_datasets"
}
```

#### dataset_statistics.json
记录整体统计信息：
```json
{
  "total_datasets_processed": 12,
  "datasets_with_errors": 0,
  "classes_with_insufficient_samples": 0,
  "total_samples_moved": 1234,
  "balance_improvement": {}
}
```

## 测试和验证

### 运行测试脚本

```bash
python test_balanced_datasets.py
```

测试脚本会验证：
- ✅ 数据集重新划分功能
- ✅ 平衡数据管理器功能
- ✅ 与原始数据集的比较
- ✅ 数据加载功能
- ✅ 元数据文件生成

### 验证平衡效果

```python
# 检查平衡效果
stats = manager.get_balanced_statistics()
for dataset_name, stat in stats.items():
    train_imbalance = stat['train_per_class']['max'] / stat['train_per_class']['min']
    test_imbalance = stat['test_per_class']['max'] / stat['test_per_class']['min']
    
    print(f"{dataset_name}:")
    print(f"  训练集不平衡比率: {train_imbalance:.2f}x")
    print(f"  测试集不平衡比率: {test_imbalance:.2f}x")
```

## 集成到现有实验

### 方法1：替换数据管理器

```python
# 原始代码
from utils.cross_domain_data_manager import CrossDomainDataManagerCore
manager = CrossDomainDataManagerCore(dataset_names=datasets, ...)

# 替换为
from utils.balanced_cross_domain_data_manager import create_balanced_data_manager
manager = create_balanced_data_manager(dataset_names=datasets, ...)
```

### 方法2：条件使用

```python
import os
from utils.cross_domain_data_manager import CrossDomainDataManagerCore
from utils.balanced_cross_domain_data_manager import create_balanced_data_manager

use_balanced = os.path.exists("balanced_datasets") and args.use_balanced

if use_balanced:
    manager = create_balanced_data_manager(
        dataset_names=datasets,
        balanced_datasets_root="balanced_datasets",
        use_balanced_datasets=True,
        **kwargs
    )
else:
    manager = CrossDomainDataManagerCore(
        dataset_names=datasets,
        **kwargs
    )
```

## 性能对比

### 数据平衡性改善

根据我们的分析，原始数据集存在严重不平衡：

| 数据集 | 原始测试样本数 | 平衡后测试样本数 | 原始不平衡比率 | 平衡后不平衡比率 |
|--------|------------------|------------------|------------------|------------------|
| dtd | 1,880 | 6,016 | 1.0x | 1.0x |
| food-101 | 25,250 | 12,928 | 1.0x | 1.0x |
| resisc45 | 25,200 | 5,760 | 1.1x | 1.0x |
| mnist | 10,000 | 1,280 | 1.27x | 1.0x |

### 实验效果预期

使用平衡数据集后，预期可以获得：

1. **更公平的性能评估**：避免大样本数据集主导整体性能
2. **更稳定的训练过程**：减少类别间的不平衡影响
3. **更准确的模型比较**：在不同数据集上的性能更具可比性

## 常见问题

### Q: 如何处理样本总数不足128的类别？

A: 系统会保留所有可用样本，并在元数据中标记为"低样本类别"。建议后续使用数据增强技术。

### Q: 可以修改每个类别的样本数量限制吗？

A: 可以，通过修改`max_samples_per_class`参数。例如设置为64或256。

### Q: 如何确保结果可重现？

A: 系统使用固定随机种子，所有采样过程都是确定性的。元数据中记录了完整的配置信息。

### Q: 平衡数据集会占用多少存储空间？

A: 大约是原始数据集的30-50%，具体取决于原始不平衡程度。

### Q: 如何回退到原始数据集？

A: 将`use_balanced_datasets`设置为`False`，或直接使用原始的`CrossDomainDataManagerCore`。

## 扩展功能

### 自定义采样策略

可以通过继承`DatasetResplitter`类来实现自定义采样策略：

```python
class CustomResplitter(DatasetResplitter):
    def _balance_dataset(self, train_data, train_targets, test_data, test_targets, num_classes):
        # 实现自定义平衡逻辑
        return super()._balance_dataset(train_data, train_targets, test_data, test_targets, num_classes)
```

### 数据增强集成

对于样本不足的类别，可以集成数据增强：

```python
def augment_samples(data, targets, target_count):
    # 实现数据增强逻辑
    augmented_data, augmented_targets = [], []
    # ... 增强逻辑 ...
    return np.array(augmented_data), np.array(augmented_targets)
```

## 贡献指南

欢迎提交Issue和Pull Request来改进这个项目：

1. **Bug报告**：请提供详细的错误信息和复现步骤
2. **功能请求**：请描述清楚期望的功能和使用场景
3. **代码贡献**：请确保代码风格一致，并添加适当的测试

## 许可证

本项目遵循与主项目相同的许可证。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交GitHub Issue
- 发送邮件至项目维护者

---

**注意**：首次运行重新划分过程可能需要较长时间，具体取决于数据集大小和系统性能。建议在空闲时间运行，或使用高性能计算资源。