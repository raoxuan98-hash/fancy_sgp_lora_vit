
# 增量分割标签映射修复方案

## 问题描述

当 `enable_incremental_split=True` 时，系统会分割跨域的每个数据集，将其子集作为独立的数据集。问题在于：

1. 每个数据集在划分时，标签是随机划分的，没有做映射
2. 创建分类器时，task_size是子集的标签数量
3. dataloader出现的样本标签超出了task_size范围，导致错误

错误信息：
```
ValueError: Label 96 exceeds class_names length 50
```

## 根本原因分析

1. **标签处理流程问题**：
   - 在 `_init_balanced_datasets` 中，所有数据集的标签都应用了全局偏移
   - 在 `_create_incremental_dataset` 中，筛选了特定类别但保持了原始的全局标签
   - 在 `get_subset` 方法中，检查标签是否超出 `cumulative_class_names` 长度，导致错误

2. **示例场景**：
   - 原始数据集有100个类别（标签0-99）
   - 应用全局偏移后变为100-199
   - 增量分割随机选择了类别[1, 3, 8]，对应全局标签[101, 103, 108]
   - 子集应该有3个类别，标签应该是0, 1, 2
   - 但实际标签是101, 103, 108，超出了范围

## 解决方案

### 核心思路
当启用增量分割时，每个子集应该被视为一个独立的"任务"，其标签应该重新映射为从0开始的连续标签。

### 实现方案

#### 1. 修改 `_create_incremental_dataset` 方法

```python
def _create_incremental_dataset(self, original_dataset: Dict, class_indices: List[int],
                               split_idx: int) -> Dict:
    """
    创建单个增量子集，包含标签重新映射
    """
    # 创建标签映射：原始全局标签 -> 新的局部标签
    original_labels = [original_dataset['global_label_offset'] + idx for idx in class_indices]
    label_mapping = {orig_label: new_label for new_label, orig_label in enumerate(sorted(original_labels))}
    
    # 筛选训练数据
    train_mask = np.isin(original_dataset['train_targets'], original_labels)
    train_data = original_dataset['train_data'][train_mask]
    train_targets = original_dataset['train_targets'][train_mask]
    
    # 应用标签映射
    remapped_train_targets = np.array([label_mapping[label] for label in train_targets])
    
    # 同样处理测试数据
    test_mask = np.isin(original_dataset['test_targets'], original_labels)
    test_data = original_dataset['test_data'][test_mask]
    test_targets = original_dataset['test_targets'][test_mask]
    remapped_test_targets = np.array([label_mapping[label] for label in test_targets])
    
    # 筛选类别名称
    class_names = [original_dataset['class_names'][idx] for idx in class_indices]
    
    # 创建子集数据集
    split_dataset = {
        'name': f"{original_dataset['name']}_split_{split_idx}",
    
    return split_dataset
```

### 2. 修复`get_subset`方法

在`utils/cross_domain_data_manager.py`文件中，需要修改`get_subset`方法中的非累积模式部分：

```python
# 非累积模式：返回当前任务的数据集，标签从上个任务结束后的累积标签开始
dataset = self.datasets[task]
if source == "train":
    data = dataset['train_data']
    targets = dataset['train_targets']
else:
    data = dataset['test_data']
    targets = dataset['test_targets']

# DEBUG: 记录非累积模式下的数据信息
targets_array = np.array(targets)
logging.info(f"[CDM] Non-cumulative mode for task {task}, source {source}:")
logging.info(f"[CDM]   Total samples: {len(data)}")
logging.info(f"[CDM]   Local label range: min={np.min(targets_array)}, max={np.max(targets_array)}")

# 对于非累积模式，使用当前任务的类别名称而不是累积类别名称
if hasattr(dataset, 'split_index') or 'split_index' in dataset:
    # 这是一个增量分割的数据集，使用其自身的类别名称
    task_class_names = dataset['class_names']
else:
    # 这是一个原始数据集，使用累积类别名称
    total_classes_up_to_task = sum(self.datasets[i]['num_classes'] for i in range(task + 1))
    task_class_names = self.global_class_names[:total_classes_up_to_task]

logging.info(f"[CDM]   Class names count: {len(task_class_names)}")
logging.info(f"[CDM]   Expected max valid label: {len(task_class_names) - 1}")

# 检查标签是否在有效范围内
if len(task_class_names) > 0:
    max_label = np.max(targets_array)
    if max_label >= len(task_class_names):
        logging.error(f"[CDM] ERROR: Max label {max_label} exceeds class_names length {len(task_class_names)}")
        logging.error(f"[CDM] Task {task} ({dataset['name']}): expected range 0-{len(task_class_names)-1}")
        logging.error(f"[CDM] Actual label range: {np.min(targets_array)}-{max_label}")
        raise ValueError(f"Label {max_label} exceeds class_names length {len(task_class_names)}")

dataset = CrossDomainSimpleDataset(
    images=data,
    labels=targets,  # 保持局部标签
    use_path=dataset['use_path'],
    class_names=task_class_names,  # 使用正确的类别名称
    templates=dataset['templates'] if dataset['templates'] else None,
    transform=transform,
    label_offset=0
)
```

### 3. 修复全局标签偏移处理

在`_create_incremental_splits`方法中，需要确保全局标签偏移正确计算：

```python
def _create_incremental_splits(self) -> None:
    """创建增量拆分，将每个数据集拆分为多个增量子集"""
    if not self.enable_incremental_split or self.num_incremental_splits <= 1:
        return
    
    incremental_datasets = []
    incremental_global_label_offset = []
    cumulative_class_names = []
    
    for dataset_idx, dataset in enumerate(self.datasets):
        dataset_name = dataset['name']
        num_classes = dataset['num_classes']
        
        logging.info(f"[BCDM] Creating incremental splits for dataset {dataset_name}: "
                    f"{num_classes} classes -> {self.num_incremental_splits} splits")
        
        # 随机拆分类别
        class_splits = self._split_classes_randomly(num_classes, self.num_incremental_splits,
                                                   self.incremental_split_seed + dataset_idx)
        
        # 为每个拆分创建数据集
        for split_idx, class_indices in enumerate(class_splits):
            split_dataset = self._create_incremental_dataset(dataset, class_indices, split_idx)
            incremental_datasets.append(split_dataset)
            
            # 计算全局标签偏移（基于之前所有拆分的类别数）
            offset = sum(d['num_classes'] for d in incremental_datasets[:-1])
            incremental_global_label_offset.append(offset)
            
            # 累积类别名称
            cumulative_class_names.extend(split_dataset['class_names'])
            
            logging.info(f"[BCDM]   Split {split_idx + 1}: {len(class_indices)} classes, "
                        f"{len(split_dataset['train_data'])} train samples, "
                        f"{len(split_dataset['test_data'])} test samples")
    
    # 更新数据集和偏移
    self.datasets = incremental_datasets
    self.global_label_offset = incremental_global_label_offset
    self.global_class_names = cumulative_class_names
    self.total_classes = sum(d['num_classes'] for d in self.datasets)
    
    logging.info(f"[BCDM] After incremental split: {len(self.datasets)} total tasks, "
                f"{self.total_classes} total classes")
```

## 实施步骤

1. 修改`_create_incremental_dataset`方法，确保标签正确映射到局部索引
2. 修改`get_subset`方法，确保类别名称与标签范围匹配
3. 修改`_create_incremental_splits`方法，确保全局标签偏移和类别名称正确累积
4. 测试修复后的代码

## 预期结果

修复后，每个增量分割的数据集将：
1. 使用局部标签（0到类别数-1）
2. 使用与标签范围匹配的类别名称
3. 正确处理全局标签偏移

这样将解决"Label 96 exceeds class_names length 50"的错误。