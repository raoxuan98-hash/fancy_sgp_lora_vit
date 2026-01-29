
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
        'train_data': train_data,
        'test_data': test_data,
        'train_targets': remapped_train_targets,  # 使用重新映射的标签
        'test_targets': remapped_test_targets,    # 使用重新映射的标签
        'num_classes': len(class_indices),
        'use_path': original_dataset['use_path'],
        'class_names': class_names,
        'templates': original_dataset['templates'],
        'original_dataset_name': original_dataset['name'],
        'split_index': split_idx,
        'original_class_indices': class_indices,
        'label_mapping': label_mapping,  # 保存标签映射
        'is_incremental_split': True     # 标记这是增量分割的子集
    }
    
    return split_dataset
```

#### 2. 修改 `CrossDomainSimpleDataset` 类

```python
class CrossDomainSimpleDataset(Dataset):
    def __init__(self,
                 images,
                 labels,
                 use_path=False,
                 class_names=None,
                 templates=None,
                 transform=None,
                 label_offset=0,
                 apply_label_mapping=False,
                 label_mapping=None):
        # ... 现有代码 ...
        self.apply_label_mapping = apply_label_mapping
        self.label_mapping = label_mapping or {}
    
    def __getitem__(self, idx):
        # ... 现有的图像加载代码 ...
        
        label = int(self.labels[idx])
        
        # 应用标签映射（如果需要）
        if self.apply_label_mapping and label in self.label_mapping:
            label = self.label_mapping[label]
        
        class_name = self.class_names[label] if self.class_names is not None else None
        return image, label, class_name
```

#### 3. 修改 `get_subset` 方法

```python
def get_subset(self, task: int, source: str = "train", cumulative: bool = False,
              mode=None, transform=None) -> Dataset:
    # ... 现有代码 ...
    
    if cumulative:
        # 累积模式处理
        # ... 现有代码 ...
    else:
        # 非累积模式：返回当前任务的数据集
        dataset = self.datasets[task]
        if source == "train":
            data = dataset['train_data']
            targets = dataset['train_targets']
        else:
            data = dataset['test_data']
            targets = dataset['test_targets']
        
        # 检查是否是增量分割的子集
        is_incremental_split = dataset.get('is_incremental_split', False)
        
        if is_incremental_split:
            # 增量子集：标签已经重新映射，使用子集的class_names
            class_names = dataset['class_names']
            apply_label_mapping = False  # 标签已经映射过
            label_mapping = None
        else:
            # 普通数据集：使用累积class_names
            total_classes_up_to_task = sum(self.datasets[i]['num_classes'] for i in range(task + 1))
            class_names = self.global_class_names[:total_classes_up_to_task]
            apply_label_mapping = False
            label_mapping = None
        
        # 创建数据集
        dataset = CrossDomainSimpleDataset(
            images=data,
            labels=targets,
            use_path=dataset['use_path'],
            class_names=class_names,
            templates=dataset['templates'] if dataset['templates'] else None,
            transform=transform,
            label_offset=0,
            apply_label_mapping=apply_label_mapping,
            label_mapping=label_mapping
        )
    
    return dataset
```

#### 4. 修改 `_create_incremental_splits` 方法

更新全局标签偏移计算，因为增量分割的子集不需要全局偏移：

```python
def _create_incremental_splits(self) -> None:
    """创建增量拆分，将每个数据集拆分为多个增量子集"""
    if not self.enable_incremental_split or self.num_incremental_splits <= 1:
        return
    
    incremental_datasets = []
3. 修改`_create_incremental_splits`方法，确保全局标签偏移和类别名称正确累积
4. 测试修复后的代码

## 预期结果

修复后，每个增量分割的数据集将：
1. 使用局部标签（0到类别数-1）
2. 使用与标签范围匹配的类别名称
3. 正确处理全局标签偏移

这样将解决"Label 96 exceeds class_names length 50"的错误。