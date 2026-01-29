
# 修复 enable_incremental_split=True 时的标签映射问题

## 问题分析

当 `enable_incremental_split=True` 时，系统会分割 cross-domain 的每个数据集，将其子集作为独立的数据集。问题在于：

1. **标签映射问题**：在创建分类器时，task_size 是子集的标签数量。但因为每个数据集在划分时候，标签也是随机划分的，没有做映射，导致 dataloader 出现的样本的标签超出了 task_size，所以报错了。

2. **全局偏移问题**：每个数据集的起点需要加上全局的 offset。例如，假设第二个数据集原本的标签范围是0到99。第一个数据集的标签范围是0-199。那么第二个数据集的标签范围得从200开始到300。

## 具体问题位置

### 1. `_create_incremental_dataset` 方法 (utils/balanced_cross_domain_data_manager.py:188)

当前代码保持了原始的全局标签，但没有进行重新映射：

```python
# 当前代码（有问题）
train_targets = original_dataset['train_targets'][train_mask]  # 保持原始全局标签
test_targets = original_dataset['test_targets'][test_mask]    # 保持原始全局标签
```

### 2. `_create_incremental_splits` 方法 (utils/balanced_cross_domain_data_manager.py:110)

在计算全局标签偏移时，没有考虑到增量拆分后的标签重新映射：

```python
# 当前代码（有问题）
offset = sum(d['num_classes'] for d in incremental_datasets[:-1])
```

## 修复方案

### 1. 修复 `_create_incremental_dataset` 方法

需要将每个子集的标签重新映射到从0开始的连续索引，同时记录原始标签到新标签的映射关系：

```python
def _create_incremental_dataset(self, original_dataset: Dict, class_indices: List[int],
                               split_idx: int, global_offset: int) -> Dict:
    """
    创建单个增量子集
    
    Args:
        original_dataset: 原始数据集
        class_indices: 该子集包含的类别索引（局部索引）
        split_idx: 子集索引
        global_offset: 全局标签偏移量
        
    Returns:
        增量子集的数据字典
    """
    # 创建标签映射：原始局部标签 -> 新的连续标签
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(class_indices)}
    
    # 筛选训练数据
    train_mask = np.isin(original_dataset['train_targets'], class_indices)
    train_data = original_dataset['train_data'][train_mask]
    train_targets_original = original_dataset['train_targets'][train_mask]
    
    # 重新映射标签到连续索引，然后加上全局偏移
    train_targets = np.array([label_mapping[label] + global_offset for label in train_targets_original])
    
    # 筛选测试数据
    test_mask = np.isin(original_dataset['test_targets'], class_indices)
    test_data = original_dataset['test_data'][test_mask]
    test_targets_original = original_dataset['test_targets'][test_mask]
    
    # 重新映射标签到连续索引，然后加上全局偏移
    test_targets = np.array([label_mapping[label] + global_offset for label in test_targets_original])
    
    # 筛选类别名称
    class_names = [original_dataset['class_names'][idx] for idx in class_indices]
    
    # 创建子集数据集
    split_dataset = {
        'name': f"{original_dataset['name']}_split_{split_idx}",
        'train_data': train_data,
        'test_data': test_data,
        'train_targets': train_targets,  # 使用重新映射后的全局标签
        'test_targets': test_targets,    # 使用重新映射后的全局标签
        'num_classes': len(class_indices),
        'use_path': original_dataset['use_path'],
        'class_names': class_names,
        'templates': original_dataset['templates'],
        'original_dataset_name': original_dataset['name'],
        'split_index': split_idx,
        'original_class_indices': class_indices,  # 保存原始类别索引映射
        'label_mapping': label_mapping,  # 保存标签映射关系
        'global_offset': global_offset  # 保存全局偏移量
    }
    
    return split_dataset
```

### 2. 修复 `_create_incremental_splits` 方法

需要正确计算每个子集的全局偏移量：

```python
def _create_incremental_splits(self) -> None:
    """创建增量拆分，将每个数据集拆分为多个增量子集"""
    if not self.enable_incremental_split or self.num_incremental_splits <= 1:
        return
    
    incremental_datasets = []
    incremental_global_label_offset = []
    
    current_global_offset = 0
    
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
            # 计算当前子集的全局偏移量
            split_global_offset = current_global_offset
            
            split_dataset = self._create_incremental_dataset(dataset, class_indices, split_idx, split_global_offset)
            incremental_datasets.append(split_dataset)
            incremental_global_label_offset.append(split_global_offset)
            
            # 更新全局偏移量，为下一个子集做准备
            current_global_offset += len(class_indices)
            
            logging.info(f"[BCDM]   Split {split_idx + 1}: {len(class_indices)} classes, "
                        f"{len(split_dataset['train_data'])} train samples, "
                        f"{len(split_dataset['test_data'])} test samples, "
                        f"global_offset={split_global_offset}")
    
    # 更新数据集和偏移
    self.datasets = incremental_datasets
    self.global_label_offset = incremental_global_label_offset
    self.total_classes = sum(d['num_classes'] for d in self.datasets)
    
    logging.info(f"[BCDM] After incremental split: {len(self.datasets)} total tasks, "
                f"{self.total_classes} total classes")
```

### 3. 修复 `get_task_size` 方法

