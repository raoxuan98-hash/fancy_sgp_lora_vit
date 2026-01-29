
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
