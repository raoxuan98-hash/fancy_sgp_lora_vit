# 修复标签映射逻辑计划

## 问题分析

根据错误信息，问题是标签范围（0-96）超过类别名称数量（50）。这个错误发生在处理`cifar100_224_split_0`数据集时。

### 根本原因

1. CIFAR-100数据集有100个类别（标签0-99）
2. 但在平衡数据集中，只有50个类别被包含在标签文件中
3. 当启用增量分割时，数据集被分割成多个子集，但标签映射逻辑有问题
4. 在`_create_incremental_dataset`方法中，标签没有被正确映射到局部索引
5. 在`get_subset`方法中，类别名称和标签范围不匹配

## 修复方案

### 1. 修复`_create_incremental_dataset`方法

在`utils/balanced_cross_domain_data_manager.py`文件中，需要修改`_create_incremental_dataset`方法：

```python
def _create_incremental_dataset(self, original_dataset: Dict, class_indices: List[int],
                               split_idx: int) -> Dict:
    """
    创建单个增量子集
    
    Args:
        original_dataset: 原始数据集
        class_indices: 该子集包含的类别索引（局部索引）
        split_idx: 子集索引
        
    Returns:
        增量子集的数据字典
    """
    # 筛选训练数据
    train_mask = np.isin(original_dataset['train_targets'], class_indices)
    train_data = original_dataset['train_data'][train_mask]
    train_targets = original_dataset['train_targets'][train_mask]
    
    # 筛选测试数据
    test_mask = np.isin(original_dataset['test_targets'], class_indices)
    test_data = original_dataset['test_data'][test_mask]
    test_targets = original_dataset['test_targets'][test_mask]
    
    # 创建标签映射：将原始全局标签映射到局部标签（0到len(class_indices)-1）
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(class_indices)}
    
    # 应用标签映射
    mapped_train_targets = np.array([label_mapping[label] for label in train_targets])
    mapped_test_targets = np.array([label_mapping[label] for label in test_targets])
    
    # 筛选类别名称 - 使用原始类别名称
    class_names = [original_dataset['class_names'][idx] for idx in class_indices]
    
    # 创建子集数据集
    split_dataset = {
        'name': f"{original_dataset['name']}_split_{split_idx}",
        'train_data': train_data,
        'test_data': test_data,
        'train_targets': mapped_train_targets,  # 使用映射后的局部标签
        'test_targets': mapped_test_targets,    # 使用映射后的局部标签
        'num_classes': len(class_indices),
        'use_path': original_dataset['use_path'],
        'class_names': class_names,
        'templates': original_dataset['templates'],
        'original_dataset_name': original_dataset['name'],
        'split_index': split_idx,
        'original_class_indices': class_indices  # 保存原始类别索引映射
    }
    
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

