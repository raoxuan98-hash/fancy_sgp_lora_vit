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
