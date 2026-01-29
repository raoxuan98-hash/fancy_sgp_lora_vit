
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
