# Cross-Domain数据集重新划分修复计划

## 问题分析

### 主要问题
在`dataset_resplitter.py`的第323-324行，当处理样本不足的类别时，代码会清空测试集数据：
```python
balanced_test_data.clear()
balanced_test_targets.clear()
```
这导致之前处理的类别的测试样本全部丢失，最终测试集几乎为空。

### 次要问题
1. 对于总样本数小于128的类别，20%/80%的分割策略可能不够合理
2. 缺乏最小样本数保护机制

## 修复方案

### 方案设计
采用**优化方案**，确保：
1. 修复清空bug
2. 对于小样本类别，采用更合理的分配策略
3. 添加最小样本数保护机制

### 具体策略

#### 1. 样本分配策略
- **总样本数 >= 128**：确保至少64个训练样本和64个测试样本
- **总样本数 < 128**：按比例分配，但确保至少1个训练样本和1个测试样本
- **修复清空bug**：不再清空测试集，而是正确追加样本

#### 2. 最小样本数保护
- 训练集：至少1个样本
- 测试集：至少1个样本
- 如果总样本数=1：分配给训练集

#### 3. 实现细节

```python
# 新的样本分配逻辑
if total_count >= self.max_samples_per_class * 2:
    # 足够样本，确保训练集和测试集各有max_samples_per_class
    train_reserve = min(self.max_samples_per_class, train_count)
    test_reserve = min(self.max_samples_per_class, test_count)
    
    if test_count < self.max_samples_per_class:
        # 从训练集补充到测试集
        needed = self.max_samples_per_class - test_count
        if train_count > needed:
            # 移动部分训练样本到测试集
            move_indices = np.random.choice(train_indices, size=needed, replace=False)
            # ... 处理移动逻辑
        else:
            # 所有训练样本都移动到测试集也不够
            # ... 处理不足情况
else:
    # 小样本情况，按比例分配
    if total_count == 1:
        # 只有一个样本，分配给训练集
        train_reserve = 1
        test_reserve = 0
    else:
        # 确保至少1个训练样本和1个测试样本
        train_reserve = max(1, min(train_count, total_count // 2))
        test_reserve = total_count - train_reserve
```

## 实施步骤

### 步骤1：修改_balance_dataset方法
- 移除清空测试集的代码
- 实现新的样本分配逻辑
- 添加最小样本数保护

### 步骤2：更新元数据记录
- 记录新的采样策略
- 更新统计信息

### 步骤3：测试验证
- 测试dtd数据集（小样本）
- 测试mnist数据集（大样本）
- 验证测试集不为空

### 步骤4：更新文档
- 更新代码注释
- 更新README文档

## 预期效果

### dtd数据集（47类，每类80个样本）
- 原始：训练集752样本，测试集64样本（几乎为空）
- 修复后：训练集~752样本，测试集~1280样本（合理分布）

### mnist数据集（10类，每类约6000/1000个样本）
- 原始：训练集1280样本，测试集1280样本
- 修复后：训练集1280样本，测试集1280样本（保持不变）

### cifar100数据集（100类，每类500/100个样本）
- 原始：训练集12800样本，测试集12800样本
- 修复后：训练集12800样本，测试集12800样本（保持不变）

## 兼容性

修复后的代码将保持与现有`BalancedCrossDomainDataManagerCore`的完全兼容，无需修改数据管理器代码。