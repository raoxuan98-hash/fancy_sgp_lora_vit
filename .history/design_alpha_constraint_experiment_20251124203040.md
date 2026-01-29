# 约束条件下α1-α2性能曲线实验设计

## 实验目标
在α1 + α2 = 1.0的约束条件下，采样不同的α1值，评估QDA和SGD分类器的性能，并绘制α1-准确度性能曲线进行对比。

## 现有代码分析

### QDA分类器参数使用
- [`QDAClassifierBuilder.build()`](classifier/da_classifier_builder.py:52)使用三个参数：
  - `qda_reg_alpha1`: 用于类内协方差矩阵正则化
  - `qda_reg_alpha2`: 用于类间协方差矩阵正则化  
  - `qda_reg_alpha3`: 用于球形正则化

### SGD分类器参数使用（当前）
- [`SGDClassifierBuilder.build()`](classifier/sgd_classifier_builder.py:22)当前只使用：
  - `alpha1`: 用于协方差矩阵正则化
  - `alpha3`: 用于球形正则化
  - **不使用alpha2参数**

## 实验设计

### 1. 修改SGD分类器
需要修改[`SGDClassifierBuilder.build()`](classifier/sgd_classifier_builder.py:22)方法，使其像QDA一样使用α2参数：

```python
# 当前实现（第59行）
covs_reg = alpha1 * covs_sym + alpha3 * sph

# 修改后
covs_reg = alpha1 * covs_sym + alpha2 * global_cov + alpha3 * sph
```

### 2. 约束条件实现
在α1 + α2 = 1.0约束下：
- 采样α1 ∈ [0, 1]
- 计算α2 = 1.0 - α1
- α3固定为较小值（如0.01）

### 3. 实验流程

#### 数据准备阶段
- 复用[`load_cross_domain_data()`](classifier_ablation/data/data_loader.py)
- 复用特征提取逻辑
- 构建高斯统计量

#### 评估阶段
对于每个α1值：
1. 计算α2 = 1.0 - α1
2. 构建QDA分类器并评估
3. 构建SGD分类器并评估
4. 记录两个分类器的准确度

#### 绘图阶段
- x轴：α1值
- y轴：准确度
- 两条曲线：QDA性能曲线 vs SGD性能曲线

### 4. 关键函数设计

#### evaluate_with_alpha_constraint()
```python
def evaluate_with_alpha_constraint(alpha1, stats, features, targets, dataset_ids, 
                                 classifier_type="qda", alpha3=0.01, device="cuda"):
    """
    在约束条件下评估分类器性能
    """
    alpha2 = 1.0 - alpha1
    
    if classifier_type == "qda":
        builder = QDAClassifierBuilder(
            qda_reg_alpha1=alpha1,
            qda_reg_alpha2=alpha2,
            qda_reg_alpha3=alpha3,
            device=device
        )
        classifier = builder.build(stats)
    elif classifier_type == "sgd":
        builder = SGDClassifierBuilder(device=device)
        classifier = builder.build(stats, alpha1=alpha1, alpha2=alpha2, alpha3=alpha3)
    
    # 评估逻辑...
    return accuracy
```

#### plot_alpha_constraint_performance()
```python
def plot_alpha_constraint_performance(alpha1_values, qda_accuracies, sgd_accuracies, save_path=None):
    """
    绘制约束条件下的性能曲线对比图
    """
    plt.figure(figsize=(3.5, 2.5))
    plt.plot(alpha1_values, qda_accuracies, 'b-', label='QDA', marker='o', markersize=4)
    plt.plot(alpha1_values, sgd_accuracies, 'r--', label='SGD', marker='s', markersize=4)
    plt.xlabel(r'$\alpha_1$')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    # 其他美化设置...
```

### 5. 实验参数设置
- α1采样点：11个点（0, 0.1, 0.2, ..., 1.0）
- α3固定值：0.01
- 模型架构：vit-b-p16-clip
- 数据集：继续使用现有的跨域数据集

### 6. 输出结果
- 性能曲线图（PNG格式，600 DPI）
- 实验数据（NPZ格式，包含α1值和两个分类器的准确度）

## 实现步骤

1. **修改SGD分类器** - 添加α2参数支持
2. **创建新实验脚本** - `exp2_alpha_constraint.py`
3. **实现评估函数** - 约束条件下的分类器评估
4. **实现绘图函数** - 双曲线对比图
5. **添加结果保存** - 数据和图表保存
6. **测试验证** - 运行完整实验流程

## 代码结构

```
classifier_ablation/experiments/
├── exp1_performance_surface.py    # 现有实验
└── exp2_alpha_constraint.py       # 新实验脚本（待创建）
```

新脚本将复用现有实验的数据加载、特征提取和评估逻辑，专注于实现约束条件下的性能对比。