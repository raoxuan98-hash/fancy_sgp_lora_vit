# 计算效率对比实验设计

## 实验目标
比较以下四种分类器在不同类别数量下的计算效率：
1. Full-rank QDA
2. Low-rank QDA
3. SGD-based linear classifier
4. LDA

## 实验变量
- **自变量**: 类别数量 (50, 100, 200, 500, 1000)
- **因变量**: 
  - 分类器构建时间 (秒)
  - 推理时间 (毫秒/样本)
  - 内存使用 (MB)

## 实验参数

### 分类器参数
- **Full-rank QDA**: 
  - qda_reg_alpha1=0.2, qda_reg_alpha2=0.2, qda_reg_alpha3=0.2
  - low_rank=False
  
- **Low-rank QDA**: 
  - qda_reg_alpha1=0.2, qda_reg_alpha2=0.2, qda_reg_alpha3=0.2
  - low_rank=True, rank=64
  
- **SGD-based linear classifier**: 
  - max_steps=5000, lr=1e-3
  - alpha1=0.5, alpha2=0.5, alpha3=0.5
  
- **LDA**: 
  - lda_reg_alpha=0.3

### 实验设置
- **重复次数**: 3次
- **测试样本数**: 1000个样本 (用于测量推理时间)
- **特征维度**: 768 (ViT-B/16特征)
- **设备**: CUDA

## 实验流程

### 1. 数据准备
- 加载完整数据集
- 根据指定类别数量随机选择类别子集
- 为每个类别子集构建训练和测试数据

### 2. 分类器构建时间测量
- 对每个分类器和类别数量组合：
  - 记录开始时间
  - 构建分类器
  - 记录结束时间
  - 计算构建时间
- 重复3次取平均值

### 3. 推理时间测量
- 对每个构建好的分类器：
  - 使用1000个测试样本
  - 记录批量推理时间
  - 计算每个样本的平均推理时间
- 重复3次取平均值

### 4. 内存使用测量
- 记录分类器构建前后的GPU内存使用情况
- 计算内存增量

## 实验输出

### 结果数据
- 每个分类器在不同类别数量下的平均构建时间
- 每个分类器在不同类别数量下的平均推理时间
- 每个分类器在不同类别数量下的内存使用量

### 可视化图表
1. 构建时间 vs 类别数量 (对数坐标)
2. 推理时间 vs 类别数量
3. 内存使用 vs 类别数量
4. 综合效率对比雷达图

## 实验脚本结构

```
exp_efficiency_comparison.py
├── 导入必要的库
├── 数据采样函数
├── 分类器构建时间测量函数
├── 推理时间测量函数
├── 内存使用测量函数
├── 主实验循环
├── 结果收集和统计
├── 结果可视化函数
└── 主函数
```

## 预期结果分析

### 复杂度分析
- **Full-rank QDA**: O(C*D²) 构建复杂度，O(C*D) 推理复杂度
- **Low-rank QDA**: O(C*D*r) 构建复杂度，O(C*r) 推理复杂度 (r=64)
- **SGD-based**: O(steps*B*D) 训练复杂度，O(D) 推理复杂度
- **LDA**: O(D²) 构建复杂度， O(D) 推理复杂度

其中C为类别数，D为特征维度(768)，r为秩(64)，B为批次大小，steps为训练步数(5000)

### 预期结论
1. 构建时间：Full-rank QDA > Low-rank QDA > SGD-based > LDA
2. 推理时间：Full-rank QDA > Low-rank QDA > LDA ≈ SGD-based
3. 内存使用：Full-rank QDA > Low-rank QDA > SGD-based > LDA
4. 随着类别数量增加，Full-rank QDA的构建时间和内存使用将呈二次增长

## 数据采样函数设计

### 函数签名
```python
def create_class_subset(
    full_dataset: Dataset,
    full_labels: torch.Tensor,
    num_classes: int,
    samples_per_class: int = 128,
    random_seed: int = 42
) -> Tuple[Dataset, torch.Tensor, List[int]]:
    """
    根据指定类别数量创建数据子集
    
    Args:
        full_dataset: 完整数据集
        full_labels: 完整标签张量
        num_classes: 目标类别数量
        samples_per_class: 每个类别的样本数
        random_seed: 随机种子
        
    Returns:
        subset_dataset: 子集数据集
        subset_labels: 子集标签
        selected_classes: 选中的类别列表
    """
```

### 实现步骤
1. **获取所有类别列表**: 从完整标签中提取唯一类别
2. **随机选择类别**: 使用随机种子选择指定数量的类别
3. **筛选数据**: 根据选中的类别筛选数据样本
4. **创建子集**: 返回筛选后的数据集和标签

### 类别数量选择策略
- 对于50-1000类别范围，确保均匀分布
- 优先选择样本数充足的类别
- 保证每个类别至少有128个样本

## 分类器构建时间测量函数设计

### 函数签名
```python
def measure_build_time(
    classifier_type: str,
    stats_dict: Dict[int, GaussianStatistics],
    device: str = "cuda",
    **kwargs
) -> Tuple[float, nn.Module, Dict[str, float]]:
    """
    测量分类器构建时间
    
    Args:
        classifier_type: 分类器类型 ("full_qda", "low_qda", "sgd_linear", "lda")
        stats_dict: 高斯统计量字典
        device: 计算设备
        **kwargs: 分类器特定参数
        
    Returns:
        build_time: 构建时间（秒）
        classifier: 构建好的分类器
        memory_usage: 内存使用信息（MB）
    """
```

### 实现步骤
1. **预热GPU**: 确保GPU处于活跃状态
2. **记录初始内存**: 记录构建前的GPU内存使用
3. **开始计时**: 使用time.time()记录开始时间
4. **构建分类器**: 根据类型构建相应的分类器
5. **结束计时**: 记录构建完成时间
6. **计算内存增量**: 记录构建后的GPU内存使用

### 分类器特定参数
- **Full-rank QDA**: low_rank=False
- **Low-rank QDA**: low_rank=True, rank=64
- **SGD-based linear**: linear=True, max_steps=5000
- **LDA**: lda_reg_alpha=0.3

## 推理时间测量函数设计

### 函数签名
```python
def measure_inference_time(
    classifier: nn.Module,
    test_features: torch.Tensor,
    batch_size: int = 64,
    warmup_runs: int = 10,
    measure_runs: int = 50,
    device: str = "cuda"
) -> Tuple[float, float, float]:
    """
    测量分类器推理时间
    
    Args:
        classifier: 训练好的分类器
        test_features: 测试特征
        batch_size: 批次大小
        warmup_runs: 预热运行次数
        measure_runs: 测量运行次数
        device: 计算设备
        
    Returns:
        avg_time_per_sample: 每个样本的平均推理时间（毫秒）
        avg_time_per_batch: 每个批次的平均推理时间（毫秒）
        throughput: 吞吐量（样本/秒）
    """
```

### 实现步骤
1. **数据准备**: 将测试数据移至GPU
2. **预热运行**: 执行多次推理以稳定GPU状态
3. **批量推理测量**:
   - 记录每个批次的推理时间
   - 重复测量多次取平均值
4. **计算指标**:
   - 每样本平均时间 = 总时间 / (批次大小 × 测量次数)
   - 吞吐量 = (批次大小 × 测量次数) / 总时间

### 测量注意事项
- 使用torch.cuda.synchronize()确保GPU操作完成
- 排除数据传输时间，仅计算纯推理时间
- 使用足够多的测量次数以获得稳定结果

## 主实验循环设计

### 实验流程
```python
def run_efficiency_experiment(
    class_counts: List[int] = [50, 100, 200, 500, 1000],
    classifier_types: List[str] = ["full_qda", "low_qda", "sgd_linear", "lda"],
    num_repeats: int = 3,
    model_name: str = "vit-b-p16-clip",
    num_shots: int = 128,
    device: str = "cuda"
) -> Dict[str, Dict[str, List[float]]]:
    """
    运行计算效率对比实验
    
    Args:
        class_counts: 类别数量列表
        classifier_types: 分类器类型列表
        num_repeats: 重复实验次数
        model_name: 模型名称
        num_shots: 每类样本数
        device: 计算设备
        
    Returns:
        results: 实验结果字典
    """
```

### 实验步骤
1. **数据准备**:
   - 加载完整数据集
   - 提取特征和标签
   - 构建完整统计量

2. **外层循环**: 遍历每个类别数量
   - 创建类别子集
   - 筛选对应的特征和统计量

3. **内层循环**: 遍历每个分类器类型
   - 重复实验指定次数
   - 测量构建时间和推理时间
   - 记录内存使用

4. **结果收集**: 保存所有测量结果

### 实验控制
- 使用固定随机种子确保可重复性
- 定期清理GPU缓存避免内存泄漏
- 记录实验进度和异常情况

## 结果收集和统计分析功能设计

### 结果数据结构
```python
results = {
    "build_time": {
        "full_qda": [list of times for each class count],
        "low_qda": [list of times for each class count],
        "sgd_linear": [list of times for each class count],
        "lda": [list of times for each class count]
    },
    "inference_time": {
        "full_qda": [list of times for each class count],
        "low_qda": [list of times for each class count],
        "sgd_linear": [list of times for each class count],
        "lda": [list of times for each class count]
    },
    "memory_usage": {
        "full_qda": [list of memory usage for each class count],
        "low_qda": [list of memory usage for each class count],
        "sgd_linear": [list of memory usage for each class count],
        "lda": [list of memory usage for each class count]
    }
}
```

### 统计分析函数
```python
def analyze_results(
    results: Dict,
    class_counts: List[int],
    classifier_types: List[str]
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    分析实验结果
    
    Returns:
        statistics: 包含平均值、标准差、置信区间等统计信息
    """
```

### 统计指标
- **平均值**: 重复实验的平均值
- **标准差**: 重复实验的标准差
- **95%置信区间**: 基于t分布的置信区间
- **增长率**: 相对于基线（最小类别数）的增长率
- **复杂度拟合**: 拟合理论复杂度曲线

## 实验限制和注意事项
1. 实验结果受硬件配置影响
2. SGD-based分类器的训练时间可能因收敛情况而变化
3. 低秩近似的效果可能因数据特性而异
4. 内存测量仅包括GPU显存，不包括CPU内存
5. 类别数量受限于数据集的实际类别数