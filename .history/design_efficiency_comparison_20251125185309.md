
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
