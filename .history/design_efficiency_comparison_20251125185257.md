
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
