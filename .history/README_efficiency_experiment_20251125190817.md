
# 分类器计算效率对比实验

本项目实现了一个全面的计算效率对比实验，用于比较不同分类器在各种类别数量下的性能表现。

## 实验概述

### 对比的分类器
1. **Full-rank QDA** - 完整秩的二次判别分析
2. **Low-rank QDA** - 低秩近似二次判别分析 (r=1, 8, 16, 32)
3. **SGD-based linear classifier** - 基于随机梯度下降的线性分类器
4. **LDA** - 线性判别分析

### 实验变量
- **自变量**: 类别数量 (50, 100, 200, 500, 1000)
- **因变量**: 
  - 分类器构建时间 (秒)
  - 推理时间 (毫秒/样本)
  - 内存使用 (MB)

### 实验设置
- **重复次数**: 3次
- **测试样本数**: 1000个样本 (用于测量推理时间)
- **特征维度**: 768 (ViT-B/16特征)
- **设备**: CUDA

## 快速开始

### 环境准备
```bash
# 安装依赖
pip install torch torchvision timm matplotlib numpy tqdm

# 设置GPU
export CUDA_VISIBLE_DEVICES=0
```

### 运行实验

#### 基本实验
```bash
python exp_efficiency_comparison.py
```

#### 自定义参数实验
```bash
python exp_efficiency_comparison.py \
    --model vit-b-p16 \
    --gpu 0 \
    --num_shots 64 \
    --class_counts 50 100 200 500 1000 \
    --num_repeats 3 \
    --output_dir 实验结果保存/效率对比实验
```

#### 小规模快速测试
```bash
python exp_efficiency_comparison.py \
    --class_counts 10 20 50 \
    --num_repeats 1 \
    --model vit-b-p16-clip
```

### 测试实验脚本
```bash
# 运行测试脚本
python test_efficiency_experiment.py
```

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | str | vit-b-p16-clip | 模型名称，支持vit-b-p16, vit-b-p16-clip等 |
| `--gpu` | str | 0 | GPU编号 |
| `--num_shots` | int | 128 | 每类样本数 |
| `--class_counts` | int+ | [50, 100, 200, 500, 1000] | 类别数量列表 |
| `--num_repeats` | int | 3 | 重复实验次数 |
| `--output_dir` | str | 实验结果保存/效率对比实验 | 结果保存目录 |

## 输出文件

实验完成后，会在指定的输出目录中生成以下文件：

### 数据文件
- `efficiency_results_{model_name}.npz` - 完整的实验结果数据
- `efficiency_results_{model_name}.csv` - CSV格式的实验结果

### 可视化图表
- `build_time_comparison.png` - 构建时间对比图
- `inference_time_comparison.png` - 推理时间对比图  
- `memory_usage_comparison.png` - 内存使用对比图

### 日志文件
- `experiment_log.txt` - 详细的实验日志

## 结果解读

### 构建时间图
- **X轴**: 类别数量（对数坐标）
