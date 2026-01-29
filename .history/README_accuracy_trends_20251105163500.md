# 准确度趋势功能说明

## 概述

本功能修改了 `main.py` 和 `trainer.py`，使得运行 `main.py` 后生成的 `aggregate_results.json` 文件包含每个任务结束后的评测结果列表，方便呈现准确度随任务数量增加的下降趋势。

## 主要修改

### 1. trainer.py 中的 aggregate_seed_results 函数

- 新增了 `per_task_accuracies` 字典，用于收集每个任务的准确度列表
- 新增了 `per_task_stats` 计算，包含每个任务的平均准确度和标准差
- 在保存的 JSON 文件中新增了 `per_task_accuracy_trends` 字段

### 2. main.py 中的 _pretty_print_aggregate 函数

- 新增了对 `per_task_accuracy_trends` 数据的显示
- 在控制台输出中添加了每个任务的准确度趋势信息

## 新的 aggregate_results.json 格式

```json
{
  "final_task_stats": {
    "SeqFT + LDA": {"mean": 68.08, "std": 0.0},
    "SeqFT + QDA": {"mean": 73.8, "std": 0.0}
  },
  "average_across_tasks_stats": {
    "SeqFT + LDA": {"mean": 74.619, "std": 0.0},
    "SeqFT + QDA": {"mean": 79.076, "std": 0.0}
  },
  "per_task_accuracy_trends": {
    "SeqFT + LDA": {
      "means": [85.5, 82.3, 78.9, 75.2, 72.1, 68.08],
      "stds": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      "num_tasks": 6
    },
    "SeqFT + QDA": {
      "means": [88.2, 85.1, 81.7, 78.3, 75.6, 73.8],
      "stds": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      "num_tasks": 6
    }
  },
  "seed_list": ["seed_1993"],
  "num_seeds": 1,
  "timestamp": "2025-11-05 16:33:00",
  "variants": ["SeqFT + LDA", "SeqFT + QDA"],
  "max_tasks": 6
}
```

## 使用方法

### 1. 运行实验

```bash
python main.py --dataset cifar100_224 --init_cls 10 --increment 10 --iterations 2000 --smart_defaults
```

### 2. 查看结果

实验完成后，在日志目录中会生成 `aggregate_results.json` 文件，其中包含 `per_task_accuracy_trends` 字段。

### 3. 可视化趋势

使用提供的可视化脚本生成准确度趋势图：

```bash
python visualize_accuracy_trends.py
```

该脚本会：
- 自动查找最新的 `aggregate_results.json` 文件
- 生成准确度随任务数量变化的趋势图
- 保存为 PNG 图片
- 在控制台输出准确度下降的统计数据

## 示例输出

```
📈 准确度趋势数据摘要:
  SeqFT + LDA:
    初始准确度: 85.50%
    最终准确度: 68.08%
    下降幅度: 17.42% (20.4%)
  SeqFT + QDA:
    初始准确度: 88.20%
    最终准确度: 73.80%
    下降幅度: 14.40% (16.3%)
```

## 测试工具

### test_aggregate_results.py

用于验证 `aggregate_results.json` 文件格式是否正确：

```bash
python test_aggregate_results.py
```

该脚本会：
- 检查文件是否包含 `per_task_accuracy_trends` 字段
- 验证数据格式是否正确
- 如果找不到有效文件，会创建模拟文件进行测试

## 注意事项

1. 现有的 `aggregate_results.json` 文件（在修改前生成的）不包含 `per_task_accuracy_trends` 字段
2. 需要使用修改后的代码重新运行实验才能获得完整的数据
3. 可视化脚本中的中文字体警告不影响功能，只是显示问题

## 文件清单

- `trainer.py` - 修改了 `aggregate_seed_results` 函数
- `main.py` - 修改了 `_pretty_print_aggregate` 函数
- `test_aggregate_results.py` - 测试脚本
- `visualize_accuracy_trends.py` - 可视化脚本
- `README_accuracy_trends.md` - 本说明文档