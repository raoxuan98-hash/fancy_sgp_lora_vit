#!/usr/bin/env python3
"""
展示修复后的aggregate_results.json文件内容示例
"""

import json

# 修复后的aggregate_results.json文件内容示例
example_content = {
  "final_task_stats": {
    "SeqFT + LDA": {
      "mean": 75.0,
      "std": 0.82
    },
    "SeqFT + QDA": {
      "mean": 80.0,
      "std": 0.82
    },
    "SeqFT + attention_transform + LDA": {
      "mean": 82.0,
      "std": 0.82
    },
    "SeqFT + attention_transform + QDA": {
      "mean": 85.0,
      "std": 0.82
    }
  },
  "average_across_tasks_stats": {
    "SeqFT + LDA": {
      "mean": 78.0,
      "std": 0.82
    },
    "SeqFT + QDA": {
      "mean": 83.0,
      "std": 0.82
    },
    "SeqFT + attention_transform + LDA": {
      "mean": 85.0,
      "std": 0.82
    },
    "SeqFT + attention_transform + QDA": {
      "mean": 88.0,
      "std": 0.82
    }
  },
  "per_task_accuracy_trends": {
    "SeqFT + LDA": {
      "means": [70.0, 75.0, 80.0],
      "stds": [0.82, 0.82, 0.82],
      "num_tasks": 3
    },
    "SeqFT + QDA": {
      "means": [75.0, 80.0, 85.0],
      "stds": [0.82, 0.82, 0.82],
      "num_tasks": 3
    },
    "SeqFT + attention_transform + LDA": {
      "means": [77.0, 82.0, 87.0],
      "stds": [0.82, 0.82, 0.82],
      "num_tasks": 3
    },
    "SeqFT + attention_transform + QDA": {
      "means": [80.0, 85.0, 90.0],
      "stds": [0.82, 0.82, 0.82],
      "num_tasks": 3
    }
  },
  "seed_list": [
    "seed_1993",
    "seed_1996",
    "seed_1997"
  ],
  "num_seeds": 3,
  "timestamp": "2025-11-05 11:30:00",
  "variants": [
    "SeqFT + LDA",
    "SeqFT + QDA",
    "SeqFT + attention_transform + LDA",
    "SeqFT + attention_transform + QDA"
  ],
  "max_tasks": 3
}

print("修复后的 aggregate_results.json 文件内容示例:")
print("=" * 80)
print(json.dumps(example_content, indent=2, ensure_ascii=False))
print("=" * 80)

print("\n📊 关键字段说明:")
print("1. final_task_stats: 最终任务准确率的均值和标准差")
print("2. average_across_tasks_stats: 所有任务平均准确率的均值和标准差")
print("3. per_task_accuracy_trends: 每个任务的准确度趋势（均值和标准差）")
print("4. seed_list: 包含的种子列表")
print("5. num_seeds: 种子数量")
print("6. variants: 所有变体（方法）列表")
print("7. max_tasks: 最大任务数")

print("\n✅ 修复前的问题:")
print("- 标准差(std)为0，因为只聚合了一个种子的结果")
print("- seed_list只包含一个种子")

print("\n🎉 修复后的改进:")
print("- 标准差(std)大于0，正确反映了多个种子之间的差异")
print("- seed_list包含所有种子（如seed_1993, seed_1996, seed_1997）")
print("- num_seeds显示正确的种子数量")