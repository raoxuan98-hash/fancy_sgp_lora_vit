#!/usr/bin/env python3
"""
可视化脚本：展示准确度随任务数量增加的下降趋势
使用aggregate_results.json中的per_task_accuracy_trends数据
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

def visualize_accuracy_trends(aggregate_file):
    """
    可视化准确度随任务数量增加的下降趋势
    
    Args:
        aggregate_file: aggregate_results.json文件路径
    """
    try:
        with open(aggregate_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return
    
    # 检查是否包含per_task_accuracy_trends字段
    if "per_task_accuracy_trends" not in data:
        print("❌ 文件中不包含per_task_accuracy_trends字段")
        print("💡 请确保使用修改后的代码运行main.py生成新的aggregate_results.json")
        return
    
    trends = data["per_task_accuracy_trends"]
    variants = list(trends.keys())
    
    if not variants:
        print("❌ 没有找到任何变体的准确度趋势数据")
        return
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # 为每个变体绘制准确度趋势线
    for variant in variants:
        trend_data = trends[variant]
        means = trend_data.get("means", [])
        stds = trend_data.get("stds", [])
        
        if not means:
            continue
        
        # 任务编号（从1开始）
        task_ids = list(range(1, len(means) + 1))
        
        # 绘制主趋势线
        plt.plot(task_ids, means, marker='o', linewidth=2, label=variant)
        
        # 绘制标准差范围（如果有）
        if stds and any(s > 0 for s in stds):
            means_array = np.array(means)
            stds_array = np.array(stds)
            plt.fill_between(task_ids, 
                           means_array - stds_array, 
                           means_array + stds_array, 
                           alpha=0.2)
    
    # 设置图形属性
    plt.xlabel('任务编号', fontsize=14)
    plt.ylabel('准确度 (%)', fontsize=14)
    plt.title('准确度随任务数量增加的变化趋势', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形
    output_file = Path(aggregate_file).parent / "accuracy_trends.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 准确度趋势图已保存到: {output_file}")
    
    # 显示图形（如果在交互环境中）
    try:
        plt.show()
    except:
        print("💡 无法显示图形，但已保存到文件")
    
    # 打印数据摘要
    print("\n📈 准确度趋势数据摘要:")
    for variant in variants:
        trend_data = trends[variant]
        means = trend_data.get("means", [])
        if means:
            initial_acc = means[0]
            final_acc = means[-1]
            drop = initial_acc - final_acc
            drop_rate = drop / initial_acc * 100
            
            print(f"  {variant}:")
            print(f"    初始准确度: {initial_acc:.2f}%")
            print(f"    最终准确度: {final_acc:.2f}%")
            print(f"    下降幅度: {drop:.2f}% ({drop_rate:.1f}%)")

def create_sample_visualization():
    """使用模拟数据创建示例可视化"""
    # 模拟数据
    sample_data = {
        "per_task_accuracy_trends": {
            "SeqFT + LDA": {
                "means": [85.5, 82.3, 78.9, 75.2, 72.1, 68.08],
                "stds": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                "num_tasks": 6
            },
            "SeqFT + QDA": {
                "means": [88.2, 85.1, 81.7, 78.3, 75.6, 73.8],
                "stds": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                "num_tasks": 6
            },
            "SeqFT + attention_transform + QDA": {
                "means": [90.1, 87.5, 84.2, 81.0, 78.5, 76.2],
                "stds": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                "num_tasks": 6
            }
        }
    }
    
    # 创建临时文件
    temp_file = Path("temp_sample_data.json")
    with open(temp_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2)
    
    print("📊 使用模拟数据创建示例可视化...")
    visualize_accuracy_trends(str(temp_file))
    
    # 清理临时文件
    temp_file.unlink()
    print("✅ 示例可视化完成")

if __name__ == "__main__":
    print("🎨 准确度趋势可视化工具")
    print("=" * 50)
    
    # 查找最新的aggregate_results.json文件
    log_dirs = [
        "sldc_logs_sgp_lora_vit_main",
        "sldc_logs_sgp_lora",
        "sldc_logs_sgp_lora_test",
        "test_results"
    ]
    
    found_files = []
    for log_dir in log_dirs:
        if os.path.exists(log_dir):
            for root, dirs, files in os.walk(log_dir):
                if "aggregate_results.json" in files:
                    found_files.append(os.path.join(root, "aggregate_results.json"))
    
    if found_files:
        # 使用最新的文件
        latest_file = max(found_files, key=os.path.getmtime)
        print(f"📄 使用文件: {latest_file}")
        visualize_accuracy_trends(latest_file)
    else:
        print("📄 未找到aggregate_results.json文件，使用模拟数据创建示例...")
        create_sample_visualization()