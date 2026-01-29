#!/usr/bin/env python3
"""
可视化cross-domain数据集的样本不平衡性
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.cross_domain_data_manager import CrossDomainDataManagerCore
import logging

def setup_visualization_style():
    """设置可视化样式"""
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
def create_dataset_comparison_plot():
    """创建数据集对比图"""
    # 默认数据集列表
    dataset_names = [
        'cifar100_224', 'cub200_224', 'resisc45', 'imagenet-r', 'caltech-101', 
        'dtd', 'fgvc-aircraft-2013b-variants102', 'food-101', 'mnist', 
        'oxford-flower-102', 'oxford-iiit-pets', 'cars196_224'
    ]
    
    # 创建数据管理器
    manager = CrossDomainDataManagerCore(
        dataset_names=dataset_names,
        shuffle=False,
        seed=0,
        num_shots=0,
        num_samples_per_task_for_evaluation=0,
        log_level=logging.WARNING
    )
    
    # 提取数据
    dataset_info = []
    for task_id in range(manager.nb_tasks):
        dataset = manager.datasets[task_id]
        dataset_info.append({
            'name': dataset['name'],
            'num_classes': dataset['num_classes'],
            'train_samples': len(dataset['train_data']),
            'test_samples': len(dataset['test_data']),
            'avg_train_per_class': len(dataset['train_data']) / dataset['num_classes'],
            'avg_test_per_class': len(dataset['test_data']) / dataset['num_classes']
        })
    
    # 按测试样本数排序
    dataset_info.sort(key=lambda x: x['test_samples'])
    
    # 创建图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Cross-Domain数据集分析', fontsize=16, fontweight='bold')
    
    # 1. 测试样本数量对比
    names = [d['name'] for d in dataset_info]
    test_samples = [d['test_samples'] for d in dataset_info]
    
    bars1 = ax1.barh(names, test_samples, color='skyblue')
    ax1.set_xlabel('测试样本数量')
    ax1.set_title('各数据集测试样本数量对比')
    ax1.grid(axis='x', alpha=0.3)
    
    # 添加数值标签
    for i, bar in enumerate(bars1):
        width = bar.get_width()
        ax1.text(width + width*0.01, bar.get_y() + bar.get_height()/2, 
                f'{int(width)}', ha='left', va='center')
    
    # 2. 类别数量对比
    num_classes = [d['num_classes'] for d in dataset_info]
    bars2 = ax2.barh(names, num_classes, color='lightcoral')
    ax2.set_xlabel('类别数量')
    ax2.set_title('各数据集类别数量对比')
    ax2.grid(axis='x', alpha=0.3)
    
    # 3. 平均每类测试样本数对比
    avg_test_per_class = [d['avg_test_per_class'] for d in dataset_info]
    bars3 = ax3.barh(names, avg_test_per_class, color='lightgreen')
    ax3.set_xlabel('平均每类测试样本数')
    ax3.set_title('各数据集平均每类测试样本数对比')
    ax3.grid(axis='x', alpha=0.3)
    
    # 4. 训练vs测试样本数散点图
    train_samples = [d['train_samples'] for d in dataset_info]
    scatter = ax4.scatter(train_samples, test_samples, 
                          c=num_classes, s=100, alpha=0.7, cmap='viridis')
    ax4.set_xlabel('训练样本数量')
    ax4.set_ylabel('测试样本数量')
    ax4.set_title('训练vs测试样本数量关系')
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('类别数量')
    
    # 添加数据集标签
    for i, name in enumerate(names):
        ax4.annotate(name, (train_samples[i], test_samples[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('cross_domain_dataset_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return dataset_info

def create_imbalance_analysis(dataset_info):
    """创建不平衡性分析图"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Cross-Domain数据集不平衡性分析', fontsize=16, fontweight='bold')
    
    names = [d['name'] for d in dataset_info]
    test_samples = [d['test_samples'] for d in dataset_info]
    num_classes = [d['num_classes'] for d in dataset_info]
    avg_test_per_class = [d['avg_test_per_class'] for d in dataset_info]
    
    # 1. 测试样本数量分布直方图
    ax1.hist(test_samples, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('测试样本数量')
    ax1.set_ylabel('数据集数量')
    ax1.set_title('测试样本数量分布')
    ax1.grid(True, alpha=0.3)
    
    # 添加统计信息
    mean_samples = np.mean(test_samples)
    std_samples = np.std(test_samples)
    ax1.axvline(mean_samples, color='red', linestyle='--', label=f'平均值: {mean_samples:.0f}')
    ax1.axvline(mean_samples + std_samples, color='orange', linestyle='--', alpha=0.7, label=f'+1σ: {mean_samples + std_samples:.0f}')
    ax1.axvline(mean_samples - std_samples, color='orange', linestyle='--', alpha=0.7, label=f'-1σ: {mean_samples - std_samples:.0f}')
    ax1.legend()
    
    # 2. 类别数量分布直方图
    ax2.hist(num_classes, bins=8, alpha=0.7, color='lightcoral', edgecolor='black')
    ax2.set_xlabel('类别数量')
    ax2.set_ylabel('数据集数量')
    ax2.set_title('类别数量分布')
    ax2.grid(True, alpha=0.3)
    
    # 3. 平均每类样本数分布
    ax3.hist(avg_test_per_class, bins=8, alpha=0.7, color='lightgreen', edgecolor='black')
    ax3.set_xlabel('平均每类测试样本数')
    ax3.set_ylabel('数据集数量')
    ax3.set_title('平均每类测试样本数分布')
    ax3.grid(True, alpha=0.3)
    
    # 4. 不平衡性指标雷达图
    categories = ['样本数量', '类别数量', '平均每类样本数']
    
    # 计算归一化值（0-1范围）
    max_test = max(test_samples)
    max_classes = max(num_classes)
    max_avg = max(avg_test_per_class)
    
    normalized_test = [s/max_test for s in test_samples]
    normalized_classes = [c/max_classes for c in num_classes]
    normalized_avg = [a/max_avg for a in avg_test_per_class]
    
    # 选择几个代表性数据集进行展示
    selected_indices = [0, len(dataset_info)//2, len(dataset_info)-1]  # 最小、中间、最大
    selected_names = [names[i] for i in selected_indices]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形
    
    ax4 = plt.subplot(2, 2, 4, projection='polar')
    
    colors = ['blue', 'green', 'red']
    for idx, dataset_idx in enumerate(selected_indices):
        values = [
            normalized_test[dataset_idx],
            normalized_classes[dataset_idx],
            normalized_avg[dataset_idx]
        ]
        values += values[:1]  # 闭合图形
        
        ax4.plot(angles, values, 'o-', linewidth=2, label=selected_names[idx], color=colors[idx])
        ax4.fill(angles, values, alpha=0.25, color=colors[idx])
    
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(categories)
    ax4.set_ylim(0, 1)
    ax4.set_title('数据集特征对比（归一化）')
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig('cross_domain_imbalance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_statistics(dataset_info):
    """创建汇总统计表格"""
    print("\n" + "="*80)
    print("数据集不平衡性详细统计")
    print("="*80)
    
    # 提取数据
    names = [d['name'] for d in dataset_info]
    test_samples = [d['test_samples'] for d in dataset_info]
    num_classes = [d['num_classes'] for d in dataset_info]
    avg_test_per_class = [d['avg_test_per_class'] for d in dataset_info]
    
    # 计算统计指标
    print(f"\n测试样本数量统计:")
    print(f"  最小值: {min(test_samples)} ({names[test_samples.index(min(test_samples))]})")
    print(f"  最大值: {max(test_samples)} ({names[test_samples.index(max(test_samples))]})")
    print(f"  平均值: {np.mean(test_samples):.2f}")
    print(f"  标准差: {np.std(test_samples):.2f}")
    print(f"  变异系数: {np.std(test_samples)/np.mean(test_samples):.3f}")
    print(f"  不平衡比率: {max(test_samples)/min(test_samples):.2f}x")
    
    print(f"\n类别数量统计:")
    print(f"  最小值: {min(num_classes)} ({names[num_classes.index(min(num_classes))]})")
    print(f"  最大值: {max(num_classes)} ({names[num_classes.index(max(num_classes))]})")
    print(f"  平均值: {np.mean(num_classes):.2f}")
    print(f"  标准差: {np.std(num_classes):.2f}")
    
    print(f"\n平均每类测试样本数统计:")
    print(f"  最小值: {min(avg_test_per_class):.2f} ({names[avg_test_per_class.index(min(avg_test_per_class))]})")
    print(f"  最大值: {max(avg_test_per_class):.2f} ({names[avg_test_per_class.index(max(avg_test_per_class))]})")
    print(f"  平均值: {np.mean(avg_test_per_class):.2f}")
    print(f"  标准差: {np.std(avg_test_per_class):.2f}")
    
    # 计算相关性
    correlation_matrix = np.corrcoef([test_samples, num_classes, avg_test_per_class])
    print(f"\n相关性矩阵:")
    print(f"  测试样本数 vs 类别数: {correlation_matrix[0,1]:.3f}")
    print(f"  测试样本数 vs 平均每类样本数: {correlation_matrix[0,2]:.3f}")
    print(f"  类别数 vs 平均每类样本数: {correlation_matrix[1,2]:.3f}")

def main():
    """主函数"""
    print("开始分析Cross-Domain数据集不平衡性...")
    
    # 设置可视化样式
    setup_visualization_style()
    
    # 创建数据集对比图
    dataset_info = create_dataset_comparison_plot()
    
    # 创建不平衡性分析图
    create_imbalance_analysis(dataset_info)
    
    # 创建汇总统计
    create_summary_statistics(dataset_info)
    
    print("\n分析完成！图像已保存为:")
    print("  - cross_domain_dataset_analysis.png")
    print("  - cross_domain_imbalance_analysis.png")

if __name__ == "__main__":
    main()