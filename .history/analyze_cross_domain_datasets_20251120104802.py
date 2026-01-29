#!/usr/bin/env python3
"""
分析cross-domain数据集中每个数据集的样本数量和类别平均数量
"""

import os
import sys
import logging
from collections import defaultdict

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.cross_domain_data_manager import CrossDomainDataManagerCore
from utils.data1 import get_dataset

def analyze_dataset(dataset_name):
    """分析单个数据集的样本数量和类别信息"""
    print(f"\n{'='*60}")
    print(f"分析数据集: {dataset_name}")
    print(f"{'='*60}")
    
    try:
        # 获取数据集
        dataset = get_dataset(dataset_name)
        
        # 获取训练和测试数据
        train_data = dataset.train_data
        train_targets = dataset.train_targets
        test_data = dataset.test_data
        test_targets = dataset.test_targets
        
        # 统计基本信息
        num_train_samples = len(train_data)
        num_test_samples = len(test_data)
        num_classes = len(dataset.class_names)
        
        # 统计每个类别的样本数量
        train_class_counts = defaultdict(int)
        test_class_counts = defaultdict(int)
        
        for label in train_targets:
            train_class_counts[label] += 1
            
        for label in test_targets:
            test_class_counts[label] += 1
        
        # 计算平均每类样本数量
        avg_train_per_class = num_train_samples / num_classes
        avg_test_per_class = num_test_samples / num_classes
        
        # 打印统计信息
        print(f"数据集名称: {dataset_name}")
        print(f"类别数量: {num_classes}")
        print(f"训练样本总数: {num_train_samples}")
        print(f"测试样本总数: {num_test_samples}")
        print(f"训练集平均每类样本数: {avg_train_per_class:.2f}")
        print(f"测试集平均每类样本数: {avg_test_per_class:.2f}")
        
        # 打印类别样本分布的详细信息
        print(f"\n训练集类别样本分布:")
        print(f"  最小样本数: {min(train_class_counts.values())}")
        print(f"  最大样本数: {max(train_class_counts.values())}")
        print(f"  标准差: {calculate_std(train_class_counts.values()):.2f}")
        
        print(f"\n测试集类别样本分布:")
        print(f"  最小样本数: {min(test_class_counts.values())}")
        print(f"  最大样本数: {max(test_class_counts.values())}")
        print(f"  标准差: {calculate_std(test_class_counts.values()):.2f}")
        
        return {
            'dataset_name': dataset_name,
            'num_classes': num_classes,
            'num_train_samples': num_train_samples,
            'num_test_samples': num_test_samples,
            'avg_train_per_class': avg_train_per_class,
            'avg_test_per_class': avg_test_per_class,
            'train_class_counts': dict(train_class_counts),
            'test_class_counts': dict(test_class_counts)
        }
        
    except Exception as e:
        print(f"分析数据集 {dataset_name} 时出错: {str(e)}")
        return None

def calculate_std(values):
    """计算标准差"""
    import numpy as np
    return np.std(list(values))

def analyze_cross_domain_datasets():
    """分析cross-domain实验中使用的所有数据集"""
    
    # 从main.py中获取默认的cross-domain数据集列表
    default_datasets = [
        'cifar100_224', 'cub200_224', 'resisc45', 'imagenet-r', 'caltech-101', 
        'dtd', 'fgvc-aircraft-2013b-variants102', 'food-101', 'mnist', 
        'oxford-flower-102', 'oxford-iiit-pets', 'cars196_224'
    ]
    
    print("Cross-Domain数据集分析报告")
    print("=" * 80)
    
    all_results = []
    
    # 分析每个数据集
    for dataset_name in default_datasets:
        result = analyze_dataset(dataset_name)
        if result:
            all_results.append(result)
    
    # 生成汇总报告
    print(f"\n{'='*80}")
    print("汇总报告")
    print(f"{'='*80}")
    
    print(f"数据集总数: {len(all_results)}")
    
    total_train_samples = sum(r['num_train_samples'] for r in all_results)
    total_test_samples = sum(r['num_test_samples'] for r in all_results)
    total_classes = sum(r['num_classes'] for r in all_results)
    
    print(f"总训练样本数: {total_train_samples}")
    print(f"总测试样本数: {total_test_samples}")
    print(f"总类别数: {total_classes}")
    
    print(f"\n各数据集测试样本数量对比:")
    print(f"{'数据集名称':<30} {'类别数':<10} {'测试样本数':<15} {'平均每类样本数':<15}")
    print("-" * 70)
    
    # 按测试样本数量排序
    sorted_results = sorted(all_results, key=lambda x: x['num_test_samples'])
    
    for result in sorted_results:
        print(f"{result['dataset_name']:<30} {result['num_classes']:<10} "
              f"{result['num_test_samples']:<15} {result['avg_test_per_class']:<15.2f}")
    
    # 分析样本不平衡性
    print(f"\n测试样本不平衡性分析:")
    test_samples = [r['num_test_samples'] for r in all_results]
    min_samples = min(test_samples)
    max_samples = max(test_samples)
    ratio = max_samples / min_samples
    
    print(f"最小测试样本数: {min_samples} ({sorted_results[0]['dataset_name']})")
    print(f"最大测试样本数: {max_samples} ({sorted_results[-1]['dataset_name']})")
    print(f"不平衡比率: {ratio:.2f}x")
    
    # 计算变异系数
    import numpy as np
    mean_samples = np.mean(test_samples)
    std_samples = np.std(test_samples)
    cv = std_samples / mean_samples
    print(f"变异系数: {cv:.3f}")
    
    return all_results

def analyze_cross_domain_manager():
    """使用CrossDomainDataManagerCore分析数据集"""
    print(f"\n{'='*80}")
    print("使用CrossDomainDataManagerCore分析")
    print(f"{'='*80}")
    
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
        num_shots=0,  # 不使用few-shot采样
        num_samples_per_task_for_evaluation=0,  # 不限制评估样本数
        log_level=logging.WARNING  # 减少日志输出
    )
    
    print(f"总任务数: {manager.nb_tasks}")
    print(f"总类别数: {manager.num_classes}")
    
    print(f"\n各任务(数据集)详细信息:")
    print(f"{'任务ID':<8} {'数据集名称':<30} {'类别数':<10} {'训练样本数':<15} {'测试样本数':<15}")
    print("-" * 80)
    
    for task_id in range(manager.nb_tasks):
        dataset_info = manager.datasets[task_id]
        train_samples = len(dataset_info['train_data'])
        test_samples = len(dataset_info['test_data'])
        num_classes = dataset_info['num_classes']
        dataset_name = dataset_info['name']
        
        print(f"{task_id:<8} {dataset_name:<30} {num_classes:<10} "
              f"{train_samples:<15} {test_samples:<15}")
    
    return manager

if __name__ == "__main__":
    # 设置日志级别
    logging.basicConfig(level=logging.WARNING)
    
    # 分析各个数据集
    results = analyze_cross_domain_datasets()
    
    # 使用CrossDomainDataManagerCore分析
    manager = analyze_cross_domain_manager()
    
    print(f"\n分析完成！")