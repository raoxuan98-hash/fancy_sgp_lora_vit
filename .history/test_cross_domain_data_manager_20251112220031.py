#!/usr/bin/env python3
"""
测试跨域数据管理器的标签偏移逻辑
"""
import logging
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.cross_domain_data_manager import CrossDomainDataManager

def test_cross_domain_data_manager():
    """测试跨域数据管理器的标签偏移逻辑"""
    
    # 设置日志级别
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 使用较小的数据集进行测试
    dataset_names = ['mnist', 'cifar100_224']  # 使用两个数据集进行测试
    
    print("=" * 60)
    print("测试跨域数据管理器的标签偏移逻辑")
    print("=" * 60)
    
    # 创建跨域数据管理器
    cdm = CrossDomainDataManager(
        dataset_names=dataset_names,
        shuffle=False,
        seed=42
    )
    
    print(f"总任务数: {cdm.nb_tasks}")
    print(f"总类别数: {cdm.num_classes}")
    
    # 测试每个任务的非累积模式
    for task_id in range(cdm.nb_tasks):
        print(f"\n--- 任务 {task_id} (非累积模式) ---")
        task_size = cdm.get_task_size(task_id)
        print(f"任务 {task_id} 的类别数: {task_size}")
        
        # 获取训练数据集
        train_dataset = cdm.get_subset(task_id, source="train", cumulative=False)
        print(f"训练集大小: {len(train_dataset)}")
        
        # 检查标签范围
        if len(train_dataset) > 0:
            # 获取前几个样本检查标签
            for i in range(min(5, len(train_dataset))):
                _, label, class_name = train_dataset[i]
                print(f"  样本 {i}: 标签={label}, 类名={class_name}")
        
        # 获取测试数据集
        test_dataset = cdm.get_subset(task_id, source="test", cumulative=False)
        print(f"测试集大小: {len(test_dataset)}")
    
    # 测试累积模式
    for task_id in range(cdm.nb_tasks):
        print(f"\n--- 任务 {task_id} (累积模式) ---")
        
        # 获取累积训练数据集
        train_dataset = cdm.get_subset(task_id, source="train", cumulative=True)
        print(f"累积训练集大小: {len(train_dataset)}")
        
        # 检查标签范围
        if len(train_dataset) > 0:
            # 获取前几个样本检查标签
            for i in range(min(5, len(train_dataset))):
                _, label, class_name = train_dataset[i]
                print(f"  样本 {i}: 标签={label}, 类名={class_name}")
        
        # 获取累积测试数据集
        test_dataset = cdm.get_subset(task_id, source="test", cumulative=True)
        print(f"累积测试集大小: {len(test_dataset)}")

if __name__ == "__main__":
    test_cross_domain_data_manager()