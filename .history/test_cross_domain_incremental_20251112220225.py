#!/usr/bin/env python3
"""
测试跨域数据管理器的类增量学习场景
"""
import logging
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_manager import DataManager

def test_cross_domain_incremental():
    """测试跨域数据管理器的类增量学习场景"""
    
    # 设置日志级别
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("=" * 60)
    print("测试跨域数据管理器的类增量学习场景")
    print("=" * 60)
    
    # 创建跨域数据管理器
    args = {
        'cross_domain_datasets': ['mnist', 'cifar100_224']  # 使用两个数据集进行测试
    }
    
    dm = DataManager(
        dataset_name='cross_domain_elevater',
        shuffle=False,
        seed=42,
        init_cls=0,  # 在跨域模式下，这个参数被忽略
        increment=0,  # 在跨域模式下，这个参数被忽略
        args=args
    )
    
    print(f"总任务数: {dm.nb_tasks}")
    
    # 模拟类增量学习过程
    for task_id in range(dm.nb_tasks):
        print(f"\n--- 任务 {task_id} ---")
        
        # 获取当前任务的大小
        task_size = dm.get_task_size(task_id)
        print(f"任务 {task_id} 的类别数: {task_size}")
        
        # 获取当前任务的类别（非累积）
        task_classes = dm.get_task_classes(task_id, cumulative=False)
        print(f"任务 {task_id} 的类别（非累积）: {task_classes[:5]}...{task_classes[-5:] if len(task_classes) > 5 else ''}")
        
        # 获取累积类别
        cum_classes = dm.get_task_classes(task_id, cumulative=True)
        print(f"任务 {task_id} 的类别（累积）: {cum_classes[:5]}...{cum_classes[-5:] if len(cum_classes) > 5 else ''}")
        
        # 获取训练数据集（非累积）
        train_dataset = dm.get_subset(task_id, source="train", cumulative=False, mode="train")
        print(f"训练集大小（非累积）: {len(train_dataset)}")
        
        # 获取训练数据集（累积）
        cum_train_dataset = dm.get_subset(task_id, source="train", cumulative=True, mode="train")
        print(f"训练集大小（累积）: {len(cum_train_dataset)}")
        
        # 检查前几个样本的标签
        print("前5个训练样本（非累积）:")
        for i in range(min(5, len(train_dataset))):
            _, label, class_name = train_dataset[i]
            print(f"  样本 {i}: 标签={label}, 类名={class_name}")
        
        print("前5个训练样本（累积）:")
        for i in range(min(5, len(cum_train_dataset))):
            _, label, class_name = cum_train_dataset[i]
            print(f"  样本 {i}: 标签={label}, 类名={class_name}")

if __name__ == "__main__":
    test_cross_domain_incremental()