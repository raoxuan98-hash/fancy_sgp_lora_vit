#!/usr/bin/env python3
"""
调试跨域训练中的数据索引错误
"""

import os
import sys
import logging
import numpy as np
from torch.utils.data import DataLoader

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(filename)s] => %(message)s')

# 导入相关模块
from utils.cross_domain_data_manager import CrossDomainDataManager
from utils.data_manager import DataManager

def test_cross_domain_data_manager():
    """测试跨域数据管理器"""
    print("=" * 80)
    print("测试跨域数据管理器")
    print("=" * 80)
    
    # 使用较小的数据集进行测试
    dataset_names = ['imagenet-r', 'caltech-101']  # 只用两个数据集测试
    
    try:
        cdm = CrossDomainDataManager(
            dataset_names=dataset_names,
            shuffle=False,
            seed=42
        )
        
        print(f"总任务数: {cdm.nb_tasks}")
        print(f"总类别数: {cdm.num_classes}")
        
        # 测试每个任务的非累积模式
        for task_id in range(cdm.nb_tasks):
            print(f"\n--- 测试任务 {task_id} (非累积模式) ---")
            task_size = cdm.get_task_size(task_id)
            print(f"任务大小: {task_size}")
            
            # 获取训练集
            train_set = cdm.get_subset(task_id, source="train", cumulative=False)
            print(f"训练集大小: {len(train_set)}")
            
            # 获取测试集
            test_set = cdm.get_subset(task_id, source="test", cumulative=False)
            print(f"测试集大小: {len(test_set)}")
            
            # 尝试访问几个样本
            try:
                for i in range(min(5, len(train_set))):
                    sample = train_set[i]
                    print(f"  样本 {i}: 标签={sample[1]}, 类名={sample[2]}")
            except Exception as e:
                print(f"  错误: {e}")
                
        # 测试累积模式（这里可能出现问题）
        print(f"\n--- 测试累积模式 ---")
        for task_id in range(cdm.nb_tasks):
            print(f"\n--- 测试任务 {task_id} (累积模式) ---")
            
            # 获取累积训练集
            try:
                cumulative_train_set = cdm.get_subset(task_id, source="train", cumulative=True)
                print(f"累积训练集大小: {len(cumulative_train_set)}")
                
                # 尝试访问几个样本
                for i in range(min(5, len(cumulative_train_set))):
                    sample = cumulative_train_set[i]
                    print(f"  样本 {i}: 标签={sample[1]}, 类名={sample[2]}")
                    
            except Exception as e:
                print(f"  累积训练集错误: {e}")
                import traceback
                traceback.print_exc()
                
            # 获取累积测试集
            try:
                cumulative_test_set = cdm.get_subset(task_id, source="test", cumulative=True)
                print(f"累积测试集大小: {len(cumulative_test_set)}")
                
                # 尝试访问几个样本
                for i in range(min(5, len(cumulative_test_set))):
                    sample = cumulative_test_set[i]
                    print(f"  样本 {i}: 标签={sample[1]}, 类名={sample[2]}")
                    
            except Exception as e:
                print(f"  累积测试集错误: {e}")
                import traceback
                traceback.print_exc()
                
    except Exception as e:
        print(f"创建 CrossDomainDataManager 失败: {e}")
        import traceback
        traceback.print_exc()

def test_data_manager():
    """测试数据管理器包装器"""
    print("\n" + "=" * 80)
    print("测试数据管理器包装器")
    print("=" * 80)
    
    args = {
        'cross_domain': True,
        'cross_domain_datasets': ['imagenet-r', 'caltech-101'],  # 只用两个数据集测试
        'init_cls': 10,
        'increment': 10
    }
    
    try:
        dm = DataManager(
            dataset_name='cross_domain_elevater',
            shuffle=False,
            seed=42,
            init_cls=10,
            increment=10,
            args=args
        )
        
        print(f"总任务数: {dm.nb_tasks}")
        
        # 测试每个任务
        for task_id in range(dm.nb_tasks):
            print(f"\n--- 测试任务 {task_id} ---")
            task_size = dm.get_task_size(task_id)
            print(f"任务大小: {task_size}")
            
            # 获取训练集（非累积）
            try:
                train_set = dm.get_subset(task_id, source="train", cumulative=False, mode="train")
                print(f"训练集大小: {len(train_set)}")
                
                # 尝试访问几个样本
                for i in range(min(3, len(train_set))):
                    sample = train_set[i]
                    print(f"  训练样本 {i}: 标签={sample[1]}, 类名={sample[2]}")
                    
            except Exception as e:
                print(f"  训练集错误: {e}")
                import traceback
                traceback.print_exc()
                
            # 获取测试集（累积）- 这里可能出现问题
            try:
                test_set = dm.get_subset(task_id, source="test", cumulative=True, mode="test")
                print(f"测试集大小: {len(test_set)}")
                
                # 尝试访问几个样本
                for i in range(min(3, len(test_set))):
                    sample = test_set[i]
                    print(f"  测试样本 {i}: 标签={sample[1]}, 类名={sample[2]}")
                    
            except Exception as e:
                print(f"  测试集错误: {e}")
                import traceback
                traceback.print_exc()
                
    except Exception as e:
        print(f"创建 DataManager 失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("开始调试跨域数据索引错误...")
    
    # 测试跨域数据管理器
    test_cross_domain_data_manager()
    
    # 测试数据管理器包装器
    test_data_manager()
    
    print("\n调试完成！")