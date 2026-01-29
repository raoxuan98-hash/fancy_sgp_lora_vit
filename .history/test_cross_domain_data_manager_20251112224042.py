import sys
import os
sys.path.append('.')

import numpy as np
from utils.cross_domain_data_manager import CrossDomainDataManagerCore

def test_label_offset():
    """测试标签偏移是否正确应用"""
    print("=== 测试跨域数据管理器的标签偏移 ===\n")
    
    # 使用较小的数据集进行测试
    dataset_names = ['cifar100_224', 'mnist', 'caltech-101']
    
    try:
        # 初始化数据管理器
        cdm = CrossDomainDataManagerCore(dataset_names, log_level=20)  # 使用INFO级别日志
        
        print(f"总任务数: {cdm.nb_tasks}")
        print(f"总类别数: {cdm.num_classes}")
        print(f"全局标签偏移: {cdm.global_label_offset}")
        print()
        
        # 检查每个数据集的标签范围
        for i, dataset in enumerate(cdm.datasets):
            train_targets = dataset['train_targets']
            test_targets = dataset['test_targets']
            
            print(f"数据集 {i} ({dataset['name']}):")
            print(f"  训练集标签范围: {np.min(train_targets)} - {np.max(train_targets)}")
            print(f"  测试集标签范围: {np.min(test_targets)} - {np.max(test_targets)}")
            print(f"  类别数: {dataset['num_classes']}")
            print(f"  期望偏移: {cdm.global_label_offset[i]}")
            print()
        
        # 测试非累积模式
        print("=== 测试非累积模式 ===")
        for task_id in range(cdm.nb_tasks):
            train_subset = cdm.get_subset(task_id, source="train", cumulative=False)
            train_labels = np.array([train_subset[i][1] for i in range(min(100, len(train_subset)))])
            
            print(f"任务 {task_id} 非累积训练集:")
            print(f"  标签范围: {np.min(train_labels)} - {np.max(train_labels)}")
            print(f"  期望范围: {cdm.global_label_offset[task_id]} - {cdm.global_label_offset[task_id] + cdm.get_task_size(task_id) - 1}")
            print()
        
        # 测试累积模式
        print("=== 测试累积模式 ===")
        for task_id in range(cdm.nb_tasks):
            train_subset = cdm.get_subset(task_id, source="train", cumulative=True)
            train_labels = np.array([train_subset[i][1] for i in range(min(100, len(train_subset)))])
            
            expected_max = sum(cdm.get_task_size(i) for i in range(task_id + 1)) - 1
            
            print(f"任务 {task_id} 累积训练集:")
            print(f"  标签范围: {np.min(train_labels)} - {np.max(train_labels)}")
            print(f"  期望范围: 0 - {expected_max}")
            print()
        
        # 检查是否有重复的数据集
        print(f"实际数据集数量: {len(cdm.datasets)}")
        print(f"预期数据集数量: {len(dataset_names)}")
        
        if len(cdm.datasets) != len(dataset_names):
            print("警告: 数据集数量不匹配，可能存在重复添加!")
        
        # 检查global_class_names是否有重复
        print(f"全局类名数量: {len(cdm.global_class_names)}")
        print(f"预期类名数量: {cdm.num_classes}")
        
        if len(cdm.global_class_names) != cdm.num_classes:
            print("警告: 全局类名数量与总类别数不匹配，可能存在重复添加!")
            
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_label_offset()