#!/usr/bin/env python3
"""
测试增量拆分标签映射修复是否有效
"""

import logging
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.balanced_cross_domain_data_manager import create_balanced_data_manager

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def test_incremental_split_label_mapping():
    """测试增量拆分的标签映射是否正确"""
    print("=" * 80)
    print("测试增量拆分的标签映射")
    print("=" * 80)
    
    # 创建平衡数据管理器（启用增量拆分）
    datasets = ['cifar100_224', 'cub200_224']
    
    manager = create_balanced_data_manager(
        dataset_names=datasets,
        balanced_datasets_root="balanced_datasets",
        use_balanced_datasets=True,
        enable_incremental_split=True,
        num_incremental_splits=3,
        incremental_split_seed=42
    )
    
    print(f"\n数据管理器创建成功:")
    print(f"  - 总任务数: {manager.nb_tasks}")
    print(f"  - 总类别数: {manager.num_classes}")
    print(f"  - 增量拆分启用: {manager.enable_incremental_split}")
    
    # 测试每个任务的标签范围
    print(f"\n测试每个任务的标签范围:")
    for task_id in range(min(6, manager.nb_tasks)):  # 测试前6个任务
        dataset_info = manager.datasets[task_id]
        global_offset = manager.global_label_offset[task_id]
        
        # 获取训练集
        train_set = manager.get_incremental_subset(
            task=task_id, source="train", cumulative=False, mode="test")
        train_loader = DataLoader(train_set, batch_size=32, shuffle=False)
        
        # 获取测试集
        test_set = manager.get_incremental_subset(
            task=task_id, source="test", cumulative=False, mode="test")
        test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
        
        # 收集所有标签
        train_labels = []
        for batch in train_loader:
            train_labels.extend(batch[1].numpy())
        
        test_labels = []
        for batch in test_loader:
            test_labels.extend(batch[1].numpy())
        
        train_labels = np.array(train_labels)
        test_labels = np.array(test_labels)
        
        print(f"\n任务 {task_id} ({dataset_info['name']}):")
        print(f"  - 数据集类别数: {dataset_info['num_classes']}")
        print(f"  - 全局偏移: {global_offset}")
        print(f"  - 训练集标签范围: {np.min(train_labels)} - {np.max(train_labels)}")
        print(f"  - 测试集标签范围: {np.min(test_labels)} - {np.max(test_labels)}")
        print(f"  - 期望标签范围: {global_offset} - {global_offset + dataset_info['num_classes'] - 1}")
        
        # 验证标签范围是否正确
        expected_min = global_offset
        expected_max = global_offset + dataset_info['num_classes'] - 1
        
        train_min_ok = np.min(train_labels) == expected_min
        train_max_ok = np.max(train_labels) == expected_max
        test_min_ok = np.min(test_labels) == expected_min
        test_max_ok = np.max(test_labels) == expected_max
        
        print(f"  - 训练集标签范围正确: {'✓' if train_min_ok and train_max_ok else '✗'}")
        print(f"  - 测试集标签范围正确: {'✓' if test_min_ok and test_max_ok else '✗'}")
        
        # 验证标签是否连续
        train_unique = np.unique(train_labels)
        test_unique = np.unique(test_labels)
        expected_labels = np.arange(expected_min, expected_max + 1)
        
        train_continuous = np.array_equal(np.sort(train_unique), expected_labels)
        test_continuous = np.array_equal(np.sort(test_unique), expected_labels)
        
        print(f"  - 训练集标签连续: {'✓' if train_continuous else '✗'}")
        print(f"  - 测试集标签连续: {'✓' if test_continuous else '✗'}")
        
        if not (train_min_ok and train_max_ok and test_min_ok and test_max_ok and 
                train_continuous and test_continuous):
            print(f"  ❌ 任务 {task_id} 测试失败!")
            return False
        else:
            print(f"  ✅ 任务 {task_id} 测试通过!")
    
    print(f"\n🎉 所有任务测试通过!")
    return True

def test_cumulative_mode():
    """测试累积模式是否正确"""
    print("\n" + "=" * 80)
    print("测试累积模式")
    print("=" * 80)
    
    # 创建平衡数据管理器（启用增量拆分）
    return True

if __name__ == "__main__":
    try:
        result = test_incremental_split_with_cross_domain()
        if result:
            print("✅ 测试通过！增量分割功能修复成功。")
            sys.exit(0)
        else:
            print("❌ 测试失败！")
            sys.exit(1)
    except Exception as e:
        print(f"❌ 测试过程中出现异常: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)