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
    datasets = ['cifar100_224', 'cub200_224']
    
    manager = create_balanced_data_manager(
        dataset_names=datasets,
        balanced_datasets_root="balanced_datasets",
        use_balanced_datasets=True,
        enable_incremental_split=True,
        num_incremental_splits=3,
        incremental_split_seed=42
    )
    
    # 测试累积模式
    for task_id in [1, 3, 5]:  # 测试几个任务
        if task_id >= manager.nb_tasks:
            continue
            
        print(f"\n测试任务 {task_id} 的累积模式:")
        
        # 获取累积测试集
        cumulative_test_set = manager.get_incremental_subset(
            task=task_id, source="test", cumulative=True, mode="test")
        cumulative_test_loader = DataLoader(cumulative_test_set, batch_size=32, shuffle=False)
        
        # 收集所有标签
        cumulative_labels = []
        for batch in cumulative_test_loader:
            cumulative_labels.extend(batch[1].numpy())
        
        cumulative_labels = np.array(cumulative_labels)
        
        # 计算期望的标签范围
        total_classes_up_to_task = sum(manager.datasets[i]['num_classes'] for i in range(task_id + 1))
        expected_min = 0
        expected_max = total_classes_up_to_task - 1
        
        print(f"  - 累积标签范围: {np.min(cumulative_labels)} - {np.max(cumulative_labels)}")
        print(f"  - 期望标签范围: {expected_min} - {expected_max}")
        
        # 验证标签范围
        min_ok = np.min(cumulative_labels) == expected_min
        max_ok = np.max(cumulative_labels) == expected_max
        
        print(f"  - 累积标签范围正确: {'✓' if min_ok and max_ok else '✗'}")
        
        # 验证标签是否包含所有期望的标签
        unique_labels = np.unique(cumulative_labels)
        expected_labels = np.arange(expected_min, expected_max + 1)
        
        # 注意：由于增量拆分，可能不是所有标签都出现，但范围应该是正确的
        range_ok = (np.min(unique_labels) >= expected_min and 
                   np.max(unique_labels) <= expected_max)
        
        print(f"  - 累积标签在期望范围内: {'✓' if range_ok else '✗'}")
        
        if not (min_ok and max_ok and range_ok):
            print(f"  ❌ 任务 {task_id} 累积模式测试失败!")
            return False
        else:
            print(f"  ✅ 任务 {task_id} 累积模式测试通过!")
    
    print(f"\n🎉 所有累积模式测试通过!")
    return True

def test_training_simulation():
    """模拟训练过程，测试是否会报错"""
    print("\n" + "=" * 80)
    print("模拟训练过程")
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
    
    # 模拟训练循环
    for task_id in range(min(3, manager.nb_tasks)):  # 只测试前3个任务
        print(f"\n模拟任务 {task_id} 训练:")
        
        try:
            # 获取训练集和测试集
            train_set = manager.get_incremental_subset(
                task=task_id, source="train", cumulative=False, mode="train")
            test_set = manager.get_incremental_subset(
                task=task_id, source="test", cumulative=True, mode="test")
            
            train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
            test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
            
            # 获取一个批次的数据
            train_batch = next(iter(train_loader))
            test_batch = next(iter(test_loader))
            
            train_inputs, train_targets = train_batch[0], train_batch[1]
            test_inputs, test_targets = test_batch[0], test_batch[1]
            
            print(f"  - 训练批次: 输入形状 {train_inputs.shape}, 目标形状 {train_targets.shape}")
            print(f"  - 训练标签范围: {train_targets.min().item()} - {train_targets.max().item()}")
            print(f"  - 测试批次: 输入形状 {test_inputs.shape}, 目标形状 {test_targets.shape}")
            print(f"  - 测试标签范围: {test_targets.min().item()} - {test_targets.max().item()}")
            
            # 模拟分类器创建（简化版）
            task_size = manager.get_task_size(task_id)
            total_classes = sum(manager.datasets[i]['num_classes'] for i in range(task_id + 1))
            
            print(f"  - 任务类别数: {task_size}")
            print(f"  - 累积类别数: {total_classes}")
            
            # 模拟标签处理
            known_classes = sum(manager.datasets[i]['num_classes'] for i in range(task_id))
            new_targets_rel = torch.where(
                train_targets - known_classes >= 0,
                train_targets - known_classes, -100)
            
            print(f"  - 已知类别数: {known_classes}")
            print(f"  - 相对标签范围: {new_targets_rel.min().item()} - {new_targets_rel.max().item()}")
            
            # 检查是否有标签超出范围
            valid_targets = new_targets_rel[new_targets_rel >= 0]
            if len(valid_targets) > 0:
                max_valid_target = valid_targets.max().item()
                if max_valid_target >= task_size:
                    print(f"  ❌ 标签 {max_valid_target} 超出任务大小 {task_size}!")
                    return False
                else:
                    print(f"  ✅ 标签范围正确 (0-{task_size-1})")
            
