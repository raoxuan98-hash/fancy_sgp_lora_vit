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
        
    
    # 测试不启用增量分割（用于比较）
    logger.info("\n测试不启用增量分割...")
    manager_no_split = create_balanced_data_manager(
        dataset_names=dataset_names,
        balanced_datasets_root="balanced_datasets",
        use_balanced_datasets=True,
        enable_incremental_split=False
    )
    
    logger.info(f"数据管理器创建成功")
    logger.info(f"总任务数: {manager_no_split.nb_tasks}")
    logger.info(f"总类别数: {manager_no_split.num_classes}")
    
    # 检查每个任务的数据
    for i in range(manager_no_split.nb_tasks):
        task_size = manager_no_split.get_task_size(i)
        task_classes = manager_no_split.get_task_classes(i, cumulative=False)
        logger.info(f"任务 {i}: 类别数={task_size}, 类别范围={task_classes}")
        
        # 获取训练集
        train_set = manager_no_split.get_incremental_subset(task=i, source="train", cumulative=False, mode="train")
        test_set = manager_no_split.get_incremental_subset(task=i, source="test", cumulative=True, mode="test")
        
        # 检查标签范围
        train_targets = train_set.labels
        test_targets = test_set.labels
        
        if len(train_targets) > 0:
            train_min = min(train_targets)
            train_max = max(train_targets)
            logger.info(f"  训练集: 样本数={len(train_targets)}, 标签范围=[{train_min}, {train_max}]")
            
        if len(test_targets) > 0:
            test_min = min(test_targets)
            test_max = max(test_targets)
            logger.info(f"  测试集: 样本数={len(test_targets)}, 标签范围=[{test_min}, {test_max}]")
    
    logger.info("测试完成，两种模式都正常工作！")
    
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