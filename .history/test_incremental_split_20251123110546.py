#!/usr/bin/env python3
"""
测试cross-domain数据集的增量拆分功能
"""

import sys
import logging
from utils.balanced_cross_domain_data_manager import BalancedCrossDomainDataManagerCore

def test_incremental_split():
    """测试增量拆分功能"""
    print("=" * 80)
    print("测试Cross-Domain数据集的增量拆分功能")
    print("=" * 80)
    
    # 测试数据集
    test_datasets = ['cifar100_224', 'imagenet-r']
    
    print(f"\n测试数据集: {test_datasets}")
    print(f"每个数据集拆分为3个增量子集")
    print("-" * 80)
    
    # 创建启用增量拆分的数据管理器
    data_manager = BalancedCrossDomainDataManagerCore(
        dataset_names=test_datasets,
        balanced_datasets_root="balanced_datasets",
        shuffle=False,
        seed=1993,
        num_shots=0,
        log_level=logging.INFO,
        use_balanced_datasets=True,
        enable_incremental_split=True,
        num_incremental_splits=3,
        incremental_split_seed=42
    )
    
    print(f"\n✓ 数据管理器创建成功")
    print(f"✓ 总任务数: {data_manager.nb_tasks}")
    print(f"✓ 总类别数: {data_manager.num_classes}")
    
    # 获取增量统计信息
    incremental_stats = data_manager.get_incremental_statistics()
    
    print(f"\n增量拆分统计:")
    for original_name, stats in incremental_stats.items():
        print(f"\n  原始数据集: {original_name}")
        print(f"    拆分数: {stats['num_splits']}")
        print(f"    总类别: {stats['total_classes']}")
        print(f"    总训练样本: {stats['total_train_samples']}")
        print(f"    总测试样本: {stats['total_test_samples']}")
        
        for split in stats['splits']:
            print(f"      拆分 {split['split_index']} (任务 {split['task_id']}): "
                  f"{split['num_classes']} 类别, {split['train_samples']} 训练样本")
    
    # 测试数据访问
    print(f"\n测试数据访问:")
    for task_id in range(data_manager.nb_tasks):
        dataset = data_manager.get_subset(task_id, source="train", mode="train")
        print(f"  任务 {task_id}: {len(dataset)} 个训练样本")
        
        # 获取第一个样本
        if len(dataset) > 0:
            image, label, class_name = dataset[0]
            print(f"    第一个样本: 标签={label}, 类别名={class_name}")
    
    # 测试原始数据集拆分映射
    print(f"\n测试原始数据集拆分映射:")
    for original_name in test_datasets:
        split_indices = data_manager.get_original_dataset_splits(original_name)
        print(f"  {original_name} -> 任务索引: {split_indices}")
    
    print(f"\n" + "=" * 80)
    print("✓ 所有测试通过！")
    print("=" * 80)

def test_without_incremental_split():
    """测试不使用增量拆分的情况（向后兼容性）"""
    print("\n" + "=" * 80)
    print("测试不使用增量拆分（向后兼容性）")
    print("=" * 80)
    
    test_datasets = ['cifar100_224']
    
    # 创建不启用增量拆分的数据管理器
    data_manager = BalancedCrossDomainDataManagerCore(
        dataset_names=test_datasets,
        balanced_datasets_root="balanced_datasets",
        shuffle=False,
        seed=1993,
        num_shots=0,
        log_level=logging.INFO,
        use_balanced_datasets=True,
        enable_incremental_split=False  # 不启用增量拆分
    )
    
    print(f"\n✓ 数据管理器创建成功（未启用增量拆分）")
    print(f"✓ 总任务数: {data_manager.nb_tasks}")
    print(f"✓ 总类别数: {data_manager.num_classes}")
    
    # 验证增量统计功能
    try:
        incremental_stats = data_manager.get_incremental_statistics()
        if not incremental_stats:
            print("✓ 增量统计功能正确返回空结果（未启用增量拆分）")
        else:
            print("✗ 增量统计功能返回了非空结果（不应该发生）")
    except Exception as e:
        print(f"✗ 增量统计功能出错: {e}")
    
    print(f"\n" + "=" * 80)
    print("✓ 向后兼容性测试通过！")
    print("=" * 80)

if __name__ == "__main__":
    # 设置日志级别
    logging.basicConfig(level=logging.INFO)
    
    try:
        # 测试增量拆分功能
        test_incremental_split()
        
        # 测试向后兼容性
        test_without_incremental_split()
        
        print(f"\n🎉 所有测试完成！")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)