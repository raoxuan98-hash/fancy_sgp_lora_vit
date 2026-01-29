#!/usr/bin/env python3
"""
测试 get_incremental_subset 方法的 cumulative 参数功能
"""

import os
import sys
import logging
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.balanced_cross_domain_data_manager import create_balanced_data_manager

def test_cumulative_functionality():
    """测试 cumulative 参数功能"""
    print("=== 测试 get_incremental_subset 的 cumulative 参数功能 ===\n")
    
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # 创建测试用的数据管理器（启用增量拆分）
    datasets = ['cifar100_224', 'cub200_224']
    
    try:
        manager = create_balanced_data_manager(
            dataset_names=datasets,
            balanced_datasets_root="balanced_datasets",
            use_balanced_datasets=True,
            enable_incremental_split=True,
            num_incremental_splits=3,
            incremental_split_seed=42
        )
        
        print(f"✅ 成功创建数据管理器")
        print(f"   总任务数: {manager.nb_tasks}")
        print(f"   总类别数: {manager.num_classes}")
        print(f"   数据集名称: {manager.dataset_names}")
        
        # 显示增量拆分统计信息
        incremental_stats = manager.get_incremental_statistics()
        print(f"\n📊 增量拆分统计信息:")
        for original_name, stat in incremental_stats.items():
            print(f"   原始数据集: {original_name}")
            print(f"     拆分数: {stat['num_splits']}")
            print(f"     总类别: {stat['total_classes']}")
            print(f"     拆分信息:")
            for split in stat['splits']:
                print(f"       拆分 {split['split_index']} (任务 {split['task_id']}): {split['num_classes']} 类别")
        
        print(f"\n🧪 测试 cumulative=False (非累积模式)")
        for task_id in range(min(3, manager.nb_tasks)):  # 测试前3个任务
            try:
                # 测试非累积模式
                subset_non_cumulative = manager.get_incremental_subset(
                    task=task_id, 
                    source="test", 
                    cumulative=False
                )
                print(f"   任务 {task_id} (非累积): 数据集长度 = {len(subset_non_cumulative)}")
                
                # 获取数据集信息
                dataset_info = manager.datasets[task_id]
                print(f"     数据集名称: {dataset_info['name']}")
                print(f"     类别数: {dataset_info['num_classes']}")
                print(f"     标签范围: {dataset_info['test_targets'].min()} - {dataset_info['test_targets'].max()}")
                
            except Exception as e:
                print(f"   ❌ 任务 {task_id} 测试失败: {e}")
                return False
        
        print(f"\n🧪 测试 cumulative=True (累积模式)")
        for task_id in range(min(3, manager.nb_tasks)):  # 测试前3个任务
            try:
                # 测试累积模式
                subset_cumulative = manager.get_incremental_subset(
                    task=task_id, 
                    source="test", 
                    cumulative=True
                )
                print(f"   任务 {task_id} (累积): 数据集长度 = {len(subset_cumulative)}")
                
                # 验证累积数据应该包含所有之前任务的数据
                total_expected_samples = 0
                for i in range(task_id + 1):
                    total_expected_samples += len(manager.datasets[i]['test_data'])
                
                if len(subset_cumulative) >= total_expected_samples:
                    print(f"     ✅ 累积模式验证通过 (期望: {total_expected_samples}, 实际: {len(subset_cumulative)})")
                else:
                    print(f"     ❌ 累积模式验证失败 (期望: {total_expected_samples}, 实际: {len(subset_cumulative)})")
                    return False
                
            except Exception as e:
                print(f"   ❌ 任务 {task_id} 累积模式测试失败: {e}")
                return False
        
        print(f"\n🧪 测试向后兼容性 (不指定 cumulative 参数)")
        for task_id in range(min(2, manager.nb_tasks)):
            try:
                # 默认应该为 cumulative=False
                subset_default = manager.get_incremental_subset(
                    task=task_id, 
                    source="test"
                )
                subset_explicit_false = manager.get_incremental_subset(
                    task=task_id, 
                    source="test", 
                    cumulative=False
                )
                
                if len(subset_default) == len(subset_explicit_false):
                    print(f"   ✅ 任务 {task_id}: 向后兼容性测试通过")
                else:
                    print(f"   ❌ 任务 {task_id}: 向后兼容性测试失败")
                    return False
                    
            except Exception as e:
                print(f"   ❌ 任务 {task_id} 向后兼容性测试失败: {e}")
                return False
        
        print(f"\n🎉 所有测试通过！get_incremental_subset 的 cumulative 参数功能正常")
        return True
        
    except Exception as e:
        print(f"❌ 创建数据管理器失败: {e}")
        return False

def test_compatibility_with_subspace_lora():
    """测试与 subspace_lora.py 的兼容性"""
    print(f"\n=== 测试与 subspace_lora.py 的兼容性 ===\n")
    
    try:
        # 模拟 subspace_lora.py 中的使用方式
        datasets = ['cifar100_224']
        
        manager = create_balanced_data_manager(
            dataset_names=datasets,
            balanced_datasets_root="balanced_datasets",
            use_balanced_datasets=True,
            enable_incremental_split=True,
            num_incremental_splits=2,
            incremental_split_seed=42
        )
        
        task_id = 0
        
        # 模拟 subspace_lora.py 中的调用方式
        print("🧪 模拟 subspace_lora.py 中的使用模式:")
        
        # 训练集 (cumulative=False)
        train_set = manager.get_incremental_subset(
            task=task_id, source="train", cumulative=False, mode="train")
        print(f"   训练集 (cumulative=False): {len(train_set)} 样本")
        
        # 测试集 (cumulative=True)
        test_set = manager.get_incremental_subset(
            task=task_id, source="test", cumulative=True, mode="test")
        print(f"   测试集 (cumulative=True): {len(test_set)} 样本")
        
        # 训练集测试模式 (cumulative=False)
        train_set_test_mode = manager.get_incremental_subset(
            task=task_id, source="train", cumulative=False, mode="test")
        print(f"   训练集测试模式 (cumulative=False): {len(train_set_test_mode)} 样本")
        
        print(f"✅ 与 subspace_lora.py 的兼容性测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 与 subspace_lora.py 的兼容性测试失败: {e}")
        return False

def main():
    """主函数"""
    print("开始测试 get_incremental_subset 的 cumulative 参数功能\n")
    
    # 检查平衡数据集是否存在
    balanced_datasets_root = Path("balanced_datasets")
    if not balanced_datasets_root.exists():
        print("⚠️  警告: balanced_datasets 目录不存在，某些测试可能失败")
    
    # 运行测试
    test1_passed = test_cumulative_functionality()
    test2_passed = test_compatibility_with_subspace_lora()
    
    if test1_passed and test2_passed:
        print(f"\n🎉 所有测试通过！功能实施成功")
        return True
    else:
        print(f"\n❌ 部分测试失败，请检查实现")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)