#!/usr/bin/env python3
"""
测试脚本：验证 num_samples_per_task_for_evaluation 功能
"""

import sys
import os
import argparse
import numpy as np
from torch.utils.data import DataLoader

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_manager import WithinDomainDataManager, CrossDomainDataManager

def test_within_domain_sampling():
    """测试域内数据管理器的采样功能"""
    print("=" * 50)
    print("测试域内数据管理器采样功能")
    print("=" * 50)
    
    # 创建数据管理器
    args = {
        'num_samples_per_task_for_evaluation': 100,  # 设置采样数量
        'dataset': 'cifar100_224'
    }
    
    dm = WithinDomainDataManager(
        dataset_name='cifar100_224',
        shuffle=False,
        seed=1993,
        init_cls=10,
        increment=10,
        args=args
    )
    
    # 获取测试数据集（不采样）
    test_set_full = dm.get_subset(task=0, source="test", cumulative=False, mode="test")
    print(f"完整测试集大小: {len(test_set_full)}")
    
    # 临时修改采样数量
    dm.num_samples_per_task_for_evaluation = 50
    
    # 获取测试数据集（采样）
    test_set_sampled = dm.get_subset(task=0, source="test", cumulative=False, mode="test")
    print(f"采样后测试集大小: {len(test_set_sampled)}")
    
    # 验证采样数量是否正确
    assert len(test_set_sampled) == 50, f"采样数量不正确: 期望50，实际{len(test_set_sampled)}"
    print("✅ 域内采样功能测试通过")
    
    return True

def test_cross_domain_sampling():
    """测试跨域数据管理器的采样功能"""
    print("\n" + "=" * 50)
    print("测试跨域数据管理器采样功能")
    print("=" * 50)
    
    # 创建数据管理器
    args = {
        'num_samples_per_task_for_evaluation': 100,  # 设置采样数量
        'cross_domain_datasets': ['imagenet-r', 'cifar100_224']
    }
    
    dm = CrossDomainDataManager(
        dataset_name='cross_domain',
        shuffle=False,
        seed=1993,
        args=args
    )
    
    # 获取第一个任务的测试数据集（不采样）
    test_set_full = dm.get_subset(task=0, source="test", cumulative=False, mode="test")
    print(f"任务0完整测试集大小: {len(test_set_full)}")
    
    # 获取第二个任务的测试数据集（不采样）
    test_set_full2 = dm.get_subset(task=1, source="test", cumulative=False, mode="test")
    print(f"任务1完整测试集大小: {len(test_set_full2)}")
    
    # 临时修改采样数量
    dm._cdm.num_samples_per_task_for_evaluation = 50
    
    # 获取测试数据集（采样）
    test_set_sampled = dm.get_subset(task=0, source="test", cumulative=False, mode="test")
    print(f"任务0采样后测试集大小: {len(test_set_sampled)}")
    
    test_set_sampled2 = dm.get_subset(task=1, source="test", cumulative=False, mode="test")
    print(f"任务1采样后测试集大小: {len(test_set_sampled2)}")
    
    # 验证采样数量是否正确
    assert len(test_set_sampled) == 50, f"任务0采样数量不正确: 期望50，实际{len(test_set_sampled)}"
    assert len(test_set_sampled2) == 50, f"任务1采样数量不正确: 期望50，实际{len(test_set_sampled2)}"
    print("✅ 跨域采样功能测试通过")
    
    return True

def test_cumulative_sampling():
    """测试累积模式下的采样功能"""
    print("\n" + "=" * 50)
    print("测试累积模式采样功能")
    print("=" * 50)
    
    # 创建数据管理器
    args = {
        'num_samples_per_task_for_evaluation': 100,  # 设置采样数量
        'cross_domain_datasets': ['imagenet-r', 'cifar100_224']
    }
    
    dm = CrossDomainDataManager(
        dataset_name='cross_domain',
        shuffle=False,
        seed=1993,
        args=args
    )
    
    # 临时修改采样数量
    dm._cdm.num_samples_per_task_for_evaluation = 200
    
    # 获取累积测试数据集（采样）
    test_set_cumulative = dm.get_subset(task=1, source="test", cumulative=True, mode="test")
    print(f"累积测试集大小: {len(test_set_cumulative)}")
    
    # 验证采样数量是否正确（累积模式下应该采样到指定数量）
    assert len(test_set_cumulative) <= 200, f"累积采样数量不正确: 期望≤200，实际{len(test_set_cumulative)}"
    print("✅ 累积模式采样功能测试通过")
    
    return True

def test_no_sampling():
    """测试不启用采样的情况"""
    print("\n" + "=" * 50)
    print("测试不启用采样的情况")
    print("=" * 50)
    
    # 创建数据管理器（不启用采样）
    args = {
        'num_samples_per_task_for_evaluation': 0,  # 不采样
        'dataset': 'cifar100_224'
    }
    
    dm = WithinDomainDataManager(
        dataset_name='cifar100_224',
        shuffle=False,
        seed=1993,
        init_cls=10,
        increment=10,
        args=args
    )
    
    # 获取测试数据集
    test_set = dm.get_subset(task=0, source="test", cumulative=False, mode="test")
    original_size = len(test_set)
    
    # 再次获取测试数据集（应该相同）
    test_set2 = dm.get_subset(task=0, source="test", cumulative=False, mode="test")
    assert len(test_set2) == original_size, "不启用采样时，数据集大小应该保持不变"
    print(f"不启用采样时测试集大小: {len(test_set2)}")
    print("✅ 不启用采样功能测试通过")
    
    return True

def main():
    """运行所有测试"""
    print("开始测试 num_samples_per_task_for_evaluation 功能")
    
    try:
        # 运行各项测试
        test_within_domain_sampling()
        test_cross_domain_sampling()
        test_cumulative_sampling()
        test_no_sampling()
        
        print("\n" + "=" * 50)
        print("🎉 所有测试通过！num_samples_per_task_for_evaluation 功能正常工作")
        print("=" * 50)
        
        return True
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)