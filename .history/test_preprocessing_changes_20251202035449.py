#!/usr/bin/env python3
"""
测试脚本：验证预处理修改是否正确匹配SLCA
"""

import sys
import torch
from torchvision import transforms
from utils.data_manager1 import IncrementalDataManager

def test_dataset_preprocessing(dataset_name: str, initial_classes: int = 10, increment_classes: int = 10):
    """测试指定数据集的预处理配置"""
    print(f"\n=== 测试 {dataset_name} 预处理配置 ===")
    
    try:
        # 创建数据管理器
        dm = IncrementalDataManager(
            dataset_name=dataset_name,
            initial_classes=initial_classes,
            increment_classes=increment_classes,
            shuffle=True,
            seed=1993
        )
        
        # 测试训练预处理
        train_transform = dm._build_transform("train")
        print(f"训练预处理变换数量: {len(train_transform.transforms)}")
        for i, transform in enumerate(train_transform.transforms):
            print(f"  {i+1}. {transform}")
        
        # 测试测试预处理
        test_transform = dm._build_transform("test")
        print(f"测试预处理变换数量: {len(test_transform.transforms)}")
        for i, transform in enumerate(test_transform.transforms):
            print(f"  {i+1}. {transform}")
        
        print(f"✓ {dataset_name} 预处理配置测试成功")
        return True
        
    except Exception as e:
        print(f"✗ {dataset_name} 预处理配置测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始验证预处理修改...")
    
    # 测试数据集列表
    test_datasets = [
        ("cifar100_224", 10, 10),
        ("imagenet-r", 20, 20),
        ("cub200_224", 20, 20),
        ("cars196_224", 20, 20),
    ]
    
    success_count = 0
    total_count = len(test_datasets)
    
    for dataset_name, init_cls, inc_cls in test_datasets:
        if test_dataset_preprocessing(dataset_name, init_cls, inc_cls):
            success_count += 1
    
    print(f"\n=== 测试总结 ===")
    print(f"成功: {success_count}/{total_count}")
    print(f"失败: {total_count - success_count}/{total_count}")
    
    if success_count == total_count:
        print("✓ 所有数据集预处理配置验证通过！")
        return 0
    else:
        print("✗ 部分数据集预处理配置验证失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())