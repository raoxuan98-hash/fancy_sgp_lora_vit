#!/usr/bin/env python3
"""
测试在实际训练场景中，启用增量拆分时是否会出现标签超出范围的问题
"""

import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.balanced_cross_domain_data_manager import create_balanced_data_manager
from models.subspace_lora import SubspaceLoRA

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def create_simple_model(num_classes):
    """创建一个简单的分类模型用于测试"""
    class SimpleModel(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
            self.classifier = nn.Linear(64, num_classes)
        
        def forward(self, x):
            features = self.feature_extractor(x)
            logits = self.classifier(features)
            return logits, features
    
    return SimpleModel(num_classes)

def test_training_with_incremental_split():
    """测试在增量拆分模式下训练是否会出错"""
    print("=" * 80)
    print("测试增量拆分模式下的训练")
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
