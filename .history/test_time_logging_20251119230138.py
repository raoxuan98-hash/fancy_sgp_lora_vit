
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试时间记录功能的脚本
"""

import torch
import logging
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 导入需要测试的模块
from classifier.classifier_builder import ClassifierReconstructor
from compensator.distribution_compensator import DistributionCompensator
from compensator.gaussian_statistics import GaussianStatistics


def create_test_data():
    """创建测试数据"""
    # 创建模拟的特征和标签
    batch_size = 32
    feature_dim = 768
    num_classes = 10
    num_samples = 1000
    
    # 生成随机数据
    features = torch.randn(num_samples, feature_dim)
    labels = torch.randint(0, num_classes, (num_samples,))
    
    # 创建数据加载器
    dataset = TensorDataset(features, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader, feature_dim, num_classes


def create_test_models(feature_dim, num_classes):
    """创建测试用的简单模型"""
    class SimpleModel(torch.nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.fc = torch.nn.Linear(input_dim, output_dim)
            
        def forward(self, x):
            return self.fc(x)
    
    model_before = SimpleModel(feature_dim, num_classes)
    model_after = SimpleModel(feature_dim, num_classes)
    
    return model_before, model_after


def create_test_gaussian_stats(feature_dim, num_classes):
    """创建测试用的高斯统计数据"""
    stats_dict = {}
    for class_id in range(num_classes):
        mean = torch.randn(feature_dim)
        cov = torch.eye(feature_dim) * 0.1 + torch.randn(feature_dim, feature_dim) * 0.01
        cov = 0.5 * (cov + cov.T)  # 确保对称
        stats_dict[class_id] = GaussianStatistics(mean, cov, cholesky=True)
    
    return stats_dict


def test_classifier_builder():
    """测试分类器构建器的时间记录功能"""
    print("\n" + "="*50)
    print("测试分类器构建器的时间记录功能")
    print("="*50)
    
    # 创建测试数据
    feature_dim = 768
