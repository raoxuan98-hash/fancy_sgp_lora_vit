#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 SGD 分类器构建器的功能
"""

import torch
import numpy as np
import logging
from compensator.gaussian_statistics import GaussianStatistics
from classifier.sgd_classifier_builder import SGDClassifierBuilder

def create_mock_stats_dict(num_classes=10, feature_dim=512):
    """创建模拟的统计数据字典"""
    stats_dict = {}
    
    for class_id in range(num_classes):
        # 创建模拟的均值向量
        mean = torch.randn(feature_dim)
        
        # 创建模拟的协方差矩阵（正定矩阵）
        cov = torch.randn(feature_dim, feature_dim)
        cov = torch.mm(cov, cov.t()) + torch.eye(feature_dim) * 1e-3
        
        # 创建 GaussianStatistics 对象
        stats = GaussianStatistics()
        stats.mean = mean
        stats.cov = cov
        stats_dict[class_id] = stats
    
    return stats_dict

def test_sgd_classifier():
    """测试 SGD 分类器的构建和训练"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(filename)s] => %(message)s')
    
    print("开始测试 SGD 分类器...")
    
    # 创建模拟数据
    print("创建模拟统计数据...")
    stats_dict = create_mock_stats_dict(num_classes=5, feature_dim=256)
    print(f"创建了 {len(stats_dict)} 个类别的统计数据，每个类别特征维度为 {stats_dict[0].mean.size(0)}")
    
    # 创建 SGD 分类器构建器
    print("创建 SGD 分类器构建器...")
    sgd_builder = SGDClassifierBuilder(
        cached_Z=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_steps=50,  # 减少步数用于测试
        lr=5e-4
    )
    
    # 构建分类器
    print("开始构建 SGD 分类器...")
    try:
        classifier = sgd_builder.build(stats_dict, linear=True)
        print("✅ SGD 分类器构建成功!")
        print(f"分类器类型: {type(classifier)}")
        print(f"分类器参数: {sum(p.numel() for p in classifier.parameters())}")
        
        # 测试分类器预测
        print("测试分类器预测...")
        test_features = torch.randn(10, 256, device=classifier[0].weight.device)
        with torch.no_grad():
            outputs = classifier(test_features)
            predictions = torch.argmax(outputs, dim=1)
            print(f"测试样本预测结果: {predictions.cpu().numpy()}")
        
        print("✅ SGD 分类器测试完成!")
        return True
        
    except Exception as e:
        print(f"❌ SGD 分类器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dynamic_sampling():
    """测试动态采样功能"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(filename)s] => %(message)s')
    
    print("\n开始测试动态采样功能...")
    
    # 创建更小的数据集用于测试
    stats_dict = create_mock_stats_dict(num_classes=3, feature_dim=128)
    
    sgd_builder = SGDClassifierBuilder(
        cached_Z=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_steps=10,  # 非常少的步数
        lr=5e-4
    )
    
    print("构建带动态采样的 SGD 分类器...")
    try:
        classifier = sgd_builder.build(stats_dict, linear=True, alpha1=1.0, alpha2=0.0, alpha3=0.5)
        print("✅ 动态采样 SGD 分类器构建成功!")
        return True
    except Exception as e:
        print(f"❌ 动态采样测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("=" * 80)
    print("SGD 分类器测试套件")
    print("=" * 80)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 测试基本功能
    success1 = test_sgd_classifier()
    
    # 测试动态采样
    success2 = test_dynamic_sampling()
    
    # 总结
    print("\n" + "=" * 80)
    print("测试结果总结:")
    print(f"基本 SGD 分类器测试: {'✅ 通过' if success1 else '❌ 失败'}")
    print(f"动态采样测试: {'✅ 通过' if success2 else '❌ 失败'}")
    
    if success1 and success2:
        print("🎉 所有测试通过! SGD 分类器功能正常。")
    else:
        print("⚠️ 部分测试失败，请检查代码。")
    
    print("=" * 80)

if __name__ == "__main__":
    main()