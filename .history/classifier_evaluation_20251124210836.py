
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
封装SGD和QDA分类器的评估过程
"""

import torch
import numpy as np
import logging
from tqdm import tqdm
from collections import defaultdict

from classifier.sgd_classifier_builder import SGDClassifierBuilder
from classifier.da_classifier_builder import QDAClassifierBuilder
from compensator.gaussian_statistics import GaussianStatistics


def build_gaussian_statistics(features, labels):
    """构建高斯统计量"""
    stats_dict = {}
    unique_labels = torch.unique(labels)
    
    for label in unique_labels:
        class_mask = (labels == label)
        class_features = features[class_mask]
        
        # 计算均值和协方差
        mean = class_features.mean(dim=0)
        centered_features = class_features - mean
        cov = torch.matmul(centered_features.t(), centered_features) / (len(class_features) - 1)
        
        # 添加正则化防止奇异
        cov += torch.eye(cov.size(0)) * 1e-3
        
        # 创建统计对象
        stats = GaussianStatistics()
        stats.mean = mean
        stats.cov = cov
        
        stats_dict[int(label)] = stats
    
    return stats_dict


def evaluate_classifiers_comprehensive(alpha1, alpha2, alpha3, dataset, 
                                     classifier_types=["sgd", "qda"],
                                     device="cuda" if torch.cuda.is_available() else "cpu",
                                     batch_size=512):
    """
    综合评估SGD和QDA分类器的性能
    
    Args:
        alpha1: 第一正则化参数
        alpha2: 第二正则化参数  
        alpha3: 第三正则化参数
        dataset: 数据集，包含train_features, train_labels, test_features, test_labels
        classifier_types: 要评估的分类器类型列表 ["sgd", "qda"]
        device: 计算设备
        batch_size: 批次大小
    
    Returns:
        results: 字典，包含每个分类器类型的class-wise准确度
                {
                    "sgd": {
                        "class_accuracies": [准确度数组],
                        "mean_accuracy": 平均准确度,
                        "class_counts": [各类别样本数]
                    },
                    "qda": {
                        "class_accuracies": [准确度数组], 
                        "mean_accuracy": 平均准确度,
                        "class_counts": [各类别样本数]
                    }
                }
    """
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(filename)s] => %(message)s')
    
    # 解析数据集
    train_features, train_labels, test_features, test_labels = dataset
    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    test_features = test_features.to(device)
    test_labels = test_labels.to(device)
    
    print(f"数据集信息:")
    print(f"  训练集形状: {train_features.shape}")
    print(f"  测试集形状: {test_features.shape}")
    print(f"  类别数: {len(torch.unique(train_labels))}")
    print(f"  α1={alpha1:.3f}, α2={alpha2:.3f}, α3={alpha3:.3f}")
    
    # 构建训练数据统计量
    print("构建训练数据统计量...")
    train_stats = build_gaussian_statistics(train_features, train_labels)
    print(f"构建了 {len(train_stats)} 个类别的统计量")
    
    results = {}
    
    # 评估每种分类器类型
    for classifier_type in classifier_types:
        print(f"\n评估 {classifier_type.upper()} 分类器...")
        
        try:
            # 构建分类器
            if classifier_type == "sgd":
                builder = SGDClassifierBuilder(
                    cached_Z=None,
                    device=device,
                    max_steps=1000,  # 适中步数
                    lr=5e-4
                )
                classifier = builder.build(
                    train_stats, 
                    linear=True,
                    alpha1=alpha1,
                    alpha2=alpha2,
                    alpha3=alpha3
                )
                
            elif classifier_type == "qda":
                builder = QDAClassifierBuilder(
                    qda_reg_alpha1=alpha1,
                    qda_reg_alpha2=alpha2,
                    qda_reg_alpha3=alpha3,
                    low_rank=True,
                    rank=64,
                    device=device
                )
                classifier = builder.build(train_stats)
            else:
                raise ValueError(f"不支持的分类器类型: {classifier_type}")
            
            # 移动到设备并设置为评估模式
            classifier.to(device)
            classifier.eval()
            classifier_device = next(classifier.parameters()).device
            
            # 创建测试数据加载器
            test_dataset = torch.utils.data.TensorDataset(test_features, test_labels)
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False
            )
            
            # 评估分类器
            all_predictions = []
            all_targets = []
            
            with torch.no_grad():
                for batch in tqdm(test_loader, desc=f"评估 {classifier_type}"):
                    inputs = batch[0].to(classifier_device)
                    targets = batch[1].to(classifier_device)
                    
                    logits = classifier(inputs)
                    predictions = torch.argmax(logits, dim=1)
                    
                    all_predictions.append(predictions.cpu())
                    all_targets.append(targets.cpu())
            
            # 合并所有预测结果
            all_predictions = torch.cat(all_predictions)
            all_targets = torch.cat(all_targets)
            
            # 计算class-wise准确度
            class_accuracies = []
            class_counts = []
            unique_classes = torch.unique(test_labels)
            
            for class_id in unique_classes:
                class_mask = (all_targets == class_id)
                class_predictions = all_predictions[class_mask]
                class_targets = all_targets[class_mask]
                
                if len(class_targets) > 0:
                    class_accuracy = (class_predictions == class_targets).float().mean().item()
                    class_accuracies.append(class_accuracy)
                    class_counts.append(len(class_targets))
                else:
                    class_accuracies.append(0.0)
                    class_counts.append(0)
            
            # 计算总体准确度
            overall_accuracy = (all_predictions == all_targets).float().mean().item()
            mean_class_accuracy = np.mean(class_accuracies)
            
            print(f"  {classifier_type.upper()} 结果:")
            print(f"    总体准确度: {overall_accuracy:.4f}")
            print(f"    Class-wise平均准确度: {mean_class_accuracy:.4f}")
            print(f"    各类别准确度: {[f'{acc:.4f}' for acc in class_accuracies]}")
            print(f"    各类别样本数: {class_counts}")
            
            # 存储结果
            results[classifier_type] = {
                "class_accuracies": class_accuracies,
                "mean_accuracy": mean_class_accuracy,
                "overall_accuracy": overall_accuracy,
                "class_counts": class_counts,
                "unique_classes": unique_classes.cpu().numpy().tolist()
            }
            
        except Exception as e:
            print(f"  ❌ {classifier_type.upper()} 评估失败: {e}")
            import traceback
            traceback.print_exc()
            
            # 记录失败结果
            results[classifier_type] = {
                "class_accuracies": [],
                "mean_accuracy": 0.0,
                "overall_accuracy": 0.0,
                "class_counts": [],
                "unique_classes": [],
                "error": str(e)
            }
        
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results


def quick_evaluate(alpha1, alpha2, alpha3, dataset, device="cuda"):
    """
    快速评估函数 - 评估SGD和QDA分类器
    
    Args:
        alpha1: 第一正则化参数
        alpha2: 第二正则化参数
        alpha3: 第三正则化参数  
        dataset: 数据集 (train_features, train_labels, test_features, test_labels)
        device: 计算设备
    
    Returns:
        dict: 包含"sgd"和"qda"结果的字典，每个结果包含"class_wise_accuracy"
    """
    results = evaluate_classifiers_comprehensive(
        alpha1=alpha1,
        alpha2=alpha2, 
        alpha3=alpha3,
        dataset=dataset,
        classifier_types=["sgd", "qda"],
        device=device
    )
    
    # 转换为简化格式
    simplified_results = {}
    for classifier_type, result in results.items():
        simplified_results[classifier_type] = {
            "class_wise_accuracy": result["mean_accuracy"]
        }
    
    return simplified_results


# 测试函数
def test_evaluation_function():
    """测试评估函数"""
    print("测试分类器评估函数...")
    
    # 创建模拟数据
    np.random.seed(42)
    torch.manual_seed(42)
    
    num_classes = 5
    feature_dim = 128
    train_samples = 1000
    test_samples = 200
    
    # 生成训练数据
    train_features = torch.randn(train_samples, feature_dim)
    train_labels = torch.randint(0, num_classes, (train_samples,))
    
    # 生成测试数据
    test_features = torch.randn(test_samples, feature_dim)
    test_labels = torch.randint(0, num_classes, (test_samples,))
    
    dataset = (train_features, train_labels, test_features, test_labels)
    
    # 测试不同alpha参数
    alpha_configs = [
        {"alpha1": 0.5, "alpha2": 0.3, "alpha3": 0.1},
        {"alpha1": 0.3, "alpha2": 0.5, "alpha3": 0.2},
        {"alpha1": 0.1, "alpha2": 0.1, "alpha3": 0.8}
    ]
    
    for i, config in enumerate(alpha_configs):
        print(f"\n测试配置 {i+1}: α1={config['alpha1']}, α2={config['alpha2']}, α3={config['alpha3']}")
        
        results = quick_evaluate(
            alpha1=config["alpha1"],
            alpha2=config["alpha2"], 
            alpha3=config["alpha3"],
            dataset=dataset,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        print(f"  SGD Class-wise准确度: {results['sgd']['class_wise_accuracy']:.4f}")
        print(f"  QDA Class-wise准确度: {results['qda']['class_wise_accuracy']:.4f}")
    
    print("\n✅ 评估函数测试完成!")


