#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理时间效率对比实验
比较Full-rank QDA, Low-rank QDA (rank-1,8,16,32), SGD-based linear classifier, LDA的推理时间
"""
# In[]

import os
import argparse
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

# 导入项目模块
from classifier.da_classifier_builder import QDAClassifierBuilder, LDAClassifierBuilder
from classifier.sgd_classifier_builder import SGDClassifierBuilder
from classifier.ncm_classifier import NCMClassifier
from classifier.gaussian_classifier import LinearLDAClassifier
from classifier_ablation.experiments.exp1_performance_surface import build_gaussian_statistics
from classifier_ablation.data.data_loader import load_cross_domain_data, create_data_loaders, create_adapt_loader
from classifier_ablation.features.feature_extractor import get_vit, adapt_backbone, extract_features_and_labels

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_gpu_memory_info() -> Dict[str, float]:
    """获取当前GPU显存信息"""
    if not torch.cuda.is_available():
        return {"allocated": 0.0, "reserved": 0.0, "max_allocated": 0.0}
    
    return {
        "allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
        "reserved": torch.cuda.memory_reserved() / 1024**3,    # GB
        "max_allocated": torch.cuda.max_memory_allocated() / 1024**3
}

def create_class_subset(
    full_features: torch.Tensor,
    full_labels: torch.Tensor,
    num_classes: int,
    samples_per_class: int = 32,
    random_seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    np.random.seed(random_seed)
    unique_classes = torch.unique(full_labels)
    
    # 按顺序选择类别
    if len(unique_classes) < num_classes:
        raise ValueError(f"数据集只有{len(unique_classes)}个类别，无法选择{num_classes}个类别")
    
    # 按顺序选择前num_classes个类别
    selected_classes = unique_classes[:num_classes].tolist()
    
    # 筛选数据
    mask = torch.tensor([label in selected_classes for label in full_labels])
    subset_features = full_features[mask]
    subset_labels = full_labels[mask]
    
    # 为每个类别限制样本数量
    final_features = []
    final_labels = []
    
    for class_id in selected_classes:
        class_mask = (subset_labels == class_id)
        class_features = subset_features[class_mask]
        
        # 随机采样指定数量的样本
        if len(class_features) > samples_per_class:
            indices = torch.randperm(len(class_features))[:samples_per_class]
            class_features = class_features[indices]
        
        final_features.append(class_features)
        final_labels.extend([class_id] * len(class_features))
    
    final_features = torch.cat(final_features, dim=0)
    final_labels = torch.tensor(final_labels)
    
    logger.info(f"创建了{num_classes}个类别的子集，共{len(final_features)}个样本")
    
    return final_features, final_labels, selected_classes

def measure_build_time(
    classifier_type: str,
    stats_dict: Dict[int, object],
    device: str = "cuda",
    **kwargs
) -> Tuple[float, nn.Module, Dict[str, float]]:
    """
    测量分类器构建时间
    
    Args:
        classifier_type: 分类器类型
        stats_dict: 高斯统计量字典
        device: 计算设备
        **kwargs: 分类器特定参数
        
    Returns:
        build_time: 构建时间（秒）
        classifier: 构建好的分类器
        memory_usage: 内存使用信息（MB）
    """
    # 预热GPU，添加错误处理
    try:
        torch.cuda.empty_cache()
        dummy = torch.randn(100, 768).to(device)
        _ = torch.mm(dummy, dummy.t())
        torch.cuda.synchronize()
    except Exception as e:
        logger.warning(f"GPU预热失败，继续执行: {e}")
        # 如果CUDA不可用，回退到CPU
        if "cuda" in device:
            device = "cpu"
    
    # 记录初始内存
    start_memory = get_gpu_memory_info()
    
    # 开始计时，安全地同步CUDA
    try:
        if torch.cuda.is_available() and "cuda" in device:
            torch.cuda.synchronize()
    except Exception as e:
        logger.warning(f"CUDA同步失败，继续执行: {e}")
    start_time = time.time()
    
    # 构建分类器
    if classifier_type == "full_qda":
        builder = QDAClassifierBuilder(
            qda_reg_alpha1=0.2,
            qda_reg_alpha2=0.2,
            qda_reg_alpha3=0.2,
            low_rank=False,
            device=device)
        
        classifier = builder.build(stats_dict)
        
    elif classifier_type.startswith("low_qda_r"):
        rank = int(classifier_type.split("_r")[1])
        builder = QDAClassifierBuilder(
            qda_reg_alpha1=0.2,
            qda_reg_alpha2=0.2,
            qda_reg_alpha3=0.2,
            low_rank=True,
            rank=rank,
            device=device
        )
        classifier = builder.build(stats_dict)
        
    elif classifier_type == "sgd_linear":
        builder = SGDClassifierBuilder(device=device, max_steps=5000, lr=1e-3)
        classifier = builder.build(stats_dict, linear=True, alpha1=0.5, alpha2=0.5, alpha3=0.5)
        
    elif classifier_type == "lda":
        builder = LDAClassifierBuilder(reg_alpha=0.3, device=device)
        classifier = builder.build(stats_dict)
        
    else:
        raise ValueError(f"不支持的分类器类型: {classifier_type}")
    
    # 结束计时，安全地同步CUDA
    try:
        if torch.cuda.is_available() and "cuda" in device:
            torch.cuda.synchronize()
    except Exception as e:
        logger.warning(f"CUDA同步失败，继续执行: {e}")
    end_time = time.time()
    
    # 计算内存增量
    end_memory = get_gpu_memory_info()
    memory_usage = {
        "allocated_mb": (end_memory["allocated"] - start_memory["allocated"]) * 1024,
        "reserved_mb": (end_memory["reserved"] - start_memory["reserved"]) * 1024
    }
    
    build_time = end_time - start_time
    
    logger.info(f"{classifier_type} 构建时间: {build_time:.4f}s, 内存使用: {memory_usage['allocated_mb']:.2f}MB")
    
    return build_time, classifier, memory_usage

def measure_inference_time(
    classifier: nn.Module,
    test_features: torch.Tensor,
    batch_size: int = 64,
    warmup_runs: int = 10,
    measure_runs: int = 50,
    device: str = "cuda"
) -> Tuple[float, float, float]:
    """
    测量分类器推理时间
    
    Args:
        classifier: 训练好的分类器
        test_features: 测试特征
        batch_size: 批次大小
        warmup_runs: 预热运行次数
        measure_runs: 测量运行次数
        device: 计算设备
        
    Returns:
        avg_time_per_sample: 每个样本的平均推理时间（毫秒）
        avg_time_per_batch: 每个批次的平均推理时间（毫秒）
        throughput: 吞吐量（样本/秒）
    """
    classifier.eval()
    
    # 安全地设置设备和移动数据
    try:
        classifier.to(device)
        test_features = test_features.to(device)
    except Exception as e:
        logger.warning(f"设备设置失败，回退到CPU: {e}")
        device = "cpu"
        classifier.to(device)
        test_features = test_features.to(device)
    
    num_samples = len(test_features)
    
    # 预热运行，添加错误处理
    with torch.no_grad():
        for _ in range(warmup_runs):
            start_idx = 0
            while start_idx < num_samples:
                end_idx = min(start_idx + batch_size, num_samples)
                batch = test_features[start_idx:end_idx]
                _ = classifier(batch)
                start_idx = end_idx
            # 安全地同步CUDA
            try:
                if torch.cuda.is_available() and "cuda" in device:
                    torch.cuda.synchronize()
            except Exception as e:
                logger.warning(f"CUDA同步失败，继续执行: {e}")
    
    # 测量运行
    total_time = 0.0
    total_batches = 0
    
    with torch.no_grad():
        for _ in range(measure_runs):
            start_time = time.time()
            
            start_idx = 0
            while start_idx < num_samples:
                end_idx = min(start_idx + batch_size, num_samples)
                batch = test_features[start_idx:end_idx]
                _ = classifier(batch)
                start_idx = end_idx
                
            # 安全地同步CUDA
            try:
                if torch.cuda.is_available() and "cuda" in device:
                    torch.cuda.synchronize()
            except Exception as e:
                logger.warning(f"CUDA同步失败，继续执行: {e}")
            end_time = time.time()
            
            total_time += (end_time - start_time)
            total_batches += 1
    
    # 计算指标
    avg_time_per_batch = (total_time / total_batches) * 1000  # 毫秒
    avg_time_per_sample = (total_time / total_batches) * 1000 / batch_size  # 毫秒/样本
    throughput = (num_samples * total_batches) / total_time  # 样本/秒
    
    logger.info(f"推理时间 - 每样本: {avg_time_per_sample:.4f}ms, 吞吐量: {throughput:.2f} samples/s")
    
    return avg_time_per_sample, avg_time_per_batch, throughput

# In[]
def initialize_results_dict(classifier_types: List[str]) -> Dict[str, Dict[str, List[float]]]:
    """初始化结果字典"""
    return {
        "inference_time": {ct: [] for ct in classifier_types}
    }

def load_and_prepare_data(num_shots: int, model_name: str):
    """加载数据并创建数据加载器"""
    logger.info("加载数据...")
    dataset, train_subsets, test_subsets = load_cross_domain_data(num_shots=num_shots, model_name=model_name)
    
    logger.info("创建数据加载器...")
    train_loader, test_loader = create_data_loaders(train_subsets, test_subsets)
    
    return dataset, train_loader, test_loader

def setup_and_adapt_model(dataset, train_subsets, model_name: str):
    """获取和适配Vision Transformer模型"""
    logger.info("获取和适配Vision Transformer模型...")
    vit = get_vit(vit_name=model_name)
    adapt_loader = create_adapt_loader(train_subsets)
    vit = adapt_backbone(vit, adapt_loader, dataset.total_classes, iterations=0)
    return vit

def extract_features(vit, dataset, train_loader, test_loader, model_name: str, num_shots: int):
    """提取特征"""
    logger.info("提取特征...")
    train_features, train_labels, train_dataset_ids, test_features, test_labels, test_dataset_ids = extract_features_and_labels(
        vit, dataset, train_loader, test_loader, model_name, num_shots=num_shots, iterations=0)
    return train_features, train_labels, test_features

def prepare_experiment_data(train_features, train_labels, test_features):
    """准备实验数据"""
    logger.info("构建高斯统计量...")
    full_stats = build_gaussian_statistics(train_features, train_labels)
    
    # 准备测试数据（用于推理时间测量）
    test_subset = test_features[:1000]  # 使用前1000个样本
    
    return full_stats, test_subset

def run_experiment_for_class_count(
    num_classes: int,
    classifier_types: List[str],
    num_repeats: int,
    train_features,
    train_labels,
    num_shots: int,
    test_subset,
    device: str,
    results: Dict[str, Dict[str, List[float]]]
):
    """为特定类别数量运行实验"""
    logger.info(f"\n{'='*60}")
    logger.info(f"测试类别数量: {num_classes}")
    logger.info(f"{'='*60}")
    
    # 创建类别子集
    subset_features, subset_labels, selected_classes = create_class_subset(
        train_features, train_labels, num_classes, samples_per_class=num_shots
    )
    
    # 构建子集统计量
    subset_stats = build_gaussian_statistics(subset_features, subset_labels)
    
    # 为每个分类器类型进行实验
    for classifier_type in classifier_types:
        logger.info(f"\n测试分类器: {classifier_type}")
        
        inference_times = []
        
        for repeat in range(num_repeats):
            logger.info(f"  重复 {repeat+1}/{num_repeats}")
            
            # 构建分类器（不测量时间）
            if classifier_type == "full_qda":
                builder = QDAClassifierBuilder(
                    qda_reg_alpha1=0.2,
                    qda_reg_alpha2=0.2,
                    qda_reg_alpha3=0.2,
                    low_rank=False,
                    device=device)
                classifier = builder.build(subset_stats)
                
            elif classifier_type.startswith("low_qda_r"):
                rank = int(classifier_type.split("_r")[1])
                builder = QDAClassifierBuilder(
                    qda_reg_alpha1=0.2,
                    qda_reg_alpha2=0.2,
                    qda_reg_alpha3=0.2,
                    low_rank=True,
                    rank=rank,
                    device=device
                )
                classifier = builder.build(subset_stats)
                
            elif classifier_type == "sgd_linear":
                builder = SGDClassifierBuilder(device=device, max_steps=5000, lr=1e-3)
                classifier = builder.build(subset_stats, linear=True, alpha1=0.5, alpha2=0.5, alpha3=0.5)
                
            elif classifier_type == "lda":
                builder = LDAClassifierBuilder(reg_alpha=0.3, device=device)
                classifier = builder.build(subset_stats)
                
            else:
                raise ValueError(f"不支持的分类器类型: {classifier_type}")
            
            # 测量推理时间
            inference_time, _, _ = measure_inference_time(
                classifier, test_subset, device=device)
            
            inference_times.append(inference_time)
        
        # 计算平均值（排除无穷大值）
        valid_inference_times = [t for t in inference_times if t != float('inf')]
        avg_inference_time = np.mean(valid_inference_times) if valid_inference_times else float('inf')
        
        results["inference_time"][classifier_type].append(avg_inference_time)
        
        logger.info(f"  平均推理时间: {avg_inference_time:.4f}ms")

def run_efficiency_experiment(
    class_counts: List[int] = [50, 100, 200, 500, 1000],
    classifier_types: List[str] = ["full_qda", "low_qda_r1", "low_qda_r8", "low_qda_r16", "low_qda_r32", "sgd_linear", "lda"],
    num_repeats: int = 3,
    model_name: str = "vit-b-p16-clip",
    num_shots: int = 128,
    device: str = "cuda"
) -> Dict[str, Dict[str, List[float]]]:
    # 初始化结果字典
    results = initialize_results_dict(classifier_types)
    
    # 加载数据
    dataset, train_loader, test_loader = load_and_prepare_data(num_shots, model_name)
    
    # 获取和适配模型
    vit = setup_and_adapt_model(dataset, train_loader, model_name)
    
    # 提取特征
    train_features, train_labels, test_features = extract_features(
        vit, dataset, train_loader, test_loader, model_name, num_shots
    )
    
    # 准备实验数据
    full_stats, test_subset = prepare_experiment_data(train_features, train_labels, test_features)
    
    # 主实验循环
    for num_classes in class_counts:
        run_experiment_for_class_count(
            num_classes, classifier_types, num_repeats,
            train_features, train_labels, num_shots, test_subset, device, results
        )
    
    return results

def plot_build_time_comparison(
    class_counts: List[int],
    build_times: Dict[str, List[float]],
    save_path: str = None,
    log_scale: bool = True
):
    """绘制构建时间对比图"""
    plt.figure(figsize=(10, 6))
    
    colors = {
        'full_qda': 'red',
        'low_qda_r1': 'orange',
        'low_qda_r8': 'yellow',
        'low_qda_r16': 'green',
        'low_qda_r32': 'blue',
        'sgd_linear': 'purple',
        'lda': 'black'
    }
    
    labels = {
        'full_qda': 'Full-rank QDA',
        'low_qda_r1': 'Low-rank QDA (r=1)',
        'low_qda_r8': 'Low-rank QDA (r=8)',
        'low_qda_r16': 'Low-rank QDA (r=16)',
        'low_qda_r32': 'Low-rank QDA (r=32)',
        'sgd_linear': 'SGD Linear',
        'lda': 'LDA'
    }
    
    for classifier_type, times in build_times.items():
        if classifier_type in colors and any(t != float('inf') for t in times):
            valid_times = [t for t in times if t != float('inf')]
            valid_counts = [c for c, t in zip(class_counts, times) if t != float('inf')]
            plt.plot(valid_counts, valid_times, 'o-', 
                    color=colors[classifier_type], 
                    label=labels[classifier_type],
                    linewidth=2, markersize=6)
    
    plt.xlabel('Number of Classes')
    plt.ylabel('Build Time (seconds)')
    plt.title('Classifier Build Time Comparison')
    
    if log_scale:
        plt.yscale('log')
        plt.xscale('log')
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"构建时间对比图已保存到: {save_path}")
    
    plt.show()

def plot_inference_time_comparison(
    class_counts: List[int],
    inference_times: Dict[str, List[float]],
    save_path: str = None
):
    """绘制推理时间对比图"""
    # IEEE单栏图片大小 (3.5英寸宽)
    plt.figure(figsize=(3.5, 2.6))
    
    # 设置颜色和线型组合
    styles = {
        'full_qda': {'color': '#1f77b4', 'linestyle': '-', 'marker': 'o'},
        'low_qda_r1': {'color': '#ff7f0e', 'linestyle': '--', 'marker': 's'},
        'low_qda_r8': {'color': '#2ca02c', 'linestyle': '-.', 'marker': '^'},
        'low_qda_r16': {'color': '#d62728', 'linestyle': ':', 'marker': 'D'},
        'low_qda_r32': {'color': '#9467bd', 'linestyle': '-', 'marker': 'v'},
        'sgd_linear': {'color': '#8c564b', 'linestyle': '--', 'marker': 'p'},
        'lda': {'color': '#e377c2', 'linestyle': '-.', 'marker': '*'}
    }
    
    labels = {
        'full_qda': 'Full-rank QDA',
        'low_qda_r1': 'Low-rank QDA (r=1)',
        'low_qda_r8': 'Low-rank QDA (r=8)',
        'low_qda_r16': 'Low-rank QDA (r=16)',
        'low_qda_r32': 'Low-rank QDA (r=32)',
        'sgd_linear': 'SGD Linear',
        'lda': 'LDA'
    }
    
    for classifier_type, times in inference_times.items():
        if classifier_type in styles and any(t != float('inf') for t in times):
            valid_times = [t for t in times if t != float('inf')]
            valid_counts = [c for c, t in zip(class_counts, times) if t != float('inf')]
            plt.plot(valid_counts, valid_times,
                    color=styles[classifier_type]['color'],
                    linestyle=styles[classifier_type]['linestyle'],
                    marker=styles[classifier_type]['marker'],
                    label=labels[classifier_type],
                    linewidth=1.5, markersize=4)
    
    plt.xlabel('Number of Classes')
    plt.ylabel('Inference Time (ms per sample)')
    plt.title('Classifier Inference Time Comparison')
    
    # 设置y轴为对数尺度
    plt.yscale('log')
    
    plt.grid(True, alpha=0.3, which='both')
    plt.legend(fontsize='small', ncol=2, columnspacing=0.5, handletextpad=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"推理时间对比图已保存到: {save_path}")
    
    plt.show()

def plot_memory_usage_comparison(
    class_counts: List[int],
    memory_usage: Dict[str, List[float]],
    save_path: str = None,
    log_scale: bool = True
):
    """绘制内存使用对比图"""
    plt.figure(figsize=(10, 6))
    
    colors = {
        'full_qda': 'red',
        'low_qda_r1': 'orange',
        'low_qda_r8': 'yellow',
        'low_qda_r16': 'green',
        'low_qda_r32': 'blue',
        'sgd_linear': 'purple',
        'lda': 'black'
    }
    
    labels = {
        'full_qda': 'Full-rank QDA',
        'low_qda_r1': 'Low-rank QDA (r=1)',
        'low_qda_r8': 'Low-rank QDA (r=8)',
        'low_qda_r16': 'Low-rank QDA (r=16)',
        'low_qda_r32': 'Low-rank QDA (r=32)',
        'sgd_linear': 'SGD Linear',
        'lda': 'LDA'
    }
    
    for classifier_type, usage in memory_usage.items():
        if classifier_type in colors and any(m != float('inf') for m in usage):
            valid_usage = [m for m in usage if m != float('inf')]
            valid_counts = [c for c, m in zip(class_counts, usage) if m != float('inf')]
            plt.plot(valid_counts, valid_usage, 'o-', 
                    color=colors[classifier_type], 
                    label=labels[classifier_type],
                    linewidth=2, markersize=6)
    
    plt.xlabel('Number of Classes')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Classifier Memory Usage Comparison')
    
    if log_scale:
        plt.yscale('log')
        plt.xscale('log')
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"内存使用对比图已保存到: {save_path}")
    
    plt.show()

def save_results(results: Dict, class_counts: List[int], model_name: str, output_dir: str):
    """保存实验结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存为NPZ文件
    save_path = os.path.join(output_dir, f"efficiency_results_{model_name}.npz")
    np.savez(save_path,
             class_counts=class_counts,
             build_time=results["build_time"],
             inference_time=results["inference_time"],
             memory_usage=results["memory_usage"])
    
    logger.info(f"实验结果已保存到: {save_path}")
    
    # 保存为CSV文件
    csv_path = os.path.join(output_dir, f"efficiency_results_{model_name}.csv")
    
    with open(csv_path, 'w') as f:
        f.write("classifier_type,class_count,build_time,inference_time,memory_usage\n")
        
        for classifier_type in results["build_time"]:
            for i, class_count in enumerate(class_counts):
                build_time = results["build_time"][classifier_type][i]
                inference_time = results["inference_time"][classifier_type][i]
                memory_usage = results["memory_usage"][classifier_type][i]
                
                f.write(f"{classifier_type},{class_count},{build_time},{inference_time},{memory_usage}\n")
    
    logger.info(f"CSV结果已保存到: {csv_path}")

# In[]
def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='分类器计算效率对比实验')
    parser.add_argument('--model', type=str, default='vit-b-p16-clip',
                        help='模型名称')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU编号')
    parser.add_argument('--num_shots', type=int, default=128,
                        help='每类样本数')
    parser.add_argument('--class_counts', type=int, nargs='+',
                        default=[50, 100, 200, 500, 1000],
                        help='类别数量列表')
    parser.add_argument('--num_repeats', type=int, default=3,
                        help='重复实验次数')
    parser.add_argument('--output_dir', type=str,
                        default='实验结果保存/效率对比实验',
                        help='结果保存目录')
    
    args = parser.parse_args(args=[])
    
    # 设置GPU和CUDA错误调试
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 启用同步CUDA执行，便于调试
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置日志文件
    log_file = os.path.join(args.output_dir, "experiment_log.txt")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info("开始计算效率对比实验")
    logger.info(f"模型: {args.model}")
    logger.info(f"GPU: {args.gpu}")
    logger.info(f"类别数量: {args.class_counts}")
    logger.info(f"重复次数: {args.num_repeats}")
    
    # 定义分类器类型
    classifier_types = ["full_qda", "low_qda_r1", "low_qda_r8", "low_qda_r16", "low_qda_r32", "sgd_linear", "lda"]
    
    # 步骤1: 初始化结果字典
    results = initialize_results_dict(classifier_types)
    logger.info("步骤1: 初始化结果字典完成")
    
    # 步骤2: 加载数据
    dataset, train_loader, test_loader = load_and_prepare_data(args.num_shots, args.model)
    logger.info("步骤2: 数据加载完成")
    
    # 步骤3: 获取和适配模型
    vit = setup_and_adapt_model(dataset, train_loader, args.model)
    logger.info("步骤3: 模型设置和适配完成")
    
    # 步骤4: 提取特征
    train_features, train_labels, test_features = extract_features(
        vit, dataset, train_loader, test_loader, args.model, args.num_shots
    )
    logger.info("步骤4: 特征提取完成")
    
    # 步骤5: 准备实验数据
    full_stats, test_subset = prepare_experiment_data(train_features, train_labels, test_features)
    logger.info("步骤5: 实验数据准备完成")
    
    # 步骤6: 主实验循环
    for num_classes in args.class_counts:
        run_experiment_for_class_count(
            num_classes, classifier_types, args.num_repeats,
            train_features, train_labels, args.num_shots, test_subset, "cuda", results
        )
    
    logger.info("步骤6: 实验循环完成")
    
    # 保存结果
    save_results(results, args.class_counts, args.model, args.output_dir)
    
    # 生成可视化图表
    plot_build_time_comparison(
        args.class_counts, results["build_time"],
        save_path=os.path.join(args.output_dir, "build_time_comparison.png")
    )
    
    plot_inference_time_comparison(
        args.class_counts, results["inference_time"],
        save_path=os.path.join(args.output_dir, "inference_time_comparison.png")
    )
    
    plot_memory_usage_comparison(
        args.class_counts, results["memory_usage"],
        save_path=os.path.join(args.output_dir, "memory_usage_comparison.png")
    )
    
    logger.info("实验完成!")

if __name__ == '__main__':
    main()
# %%
