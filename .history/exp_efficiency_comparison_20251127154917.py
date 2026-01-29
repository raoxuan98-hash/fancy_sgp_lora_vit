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
        
        inference_times = []

        for _ in range(num_repeats):
            # 测量推理时间
            inference_time, _, _ = measure_inference_time(
                classifier, test_subset, device=device)
            inference_times.append(inference_time)
        
        # 计算平均值（排除无穷大值）
        valid_inference_times = [t for t in inference_times if t != float('inf')]
        avg_inference_time = float(np.mean(valid_inference_times)) if valid_inference_times else float('inf')
        results["inference_time"][classifier_type].append(avg_inference_time)
        logger.info(f"  平均推理时间: {avg_inference_time:.4f}ms")

def run_efficiency_experiment(
    class_counts: List[int] = [50, 100],
    classifier_types: List[str] = ["full_qda", "low_qda_r1", "low_qda_r8", "low_qda_r16", "low_qda_r32", "sgd_linear", "lda"],
    num_repeats: int = 3,
    model_name: str = "vit-b-p16-clip",
    num_shots: int = 128,
    device: str = "cuda"
) -> Dict[str, Dict[str, List[float]]]:
    results = initialize_results_dict(classifier_types)
    dataset, train_loader, test_loader = load_and_prepare_data(num_shots, model_name)
    
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
            train_features, train_labels, num_shots, test_subset, device, results)
    
    return results

def plot_inference_time_comparison(
    class_counts: List[int],
    inference_times: Dict[str, List[float]],
    save_path: Optional[str] = None
):
    """绘制优化的推理时间对比图（符合IEEE单栏图表规范）"""
    # IEEE单栏图片标准尺寸 (3.5英寸宽)
    plt.figure(figsize=(3.5, 2.6))
    
    # 优化颜色方案 - 使用ColorBrewer 10-class色盲友好方案
    styles = {
        'full_qda': {'color': '#e41a1c', 'linestyle': '-', 'marker': 'o', 'linewidth': 1.8},
        'low_qda_r1': {'color': '#377eb8', 'linestyle': '--', 'marker': 's', 'linewidth': 1.6},
        'low_qda_r8': {'color': '#4daf4a', 'linestyle': '-.', 'marker': '^', 'linewidth': 1.6},
        'low_qda_r16': {'color': '#984ea3', 'linestyle': ':', 'marker': 'D', 'linewidth': 1.6},
        'low_qda_r32': {'color': '#ff7f00', 'linestyle': '-', 'marker': 'v', 'linewidth': 1.6},
        'sgd_linear': {'color': '#a65628', 'linestyle': '--', 'marker': 'p', 'linewidth': 1.6},
        'lda': {'color': '#f781bf', 'linestyle': '-.', 'marker': '*', 'linewidth': 1.6}
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
    
    # 提取有效数据点（排除inf值）
    valid_data = {}
    for classifier_type, times in inference_times.items():
        if classifier_type in styles:
            valid_mask = [t != float('inf') for t in times]
            valid_counts = [c for c, m in zip(class_counts, valid_mask) if m]
            valid_times = [t for t, m in zip(times, valid_mask) if m]
            if valid_times:
                valid_data[classifier_type] = (valid_counts, valid_times)
    
    # 绘制所有有效数据
    for classifier_type, (valid_counts, valid_times) in valid_data.items():
        plt.plot(valid_counts, valid_times,
                color=styles[classifier_type]['color'],
                linestyle=styles[classifier_type]['linestyle'],
                marker=styles[classifier_type]['marker'],
                label=labels[classifier_type],
                linewidth=styles[classifier_type]['linewidth'],
                markersize=4,
                markeredgewidth=0.5)
    
    # 优化坐标轴范围
    plt.xlim(min(class_counts) - 20, max(class_counts) + 20)
    plt.ylim(0.005, 2.5)  # 优化y轴范围，聚焦有效数据区域
    
    # 优化坐标轴标签
    plt.xlabel('Number of Classes', fontsize=9, labelpad=2)
    plt.ylabel('Inference Time (ms per sample)', fontsize=9, labelpad=2)
    
    # 优化标题（移除，IEEE图表通常不需要标题）
    # plt.title('Classifier Inference Time Comparison', fontsize=10)
    
    # 优化y轴为对数尺度
    plt.yscale('log')
    
    # 优化网格线
    plt.grid(True, which='major', linestyle='-', alpha=0.25, linewidth=0.5)
    plt.grid(True, which='minor', linestyle=':', alpha=0.15, linewidth=0.3)
    
    # 优化图例：置于底部外部，紧凑排列
    plt.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.2),
        ncol=4,
        fontsize=7,
        frameon=True,
        framealpha=0.9,
        edgecolor='#e0e0e0',
        columnspacing=0.8,
        handletextpad=0.4,
        borderpad=0.3
    )
    
    # 优化刻度标签
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    
    # 紧凑布局，为图例留出空间
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # 为底部图例预留空间
    
    # 保存或显示
    if save_path:
        plt.savefig(
            save_path,
            dpi=600,
            bbox_inches='tight',
            pad_inches=0.02,
            format='pdf'  # IEEE推荐使用PDF格式
        )
        logger.info(f"优化的推理时间对比图已保存到: {save_path}")
    
    plt.show()

def save_results(results: Dict[str, Dict[str, List[float]]], class_counts: List[int],
                model_name: str, output_dir: str = "实验结果保存"):
    """保存实验结果到CSV文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, f"inference_results_{model_name}.csv")
    
    with open(csv_path, 'w') as f:
        f.write("classifier_type,class_count,inference_time\n")
        for classifier_type in results["inference_time"]:
            for i, class_count in enumerate(class_counts):
                inference_time = results["inference_time"][classifier_type][i]
                f.write(f"{classifier_type},{class_count},{inference_time}\n")
    
    logger.info(f"结果已保存到: {csv_path}")
    return csv_path

def load_results(model_name: str, output_dir: str = "实验结果保存") -> Tuple[Optional[Dict], Optional[List[int]]]:
    """从CSV文件加载实验结果"""
    csv_path = os.path.join(output_dir, f"inference_results_{model_name}.csv")
    
    if not os.path.exists(csv_path):
        logger.warning(f"结果文件不存在: {csv_path}")
        return None, None
    
    results = {"inference_time": {}}
    class_counts_set = set()
    
    with open(csv_path, 'r') as f:
        lines = f.readlines()
        header = lines[0].strip().split(',')
        
        for line in lines[1:]:
            classifier_type, class_count, inference_time = line.strip().split(',')
            class_count = int(class_count)
            inference_time = float(inference_time)
            
            class_counts_set.add(class_count)
            
            if classifier_type not in results["inference_time"]:
                results["inference_time"][classifier_type] = {}
            results["inference_time"][classifier_type][class_count] = inference_time
    
    # 转换为列表格式
    class_counts = sorted(list(class_counts_set))
    
    for classifier_type in results["inference_time"]:
        time_dict = results["inference_time"][classifier_type]
        results["inference_time"][classifier_type] = [time_dict[cc] for cc in class_counts]
    
    logger.info(f"已从 {csv_path} 加载结果")
    return results, class_counts

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='推理时间效率对比实验')
    
    # 实验参数
    parser.add_argument('--class_counts', type=int, nargs='+', default=[50, 100, 200, 400, 800],
                       help='测试的类别数量列表')
    parser.add_argument('--classifier_types', type=str, nargs='+',
                       default=["full_qda", "low_qda_r1", "low_qda_r8", "low_qda_r16", "low_qda_r32", "sgd_linear", "lda"],
                       help='测试的分类器类型列表')
    parser.add_argument('--num_repeats', type=int, default=3,
                       help='每个实验的重复次数')
    parser.add_argument('--model_name', type=str, default="vit-b-p16-clip",
                       help='使用的模型名称')
    parser.add_argument('--num_shots', type=int, default=128,
                       help='每个类别的样本数量')
    parser.add_argument('--device', type=str, default="cuda",
                       help='计算设备')
    
    # 输出控制
    parser.add_argument('--output_dir', type=str, default="实验结果保存",
                       help='结果保存目录')
    parser.add_argument('--save_plot', type=str, default="实验结果保存/效率对比实验",
                       help='图片保存路径（可选）')
    
    # 功能控制
    parser.add_argument('--load_only',type=bool, default=True,
                       help='仅加载已有结果并绘图，不运行新实验')
    parser.add_argument('--plot_only', type=bool, default=True,
                       help='仅绘图，需要已有结果')
    
    return parser.parse_args()

def set_global_variables(results: Dict[str, Dict[str, List[float]]], class_counts: List[int]):
    """设置全局变量，方便后期绘图"""
    global inference_results, inference_class_counts
    inference_results = results
    inference_class_counts = class_counts
    logger.info("已设置全局变量 inference_results 和 inference_class_counts")

if __name__ == '__main__':
    # 解析命令行参数
    args = parse_args()
    
    # 设置日志级别
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 如果只是加载已有结果
    if args.load_only or args.plot_only:
        results, class_counts = load_results(args.model_name, args.output_dir)
        
        if results is None or class_counts is None:
            logger.error("无法加载结果，请先运行实验或检查文件路径")
            exit(1)
        
        # 设置全局变量
        set_global_variables(results, class_counts)
        
        # 绘图
        plot_inference_time_comparison(
            class_counts,
            results["inference_time"],
            save_path=args.save_plot
        )
        
        logger.info("完成结果加载和绘图")
        exit(0)
    
    # 运行完整实验
    logger.info("开始运行推理时间效率对比实验...")
    logger.info(f"实验参数: 类别数量={args.class_counts}, 分类器类型={args.classifier_types}")
    logger.info(f"模型={args.model_name}, 每类样本数={args.num_shots}, 重复次数={args.num_repeats}")
    
    # 运行实验
    results = run_efficiency_experiment(
        class_counts=args.class_counts,
        classifier_types=args.classifier_types,
        num_repeats=args.num_repeats,
        model_name=args.model_name,
        num_shots=args.num_shots,
        device=args.device
    )
    
    # 保存结果
    save_results(results, args.class_counts, args.model_name, args.output_dir)
    
    # 设置全局变量
    set_global_variables(results, args.class_counts)
    
    # 绘图
    plot_inference_time_comparison(
        args.class_counts,
        results["inference_time"],
        save_path=args.save_plot
    )
    
    logger.info("实验完成！")
