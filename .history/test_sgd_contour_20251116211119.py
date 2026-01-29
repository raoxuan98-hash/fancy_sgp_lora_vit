#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本：验证SGD分类器等高线图生成功能
"""

import os
import sys
sys.path.append('/home/raoxuan/projects/fancy_sgp_lora_vit')

# 导入必要的模块
import numpy as np
import matplotlib.pyplot as plt
import torch
from classifier_ablation.experiments.exp4_sgd_contour import (
    build_gaussian_statistics,
    grid_search_sgd_alpha1_alpha2,
    plot_sgd_alpha1_alpha2_contour,
    save_sgd_results
)

def create_mock_data():
    """
    创建模拟数据用于测试
    """
    # 创建模拟的高斯统计量
    num_classes = 5
    feature_dim = 768
    
    stats = {}
    for i in range(num_classes):
        mean = torch.randn(feature_dim) * 0.1
        cov = torch.eye(feature_dim) * 0.1
        stats[i] = type('GaussianStatistics', (), {
            'mean': mean,
            'cov': cov
        })()
    
    # 创建模拟的测试数据
    test_features = torch.randn(100, feature_dim)
    test_labels = torch.randint(0, num_classes, (100,))
    test_dataset_ids = torch.zeros(100)
    
    # 生成缓存的随机向量
    cached_Z = torch.randn(1024, feature_dim)
    
    return stats, test_features, test_labels, test_dataset_ids, cached_Z

def test_grid_search():
    """
    测试网格搜索功能
    """
    print("测试SGD分类器网格搜索功能...")
    
    # 创建模拟数据
    stats, test_features, test_labels, test_dataset_ids, cached_Z = create_mock_data()
    
    # 设置较小的参数范围用于测试
    alpha1_values, alpha2_values, accuracy_matrix = grid_search_sgd_alpha1_alpha2(
        stats, test_features, test_labels, test_dataset_ids,
        alpha1_min=0.9, alpha1_max=1.1, alpha2_min=0.0, alpha2_max=0.5,
        alpha1_points=3, alpha2_points=3,
        sgd_epochs=1, sgd_lr=1e-3, cached_Z=cached_Z,
        return_class_wise=True, device="cpu"  # 使用CPU进行测试
    )
    
    print(f"alpha1_values: {alpha1_values}")
    print(f"alpha2_values: {alpha2_values}")
    print(f"accuracy_matrix shape: {accuracy_matrix.shape}")
    print(f"accuracy_matrix:\n{accuracy_matrix}")
    
    return alpha1_values, alpha2_values, accuracy_matrix

def test_plot_contour():
    """
    测试等高线图绘制功能
    """
    print("\n测试SGD分类器等高线图绘制功能...")
    
    # 获取网格搜索结果
    alpha1_values, alpha2_values, accuracy_matrix = test_grid_search()
    
    # 测试绘制等高线图
    plt.figure(figsize=(5, 4))  # 使用较小的尺寸便于显示
    
    # 创建网格
    alpha1_grid, alpha2_grid = np.meshgrid(alpha1_values, alpha2_values)
    
    # 绘制等高线图
    contour = plt.contourf(alpha1_grid, alpha2_grid, accuracy_matrix.T, levels=10, cmap='viridis')
    plt.colorbar(contour)
    
    # 添加等高线
    plt.contour(alpha1_grid, alpha2_grid, accuracy_matrix.T, levels=10, colors='black', alpha=0.5)
    plt.clabel(plt.contour(alpha1_grid, alpha2_grid, accuracy_matrix.T, levels=10), inline=True, fontsize=8)
    
    plt.xlabel(r'$\alpha_1^{\rm SGD}$')
    plt.ylabel(r'$\alpha_2^{\rm SGD}$')
    plt.title('SGD Classifier Parameter Sensitivity (Test)')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 找到最佳准确率及其对应的参数
    max_idx = np.unravel_index(np.argmax(accuracy_matrix), accuracy_matrix.shape)
    best_alpha1 = alpha1_values[max_idx[0]]
    best_alpha2 = alpha2_values[max_idx[1]]
    best_acc = accuracy_matrix[max_idx]
    plt.plot(best_alpha1, best_alpha2, 'r*', markersize=10, label=f'Best: ({best_alpha1:.3f}, {best_alpha2:.3f}) = {best_acc:.2f}%')
    
    plt.legend()
    plt.tight_layout()
    
    # 保存测试图像
    test_output_dir = "test_output"
    os.makedirs(test_output_dir, exist_ok=True)
    test_image_path = os.path.join(test_output_dir, "test_sgd_contour.png")
    plt.savefig(test_image_path, dpi=300, bbox_inches='tight')
    print(f"测试等高线图已保存到: {test_image_path}")
    
    plt.close()  # 关闭图像以释放内存
    
    return test_image_path

def test_save_results():
    """
    测试结果保存功能
    """
    print("\n测试SGD分类器结果保存功能...")
    
    # 获取网格搜索结果
    alpha1_values, alpha2_values, accuracy_matrix = test_grid_search()
    
    # 测试保存结果
    test_output_dir = "test_output"
    model_name = "test-model"
    save_path = save_sgd_results(alpha1_values, alpha2_values, accuracy_matrix, model_name, test_output_dir)
    
    # 验证保存的文件
    if os.path.exists(save_path):
        print(f"结果已成功保存到: {save_path}")
        
        # 尝试加载保存的数据
        loaded_data = np.load(save_path)
        print(f"加载的数据键: {list(loaded_data.keys())}")
        print(f"alpha1_values形状: {loaded_data['alpha1_values'].shape}")
        print(f"alpha2_values形状: {loaded_data['alpha2_values'].shape}")
        print(f"accuracy_matrix形状: {loaded_data['accuracy_matrix'].shape}")
        
        return save_path
    else:
        print(f"保存失败: 文件不存在 {save_path}")
        return None

if __name__ == "__main__":
    print("开始测试SGD分类器等高线图生成功能...")
    print("="*60)
    
    try:
        # 测试结果保存功能
        save_path = test_save_results()
        
        # 测试等高线图绘制功能
        image_path = test_plot_contour()
        
        print("\n" + "="*60)
        print("测试完成!")
        print(f"结果文件: {save_path}")
        print(f"图像文件: {image_path}")
        print("="*60)
        
    except Exception as e:
        print(f"测试过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()