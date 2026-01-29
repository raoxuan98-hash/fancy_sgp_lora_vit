
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试计算效率对比实验脚本
用于验证exp_efficiency_comparison.py的基本功能
"""

import os
import sys
import subprocess
import numpy as np
import torch

def test_basic_functionality():
    """测试基本功能"""
    print("测试1: 基本功能测试")
    
    # 小规模测试命令
    cmd = [
        "python", "exp_efficiency_comparison.py",
        "--class_counts", "10", "20",
        "--num_repeats", "1",
        "--model", "vit-b-p16-clip",
        "--gpu", "0"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30分钟超时
        if result.returncode == 0:
            print("✓ 基本功能测试通过")
            return True
        else:
            print(f"✗ 基本功能测试失败: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("✗ 基本功能测试超时")
        return False
    except Exception as e:
        print(f"✗ 基本功能测试异常: {str(e)}")
        return False

def test_output_files():
    """测试输出文件生成"""
    print("\n测试2: 输出文件测试")
    
    output_dir = "实验结果保存/效率对比实验"
    expected_files = [
        "efficiency_results_vit-b-p16-clip.npz",
        "efficiency_results_vit-b-p16-clip.csv",
        "build_time_comparison.png",
        "inference_time_comparison.png",
        "memory_usage_comparison.png",
        "experiment_log.txt"
    ]
    
    all_exist = True
    for file in expected_files:
        file_path = os.path.join(output_dir, file)
        if os.path.exists(file_path):
            print(f"✓ {file} 存在")
        else:
            print(f"✗ {file} 不存在")
            all_exist = False
    
    return all_exist

def test_data_loading():
    """测试数据加载功能"""
    print("\n测试3: 数据加载测试")
    
    try:
        # 测试数据加载
        from classifier_ablation.data.data_loader import load_cross_domain_data
        from classifier_ablation.features.feature_extractor import get_vit
        
        # 加载数据
        dataset, train_subsets, test_subsets = load_cross_domain_data(
            num_shots=50, model_name="vit-b-p16-clip"
        )
        
        # 获取模型
        vit = get_vit(vit_name="vit-b-p16-clip")
        
        print(f"✓ 数据加载成功，总类别数: {dataset.total_classes}")
