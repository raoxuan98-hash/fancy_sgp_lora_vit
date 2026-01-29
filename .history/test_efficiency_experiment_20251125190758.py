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
        print(f"✓ 模型加载成功，特征维度: {768}")
        
        return True
        
    except Exception as e:
        print(f"✗ 数据加载测试失败: {str(e)}")
        return False

def test_gpu_availability():
    """测试GPU可用性"""
    print("\n测试4: GPU可用性测试")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"✓ GPU可用，设备数量: {gpu_count}")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {gpu_name}, 内存: {gpu_memory:.2f}GB")
        
        return True
    else:
        print("✗ GPU不可用")
        return False

def test_memory_function():
    """测试内存测量函数"""
    print("\n测试5: 内存测量功能测试")
    
    try:
        # 导入实验脚本中的函数
        sys.path.append('.')
        from exp_efficiency_comparison import get_gpu_memory_info
        
        # 测试内存测量
        memory_info = get_gpu_memory_info()
        
        if isinstance(memory_info, dict) and all(key in memory_info for key in ["allocated", "reserved", "max_allocated"]):
            print("✓ 内存测量函数正常")
            print(f"  已分配: {memory_info['allocated']:.2f}GB")
            print(f"  已保留: {memory_info['reserved']:.2f}GB")
            print(f"  最大分配: {memory_info['max_allocated']:.2f}GB")
            return True
        else:
            print("✗ 内存测量函数返回格式错误")
            return False
            
    except Exception as e:
        print(f"✗ 内存测量功能测试失败: {str(e)}")
        return False

def main():
    """主测试函数"""
    print("开始测试计算效率对比实验脚本")
    print("="*60)
    
    # 设置环境
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    tests = [
        test_gpu_availability,
        test_data_loading,
        test_memory_function,
        test_basic_functionality,
        test_output_files
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "="*60)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("✓ 所有测试通过，实验脚本可以正常使用")
        return 0
    else:
        print("✗ 部分测试失败，请检查实验脚本")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)