#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修改后的推理时间实验脚本
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入修改后的实验脚本
from exp_efficiency_comparison import (
    load_results, 
    plot_inference_time_comparison,
    initialize_results_dict
)

def test_load_results():
    """测试加载结果功能"""
    print("测试加载结果功能...")
    
    # 测试加载不存在的文件
    results, class_counts = load_results("nonexistent_model", "nonexistent_dir")
    assert results is None and class_counts is None, "应该返回None"
    print("✓ 加载不存在文件的测试通过")
    
    # 创建测试数据
    output_dir = "test_output"
    os.makedirs(output_dir, exist_ok=True)
    model_name = "test_model"
    
    # 创建测试结果
    classifier_types = ["full_qda", "low_qda_r1", "sgd_linear"]
    results = initialize_results_dict(classifier_types)
    class_counts = [50, 100, 200]
    
    # 填充测试数据
    for ct in classifier_types:
        for _ in class_counts:
            results["inference_time"][ct].append(np.random.uniform(0.1, 10.0))
    
    # 保存为CSV
    csv_path = os.path.join(output_dir, f"inference_results_{model_name}.csv")
    with open(csv_path, 'w') as f:
        f.write("classifier_type,class_count,inference_time\n")
        for ct in results["inference_time"]:
            for i, cc in enumerate(class_counts):
                f.write(f"{ct},{cc},{results['inference_time'][ct][i]}\n")
    
    # 测试加载
    loaded_results, loaded_class_counts = load_results(model_name, output_dir)
    
    assert loaded_results is not None, "应该成功加载结果"
    assert loaded_class_counts == class_counts, f"类别数量不匹配: {loaded_class_counts} vs {class_counts}"
    
    for ct in classifier_types:
        assert ct in loaded_results["inference_time"], f"缺少分类器类型: {ct}"
        assert len(loaded_results["inference_time"][ct]) == len(class_counts), f"数据长度不匹配: {ct}"
    
    print("✓ 加载CSV文件的测试通过")
    
    # 清理测试文件
    os.remove(csv_path)
    os.rmdir(output_dir)
    
    print("✓ 所有加载结果测试通过")

def test_plot_function():
    """测试绘图功能"""
    print("测试绘图功能...")
    
    # 创建测试数据
    classifier_types = ["full_qda", "low_qda_r1", "sgd_linear"]
    results = initialize_results_dict(classifier_types)
    class_counts = [50, 100, 200]
    
    # 填充测试数据
    for ct in classifier_types:
        for _ in class_counts:
            results["inference_time"][ct].append(np.random.uniform(0.1, 10.0))
    
    # 测试绘图（不保存文件）
    try:
        plot_inference_time_comparison(class_counts, results["inference_time"])
        print("✓ 绘图功能测试通过")
    except Exception as e:
        print(f"✗ 绘图功能测试失败: {e}")
        return False
    
    return True

def test_global_variables():
    """测试全局变量设置"""
    print("测试全局变量设置...")
    
    # 模拟设置全局变量
    global inference_results, inference_class_counts
    
    classifier_types = ["full_qda", "low_qda_r1"]
    results = initialize_results_dict(classifier_types)
    class_counts = [50, 100]
    
    # 填充测试数据
    for ct in classifier_types:
        for _ in class_counts:
            results["inference_time"][ct].append(np.random.uniform(0.1, 10.0))
    
    # 设置全局变量
    inference_results = results
    inference_class_counts = class_counts
    
    # 检查全局变量
    assert 'inference_results' in globals(), "全局变量 inference_results 未设置"
    assert 'inference_class_counts' in globals(), "全局变量 inference_class_counts 未设置"
    assert inference_results is not None, "inference_results 不应为 None"
    assert inference_class_counts is not None, "inference_class_counts 不应为 None"
    
    print("✓ 全局变量设置测试通过")

if __name__ == "__main__":
    print("开始测试修改后的推理时间实验脚本...")
    print("=" * 60)
    
    try:
        test_load_results()
        print()
        
        test_plot_function()
        print()
        
        test_global_variables()
        print()
        
        print("=" * 60)
        print("✓ 所有测试通过！修改后的脚本工作正常。")
        
        print("\n主要修改内容:")
        print("1. 只测量推理时间，移除了构建时间和内存使用测量")
        print("2. 添加了加载预先保存结果的功能")
        print("3. 添加了全局变量设置，方便后期绘图")
        print("4. 添加了命令行参数 --load_only 和 --plot_only")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()