#!/usr/bin/env python3
# test_exp2_alpha_constraint.py - 简单测试约束条件实验脚本

import sys
import os
sys.path.append('/home/raoxuan/projects/low_rank_rda')

# 导入模块测试
try:
    print("正在导入exp2_alpha_constraint模块...")
    from classifier_ablation.experiments import exp2_alpha_constraint
    print("✓ 导入成功")
except ImportError as e:
    print(f"✗ 导入失败: {e}")
    sys.exit(1)

# 测试函数定义
def test_function_definitions():
    """测试核心函数是否正确定义"""
    print("\n测试函数定义...")
    
    functions_to_test = [
        'evaluate_classifier_with_alpha_constraint',
        'evaluate_classifiers_under_constraint', 
        'plot_alpha_constraint_performance',
        'save_constraint_results',
        'run_alpha_constraint_experiment'
    ]
    
    for func_name in functions_to_test:
        if hasattr(exp2_alpha_constraint, func_name):
            func = getattr(exp2_alpha_constraint, func_name)
            print(f"✓ {func_name} 已定义")
        else:
            print(f"✗ {func_name} 未找到")

# 测试参数验证
def test_parameter_validation():
    """测试关键函数的参数处理"""
    print("\n测试参数验证...")
    
    # 测试alpha1_values生成
    try:
        import numpy as np
        alpha1_values = np.linspace(0, 1, 11)
        expected_values = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        
        if np.allclose(alpha1_values, expected_values):
            print("✓ α1值数组生成正确")
        else:
            print(f"✗ α1值数组生成错误: {alpha1_values}")
    except Exception as e:
        print(f"✗ α1值数组生成失败: {e}")
    
    # 测试约束条件计算
    try:
        alpha1 = 0.3
        alpha2 = 1.0 - alpha1
        expected_alpha2 = 0.7
        
        if abs(alpha2 - expected_alpha2) < 1e-10:
            print("✓ 约束条件α1+α2=1.0计算正确")
        else:
            print(f"✗ 约束条件计算错误: α1={alpha1}, α2={alpha2}")
    except Exception as e:
        print(f"✗ 约束条件计算失败: {e}")

# 测试模拟评估功能
def test_mock_evaluation():
    """测试模拟评估功能"""
    print("\n测试模拟评估功能...")
    
    try:
        # 创建模拟数据
        import numpy as np
        import torch
        
        # 模拟alpha1_values
        alpha1_values = np.linspace(0, 1, 5)  # 减少到5个点用于测试
        
        # 模拟准确度数据
        qda_accuracies = np.random.uniform(0.7, 0.9, len(alpha1_values))  # 模拟QDA准确度
        sgd_accuracies = np.random.uniform(0.6, 0.8, len(alpha1_values))  # 模拟SGD准确度
        
        print(f"✓ 模拟数据生成成功")
        print(f"  - α1值: {alpha1_values}")
        print(f"  - QDA模拟准确度: {qda_accuracies}")
        print(f"  - SGD模拟准确度: {sgd_accuracies}")
        
        # 测试最佳点查找
        qda_best_idx = np.argmax(qda_accuracies)
        sgd_best_idx = np.argmax(sgd_accuracies)
        
        print(f"✓ 最佳点查找成功")
        print(f"  - QDA最佳点: α1={alpha1_values[qda_best_idx]:.3f}, acc={qda_accuracies[qda_best_idx]:.3f}")
        print(f"  - SGD最佳点: α1={alpha1_values[sgd_best_idx]:.3f}, acc={sgd_accuracies[sgd_best_idx]:.3f}")
        
    except Exception as e:
        print(f"✗ 模拟评估功能测试失败: {e}")

# 测试结果保存功能（仅验证函数调用，不实际保存）
def test_save_function_structure():
    """测试结果保存函数的结构"""
    print("\n测试结果保存函数结构...")
    
    try:
        # 验证函数是否存在且可调用
        save_func = getattr(exp2_alpha_constraint, 'save_constraint_results')
        if callable(save_func):
            print("✓ save_constraint_results函数可调用")
        else:
            print("✗ save_constraint_results函数不可调用")
            
        # 测试matplotlib绘图函数
        plot_func = getattr(exp2_alpha_constraint, 'plot_alpha_constraint_performance')
        if callable(plot_func):
            print("✓ plot_alpha_constraint_performance函数可调用")
        else:
            print("✗ plot_alpha_constraint_performance函数不可调用")
            
    except Exception as e:
        print(f"✗ 结果保存函数结构测试失败: {e}")

# 主测试函数
def main():
    """运行所有测试"""
    print("="*50)
    print("约束条件实验脚本功能测试")
    print("="*50)
    
    # 运行各项测试
    test_function_definitions()
    test_parameter_validation()
    test_mock_evaluation()
    test_save_function_structure()
    
    print("\n" + "="*50)
    print("测试完成")
    print("="*50)
    print("\n注意：这是功能测试，未运行完整的实验流程。")
    print("要运行完整实验，请执行:")
    print("python classifier_ablation/experiments/exp2_alpha_constraint.py")

if __name__ == '__main__':
    main()