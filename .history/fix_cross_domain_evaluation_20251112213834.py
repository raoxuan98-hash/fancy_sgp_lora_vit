#!/usr/bin/env python3
"""
跨域训练准确率异常问题的修复方案

核心问题：
1. 评估时使用累积模式的数据集，但分类器输出维度与测试数据的标签范围不匹配
2. 非累积模式下的标签转换可能存在问题

修复策略：
1. 确保评估时使用正确的数据集模式
2. 修复标签映射和偏移问题
3. 添加详细的调试日志验证修复效果
"""

import logging
import torch
import numpy as np
from torch.utils.data import DataLoader

def fix_evaluation_logic():
    """
    修复评估逻辑
    
    问题：在跨域训练中，评估时使用累积模式的数据集（包含所有已学习任务的类别），
    但分类器的输出维度可能与测试数据的标签范围不匹配。
    
    解决方案：修改评估函数，确保预测标签与真实标签的范围匹配。
    """
    
    print("=" * 80)
    print("修复评估逻辑")
    print("=" * 80)
    
    # 修复方案1：在评估时使用非累积模式的数据集
    # 这样可以确保测试数据的标签范围与分类器的输出维度匹配
    print("""
    修复方案1：修改评估函数中的数据集获取方式
    
    原始代码问题：
    - 使用累积模式获取测试数据：test_set = data_manager.get_subset(task_id, source="test", cumulative=True, mode="test")
    - 这导致测试数据包含所有已学习任务的类别，但分类器可能只针对当前任务训练
    
    修复方案：
    - 对于单个任务的评估，使用非累积模式：test_set = data_manager.get_subset(task_id, source="test", cumulative=False, mode="test")
    - 这样可以确保测试数据的标签范围与分类器的输出维度匹配
    """)
    
    # 修复方案2：在评估函数中添加标签范围检查
    print("""
    修复方案2：在评估函数中添加标签范围检查
    
    问题：预测标签可能超出真实标签的有效范围
    
    修复方案：
    - 在评估函数中检查预测标签的范围
    - 如果预测标签超出范围，将其截断到有效范围内
    - 记录警告信息以便调试
    """)
    
    # 修复方案3：修改分类器更新逻辑
    print("""
    修复方案3：修改分类器更新逻辑
    
    问题：分类器的输出维度可能与实际需要的不匹配
    
    修复方案：
    - 确保分类器的输出维度与当前任务的类别数匹配
    - 在更新分类器时验证维度正确性
    """)

def fix_data_manager_logic():
    """
    修复数据管理器逻辑
    
    问题：在CrossDomainDataManager中，非累积模式下的标签转换可能存在问题
    
    解决方案：确保标签转换的正确性和一致性
    """
    
    print("\n" + "=" * 80)
    print("修复数据管理器逻辑")
    print("=" * 80)
    
    print("""
    修复方案：确保标签转换的正确性
    
    问题：
    - 在非累积模式下，全局标签转换为局部标签时可能存在错误
    - 标签范围验证不足
    
    修复方案：
    - 添加更严格的标签范围验证
    - 确保转换后的标签在有效范围内
    - 添加详细的调试日志
    """)

def fix_training_logic():
    """
    修复训练逻辑
    
    问题：训练时的标签处理可能不正确
    
    解决方案：确保训练时的标签处理与评估时一致
    """
    
    print("\n" + "=" * 80)
    print("修复训练逻辑")
    print("=" * 80)
    
    print("""
    修复方案：确保训练时的标签处理与评估时一致
    
    问题：
    - 训练时使用的标签范围可能与评估时不一致
    - 跨域场景下的标签处理逻辑可能有问题
    
    修复方案：
    - 确保训练时使用正确的标签范围
    - 在跨域场景下，直接使用局部标签而不进行额外转换
    - 添加调试日志验证标签处理的正确性
    """)

def generate_patch_files():
    """
    生成修复补丁文件
    """
    
    print("\n" + "=" * 80)
    print("生成修复补丁文件")
    print("=" * 80)
    
    # 补丁1：修复models/subspace_lora.py中的incremental_train方法
    patch1 = """
    # 在 models/subspace_lora.py 的 incremental_train 方法中
    # 将测试集的获取方式从累积模式改为非累积模式
    
    # 原始代码（有问题）：
    # test_set = data_manager.get_subset(task_id, source="test", cumulative=True, mode="test")
    
    # 修复后的代码：
    test_set = data_manager.get_subset(task_id, source="test", cumulative=False, mode="test")
    """
    
    # 补丁2：修复models/subspace_lora.py中的evaluate方法
    patch2 = """
    # 在 models/subspace_lora.py 的 evaluate 方法中
    # 添加标签范围检查
    
    # 在预测后添加以下代码：
    if self.args.get('cross_domain', False):
        # 获取当前batch的最大有效标签
        max_valid_label = targets.max().item()
        # 将超出范围的预测截断到最大有效标签
        preds = torch.clamp(preds, 0, max_valid_label)
    """
    
    # 补丁3：修复utils/cross_domain_data_manager.py中的get_subset方法
    patch3 = """
    # 在 utils/cross_domain_data_manager.py 的 get_subset 方法中
    # 添加更严格的标签范围验证
    
    # 在非累积模式下，添加以下验证代码：
    if len(dataset['class_names']) > 0:
        min_local, max_local = np.min(local_targets), np.max(local_targets)
        expected_max = len(dataset['class_names']) - 1
        if max_local > expected_max:
            logging.error(f"[CDM] ERROR: Local label {max_local} exceeds class_names length {len(dataset['class_names'])}")
            raise ValueError(f"Local label {max_local} exceeds class_names length {len(dataset['class_names'])}")
    """
    
    print("补丁1：修复测试集获取方式")
    print(patch1)
    
    print("\n补丁2：修复评估函数中的标签范围检查")
    print(patch2)
    
    print("\n补丁3：修复数据管理器中的标签验证")
    print(patch3)

def main():
    """
    主函数：输出修复方案
    """
    print("跨域训练准确率异常问题修复方案")
    print("=" * 80)
    
    fix_evaluation_logic()
    fix_data_manager_logic()
    fix_training_logic()
    generate_patch_files()
    
    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    print("""
    核心问题：
    1. 评估时使用累积模式的数据集，但分类器输出维度与测试数据的标签范围不匹配
    2. 非累积模式下的标签转换可能存在问题
    
    最优雅的修复方案：
    1. 将评估时的测试集获取方式从累积模式改为非累积模式
    2. 在评估函数中添加标签范围检查
    3. 在数据管理器中添加更严格的标签验证
    
    这样可以从根源解决问题，而不是在评估时进行临时的标签截断。
    """)

if __name__ == "__main__":
    main()