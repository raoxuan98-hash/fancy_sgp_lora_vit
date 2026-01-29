#!/usr/bin/env python3
"""
测试脚本：验证cross-domain评估中的调试日志
"""
import sys
import os
import logging
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from torch.utils.data import DataLoader, TensorDataset

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.subspace_lora import SubspaceLoRA

def setup_simple_test():
    """创建一个简单的测试来验证调试日志"""
    print("=== 创建简单测试验证调试日志 ===")
    
    # 创建模拟的模型参数
    mock_args = {
        'cross_domain': True,
        'batch_size': 32,
        'lrate': 0.0001,
        'weight_decay': 0.0005,
        'optimizer': 'adamw',
        'iterations': 100,
        'warmup_ratio': 0.1,
        'gamma_kd': 1.0,
        'kd_type': 'none',
        'use_aux_for_kd': False,
        'update_teacher_each_task': False,
        'auxiliary_data_size': 1000,
        'lda_reg_alpha': 1.0,
        'qda_reg_alpha1': 1.0,
        'qda_reg_alpha2': 1.0,
        'qda_reg_alpha3': 1.0,
        'distillation_transform': 'identity',
        'eval_only': False,
        'log_path': './test_logs',
        'memory_size': 0,
        'memory_per_class': 0,
        'fixed_memory': False,
        'init_cls': 0,
        'increment': 0,
        'seed': 42,
        'auxiliary_data_path': './data',
        'aux_dataset': 'cifar10'
    }
    
    # 创建模拟的模型
    with patch('models.subspace_lora.BaseNet'):
        with patch('models.subspace_lora.DistributionCompensator'):
            with patch('models.subspace_lora.ClassifierReconstructor'):
                with patch('models.subspace_lora.Distiller'):
                    model = SubspaceLoRA(mock_args)
    
    # 设置当前任务ID为1（表示只学习了第一个任务）
    model.current_task_id = 1
    
    # 模拟网络的前向传播
    def mock_forward_features(inputs):
        batch_size = inputs.size(0)
        return torch.randn(batch_size, 512)
    
    model.network = Mock()
    model.network.forward_features = mock_forward_features
    model.network.eval = Mock()
    
    # 创建模拟的数据管理器
    mock_data_manager = Mock()
    mock_data_manager.nb_tasks = 3  # 总共3个任务
    mock_data_manager.dataset_names = ['imagenet-r', 'caltech-101', 'dtd']
    mock_data_manager.global_label_offset = [0, 100, 110]
    mock_data_manager.get_task_size = Mock(side_effect=lambda task_id: [100, 10, 101][task_id])
    model.data_manager = mock_data_manager
    
    # 创建模拟的数据加载器，包含所有任务的数据
    # 创建一些样本，每个样本属于不同的任务
    inputs = torch.randn(64, 3, 224, 224)
    # 创建标签：前20个属于任务0，中间20个属于任务1，最后24个属于任务2
    targets = torch.cat([
        torch.randint(0, 100, (20,)),  # 任务0的标签
        torch.randint(100, 110, (20,)),  # 任务1的标签
        torch.randint(110, 211, (24,))  # 任务2的标签
    ])
    
    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # 创建模拟的fc_dict
    fc_dict = {}
    for variant_name in ['SeqFT + LDA']:
        fc = Mock(spec=nn.Linear)
        def mock_fc_forward(features):
            batch_size = features.size(0)
            # 返回模拟的logits
            return torch.randn(batch_size, 211)  # 总共211个类别
        fc.side_effect = mock_fc_forward
        fc.to = Mock()
        fc_dict[variant_name] = fc
    
    return model, loader, fc_dict

def main():
    """主测试函数"""
    print("=" * 60)
    print("测试cross-domain评估的调试日志")
    print("=" * 60)
    
    # 设置日志级别
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(filename)s] => %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 创建测试数据
    model, loader, fc_dict = setup_simple_test()
    
    print(f"\n当前任务ID: {model.current_task_id}")
    print(f"总任务数: {model.data_manager.nb_tasks}")
    print(f"应该评估的任务: 0 到 {model.current_task_id - 1}")
    
    print("\n=== 调用 _evaluate_cross_domain 方法 ===")
    results = model._evaluate_cross_domain(loader, fc_dict)
    
    print(f"\n=== 返回的结果 ===")
    for name, acc in results.items():
        print(f"{name}: {acc:.2f}%")
    
    print("\n=== 验证假设 ===")
    print("1. 检查调试日志是否显示当前任务ID和总任务数")
    print("2. 检查是否区分了已学习和未学习的任务")
    print("3. 检查是否显示了两种不同的平均准确度计算方式")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)