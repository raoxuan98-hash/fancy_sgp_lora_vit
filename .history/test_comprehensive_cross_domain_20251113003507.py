#!/usr/bin/env python3
"""
全面测试修复后的cross-domain评估功能
测试不同任务ID下的行为
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

def create_test_model(task_id):
    """创建测试模型，设置指定的任务ID"""
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
    
    # 设置当前任务ID
    model.current_task_id = task_id
    
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
    inputs = torch.randn(64, 3, 224, 224)
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
            return torch.randn(batch_size, 211)
        fc.side_effect = mock_fc_forward
        fc.to = Mock()
        fc_dict[variant_name] = fc
    
    return model, loader, fc_dict

def test_task_id_1():
    """测试任务ID为1的情况（只学习了第一个任务）"""
    print("\n" + "="*50)
