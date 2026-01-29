#!/usr/bin/env python3
"""
测试 models/subspace_lora.py 中的跨域评估功能
特别测试 _evaluate_cross_domain() 方法的实现
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
from utils.data_manager import CrossDomainDataManager

def setup_mock_data():
    """创建模拟数据用于测试"""
    # 创建模拟的跨域数据管理器
    mock_data_manager = Mock(spec=CrossDomainDataManager)
    mock_data_manager.nb_tasks = 3
    mock_data_manager.dataset_names = ['cifar100_224', 'mnist', 'caltech-101']
    mock_data_manager.global_label_offset = [0, 100, 110]  # 每个数据集的标签偏移
    mock_data_manager.get_task_size = Mock(side_effect=lambda task_id: [100, 10, 101][task_id])
    
    # 创建模拟的数据加载器
    # 为每个任务创建不同的数据，模拟跨域场景
    batch_size = 32
    num_batches = 2
    
    # 创建模拟数据：每个batch包含来自不同任务的数据
    all_inputs = []
    all_targets = []
    
    for batch_idx in range(num_batches):
        # 为每个任务创建一些样本
        batch_inputs = []
        batch_targets = []
        
        for task_id in range(mock_data_manager.nb_tasks):
            # 每个任务创建几个样本
            task_samples = 8
            task_start = mock_data_manager.global_label_offset[task_id]
            task_end = task_start + mock_data_manager.get_task_size(task_id)
            
            # 随机生成标签在任务范围内
            task_labels = torch.randint(task_start, min(task_end, task_start + 10), (task_samples,))
            task_images = torch.randn(task_samples, 3, 224, 224)
            
            batch_inputs.append(task_images)
            batch_targets.append(task_labels)
        
        # 合并所有任务的样本
        batch_inputs = torch.cat(batch_inputs, dim=0)
        batch_targets = torch.cat(batch_targets, dim=0)
        
        all_inputs.append(batch_inputs)
        all_targets.append(batch_targets)
    
    # 创建模拟的数据加载器
    dataset = TensorDataset(torch.cat(all_inputs, dim=0), torch.cat(all_targets, dim=0))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    return mock_data_manager, loader

def setup_mock_model():
    """创建模拟的 SubspaceLoRA 模型"""
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
