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
