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
        'eval_only': False,
        'log_path': './test_logs'
    }
    
    # 创建模拟的模型
    with patch('models.subspace_lora.BaseNet'):
        with patch('models.subspace_lora.DistributionCompensator'):
            with patch('models.subspace_lora.ClassifierReconstructor'):
                with patch('models.subspace_lora.Distiller'):
                    model = SubspaceLoRA(mock_args)
    
    # 模拟网络的前向传播
    def mock_forward_features(inputs):
        # 返回模拟的特征
        batch_size = inputs.size(0)
        return torch.randn(batch_size, 512)
    
    model.network = Mock()
    model.network.forward_features = mock_forward_features
    model.network.eval = Mock()
    
    # 模拟 fc_dict，包含多个分类器变体
    fc_dict = {}
    for variant_name in ['variant1', 'variant2', 'variant3']:
        fc = Mock(spec=nn.Linear)
        # 模拟分类器的输出，返回不同准确度的预测
        def mock_fc_forward(self, features, variant_name=variant_name):
            batch_size = features.size(0)
            # 返回模拟的logits
            return torch.randn(batch_size, 100)  # 假设有100个类别
        
        fc.side_effect = lambda features, variant_name=variant_name: mock_fc_forward(fc, features, variant_name)
        fc.to = Mock()
        fc_dict[variant_name] = fc
    
    return model, fc_dict

def test_syntax_errors():
    """测试代码是否有语法错误"""
    print("=== 测试代码语法错误 ===")
    try:
        # 尝试导入模块
        from models.subspace_lora import SubspaceLoRA
        print("✓ 代码导入成功，没有语法错误")
        return True
    except SyntaxError as e:
        print(f"✗ 语法错误: {e}")
        return False
    except Exception as e:
        print(f"✗ 其他错误: {e}")
        return False

def test_evaluate_cross_domain():
    """测试 _evaluate_cross_domain() 方法"""
    print("\n=== 测试 _evaluate_cross_domain() 方法 ===")
    
    try:
        # 设置模拟数据和模型
        mock_data_manager, loader = setup_mock_data()
        model, fc_dict = setup_mock_model()
        
        # 设置模型的 data_manager 属性
        model.data_manager = mock_data_manager
        
        # 设置日志级别
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        
        # 调用 _evaluate_cross_domain 方法
        print("调用 _evaluate_cross_domain() 方法...")
        results = model._evaluate_cross_domain(loader, fc_dict)
        
        # 验证结果
        print(f"返回的结果: {results}")
        
        # 检查结果是否包含所有变体
        for variant_name in fc_dict.keys():
            if variant_name not in results:
                print(f"✗ 缺少变体 {variant_name} 的结果")
                return False
            print(f"✓ 变体 {variant_name} 的平均准确度: {results[variant_name]:.2f}%")
        
        # 检查结果是否为浮点数
        for variant_name, accuracy in results.items():
            if not isinstance(accuracy, (float, np.floating)):
                print(f"✗ 变体 {variant_name} 的准确度不是浮点数: {type(accuracy)}")
                return False
            if not (0 <= accuracy <= 100):
                print(f"✗ 变体 {variant_name} 的准确度不在0-100范围内: {accuracy}")
                return False
        
        print("✓ _evaluate_cross_domain() 方法测试通过")
        return True
        
    except Exception as e:
        print(f"✗ _evaluate_cross_domain() 方法测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_evaluation_logic():
    """测试评估逻辑是否按数据集平均"""
    print("\n=== 测试评估逻辑是否按数据集平均 ===")
    
    try:
        # 设置模拟数据和模型
        mock_data_manager, loader = setup_mock_data()
        model, fc_dict = setup_mock_model()
        
        # 设置模型的 data_manager 属性
        model.data_manager = mock_data_manager
        
