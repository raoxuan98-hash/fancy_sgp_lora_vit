#!/usr/bin/env python3
"""
调试跨域训练中的准确率异常问题
"""

import logging
import torch
import numpy as np
from torch.utils.data import DataLoader

from utils.cross_domain_data_manager import CrossDomainDataManager, DataManager
from models.subspace_lora import SubspaceLoRA

# 设置日志级别为DEBUG以查看更多信息
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(filename)s] => %(message)s')

def test_cross_domain_data_manager():
    """测试CrossDomainDataManager的标签处理"""
    print("=" * 80)
    print("测试 CrossDomainDataManager 的标签处理")
    print("=" * 80)
    
    # 使用简化的数据集列表进行测试
    dataset_names = ['caltech-101', 'dtd', 'eurosat_clip']
    
    cdm = CrossDomainDataManager(
        dataset_names=dataset_names,
        shuffle=False,
        seed=1993
    )
    
    print(f"总数据集数量: {cdm.nb_tasks}")
    print(f"总类别数: {cdm.num_classes}")
    
    # 测试每个任务的非累积模式数据集
    for task_id in range(cdm.nb_tasks):
        print(f"\n--- 任务 {task_id} ({dataset_names[task_id]}) ---")
        
        # 获取训练集（非累积模式）
        train_dataset = cdm.get_subset(
            task_id=task_id, 
            source="train", 
            cumulative=False,
            transform=None
        )
        
        # 获取测试集（累积模式）
        test_dataset = cdm.get_subset(
            task_id=task_id, 
            source="test", 
            cumulative=True,
            transform=None
        )
        
        print(f"训练集大小: {len(train_dataset)}")
        print(f"测试集大小: {len(test_dataset)}")
        
        # 检查训练集的标签范围
        train_labels = []
        train_class_names = []
        for i in range(min(10, len(train_dataset))):
            _, label, class_name = train_dataset[i]
            train_labels.append(label)
            if class_name is not None:
                train_class_names.append(class_name)
        
        print(f"训练集前10个标签: {train_labels}")
        print(f"训练集标签范围: {min(train_labels)}-{max(train_labels)}")
        print(f"训练集类名数量: {len(train_dataset.class_names) if train_dataset.class_names else 0}")
        
        # 检查测试集的标签范围
        test_labels = []
        for i in range(min(10, len(test_dataset))):
            _, label, _ = test_dataset[i]
            test_labels.append(label)
        
        print(f"测试集前10个标签: {test_labels}")
        print(f"测试集标签范围: {min(test_labels)}-{max(test_labels)}")
        print(f"测试集类名数量: {len(test_dataset.class_names) if test_dataset.class_names else 0}")

def test_data_manager_wrapper():
    """测试DataManager包装器"""
    print("\n" + "=" * 80)
    print("测试 DataManager 包装器")
    print("=" * 80)
    
    args = {
        'dataset': 'cross_domain_elevater',
        'cross_domain': True,
        'cross_domain_datasets': ['caltech-101', 'dtd', 'eurosat_clip'],
        'shuffle': False,
        'seed': 1993,
        'init_cls': 10,
        'increment': 10
    }
    
    dm = DataManager(
        dataset_name=args['dataset'],
        shuffle=args['shuffle'],
        seed=args['seed'],
        init_cls=args['init_cls'],
        increment=args['increment'],
        args=args
    )
    
    print(f"总任务数: {dm.nb_tasks}")
    
    # 测试第一个任务
    task_id = 0
    train_dataset = dm.get_subset(
        task=task_id, 
        source="train", 
        cumulative=False, 
        mode="train"
    )
    
    test_dataset = dm.get_subset(
        task=task_id, 
        source="test", 
        cumulative=True, 
        mode="test"
    )
    
    print(f"任务 {task_id} 训练集大小: {len(train_dataset)}")
    print(f"任务 {task_id} 测试集大小: {len(test_dataset)}")
    
    # 检查DataLoader
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
    
    for batch_idx, (images, labels, class_names) in enumerate(train_loader):
        if batch_idx >= 2:  # 只检查前两个batch
            break
        
        print(f"\nBatch {batch_idx}:")
        print(f"  图像形状: {images.shape}")
        print(f"  标签: {labels.tolist()}")
        print(f"  标签范围: {labels.min().item()}-{labels.max().item()}")
        print(f"  类名: {class_names[:2] if class_names else None}")  # 只显示前两个

def test_model_initialization():
    """测试模型初始化和分类器维度"""
    print("\n" + "=" * 80)
    print("测试模型初始化和分类器维度")
    print("=" * 80)
    
    args = {
        'dataset': 'cross_domain_elevater',
        'cross_domain': True,
        'cross_domain_datasets': ['caltech-101', 'dtd', 'eurosat_clip'],
        'shuffle': False,
        'seed': 1993,
        'init_cls': 10,
        'increment': 10,
        'model_name': 'sldc',
        'vit_type': 'vit-b-p16-mocov3',
        'lora_type': 'sgp_lora',
        'lora_rank': 4,
        'batch_size': 4,
        'iterations': 10,  # 只训练很少的步数用于测试
        'lrate': 1e-4,
        'weight_decay': 3e-5,
        'optimizer': 'adamw',
        'gamma_kd': 0.0,
        'weight_temp': 2.0,
        'weight_kind': 'log1p',
        'weight_p': 1.0,
        'lda_reg_alpha': 0.1,
        'qda_reg_alpha1': 0.2,
        'qda_reg_alpha2': 0.9,
        'qda_reg_alpha3': 0.2,
        'auxiliary_data_size': 1024,
        'auxiliary_data_path': '/data1/open_datasets',
        'aux_dataset': 'imagenet',
        'eval_only': True
    }
    
    # 初始化数据管理器
    dm = DataManager(
        dataset_name=args['dataset'],
        shuffle=args['shuffle'],
        seed=args['seed'],
        init_cls=args['init_cls'],
        increment=args['increment'],
        args=args
    )
    
    # 初始化模型
    model = SubspaceLoRA(args)
    
    print(f"模型特征维度: {model.network.feature_dim}")
    
    # 模拟训练第一个任务
    task_id = 0
    print(f"\n模拟训练任务 {task_id}...")
    
    # 获取数据
    train_dataset = dm.get_subset(
        task=task_id, 
        source="train", 
        cumulative=False, 
        mode="train"
    )
    
    test_dataset = dm.get_subset(
        task=task_id, 
        source="test", 
        cumulative=True, 
        mode="test"
    )
    
    # 更新分类器
    task_size = dm.get_task_size(task_id)
    model.network.update_fc(task_size)
    
    print(f"任务 {task_id} 类别数: {task_size}")
    print(f"分类器输出维度: {model.network.fc.current_output_size}")
    
    # 检查DataLoader
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    # 检查第一个batch
    for batch_idx, (images, labels, class_names) in enumerate(train_loader):
        if batch_idx >= 1:  # 只检查第一个batch
            break
        
        print(f"\n训练 Batch {batch_idx}:")
        print(f"  图像形状: {images.shape}")
        print(f"  标签: {labels.tolist()}")
        print(f"  标签范围: {labels.min().item()}-{labels.max().item()}")
        
        # 前向传播
        with torch.no_grad():
            feats = model.network.forward_features(images)
            logits = model.network.fc(feats)
            
        print(f"  特征形状: {feats.shape}")
        print(f"  Logits形状: {logits.shape}")
        print(f"  预测: {logits.argmax(dim=1).tolist()}")
        
        # 检查预测是否在有效范围内
        preds = logits.argmax(dim=1)
        max_label = labels.max().item()
        max_pred = preds.max().item()
        
        if max_pred > max_label:
            print(f"  警告: 预测标签 {max_pred} 超出真实标签范围 {max_label}")
        else:
            print(f"  预测标签在有效范围内")
    
    # 检查测试集
    for batch_idx, (images, labels, class_names) in enumerate(test_loader):
        if batch_idx >= 1:  # 只检查第一个batch
            break
        
        print(f"\n测试 Batch {batch_idx}:")
        print(f"  图像形状: {images.shape}")
        print(f"  标签: {labels.tolist()}")
        print(f"  标签范围: {labels.min().item()}-{labels.max().item()}")
        
        # 前向传播
        with torch.no_grad():
            feats = model.network.forward_features(images)
            logits = model.network.fc(feats)
            
        print(f"  特征形状: {feats.shape}")
        print(f"  Logits形状: {logits.shape}")
        print(f"  预测: {logits.argmax(dim=1).tolist()}")
        
        # 检查预测是否在有效范围内
        preds = logits.argmax(dim=1)
        max_label = labels.max().item()
        max_pred = preds.max().item()
        
        if max_pred > max_label:
            print(f"  警告: 预测标签 {max_pred} 超出真实标签范围 {max_label}")
        else:
            print(f"  预测标签在有效范围内")

if __name__ == "__main__":
    test_cross_domain_data_manager()
    test_data_manager_wrapper()
    test_model_initialization()