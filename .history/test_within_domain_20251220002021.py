#!/usr/bin/env python3
"""
简单的测试脚本，用于验证 within_domain 数据加载功能
"""
import os
import sys
import torch

# 根据你的环境保留路径设置
sys.path.append('/home/raoxuan/projects/low_rank_rda')
try:
    os.chdir('/home/raoxuan/projects/low_rank_rda')
    print("当前工作目录:", os.getcwd())
except FileNotFoundError:
    print("注意: 目录不存在，请检查路径。当前在:", os.getcwd())

from classifier_ablation.data.within_domain_data_loader import load_within_domain_data, create_data_loaders
from classifier_ablation.features.feature_extractor import get_vit

def main():
    print("开始测试 within_domain 数据加载...")
    
    try:
        # 1. 加载数据
        print("\n1. 加载 cifar100_224 数据集...")
        dataset, train_subset, test_subset = load_within_domain_data(
            dataset_name="cifar100_224", 
            init_cls=5, 
            increment=5, 
            model_name="vit-b-p16", 
            seed=42
        )
        print(f"   数据集加载成功，任务数: {dataset.nb_tasks}")
        
        # 2. 创建数据加载器
        print("\n2. 创建数据加载器...")
        train_loader, test_loader = create_data_loaders(train_subset, test_subset, batch_size=8, num_workers=0)
        print(f"   训练加载器创建成功，批次数: {len(train_loader)}")
        print(f"   测试加载器创建成功，批次数: {len(test_loader)}")
        
        # 3. 测试数据加载
        print("\n3. 测试数据加载...")
        train_batch = next(iter(train_loader))
        test_batch = next(iter(test_loader))
        
        print(f"   训练批次形状: {train_batch[0].shape}, 标签形状: {train_batch[1].shape}")
        print(f"   测试批次形状: {test_batch[0].shape}, 标签形状: {test_batch[1].shape}")
        
        # 4. 加载模型
        print("\n4. 加载 ViT 模型...")
        vit = get_vit(vit_name="vit-b-p16")
        print(f"   模型加载成功")
        
        # 5. 测试模型前向传播
        print("\n5. 测试模型前向传播...")
        with torch.no_grad():
            train_features = vit(train_batch[0])
            test_features = vit(test_batch[0])
            
        print(f"   训练特征形状: {train_features.shape}")
        print(f"   测试特征形状: {test_features.shape}")
        
        print("\n✅ 所有测试通过！within_domain 数据加载功能正常工作。")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)