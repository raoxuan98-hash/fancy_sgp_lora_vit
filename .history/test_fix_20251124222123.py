#!/usr/bin/env python3
"""
测试脚本：验证修复是否有效
"""
import os
os.chdir('/home/raoxuan/projects/low_rank_rda')
print("当前工作目录:", os.getcwd())

# 测试数据加载
try:
    from classifier_ablation.data.data_loader import load_cross_domain_data, create_data_loaders
    
    print("测试数据加载...")
    dataset, train_subsets, test_subsets = load_cross_domain_data(num_shots=16, model_name="vit-b-p16")
    
    print("创建数据加载器...")
    train_loader, test_loader = create_data_loaders(train_subsets, test_subsets, batch_size=4, num_workers=0)
    
    print("测试数据迭代...")
    for i, batch in enumerate(train_loader):
        if i >= 3:  # 只测试前3个批次
            break
        images, labels, class_names = batch
        print(f"批次 {i}: 图像形状 {images.shape}, 标签 {labels}, 类别名称数量 {len(class_names) if class_names else 0}")
        
        # 检查标签是否在有效范围内
        if class_names is not None:
