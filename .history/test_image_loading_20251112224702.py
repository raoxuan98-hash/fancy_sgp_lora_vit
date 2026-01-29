import sys
import os
sys.path.append('.')

import numpy as np
from utils.cross_domain_data_manager import CrossDomainDataManagerCore

def test_image_loading():
    """测试累积数据集后图片是否能正常读取"""
    print("=== 测试累积数据集的图片读取 ===\n")
    
    # 使用较小的数据集进行测试
    dataset_names = ['cifar100_224', 'mnist', 'caltech-101']
    
    try:
        # 初始化数据管理器
        cdm = CrossDomainDataManagerCore(dataset_names, log_level=20)
        
        # 检查每个数据集的数据类型
        print("=== 检查数据集类型 ===")
        for i, dataset in enumerate(cdm.datasets):
            print(f"数据集 {i} ({dataset['name']}):")
            print(f"  use_path: {dataset['use_path']}")
            print(f"  train_data类型: {type(dataset['train_data'])}")
            print(f"  train_data形状: {dataset['train_data'].shape if hasattr(dataset['train_data'], 'shape') else 'N/A'}")
            print(f"  train_targets范围: {np.min(dataset['train_targets'])} - {np.max(dataset['train_targets'])}")
            print()
        
        # 测试非累积模式的图片读取
        print("=== 测试非累积模式的图片读取 ===")
        for task_id in range(cdm.nb_tasks):
            train_subset = cdm.get_subset(task_id, source="train", cumulative=False)
            print(f"任务 {task_id} 非累积模式:")
            
            try:
                # 尝试读取前3个样本
                for i in range(3):
                    image, label, class_name = train_subset[i]
                    print(f"  样本 {i}: 标签={label}, 类名={class_name}, 图像类型={type(image)}, 图像大小={image.size if hasattr(image, 'size') else 'N/A'}")
                print("  ✓ 非累积模式图片读取成功")
            except Exception as e:
                print(f"  ✗ 非累积模式图片读取失败: {e}")
            print()
        
        # 测试累积模式的图片读取
        print("=== 测试累积模式的图片读取 ===")
        for task_id in range(cdm.nb_tasks):
            train_subset = cdm.get_subset(task_id, source="train", cumulative=True)
            print(f"任务 {task_id} 累积模式:")
            
            try:
                # 尝试读取前3个样本
                for i in range(3):
                    image, label, class_name = train_subset[i]
                    print(f"  样本 {i}: 标签={label}, 类名={class_name}, 图像类型={type(image)}, 图像大小={image.size if hasattr(image, 'size') else 'N/A'}")
                print("  ✓ 累积模式图片读取成功")
            except Exception as e:
                print(f"  ✗ 累积模式图片读取失败: {e}")
                import traceback
                traceback.print_exc()
            print()
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_image_loading()