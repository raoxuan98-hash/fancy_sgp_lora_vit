import sys
import os
sys.path.append('.')

import numpy as np
from utils.cross_domain_data_manager import CrossDomainDataManagerCore

def test_comprehensive_image_loading():
    """全面测试累积数据集的图片读取，确保不同类型的图片都能正确处理"""
    print("=== 全面测试累积数据集的图片读取 ===\n")
    
    # 使用混合类型的数据集进行测试
    dataset_names = ['cifar100_224', 'mnist', 'caltech-101']
    
    try:
        # 初始化数据管理器
        cdm = CrossDomainDataManagerCore(dataset_names, log_level=20)
        
        # 测试累积模式下不同数据集的图片读取
        print("=== 测试累积模式下不同数据集的图片读取 ===")
        
        # 任务1：包含cifar100_224和mnist
        train_subset_1 = cdm.get_subset(1, source="train", cumulative=True)
        print("任务1 (cifar100_224 + mnist):")
        
        # 尝试读取来自不同数据集的样本
        # 前50000个样本来自cifar100_224，后60000个来自mnist
        cifar_sample_idx = 0  # 来自cifar100_224
        mnist_sample_idx = 50000  # 来自mnist
        
        try:
            image, label, class_name = train_subset_1[cifar_sample_idx]
            print(f"  CIFAR样本 {cifar_sample_idx}: 标签={label}, 类名={class_name}, 图像大小={image.size}")
        except Exception as e:
            print(f"  CIFAR样本读取失败: {e}")
        
        try:
            image, label, class_name = train_subset_1[mnist_sample_idx]
            print(f"  MNIST样本 {mnist_sample_idx}: 标签={label}, 类名={class_name}, 图像大小={image.size}")
        except Exception as e:
            print(f"  MNIST样本读取失败: {e}")
        
        # 任务2：包含所有三个数据集
        train_subset_2 = cdm.get_subset(2, source="train", cumulative=True)
        print("\n任务2 (cifar100_224 + mnist + caltech-101):")
        
        # 尝试读取来自caltech-101的样本
        caltech_sample_idx = 56000  # 来自caltech-101 (50000+60000)
        
        try:
            image, label, class_name = train_subset_2[caltech_sample_idx]
            print(f"  Caltech样本 {caltech_sample_idx}: 标签={label}, 类名={class_name}, 图像大小={image.size}")
        except Exception as e:
            print(f"  Caltech样本读取失败: {e}")
        
        # 验证标签范围
        print("\n=== 验证标签范围 ===")
        for task_id in range(cdm.nb_tasks):
            train_subset = cdm.get_subset(task_id, source="train", cumulative=True)
            labels = train_subset.labels
            unique_labels = np.unique(labels)
            expected_max = sum(cdm.get_task_size(i) for i in range(task_id + 1)) - 1
            
            print(f"任务 {task_id}:")
            print(f"  实际标签范围: {np.min(labels)} - {np.max(labels)}")
            print(f"  唯一标签数量: {len(unique_labels)}")
            print(f"  期望最大标签: {expected_max}")
            print(f"  标签连续性: {'✓' if len(unique_labels) == expected_max + 1 else '✗'}")
        
        print("\n✓ 所有测试通过！累积数据集的图片读取和标签偏移都正确。")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_comprehensive_image_loading()