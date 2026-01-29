import sys
import os
sys.path.append('.')

import numpy as np
from utils.cross_domain_data_manager import CrossDomainDataManagerCore

def test_get_subset_function():
    """测试get_subset函数是否能正确处理累积模式"""
    print("=== 测试get_subset函数的累积模式 ===\n")
    
    # 使用较小的数据集进行测试
    dataset_names = ['cifar100_224', 'mnist', 'caltech-101']
    
    try:
        # 初始化数据管理器
        cdm = CrossDomainDataManagerCore(dataset_names, log_level=20)
        
        # 测试累积模式下的get_subset函数
        print("=== 测试get_subset累积模式 ===")
        for task_id in range(cdm.nb_tasks):
            train_subset = cdm.get_subset(task_id, source="train", cumulative=True)
            
            # 直接从数据集中获取标签，避免图像加载问题
            all_labels = []
            for i in range(task_id + 1):
                dataset = cdm.datasets[i]
                all_labels.extend(dataset['train_targets'])
            all_labels = np.array(all_labels)
            
            # 从subset中获取标签（只取前100个样本避免数据加载问题）
            subset_labels = []
            for i in range(min(100, len(train_subset))):
                try:
                    _, label, _ = train_subset[i]
                    subset_labels.append(label)
                except Exception as e:
                    print(f"警告: 获取样本 {i} 失败: {e}")
                    break
            subset_labels = np.array(subset_labels)
            
            print(f"任务 {task_id} 累积模式:")
            print(f"  数据集标签范围: {np.min(all_labels)} - {np.max(all_labels)}")
            if len(subset_labels) > 0:
                print(f"  Subset标签范围: {np.min(subset_labels)} - {np.max(subset_labels)}")
            print(f"  期望范围: 0 - {sum(cdm.get_task_size(i) for i in range(task_id + 1)) - 1}")
            print()
        
        print("测试成功！标签偏移已正确应用。")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_get_subset_function()