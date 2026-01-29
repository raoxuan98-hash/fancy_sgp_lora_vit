import sys
import os
sys.path.append('.')

import numpy as np
from utils.cross_domain_data_manager import CrossDomainDataManagerCore

def test_simple_label_offset():
    """简单测试标签偏移是否正确应用"""
    print("=== 简单测试标签偏移 ===\n")
    
    # 使用较小的数据集进行测试
    dataset_names = ['cifar100_224', 'mnist', 'caltech-101']
    
    try:
        # 初始化数据管理器
        cdm = CrossDomainDataManagerCore(dataset_names, log_level=20)
        
        # 测试get_subset函数中的累积模式
        print("=== 测试get_subset累积模式中的标签处理 ===")
        for task_id in range(cdm.nb_tasks):
            # 获取累积模式的数据集
            train_subset = cdm.get_subset(task_id, source="train", cumulative=True)
            
            # 检查数据集内部的标签数组
            subset_labels = train_subset.labels
            print(f"任务 {task_id} 累积模式:")
            print(f"  Subset内部标签范围: {np.min(subset_labels)} - {np.max(subset_labels)}")
            print(f"  期望范围: 0 - {sum(cdm.get_task_size(i) for i in range(task_id + 1)) - 1}")
            
            # 验证标签是否连续
            unique_labels = np.unique(subset_labels)
            expected_labels = np.arange(0, sum(cdm.get_task_size(i) for i in range(task_id + 1)))
            
            if np.array_equal(unique_labels, expected_labels):
                print(f"  ✓ 标签连续且正确")
            else:
                print(f"  ✗ 标签不连续或不正确")
                print(f"    实际唯一标签: {unique_labels[:10]}...{unique_labels[-10:] if len(unique_labels) > 10 else ''}")
                print(f"    期望标签: {expected_labels[:10]}...{expected_labels[-10:] if len(expected_labels) > 10 else ''}")
            print()
        
        print("测试完成！")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple_label_offset()