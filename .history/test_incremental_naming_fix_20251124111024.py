#!/usr/bin/env python3
"""
测试增量拆分任务命名显示的修复效果
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, os.path.abspath('.'))

from models.subspace_lora import SubspaceLoRA
from utils.balanced_cross_domain_data_manager import BalancedCrossDomainDataManagerCore

def test_incremental_task_naming():
    """测试增量拆分后任务名称显示是否正确"""
    
    print("🧪 测试增量拆分任务命名显示修复")
    print("=" * 60)
    
    # 创建临时目录用于测试
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # 创建平衡数据管理器，启用增量拆分
            dataset_names = ['cifar100_224', 'imagenet-r', 'cars196_224']
            
            print(f"📊 原始数据集列表: {dataset_names}")
            
            # 创建数据管理器
            data_manager = BalancedCrossDomainDataManagerCore(
                dataset_names=dataset_names,
                balanced_datasets_root="balanced_datasets",
                shuffle=False,
                seed=42,
                num_shots=0,
                use_balanced_datasets=False,  # 使用原始数据集
                enable_incremental_split=True,
                num_incremental_splits=2,
                incremental_split_seed=42
            )
            
            print(f"📈 增量拆分后任务数量: {data_manager.nb_tasks}")
            print(f"📈 任务数据集信息:")
            for i, dataset in enumerate(data_manager.datasets):
                original_name = dataset.get('original_dataset_name', dataset['name'])
                print(f"  任务 {i}: {dataset['name']} -> 原始名称: {original_name}")
            
            # 模拟analyze_task_results函数中的数据集名称映射逻辑
            def test_dataset_name_mapping(data_manager, task_id, dataset_names):
                """测试数据集名称映射逻辑"""
                if hasattr(data_manager, 'datasets') and task_id < len(data_manager.datasets):
                    dataset_info = data_manager.datasets[task_id]
                    if 'original_dataset_name' in dataset_info:
                        # 增量拆分情况：使用原始数据集名称
                        dataset_name = dataset_info['original_dataset_name']
                    elif 'name' in dataset_info:
                        # 普通情况：使用数据集名称
                        dataset_name = dataset_info['name']
                    elif dataset_names and task_id < len(dataset_names):
                        # 回退到传入的dataset_names
                        dataset_name = dataset_names[task_id]
                    else:
                        # 最后回退
                        dataset_name = f"Task {task_id}"
                else:
                    # 回退方案
                    dataset_name = dataset_names[task_id] if dataset_names and task_id < len(dataset_names) else f"Task {task_id}"
                
                # 清理数据集名称，使其更美观
                if dataset_name.endswith('_split_0') or dataset_name.endswith('_split_1'):
                    dataset_name = dataset_name.split('_split_')[0]
                elif dataset_name.endswith('_224'):
                    # 保持_224后缀以区分不同分辨率
                    pass
                
                return dataset_name
            
            # 测试数据集名称映射
            print(f"\n🧪 测试数据集名称映射:")
            for task_id in range(data_manager.nb_tasks):
                mapped_name = test_dataset_name_mapping(data_manager, task_id, dataset_names)
                print(f"  任务 {task_id} -> {mapped_name}")
            
            # 验证映射结果
            expected_mapping = {
                0: 'cifar100_224',
                1: 'cifar100_224', 
                2: 'imagenet-r',
                3: 'imagenet-r',
                4: 'cars196_224',
                5: 'cars196_224'
            }
            
            print(f"\n✅ 验证映射结果:")
            success_count = 0
            for task_id in range(data_manager.nb_tasks):
                mapped_name = test_dataset_name_mapping(data_manager, task_id, dataset_names)
                expected_name = expected_mapping.get(task_id, f"Task {task_id}")
                
                if mapped_name == expected_name:
                    print(f"  ✅ 任务 {task_id}: {mapped_name} (正确)")
                    success_count += 1
                else:
                    print(f"  ❌ 任务 {task_id}: {mapped_name} (期望: {expected_name})")
            
            success_rate = success_count / data_manager.nb_tasks * 100
            print(f"\n📊 测试结果: {success_count}/{data_manager.nb_tasks} ({success_rate:.1f}%) 成功")
            
            if success_rate >= 90:
                print("🎉 增量拆分任务命名显示修复成功！")
                return True
            else:
                print("❌ 增量拆分任务命名显示修复失败！")
                return False
                
        except Exception as e:
            print(f"❌ 测试执行出错: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    success = test_incremental_task_naming()
    sys.exit(0 if success else 1)