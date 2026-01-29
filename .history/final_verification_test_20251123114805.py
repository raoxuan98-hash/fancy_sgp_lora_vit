#!/usr/bin/env python3
"""
最终验证测试：确保 subspace_lora.py 中的 get_incremental_subset 调用正常工作
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.balanced_cross_domain_data_manager import create_balanced_data_manager

def test_subspace_lora_pattern():
    """
    测试模拟 subspace_lora.py 中的使用模式
    """
    print("=== 验证 subspace_lora.py 中的 get_incremental_subset 使用 ===\n")
    
    # 模拟两种常见的配置
    test_configs = [
        {"enable_incremental_split": False},
        {"enable_incremental_split": True, "num_incremental_splits": 3}
    ]
    
    config_names = ["常规模式", "增量拆分模式"]
    
    for i, config in enumerate(test_configs):
        config_name = config_names[i]
        print(f"🧪 测试配置: {config_name}")
        print(f"   enable_incremental_split: {config['enable_incremental_split']}")
        
        try:
            # 创建数据管理器
            manager = create_balanced_data_manager(
                dataset_names=['cifar100_224'],
                balanced_datasets_root="balanced_datasets",
                use_balanced_datasets=True,
                **config
            )
            
            print(f"   ✅ 数据管理器创建成功")
            print(f"      总任务数: {manager.nb_tasks}")
            print(f"      总类别数: {manager.num_classes}")
            
            # 模拟 subspace_lora.py 中的调用模式
            task_id = 0
            task_size = manager.get_task_size(task_id)
            
            print(f"   🧪 模拟 subspace_lora.py 调用模式:")
            
            # 这是 subspace_lora.py 中实际使用的调用
            train_set = manager.get_incremental_subset(
                task=task_id, source="train", cumulative=False, mode="train")
            test_set = manager.get_incremental_subset(
                task=task_id, source="test", cumulative=True, mode="test")
            train_set_test_mode = manager.get_incremental_subset(
                task=task_id, source="train", cumulative=False, mode="test")
            
            print(f"      ✅ 任务 {task_id} 调用成功")
            print(f"         任务大小: {task_size} 类别")
            print(f"         训练集大小: {len(train_set)} 样本")
            print(f"         测试集大小: {len(test_set)} 样本 (累积模式)")
            print(f"         训练测试集大小: {len(train_set_test_mode)} 样本")
            
            # 验证累积模式的正确性
            if config["enable_incremental_split"]:
                # 在增量拆分模式下，测试集应该包含当前任务的所有数据
                print(f"      🔍 验证累积模式:")
                expected_test_size = sum(len(manager.datasets[i]['test_data']) 
                                       for i in range(min(task_id + 1, manager.nb_tasks)))
                if len(test_set) == expected_test_size:
                    print(f"         ✅ 累积模式验证通过: {len(test_set)} == {expected_test_size}")
                else:
                    print(f"         ❌ 累积模式验证失败: {len(test_set)} != {expected_test_size}")
                    return False
            else:
                # 在常规模式下，测试集应该包含从任务0到当前任务的所有数据
                print(f"      🔍 验证累积模式:")
                expected_test_size = len(manager.datasets[0]['test_data'])
                if len(test_set) == expected_test_size:
                    print(f"         ✅ 累积模式验证通过: {len(test_set)} == {expected_test_size}")
                else:
                    print(f"         ❌ 累积模式验证失败: {len(test_set)} != {expected_test_size}")
                    return False
            
            print(f"      ✅ {config_name} 测试通过\n")
            
        except Exception as e:
            print(f"      ❌ {config_name} 测试失败: {e}")
            return False
    
    return True

def test_backwards_compatibility():
    """
    测试向后兼容性：确保原有的 get_subset 调用仍然有效
    """
    print("=== 验证向后兼容性 ===\n")
    
    try:
        manager = create_balanced_data_manager(
            dataset_names=['cifar100_224'],
            balanced_datasets_root="balanced_datasets",
            use_balanced_datasets=True,
            enable_incremental_split=False
        )
        
        print("🧪 测试原有的 get_subset 方法仍然有效:")
        
        # 使用原有的 get_subset 方法
        train_set_old = manager.get_subset(
            task=0, source="train", cumulative=False, mode="train")
        test_set_old = manager.get_subset(
            task=0, source="test", cumulative=True, mode="test")
        
        # 使用新的 get_incremental_subset 方法
        train_set_new = manager.get_incremental_subset(
            task=0, source="train", cumulative=False, mode="train")
        test_set_new = manager.get_incremental_subset(
            task=0, source="test", cumulative=True, mode="test")
        
        # 比较结果
        if (len(train_set_old) == len(train_set_new) and 
            len(test_set_old) == len(test_set_new)):
            print("   ✅ 向后兼容性测试通过")
            print(f"      训练集: {len(train_set_old)} == {len(train_set_new)}")
            print(f"      测试集: {len(test_set_old)} == {len(test_set_new)}")
            return True
        else:
            print("   ❌ 向后兼容性测试失败")
            return False
            
    except Exception as e:
        print(f"   ❌ 向后兼容性测试失败: {e}")
        return False

def main():
    """主函数"""
    print("开始最终验证测试\n")
    
    # 运行测试
    test1_passed = test_subspace_lora_pattern()
    test2_passed = test_backwards_compatibility()
    
    if test1_passed and test2_passed:
        print("🎉 所有验证测试通过！")
        print("\n📋 总结:")
        print("   ✅ get_incremental_subset 在 subspace_lora.py 中正常工作")
        print("   ✅ 支持 cumulative 参数")
        print("   ✅ 支持 enable_incremental_split=True/False 两种配置")
        print("   ✅ 保持向后兼容性")
        print("   ✅ 代码替换成功")
        return True
    else:
        print("❌ 部分测试失败")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)