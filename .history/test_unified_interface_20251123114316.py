#!/usr/bin/env python3
"""
测试统一接口：get_incremental_subset 可以在所有情况下使用
"""

from utils.balanced_cross_domain_data_manager import create_balanced_data_manager

def test_unified_interface():
    """测试统一接口在各种配置下都能正常工作"""
    print("=== 测试统一接口 get_incremental_subset ===\n")
    
    datasets = ['cifar100_224']
    
    # 测试场景1: enable_incremental_split = False
    print("📋 场景1: enable_incremental_split = False")
    manager1 = create_balanced_data_manager(
        dataset_names=datasets,
        balanced_datasets_root="balanced_datasets",
        use_balanced_datasets=True,
        enable_incremental_split=False
    )
    
    print(f"   配置: 总任务数={manager1.nb_tasks}, 增量拆分={manager1.enable_incremental_split}")
    
    try:
        # 现在应该可以统一使用 get_incremental_subset
        train_set = manager1.get_incremental_subset(
            task=0, source="train", cumulative=False, mode="train")
        test_set = manager1.get_incremental_subset(
            task=0, source="test", cumulative=True, mode="test")
        
        print(f"   ✅ 成功: 训练集={len(train_set)}, 测试集={len(test_set)}")
        
    except Exception as e:
        print(f"   ❌ 失败: {e}")
        return False
    
    # 测试场景2: enable_incremental_split = True
    print(f"\n📋 场景2: enable_incremental_split = True")
    manager2 = create_balanced_data_manager(
        dataset_names=datasets,
        balanced_datasets_root="balanced_datasets",
        use_balanced_datasets=True,
        enable_incremental_split=True,
        num_incremental_splits=2
    )
    
    print(f"   配置: 总任务数={manager2.nb_tasks}, 增量拆分={manager2.enable_incremental_split}")
    
    try:
        # 统一使用 get_incremental_subset
        for task_id in range(manager2.nb_tasks):
            train_set = manager2.get_incremental_subset(
                task=task_id, source="train", cumulative=False, mode="train")
            test_set = manager2.get_incremental_subset(
                task=task_id, source="test", cumulative=True, mode="test")
            
            print(f"   ✅ 任务 {task_id}: 训练集={len(train_set)}, 测试集={len(test_set)}")
        
    except Exception as e:
        print(f"   ❌ 失败: {e}")
        return False
    
    # 测试场景3: 在 subspace_lora.py 中的典型用法
    print(f"\n📋 场景3: 模拟 subspace_lora.py 用法")
    
    # 场景3a: 常规模式
    print(f"   3a. 常规模式:")
    manager3a = create_balanced_data_manager(
        dataset_names=['cifar100_224', 'cub200_224'],
        balanced_datasets_root="balanced_datasets",
        use_balanced_datasets=True,
        enable_incremental_split=False
    )
    
    try:
        task_id = 0
        train_set = manager3a.get_incremental_subset(
            task=task_id, source="train", cumulative=False, mode="train")
        test_set = manager3a.get_incremental_subset(
            task=task_id, source="test", cumulative=True, mode="test")
        train_set_test_mode = manager3a.get_incremental_subset(
            task=task_id, source="train", cumulative=False, mode="test")
        
        print(f"      ✅ 任务 {task_id}: 训练={len(train_set)}, 测试={len(test_set)}, 训练测试={len(train_set_test_mode)}")
        
    except Exception as e:
        print(f"      ❌ 失败: {e}")
        return False
    
    # 场景3b: 增量拆分模式
    print(f"   3b. 增量拆分模式:")
    manager3b = create_balanced_data_manager(
        dataset_names=['cifar100_224'],
        balanced_datasets_root="balanced_datasets",
        use_balanced_datasets=True,
        enable_incremental_split=True,
        num_incremental_splits=3
    )
    
    try:
        task_id = 1  # 第二个增量任务
        train_set = manager3b.get_incremental_subset(
            task=task_id, source="train", cumulative=False, mode="train")
        test_set = manager3b.get_incremental_subset(
            task=task_id, source="test", cumulative=True, mode="test")
        train_set_test_mode = manager3b.get_incremental_subset(
            task=task_id, source="train", cumulative=False, mode="test")
        
        print(f"      ✅ 任务 {task_id}: 训练={len(train_set)}, 测试={len(test_set)}, 训练测试={len(train_set_test_mode)}")
        
    except Exception as e:
        print(f"      ❌ 失败: {e}")
        return False
    
    return True

def demonstrate_simplified_usage():
    """演示简化的使用方式"""
    print(f"\n=== 简化使用指南 ===\n")
    
    print("🎯 现在可以统一使用 get_incremental_subset 方法！")
    print()
    print("无论 enable_incremental_split 是 True 还是 False，都可以使用:")
    print()
    
    print("# 数据管理器创建（根据需要设置 enable_incremental_split）")
    print("manager = create_balanced_data_manager(")
    print("    dataset_names=['cifar100_224', 'cub200_224'],")
    print("    balanced_datasets_root='balanced_datasets',")
    print("    use_balanced_datasets=True,")
    print("    enable_incremental_split=True,  # 可选：True 或 False")
    print("    num_incremental_splits=3       # 仅在 enable_incremental_split=True 时使用")
    print(")")
    print()
    
    print("# 统一的数据获取方式")
    print("for task_id in range(manager.nb_tasks):")
    print("    train_set = manager.get_incremental_subset(")
    print("        task=task_id, source='train', cumulative=False, mode='train')")
    print("    test_set = manager.get_incremental_subset(")
    print("        task=task_id, source='test', cumulative=True, mode='test')")
    print("    train_set_test_mode = manager.get_incremental_subset(")
    print("        task=task_id, source='train', cumulative=False, mode='test')")
    print("    # 使用这些数据集进行训练和评估...")
    print()
    
    print("✅ 好处:")
    print("   - 无需根据配置动态选择方法")
    print("   - 代码更简洁、更易维护")
    print("   - 在 subspace_lora.py 中可以直接替换原有的 get_subset 调用")

def main():
    """主函数"""
    success = test_unified_interface()
    
    if success:
        print(f"\n🎉 所有测试通过！get_incremental_subset 现在是真正的统一接口")
        demonstrate_simplified_usage()
    else:
        print(f"\n❌ 测试失败，请检查实现")
    
    return success

if __name__ == "__main__":
    main()