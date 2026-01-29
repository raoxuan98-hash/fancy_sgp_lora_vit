#!/usr/bin/env python3
"""
根据 enable_incremental_split 配置选择正确的数据获取方法
"""

from utils.balanced_cross_domain_data_manager import create_balanced_data_manager

def demonstrate_correct_usage():
    """
    演示在不同配置下应该如何选择方法
    """
    print("=== 根据 enable_incremental_split 配置选择正确方法 ===\n")
    
    datasets = ['cifar100_224', 'cub200_224']
    
    # 场景1: enable_incremental_split = False
    print("📋 场景1: enable_incremental_split = False")
    print("   - 每个原始数据集对应一个任务")
    print("   - 总任务数 = 数据集数量")
    print("   - 应该使用: get_subset 方法\n")
    
    try:
        manager_no_split = create_balanced_data_manager(
            dataset_names=datasets,
            balanced_datasets_root="balanced_datasets",
            use_balanced_datasets=True,
            enable_incremental_split=False  # 不启用增量拆分
        )
        
        print(f"✅ 成功创建数据管理器:")
        print(f"   - 总任务数: {manager_no_split.nb_tasks} (应该等于数据集数量)")
        print(f"   - 总类别数: {manager_no_split.num_classes}")
        print(f"   - 增量拆分启用: {manager_no_split.enable_incremental_split}")
        
        # 演示正确的使用方式
        print(f"\n🧪 正确使用 get_subset 方法:")
        for task_id in range(manager_no_split.nb_tasks):
            # 使用 get_subset (NOT get_incremental_subset)
            train_set = manager_no_split.get_subset(
                task=task_id, source="train", cumulative=False, mode="train")
            test_set = manager_no_split.get_subset(
                task=task_id, source="test", cumulative=True, mode="test")
            
            print(f"   任务 {task_id}:")
            print(f"     训练集: {len(train_set)} 样本")
            print(f"     测试集: {len(test_set)} 样本")
            
            # 获取数据集信息
            dataset_info = manager_no_split.datasets[task_id]
            print(f"     数据集: {dataset_info['name']}")
        
        print(f"   ✅ 使用 get_subset 成功")
        
    except Exception as e:
        print(f"   ❌ 错误: {e}")
    
    # 场景2: enable_incremental_split = True
    print(f"\n📋 场景2: enable_incremental_split = True")
    print("   - 每个原始数据集会被拆分为多个子任务")
    print("   - 总任务数 > 数据集数量")
    print("   - 推荐使用: get_incremental_subset 方法")
    print("   - 也可以使用: get_subset 方法")
    
    try:
        manager_with_split = create_balanced_data_manager(
            dataset_names=datasets[:1],  # 只用一个数据集演示
            balanced_datasets_root="balanced_datasets",
            use_balanced_datasets=True,
            enable_incremental_split=True,  # 启用增量拆分
            num_incremental_splits=3
        )
        
        print(f"\n✅ 成功创建数据管理器:")
        print(f"   - 总任务数: {manager_with_split.nb_tasks} (大于数据集数量)")
        print(f"   - 总类别数: {manager_with_split.num_classes}")
        print(f"   - 增量拆分启用: {manager_with_split.enable_incremental_split}")
        
        # 演示两种方法都可以使用
        print(f"\n🧪 方法1: 使用 get_incremental_subset (推荐)")
        for task_id in range(min(2, manager_with_split.nb_tasks)):
            try:
                # 推荐使用 get_incremental_subset
                train_set = manager_with_split.get_incremental_subset(
                    task=task_id, source="train", cumulative=False, mode="train")
                test_set = manager_with_split.get_incremental_subset(
                    task=task_id, source="test", cumulative=True, mode="test")
                
                print(f"   任务 {task_id}:")
                print(f"     训练集: {len(train_set)} 样本")
                print(f"     测试集: {len(test_set)} 样本")
                
            except Exception as e:
                print(f"     ❌ 任务 {task_id} 失败: {e}")
        
        print(f"\n🧪 方法2: 使用 get_subset (也支持)")
        for task_id in range(min(2, manager_with_split.nb_tasks)):
            try:
                # 也可以使用 get_subset
                train_set = manager_with_split.get_subset(
                    task=task_id, source="train", cumulative=False, mode="train")
                test_set = manager_with_split.get_subset(
                    task=task_id, source="test", cumulative=True, mode="test")
                
                print(f"   任务 {task_id}:")
                print(f"     训练集: {len(train_set)} 样本")
                print(f"     测试集: {len(test_set)} 样本")
                
            except Exception as e:
                print(f"     ❌ 任务 {task_id} 失败: {e}")
        
        print(f"   ✅ 两种方法都可以正常工作")
        
    except Exception as e:
        print(f"   ❌ 错误: {e}")

def show_error_case():
    """
    展示错误使用的情况
    """
    print(f"\n❌ 错误使用示例:")
    
    # 创建不启用增量拆分的管理器
    manager = create_balanced_data_manager(
        dataset_names=['cifar100_224'],
        balanced_datasets_root="balanced_datasets",
        use_balanced_datasets=True,
        enable_incremental_split=False
    )
    
    print("当 enable_incremental_split=False 时，使用 get_incremental_subset 会出错:")
    
    try:
        # 错误：在不启用增量拆分的情况下使用 get_incremental_subset
        train_set = manager.get_incremental_subset(
            task=0, source="train", cumulative=False)
        print("   ✅ 意外成功")
    except ValueError as e:
        print(f"   ❌ 预期的错误: {e}")
        print(f"   解决方案: 使用 get_subset 代替")

def create_unified_interface_example():
    """
    创建统一接口的示例
    """
    print(f"\n=== 统一接口示例 ===\n")
    
    def get_data_subset(manager, task_id, source="train", cumulative=False, mode=None):
        """
        统一的数据获取接口，根据管理器类型自动选择正确的方法
        """
        if manager.enable_incremental_split:
            # 启用增量拆分时，使用 get_incremental_subset
            return manager.get_incremental_subset(
                task=task_id, source=source, cumulative=cumulative, mode=mode)
        else:
            # 未启用增量拆分时，使用 get_subset
            return manager.get_subset(
                task=task_id, source=source, cumulative=cumulative, mode=mode)
    
    print("统一接口使用示例:")
    
    # 测试场景1: enable_incremental_split = False
    manager1 = create_balanced_data_manager(
        dataset_names=['cifar100_224'],
        balanced_datasets_root="balanced_datasets",
        use_balanced_datasets=True,
        enable_incremental_split=False
    )
    
    print(f"\n场景1 (enable_incremental_split=False):")
    subset1 = get_data_subset(manager1, 0, source="test", cumulative=False)
    print(f"   数据集大小: {len(subset1)}")
    
    # 测试场景2: enable_incremental_split = True
    manager2 = create_balanced_data_manager(
        dataset_names=['cifar100_224'],
        balanced_datasets_root="balanced_datasets",
        use_balanced_datasets=True,
        enable_incremental_split=True,
        num_incremental_splits=2
    )
    
    print(f"\n场景2 (enable_incremental_split=True):")
    subset2 = get_data_subset(manager2, 0, source="test", cumulative=False)
    print(f"   数据集大小: {len(subset2)}")
    
    print(f"\n✅ 统一接口可以在两种场景下正常工作")

def main():
    """主函数"""
    demonstrate_correct_usage()
    show_error_case()
    create_unified_interface_example()
    
    print(f"\n=== 总结 ===")
    print(f"📌 enable_incremental_split=False:")
    print(f"   - 使用 get_subset 方法")
    print(f"   - 每个原始数据集对应一个任务")
    print(f"")
    print(f"📌 enable_incremental_split=True:")
    print(f"   - 推荐使用 get_incremental_subset 方法 (语义更明确)")
    print(f"   - 也可以使用 get_subset 方法")
    print(f"   - 每个原始数据集被拆分为多个子任务")
    print(f"")
    print(f"🎯 最佳实践: 根据配置动态选择方法，或使用统一包装函数")

if __name__ == "__main__":
    main()