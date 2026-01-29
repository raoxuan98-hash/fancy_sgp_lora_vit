#!/usr/bin/env python3
"""
在 subspace_lora.py 中使用修改后的 get_incremental_subset 方法的示例
"""

from utils.balanced_cross_domain_data_manager import create_balanced_data_manager

def demonstrate_usage():
    """
    演示如何在 subspace_lora.py 中使用 get_incremental_subset 方法
    """
    print("=== 演示 subspace_lora.py 中的使用方式 ===\n")
    
    # 创建平衡数据管理器（启用增量拆分）
    datasets = ['cifar100_224', 'cub200_224']
    
    manager = create_balanced_data_manager(
        dataset_names=datasets,
        balanced_datasets_root="balanced_datasets",
        use_balanced_datasets=True,
        enable_incremental_split=True,
        num_incremental_splits=3,
        incremental_split_seed=42
    )
    
    print(f"数据管理器创建成功:")
    print(f"  - 总任务数: {manager.nb_tasks}")
    print(f"  - 总类别数: {manager.num_classes}")
    print(f"  - 增量拆分启用: {manager.enable_incremental_split}")
    
    # 模拟 subspace_lora.py 中的训练循环
    print(f"\n🔄 模拟 subspace_lora.py 的训练流程:")
    
    for task_id in range(min(3, manager.nb_tasks)):  # 演示前3个任务
        print(f"\n--- 任务 {task_id} ---")
        
        # ======= 原来的方式 (在 subspace_lora.py 第229-234行) =======
        # train_set = data_manager.get_subset(
        #     task=task_id, source="train", cumulative=False, mode="train")
        # test_set = data_manager.get_subset(
        #     task=task_id, source="test", cumulative=True, mode="test")
        # train_set_test_mode = data_manager.get_subset(
        #     task=task_id, source="train", cumulative=False, mode="test")
        
        # ======= 新的方式 (使用 get_incremental_subset) =======
        train_set = manager.get_incremental_subset(
            task=task_id, source="train", cumulative=False, mode="train")
        test_set = manager.get_incremental_subset(
            task=task_id, source="test", cumulative=True, mode="test")
        train_set_test_mode = manager.get_incremental_subset(
            task=task_id, source="train", cumulative=False, mode="test")
        
        # 显示结果信息
        print(f"  训练集 (cumulative=False): {len(train_set)} 样本")
        print(f"  测试集 (cumulative=True): {len(test_set)} 样本")
        print(f"  训练集测试模式 (cumulative=False): {len(train_set_test_mode)} 样本")
        
        # 验证累积模式的正确性
        if task_id > 0:
            expected_cumulative_size = sum(len(manager.datasets[i]['test_data']) for i in range(task_id + 1))
            if len(test_set) == expected_cumulative_size:
                print(f"  ✅ 累积模式验证通过")
            else:
                print(f"  ❌ 累积模式验证失败: 期望 {expected_cumulative_size}, 实际 {len(test_set)}")
        
        # 模拟训练过程
        print(f"  🏋️  训练模型...")
        print(f"  🧪 评估模型...")
        
        # 获取任务信息
        dataset_info = manager.datasets[task_id]
        print(f"  📊 数据集: {dataset_info['name']}")
        print(f"  📚 类别数: {dataset_info['num_classes']}")
        
        if task_id == 0:
            print(f"  ℹ️  首个任务，仅使用当前任务数据")
        else:
            previous_classes = sum(manager.datasets[i]['num_classes'] for i in range(task_id))
            total_classes = previous_classes + dataset_info['num_classes']
            print(f"  ℹ️  累积 {previous_classes} + {dataset_info['num_classes']} = {total_classes} 个类别")

def show_migration_guide():
    """
    展示从 get_subset 迁移到 get_incremental_subset 的指南
    """
    print(f"\n=== 迁移指南：从 get_subset 到 get_incremental_subset ===\n")
    
    print("在 subspace_lora.py 中进行以下替换:")
    print()
    
    print("1. 在文件顶部添加或确认导入:")
    print("   from utils.balanced_cross_domain_data_manager import create_balanced_data_manager")
    print()
    
    print("2. 替换数据管理器的创建:")
    print("   # 原来的方式")
    print("   data_manager = CrossDomainDataManagerCore(dataset_names, ...)")
    print()
    print("   # 新的方式")
    print("   data_manager = create_balanced_data_manager(")
    print("       dataset_names=dataset_names,")
    print("       balanced_datasets_root='balanced_datasets',")
    print("       use_balanced_datasets=True,")
    print("       enable_incremental_split=True,  # 启用增量拆分")
    print("       num_incremental_splits=3,")
    print("       incremental_split_seed=42")
    print("   )")
    print()
    
    print("3. 替换 get_subset 调用为 get_incremental_subset:")
    print("   # 原来的方式")
    print("   train_set = data_manager.get_subset(")
    print("       task=task_id, source='train', cumulative=False, mode='train')")
    print("   test_set = data_manager.get_subset(")
    print("       task=task_id, source='test', cumulative=True, mode='test')")
    print()
    print("   # 新的方式 (推荐)")
    print("   train_set = data_manager.get_incremental_subset(")
    print("       task=task_id, source='train', cumulative=False, mode='train')")
    print("   test_set = data_manager.get_incremental_subset(")
    print("       task=task_id, source='test', cumulative=True, mode='test')")
    print()
    
    print("4. 好处:")
    print("   ✅ 语义更明确 - 明确表示这是增量学习场景")
    print("   ✅ 增强的类型安全性 - 专门为增量拆分设计")
    print("   ✅ 更好的错误处理 - 当未启用增量拆分时会抛出明确的错误")
    print("   ✅ 保持向后兼容 - 仍然支持原有的参数和行为")

def main():
    """主函数"""
    demonstrate_usage()
    show_migration_guide()
    
    print(f"\n🎉 演示完成！get_incremental_subset 方法现在完全支持 cumulative 参数")
    print(f"   可以无缝替换 subspace_lora.py 中的 get_subset 调用")

if __name__ == "__main__":
    main()