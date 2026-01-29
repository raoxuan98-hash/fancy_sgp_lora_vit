#!/usr/bin/env python3
"""
平衡数据集使用示例
演示如何使用重新划分后的平衡数据集进行实验
"""

import os
import sys
import logging

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset_resplitter import DatasetResplitter
from utils.balanced_cross_domain_data_manager import create_balanced_data_manager

def example_resplit_datasets():
    """示例：重新划分数据集"""
    print("=" * 60)
    print("示例1：重新划分数据集")
    print("=" * 60)
    
    # 选择几个小数据集进行演示
    demo_datasets = ['dtd', 'mnist', 'cifar100_224']
    
    # 创建重新划分器
    resplitter = DatasetResplitter(
        max_samples_per_class=64,  # 使用64个样本进行快速演示
        seed=42,
        output_dir="example_balanced_datasets"
    )
    
    # 处理数据集
    results = resplitter.resplit_all_datasets(demo_datasets)
    
    print("数据集重新划分完成！")
    for dataset_name, result in results.items():
        if 'error' in result:
            print(f"❌ {dataset_name}: {result['error']}")
        else:
            print(f"✅ {dataset_name}: 处理成功")
    
    return True

def example_use_balanced_manager():
    """示例：使用平衡数据管理器"""
    print("\n" + "=" * 60)
    print("示例2：使用平衡数据管理器")
    print("=" * 60)
    
    # 使用刚刚创建的平衡数据集
    demo_datasets = ['dtd', 'mnist', 'cifar100_224']
    
    # 创建平衡数据管理器
    manager = create_balanced_data_manager(
        dataset_names=demo_datasets,
        balanced_datasets_root="example_balanced_datasets",
        use_balanced_datasets=True,
        log_level=logging.WARNING
    )
    
    print(f"✅ 平衡数据管理器创建成功")
    print(f"   总任务数: {manager.nb_tasks}")
    print(f"   总类别数: {manager.num_classes}")
    
    # 获取统计信息
    stats = manager.get_balanced_statistics()
    print(f"\n📊 平衡后统计信息:")
    for dataset_name, stat in stats.items():
        print(f"   {dataset_name}:")
        print(f"     训练样本: {stat['total_train_samples']}")
        print(f"     测试样本: {stat['total_test_samples']}")
        print(f"     训练每类: min={stat['train_per_class']['min']}, "
              f"max={stat['train_per_class']['max']}")
        print(f"     测试每类: min={stat['test_per_class']['min']}, "
              f"max={stat['test_per_class']['max']}")
    
    return True

def example_data_loading():
    """示例：数据加载和使用"""
    print("\n" + "=" * 60)
    print("示例3：数据加载和使用")
    print("=" * 60)
    
    demo_datasets = ['dtd', 'mnist']
    
    # 创建平衡数据管理器
    manager = create_balanced_data_manager(
        dataset_names=demo_datasets,
        balanced_datasets_root="example_balanced_datasets",
        use_balanced_datasets=True,
        log_level=logging.WARNING
    )
    
    # 演示数据加载
    for task_id in range(manager.nb_tasks):
        dataset_name = manager.datasets[task_id]['name']
        print(f"\n🔍 加载数据集 {dataset_name} (任务 {task_id}):")
        
        try:
            # 加载训练集
            train_dataset = manager.get_subset(task_id, source="train", mode="train")
            train_length = len(train_dataset) if hasattr(train_dataset, '__len__') else 'unknown'
            print(f"   ✅ 训练集加载成功: {train_length} 个样本")
            
            # 加载测试集
            test_dataset = manager.get_subset(task_id, source="test", mode="test")
            test_length = len(test_dataset) if hasattr(test_dataset, '__len__') else 'unknown'
            print(f"   ✅ 测试集加载成功: {test_length} 个样本")
            
            # 获取第一个样本查看
            try:
                train_sample_len = len(train_dataset)
                test_sample_len = len(test_dataset)
            except:
                train_sample_len = 0
                test_sample_len = 0
                
            if train_sample_len > 0 and test_sample_len > 0:
                train_sample, train_label, train_class_name = train_dataset[0]
                test_sample, test_label, test_class_name = test_dataset[0]
                
                print(f"   📷 训练样本形状: {train_sample.shape}")
                print(f"   🏷️  训练标签: {train_label}, 类名: {train_class_name}")
                print(f"   📷 测试样本形状: {test_sample.shape}")
                print(f"   🏷️  测试标签: {test_label}, 类名: {test_class_name}")
                
        except Exception as e:
            print(f"   ❌ 加载失败: {str(e)}")
            return False
    
    return True

def example_comparison():
    """示例：与原始数据集比较"""
    print("\n" + "=" * 60)
    print("示例4：与原始数据集比较")
    print("=" * 60)
    
    demo_datasets = ['dtd', 'mnist']
    
    # 创建平衡数据管理器
    balanced_manager = create_balanced_data_manager(
        dataset_names=demo_datasets,
        balanced_datasets_root="example_balanced_datasets",
        use_balanced_datasets=True,
        log_level=logging.WARNING
    )
    
    # 获取比较结果
    comparison = balanced_manager.compare_with_original()
    
    print("📊 原始 vs 平衡数据集比较:")
    for dataset_name, comp in comparison.items():
        print(f"\n🔍 {dataset_name}:")
        
        orig = comp['original']
        bal = comp['balanced']
        
        print(f"   训练样本: {orig['total_train_samples']} → {bal['total_train_samples']}")
        print(f"   测试样本: {orig['total_test_samples']} → {bal['total_test_samples']}")
        
        print(f"   训练每类范围: {orig['train_per_class_stats']['min']}-{orig['train_per_class_stats']['max']} "
              f"→ {bal['train_per_class_stats']['min']}-{bal['train_per_class_stats']['max']}")
        
        print(f"   测试每类范围: {orig['test_per_class_stats']['min']}-{orig['test_per_class_stats']['max']} "
              f"→ {bal['test_per_class_stats']['min']}-{bal['test_per_class_stats']['max']}")
        
        # 计算改善程度
        if orig['test_per_class_stats']['min'] > 0:
            orig_imbalance = orig['test_per_class_stats']['max'] / orig['test_per_class_stats']['min']
            bal_imbalance = bal['test_per_class_stats']['max'] / bal['test_per_class_stats']['min']
            
            print(f"   测试集不平衡比率: {orig_imbalance:.2f}x → {bal_imbalance:.2f}x")
            
            if bal_imbalance < orig_imbalance:
                print(f"   ✅ 不平衡性改善了 {(orig_imbalance/bal_imbalance):.2f}x")
            else:
                print(f"   ⚠️  不平衡性未改善")
    
    return True

def main():
    """主函数"""
    print("🚀 平衡数据集使用示例")
    
    # 设置日志级别
    logging.basicConfig(level=logging.WARNING)
    
    examples = [
        ("重新划分数据集", example_resplit_datasets),
        ("使用平衡数据管理器", example_use_balanced_manager),
        ("数据加载和使用", example_data_loading),
        ("与原始数据集比较", example_comparison)
    ]
    
    results = []
    for example_name, example_func in examples:
        try:
            result = example_func()
            results.append((example_name, result))
        except Exception as e:
            print(f"❌ 示例 '{example_name}' 出现异常: {str(e)}")
            results.append((example_name, False))
    
    # 输出结果摘要
    print("\n" + "=" * 60)
    print("示例运行结果摘要")
    print("=" * 60)
    
    passed = 0
    for example_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{example_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{len(results)} 个示例通过")
    
    if passed == len(results):
        print("🎉 所有示例运行成功！")
        print("\n💡 提示：")
        print("   - 平衡数据集已保存在 example_balanced_datasets/")
        print("   - 元数据保存在 example_balanced_datasets/metadata/")
        print("   - 可以在实验中使用 BalancedCrossDomainDataManagerCore")
        print("   - 查看 README_balanced_datasets.md 获取详细文档")
        return True
    else:
        print("⚠️  部分示例运行失败，请检查上述错误信息。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)