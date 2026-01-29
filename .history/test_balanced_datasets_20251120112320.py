#!/usr/bin/env python3
"""
测试平衡数据集的脚本
验证数据集重新划分和新的数据管理器是否正常工作
"""

import os
import sys
import logging
import json
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset_resplitter import DatasetResplitter
from utils.balanced_cross_domain_data_manager import create_balanced_data_manager
from utils.cross_domain_data_manager import CrossDomainDataManagerCore

def test_dataset_resplitter():
    """测试数据集重新划分器"""
    print("=" * 60)
    print("测试数据集重新划分器")
    print("=" * 60)
    
    # 只测试几个小数据集以节省时间
    test_datasets = ['dtd', 'mnist', 'cifar100_224']
    
    # 创建重新划分器
    resplitter = DatasetResplitter(
        max_samples_per_class=128,
        seed=42,
        output_dir="test_balanced_datasets"
    )
    
    # 处理测试数据集
    results = resplitter.resplit_all_datasets(test_datasets)
    
    # 检查结果
    for dataset_name, result in results.items():
        if 'error' in result:
            print(f"❌ {dataset_name}: {result['error']}")
        else:
            print(f"✅ {dataset_name}: 处理成功")
            if 'classes_with_insufficient_samples' in result:
                insufficient = result['classes_with_insufficient_samples']
                if insufficient:
                    print(f"   ⚠️  有 {len(insufficient)} 个类别样本不足128")
                else:
                    print(f"   ✅ 所有类别都有足够的样本")
    
    return True

def test_balanced_data_manager():
    """测试平衡数据管理器"""
    print("\n" + "=" * 60)
    print("测试平衡数据管理器")
    print("=" * 60)
    
    # 测试数据集
    test_datasets = ['dtd', 'mnist', 'cifar100_224']
    
    try:
        # 创建平衡数据管理器
        balanced_manager = create_balanced_data_manager(
            dataset_names=test_datasets,
            balanced_datasets_root="test_balanced_datasets",
            use_balanced_datasets=True,
            log_level=logging.WARNING
        )
        
        print(f"✅ 平衡数据管理器创建成功")
        print(f"   总任务数: {balanced_manager.nb_tasks}")
        print(f"   总类别数: {balanced_manager.num_classes}")
        
        # 测试获取数据集
        for task_id in range(balanced_manager.nb_tasks):
            dataset_info = balanced_manager.datasets[task_id]
            train_samples = len(dataset_info['train_data'])
            test_samples = len(dataset_info['test_data'])
            num_classes = dataset_info['num_classes']
            dataset_name = dataset_info['name']
            
            print(f"\n📊 数据集 {dataset_name} (任务 {task_id}):")
            print(f"   类别数: {num_classes}")
            print(f"   训练样本: {train_samples}")
            print(f"   测试样本: {test_samples}")
            print(f"   平均每类训练样本: {train_samples/num_classes:.2f}")
            print(f"   平均每类测试样本: {test_samples/num_classes:.2f}")
        
        # 获取统计信息
        stats = balanced_manager.get_balanced_statistics()
        print(f"\n📈 平衡后统计信息:")
        for dataset_name, stat in stats.items():
            print(f"   {dataset_name}:")
            print(f"     训练每类: min={stat['train_per_class']['min']}, "
                  f"max={stat['train_per_class']['max']}, "
                  f"mean={stat['train_per_class']['mean']:.2f}")
            print(f"     测试每类: min={stat['test_per_class']['min']}, "
                  f"max={stat['test_per_class']['max']}, "
                  f"mean={stat['test_per_class']['mean']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 平衡数据管理器测试失败: {str(e)}")
        return False

def test_comparison_with_original():
    """测试与原始数据集的比较"""
    print("\n" + "=" * 60)
    print("测试与原始数据集的比较")
    print("=" * 60)
    
    test_datasets = ['dtd', 'mnist', 'cifar100_224']
    
    try:
        # 创建平衡数据管理器
        balanced_manager = create_balanced_data_manager(
            dataset_names=test_datasets,
            balanced_datasets_root="test_balanced_datasets",
            use_balanced_datasets=True,
            log_level=logging.WARNING
        )
        
        # 创建原始数据管理器
        original_manager = CrossDomainDataManagerCore(
            dataset_names=test_datasets,
            shuffle=False,
            seed=0,
            num_shots=0,
            num_samples_per_task_for_evaluation=0,
            log_level=logging.WARNING
        )
        
        # 比较统计信息
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
            orig_imbalance = orig['test_per_class_stats']['max'] / orig['test_per_class_stats']['min'] if orig['test_per_class_stats']['min'] > 0 else float('inf')
            bal_imbalance = bal['test_per_class_stats']['max'] / bal['test_per_class_stats']['min'] if bal['test_per_class_stats']['min'] > 0 else float('inf')
            
            print(f"   测试集不平衡比率: {orig_imbalance:.2f}x → {bal_imbalance:.2f}x")
            
            if bal_imbalance < orig_imbalance:
                print(f"   ✅ 不平衡性改善了 {(orig_imbalance/bal_imbalance):.2f}x")
            else:
                print(f"   ⚠️  不平衡性未改善")
        
        return True
        
    except Exception as e:
        print(f"❌ 比较测试失败: {str(e)}")
        return False

def test_data_loading():
    """测试数据加载功能"""
    print("\n" + "=" * 60)
    print("测试数据加载功能")
    print("=" * 60)
    
    test_datasets = ['dtd', 'mnist']
    
    try:
        # 创建平衡数据管理器
        balanced_manager = create_balanced_data_manager(
            dataset_names=test_datasets,
            balanced_datasets_root="test_balanced_datasets",
            use_balanced_datasets=True,
            log_level=logging.WARNING
        )
        
        # 测试获取数据集
        for task_id in range(balanced_manager.nb_tasks):
            print(f"\n🔍 测试任务 {task_id} ({balanced_manager.datasets[task_id]['name']}):")
            
            # 测试训练集
            try:
                train_dataset = balanced_manager.get_subset(task_id, source="train", mode="train")
                print(f"   ✅ 训练集加载成功: {len(train_dataset)} 个样本")
                
                # 测试获取第一个样本
                if len(train_dataset) > 0:
                    sample, label, class_name = train_dataset[0]
                    print(f"   📷 样本形状: {sample.shape if hasattr(sample, 'shape') else type(sample)}")
                    print(f"   🏷️  标签: {label}, 类名: {class_name}")
                
            except Exception as e:
                print(f"   ❌ 训练集加载失败: {str(e)}")
                return False
            
            # 测试测试集
            try:
                test_dataset = balanced_manager.get_subset(task_id, source="test", mode="test")
                print(f"   ✅ 测试集加载成功: {len(test_dataset)} 个样本")
                
                # 测试获取第一个样本
                if len(test_dataset) > 0:
                    sample, label, class_name = test_dataset[0]
                    print(f"   📷 样本形状: {sample.shape if hasattr(sample, 'shape') else type(sample)}")
                    print(f"   🏷️  标签: {label}, 类名: {class_name}")
                
            except Exception as e:
                print(f"   ❌ 测试集加载失败: {str(e)}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ 数据加载测试失败: {str(e)}")
        return False

def check_metadata_files():
    """检查元数据文件是否正确生成"""
    print("\n" + "=" * 60)
    print("检查元数据文件")
    print("=" * 60)
    
    metadata_dir = Path("test_balanced_datasets/metadata")
    
    if not metadata_dir.exists():
        print("❌ 元数据目录不存在")
        return False
    
    required_files = [
        "original_distribution.json",
        "balanced_distribution.json", 
        "sampling_config.json",
        "dataset_statistics.json"
    ]
    
    all_exist = True
    for filename in required_files:
        file_path = metadata_dir / filename
        if file_path.exists():
            print(f"✅ {filename} 存在")
            
            # 检查文件内容
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                print(f"   📄 包含 {len(data)} 个条目")
            except Exception as e:
                print(f"   ⚠️  读取文件时出错: {str(e)}")
        else:
            print(f"❌ {filename} 不存在")
            all_exist = False
    
    return all_exist

def main():
    """主测试函数"""
    print("🧪 开始测试平衡数据集系统")
    
    # 设置日志级别
    logging.basicConfig(level=logging.WARNING)
    
    tests = [
        ("数据集重新划分", test_dataset_resplitter),
        ("平衡数据管理器", test_balanced_data_manager),
        ("与原始数据集比较", test_comparison_with_original),
        ("数据加载功能", test_data_loading),
        ("元数据文件检查", check_metadata_files)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ 测试 '{test_name}' 出现异常: {str(e)}")
            results.append((test_name, False))
    
    # 输出测试结果摘要
    print("\n" + "=" * 60)
    print("测试结果摘要")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{len(results)} 个测试通过")
    
    if passed == len(results):
        print("🎉 所有测试通过！平衡数据集系统工作正常。")
        return True
    else:
        print("⚠️  部分测试失败，请检查上述错误信息。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)