#!/usr/bin/env python3
"""
测试 BalancedCrossDomainDataManagerCore 的所有功能

这个测试脚本包含了以下测试：
1. 基本初始化测试
2. 平衡数据集加载测试
3. 增量拆分功能测试
4. 统计信息功能测试
5. 与原始数据集比较测试
6. 错误处理测试
7. 小样本学习功能测试
"""

import os
import sys
import logging
import numpy as np
from pathlib import Path
import tempfile
import shutil
import json
from typing import List, Dict

# 添加项目根目录到路径
sys.path.append('/home/raoxuan/projects/fancy_sgp_lora_vit')

# 导入要测试的模块
from utils.balanced_cross_domain_data_manager import (
    BalancedCrossDomainDataManagerCore,
    create_balanced_data_manager
)


class TestBalancedCrossDomainDataManager:
    """测试 BalancedCrossDomainDataManagerCore 的所有功能"""
    
    def __init__(self):
        self.test_dir = Path("test_balanced_cross_domain_data_manager")
        self.temp_dir = None
        
    def setup(self):
        """设置测试环境"""
        print("🔧 设置测试环境...")
        
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp(prefix="bcdm_test_")
        self.test_balanced_dir = Path(self.temp_dir) / "balanced_datasets"
        self.test_balanced_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建测试数据集
        self._create_test_balanced_dataset()
        
    def teardown(self):
        """清理测试环境"""
        print("🧹 清理测试环境...")
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def _create_test_balanced_dataset(self):
        """创建一个简单的测试平衡数据集"""
        print("📦 创建测试平衡数据集...")
        
        # 创建 CIFAR-100 测试数据集
        test_dataset_dir = self.test_balanced_dir / "cifar100_224"
        test_dataset_dir.mkdir(exist_ok=True)
        
        # 创建标签文件
        label_file = test_dataset_dir / "label.txt"
        class_names = [f"class_{i}" for i in range(10)]  # 10个测试类
        with open(label_file, 'w') as f:
            f.write('\n'.join(class_names))
        
        # 创建训练和测试目录
        train_dir = test_dataset_dir / "train"
        test_dir = test_dataset_dir / "test"
        train_dir.mkdir(exist_ok=True)
        test_dir.mkdir(exist_ok=True)
        
        # 创建每个类别的目录和文件
        for class_id in range(10):
            train_class_dir = train_dir / str(class_id)
            test_class_dir = test_dir / str(class_id)
            train_class_dir.mkdir(exist_ok=True)
            test_class_dir.mkdir(exist_ok=True)
            
            # 创建训练文件（每个类5个样本）
            for i in range(5):
                train_file = train_class_dir / f"train_{class_id}_{i}.txt"
                train_file.write_text(f"train_data_{class_id}_{i}")
            
            # 创建测试文件（每个类3个样本）
            for i in range(3):
                test_file = test_class_dir / f"test_{class_id}_{i}.txt"
                test_file.write_text(f"test_data_{class_id}_{i}")
    
    def test_basic_initialization(self):
        """测试基本初始化功能"""
        print("\n🔍 测试 1: 基本初始化")
        
        try:
            # 测试创建基本数据管理器
            manager = BalancedCrossDomainDataManagerCore(
                dataset_names=["cifar100_224"],
                balanced_datasets_root=str(self.test_balanced_dir),
                use_balanced_datasets=True,
                log_level=logging.ERROR  # 减少输出
            )
            
            # 验证基本属性
            assert manager.nb_tasks == 1, f"任务数应为1，实际为{manager.nb_tasks}"
            assert manager.total_classes == 10, f"总类别数应为10，实际为{manager.total_classes}"
            assert len(manager.datasets) == 1, f"数据集数应为1，实际为{len(manager.datasets)}"
            
            print("✅ 基本初始化测试通过")
            return True
            
        except Exception as e:
            print(f"❌ 基本初始化测试失败: {str(e)}")
            return False
    
    def test_balance_dataset_loading(self):
        """测试平衡数据集加载"""
        print("\n🔍 测试 2: 平衡数据集加载")
        
        try:
            manager = BalancedCrossDomainDataManagerCore(
                dataset_names=["cifar100_224"],
                balanced_datasets_root=str(self.test_balanced_dir),
                use_balanced_datasets=True,
                log_level=logging.ERROR
            )
            
            dataset = manager.datasets[0]
            
            # 验证数据集信息
            assert dataset['name'] == "cifar100_224", f"数据集名称错误"
            assert dataset['num_classes'] == 10, f"类别数错误"
            assert len(dataset['train_data']) == 50, f"训练样本数错误：{len(dataset['train_data'])}"
            assert len(dataset['test_data']) == 30, f"测试样本数错误：{len(dataset['test_data'])}"
            assert len(dataset['class_names']) == 10, f"类名数错误"
            
            # 验证标签范围
            train_targets = dataset['train_targets']
            test_targets = dataset['test_targets']
            assert np.min(train_targets) == 0, f"训练标签最小值错误"
            assert np.max(train_targets) == 9, f"训练标签最大值错误"
            assert np.min(test_targets) == 0, f"测试标签最小值错误"
            assert np.max(test_targets) == 9, f"测试标签最大值错误"
            
            print("✅ 平衡数据集加载测试通过")
            return True
            
        except Exception as e:
            print(f"❌ 平衡数据集加载测试失败: {str(e)}")
            return False
    
    def test_few_shot_sampling(self):
        """测试小样本学习功能"""
        print("\n🔍 测试 3: 小样本学习")
        
        try:
            manager = BalancedCrossDomainDataManagerCore(
                dataset_names=["cifar100_224"],
                balanced_datasets_root=str(self.test_balanced_dir),
                use_balanced_datasets=True,
                num_shots=2,  # 每类2个样本
                log_level=logging.ERROR
            )
            
            dataset = manager.datasets[0]
            
            # 验证小样本采样结果
            expected_samples = 10 * 2  # 10类，每类2个样本
            actual_samples = len(dataset['train_data'])
            
            assert actual_samples == expected_samples, f"小样本采样后样本数错误：期望{expected_samples}，实际{actual_samples}"
            
            print("✅ 小样本学习测试通过")
            return True
            
        except Exception as e:
            print(f"❌ 小样本学习测试失败: {str(e)}")
            return False
    
    def test_incremental_splits(self):
        """测试增量拆分功能"""
        print("\n🔍 测试 4: 增量拆分")
        
        try:
            manager = BalancedCrossDomainDataManagerCore(
                dataset_names=["cifar100_224"],
                balanced_datasets_root=str(self.test_balanced_dir),
                use_balanced_datasets=True,
                enable_incremental_split=True,
                num_incremental_splits=3,
                incremental_split_seed=42,
                log_level=logging.ERROR
            )
            
            # 验证增量拆分结果
            assert manager.nb_tasks == 3, f"增量拆分后任务数应为3，实际为{manager.nb_tasks}"
            assert manager.total_classes == 10, f"总类别数仍应为10，实际为{manager.total_classes}"
            
            # 验证每个拆分的类别数
            for i in range(3):
                task_classes = manager.get_task_size(i)
                assert task_classes > 0, f"任务{i}的类别数应为正数，实际为{task_classes}"
            
            # 验证增量子集获取
            subset = manager.get_incremental_subset(0, "train")
            assert len(subset) > 0, "增量子集应不为空"
            
            print("✅ 增量拆分测试通过")
            return True
            
        except Exception as e:
            print(f"❌ 增量拆分测试失败: {str(e)}")
            return False
    
    def test_statistics(self):
        """测试统计信息功能"""
        print("\n🔍 测试 5: 统计信息")
        
        try:
            manager = BalancedCrossDomainDataManagerCore(
                dataset_names=["cifar100_224"],
                balanced_datasets_root=str(self.test_balanced_dir),
                use_balanced_datasets=True,
                log_level=logging.ERROR
            )
            
            # 测试基本统计信息
            stats = manager.get_balanced_statistics()
            
            assert "cifar100_224" in stats, "统计信息应包含数据集名称"
            cifar_stats = stats["cifar100_224"]
            
            # 验证统计指标
            assert cifar_stats['num_classes'] == 10, f"类别数统计错误"
            assert cifar_stats['total_train_samples'] == 50, f"训练样本总数统计错误"
            assert cifar_stats['total_test_samples'] == 30, f"测试样本总数统计错误"
            
            # 验证每类统计
            assert 'train_per_class' in cifar_stats, "应包含每类训练统计"
            assert 'test_per_class' in cifar_stats, "应包含每类测试统计"
            
            # 测试增量统计信息（启用增量拆分）
            manager_inc = BalancedCrossDomainDataManagerCore(
                dataset_names=["cifar100_224"],
                balanced_datasets_root=str(self.test_balanced_dir),
                use_balanced_datasets=True,
                enable_incremental_split=True,
                num_incremental_splits=2,
                log_level=logging.ERROR
            )
            
            inc_stats = manager_inc.get_incremental_statistics()
            assert "cifar100_224" in inc_stats, "增量统计信息应包含数据集名称"
            
            print("✅ 统计信息测试通过")
            return True
            
        except Exception as e:
            print(f"❌ 统计信息测试失败: {str(e)}")
            return False
    
    def test_subset_creation(self):
        """测试子集创建功能"""
        print("\n🔍 测试 6: 子集创建")
        
        try:
            manager = BalancedCrossDomainDataManagerCore(
                dataset_names=["cifar100_224"],
                balanced_datasets_root=str(self.test_balanced_dir),
                use_balanced_datasets=True,
                log_level=logging.ERROR
            )
            
            # 测试训练子集
            train_subset = manager.get_subset(0, "train")
            assert len(train_subset) == 50, f"训练子集大小错误：{len(train_subset)}"
            
            # 测试测试子集
            test_subset = manager.get_subset(0, "test")
            assert len(test_subset) == 30, f"测试子集大小错误：{len(test_subset)}"
            
            # 测试累积模式
            cumulative_subset = manager.get_subset(0, "train", cumulative=True)
            assert len(cumulative_subset) == 50, f"累积子集大小错误：{len(cumulative_subset)}"
            
            # 测试数据加载
            sample = train_subset[0]
            assert len(sample) == 3, f"样本应有3个元素（图像、标签、类名），实际{len(sample)}"
            
            print("✅ 子集创建测试通过")
            return True
            
        except Exception as e:
            print(f"❌ 子集创建测试失败: {str(e)}")
            return False
    
    def test_helper_functions(self):
        """测试辅助函数"""
        print("\n🔍 测试 7: 辅助函数")
        
        try:
            manager = BalancedCrossDomainDataManagerCore(
                dataset_names=["cifar100_224"],
                balanced_datasets_root=str(self.test_balanced_dir),
                use_balanced_datasets=True,
                enable_incremental_split=True,
                num_incremental_splits=2,
                log_level=logging.ERROR
            )
            
            # 测试任务类获取
            task_classes = manager.get_task_classes(0, cumulative=False)
            assert len(task_classes) > 0, "任务类别列表不应为空"
            
            cumulative_classes = manager.get_task_classes(0, cumulative=True)
            assert len(cumulative_classes) >= len(task_classes), "累积类别数应不小于单任务类别数"
            
            # 测试原始数据集拆分获取
            original_splits = manager.get_original_dataset_splits("cifar100_224")
            assert len(original_splits) == 2, f"原始数据集拆分数应为2，实际{len(original_splits)}"
            
            # 测试工厂函数
            factory_manager = create_balanced_data_manager(
                dataset_names=["cifar100_224"],
                balanced_datasets_root=str(self.test_balanced_dir),
                log_level=logging.ERROR
            )
            assert factory_manager.nb_tasks == 1, "工厂函数创建的管理器任务数错误"
            
            print("✅ 辅助函数测试通过")
            return True
            
        except Exception as e:
            print(f"❌ 辅助函数测试失败: {str(e)}")
            return False
    
    def test_error_handling(self):
        """测试错误处理"""
        print("\n🔍 测试 8: 错误处理")
        
        try:
            # 测试不存在的平衡数据集
            try:
                manager = BalancedCrossDomainDataManagerCore(
                    dataset_names=["nonexistent_dataset"],
                    balanced_datasets_root=str(self.test_balanced_dir),
                    use_balanced_datasets=True,
                    log_level=logging.ERROR
                )
                # 应该能够创建，但不包含任何数据集
                assert manager.nb_tasks == 0, "不存在的数据集应该创建空管理器"
                print("✅ 不存在数据集的错误处理正确")
            except Exception:
                print("❌ 不存在数据集的错误处理失败")
                return False
            
            # 测试无效的增量拆分参数
            try:
                manager = BalancedCrossDomainDataManagerCore(
                    dataset_names=["cifar100_224"],
                    balanced_datasets_root=str(self.test_balanced_dir),
                    use_balanced_datasets=True,
                    enable_incremental_split=True,
                    num_incremental_splits=0,  # 无效参数
                    log_level=logging.ERROR
                )
                # 应该正常工作（禁用增量拆分）
                print("✅ 无效增量拆分参数处理正确")
            except Exception as e:
                print(f"❌ 无效增量拆分参数处理失败: {str(e)}")
                return False
            
            print("✅ 错误处理测试通过")
            return True
            
        except Exception as e:
            print(f"❌ 错误处理测试失败: {str(e)}")
            return False
    
    def test_integration(self):
        """集成测试：完整工作流程"""
        print("\n🔍 测试 9: 集成测试")
        
        try:
            # 创建完整的数据管理器
            manager = create_balanced_data_manager(
                dataset_names=["cifar100_224"],
                balanced_datasets_root=str(self.test_balanced_dir),
                use_balanced_datasets=True,
                enable_incremental_split=True,
                num_incremental_splits=2,
                num_shots=3,
                incremental_split_seed=42,
                log_level=logging.ERROR
            )
            
            # 验证完整工作流程
            assert manager.nb_tasks > 0, "任务数应大于0"
            
            # 获取统计信息
            stats = manager.get_balanced_statistics()
            assert len(stats) > 0, "统计信息应不为空"
            
            # 获取增量统计信息
            inc_stats = manager.get_incremental_statistics()
            assert len(inc_stats) > 0, "增量统计信息应不为空"
            
            # 测试所有任务的数据加载
            for task_id in range(manager.nb_tasks):
                train_subset = manager.get_subset(task_id, "train")
                test_subset = manager.get_subset(task_id, "test")
                
                assert len(train_subset) > 0, f"任务{task_id}的训练子集不应为空"
                assert len(test_subset) > 0, f"任务{task_id}的测试子集不应为空"
                
                # 测试样本
                sample = train_subset[0]
                assert len(sample) == 3, f"任务{task_id}样本格式错误"
            
            print("✅ 集成测试通过")
            return True
            
        except Exception as e:
            print(f"❌ 集成测试失败: {str(e)}")
            return False
    
    def run_all_tests(self):
        """运行所有测试"""
        print("🚀 开始测试 BalancedCrossDomainDataManagerCore")
        print("=" * 60)
        
        # 设置测试环境
        self.setup()
        
        # 测试列表
        tests = [
            ("基本初始化", self.test_basic_initialization),
            ("平衡数据集加载", self.test_balance_dataset_loading),
            ("小样本学习", self.test_few_shot_sampling),
            ("增量拆分", self.test_incremental_splits),
            ("统计信息", self.test_statistics),
            ("子集创建", self.test_subset_creation),
            ("辅助函数", self.test_helper_functions),
            ("错误处理", self.test_error_handling),
            ("集成测试", self.test_integration)
        ]
        
        # 执行测试
        passed = 0
        failed = 0
        
        for test_name, test_func in tests:
            try:
                if test_func():
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"❌ {test_name}测试异常: {str(e)}")
                failed += 1
        
        # 清理测试环境
        self.teardown()
        
        # 输出结果
        print("\n" + "=" * 60)
        print("🏁 测试完成!")
        print(f"✅ 通过: {passed}")
        print(f"❌ 失败: {failed}")
        print(f"📊 总计: {passed + failed}")
        
        if failed == 0:
            print("🎉 所有测试都通过了！")
            return True
        else:
            print("⚠️  有测试失败，请检查代码")
            return False


def main():
    """主函数"""
    tester = TestBalancedCrossDomainDataManager()
    success = tester.run_all_tests()
    
    # 根据测试结果设置退出代码
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()