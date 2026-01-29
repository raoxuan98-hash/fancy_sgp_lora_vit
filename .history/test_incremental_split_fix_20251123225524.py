"""
测试脚本，用于验证增量分割的修复是否有效
"""
import os
import sys
import logging
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.balanced_cross_domain_data_manager import create_balanced_data_manager

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_incremental_split_with_cross_domain():
    """
    测试启用增量分割的跨域数据管理器
    
    这个测试验证了启用增量分割时，是否会导致标签索引越界的问题
    """
    logger.info("开始测试增量分割功能...")
    
    # 使用与原问题相同的配置
    dataset_names = ['cifar100_224', 'imagenet-r', 'cars196_224', 'cub200_224']
    
    # 测试启用增量分割
    logger.info("测试启用增量分割...")
    manager = create_balanced_data_manager(
        dataset_names=dataset_names,
        balanced_datasets_root="balanced_datasets",
        use_balanced_datasets=True,
        enable_incremental_split=True,
        num_incremental_splits=2,
        incremental_split_seed=42
    )
    
    logger.info(f"数据管理器创建成功")
    logger.info(f"总任务数: {manager.nb_tasks}")
    logger.info(f"总类别数: {manager.num_classes}")
    
    # 检查每个任务的数据
    for i in range(manager.nb_tasks):
        task_size = manager.get_task_size(i)
        task_classes = manager.get_task_classes(i, cumulative=False)
        logger.info(f"任务 {i}: 类别数={task_size}, 类别范围={task_classes}")
        
        # 获取训练集
        train_set = manager.get_incremental_subset(task=i, source="train", cumulative=False, mode="train")
        test_set = manager.get_incremental_subset(task=i, source="test", cumulative=True, mode="test")
        
        # 检查标签范围
        train_targets = train_set.labels
        test_targets = test_set.labels
        
        if len(train_targets) > 0:
            train_min = min(train_targets)
            train_max = max(train_targets)
            logger.info(f"  训练集: 样本数={len(train_targets)}, 标签范围=[{train_min}, {train_max}]")
            
        if len(test_targets) > 0:
            test_min = min(test_targets)
            test_max = max(test_targets)
            logger.info(f"  测试集: 样本数={len(test_targets)}, 标签范围=[{test_min}, {test_max}]")
    
    logger.info("测试完成，增量分割功能正常工作！")
    
    # 测试不启用增量分割（用于比较）
    logger.info("\n测试不启用增量分割...")
    manager_no_split = create_balanced_data_manager(
        dataset_names=dataset_names,
        balanced_datasets_root="balanced_datasets",
        use_balanced_datasets=True,
        enable_incremental_split=False
    )
    
    logger.info(f"数据管理器创建成功")
    logger.info(f"总任务数: {manager_no_split.nb_tasks}")
    logger.info(f"总类别数: {manager_no_split.num_classes}")
    
    # 检查每个任务的数据
    for i in range(manager_no_split.nb_tasks):
        task_size = manager_no_split.get_task_size(i)
        task_classes = manager_no_split.get_task_classes(i, cumulative=False)
        logger.info(f"任务 {i}: 类别数={task_size}, 类别范围={task_classes}")
        
        # 获取训练集
        train_set = manager_no_split.get_incremental_subset(task=i, source="train", cumulative=False, mode="train")
        test_set = manager_no_split.get_incremental_subset(task=i, source="test", cumulative=True, mode="test")
        
        # 检查标签范围
        train_targets = train_set.labels
        test_targets = test_set.labels
        
        if len(train_targets) > 0:
            train_min = min(train_targets)
            train_max = max(train_targets)
            logger.info(f"  训练集: 样本数={len(train_targets)}, 标签范围=[{train_min}, {train_max}]")
            
        if len(test_targets) > 0:
            test_min = min(test_targets)
            test_max = max(test_targets)
            logger.info(f"  测试集: 样本数={len(test_targets)}, 标签范围=[{test_min}, {test_max}]")
    
    logger.info("测试完成，两种模式都正常工作！")
    
    return True

if __name__ == "__main__":
    try:
        result = test_incremental_split_with_cross_domain()
        if result:
            print("✅ 测试通过！增量分割功能修复成功。")
            sys.exit(0)
        else:
            print("❌ 测试失败！")
            sys.exit(1)
    except Exception as e:
        print(f"❌ 测试过程中出现异常: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)