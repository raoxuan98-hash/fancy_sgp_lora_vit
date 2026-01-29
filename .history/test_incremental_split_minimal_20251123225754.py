
"""
最小测试脚本，用于验证增量分割的修复是否有效
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

def test_incremental_split_minimal():
    """
    测试启用增量分割的基本功能
    """
    logger.info("开始测试增量分割功能...")
    
    # 使用与原问题相同的配置，但使用较小的数据集
    dataset_names = ['cifar100_224', 'imagenet-r']
    
    # 测试启用增量分割
    logger.info("测试启用增量分割...")
    try:
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
        
        # 检查每个任务的基本信息
        for i in range(manager.nb_tasks):
            task_size = manager.get_task_size(i)
            task_classes = manager.get_task_classes(i, cumulative=False)
            logger.info(f"任务 {i}: 类别数={task_size}, 类别范围={task_classes}")
            
            # 尝试访问task_classes以确保它们是有效的
            if task_classes and len(task_classes) > 0:
                logger.info(f"  类别范围检查: 最小={min(task_classes)}, 最大={max(task_classes)}")
        
        logger.info("✅ 测试启用增量分割成功！")
        
    except Exception as e:
        logger.error(f"❌ 测试启用增量分割失败: {e}")
        raise
    
    # 测试不启用增量分割（用于比较）
    logger.info("\n测试不启用增量分割...")
    try:
        manager_no_split = create_balanced_data_manager(
            dataset_names=dataset_names,
            balanced_datasets_root="balanced_datasets",
            use_balanced_datasets=True,
            enable_incremental_split=False
        )
        
        logger.info(f"数据管理器创建成功")
