"""
SGP-LoRA-VIT 实验结果收集和存储系统

该模块提供了统一的实验结果管理功能，包括：
1. 标准化的实验结果存储结构
2. 实验配置和元数据管理
3. 跨域和域内实验结果管理
4. 多种子统计分析
5. 数据迁移工具
"""

from .core.config import ExperimentConfig
from .core.storage import ExperimentStorage
from .core.collector import ResultCollector
from .core.processor import ResultProcessor
from .core.registry import ExperimentRegistry

__version__ = "1.0.0"
__all__ = [
    "ExperimentConfig",
    "ExperimentStorage", 
    "ResultCollector",
    "ResultProcessor",
    "ExperimentRegistry"
]