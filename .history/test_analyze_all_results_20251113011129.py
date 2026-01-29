#!/usr/bin/env python3
"""
测试 analyze_all_results 函数的正确性
"""

import sys
import os
import logging

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trainer import analyze_all_results

def test_analyze_all_results():
    """测试 analyze_all_results 函数"""
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(filename)s] => %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    # 创建模拟的 all_results 数据
    mock_all_results = {
        "seed_1993": {
            "last_task_id": 2,
            "last_task_accuracies": {
                "lda": 82.1,
                "qda": 84.7
            },
            "average_accuracies": {
                "lda": 75.3,
                "qda": 78.17
            },
            "per_task_results": {
                0: {"lda": 75.5, "qda": 78.2},
                1: {"lda": 68.3, "qda": 71.6},
                2: {"lda": 82.1, "qda": 84.7}
            },
            "log_path": "/path/to/seed_1993/logs"
        },
        "seed_1996": {
            "last_task_id": 2,
            "last_task_accuracies": {
                "lda": 81.8,
                "qda": 84.2
            },
            "average_accuracies": {
                "lda": 74.9,
                "qda": 77.8
            },
            "per_task_results": {
                0: {"lda": 75.2, "qda": 77.9},
                1: {"lda": 67.8, "qda": 71.2},
                2: {"lda": 81.8, "qda": 84.2}
            },
            "log_path": "/path/to/seed_1996/logs"
        },
        "seed_1997": {
            "last_task_id": 2,
            "last_task_accuracies": {
                "lda": 82.5,
                "qda": 85.1
            },
            "average_accuracies": {
                "lda": 75.7,
                "qda": 78.5
            },
            "per_task_results": {
                0: {"lda": 75.8, "qda": 78.5},
                1: {"lda": 68.7, "qda": 71.9},
