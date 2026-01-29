#!/usr/bin/env python3
"""
测试 analyze_all_results 函数的正确性
"""

import sys
import os
import logging
import json

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
                2: {"lda": 82.5, "qda": 85.1}
            },
            "log_path": "/path/to/seed_1997/logs"
        }
    }
    
    # 模拟数据集名称
    dataset_names = ["CIFAR-100", "CUB200", "Cars196"]
    
    # 测试函数
    print("=" * 80)
    print("🧪 开始测试 analyze_all_results 函数")
    print("=" * 80)
    
    try:
        # 测试保存JSON功能
        output_path = "./test_statistics.json"
        statistics_results = analyze_all_results(mock_all_results, dataset_names, save_json=True, output_path=output_path)
        
        print("\n🔍 检查返回的统计结果结构:")
        print(f"  - 包含summary: {'summary' in statistics_results}")
        print(f"  - 包含variants: {'variants' in statistics_results}")
        print(f"  - 包含overall_summary: {'overall_summary' in statistics_results}")
        
    """测试边界情况"""
    print("\n" + "=" * 80)
    print("🧪 开始测试边界情况")
    print("=" * 80)
    
    # 测试空字典
    print("\n📋 测试空字典:")
    analyze_all_results({})
    
    # 测试单个种子
    print("\n📋 测试单个种子:")
    single_seed = {
        "seed_1993": {
            "last_task_id": 0,
            "last_task_accuracies": {"lda": 82.1},
            "average_accuracies": {"lda": 75.3},
            "per_task_results": {0: {"lda": 75.5}},
            "log_path": "/path/to/seed_1993/logs"
        }
    }
    analyze_all_results(single_seed, ["CIFAR-100"])
    
    # 测试缺少某些字段的情况
    print("\n📋 测试缺少字段的情况:")
    incomplete_results = {
        "seed_1993": {
            "last_task_accuracies": {"lda": 82.1}
            # 缺少其他字段
        },
        "seed_1996": {
            "average_accuracies": {"qda": 77.8}
            # 缺少其他字段
        }
    }
    analyze_all_results(incomplete_results)
    
    print("\n✅ 边界情况测试完成！")

if __name__ == "__main__":
    success = test_analyze_all_results()
    test_edge_cases()
    
    if success:
        print("\n🎉 所有测试通过！")
        sys.exit(0)
    else:
        print("\n💥 部分测试失败！")
        sys.exit(1)