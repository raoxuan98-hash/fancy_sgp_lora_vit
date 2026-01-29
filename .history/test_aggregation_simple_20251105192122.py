#!/usr/bin/env python3
"""
简单测试脚本：验证多种子结果聚合逻辑是否正确工作
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path

def test_aggregation_logic():
    """测试聚合逻辑，不运行实际训练"""
    
    # 创建临时目录
    temp_dir = Path(tempfile.mkdtemp(prefix="test_aggregation_"))
    print(f"📁 创建临时测试目录: {temp_dir}")
    
    try:
        # 模拟多个种子的结果数据
        seed_results = {
            "seed_1993": {
                "last_task_accuracies": {
                    "SeqFT + LDA": 75.0,
                    "SeqFT + QDA": 80.0,
                    "SeqFT + attention_transform + LDA": 82.0,
                    "SeqFT + attention_transform + QDA": 85.0,
                },
                "average_accuracies": {
                    "SeqFT + LDA": 78.0,
                    "SeqFT + QDA": 83.0,
                    "SeqFT + attention_transform + LDA": 85.0,
                    "SeqFT + attention_transform + QDA": 88.0,
                },
                "per_task_results": {
                    0: {
                        "SeqFT + LDA": 70.0,
                        "SeqFT + QDA": 75.0,
                        "SeqFT + attention_transform + LDA": 77.0,
                        "SeqFT + attention_transform + QDA": 80.0,
                    },
                    1: {
                        "SeqFT + LDA": 75.0,
                        "SeqFT + QDA": 80.0,
                        "SeqFT + attention_transform + LDA": 82.0,
                        "SeqFT + attention_transform + QDA": 85.0,
                    },
                    2: {
                        "SeqFT + LDA": 80.0,
                        "SeqFT + QDA": 85.0,
                        "SeqFT + attention_transform + LDA": 87.0,
                        "SeqFT + attention_transform + QDA": 90.0,
                    }
                },
                "shared_log_dir": str(temp_dir)
            },
            "seed_1996": {
                "last_task_accuracies": {
                    "SeqFT + LDA": 74.0,
                    "SeqFT + QDA": 79.0,
                    "SeqFT + attention_transform + LDA": 81.0,
                    "SeqFT + attention_transform + QDA": 84.0,
                },
                "average_accuracies": {
                    "SeqFT + LDA": 77.0,
                    "SeqFT + QDA": 82.0,
                    "SeqFT + attention_transform + LDA": 84.0,
                    "SeqFT + attention_transform + QDA": 87.0,
                },
                "per_task_results": {
                    0: {
                        "SeqFT + LDA": 69.0,
                        "SeqFT + QDA": 74.0,
                        "SeqFT + attention_transform + LDA": 76.0,
                        "SeqFT + attention_transform + QDA": 79.0,
                    },
                    1: {
                        "SeqFT + LDA": 74.0,
                        "SeqFT + QDA": 79.0,
                        "SeqFT + attention_transform + LDA": 81.0,
                        "SeqFT + attention_transform + QDA": 84.0,
                    },
                    2: {
                        "SeqFT + LDA": 79.0,
                        "SeqFT + QDA": 84.0,
                        "SeqFT + attention_transform + LDA": 86.0,
                        "SeqFT + attention_transform + QDA": 89.0,
                    }
                },
                "shared_log_dir": str(temp_dir)
            },
            "seed_1997": {
                "last_task_accuracies": {
                    "SeqFT + LDA": 76.0,
                    "SeqFT + QDA": 81.0,
                    "SeqFT + attention_transform + LDA": 83.0,
                    "SeqFT + attention_transform + QDA": 86.0,
                },
                "average_accuracies": {
                    "SeqFT + LDA": 79.0,
                    "SeqFT + QDA": 84.0,
                    "SeqFT + attention_transform + LDA": 86.0,
                    "SeqFT + attention_transform + QDA": 89.0,
                },
                "per_task_results": {
                    0: {
                        "SeqFT + LDA": 71.0,
                        "SeqFT + QDA": 76.0,
                        "SeqFT + attention_transform + LDA": 78.0,
                        "SeqFT + attention_transform + QDA": 81.0,
                    },
                    1: {
                        "SeqFT + LDA": 76.0,
                        "SeqFT + QDA": 81.0,
                        "SeqFT + attention_transform + LDA": 83.0,
                        "SeqFT + attention_transform + QDA": 86.0,
                    },
                    2: {
                        "SeqFT + LDA": 81.0,
                        "SeqFT + QDA": 86.0,
                        "SeqFT + attention_transform + LDA": 88.0,
                        "SeqFT + attention_transform + QDA": 91.0,
                    }
                },
                "shared_log_dir": str(temp_dir)
            }
        }
        
        # 导入聚合函数
        from trainer import aggregate_seed_results
        
        print("🧪 开始测试聚合逻辑...")
        print(f"🌱 测试种子数量: {len(seed_results)}")
        
        # 运行聚合函数
        aggregated = aggregate_seed_results(seed_results)
        
        # 检查聚合结果
        assert 'final_task' in aggregated, "聚合结果中缺少'final_task'"
        assert 'average_across_tasks' in aggregated, "聚合结果中缺少'average_across_tasks'"
        assert 'per_task_accuracy_trends' in aggregated, "聚合结果中缺少'per_task_accuracy_trends'"
        
        # 检查聚合结果文件是否存在
        aggregate_file = temp_dir / "aggregate_results.json"
        assert aggregate_file.exists(), "聚合结果文件不存在"
        
        # 检查聚合结果文件内容
        with open(aggregate_file, 'r', encoding='utf-8') as f:
            aggregate_data = json.load(f)
        
        assert 'final_task_stats' in aggregate_data, "聚合文件中缺少'final_task_stats'"
        assert 'average_across_tasks_stats' in aggregate_data, "聚合文件中缺少'average_across_tasks_stats'"
        assert 'seed_list' in aggregate_data, "聚合文件中缺少'seed_list'"
        assert 'num_seeds' in aggregate_data, "聚合文件中缺少'num_seeds'"
        
        # 检查种子列表
        seed_list = aggregate_data['seed_list']
        assert len(seed_list) == len(seed_results), "聚合文件中的种子数量不匹配"
        
        # 检查标准差是否为0（如果是0，说明没有正确聚合多个种子）
        print("\n📊 聚合结果检查:")
        for variant, stats in aggregate_data['final_task_stats'].items():
            mean = stats['mean']
            std = stats['std']
            
            # 计算期望的平均值和标准差
            values = [seed_results[f"seed_{seed}"]["last_task_accuracies"][variant] for seed in [1993, 1996, 1997]]
            expected_mean = sum(values) / len(values)
            expected_std = (sum((x - expected_mean) ** 2 for x in values) / len(values)) ** 0.5
            
            print(f"  变体 {variant}:")
            print(f"    期望均值: {expected_mean:.2f}%, 实际均值: {mean:.2f}%")
            print(f"    期望标准差: {expected_std:.2f}%, 实际标准差: {std:.2f}%")
            
            assert abs(mean - expected_mean) < 0.01, f"均值计算错误: {mean} vs {expected_mean}"
            assert abs(std - expected_std) < 0.01, f"标准差计算错误: {std} vs {expected_std}"
            
            if std > 0.0:
                print(f"    ✅ 标准差为{std:.2f}，聚合正常")
            else:
                print(f"    ❌ 标准差为0，聚合异常")
        
        print("\n🎉 测试通过！多种子结果聚合逻辑工作正常。")
        print(f"📁 聚合结果保存在: {aggregate_file}")
        print(f"🌱 包含种子: {seed_list}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir)
        print(f"🧹 清理临时目录: {temp_dir}")

if __name__ == "__main__":
    success = test_aggregation_logic()
    sys.exit(0 if success else 1)