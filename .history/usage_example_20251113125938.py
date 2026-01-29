#!/usr/bin/env python3
"""
使用示例：展示如何使用 num_samples_per_task_for_evaluation 参数进行快速评估
"""

import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def example_usage():
    """展示如何使用 num_samples_per_task_for_evaluation 参数"""
    
    print("=" * 60)
    print("num_samples_per_task_for_evaluation 参数使用示例")
    print("=" * 60)
    
    print("\n1. 域内场景使用示例：")
    print("python main.py --dataset cifar100_224 --init_cls 10 --increment 10 \\")
    print("               --num_samples_per_task_for_evaluation 500 \\")
    print("               --iterations 1000 --lrate 1e-4")
    
    print("\n2. 跨域场景使用示例：")
    print("python main.py --cross_domain True \\")
    print("               --cross_domain_datasets imagenet-r cifar100_224 \\")
    print("               --num_samples_per_task_for_evaluation 200 \\")
    print("               --iterations 1000 --lrate 1e-4")
    
    print("\n3. 不启用采样（默认行为）：")
    print("python main.py --dataset cifar100_224 --init_cls 10 --increment 10 \\")
    print("               --num_samples_per_task_for_evaluation 0 \\")
    print("               --iterations 1000 --lrate 1e-4")
    
    print("\n4. 参数说明：")
    print("  --num_samples_per_task_for_evaluation:")
    print("    - 默认值：0（不启用采样）")
    print("    - 设置为正整数时，每个测试任务将只使用指定数量的样本进行评估")
    print("    - 这可以大大减少评估时间，特别适用于开发和调试阶段")
    print("    - 采样是随机的，但使用相同的种子可以保证结果可重现")
    
    print("\n5. 使用场景：")
    print("  - 开发阶段：快速验证模型是否正常工作")
    print("  - 调试阶段：快速定位问题")
