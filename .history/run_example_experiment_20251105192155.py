#!/usr/bin/env python3
"""
示例脚本：展示如何使用修改后的代码运行多种子实验
"""

import os
import sys
import json
import subprocess
from pathlib import Path

def run_example_experiment():
    """运行一个示例实验，展示多种子聚合功能"""
    
    print("🚀 运行示例实验，展示多种子聚合功能")
    print("=" * 80)
    
    # 示例命令
    cmd = [
        "python", "main.py",
        "--dataset", "cifar100_224",
        "--smart_defaults",
        "--lora_type", "sgp_lora",
        "--weight_temp", "2.0",
        "--weight_kind", "log1p",
        "--weight_p", "1.0",
        "--seed_list", "1993", "1996", "1997",
        "--test"  # 使用测试模式，减少运行时间
    ]
    
    print("📋 运行命令:")
    print(" ".join(cmd))
    print("\n⏳ 开始运行实验...")
    print("注意：由于使用了--test参数，这将是一个快速测试")
    print("=" * 80)
    
    try:
        # 运行命令
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        print("✅ 实验完成！")
        print("\n📊 输出摘要:")
        print(result.stdout)
        
        # 查找并显示聚合结果文件
        log_dirs = list(Path("sldc_logs_test_user").glob("**/aggregate_results.json"))
        if log_dirs:
            aggregate_file = log_dirs[0]
            print(f"\n📁 聚合结果文件: {aggregate_file}")
            
            # 读取并显示聚合结果
            with open(aggregate_file, 'r', encoding='utf-8') as f:
                aggregate_data = json.load(f)
            
            print("\n📈 聚合统计:")
            for variant, stats in aggregate_data['final_task_stats'].items():
                mean = stats['mean']
                std = stats['std']
                print(f"  {variant:<30} : {mean:.2f}% ± {std:.2f}%")
            
            print(f"\n🌱 包含种子: {aggregate_data['seed_list']}")
            print(f"🔢 种子数量: {aggregate_data['num_seeds']}")
            
            # 检查最后一个变体的标准差
            last_variant = list(aggregate_data['final_task_stats'].keys())[-1]
            last_std = aggregate_data['final_task_stats'][last_variant]['std']
            if last_std > 0:
                print("\n✅ 多种子聚合成功！标准差大于0，表明正确聚合了多个种子的结果。")
            else:
                print("\n⚠️ 警告：标准差为0，可能没有正确聚合多个种子的结果。")
        else:
            print("\n❌ 未找到聚合结果文件")
            
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 实验失败: {e}")
        print("错误输出:")
        print(e.stderr)
        return False
        
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        return False

if __name__ == "__main__":
    success = run_example_experiment()
    sys.exit(0 if success else 1)