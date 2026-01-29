#!/usr/bin/env python3
"""
聚合多个随机种子的结果
用于修复aggregate_results.json没有聚合三个随机种子结果的问题
"""

import json
import os
import glob
import re
import numpy as np
import time
from pathlib import Path

def extract_results_from_log(log_path):
    """从record.log文件中提取最终结果"""
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找最后一个Incremental Learning Evaluation Analysis部分
        # 使用正则表达式匹配
        last_task_pattern = r"Last Task ID: (\d+)"
        match = re.search(last_task_pattern, content)
        
        if not match:
            print(f"警告: 在 {log_path} 中未找到最后任务ID")
            return None
        
        task_id = int(match.group(1))
        
        # 查找最终任务准确度部分
        final_task_pattern = r"   ── Final Task Accuracy (%) ──"
        final_task_match = re.search(final_task_pattern, content[match.start():])
        
        if not final_task_match:
            print(f"警告: 在 {log_path} 中未找到最终任务准确度")
            return None
            
        # 提取所有变体的准确度
        accuracies = {}
        lines = content[final_task_match.start():]
        
        for line in lines:
            if "SeqFT +" in line and ":" in line:
                parts = line.split(":")
                variant = parts[0].strip()
                accuracy = float(parts[1].strip().replace("%", ""))
                accuracies[variant] = accuracy
        
        # 查找平均准确度部分
        avg_task_pattern = r"   ── Average Accuracy Across Tasks (%) ──"
        avg_task_match = re.search(avg_task_pattern, content[final_task_match.start():])
        
        avg_accuracies = {}
        if avg_task_match:
            lines = content[avg_task_match.start():]
            for line in lines:
                if "SeqFT +" in line and ":" in line:
                    parts = line.split(":")
                    variant = parts[0].strip()
                    accuracy = float(parts[1].strip().replace("%", ""))
                    avg_accuracies[variant] = accuracy
        
        # 查找每个任务的准确度趋势
        per_task_pattern = r"   ── Per-Task Accuracy Trends (Mean ± Std) ──"
        per_task_match = re.search(per_task_pattern, content[final_task_match.start():])
        
        per_task_trends = {}
        if per_task_match:
            lines = content[per_task_match.start():]
            for line in lines:
                if "SeqFT +" in line and "→" in line:
                    parts = line.split(":")
                    variant = parts[0].strip()
                    trends = parts[1].strip().split(" → ")
                    per_task_trends[variant] = trends
        
        return {
            "seed": os.path.basename(os.path.dirname(log_path)),
            "task_id": task_id,
            "last_task_accuracies": accuracies,
            "average_accuracies": avg_accuracies,
            "per_task_trends": per_task_trends
        }
    except Exception as e:
        print(f"处理 {log_path} 时出错: {e}")
        return None

def aggregate_multiple_seeds(log_dir):
    """聚合多个种子的结果"""
    # 查找所有种子的record.log文件
    log_files = glob.glob(os.path.join(log_dir, "opt-*/record.log"))
    
    if len(log_files) == 0:
        print(f"在 {log_dir} 中未找到任何record.log文件")
        return
    
    print(f"找到 {len(log_files)} 个种子日志文件")
    
    # 提取每个种子的结果
    seed_results = {}
    for log_file in log_files:
        result = extract_results_from_log(log_file)
        if result:
            seed_key = f"seed_{result['seed']}"
            seed_results[seed_key] = result
    
    if len(seed_results) < 2:
        print(f"只找到 {len(seed_results)} 个有效的种子结果，需要至少2个种子才能计算标准差")
        return
    
    print(f"成功提取了 {len(seed_results)} 个种子的结果")
    
    # 获取所有变体名称
    all_variants = set()
    for result in seed_results.values():
        all_variants.update(result["last_task_accuracies"].keys())
        all_variants.update(result["average_accuracies"].keys())
    
    # 初始化聚合容器
    final_task_values = {variant: [] for variant in all_variants}
    avg_task_values = {variant: [] for variant in all_variants}
    per_task_trends = {variant: [] for variant in all_variants}
    
    # 收集数据
    for seed_key, result in seed_results.items():
        for variant in all_variants:
            final_task_values[variant].append(result["last_task_accuracies"].get(variant, 0.0))
            avg_task_values[variant].append(result["average_accuracies"].get(variant, 0.0))
            
            # 收集每个任务的准确度趋势
            if variant in result["per_task_trends"]:
                trends = result["per_task_trends"][variant]
                per_task_trends[variant].append(trends)
    
    # 计算均值和标准差
    final_task_stats = {}
    avg_task_stats = {}
    per_task_stats = {}
    
    for variant in all_variants:
        f_vals = np.array(final_task_values[variant])
        a_vals = np.array(avg_task_values[variant])
        
        final_task_stats[variant] = {
            "mean": float(np.mean(f_vals)),
            "std": float(np.std(f_vals))
        }
        
        avg_task_stats[variant] = {
            "mean": float(np.mean(a_vals)),
            "std": float(np.std(a_vals))
        }
        
        # 计算每个任务的准确度趋势
        if per_task_trends[variant]:
            # 将所有种子的任务准确度转换为numpy数组
            task_arrays = np.array(per_task_trends[variant])
            # 计算每个任务的平均值和标准差
            task_means = np.mean(task_arrays, axis=0).tolist()
            task_stds = np.std(task_arrays, axis=0).tolist()
            per_task_stats[variant] = {
                "means": task_means,
                "stds": task_stds,
                "num_tasks": len(task_means)
            }
    
    # 保存聚合结果
    aggregate_file = os.path.join(log_dir, "aggregate_results.json")
    
    save_data = {
        "final_task_stats": final_task_stats,
        "average_across_tasks_stats": avg_task_stats,
        "per_task_accuracy_trends": per_task_stats,
        "seed_list": list(seed_results.keys()),
        "num_seeds": len(seed_results),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "variants": sorted(all_variants),
        "max_tasks": max([per_task_stats.get(variant, {}).get("num_tasks", 0) for variant in per_task_stats])
    }
    
    with open(aggregate_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    print(f"聚合结果已保存到: {aggregate_file}")
    
    # 打印聚合结果摘要
    print("\n=== 聚合结果摘要 ===")
    print(f"种子数量: {len(seed_results)}")
    print("\n最终任务准确度 (Mean ± Std):")
    for variant in sorted(all_variants):
        mean, std = final_task_stats[variant]["mean"], final_task_stats[variant]["std"]
        print(f"  {variant:<20} : {mean:.2f}% ± {std:.2f}%")
    
    print("\n平均任务准确度 (Mean ± Std):")
    for variant in sorted(all_variants):
        mean, std = avg_task_stats[variant]["mean"], avg_task_stats[variant]["std"]
        print(f"  {variant:<20} : {mean:.2f}% ± {std:.2f}%")
    
    print("\n每个任务的准确度趋势 (Mean ± Std):")
    for variant in sorted(all_variants):
        if variant in per_task_stats:
            task_means = per_task_stats[variant]["means"]
            task_stds = per_task_stats[variant]["stds"]
            trend_str = " → ".join([f"{m:.2f}%±{s:.2f}%" for m, s in zip(task_means, task_stds)])
            print(f"  {variant:<20} : {trend_str}")

def main():
    """主函数"""
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description="聚合多个随机种子的结果")
    parser.add_argument("--log_dir", type=str, 
                       default="sldc_logs_sgp_lora_vit_main/cars196_224_vit-b-p16-mocov3/init-20_inc-20/lrank-4_ltype-nsp_lora/eps-0.05_w-0.0/",
                       help="包含种子日志文件的目录路径")
    
    args = parser.parse_args()
    
    print("开始聚合多个随机种子的结果...")
    aggregate_multiple_seeds(args.log_dir)

if __name__ == "__main__":
    main()