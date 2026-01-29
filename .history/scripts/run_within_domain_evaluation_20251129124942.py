#!/usr/bin/env python3
"""
Within-domain数据集评估脚本
使用vit-b-p16架构，评估四个数据集，每个数据集使用一个GPU
每个数据集三个随机种子(1993, 1996, 1997)，顺序执行
使用basic_lora和smart_defaults
"""

import os
import sys
import subprocess
import datetime
import argparse
from pathlib import Path

def run_command(cmd, gpu_id, log_file):
    """执行命令并记录日志"""
    # 设置CUDA设备
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    print(f"执行命令: {' '.join(cmd)}")
    print(f"使用GPU: {gpu_id}")
    print(f"日志文件: {log_file}")
    
    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            env=env
        )
        
        # 实时输出并记录到日志文件
        for line in iter(process.stdout.readline, ''):
            print(line.rstrip())
            f.write(line)
            f.flush()
        
        return_code = process.wait()
        
    return return_code == 0

def main():
    parser = argparse.ArgumentParser(description='Within-domain数据集评估')
    parser.add_argument('--datasets', nargs='+', 
                       default=['imagenet-r', 'cifar100_224', 'cub200_224', 'cars196_224'],
                       help='要评估的数据集列表')
    parser.add_argument('--seeds', nargs='+', type=int,
                       default=[1993, 1996, 1997],
                       help='随机种子列表')
    parser.add_argument('--gpus', nargs='+', type=int,
                       default=[0, 1, 2, 3],
                       help='GPU列表')
    parser.add_argument('--vit_type', type=str, default='vit-b-p16',
                       help='ViT架构类型')
    parser.add_argument('--lora_type', type=str, default='basic_lora',
                       help='LoRA类型')
    parser.add_argument('--smart_defaults', action='store_true', default=True,
                       help='是否使用smart_defaults')
    parser.add_argument('--dry_run', action='store_true',
                       help='只显示将要执行的命令，不实际运行')
    
    args = parser.parse_args()
    
    # 创建日志目录
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
    log_dir = Path(f"within_domain_evaluation_logs_{timestamp}")
    log_dir.mkdir(exist_ok=True)
    
    print("="*50)
    print("开始within-domain数据集评估...")
    print(f"架构: {args.vit_type}")
    print(f"LoRA类型: {args.lora_type}")
    print(f"数据集: {args.datasets}")
    print(f"随机种子: {args.seeds}")
    print(f"GPU列表: {args.gpus}")
    print(f"使用smart_defaults: {args.smart_defaults}")
    print(f"日志目录: {log_dir}")
    print("="*50)
    
    success_count = 0
    total_count = 0
    
    # 遍历每个数据集
    for i, dataset in enumerate(args.datasets):
        if i >= len(args.gpus):
            print(f"警告: 数据集数量({len(args.datasets)})超过GPU数量({len(args.gpus)})")
            gpu_id = args.gpus[-1]  # 使用最后一个GPU
        else:
            gpu_id = args.gpus[i]
        
        print("\n" + "="*50)
        print(f"开始处理数据集: {dataset} (GPU: {gpu_id})")
        print("="*50)
        
        # 为当前数据集创建日志子目录
        dataset_log_dir = log_dir / f"{dataset}_{args.vit_type}"
        dataset_log_dir.mkdir(exist_ok=True)
        
        # 遍历每个随机种子（顺序执行）
        for seed in args.seeds:
            print("\n" + "-"*40)
            print(f"数据集: {dataset}, 随机种子: {seed}, GPU: {gpu_id}")
            print("-"*40)
            
            # 构建命令
            cmd = [
                "python", "main.py",
                "--dataset", dataset,
                "--vit_type", args.vit_type,
                "--lora_type", args.lora_type,
                "--seed_list", str(seed),
                "--cross_domain", "False"
            ]
            
            if args.smart_defaults:
                cmd.append("--smart_defaults")
            
            # 设置日志文件路径
            log_file = dataset_log_dir / f"seed_{seed}.log"
            
            total_count += 1
            
            if args.dry_run:
                print(f"[DRY RUN] 将要执行: {' '.join(cmd)}")
                print(f"[DRY RUN] 日志文件: {log_file}")
                success_count += 1
                continue
            
            # 执行命令
            success = run_command(cmd, gpu_id, log_file)
            
            if success:
                print(f"✓ 数据集 {dataset} 种子 {seed} 执行成功")
                success_count += 1
            else:
                print(f"✗ 数据集 {dataset} 种子 {seed} 执行失败，请检查日志: {log_file}")
    
    print("\n" + "="*50)
    print("执行结果统计:")
    print(f"成功: {success_count}/{total_count}")
    print(f"日志保存在: {log_dir}")
    print("="*50)
    
    return 0 if success_count == total_count else 1

if __name__ == "__main__":
    sys.exit(main())