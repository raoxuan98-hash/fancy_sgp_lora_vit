#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提取full_method实验结果的脚本
从logs目录中的full_method实验日志文件中提取关键信息并整理成多维列表
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

def parse_log_file(log_path: str) -> Dict:
    """
    解析单个日志文件，提取实验参数和结果
    
    Args:
        log_path: 日志文件路径
        
    Returns:
        包含实验参数和结果的字典
    """
    result = {
        'log_path': log_path,
        'dataset': None,
        'weight_temp': None,
        'weight_p': None,
        'seed': None,
        'final_accuracies': {},  # 各变体的最终任务准确率
        'average_accuracies': {},  # 各变体的平均准确率
        'best_final': None,  # 最终任务最佳结果
        'best_average': None,  # 平均准确率最佳结果
    }
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 从文件名提取基本信息
        filename = os.path.basename(log_path)
        # 例如: cifar100_224_temp1.0_p1.0_seed1993.log
        match = re.match(r'(.+)_temp(.+)_p(.+)_seed(.+)\.log', filename)
        if match:
            result['dataset'] = match.group(1)
            result['weight_temp'] = float(match.group(2))
            result['weight_p'] = float(match.group(3))
            result['seed'] = int(match.group(4))
        
        # 提取参数信息
        param_patterns = {
            'dataset': r'dataset: (.+)',
            'weight_temp': r'weight_temp: (.+)',
            'weight_p': r'weight_p: (.+)',
            'seed': r'seed: (.+)',
        }
        
        for key, pattern in param_patterns.items():
            matches = re.findall(pattern, content)
            if matches and result[key] is None:
                if key in ['weight_temp', 'weight_p', 'seed']:
                    try:
                        result[key] = type(result[key])(matches[-1]) if result[key] is not None else float(matches[-1]) if '.' in matches[-1] else int(matches[-1])
                    except (ValueError, TypeError):
                        result[key] = matches[-1]
                else:
                    result[key] = matches[-1]
        
        # 提取变体名称和对应的准确率
        variant_pattern = r'(\S+(?:\s+\+\s+\S+)*)\s*:\s*([\d.]+)%'
        
        # 查找所有准确率报告部分 - 修改正则表达式以匹配实际格式
        accuracy_sections = re.findall(r'Final Task Accuracy.*?\n((?:\s*[^:]+:\s*[\d.]+%\s*\n)+)', content, re.DOTALL)
        if accuracy_sections:
            for section in accuracy_sections:
                variants = re.findall(variant_pattern, section)
                for variant, accuracy in variants:
                    result['final_accuracies'][variant.strip()] = float(accuracy)
        
        avg_sections = re.findall(r'Average Accuracy Across Tasks.*?\n((?:\s*[^:]+:\s*[\d.]+%\s*\n)+)', content, re.DOTALL)
        if avg_sections:
            for section in avg_sections:
                variants = re.findall(variant_pattern, section)
                for variant, accuracy in variants:
                    result['average_accuracies'][variant.strip()] = float(accuracy)
        
        # 如果上面的方法没有提取到数据，尝试另一种方法
        if not result['final_accuracies']:
            # 尝试提取最后的聚合结果
            final_matches = re.findall(r'([A-Za-z\s+]+):\s*([\d.]+)%\s*±\s*[\d.]+%', content)
            for match in final_matches:
                method = match[0].strip()
                accuracy = float(match[1])
                result['final_accuracies'][method] = accuracy
                
        if not result['average_accuracies']:
            # 尝试提取最后的聚合结果
            avg_matches = re.findall(r'([A-Za-z\s+]+):\s*([\d.]+)%\s*±\s*[\d.]+%', content)
            for match in avg_matches:
                method = match[0].strip()
                accuracy = float(match[1])
                result['average_accuracies'][method] = accuracy
        
        # 提取最佳结果
        summary_pattern = r"Best in Final Task: '([^']+)' \| Best Average: '([^']+)'"
        summary_matches = re.findall(summary_pattern, content)
        if summary_matches:
            best_final_candidate = summary_matches[-1][0]
            best_average_candidate = summary_matches[-1][1]
            
            # 检查最佳结果是否在准确率字典中，如果不在则使用最高准确率的变体
            if best_final_candidate in result['final_accuracies']:
                result['best_final'] = best_final_candidate
            elif result['final_accuracies']:
                result['best_final'] = max(result['final_accuracies'].items(), key=lambda x: x[1])[0]
            
            if best_average_candidate in result['average_accuracies']:
                result['best_average'] = best_average_candidate
            elif result['average_accuracies']:
                result['best_average'] = max(result['average_accuracies'].items(), key=lambda x: x[1])[0]
        
        # 如果没有找到summary，从准确率中推断最佳结果
        if not result['best_final'] and result['final_accuracies']:
            result['best_final'] = max(result['final_accuracies'].items(), key=lambda x: x[1])[0]
        if not result['best_average'] and result['average_accuracies']:
            result['best_average'] = max(result['average_accuracies'].items(), key=lambda x: x[1])[0]
            
    except Exception as e:
        print(f"解析日志文件 {log_path} 时出错: {e}")
        
    return result

def find_full_method_logs(logs_dir: str = "logs") -> List[str]:
    """
    查找所有full_method相关的日志文件
    
    Args:
        logs_dir: 日志目录路径
        
    Returns:
        日志文件路径列表
    """
    log_files = []
    logs_path = Path(logs_dir)
    
    # 查找所有full_method_*目录
    for method_dir in logs_path.glob("full_method_*"):
        if method_dir.is_dir():
            # 查找目录中的所有.log文件
            for log_file in method_dir.glob("*.log"):
                log_files.append(str(log_file))
    
    return sorted(log_files)

def organize_results(results: List[Dict]) -> Dict:
    """
    将结果组织成多维结构
    
    Args:
        results: 解析后的日志结果列表
        
    Returns:
        组织后的结果字典
    """
    organized = {}
    
    for result in results:
        if not result['dataset']:
            continue
            
        dataset = result['dataset']
        weight_temp = result['weight_temp']
        weight_p = result['weight_p']
        
        if dataset not in organized:
            organized[dataset] = {}
            
        if weight_temp not in organized[dataset]:
            organized[dataset][weight_temp] = {}
            
        organized[dataset][weight_temp][weight_p] = result
    
    return organized

def print_results_summary(organized: Dict):
    """
    打印结果摘要
    
    Args:
        organized: 组织后的结果字典
    """
    print("=" * 80)
    print("FULL_METHOD 实验结果摘要")
    print("=" * 80)
    
    for dataset, temp_data in organized.items():
        print(f"\n数据集: {dataset}")
        print("-" * 60)
        
        for weight_temp, p_data in sorted(temp_data.items()):
            print(f"\n权重温度: {weight_temp}")
            print("-" * 40)
            
            for weight_p, result in sorted(p_data.items()):
                print(f"\n权重P: {weight_p}")
                
                if result['best_final'] and result['best_final'] in result['final_accuracies']:
                    best_final_acc = result['final_accuracies'][result['best_final']]
                    print(f"  最佳最终任务准确率: {result['best_final']} - {best_final_acc:.2f}%")
                
                if result['best_average'] and result['best_average'] in result['average_accuracies']:
                    best_avg_acc = result['average_accuracies'][result['best_average']]
                    print(f"  最佳平均准确率: {result['best_average']} - {best_avg_acc:.2f}%")

def export_to_csv(organized: Dict, output_file: str = "full_method_results.csv"):
    """
    将结果导出为CSV格式
    
    Args:
        organized: 组织后的结果字典
        output_file: 输出文件路径
    """
    import csv
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'dataset', 'weight_temp', 'weight_p', 'seed',
            'best_final_variant', 'best_final_accuracy',
            'best_average_variant', 'best_average_accuracy'
        ]
        
        # 添加所有变体的准确率列
        all_variants = set()
        for dataset in organized.values():
            for temp in dataset.values():
                for result in temp.values():
                    all_variants.update(result['final_accuracies'].keys())
        
        for variant in sorted(all_variants):
            fieldnames.append(f"final_{variant}")
            fieldnames.append(f"average_{variant}")
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for dataset, temp_data in organized.items():
            for weight_temp, p_data in sorted(temp_data.items()):
                for weight_p, result in sorted(p_data.items()):
                    row = {
                        'dataset': dataset,
                        'weight_temp': weight_temp,
                        'weight_p': weight_p,
                        'seed': result['seed'],
                        'best_final_variant': result['best_final'],
                        'best_final_accuracy': result['final_accuracies'].get(result['best_final'], ''),
                        'best_average_variant': result['best_average'],
                        'best_average_accuracy': result['average_accuracies'].get(result['best_average'], ''),
                    }
                    
                    # 添加所有变体的准确率
                    for variant in sorted(all_variants):
                        row[f"final_{variant}"] = result['final_accuracies'].get(variant, '')
                        row[f"average_{variant}"] = result['average_accuracies'].get(variant, '')
                    
                    writer.writerow(row)
    
    print(f"\n结果已导出到: {output_file}")

def main():
    """主函数"""
    print("开始提取full_method实验结果...")
    
    # 查找所有日志文件
    log_files = find_full_method_logs()
    print(f"找到 {len(log_files)} 个日志文件")
    
    # 解析所有日志文件
    results = []
    for log_file in log_files:
        print(f"解析: {log_file}")
        result = parse_log_file(log_file)
        results.append(result)
    
    # 组织结果
    organized = organize_results(results)
    
    # 打印摘要
    print_results_summary(organized)
    
    # 导出为CSV
    export_to_csv(organized)
    
    # 保存详细结果为JSON
    with open("full_method_results.json", "w", encoding="utf-8") as f:
        json.dump(organized, f, indent=2, ensure_ascii=False)
    
    print(f"\n详细结果已保存到: full_method_results.json")
    print("提取完成!")

if __name__ == "__main__":
    main()