#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将full_method实验结果整理成多维列表格式，便于分析和比较
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

def load_results():
    """加载实验结果"""
    with open("full_method_results.json", "r", encoding="utf-8") as f:
        return json.load(f)

def create_multidimensional_list(results):
    """
    将结果组织成多维列表格式
    
    Args:
        results: 从JSON文件加载的结果
        
    Returns:
        多维列表结构的结果
    """
    # 获取所有数据集、权重温度和权重P值
    datasets = sorted(results.keys())
    weight_temps = set()
    weight_ps = set()
    
    for dataset in datasets:
        weight_temps.update(results[dataset].keys())
        for temp in results[dataset].keys():
            weight_ps.update(results[dataset][temp].keys())
    
    weight_temps = sorted(weight_temps, key=lambda x: float(x))
    weight_ps = sorted(weight_ps, key=lambda x: float(x))
    
    # 创建多维列表
    multidimensional = {
        'datasets': datasets,
        'weight_temps': weight_temps,
        'weight_ps': weight_ps,
        'results': {}
    }
    
    # 填充结果数据
    for dataset in datasets:
        multidimensional['results'][dataset] = {}
        for temp in weight_temps:
            multidimensional['results'][dataset][str(temp)] = {}
            for p in weight_ps:
                if str(temp) in results[dataset] and str(p) in results[dataset][str(temp)]:
                    result = results[dataset][str(temp)][str(p)]
                    multidimensional['results'][dataset][str(temp)][str(p)] = {
                        'best_final_variant': result['best_final'],
                        'best_final_accuracy': result['final_accuracies'].get(result['best_final'], 0),
                        'best_average_variant': result['best_average'],
                        'best_average_accuracy': result['average_accuracies'].get(result['best_average'], 0),
                        'all_final_accuracies': result['final_accuracies'],
                        'all_average_accuracies': result['average_accuracies']
                    }
                else:
                    # 如果没有对应的结果，填充空值
                    multidimensional['results'][dataset][str(temp)][str(p)] = {
                        'best_final_variant': None,
                        'best_final_accuracy': 0,
                        'best_average_variant': None,
                        'best_average_accuracy': 0,
                        'all_final_accuracies': {},
                        'all_average_accuracies': {}
                    }
    
    return multidimensional

def create_accuracy_matrix(multidimensional, accuracy_type='best_final_accuracy'):
    """
    创建准确率矩阵，便于可视化比较
    
    Args:
        multidimensional: 多维列表格式的结果
        accuracy_type: 准确率类型 ('best_final_accuracy' 或 'best_average_accuracy')
        
    Returns:
        每个数据集的准确率矩阵字典
    """
    datasets = multidimensional['datasets']
    weight_temps = multidimensional['weight_temps']
    weight_ps = multidimensional['weight_ps']
    
    matrices = {}
    
    for dataset in datasets:
        # 创建矩阵
        matrix = np.zeros((len(weight_temps), len(weight_ps)))
        
        for i, temp in enumerate(weight_temps):
            for j, p in enumerate(weight_ps):
                matrix[i, j] = multidimensional['results'][dataset][str(temp)][str(p)][accuracy_type]
        
        matrices[dataset] = pd.DataFrame(
            matrix, 
            index=weight_temps, 
            columns=weight_ps
        )
    
    return matrices

def print_summary(multidimensional):
    """打印结果摘要"""
    datasets = multidimensional['datasets']
    weight_temps = multidimensional['weight_temps']
    weight_ps = multidimensional['weight_ps']
    
    print("=" * 80)
    print("FULL_METHOD 实验结果多维列表摘要")
    print("=" * 80)
    
    for dataset in datasets:
        print(f"\n数据集: {dataset}")
        print("-" * 60)
        
        # 找出最佳配置
        best_config = None
        best_acc = 0
        
        for temp in weight_temps:
            for p in weight_ps:
                result = multidimensional['results'][dataset][str(temp)][str(p)]
                if result['best_final_accuracy'] > best_acc:
                    best_acc = result['best_final_accuracy']
                    best_config = (temp, p, result['best_final_variant'])
        
        if best_config:
            print(f"最佳配置: 权重温度={best_config[0]}, 权重P={best_config[1]}, 变体={best_config[2]}")
            print(f"最佳准确率: {best_acc:.2f}%")
        
        # 打印准确率矩阵
        print("\n最终任务准确率矩阵 (%):")
        matrix = create_accuracy_matrix(multidimensional, 'best_final_accuracy')[dataset]
        print(matrix.round(2))
        
        print("\n平均任务准确率矩阵 (%):")
        matrix = create_accuracy_matrix(multidimensional, 'best_average_accuracy')[dataset]
        print(matrix.round(2))

def export_multidimensional_results(multidimensional, output_file="full_method_multidimensional.json"):
    """导出多维列表结果到JSON文件"""
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(multidimensional, f, indent=2, ensure_ascii=False)
    print(f"\n多维列表结果已保存到: {output_file}")

def export_accuracy_matrices(multidimensional):
    """导出准确率矩阵到CSV文件"""
    final_matrices = create_accuracy_matrix(multidimensional, 'best_final_accuracy')
    avg_matrices = create_accuracy_matrix(multidimensional, 'best_average_accuracy')
    
    for dataset, matrix in final_matrices.items():
        filename = f"accuracy_matrix_final_{dataset}.csv"
        matrix.to_csv(filename)
        print(f"最终任务准确率矩阵已保存到: {filename}")
    
    for dataset, matrix in avg_matrices.items():
        filename = f"accuracy_matrix_average_{dataset}.csv"
        matrix.to_csv(filename)
        print(f"平均任务准确率矩阵已保存到: {filename}")

def main():
    """主函数"""
    print("整理full_method实验结果为多维列表格式...")
    
    # 加载结果
    results = load_results()
    
    # 创建多维列表
    multidimensional = create_multidimensional_list(results)
    
    # 打印摘要
    print_summary(multidimensional)
    
    # 导出结果
    export_multidimensional_results(multidimensional)
    export_accuracy_matrices(multidimensional)
    
    print("\n整理完成!")

if __name__ == "__main__":
    main()