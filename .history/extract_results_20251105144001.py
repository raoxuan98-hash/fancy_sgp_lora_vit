#!/usr/bin/env python3
import os
import re
import json

def extract_final_accuracy(log_file):
    """从日志文件中提取最终任务准确率"""
    with open(log_file, 'r') as f:
        content = f.read()
    
    # 提取数据集名称
    dataset_match = re.search(r'dataset: (\S+)', content)
    dataset = dataset_match.group(1) if dataset_match else "unknown"
    
    # 提取weight_temp和weight_p
    temp_match = re.search(r'weight_temp: (\S+)', content)
    weight_temp = temp_match.group(1) if temp_match else "unknown"
    
    p_match = re.search(r'weight_p: (\S+)', content)
    weight_p = p_match.group(1) if p_match else "unknown"
    
    # 提取最终的Aggregated Final Task Accuracy
    # 查找attention_transform + QDA的结果
    pattern = r'SeqFT \+ attention_transform \+ QDA : ([\d.]+)%'
    matches = re.findall(pattern, content)
    
    if matches:
        # 取最后一个匹配项（最终结果）
        accuracy = float(matches[-1])
    else:
        accuracy = None
    
    return dataset, weight_temp, weight_p, accuracy

def main():
    log_dir = "logs/full_method_20251105_121524"
    results = []
    
    # 遍历所有日志文件
    for filename in os.listdir(log_dir):
        if filename.endswith('.log'):
            log_path = os.path.join(log_dir, filename)
            dataset, weight_temp, weight_p, accuracy = extract_final_accuracy(log_path)
            
            if accuracy is not None:
                results.append({
                    'dataset': dataset,
                    'weight_temp': float(weight_temp),
                    'weight_p': float(weight_p),
                    'accuracy': accuracy
                })
    
    # 按数据集、weight_temp和weight_p排序
    results.sort(key=lambda x: (x['dataset'], x['weight_temp'], x['weight_p']))
    
    # 打印表格
