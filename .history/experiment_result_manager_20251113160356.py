
"""
实验结果管理器 - 实现改进的统一存储结构

该模块提供了一个统一的实验结果收集和存储解决方案，与现有的trainer.py兼容，
同时实现改进的存储结构设计。
"""

import os
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import numpy as np
import logging


class ExperimentResultManager:
    """
    实验结果管理器，负责将现有的实验结果转换为改进的统一存储结构
    """
    
    def __init__(self, base_dir: str = "experiment_results"):
        """
        初始化实验结果管理器
        
        Args:
            base_dir: 实验结果存储的基础目录
        """
        self.base_dir = Path(base_dir)
        self.experiments_dir = self.base_dir / "experiments"
        self.raw_results_dir = self.base_dir / "raw_results"
        
        # 创建基础目录结构
        self._create_directory_structure()
        
        # 初始化实验注册表
        self.registry_file = self.experiments_dir / "metadata" / "experiment_registry.json"
        self.experiment_registry = self._load_experiment_registry()
        
    def _create_directory_structure(self):
        """创建改进的存储目录结构"""
        # 实验配置目录
        dirs = [
            self.experiments_dir / "configs" / "main_experiments" / "cross_domain",
            self.experiments_dir / "configs" / "main_experiments" / "within_domain",
            self.experiments_dir / "configs" / "ablation_experiments" / "cross_domain",
            self.experiments_dir / "configs" / "ablation_experiments" / "within_domain",
            self.experiments_dir / "configs" / "parameter_sensitivity" / "cross_domain",
            self.experiments_dir / "configs" / "parameter_sensitivity" / "within_domain",
            self.experiments_dir / "metadata",
            self.raw_results_dir / "cross_domain",
            self.raw_results_dir / "within_domain"
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # 创建元数据文件
        self._create_metadata_files()
    
    def _create_metadata_files(self):
        """创建元数据文件"""
        metadata_dir = self.experiments_dir / "metadata"
        
        # 数据集信息文件
        dataset_info = {
            "cross_domain_datasets": {
                "imagenet-r": {
                    "name": "ImageNet-R",
                    "description": "ImageNet Rendition dataset",
                    "num_classes": 200,
                    "image_size": 224
                },
                "caltech-101": {
                    "name": "Caltech-101",
                    "description": "Caltech-101 dataset",
                    "num_classes": 101,
                    "image_size": 224
                },
                "dtd": {
                    "name": "Describable Textures Dataset",
                    "description": "DTD dataset",
                    "num_classes": 47,
                    "image_size": 224
                }
            },
            "within_domain_datasets": {
                "cifar100_224": {
                    "name": "CIFAR-100",
                    "description": "CIFAR-100 dataset resized to 224x224",
                    "num_classes": 100,
                    "image_size": 224
                },
                "cub200_224": {
                    "name": "CUB-200-2011",
                    "description": "Caltech-UCSD Birds-200-2011 dataset",
                    "num_classes": 200,
                    "image_size": 224
                },
                "cars196_224": {
                    "name": "Stanford Cars-196",
                    "description": "Stanford Cars dataset",
                    "num_classes": 196,
                    "image_size": 224
                }
            }
        }
        
        dataset_info_file = metadata_dir / "dataset_info.json"
        if not dataset_info_file.exists():
            with open(dataset_info_file, 'w', encoding='utf-8') as f:
                json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    
    def _load_experiment_registry(self) -> Dict:
        """加载实验注册表"""
        if self.registry_file.exists():
            with open(self.registry_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return {
                "experiments": {},
                "last_experiment_id": 0,
                "created_at": datetime.now().isoformat()
            }
    
    def _save_experiment_registry(self):
        """保存实验注册表"""
        with open(self.registry_file, 'w', encoding='utf-8') as f:
            json.dump(self.experiment_registry, f, ensure_ascii=False, indent=2)
    
    def register_experiment(self, experiment_config: Dict, original_log_path: str) -> str:
        """
        注册新实验并生成实验ID
        
        Args:
            experiment_config: 实验配置
            original_log_path: 原始日志路径
            
        Returns:
            实验ID
        """
        # 生成新的实验ID
        self.experiment_registry["last_experiment_id"] += 1
        experiment_id = f"exp_{self.experiment_registry['last_experiment_id']:06d}"
        
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 确定实验类型
        is_cross_domain = experiment_config.get('cross_domain', False)
        experiment_type = "cross_domain" if is_cross_domain else "within_domain"
        
        # 注册实验信息
        self.experiment_registry["experiments"][experiment_id] = {
            "experiment_id": experiment_id,
            "timestamp": timestamp,
            "experiment_type": experiment_type,
            "config": experiment_config,
            "original_log_path": original_log_path,
            "status": "registered",
            "created_at": datetime.now().isoformat()
        }
        
        # 保存注册表
        self._save_experiment_registry()
        
        return experiment_id
    
    def migrate_existing_results(self, source_log_dir: str = "sldc_logs_sgp_lora_vit_main"):
        """
        迁移现有的实验结果到新的存储结构
        
        Args:
            source_log_dir: 现有日志目录
        """
        source_path = Path(source_log_dir)
        if not source_path.exists():
            logging.warning(f"源日志目录不存在: {source_path}")
            return
        
        logging.info(f"开始迁移实验结果从 {source_path} 到 {self.base_dir}")
        
        # 遍历所有数据集
        for dataset_model_dir in source_path.iterdir():
            if not dataset_model_dir.is_dir():
                continue
                
            # 解析数据集和模型类型
            parts = dataset_model_dir.name.split('_')
            if len(parts) < 2:
                continue
                
            dataset = '_'.join(parts[:-1])
            model_type = parts[-1]
            
            # 遍历所有任务设置
            for task_dir in dataset_model_dir.iterdir():
                if not task_dir.is_dir():
                    continue
                
                # 遍历所有LoRA配置
                for lora_dir in task_dir.iterdir():
                    if not lora_dir.is_dir():
                        continue
                    
                    # 遍历所有方法参数
                    for method_dir in lora_dir.iterdir():
                        if not method_dir.is_dir():
                            continue
                        
