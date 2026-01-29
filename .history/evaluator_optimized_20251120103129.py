
"""
高性能评估模块 - 解决评估环节耗时问题
主要优化点：
1. 批量特征提取，避免重复计算
2. 矩阵化分类器评估，并行计算所有分类器
3. 智能采样机制，减少评估数据量
4. 特征缓存，避免重复计算
5. 自适应评估频率
"""

import time
import torch
import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional
from torch.utils.data import DataLoader, Subset
from collections import defaultdict


class OptimizedEvaluator:
    """
    优化的评估器，大幅提升评估速度
    """
    
    def __init__(self, network, device: torch.device, args: Dict[str, Any]):
        self.network = network
        self.device = device
        self.args = args
        
        # 评估配置
        self.eval_sample_ratio = args.get('eval_sample_ratio', 1.0)  # 采样比例
        self.eval_interval = args.get('eval_interval', 1)  # 评估间隔
        self.use_feature_cache = args.get('use_feature_cache', True)  # 是否使用特征缓存
        self.eval_batch_size = args.get('eval_batch_size', 256)  # 评估批次大小
        
        # 特征缓存
        self.feature_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # 性能统计
        self.eval_times = []
        
    def clear_cache(self):
        """清空特征缓存"""
        self.feature_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / max(total_requests, 1)
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.feature_cache)
        }
    
    def _create_sampled_loader(self, original_loader: DataLoader, sample_ratio: float) -> DataLoader:
        """创建采样的数据加载器"""
        if sample_ratio >= 1.0:
            return original_loader
            
        dataset = original_loader.dataset
        dataset_size = len(dataset)
        sample_size = int(dataset_size * sample_ratio)
        
        # 随机采样
        indices = torch.randperm(dataset_size)[:sample_size]
        sampled_dataset = Subset(dataset, indices)
        
        # 创建新的数据加载器
        return DataLoader(
            sampled_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=original_loader.num_workers,
            pin_memory=original_loader.pin_memory
        )
    
    def _extract_features_batch(self, loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        """批量提取特征，避免重复计算"""
        all_features = []
        all_targets = []
        
        # 检查缓存
        cache_key = f"features_{id(loader.dataset)}"
        if cache_key in self.feature_cache and self.use_feature_cache:
            self.cache_hits += 1
