# -*- coding: utf-8 -*-
import copy
import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from compensator.gaussian_statistics import GaussianStatistics
from compensator.sldc_linear import LinearCompensator
from compensator.sldc_weaknonlinear import WeakNonlinearCompensator
from compensator.sldc_attention import AttentionCompensator


class DistributionCompensatorDistributionCompensator:
    """
    负责从特征对 (features_before, features_after) 构建多种补偿变体。
    输出 variants: Dict[str, Dict[int, GaussianStatistics]]
    """
    
    def __init__(
        self, 
        device: str = "cuda", 
        auxiliary_data_size: int = 1024
    ):
        self.device = device
        self.auxiliary_data_size = auxiliary_data_size

        # 特征和缓存相关
        self.feature_dim = None
        self.cached_Z = None
        self.aux_loader = None

        # 变体存储
        self.variants = self._initialize_variants()

    def _initialize_variants(self) -> Dict[str, Dict]:
        """初始化所有补偿变体结构"""
        return {
            "SeqFT": {},
            "SeqFT + linear_transform": {},
            "SeqFT + weaknonlinear_transform": {},
            "SeqFT + attention_transform": {},
        }

    # ============================================================
    #                  特征抽取与统计构建
    # ============================================================
    
    @torch.no_grad()
    def extract_features_before_after(
        self, 
        model_before: nn.Module, 
        model_after: nn.Module, 
        data_loader: DataLoader
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """同时提取前后模型的特征"""
        model_before.eval()
        model_after.eval()
        model_before.to(self.device)
        model_after.to(self.device)

        feats_before, feats_after, labels = [], [], []
        for batch in data_loader:
            inputs = batch[0]
            targets = batch[1]
            inputs = inputs.to(self.device)
            feats_before.append(model_before(inputs).cpu())
            feats_after.append(model_after(inputs).cpu())
            labels.append(targets)
            
        return (
            torch.cat(feats_before),
            torch.cat(feats_after), 
            torch.cat(labels)
        )

    @torch.no_grad()
    def _extract_combined_features(
        self,
        model_before: nn.Module,
        model_after: nn.Module,
        current_loader: DataLoader
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        一次性提取当前数据和辅助数据的组合特征，避免重复计算
        
        Returns:
            Tuple: (current_before, current_after, current_labels, combined_before, combined_after)
        """
        # 提取当前任务特征
        current_before, current_after, current_labels = self.extract_features_before_after(
            model_before, model_after, current_loader
        )
        
        # 初始化组合特征
        combined_before, combined_after = current_before, current_after
        
        # 如果有辅助数据，合并特征
        if self.aux_loader is not None:
            aux_before, aux_after, _ = self.extract_features_before_after(
                model_before, model_after, self.aux_loader
            )
            combined_before = torch.cat([current_before, aux_before])
            combined_after = torch.cat([current_after, aux_after])
            
        return current_before, current_after, current_labels, combined_before, combined_after

    def _build_gaussian_statistics(
        self, 
        features: torch.Tensor, 
        labels: torch.Tensor
    ) -> Dict[int, GaussianStatistics]:
        """为每个类别构建高斯统计量"""
        features = features.cpu()
        labels = labels.cpu()
        unique_labels = torch.unique(labels)
        
        stats = {}
        for lbl in unique_labels:
            mask = (labels == lbl)
            feats_class = features[mask]
            
            # 计算均值和协方差
            mu = feats_class.mean(0)
            if feats_class.size(0) >= 2:
                cov = torch.cov(feats_class.T)
            else:
                cov = torch.eye(feats_class.size(1)) * 1e-4
                
            stats[int(lbl.item())] = GaussianStatistics(mu, cov)
            
        return stats

    # ============================================================
    #                  补偿器计算与变换
    # ============================================================
    
    def _compute_linear_transform(
        self, 
        f_before: torch.Tensor, 
        f_after: torch.Tensor, 
        gamma: float = 0.1, 
        temp: float = 1.0
    ) -> LinearCompensator:
        """计算线性变换补偿器"""
        compensator = LinearCompensator(
            input_dim=f_before.size(1),
            gamma=gamma,
            temp=temp,
            device=self.device,
        )
        compensator.train(f_before.to(self.device), f_after.to(self.device))
        return compensator

    def _compute_weaknonlinear_transform(
        self, 
        f_before: torch.Tensor, 
        f_after: torch.Tensor
    ) -> WeakNonlinearCompensator:
        """计算弱非线性变换补偿器"""
        compensator = WeakNonlinearCompensator(
            input_dim=f_before.size(1),
            device=self.device,
        )
        compensator.train(f_before.to(self.device), f_after.to(self.device))
        return compensator

    def _compute_attention_transform(
        self, 
        f_before: torch.Tensor, 
        f_after: torch.Tensor
    ) -> AttentionCompensator:
        """计算注意力变换补偿器"""
        compensator = AttentionCompensator(
            input_dim=f_before.size(1),
            device=self.device,
        )
        compensator.train(f_before.to(self.device), f_after.to(self.device))
        return compensator

    def _update_variants_with_transforms(
        self, 
        task_id: int,
        current_stats: Dict[int, GaussianStatistics],
        combined_before: torch.Tensor,
        combined_after: torch.Tensor
    ):
        """使用各种变换更新变体"""
        if task_id <= 1:
            # 对于第一个任务，直接使用当前统计量初始化所有变体
            for variant_key in self.variants:
                if variant_key != "SeqFT":  # SeqFT已经在主流程中更新
                    self.variants[variant_key].update(copy.deepcopy(current_stats))
            return
            
        # 计算各种变换
        transforms = {
            "linear_transform": self._compute_linear_transform(combined_before, combined_after),
            "weaknonlinear_transform": self._compute_weaknonlinear_transform(combined_before, combined_after),
            "attention_transform": self._compute_attention_transform(combined_before, combined_after),
        }
        
        # 应用变换到现有统计量并更新
        for transform_name, transform in transforms.items():
            variant_key = f"SeqFT + {transform_name}"
            
            # 对现有统计量进行补偿
            compensated_existing_stats = transform.compensate(self.variants[variant_key])
            
            # 添加当前任务的统计量
            compensated_existing_stats.update(copy.deepcopy(current_stats))
            
            self.variants[variant_key] = compensated_existing_stats

    # ============================================================
    #                  主入口方法
    # ============================================================
    
    def build_all_variants(
        self, 
        task_id: int, 
        model_before: nn.Module, 
        model_after: nn.Module, 
        data_loader: DataLoader
    ) -> Dict[str, Dict[int, GaussianStatistics]]:
        """
        为当前任务构建所有补偿变体
        
        Args:
            task_id: 当前任务ID
            model_before: 微调前的模型
            model_after: 微调后的模型  
            data_loader: 当前任务的数据加载器
            
        Returns:
            包含所有补偿变体的字典
        """
        # 参数验证
        if task_id < 0:
            raise ValueError("task_id must be non-negative")
            
        # 一次性提取所有需要的特征，避免重复计算
        (current_before, current_after, current_labels, 
         combined_before, combined_after) = self._extract_combined_features(
            model_before, model_after, data_loader
        )
        
        # 初始化特征维度缓存
        if self.feature_dim is None:
            self.feature_dim = current_after.size(1)
            self.cached_Z = torch.randn(50000, self.feature_dim)
        
        # 构建当前任务的统计量
        current_stats = self._build_gaussian_statistics(current_after, current_labels)
        
        # 更新基础变体
        self.variants["SeqFT"].update(copy.deepcopy(current_stats))
        
        # 更新各种变换变体
        self._update_variants_with_transforms(
            task_id, current_stats, combined_before, combined_after
        )
        
        logging.info(f"[INFO] DistributionCompensator built {len(self.variants)} variants for task {task_id}.")
        return self.variants

    def set_auxiliary_loader(self, aux_loader: DataLoader):
        """设置辅助数据加载器"""
        self.aux_loader = aux_loader

    def clear_cache(self):
        """清除缓存"""
        self.cached_Z = None

    def get_variant_names(self) -> list:
        """获取所有变体名称"""
        return list(self.variants.keys())