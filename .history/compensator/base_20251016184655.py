# compensator/base_compensator.py
from abc import ABC, abstractmethod
from typing import Dict
import torch
from models.gaussian_statistics import GaussianStatistics


class BaseCompensator(ABC):
    """所有补偿器的抽象基类：包含 train() + compensate() 两阶段"""

    def __init__(self, input_dim: int, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.input_dim = input_dim
        self.device = device
        self.is_trained = False

    @abstractmethod
    def train(self, features_before: torch.Tensor, features_after: torch.Tensor):
        """拟合补偿模型"""
        pass

    @abstractmethod
    def compensate(self, stats_dict: Dict[int, GaussianStatistics]) -> Dict[int, GaussianStatistics]:
        """对高斯分布统计进行补偿"""
        pass

