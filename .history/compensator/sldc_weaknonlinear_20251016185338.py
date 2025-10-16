# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from compensator.gaussian_statistics import GaussianStatistics


class ResidMLP(nn.Module):
    """
    弱非线性残差变换器，用于模拟特征空间的小幅漂移。
    结构：线性恒等层 + ReLU MLP + softmax 融合。
    """
    def __init__(self, dim):
        super().__init__()
        self.log_scale = nn.Parameter(torch.tensor(0.0))
        self.fc1 = nn.Linear(dim, dim, bias=False)
        self.fc1.weight.data = torch.eye(dim)
        self.fc2 = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.ReLU(),
            nn.Linear(dim, dim, bias=False)
        )
        self.alphas = nn.Parameter(torch.tensor([1.0, 0.0]))

    def forward(self, x):
        scale = torch.exp(self.log_scale)
        weights = F.softmax(self.alphas / scale, dim=0)
        y1 = self.fc1(x)
        y2 = self.fc2(x)
        return weights[0] * y1 + weights[1] * y2

    def reg_loss(self):
        weights = F.softmax(self.alphas, dim=0)
        return (weights[0] - 1.0) ** 2


class WeakNonlinearTransform:
    """
    训练一个弱非线性 MLP 变换，用于漂移补偿。
    """
    def __init__(self, input_dim: int):
        self.net = ResidMLP(input_dim)
        self.is_trained = False

    def train(self, features_before: torch.Tensor, features_after: torch.Tensor, epochs: int = 4000, lr: float = 0.001):
        device = features_before.device
        self.net = self.net.to(device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr / 3)
        crit = nn.MSELoss()
        X = F.normalize(features_before, dim=-1)
        Y = F.normalize(features_after, dim=-1)
        for ep in range(epochs):
            idx = torch.randint(0, X.size(0), (64,), device=device)
            pred = self.net(X[idx])
            loss = crit(pred, Y[idx]) + 0.5 * self.net.reg_loss()
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
            if (ep + 1) % 1000 == 0:
                logging.info(f"[WeakNonlinearTransform] Epoch {ep+1}/{epochs}, loss={loss.item():.6f}")
        self.is_trained = True
        return self.net

    @torch.no_grad()
    def transform_features(self, features: torch.Tensor) -> torch.Tensor:
        if not self.is_trained:
            raise ValueError("非线性变换器尚未训练。")
        norms = features.norm(dim=1, keepdim=True)
        return self.net(features / norms) * norms

    @torch.no_grad()
    def transform_stats(self, stats_dict, n_samples: int = 5000):
        """对每个类的统计进行采样-变换-再估计"""
        if not self.is_trained:
            raise ValueError("非线性变换器尚未训练。")
        device = next(self.net.parameters()).device
        new_stats = {}
        for cid, stat in stats_dict.items():
            samples = stat.sample(n_samples).to(device)
            transformed = self.transform_features(samples)
            mu_new = transformed.mean(0).cpu()
            cov_new = torch.cov(transformed.T).cpu()
            new_stats[cid] = GaussianStatistics(mu_new, cov_new, stat.reg)
        return new_stats
