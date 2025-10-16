# -*- coding: utf-8 -*-
import math
import copy
import numpy as np
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, TensorDataset
from sklearn.cluster import KMeans

def cholesky_manual_stable(matrix: torch.Tensor, reg: float = 1e-5) -> torch.Tensor:
    n = matrix.size(0)
    L = torch.zeros_like(matrix)
    reg_eye = reg * torch.eye(n, device=matrix.device, dtype=matrix.dtype)
    matrix = matrix + reg_eye

    for j in range(n):
        s_diag = torch.sum(L[j, :j] ** 2, dim=0)
        diag = matrix[j, j] - s_diag
        L[j, j] = torch.sqrt(torch.clamp(diag, min=1e-8))

        if j < n - 1:
            s_off = L[j + 1:, :j] @ L[j, :j]
            L[j + 1:, j] = (matrix[j + 1:, j] - s_off) / L[j, j]
    return L

def symmetric_cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor,
                                 sce_a: float = 0.5, sce_b: float = 0.5) -> torch.Tensor:
    pred = F.softmax(logits, dim=1)
    pred = torch.clamp(pred, min=1e-7, max=1.0)

    label_one_hot = F.one_hot(targets, num_classes=pred.size(1)).float().to(pred.device)
    label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)

    ce_loss = -(label_one_hot * torch.log(pred)).sum(dim=1).mean()
    rce_loss = -(pred * torch.log(label_one_hot)).sum(dim=1).mean()
    return sce_a * ce_loss + sce_b * rce_loss


class ResidMLP(nn.Module):
    def __init__(self, dim):
        super(ResidMLP, self).__init__()
        self.log_scale = nn.Parameter(torch.tensor(0.0))
        self.fc1 = nn.Linear(dim, dim, bias=False)
        self.fc1.weight.data = torch.eye(dim)

        self.fc2 = nn.Sequential(    
            nn.Linear(dim, dim, bias=False),
            nn.ReLU(),
            nn.Linear(dim, dim, bias=False))
        
        self.alphas =  nn.Parameter(torch.tensor([1.0, 0.0]))

    def forward(self, x):
        scale = torch.exp(self.log_scale)
        weights = F.softmax(self.alphas / scale, dim=0)
        y1 = self.fc1(x)
        y2 = self.fc2(x)
        y = weights[0] * y1 + weights[1] * y2
        return y 
    
    def reg_loss(self):
        weights = F.softmax(self.alphas, dim=0)
        return (weights[0] - 1.0) ** 2
    
class GaussianStatistics:
    def __init__(self, mean: torch.Tensor, cov: torch.Tensor, reg: float = 1e-5):
        if mean.dim() == 2 and mean.size(0) == 1:
            mean = mean.squeeze(0)
        assert mean.dim() == 1, "GaussianStatistics.mean should be a 1D vector (D,)"
        self.mean = mean
        self.cov = cov
        self.reg = reg
        self.L = cholesky_manual_stable(cov, reg=reg)
        self.L_weighted = None
        self.weighted_cov = None

    def initialize_weighted_covariance(self, weighted_cov: torch.Tensor):
        self.weighted_cov = weighted_cov
        self.L_weighted = cholesky_manual_stable(weighted_cov, reg=self.reg)

    def sample(self, n_samples: int = None, cached_eps: torch.Tensor = None, use_weighted_cov: bool = False) -> torch.Tensor:
        device = self.mean.device
        d = self.mean.size(0)

        if cached_eps is None:
            eps = torch.randn(n_samples, d, device=device)
        else:
            eps = cached_eps.to(device)
            n_samples = eps.size(0)

        L = self.L_weighted if (use_weighted_cov and self.L_weighted is not None) else self.L
        samples = self.mean.unsqueeze(0) + eps @ L.t()
        return samples


class WeakNonlinearTransform:
    def __init__(self, input_dim: int):
        self.net = ResidMLP(input_dim)
        self.is_trained = False
        
    def train(self, features_before: torch.Tensor, features_after: torch.Tensor, 
              epochs: int = 4000, lr: float = 0.001):
        """训练非线性变换网络"""
        device = features_before.device
        self.net = self.net.to(device)
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/3)
        criterion = nn.MSELoss()
    
        X = F.normalize(features_before, dim=-1)
        Y = F.normalize(features_after, dim=-1)

        for epoch in range(epochs):
            optimizer.zero_grad()
            idx = torch.randint(0, X.size(0), (64,)).to(device)
            x = X[idx]; y = Y[idx]
            pred = self.net(x)
            loss = criterion(pred, y) + 0.5 * self.net.reg_loss()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if (epoch + 1) % 1000 == 0:
                print(f"弱非线性变换训练step {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
        
        self.is_trained = True
        return self.net
    
    def transform_features(self, features: torch.Tensor) -> torch.Tensor:
        """变换特征"""
        if not self.is_trained:
            raise ValueError("非线性变换器尚未训练")
        with torch.no_grad():
            norms = features.norm(dim=1, keepdim=True)
            return self.net(features / norms) * norms
    
    def transform_stats(self, stats_dict: Dict[int, GaussianStatistics], 
                       n_samples: int = 5000) -> Dict[int, GaussianStatistics]:
        """通过采样变换高斯统计量"""
        if not self.is_trained:
            raise ValueError("非线性变换器尚未训练")
            
        transformed_stats = {}
        device = next(self.net.parameters()).device
        
        for cid, stat in stats_dict.items():
            # 采样并变换
            samples = stat.sample(n_samples).to(device)
            transformed_samples = self.transform_features(samples)
            
            # 计算新的统计量
            new_mean = transformed_samples.mean(dim=0).cpu()
            new_cov = torch.cov(transformed_samples.T).cpu()
    
            transformed_stats[cid] = GaussianStatistics(new_mean, new_cov, stat.reg)
        return transformed_stats

class FeCAMClassifier(nn.Module):
    """
    非参判别头：基于每类的 (mean, cov) 做收缩 → 相关系数化 → Cholesky，
    前向时按马氏距离评分。可选 logdet 偏置与温度缩放。
    """
    def __init__(
        self,
        stats: Dict[int, GaussianStatistics],
        shrink_gamma_diag: float = 1e-4,
        shrink_gamma_off: float = 1e-4,
        add_logdet_bias: bool = False,
        temperature: float = 1.0,
        use_tukey: bool = False,     # 仅当统计量基于 Tukey 变换特征计算时再打开
        eps: float = 1e-6,
    ):
        super().__init__()
        assert len(stats) > 0, "FeCAMClassifier: empty stats."

        # 确定类别顺序与维度
        class_ids = sorted(list(stats.keys()))
        D = stats[class_ids[0]].mean.numel()
        C = len(class_ids)

        mus = torch.stack([stats[c].mean for c in class_ids], dim=0)       # (C, D)
        covs = torch.stack([stats[c].cov for c in class_ids], dim=0)       # (C, D, D)

        # —— 协方差收缩：Σ^s = Σ + γd * mean(diag) * I + γo * mean(offdiag) * (1 - I)
        eye = torch.eye(D, dtype=covs.dtype, device=covs.device).unsqueeze(0)  # (1, D, D)
        diag_mask = eye.bool()
        off_mask = ~diag_mask
        diag_mean = covs[:, diag_mask].view(C, D).mean(dim=1).view(C, 1, 1)    # (C,1,1)
        off_means = []
        for k in range(C):
            off_vals = covs[k][off_mask[0]].view(D, D-1)
            off_means.append(off_vals.mean())
        off_mean = torch.stack(off_means, dim=0).view(C, 1, 1)                 # (C,1,1)

        covs_shrunk = covs + shrink_gamma_diag * diag_mean * eye + shrink_gamma_off * off_mean * (1.0 - eye)

        # —— 相关系数化：R = D^{-1/2} Σ^s D^{-1/2}
        std = torch.sqrt(torch.clamp(torch.diagonal(covs_shrunk, dim1=1, dim2=2), min=eps))  # (C, D)
        inv_std = 1.0 / torch.clamp(std, min=eps)                                            # (C, D)
        # 逐类做外积归一：R_ij = Σ_ij / (σ_i σ_j)
        R = covs_shrunk * inv_std.unsqueeze(2) * inv_std.unsqueeze(1)                        # (C, D, D)

        # —— Cholesky 分解（稳定）
        L_list = []
        logdet_list = []
        for k in range(C):
            Lk = cholesky_manual_stable(R[k], reg=eps)      # (D, D), 下三角
            L_list.append(Lk)
            # log|R| = 2 * sum(log(diag(L)))
            logdet_list.append(2.0 * torch.log(torch.diag(Lk) + eps).sum())
        L = torch.stack(L_list, dim=0)                      # (C, D, D)
        logdet = torch.stack(logdet_list, dim=0)            # (C,)

        # 注册为 buffer（随 device 移动，且不参与梯度）
        self.register_buffer("mu", mus)                     # (C, D)
        self.register_buffer("std", std)                    # (C, D)
        self.register_buffer("L", L)                        # (C, D, D)
        self.register_buffer("logdetR", logdet)             # (C,)
        self.class_ids = class_ids
        self.add_logdet_bias = add_logdet_bias
        self.temperature = temperature
        self.use_tukey = use_tukey
        self.eps = eps

    @staticmethod
    def _tukey_power(x: torch.Tensor, lam: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
        # 与 Box-Cox 类似的幂变换；这里对正特征安全使用（如有负值建议先做非负化或改用 sign-preserving）
        if lam == 0.0:
            return torch.log(torch.clamp(x, min=eps))
        return (torch.clamp(x, min=eps) ** lam - 1.0) / lam

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, D) 特征。输出 logits: (N, C)，按 -0.5*Mahalanobis 距离（可选加 logdet 偏置）。
        """
        device = x.device
        mu = self.mu.to(device)           # (C, D)
        std = self.std.to(device)         # (C, D)
        L = self.L.to(device)             # (C, D, D)
        logdetR = self.logdetR.to(device) # (C,)

        if self.use_tukey:
            x = self._tukey_power(x, lam=0.5, eps=self.eps)

        # diff & whitening
        # diff: (N, C, D)
        diff = x.unsqueeze(1) - mu.unsqueeze(0)
        w = diff / torch.clamp(std.unsqueeze(0), min=self.eps)  # (N, C, D)

        # 逐类用 Cholesky 下三角解 L y = w^T，d = ||y||^2
        N, C, D = w.shape
        dists = []
        for k in range(C):
            wk = w[:, k, :].T                                 # (D, N)
            # 解 y: L y = wk
            yk = torch.linalg.solve_triangular(L[k], wk, upper=False)  # (D, N)
            dk = (yk ** 2).sum(dim=0)                         # (N,)
            dists.append(dk)
        dists = torch.stack(dists, dim=1)                     # (N, C)

        logits = -0.5 * dists / max(self.temperature, self.eps)  # (N, C)
        if self.add_logdet_bias:
            logits = logits - 0.5 * logdetR.unsqueeze(0)     # 加上 -0.5 log|R|

        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).argmax(dim=1)

    @property
    def num_classes(self) -> int:
        return len(self.class_ids)

class FeCAMHead(nn.Module):
    """
    基于每类 (mean, cov) 的非参判别头（FeCAM）：
      1) 协方差收缩（对角与非对角分别收缩）
      2) 相关系数化（对角=1）
      3) 稳定 Cholesky
      4) 前向以马氏距离打分：logits = -0.5 * d / T （可选 -0.5 * log|R| 偏置）

    依赖：GaussianStatistics, cholesky_manual_stable
    """
    def __init__(
        self,
        stats: Dict[int, GaussianStatistics],
        shrink_gamma_diag: float = 1e-4,
        shrink_gamma_off: float = 1e-4,
        add_logdet_bias: bool = False,
        temperature: float = 1.0,
        use_tukey: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert len(stats) > 0, "FeCAMHead: stats 为空。"

        class_ids = sorted(list(stats.keys()))
        D = stats[class_ids[0]].mean.numel()
        C = len(class_ids)

        # (C,D), (C,D,D)
        mus = torch.stack([stats[c].mean for c in class_ids], dim=0)
        covs = torch.stack([stats[c].cov for c in class_ids], dim=0)

        # —— 协方差收缩：Σ^s = Σ + γd * mean(diag) * I + γo * mean(offdiag) * (1 - I)
        eye = torch.eye(D, dtype=covs.dtype, device=covs.device).unsqueeze(0)  # (1,D,D)

        # 每类对角均值 → (C,1,1)
        diag_vals = torch.diagonal(covs, dim1=1, dim2=2)                      # (C, D)
        diag_mean = diag_vals.mean(dim=1, keepdim=True).unsqueeze(-1)         # (C,1,1)

        # 每类非对角均值 → (C,1,1)
        total_sum = covs.sum(dim=(1, 2))                                      # (C,)
        diag_sum = diag_vals.sum(dim=1)                                       # (C,)
        off_cnt = D * (D - 1)
        off_mean = ((total_sum - diag_sum) / max(off_cnt, 1)).view(C, 1, 1)   # (C,1,1)

        covs_shrunk = covs + shrink_gamma_diag * diag_mean * eye + shrink_gamma_off * off_mean * (1.0 - eye)

        # —— 相关系数化：R = D^{-1/2} Σ^s D^{-1/2}
        std = torch.sqrt(torch.clamp(torch.diagonal(covs_shrunk, dim1=1, dim2=2), min=eps))  # (C, D)
        inv_std = 1.0 / torch.clamp(std, min=eps)
        R = covs_shrunk * inv_std.unsqueeze(2) * inv_std.unsqueeze(1)                          # (C, D, D)

        # —— 稳定 Cholesky + logdet
        L_list, logdet_list = [], []
        for k in range(C):
            Lk = cholesky_manual_stable(R[k], reg=eps)  # (D, D)
            L_list.append(Lk)
            logdet_list.append(2.0 * torch.log(torch.diag(Lk) + eps).sum())
        L = torch.stack(L_list, dim=0)                  # (C, D, D)
        logdet = torch.stack(logdet_list, dim=0)        # (C,)

        # 注册 buffer（随 device 移动，不参与梯度）
        self.register_buffer("mu", mus)                 # (C, D)
        self.register_buffer("std", std)                # (C, D)
        self.register_buffer("L", L)                    # (C, D, D)
        self.register_buffer("logdetR", logdet)         # (C,)

        # 运行时配置
        self.class_ids = class_ids
        self.add_logdet_bias = add_logdet_bias
        self.temperature = temperature
        self.use_tukey = use_tukey
        self.eps = eps

    @staticmethod
    def _tukey_power(x: torch.Tensor, lam: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
        # 若你的特征可能有非正值，建议改为 sign-preserving 版本
        if lam == 0.0:
            return torch.log(torch.clamp(x, min=eps))
        return (torch.clamp(x, min=eps) ** lam - 1.0) / lam

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, D) 特征
        return: logits (N, C)
        """
        device = x.device
        mu = self.mu.to(device)           # (C, D)
        std = self.std.to(device)         # (C, D)
        L = self.L.to(device)             # (C, D, D)
        logdetR = self.logdetR.to(device) # (C,)

        if self.use_tukey:
            x = self._tukey_power(x, lam=0.5, eps=self.eps)

        # 标准化差分
        diff = x.unsqueeze(1) - mu.unsqueeze(0)                         # (N, C, D)
        w = diff / torch.clamp(std.unsqueeze(0), min=self.eps)          # (N, C, D)

        # 逐类解三角方程，得到马氏距离
        N, C, D = w.shape
        dists = []
        for k in range(C):
            wk = w[:, k, :].T                                           # (D, N)
            yk = torch.linalg.solve_triangular(L[k], wk, upper=False)   # (D, N)
            dk = (yk ** 2).sum(dim=0)                                   # (N,)
            dists.append(dk)
        dists = torch.stack(dists, dim=1)                               # (N, C)

        logits = -0.5 * dists / max(self.temperature, self.eps)
        if self.add_logdet_bias:
            logits = logits - 0.5 * logdetR.unsqueeze(0)
        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).argmax(dim=1)

    @property
    def num_classes(self) -> int:
        return len(self.class_ids)



class Drift_Compensator(object):
    def __init__(self, args):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.temp = 1.0
        self.gamma = args.get('gamma', 1e-4)
        self.auxiliary_data_size = args.get('auxiliary_data_size', 1024)
        self.args = args
        self.compensate = args.get('compensate', True)
        self.use_nonlinear = args.get('use_weaknonlinear', True)
        
        self.cached_Z = None
        self.aux_loader = None
        self.linear_transforms = {}
        self.linear_transforms_current_only = {}  # 仅使用当前任务数据的线性变换
        self.weaknonlinear_transforms = {}
        self.weaknonlinear_transforms_current_only = {}  # 仅使用当前任务数据的弱非线性变换

    @torch.no_grad()
    def extract_features_before_after(self, model_before, model_after, data_loader):
        model_before, model_after = model_before.to(self.device), model_after.to(self.device)
        model_before.eval()
        model_after.eval()

        feats_before, feats_after, targets = [], [], []
        for batch in data_loader:
            inputs, batch_targets = batch[0], batch[1]
            inputs = inputs.to(self.device)
            feats_before.append(model_before(inputs).cpu())
            feats_after.append(model_after(inputs).cpu())
            targets.append(batch_targets)

        feats_before = torch.cat(feats_before)
        feats_after = torch.cat(feats_after)
        targets = torch.cat(targets)
        return feats_before, feats_after, targets

    def compute_linear_transform(self, features_before: torch.Tensor, features_after: torch.Tensor, normalize: bool = True):
        """基于前/后特征构建线性补偿矩阵 W"""
        print("基于当前任务的前后特征构建线性补偿器(alpha_1-SLDC)")

        device = self.device
        features_before = features_before.to(device)
        features_after = features_after.to(device)

        if normalize:
            X = F.normalize(features_before, dim=1)
            Y = F.normalize(features_after, dim=1)
        else:
            X = features_before
            Y = features_after

        n_samples, dim = X.size()
        XTX = X.T @ X + self.gamma * torch.eye(dim, device=device)
        XTY = X.T @ Y
        W_global = torch.linalg.solve(XTX, XTY)  # (D, D)
        weight = math.exp(-n_samples / (self.temp * dim))
        W_global = (1 - weight) * W_global + weight * torch.eye(dim, device=device)

        feats_new_after_pred = features_before @ W_global
        feat_diffs = (features_after - features_before).norm(dim=1).mean().item()
        feat_diffs_pred = (features_after - feats_new_after_pred).norm(dim=1).mean().item()

        s = torch.linalg.svdvals(W_global)
        max_singular = s[0].item()
        min_singular = s[-1].item()

        print(
            f"仿射变换矩阵对角线元素均值：{W_global.diag().mean().item():.4f}，"
            f"融合权重：{weight:.4f}，样本数量：{n_samples}；"
            f"线性修正前差异：{feat_diffs:.4f}；修正后差异：{feat_diffs_pred:.4f}；"
            f"最大奇异值：{max_singular:.2f}；最小奇异值：{min_singular:.2f}")
        return W_global

    def compute_weaknonlinear_transform(self, features_before: torch.Tensor, features_after: torch.Tensor) -> WeakNonlinearTransform:
        """基于前/后特征训练弱非线性变换器"""
        print("基于当前任务的前后特征构建弱非线性补偿器")
        device = self.device
        features_before = features_before.to(device)
        features_after = features_after.to(device)
        
        transform = WeakNonlinearTransform(
            input_dim=features_before.size(1))
        
        transform.train(features_before, features_after)
        
        # 评估变换效果
        with torch.no_grad():
            transformed_features = transform.transform_features(features_before)
            feat_diffs = (features_after - features_before).norm(dim=1).mean().item()
            feat_diffs_nonlinear = (features_after - transformed_features).norm(dim=1).mean().item()
            print(f"非线性修正前差异：{feat_diffs:.4f}；修正后差异：{feat_diffs_nonlinear:.4f}")
        
        return transform

    def semantic_drift_compensation(self, old_stats_dict: Dict[int, GaussianStatistics], 
                                  features_before: torch.Tensor, features_after: torch.Tensor,
                                  labels: torch.Tensor, use_auxiliary: bool = False) -> Dict[int, GaussianStatistics]:
        """
        实现语义漂移补偿(SDC)方法
        基于论文中的方法：通过当前任务数据的漂移向量场来估计旧原型的漂移
        """
        if not old_stats_dict:
            return {}
            
        device = self.device
        features_before = features_before.to(device)
        features_after = features_after.to(device)
        
        # 计算当前任务数据的漂移向量
        drift_vectors = features_after - features_before  # (N_current, D)
        
        # 构建补偿后的统计量字典
        compensated_stats = {}
        
        for class_id, old_stat in old_stats_dict.items():
            old_prototype = old_stat.mean.to(device)  # (D,)
            
            # 计算当前任务特征与旧原型的距离
            distances = torch.cdist(features_before, old_prototype.unsqueeze(0))  # (N_current, 1)
            distances = distances.squeeze(1)  # (N_current,)
            
            # 使用高斯核计算权重（距离越近权重越大）
            sigma = 0.3  # 与论文中保持一致
            weights = torch.exp(-distances**2 / (2 * sigma**2))  # (N_current,)
            
            # 计算加权平均漂移向量
            weighted_drift = (weights.unsqueeze(1) * drift_vectors).sum(dim=0) / (weights.sum() + 1e-8)
            
            # 补偿原型：新原型 = 旧原型 + 加权漂移
            compensated_mean = old_prototype + weighted_drift
            
            # 保持单位协方差不变
            d = old_stat.mean.size(0)
            unit_cov = torch.eye(d, device=device, dtype=old_stat.mean.dtype)
            
            compensated_stats[class_id] = GaussianStatistics(
                compensated_mean.cpu(), unit_cov, old_stat.reg)
            
            if class_id == list(old_stats_dict.keys())[0]:  # 只打印第一个类的调试信息
                drift_norm = weighted_drift.norm().item()
                print(f"[SDC] Class {class_id}: drift norm = {drift_norm:.4f}, "
                      f"mean weight = {weights.mean().item():.4f}")
        
        return compensated_stats

    # --------------------- 构建 per-class 统计 ---------------------
    def _build_stats(self, features: torch.Tensor, labels: torch.Tensor) -> Dict[int, object]:
        """仅使用类内协方差"""
        device = features.device
        unique_labels = torch.unique(labels)
        stats = {}

        for lbl in unique_labels:
            mask = (labels == lbl)
            class_feats = features[mask]  # (N_c, D)

            class_mean = class_feats.mean(dim=0)  # (D,)

            if class_feats.size(0) >= 2:
                class_cov = torch.cov(class_feats.T) 
            else:
                d = class_feats.size(1)
                class_cov = torch.eye(d, device=device, dtype=class_feats.dtype) * 1e-4

            cid = int(lbl.item())
            stats[cid] = GaussianStatistics(class_mean, class_cov)
        return stats

    def build_all_variants(self, task_id: int, model_before: nn.Module, model_after: nn.Module, data_loader):
        feats_before, feats_after, labels = self.extract_features_before_after(
            model_before, model_after, data_loader)

        if self.cached_Z is None:
            self.cached_Z = torch.randn(50000, feats_after.size(1))

        W = None
        W_current_only = None
        weaknonlinear_transform = None
        weaknonlinear_transform_current_only = None
        
        if self.compensate and task_id > 1:
            # 使用当前任务数据构建线性补偿（不使用辅助数据）
            W_current_only = self.compute_linear_transform(feats_before, feats_after)
            self.linear_transforms_current_only[task_id] = W_current_only.cpu()
            
            # 使用当前任务数据构建弱非线性补偿（不使用辅助数据）
            if self.use_nonlinear:
                weaknonlinear_transform_current_only = self.compute_weaknonlinear_transform(feats_before, feats_after)
                self.weaknonlinear_transforms_current_only[task_id] = weaknonlinear_transform_current_only
            
            # 原有的使用辅助数据的补偿方法
            aux_loader = self.get_aux_loader(self.args)
            feats_aux_before, feats_aux_after, _ = self.extract_features_before_after(model_before, model_after, aux_loader)
            
            # 合并当前任务和辅助数据
            feats_before_combined = torch.cat([feats_before, feats_aux_before], dim=0)
            feats_after_combined = torch.cat([feats_after, feats_aux_after], dim=0)

            # 计算线性变换（使用辅助数据）
            W = self.compute_linear_transform(feats_before_combined, feats_after_combined)
            self.linear_transforms[task_id] = W.cpu()
            
            # 计算非线性变换（使用辅助数据）
            if self.use_nonlinear:
                weaknonlinear_transform = self.compute_weaknonlinear_transform(feats_before_combined, feats_after_combined)
                self.weaknonlinear_transforms[task_id] = weaknonlinear_transform

        # 构建基础统计
        stats = self._build_stats(features=feats_after, labels=labels)
        
        # 构建仅使用单位协方差的统计（用于LDC和SDC变体）
        stats_unit_cov = {}
        for cid, stat in stats.items():
            # 保持均值不变，但使用单位矩阵作为协方差
            d = stat.mean.size(0)
            unit_cov = torch.eye(d, device=stat.mean.device, dtype=stat.mean.dtype)
            stats_unit_cov[cid] = GaussianStatistics(stat.mean, unit_cov, stat.reg)

        # 初始化变体字典
        if not hasattr(self, 'variants') or len(self.variants) == 0:
            self.variants = {
                "SeqFT": {},                    # 原始特征
                "SeqFT without Cov": {},        # 新增：仅用类均值，协方差为单位矩阵
                "alpha_1-SLDC + ADE": {},       # 线性补偿（使用辅助数据）
                "alpha_2-SLDC + ADE": {},       # 弱非线性补偿（使用辅助数据）
                "alpha_1-SLDC": {},             # 线性补偿（仅使用当前任务数据）
                "alpha_2-SLDC": {},             # 弱非线性补偿（仅使用当前任务数据）
                "LDC": {},                      # 线性补偿+单位协方差（仅使用当前任务数据）
                "LDC + ADE": {},                # 线性补偿+单位协方差（使用辅助数据）
                "SDC": {},                      # 语义漂移补偿+单位协方差（仅使用当前任务数据）
                "SDC + ADE": {},                # 语义漂移补偿+单位协方差（使用辅助数据）
            }

        # 更新基础版本（无补偿）
        self.variants["SeqFT"].update(copy.deepcopy(stats))
        
        # 新增：SeqFT without Cov版本（仅使用单位协方差）
        self.variants["SeqFT without Cov"].update(copy.deepcopy(stats_unit_cov))
        
        if self.compensate and task_id > 1:
            # ================== 原有变体 ==================
            # 线性补偿版本（使用辅助数据）：变换旧任务统计量，添加新任务统计量
            if "alpha_1-SLDC + ADE" in self.variants:
                stats_compensated = self.transform_stats_with_W(self.variants['alpha_1-SLDC + ADE'], W)
            else:
                stats_compensated = {}
            self.variants["alpha_1-SLDC + ADE"] = stats_compensated
            self.variants["alpha_1-SLDC + ADE"].update(copy.deepcopy(stats))
            
            # 线性补偿版本（仅使用当前任务数据）
            if "alpha_1-SLDC" in self.variants:
                stats_compensated_current = self.transform_stats_with_W(self.variants['alpha_1-SLDC'], W_current_only)
            else:
                stats_compensated_current = {}
            self.variants["alpha_1-SLDC"] = stats_compensated_current
            self.variants["alpha_1-SLDC"].update(copy.deepcopy(stats))
            
            # 弱非线性补偿版本（使用辅助数据）
            if self.use_nonlinear and weaknonlinear_transform is not None:
                if "alpha_2-SLDC + ADE" in self.variants:
                    stats_weaknonlinear = weaknonlinear_transform.transform_stats(self.variants["alpha_2-SLDC + ADE"])
                else:
                    stats_weaknonlinear = {}
                self.variants["alpha_2-SLDC + ADE"] = stats_weaknonlinear
                self.variants["alpha_2-SLDC + ADE"].update(copy.deepcopy(stats))
            
            # 弱非线性补偿版本（仅使用当前任务数据）
            if self.use_nonlinear and weaknonlinear_transform_current_only is not None:
                if "alpha_2-SLDC" in self.variants:
                    stats_weaknonlinear_current = weaknonlinear_transform_current_only.transform_stats(self.variants["alpha_2-SLDC"])
                else:
                    stats_weaknonlinear_current = {}
                self.variants["alpha_2-SLDC"] = stats_weaknonlinear_current
                self.variants["alpha_2-SLDC"].update(copy.deepcopy(stats))
                
            # LDC版本（线性补偿+单位协方差，仅使用当前任务数据）
            if "LDC" in self.variants:
                # 对旧任务的单位协方差统计进行线性变换
                stats_ldc_compensated = self.transform_stats_with_W(self.variants['LDC'], W_current_only)
            else:
                stats_ldc_compensated = {}
            self.variants["LDC"] = stats_ldc_compensated
            self.variants["LDC"].update(copy.deepcopy(stats_unit_cov))  # 新任务使用单位协方差
            
            # LDC + ADE版本（线性补偿+单位协方差，使用辅助数据）
            if "LDC + ADE" in self.variants:
                # 对旧任务的单位协方差统计进行线性变换
                stats_ldc_ade_compensated = self.transform_stats_with_W(self.variants['LDC + ADE'], W)
            else:
                stats_ldc_ade_compensated = {}
            self.variants["LDC + ADE"] = stats_ldc_ade_compensated
            self.variants["LDC + ADE"].update(copy.deepcopy(stats_unit_cov))  # 新任务使用单位协方差
            
            # ================== 新增SDC变体 ==================
            # SDC版本（语义漂移补偿+单位协方差，仅使用当前任务数据）
            if "SDC" in self.variants:
                # 使用SDC方法补偿旧任务统计量
                stats_sdc_compensated = self.semantic_drift_compensation(
                    self.variants['SDC'], feats_before, feats_after, labels, use_auxiliary=False)
            else:
                stats_sdc_compensated = {}
            self.variants["SDC"] = stats_sdc_compensated
            self.variants["SDC"].update(copy.deepcopy(stats_unit_cov))  # 新任务使用单位协方差
            
            # SDC + ADE版本（语义漂移补偿+单位协方差，使用辅助数据）
            if "SDC + ADE" in self.variants:
                # 使用SDC方法补偿旧任务统计量（使用辅助数据）
                stats_sdc_ade_compensated = self.semantic_drift_compensation(
                    self.variants['SDC + ADE'], feats_before_combined, feats_after_combined, labels, use_auxiliary=True)
            else:
                stats_sdc_ade_compensated = {}
            self.variants["SDC + ADE"] = stats_sdc_ade_compensated
            self.variants["SDC + ADE"].update(copy.deepcopy(stats_unit_cov))  # 新任务使用单位协方差
            
        else:
            # 任务1：所有变体都直接使用当前统计量
            self.variants["alpha_1-SLDC + ADE"].update(copy.deepcopy(stats))
            self.variants["alpha_2-SLDC + ADE"].update(copy.deepcopy(stats))
            self.variants["alpha_1-SLDC"].update(copy.deepcopy(stats))
            self.variants["alpha_2-SLDC"].update(copy.deepcopy(stats))
            # 对于LDC变体，任务1使用单位协方差
            self.variants["LDC"].update(copy.deepcopy(stats_unit_cov))
            self.variants["LDC + ADE"].update(copy.deepcopy(stats_unit_cov))
            # 对于SDC变体，任务1使用单位协方差
            self.variants["SDC"].update(copy.deepcopy(stats_unit_cov))
            self.variants["SDC + ADE"].update(copy.deepcopy(stats_unit_cov))

        print(f"[INFO] Built distribution variants: {list(self.variants.keys())}, num classes: {len(stats)}")
        return self.variants

    def transform_stats_with_W(self, stats_dict: Dict[int, object], W: torch.Tensor) -> Dict[int, object]:
        """x' = x @ W  =>  μ' = μ @ W,  Σ' = W^T Σ W"""
        W = W.cpu()
        if stats_dict is None or len(stats_dict) == 0:
            return {}
        WT = W.t()
        out = {}
        for cid, stat in stats_dict.items():
            mean = stat.mean @ W             # (D,)
            cov = WT @ stat.cov @ W + 1e-3 * torch.eye(stat.cov.size(0), device=stat.cov.device) 
            new_stat = GaussianStatistics(mean, cov, reg=stat.reg)
            out[cid] = new_stat
        return out


    def train_classifier_with_cached_samples(self, fc: nn.Module, stats: Dict[int, object],
                                             epochs: int = 5) -> nn.Module:
        epochs = 5
        num_samples_per_class = 1024
        batch_size = max(32 * max(1, len(stats) // 10), 32)
        lr = 0.01

        fc = copy.deepcopy(fc).to(self.device)
        optimizer = torch.optim.SGD(fc.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs, eta_min=lr / 10)

        cached_Z = self.cached_Z.to(self.device)

        all_samples, all_targets = [], []
        class_means = []

        for class_id, gauss in stats.items():
            class_mean = gauss.mean.to(self.device)
            class_means.append(class_mean)

            L_matrix = gauss.L.to(self.device)

            # 从缓存噪声取一段
            start_idx = (int(class_id) * num_samples_per_class) % cached_Z.size(0)
            end_idx = start_idx + num_samples_per_class
            if end_idx > cached_Z.size(0):
                Z = torch.cat([cached_Z[start_idx:], cached_Z[: end_idx - cached_Z.size(0)]], dim=0)
            else:
                Z = cached_Z[start_idx:end_idx]

            # mean + Z @ L^T
            samples = class_mean + Z @ L_matrix.t()
            targets = torch.full((num_samples_per_class,), int(class_id), device=self.device)

            all_samples.append(samples)
            all_targets.append(targets)

        class_means = torch.stack(class_means)
        inputs = torch.cat(all_samples, dim=0).detach().clone()
        targets = torch.cat(all_targets, dim=0).detach().clone()

        for epoch in range(epochs):
            perm = torch.randperm(inputs.size(0), device=self.device)
            inputs_shuffled = inputs[perm]
            targets_shuffled = targets[perm]

            losses = 0.0
            num_samples = inputs.size(0)
            num_complete_batches = num_samples // batch_size

            for batch_idx in range(num_complete_batches + 1):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, num_samples)
                if start_idx >= end_idx:
                    continue

                inp = inputs_shuffled[start_idx:end_idx]
                tgt = targets_shuffled[start_idx:end_idx]

                optimizer.zero_grad()
                logits = fc(inp)

                loss = symmetric_cross_entropy_loss(logits, tgt)

                loss.backward()
                optimizer.step()
                losses += loss.item() * (end_idx - start_idx)

            loss_epoch = losses / num_samples
            if (epoch + 1) % 2 == 0:
                print(f"[INFO] Cached-sample classifier training: Epoch {epoch + 1}, Loss: {loss_epoch:.4f}")
            scheduler.step()

        return fc

    def refine_classifiers_from_variants(self, fc: nn.Module, epochs: int = 5) -> Dict[str, nn.Module]:
        """
        对 self.variants 里的每个分布各训练一个（线性）分类器；
        另外基于 SeqFT / alpha_1-SLDC / alpha_2-SLDC 的统计，构建并返回 3 个 FeCAM 头（nn.Module）。
        """
        assert hasattr(self, 'variants') and len(self.variants) > 0, "No variants found. Call build_all_variants first."
        out = {}

        # 1) 训练原有线性分类器
        for name, stats in self.variants.items():
            clf = self.train_classifier_with_cached_samples(fc, stats, epochs=epochs)
            out[name] = clf
        print(f"[INFO] Cached-sample linear classifiers trained: {len(self.variants)} variants.")

        # 2) 额外构建并返回 3 个 FeCAM 模块（使用相应变体的均值与协方差）
        fecam_cfg = {
            "shrink_gamma_diag": self.args.get("fecam_shrink_diag", 1e-4),
            "shrink_gamma_off":  self.args.get("fecam_shrink_off",  1e-4),
            "add_logdet_bias":   self.args.get("fecam_logdet_bias", False),
            "temperature":       self.args.get("fecam_temperature", 1.0),
            "use_tukey":         self.args.get("fecam_use_tukey", False),
        }

        def _maybe_add_fecam(variant_key: str, disp_name: str):
            stats = self.variants.get(variant_key, None)
            if stats is None or len(stats) == 0:
                print(f"[FeCAM] Skip {variant_key}: empty stats.")
                return
            head = FeCAMHead(
                stats=stats,
                shrink_gamma_diag=fecam_cfg["shrink_gamma_diag"],
                shrink_gamma_off=fecam_cfg["shrink_gamma_off"],
                add_logdet_bias=fecam_cfg["add_logdet_bias"],
                temperature=fecam_cfg["temperature"],
                use_tukey=fecam_cfg["use_tukey"],
            ).to(self.device).eval()
            out[disp_name] = head
            print(f"[INFO] Added {disp_name} to refined classifiers.")

        _maybe_add_fecam("SeqFT",          "FeCAM (SeqFT)")
        _maybe_add_fecam("alpha_1-SLDC",   "FeCAM (alpha_1-SLDC)")
        _maybe_add_fecam("alpha_2-SLDC",   "FeCAM (alpha_2-SLDC)")

        print(f"[INFO] Trained {len(self.variants)} linear heads + added 3 FeCAM heads (if available). Total: {len(out)}")
        return out


    def initialize_aux_loader(self, train_set):
        if self.aux_loader is not None:
            return self.aux_loader
        self.aux_loader = DataLoader(train_set, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
        return self.aux_loader
    
    def get_aux_loader(self, args):
        return self.aux_loader