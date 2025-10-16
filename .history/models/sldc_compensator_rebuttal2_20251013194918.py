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
      
        # 在初始化时计算Cholesky分解
        self.L = cholesky_manual_stable(cov, reg=reg)
        self.L_weighted = None
        self.weighted_cov = None

    def to(self, device):
        """移动统计量到指定设备"""
        self.mean = self.mean.to(device)
        self.cov = self.cov.to(device)
        self.L = self.L.to(device)
        if self.L_weighted is not None:
            self.L_weighted = self.L_weighted.to(device)
        if self.weighted_cov is not None:
            self.weighted_cov = self.weighted_cov.to(device)
        return self

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
          
            new_mean = transformed_samples.mean(dim=0).cpu()
            new_cov = torch.cov(transformed_samples.T).cpu()
  
            transformed_stats[cid] = GaussianStatistics(new_mean, new_cov, stat.reg)
        return transformed_stats


class RegularizedGaussianClassifier(nn.Module):
    def __init__(self, stats_dict: Dict[int, 'GaussianStatistics'], 
                 class_priors: Dict[int, float] = None, 
                 mode: str = "qda",
                 reg_alpha: float = 0.1,
                 reg_type: str = "shrinkage"):
        super().__init__()
        self.mode = mode
        self.reg_alpha = reg_alpha
        self.reg_type = reg_type
        self.class_ids = sorted(stats_dict.keys())
        self.num_classes = len(self.class_ids)
        self.epsilon = 1e-2

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- 类先验 ---
        if class_priors is None:
            self.priors = {cid: 1.0 / self.num_classes for cid in self.class_ids}
        else:
            self.priors = class_priors

        # --- 计算全局统计 ---
        means_list = []
        covs_list = []
        for cid in self.class_ids:
            stat = stats_dict[cid]
            means_list.append(stat.mean.float().to(device))
            covs_list.append(stat.cov.float().to(device))

        means_tensor = torch.stack(means_list)  # [C, D]
        covs_tensor = torch.stack(covs_list)    # [C, D, D]
        global_mean = means_tensor.mean(dim=0)
        global_cov = covs_tensor.mean(dim=0)

        self.global_mean = nn.Parameter(global_mean.cpu(), requires_grad=False)
        self.global_cov = nn.Parameter(global_cov.cpu(), requires_grad=False)

        # --- 定义正则化函数 ---
        def regularize_covariance(cov: torch.Tensor) -> torch.Tensor:
            d = cov.size(-1)
            if self.reg_type == "shrinkage":
                return (1 - self.reg_alpha) * cov + self.reg_alpha * global_cov
            elif self.reg_type == "diagonal":
                diag_cov = torch.diag_embed(torch.diagonal(cov, dim1=-2, dim2=-1))
                return (1 - self.reg_alpha) * cov + self.reg_alpha * diag_cov
            elif self.reg_type == "combined":
                diag_global = torch.diag_embed(torch.diagonal(global_cov, dim1=-2, dim2=-1))
                target = 0.5 * global_cov + 0.5 * diag_global
                return (1 - self.reg_alpha) * cov + self.reg_alpha * target
            elif self.reg_type == "spherical":
                trace = torch.trace(cov) / d
                spherical = trace * torch.eye(d, device=cov.device)
                return (1 - self.reg_alpha) * cov + self.reg_alpha * spherical
            else:
                return cov

        # --- 批量正则化 + 求逆 + logdet（全 GPU）---
        covs_tensor = torch.stack([regularize_covariance(cov) for cov in covs_tensor])
        covs_tensor += self.epsilon * torch.eye(covs_tensor.size(-1), device=device)

        # GPU批量求逆与logdet
        cov_invs = torch.linalg.pinv(covs_tensor)       # [C, D, D]
        log_dets = torch.logdet(covs_tensor)            # [C]

        # --- 存储 ---
        self.means = nn.ParameterDict()
        self.regularized_covs = nn.ParameterDict()
        self.cov_invs = nn.ParameterDict()
        self.log_dets = nn.ParameterDict()

        for i, cid in enumerate(self.class_ids):
            self.means[str(cid)] = nn.Parameter(means_tensor[i].cpu(), requires_grad=False)
            self.regularized_covs[str(cid)] = nn.Parameter(covs_tensor[i].cpu(), requires_grad=False)
            self.cov_invs[str(cid)] = nn.Parameter(cov_invs[i].cpu(), requires_grad=False)
            self.log_dets[str(cid)] = nn.Parameter(log_dets[i].unsqueeze(0).cpu(), requires_grad=False)

        # --- LDA 模式共享协方差 ---
        if self.mode == "lda":
            shared_cov = covs_tensor.mean(dim=0)
            shared_cov_inv = torch.linalg.pinv(shared_cov)
            shared_log_det = torch.logdet(shared_cov)
            self.shared_cov = nn.Parameter(shared_cov.cpu(), requires_grad=False)
            self.shared_cov_inv = nn.Parameter(shared_cov_inv.cpu(), requires_grad=False)
            self.shared_log_det = nn.Parameter(shared_log_det.unsqueeze(0).cpu(), requires_grad=False)

        print(f"[INFO] GaussianClassifier initialized on {device} with {self.num_classes} classes (mode={mode})")

    # --- 判别函数 ---
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        batch_size = x.size(0)
        logits = torch.zeros(batch_size, self.num_classes, device=device)

        for idx, cid in enumerate(self.class_ids):
            if self.mode == "qda":
                logits[:, idx] = self._qda_discriminant(x, cid)
            else:
                logits[:, idx] = self._lda_discriminant(x, cid)
        return logits

    def _qda_discriminant(self, x: torch.Tensor, class_id: int) -> torch.Tensor:
        cid_str = str(class_id)
        mean = self.means[cid_str].to(x.device)
        cov_inv = self.cov_invs[cid_str].to(x.device)
        log_det = self.log_dets[cid_str].to(x.device)
        prior = torch.log(torch.tensor(self.priors[class_id], device=x.device))

        x_centered = x - mean.unsqueeze(0)
        mahalanobis = 0.5 * torch.sum(x_centered @ cov_inv * x_centered, dim=1)
        return -mahalanobis - 0.5 * log_det + prior

    def _lda_discriminant(self, x: torch.Tensor, class_id: int) -> torch.Tensor:
        cid_str = str(class_id)
        mean = self.means[cid_str].to(x.device)
        shared_cov_inv = self.shared_cov_inv.to(x.device)
        prior = torch.log(torch.tensor(self.priors[class_id], device=x.device))
        term1 = x @ shared_cov_inv @ mean
        term2 = 0.5 * mean @ shared_cov_inv @ mean
        return term1 - term2 + prior

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return torch.argmax(self.forward(x), dim=1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(x), dim=1)

  

class Drift_Compensator(object):
    def __init__(self, args):
        # 设备 & 基本超参
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.temp = 1.0
        self.gamma = args.get('gamma', 1e-4)
        self.auxiliary_data_size = args.get('auxiliary_data_size', 1024)
        self.args = args
        self.compensate = args.get('compensate', True)
        self.use_nonlinear = args.get('use_weaknonlinear', True)

        # === DPCR 关键超参（可调） ===
        self.energy = args.get('dpcr_energy', 0.95)
        self.r_cap  = args.get('dpcr_r_cap', 256)

        # 缓存 & 容器
        self.cached_Z = None
        self.aux_loader = None
        self.linear_transforms = {}
        self.linear_transforms_current_only = {}
        self.weaknonlinear_transforms = {}
        self.weaknonlinear_transforms_current_only = {}
        self.feature_dim = None

    @torch.no_grad()
    def extract_features_before_after(self, model_before, model_after, data_loader):
        """从同一批数据上抽取“前后模型”的特征用于补偿器估计。"""
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
        """基于前/后特征构建线性补偿矩阵 W（岭回归闭式解）。"""
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
        W_global = torch.linalg.solve(XTX, XTY)

        # 温和收缩到单位映射，避免数值炸裂
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

    def compute_weaknonlinear_transform(self, features_before: torch.Tensor, features_after: torch.Tensor) -> 'WeakNonlinearTransform':
        """基于前/后特征训练弱非线性变换器（Residual MLP）。"""
        print("基于当前任务的前后特征构建弱非线性补偿器")
        device = self.device
        features_before = features_before.to(device)
        features_after = features_after.to(device)

        transform = WeakNonlinearTransform(input_dim=features_before.size(1))
        transform.train(features_before, features_after)

        with torch.no_grad():
            transformed_features = transform.transform_features(features_before)
            feat_diffs = (features_after - features_before).norm(dim=1).mean().item()
            feat_diffs_nonlinear = (features_after - transformed_features).norm(dim=1).mean().item()
            print(f"非线性修正前差异：{feat_diffs:.4f}；修正后差异：{feat_diffs_nonlinear:.4f}")

        return transform

    def semantic_drift_compensation(self, old_stats_dict: Dict[int, 'GaussianStatistics'],
                                    features_before: torch.Tensor, features_after: torch.Tensor,
                                    labels: torch.Tensor, use_auxiliary: bool = False) -> Dict[int, 'GaussianStatistics']:
        if not old_stats_dict:
            return {}

        device = self.device
        features_before = features_before.to(device)
        features_after = features_after.to(device)

        drift_vectors = features_after - features_before
        compensated_stats = {}

        for class_id, old_stat in old_stats_dict.items():
            old_prototype = old_stat.mean.to(device)
            distances = torch.cdist(features_before, old_prototype.unsqueeze(0)).squeeze(1)
          
            # 关键修改：数值稳定的权重计算
            # 1. 自适应设置sigma - 基于距离的统计特性
            sigma = distances.median().item() * 0.5  # 使用中位数的一半作为sigma
            sigma = max(sigma, 1.0)  # 确保sigma不会太小
          
            # 2. 数值稳定的权重计算
            # 先计算负的平方距离（不除2sigma^2）
            neg_sq_distances = -distances**2
          
            # 找到最大值用于数值稳定性
            max_neg_sq = neg_sq_distances.max()
          
            # 使用log-sum-exp技巧的变体来避免数值下溢
            # weights = exp(neg_sq_distances/(2*sigma^2) - max_neg_sq/(2*sigma^2)) * exp(max_neg_sq/(2*sigma^2))
            scaled_neg_sq = (neg_sq_distances - max_neg_sq) / (2 * sigma**2)
            weights = torch.exp(scaled_neg_sq)
          
            # 现在weights的最大值是1，避免了数值下溢
            weight_sum = weights.sum()
          
            if weight_sum < 1e-20:  # 如果权重和仍然太小，使用均匀权重
                print(f"[SDC-WARN] Class {class_id}: weights too small ({weight_sum:.2e}), using uniform weights")
                weights = torch.ones_like(weights) / len(weights)
                weight_sum = 1.0
          
            weighted_drift = (weights.unsqueeze(1) * drift_vectors).sum(dim=0) / weight_sum
            compensated_mean = old_prototype + weighted_drift

            d = old_stat.mean.size(0)
            unit_cov = torch.eye(d, device=device, dtype=old_stat.mean.dtype)
            compensated_stats[class_id] = GaussianStatistics(compensated_mean.cpu(), unit_cov, old_stat.reg)

            if class_id == list(old_stats_dict.keys())[0]:
                drift_norm = weighted_drift.norm().item()
                mean_weight = weights.mean().item()
                print(f"[SDC] Class {class_id}: sigma = {sigma:.4f}")
                print(f"[SDC] distances: mean = {distances.mean():.4f}, max = {distances.max():.4f}")
                print(f"[SDC] weights: mean = {mean_weight:.6f}, sum = {weight_sum:.6f}, max = {weights.max():.6f}")
                print(f"[SDC] drift norm = {drift_norm:.4f}")

        return compensated_stats


    def _build_stats(self, features: torch.Tensor, labels: torch.Tensor) -> Dict[int, object]:
        """仅使用类内协方差构建（μ, Σ）。"""
        features = features.cpu()
        labels = labels.cpu()

        unique_labels = torch.unique(labels)
        stats = {}

        for lbl in unique_labels:
            mask = (labels == lbl)
            class_feats = features[mask]
            class_mean = class_feats.mean(dim=0)

            if class_feats.size(0) >= 2:
                class_cov = torch.cov(class_feats.T)
            else:
                d = class_feats.size(1)
                class_cov = torch.eye(d, dtype=class_feats.dtype) * 1e-4

            cid = int(lbl.item())
            stats[cid] = GaussianStatistics(class_mean, class_cov)
        return stats

    # =========================
    #       DPCR: 核心新增
    # =========================
    @torch.no_grad()
    def _compute_class_projectors(self, feats_before: torch.Tensor, labels: torch.Tensor):
        """
        对每个类 c，用未中心化协方差 Φ_c = X_c^T X_c 的特征分解，取累计能量>=self.energy 的主成分子空间 V_r，
        得到类行空间近似投影基 V_r（投影为 V_r V_r^T）。
        返回 {cid: V_r}。
        """
        X = F.normalize(feats_before, dim=1).cpu()
        y = labels.cpu()
        proj_basis = {}

        for cid in torch.unique(y).tolist():
            mask = (y == cid)
            Xc = X[mask]  # [Nc, d]
            if Xc.size(0) == 0:
                continue

            # 未中心化协方差（行空间）
            Phi = Xc.T @ Xc  # [d, d], PSD
            evals, evecs = torch.linalg.eigh(Phi)  # 升序
            evals = torch.clamp(evals, min=0)

            if evals.sum() <= 0:
                # 极端退化：取单位基的前 r_cap 列
                V_r = torch.eye(Xc.size(1))[:, :min(self.r_cap, Xc.size(1))]  # [d, r]
            else:
                w = evals / (evals.sum() + 1e-12)
                # 从最大特征值开始累计
                w_rev = torch.flip(w, dims=[0])
                cumsum_rev = torch.cumsum(w_rev, dim=0)
                # 选择最小 k 使累计能量 >= self.energy
                idx = int((cumsum_rev >= self.energy).nonzero(as_tuple=False)[0].item() + 1)
                k = min(idx, self.r_cap, Xc.size(1))
                V_r = evecs[:, -k:]  # 取最大的 k 个特征向量
            proj_basis[int(cid)] = V_r.contiguous()
        return proj_basis

    @torch.no_grad()
    def _build_classwise_W(self, W: torch.Tensor, feats_before: torch.Tensor, labels: torch.Tensor):
        """给定全局 W，构造 {cid: W_c = W @ (V_r V_r^T)}。"""
        if W is None:
            return {}
        W = W.detach().cpu()
        basis = self._compute_class_projectors(feats_before, labels)  # {cid: V_r}
        W_dict = {}
        for cid, V_r in basis.items():
            W_c = (W @ (V_r @ V_r.T)).contiguous()  # [d, d]
            W_dict[cid] = W_c
        return W_dict

    def transform_stats_with_W_classwise(self, stats_dict: Dict[int, object], W_dict: Dict[int, torch.Tensor]):
        """
        类专属线性变换：x' = x @ W_c  =>  μ' = μ @ W_c, Σ' = W_c^T Σ W_c
        """
        if not stats_dict or not W_dict:
            return {}
        out = {}
        for cid, stat in stats_dict.items():
            Wc = W_dict.get(cid, None)
            if Wc is None:
                # 若缺少该类的 Wc，则保持原样（也可退回全局 W）
                out[cid] = GaussianStatistics(stat.mean.clone(), stat.cov.clone(), stat.reg)
                continue
            Wc = Wc.to(stat.mean.device)
            WTc = Wc.t()
            mean = stat.mean @ Wc
            cov  = WTc @ stat.cov @ Wc
            cov  = cov + 1e-6 * torch.eye(cov.size(0), device=cov.device, dtype=cov.dtype)  # 稳定化
            out[cid] = GaussianStatistics(mean, cov, stat.reg)
        return out
    # =========================
    #     DPCR: 核心新增结束
    # =========================

    def build_all_variants(self, task_id: int, model_before: nn.Module, model_after: nn.Module, data_loader):
        """构建各类"分布变体"统计量（含 DPCR / DPCR + ADE）。"""
        feats_before, feats_after, labels = self.extract_features_before_after(
            model_before, model_after, data_loader)

        # 设置特征维度
        if self.feature_dim is None:
            self.feature_dim = feats_after.size(1)

        if self.cached_Z is None:
            self.cached_Z = torch.randn(50000, feats_after.size(1))

        W = None
        W_current_only = None
        weaknonlinear_transform = None
        weaknonlinear_transform_current_only = None

        if self.compensate and task_id > 1:
            # —— 仅用当前任务拟合 W
            W_current_only = self.compute_linear_transform(feats_before, feats_after)
            self.linear_transforms_current_only[task_id] = W_current_only.cpu()

            if self.use_nonlinear:
                weaknonlinear_transform_current_only = self.compute_weaknonlinear_transform(feats_before, feats_after)
                self.weaknonlinear_transforms_current_only[task_id] = weaknonlinear_transform_current_only

            # —— ADE：拼接辅助数据拟合更鲁棒的 W
            aux_loader = self.get_aux_loader(self.args)
            if aux_loader is not None:
                feats_aux_before, feats_aux_after, _ = self.extract_features_before_after(model_before, model_after, aux_loader)
                feats_before_combined = torch.cat([feats_before, feats_aux_before], dim=0)
                feats_after_combined = torch.cat([feats_after, feats_aux_after], dim=0)

                W = self.compute_linear_transform(feats_before_combined, feats_after_combined)
                self.linear_transforms[task_id] = W.cpu()

                if self.use_nonlinear:
                    weaknonlinear_transform = self.compute_weaknonlinear_transform(feats_before_combined, feats_after_combined)
                    self.weaknonlinear_transforms[task_id] = weaknonlinear_transform
            else:
                # 如果没有ADE数据，使用当前任务的W
                W = W_current_only
                weaknonlinear_transform = weaknonlinear_transform_current_only

        # 基于"当前特征（after）"计算原始统计量
        stats = self._build_stats(features=feats_after, labels=labels)
      
        # 单位协方差版本（用于 LDC 系列）
        stats_unit_cov = {}
        for cid, stat in stats.items():
            d = stat.mean.size(0)
            unit_cov = torch.eye(d, device=stat.mean.device, dtype=stat.mean.dtype)
            stats_unit_cov[cid] = GaussianStatistics(stat.mean, unit_cov, stat.reg)

        # 初始化 variants 字典（新增：SeqFT + LDA / SeqFT + QDA）
        if not hasattr(self, 'variants') or len(self.variants) == 0:
            self.variants = {
                "SeqFT": {},
                "SeqFT without Cov": {},
                "SeqFT + LDA": {},          # 新增
                "SeqFT + QDA": {},          # 新增
                "alpha_1-SLDC + ADE": {},
                "alpha_2-SLDC + ADE": {},
                "alpha_1-SLDC": {},
                "alpha_2-SLDC": {},
                "LDC": {},
                "LDC + ADE": {},
                "SDC": {},
                "SDC + ADE": {},
                # 新增：DPCR（类行空间收缩）
                "DPCR": {},
                "DPCR + ADE": {},
            }

        # 直接更新"当前任务"的统计量
        self.variants["SeqFT"].update(copy.deepcopy(stats))
        self.variants["SeqFT without Cov"].update(copy.deepcopy(stats_unit_cov))
        # 新增：与 SeqFT 一致（真实协方差）
        self.variants["SeqFT + LDA"].update(copy.deepcopy(stats))
        self.variants["SeqFT + QDA"].update(copy.deepcopy(stats))

        if self.compensate and task_id > 1:
            # ============ 线性补偿（全局 W） ============
            if "alpha_1-SLDC + ADE" in self.variants and W is not None:
                stats_compensated = self.transform_stats_with_W(self.variants['alpha_1-SLDC + ADE'], W)
            else:
                stats_compensated = {}
            self.variants["alpha_1-SLDC + ADE"] = stats_compensated
            self.variants["alpha_1-SLDC + ADE"].update(copy.deepcopy(stats))

            if "alpha_1-SLDC" in self.variants and W_current_only is not None:
                stats_compensated_current = self.transform_stats_with_W(self.variants['alpha_1-SLDC'], W_current_only)
            else:
                stats_compensated_current = {}
            self.variants["alpha_1-SLDC"] = stats_compensated_current
            self.variants["alpha_1-SLDC"].update(copy.deepcopy(stats))

            # ============ 弱非线性补偿 ============
            if self.use_nonlinear and weaknonlinear_transform is not None:
                if "alpha_2-SLDC + ADE" in self.variants:
                    stats_weaknonlinear = weaknonlinear_transform.transform_stats(self.variants["alpha_2-SLDC + ADE"])
                else:
                    stats_weaknonlinear = {}
                self.variants["alpha_2-SLDC + ADE"] = stats_weaknonlinear
                self.variants["alpha_2-SLDC + ADE"].update(copy.deepcopy(stats))

            if self.use_nonlinear and weaknonlinear_transform_current_only is not None:
                if "alpha_2-SLDC" in self.variants:
                    stats_weaknonlinear_current = weaknonlinear_transform_current_only.transform_stats(self.variants["alpha_2-SLDC"])
                else:
                    stats_weaknonlinear_current = {}
                self.variants["alpha_2-SLDC"] = stats_weaknonlinear_current
                self.variants["alpha_2-SLDC"].update(copy.deepcopy(stats))

            # ============ LDC（单位协方差） ============
            if "LDC" in self.variants and W_current_only is not None:
                stats_ldc_compensated = self.transform_stats_with_W(self.variants['LDC'], W_current_only)
            else:
                stats_ldc_compensated = {}
            self.variants["LDC"] = stats_ldc_compensated
            self.variants["LDC"].update(copy.deepcopy(stats_unit_cov))

            if "LDC + ADE" in self.variants and W is not None:
                stats_ldc_ade_compensated = self.transform_stats_with_W(self.variants['LDC + ADE'], W)
            else:
                stats_ldc_ade_compensated = {}
            self.variants["LDC + ADE"] = stats_ldc_ade_compensated
            self.variants["LDC + ADE"].update(copy.deepcopy(stats_unit_cov))

            # ============ SDC ============
            if "SDC" in self.variants:
                stats_sdc_compensated = self.semantic_drift_compensation(
                    self.variants['SDC'], feats_before, feats_after, labels, use_auxiliary=False)
            else:
                stats_sdc_compensated = {}
            self.variants["SDC"] = stats_sdc_compensated
            self.variants["SDC"].update(copy.deepcopy(stats_unit_cov))

            if "SDC + ADE" in self.variants and aux_loader is not None:
                # 这里用 ADE 合并后的特征进行 SDC
                stats_sdc_ade_compensated = self.semantic_drift_compensation(
                    self.variants['SDC + ADE'], feats_before_combined, feats_after_combined, labels, use_auxiliary=True)
            else:
                stats_sdc_ade_compensated = {}
            self.variants["SDC + ADE"] = stats_sdc_ade_compensated
            self.variants["SDC + ADE"].update(copy.deepcopy(stats_unit_cov))


            print("\n[DPCR] Performing real DPCR correction ...")
            dpcrr_corrected_stats = {}

            # 历史类来自 DPCR（上一轮构建时保存的旧类统计）
            if "DPCR" in self.variants and len(self.variants["DPCR"]) > 0:
                old_stats = self.variants["DPCR"]
                for cid, st in old_stats.items():
                    mu_old = st.mean.to(self.device).float()
                    cov_old = st.cov.to(self.device).float()
                    M2 = cov_old 
                    U_r = self._principal_subspace_from_cov(M2)
                    P_ic = W_current_only @ (U_r @ U_r.T)

                    # 校正旧类统计到 θ_t 域
                    mu_hat = mu_old @ P_ic
                    cov_hat = P_ic.T @ cov_old @ P_ic
                    cov_hat = 0.5 * (cov_hat + cov_hat.T) + 1e-6 * torch.eye(cov_hat.size(0), device=self.device)

                    dpcrr_corrected_stats[cid] = GaussianStatistics(mu_hat.cpu(), cov_hat.cpu())

            merged_stats = copy.deepcopy(dpcrr_corrected_stats)
            merged_stats.update(copy.deepcopy(stats))
            self.variants["DPCR"] = merged_stats
            print(f"[INFO] DPCR variant built with {len(merged_stats)} classes ")


            dpcrr_corrected_stats = {}
            if "DPCR + ADE" in self.variants and len(self.variants["DPCR + ADE"]) > 0:
                old_stats = self.variants["DPCR + ADE"]
                for cid, st in old_stats.items():
                    mu_old = st.mean.to(self.device).float()
                    cov_old = st.cov.to(self.device).float()
                    M2 = cov_old 
                    U_r = self._principal_subspace_from_cov(M2)
                    P_ic = W @ (U_r @ U_r.T)

                    # 校正旧类统计到 θ_t 域
                    mu_hat = mu_old @ P_ic
                    cov_hat = P_ic.T @ cov_old @ P_ic
                    cov_hat = 0.5 * (cov_hat + cov_hat.T) + 1e-6 * torch.eye(cov_hat.size(0), device=self.device)

                    dpcrr_corrected_stats[cid] = GaussianStatistics(mu_hat.cpu(), cov_hat.cpu())

            merged_stats = copy.deepcopy(dpcrr_corrected_stats)
            merged_stats.update(copy.deepcopy(stats))
            self.variants["DPCR + ADE "] = merged_stats
            print(f"[INFO] DPCR + ADE variant built with {len(merged_stats)} classes ")            


        else:
            # 首任务或不做补偿：所有变体都先接入"当前统计量"以建立键空间
            self.variants["alpha_1-SLDC + ADE"].update(copy.deepcopy(stats))
            self.variants["alpha_2-SLDC + ADE"].update(copy.deepcopy(stats))
            self.variants["alpha_1-SLDC"].update(copy.deepcopy(stats))
            self.variants["alpha_2-SLDC"].update(copy.deepcopy(stats))
            self.variants["LDC"].update(copy.deepcopy(stats_unit_cov))
            self.variants["LDC + ADE"].update(copy.deepcopy(stats_unit_cov))
            self.variants["SDC"].update(copy.deepcopy(stats_unit_cov))
            self.variants["SDC + ADE"].update(copy.deepcopy(stats_unit_cov))
            self.variants["DPCR"].update(copy.deepcopy(stats))
            self.variants["DPCR + ADE"].update(copy.deepcopy(stats))
            # 新增：首任务也建立 SeqFT + LDA/QDA 的键空间（与 SeqFT 一致）
            self.variants["SeqFT + LDA"].update(copy.deepcopy(stats))
            self.variants["SeqFT + QDA"].update(copy.deepcopy(stats))

        print(f"[INFO] Built distribution variants: {list(self.variants.keys())}, num classes: {len(stats)}")
      
        # 打印每个变体的类别数量
        for name, variant_stats in self.variants.items():
            print(f"  {name}: {len(variant_stats)} classes")
      
        return self.variants

    @staticmethod
    def _principal_subspace_from_cov(Phi: torch.Tensor, energy: float = 0.95, r_cap: int = 256) -> torch.Tensor:
        """
        从非中心二阶矩(或任意PSD矩阵) Phi 中提取累计能量>=energy 的主成分子空间 U_r
        返回 U_r: [d, r]
        """
        d = Phi.size(0)
        # 数值稳定化
        Phi = 0.5 * (Phi + Phi.T)
        evals, evecs = torch.linalg.eigh(Phi)  # 升序
        evals = torch.clamp(evals, min=0)
        if torch.sum(evals) <= 0:
            k = min(r_cap, d)
            return torch.eye(d, device=Phi.device, dtype=Phi.dtype)[:, :k]

        # 按能量从大到小取前k个
        evals_desc, idx = torch.sort(evals, descending=True)
        evecs_desc = evecs[:, idx]
        cumsum = torch.cumsum(evals_desc, dim=0)
        ratio = cumsum / (evals_desc.sum() + 1e-12)
        k = int((ratio >= energy).nonzero(as_tuple=False)[0].item() + 1)
        k = min(k, r_cap, d)
        U_r = evecs_desc[:, :k].contiguous()
        return U_r


    def transform_stats_with_W(self, stats_dict: Dict[int, object], W: torch.Tensor) -> Dict[int, object]:
        """全局线性变换：x' = x @ W  =>  μ' = μ @ W,  Σ' = W^T Σ W"""
        if W is None or stats_dict is None or len(stats_dict) == 0:
            return {}
        W = W.cpu()
        WT = W.t()
        out = {}
        for cid, stat in stats_dict.items():
            mean = stat.mean @ W
            cov = WT @ stat.cov @ W + 1e-3 * torch.eye(stat.cov.size(0), device=stat.cov.device)
            new_stat = GaussianStatistics(mean, cov, reg=stat.reg)
            out[cid] = new_stat
        return out

    def train_classifier_with_cached_samples(self, fc: nn.Module, stats: Dict[int, object],
                                             epochs: int = 5) -> nn.Module:
        """用缓存噪声从高斯统计采样进行 SGD 训练的线性分类器。"""
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

            start_idx = (int(class_id) * num_samples_per_class) % cached_Z.size(0)
            end_idx = start_idx + num_samples_per_class
            if end_idx > cached_Z.size(0):
                Z = torch.cat([cached_Z[start_idx:], cached_Z[: end_idx - cached_Z.size(0)]], dim=0)
            else:
                Z = cached_Z[start_idx:end_idx]

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

    def refine_classifiers_from_variants(self, fc: nn.Module, epochs: int = 5, 
                                        reg_alpha: float = 0.4, reg_type: str = "shrinkage") -> Dict[str, nn.Module]:
        """
        对 self.variants 中每个分布训练/构建一个分类器，根据变体名称使用不同的分类器类型。
        """
        assert hasattr(self, 'variants') and len(self.variants) > 0, "No variants found. Call build_all_variants first."

        out = {}
      
        for name, stats in self.variants.items():
            if len(stats) == 0:
                print(f"[WARNING] Variant '{name}' has no statistics, skipping...")
                continue
              
            try:
                class_priors = {cid: 1.0 / len(stats) for cid in stats.keys()}
              
                # LDC、SDC 系列 -> 使用 SGD 训练
                if name in ["LDC", "LDC + ADE", "SDC", "SDC + ADE"]:
                    clf = self.train_classifier_with_cached_samples(fc, stats, epochs=epochs)
                    out[name] = clf
                    print(f"[INFO] {name}: SGD with {len(stats)} classes")

                # SeqFT without Cov -> LDA（共享协方差）
                elif name == "SeqFT without Cov":
                    clf = RegularizedGaussianClassifier(
                        stats_dict=stats,
                        class_priors=class_priors,
                        mode="lda",               
                        reg_alpha=reg_alpha,
                        reg_type=reg_type
                    )
                    clf = clf.to(self.device)
                    out[name] = clf
                    print(f"[INFO] {name}: LDA with {len(stats)} classes")

                # DPCR 系列 -> QDA（类内各自协方差）
                elif name in ["DPCR", "DPCR + ADE"]:
                    clf = RegularizedGaussianClassifier(
                        stats_dict=stats,
                        class_priors=class_priors,
                        mode="qda",
                        reg_alpha=reg_alpha,
                        reg_type=reg_type
                    )
                    clf = clf.to(self.device)
                    out[name] = clf
                    print(f"[INFO] {name}: QDA with {len(stats)} classes")

                # 新增：SeqFT + LDA -> LDA（直接用当前统计量）
                elif name == "SeqFT + LDA":
                    clf = RegularizedGaussianClassifier(
                        stats_dict=stats,
                        class_priors=class_priors,
                        mode="lda",
                        reg_alpha=reg_alpha,
                        reg_type=reg_type
                    )
                    clf = clf.to(self.device)
                    out[name] = clf
                    print(f"[INFO] {name}: LDA with {len(stats)} classes")

                # 新增：SeqFT + QDA -> QDA
                elif name == "SeqFT + QDA":
                    clf = RegularizedGaussianClassifier(
                        stats_dict=stats,
                        class_priors=class_priors,
                        mode="qda",
                        reg_alpha=reg_alpha,
                        reg_type=reg_type
                    )
                    clf = clf.to(self.device)
                    out[name] = clf
                    print(f"[INFO] {name}: QDA with {len(stats)} classes")

                elif name == "SeqFT":
                    clf = self.train_classifier_with_cached_samples(fc, stats, epochs=epochs)
                    out[name] = clf
                    print(f"[INFO] {name}: SGD with {len(stats)} classes")

                # alpha_1/alpha_2 系列 -> 同时导出 LDA/QDA/SGD 三个头
                elif name in ["alpha_1-SLDC", "alpha_2-SLDC", "alpha_1-SLDC + ADE", "alpha_2-SLDC + ADE"]:
                    # LDA
                    lda_clf = RegularizedGaussianClassifier(
                        stats_dict=stats,
                        class_priors=class_priors,
                        mode="lda",
                        reg_alpha=reg_alpha,
                        reg_type=reg_type
                    ).to(self.device)
                    out[f"{name} (LDA)"] = lda_clf

                    # QDA
                    qda_clf = RegularizedGaussianClassifier(
                        stats_dict=stats,
                        class_priors=class_priors,
                        mode="qda",
                        reg_alpha=reg_alpha,
                        reg_type=reg_type
                    ).to(self.device)
                    out[f"{name} (QDA)"] = qda_clf

                    # SGD
                    sgd_clf = self.train_classifier_with_cached_samples(fc, stats, epochs=epochs)
                    out[f"{name} (SGD)"] = sgd_clf

                    print(f"[INFO] {name}: LDA/QDA/SGD with {len(stats)} classes")

                else:
                    print(f"[WARNING] Unknown variant '{name}', using SGD as default")
                    clf = self.train_classifier_with_cached_samples(fc, stats, epochs=epochs)
                    out[name] = clf

                # 简单运行一次前向，检查设备一致性
                with torch.no_grad():
                    test_input = torch.randn(2, list(stats.values())[0].mean.size(0), device=self.device)
                    if name in ["alpha_1-SLDC", "alpha_2-SLDC", "alpha_1-SLDC + ADE", "alpha_2-SLDC + ADE"]:
                        _ = out[f"{name} (LDA)"](test_input)
                        _ = out[f"{name} (QDA)"](test_input)
                        _ = out[f"{name} (SGD)"](test_input)
                    else:
                        _ = out[name](test_input)

            except Exception as e:
                print(f"[ERROR] Failed to create classifier for variant '{name}': {e}")
                print(f"[INFO] Falling back to SGD for variant '{name}'")
                clf = self.train_classifier_with_cached_samples(fc, stats, epochs=epochs)
                out[name] = clf

        print(f"[INFO] Created {len(out)} classifiers from variants.")

        for key, value in out.items():
            out[key] = value.cpu()
        return out


    def initialize_aux_loader(self, train_set):
        """外部传入一个 DataSet（如 replay/额外数据），用于 ADE 拟合 W。"""
        if self.aux_loader is not None:
            return self.aux_loader
        self.aux_loader = DataLoader(train_set, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
        return self.aux_loader

    def get_aux_loader(self, args):
        """若未初始化，将返回 None；使用前请先调用 initialize_aux_loader。"""
        return self.aux_loader
