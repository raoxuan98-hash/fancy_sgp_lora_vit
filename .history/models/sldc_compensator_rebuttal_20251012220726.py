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


class LDC_Compensator:
    """
    LDC 原型补偿器（纯用归一化类均值）。
    不再内部学习线性投影；由外部传入 α1-SLDC 的 W，用于只投影历史原型。
    """
    def __init__(self, args):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.args = args
        self.compensate = args.get('compensate', True)

        self.task_id = 0
        self.prototypes_current_space: Dict[int, torch.Tensor] = {}
        self.per_task_raw_means: Dict[int, Dict[int, torch.Tensor]] = {}

    def _compute_class_means(self, features: torch.Tensor, labels: torch.Tensor) -> Dict[int, torch.Tensor]:
        features = features.to(self.device)
        labels = labels.to(self.device)
        means = {}
        for lbl in torch.unique(labels):
            m = features[labels == lbl].mean(dim=0)
            means[int(lbl.item())] = F.normalize(m.unsqueeze(0), dim=1).squeeze(0).detach().cpu()
        return means

    def update_task_with_W(
        self,
        task_id: int,
        feats_after: torch.Tensor,
        labels: torch.Tensor,
        W_prev_to_curr = None,
    ):
        """
        外部传入 α1-SLDC 求得的线性变换 W（t-1 -> t）。
        规则：
          - task 1：仅计算并保存当前任务原型；
          - task t>1：若提供了 W，则仅对“历史原型”做 proto @ W 并归一化；随后合并“当前任务”的新原型（不投影）。
        """
        self.task_id = task_id

        # 当前任务在“after（当前）空间”的原型（不投影）
        current_means = self._compute_class_means(feats_after, labels)
        self.per_task_raw_means[task_id] = current_means

        if (not self.compensate) or task_id == 1 or (W_prev_to_curr is None):
            # 第一任务 / 禁用补偿 / 未提供 W：直接并入
            self.prototypes_current_space.update(current_means)
            return

        # 仅对“历史原型”进行投影到新空间
        W = W_prev_to_curr.detach().to(self.device)  # (D, D)
        updated = {}
        with torch.no_grad():
            for cid, proto in self.prototypes_current_space.items():
                p = proto.to(self.device).unsqueeze(0)    # (1, D)
                q = p @ W                                  # (1, D)
                q = F.normalize(q, dim=1).squeeze(0).cpu()
                updated[cid] = q
        self.prototypes_current_space = updated

        # 合并当前任务原型（不投影）
        self.prototypes_current_space.update(current_means)

    def get_prototypes(self) -> Dict[int, torch.Tensor]:
        return copy.deepcopy(self.prototypes_current_space)
        
    def get_classifier(self, feature_dim: int, num_classes: int) -> nn.Module:
        """
        返回基于当前空间原型的 NCM 分类器（LDC_Classifier）。
        """
        return LDC_Classifier(feature_dim, num_classes, self.get_prototypes(), self.device)

class LDC_Classifier(nn.Module):
    """
    非参数 NCM 分类器：用当前空间的类原型做余弦相似度分类。
    - 支持 num_classes >= len(prototypes) 的情形，缺失类别用大负数填充 logits 列。
    """
    def __init__(self, feature_dim: int, num_classes: int, prototypes: Dict[int, torch.Tensor], device: str):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.device = device
        self.prototypes = prototypes  # {cid: proto(D,) …}
        self._build_prototype_matrix()

    def _build_prototype_matrix(self):
        if not self.prototypes:
            self.register_buffer("prototype_matrix", torch.empty(0, self.feature_dim))
            self.class_ids = []
            return
        class_ids = sorted(self.prototypes.keys())
        self.class_ids = class_ids
        mats = [F.normalize(self.prototypes[cid].to(self.device).unsqueeze(0), dim=1).squeeze(0)
                for cid in class_ids]
        P = torch.stack(mats, dim=0)  # (C_have, D)
        self.register_buffer("prototype_matrix", P)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.prototype_matrix.numel() == 0:
            # 无原型时返回全零（或小随机），以免数值炸掉
            return torch.zeros(features.size(0), self.num_classes, device=features.device)

        x = F.normalize(features.to(self.device), dim=1)                 # (B, D)
        sim = x @ self.prototype_matrix.t()                              # (B, C_have)
        dist = 1.0 - sim                                                 # (B, C_have)
        logits_partial = -dist                                           # (B, C_have)

        # 组装成 (B, num_classes)，缺失类置为大负数
        B = features.size(0)
        logits_full = torch.full((B, self.num_classes),
                                 fill_value=-1e4, device=self.device, dtype=logits_partial.dtype)
        if len(self.class_ids) > 0:
            idx = torch.tensor(self.class_ids, device=self.device, dtype=torch.long)
            logits_full.index_copy_(dim=1, index=idx, source=logits_partial)
        return logits_full

    def update_prototypes(self, new_prototypes: Dict[int, torch.Tensor]):
        self.prototypes.update(new_prototypes)
        self._build_prototype_matrix()


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
        
        # 新增LDC补偿器
        self.ldc_compensator = LDC_Compensator(args)

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
            # —— α1-SLDC（仅当前数据）的 W ——（推荐给 LDC 使用）
            W_current_only = self.compute_linear_transform(feats_before, feats_after)
            self.linear_transforms_current_only[task_id] = W_current_only.cpu()

            # —— α2（弱非线性）的当前数据版（可选）——
            if self.use_nonlinear:
                weaknonlinear_transform_current_only = self.compute_weaknonlinear_transform(feats_before, feats_after)
                self.weaknonlinear_transforms_current_only[task_id] = weaknonlinear_transform_current_only

            # —— ADE 版（含辅助数据）的 W/非线性（供其它变体使用）——
            aux_loader = self.get_aux_loader(self.args)
            feats_aux_before, feats_aux_after, _ = self.extract_features_before_after(model_before, model_after, aux_loader)
            feats_before_combined = torch.cat([feats_before, feats_aux_before], dim=0)
            feats_after_combined  = torch.cat([feats_after,  feats_aux_after],  dim=0)

            W = self.compute_linear_transform(feats_before_combined, feats_after_combined)
            self.linear_transforms[task_id] = W.cpu()

            if self.use_nonlinear:
                weaknonlinear_transform = self.compute_weaknonlinear_transform(feats_before_combined, feats_after_combined)
                self.weaknonlinear_transforms[task_id] = weaknonlinear_transform

        # —— 基础统计（给非 LDC 变体用）——
        stats = self._build_stats(features=feats_after, labels=labels)

        # —— LDC：用 α1-SLDC 的 W（仅当前数据）来投影历史原型，再合并“当前任务原型” —— 
        if task_id == 1:
            self.ldc_compensator.update_task_with_W(task_id, feats_after, labels, W_prev_to_curr=None)
        else:
            self.ldc_compensator.update_task_with_W(task_id, feats_after, labels, W_prev_to_curr=W_current_only)

        # —— 构造 LDC 变体（把原型包成 GaussianStatistics 以复用后续训练接口）——
        ldc_prototypes = self.ldc_compensator.get_prototypes()
        ldc_stats = {}
        for cid, proto in ldc_prototypes.items():
            d = proto.size(0)
            cov = torch.eye(d) * 1e-4  # LDC 不关心协方差，给个小对角即可
            ldc_stats[cid] = GaussianStatistics(proto, cov, reg=1e-5)

        # —— 初始化/更新各变体字典 —— 
        if not hasattr(self, 'variants') or len(self.variants) == 0:
            self.variants = {
                "SeqFT": {},
                "alpha_1-SLDC + ADE": {},
                "alpha_2-SLDC + ADE": {},
                "alpha_1-SLDC": {},
                "alpha_2-SLDC": {},
                "LDC": {},  # 新
            }

        self.variants["SeqFT"].update(copy.deepcopy(stats))
        self.variants["LDC"] = ldc_stats

        if self.compensate and task_id > 1:
            # α1 + ADE
            stats_comp_ADE = self.transform_stats_with_W(self.variants['alpha_1-SLDC + ADE'], W)
            self.variants["alpha_1-SLDC + ADE"] = stats_comp_ADE
            self.variants["alpha_1-SLDC + ADE"].update(copy.deepcopy(stats))

            # α1（仅当前）
            stats_comp_cur = self.transform_stats_with_W(self.variants['alpha_1-SLDC'], W_current_only)
            self.variants["alpha_1-SLDC"] = stats_comp_cur
            self.variants["alpha_1-SLDC"].update(copy.deepcopy(stats))

            # α2 + ADE
            if self.use_nonlinear and weaknonlinear_transform is not None:
                stats_nl_ADE = weaknonlinear_transform.transform_stats(self.variants["alpha_2-SLDC + ADE"])
                self.variants["alpha_2-SLDC + ADE"] = stats_nl_ADE
                self.variants["alpha_2-SLDC + ADE"].update(copy.deepcopy(stats))

            # α2（仅当前）
            if self.use_nonlinear and weaknonlinear_transform_current_only is not None:
                stats_nl_cur = weaknonlinear_transform_current_only.transform_stats(self.variants["alpha_2-SLDC"])
                self.variants["alpha_2-SLDC"] = stats_nl_cur
                self.variants["alpha_2-SLDC"].update(copy.deepcopy(stats))
        else:
            # 任务1：所有非 LDC 变体直接存当前统计
            self.variants["alpha_1-SLDC + ADE"].update(copy.deepcopy(stats))
            self.variants["alpha_2-SLDC + ADE"].update(copy.deepcopy(stats))
            self.variants["alpha_1-SLDC"].update(copy.deepcopy(stats))
            self.variants["alpha_2-SLDC"].update(copy.deepcopy(stats))

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
        """对 self.variants 里的每个分布各训练一个分类器"""
        assert hasattr(self, 'variants') and len(self.variants) > 0, "No variants found. Call build_all_variants first."
        out = {}
        for name, stats in self.variants.items():
            if name == "LDC":
                # 对于LDC变体，使用专门的LDC分类器
                feature_dim = next(iter(stats.values())).mean.size(0)
                num_classes = len(stats)
                ldc_classifier = self.ldc_compensator.get_classifier(feature_dim, num_classes)
                out[name] = ldc_classifier
            else:
                # 其他变体使用原有的分类器训练方法
                clf = self.train_classifier_with_cached_samples(fc, stats, epochs=epochs)
                out[name] = clf
        print(f"[INFO] Trained {len(out)} classifiers from variants.")
        return out

    def initialize_aux_loader(self, train_set):
        if self.aux_loader is not None:
            return self.aux_loader
        self.aux_loader = DataLoader(train_set, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
        return self.aux_loader
    
    def get_aux_loader(self, args):
        return self.aux_loader


# 测试代码
if __name__ == "__main__":
    # 测试LDC补偿器
    args = {'compensate': True, 'use_weaknonlinear': True}
    
    # 创建模拟数据
    feature_dim = 128
    num_classes = 10
    num_samples = 100
    
    # 模拟特征和标签
    features_before = torch.randn(num_samples, feature_dim)
    features_after = torch.randn(num_samples, feature_dim)
    labels = torch.randint(0, num_classes, (num_samples,))
    
    # 模拟模型
    class MockModel(nn.Module):
        def __init__(self, feature_dim):
            super().__init__()
            self.feature_dim = feature_dim
            
        def forward(self, x):
            return torch.randn(x.size(0), self.feature_dim)
    
    model_before = MockModel(feature_dim)
    model_after = MockModel(feature_dim)
    
    # 模拟数据加载器
    dataset = TensorDataset(features_before, labels)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    class_list = list(range(num_classes))
    
    # 测试LDC补偿器
    ldc_compensator = LDC_Compensator(args)
    ldc_compensator.update_task(1, model_before, model_after, data_loader, class_list)
    
    # 获取分类器
    classifier = ldc_compensator.get_classifier(feature_dim, num_classes)
    
    # 测试分类器
    test_features = torch.randn(5, feature_dim)
    logits = classifier(test_features)
    print(f"LDC分类器输出形状: {logits.shape}")
    print("LDC补偿器测试完成!")