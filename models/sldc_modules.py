# -*- coding: utf-8 -*-
import math
import copy
import numpy as np
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.cluster import KMeans
<<<<<<< HEAD
from torch.utils.data import DataLoader, TensorDataset
=======

>>>>>>> 4816499fc2b904e2d81571b705b8f392a7bd6601

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def cholesky_manual_stable(matrix: torch.Tensor, reg: float = 1e-5) -> torch.Tensor:
    """稳定的手动 Cholesky 分解，自动添加正则化 (matrix must be SPD-ish)"""
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
    """对称交叉熵损失：CE + RCE"""
    pred = F.softmax(logits, dim=1)
    pred = torch.clamp(pred, min=1e-7, max=1.0)

    label_one_hot = F.one_hot(targets, num_classes=pred.size(1)).float().to(pred.device)
    label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)

    ce_loss = -(label_one_hot * torch.log(pred)).sum(dim=1).mean()
    rce_loss = -(pred * torch.log(label_one_hot)).sum(dim=1).mean()

    return sce_a * ce_loss + sce_b * rce_loss


# -----------------------------------------------------------------------------
# Statistics containers
# -----------------------------------------------------------------------------
class GaussianStatistics:
    """单均值高斯：mean 为 (D,)，cov 为 (D, D)"""

    def __init__(self, mean: torch.Tensor, cov: torch.Tensor, reg: float = 1e-5):
        # mean 需要是 (D,)
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

    def sample(self, n_samples: int, cached_eps: torch.Tensor = None, use_weighted_cov: bool = True) -> torch.Tensor:
        device = self.mean.device
        d = self.mean.size(0)

        if cached_eps is None:
            eps = torch.randn(n_samples, d, device=device)
        else:
            eps = cached_eps.to(device)
            n_samples = eps.size(0)

        L = self.L_weighted if (use_weighted_cov and self.L_weighted is not None) else self.L
        samples = self.mean.unsqueeze(0) + eps @ L.t()  # (n_samples, D)
        return samples


class MultiMeanGaussianStatistics:
    """
    多均值（K 个分量），共享同一协方差。
    means: (K, D), probs: (K,), cov: (D, D)
    """

    def __init__(self, means: torch.Tensor, cov: torch.Tensor, probs: torch.Tensor, reg: float = 1e-5):
        if isinstance(means, list):
            means = torch.stack(means, dim=0)
        assert means.dim() == 2, "MultiMeanGaussianStatistics.means should be (K, D)"
        assert probs.dim() == 1 and probs.size(0) == means.size(0), "probs shape must be (K,) and match K"
        self.means = means
        self.probs = probs
        self.cov = cov
        self.reg = reg

        self.L = cholesky_manual_stable(cov, reg=reg)
        self.L_weighted = None
        self.weighted_cov = None

    def initialize_weighted_covariance(self, weighted_cov: torch.Tensor):
        self.weighted_cov = weighted_cov
        self.L_weighted = cholesky_manual_stable(weighted_cov, reg=self.reg)

    def sample(self, n_samples: int, cached_eps: torch.Tensor = None, use_weighted_cov: bool = True) -> torch.Tensor:
        device = self.means.device
        d = self.means.size(1)

        if cached_eps is None:
            eps = torch.randn(n_samples, d, device=device)
        else:
            eps = cached_eps.to(device)
            n_samples = eps.size(0)

        indices = torch.multinomial(self.probs, n_samples, replacement=True)  # (n_samples,)
        selected_means = self.means[indices]  # (n_samples, d)
        L = self.L_weighted if (use_weighted_cov and self.L_weighted is not None) else self.L
        noise = eps @ L.t()
        samples = selected_means + noise
        return samples


# -----------------------------------------------------------------------------
# Covariance smoothing across classes
# -----------------------------------------------------------------------------
def compute_weighted_covariances(stats_dict: Dict[int, object], temperature: float = 1.0, device: str = "cpu"):
    """
    基于类中心相似度，对协方差做加权平滑：Σ_i^w = Σ_j a_ij Σ_j 。
    对单均值使用 mean，对多均值使用其均值的均值。
    """
    if len(stats_dict) == 0:
        return stats_dict

    class_ids = list(stats_dict.keys())
    # 正确的类型判断
    if isinstance(stats_dict[class_ids[0]], MultiMeanGaussianStatistics):
        means = torch.stack([stats_dict[cid].means.mean(dim=0).to(device) for cid in class_ids])
    else:
        means = torch.stack([stats_dict[cid].mean.to(device) for cid in class_ids])

    covs = torch.stack([stats_dict[cid].cov.to(device) for cid in class_ids])  # (C, D, D)
    means_norm = F.normalize(means, dim=1)
    sim_matrix = torch.mm(means_norm, means_norm.t())  # (C, C)
    weights = F.softmax(sim_matrix / temperature, dim=1)  # (C, C)
    weighted_covs = torch.einsum('ij,jkl->ikl', weights, covs)  # (C, D, D)

    for idx, cid in enumerate(class_ids):
        stats_dict[cid].initialize_weighted_covariance(weighted_covs[idx])

    print(f"[INFO] Updated weighted covariances for {len(class_ids)} classes using mean similarity weighting.")
    return stats_dict


# -----------------------------------------------------------------------------
# Drift Compensator
# -----------------------------------------------------------------------------
class Drift_Compensator(object):
    def __init__(self, args):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # 旧接口保留（可选）
        self.original_stats = {}
        self.multi_mean_stats = {}
        self.original_stats_shared_covariance = {}
        self.multi_mean_stats_shared_covariance = {}

        self.linear_original_stats = {}
        self.linear_multi_mean_stats = {}
        self.linear_original_stats_shared_covariance = {}
        self.linear_multi_mean_stats_shared_covariance = {}

        # 参数
        self.n_clusters_per_class = args.get('n_class_clusters', 3)
        self.n_task_clusters = args.get('n_task_clusters', 3)
        self.cov_weighting_temperature = args.get('cov_weighting_temperature', 0.2)

        self.alpha_t = args.get('alpha_t', 1.0)
        self.gamma_1 = args.get('gamma_1', 1e-4)
        self.auxiliary_data_size = args.get('auxiliary_data_size', 5000)
        self.args = args
        self.compensate = args.get('compensate', True)

        self.covariance_sharing_mode = args.get('covariance_sharing_mode', 'per_class')
        assert self.covariance_sharing_mode in {'per_class', 'task_wise', 'global'}, \
            "covariance_sharing_mode must be one of: 'per_class', 'task_wise', 'global'"

        self.cached_Z = None
        self.aux_loader = None
        self._task_cov_cache = None  # (task_means, task_covs)
        self.variants = {}           # 8 种分布变体存放处

    # --------------------- 数据/工具 ---------------------
    def _enforce_min_distance(self, centers: np.ndarray, min_dist: float = 0.5) -> np.ndarray:
        """确保聚类中心之间最小距离，避免坍缩"""
        centers = centers.copy()
        n = len(centers)
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < min_dist:
                    direction = centers[j] - centers[i]
                    direction /= (np.linalg.norm(direction) + 1e-8)
                    centers[j] = centers[i] + direction * min_dist
        return centers

    @torch.no_grad()
    def extract_features_before_after(self, model_before, model_after, data_loader):
        model_before, model_after = model_before.to(self.device), model_after.to(self.device)
        model_before.eval()
        model_after.eval()

        feats_before, feats_after, targets = [], [], []
        # 期望 data_loader 产出: (batch_indices, inputs, batch_targets)
        for batch in data_loader:
            if len(batch) == 3:
                _, inputs, batch_targets = batch
            else:
                # 兼容仅 (inputs, targets)
                inputs, batch_targets = batch
            inputs = inputs.to(self.device)
            feats_before.append(model_before(inputs).cpu())
            feats_after.append(model_after(inputs).cpu())
            targets.append(batch_targets)

        feats_before = torch.cat(feats_before)
        feats_after = torch.cat(feats_after)
        targets = torch.cat(targets)
        return feats_before, feats_after, targets

    @torch.no_grad()
    def extract_features_before_after_for_auxiliary_data(self, model_before, model_after, data_loader):
        model_before, model_after = model_before.to(self.device), model_after.to(self.device)
        model_before.eval()
        model_after.eval()

        feats_before, feats_after = [], []
        # 期望 aux_loader 产出: (inputs, batch_targets) 或 (inputs, )
        for batch in data_loader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 1:
                inputs = batch[0]
            else:
                inputs = batch
            inputs = inputs.to(self.device)
            feats_before.append(model_before(inputs).cpu())
            feats_after.append(model_after(inputs).cpu())

        feats_before = torch.cat(feats_before)
        feats_after = torch.cat(feats_after)
        return feats_before, feats_after

    def compute_clusters(self, features: torch.Tensor, min_distance: float = 0.5, compute_covariances: bool = True):
        """对特征做 KMeans，返回簇均值（和可选协方差）"""
        features_np = features.cpu().numpy()
        kmeans = KMeans(n_clusters=self.n_task_clusters, n_init=20, init='k-means++').fit(features_np)
        centers = kmeans.cluster_centers_
        centers = self._enforce_min_distance(centers, min_distance)
        labels = kmeans.labels_

        cluster_means = []
        cluster_covs = []
        for i in range(self.n_task_clusters):
            cluster_mask = labels == i
            cluster_features = torch.from_numpy(features_np[cluster_mask])
            cluster_mean = cluster_features.mean(dim=0)
            cluster_means.append(cluster_mean)
            if compute_covariances:
                cluster_cov = torch.cov(cluster_features.T)
                cluster_covs.append(cluster_cov)

        if compute_covariances:
            return torch.stack(cluster_means), torch.stack(cluster_covs)
        else:
            return torch.stack(cluster_means)

    # --------------------- 线性变换 ---------------------
    def compute_linear_transform(self, features_before: torch.Tensor, features_after: torch.Tensor, normalize: bool = True):
        """基于前/后特征构建线性补偿矩阵 W（无偏置）"""
        print("基于当前任务的前后特征构建仿射补偿器(alpha_1-SLDC-Affine) - 无偏置版本")
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
<<<<<<< HEAD
        XTX = X.T @ X + 1e-4 * torch.eye(dim, device=device)
=======
        XTX = X.T @ X + self.gamma_1 * torch.eye(dim, device=device)
>>>>>>> 4816499fc2b904e2d81571b705b8f392a7bd6601
        XTY = X.T @ Y
        W_global = torch.linalg.solve(XTX, XTY)  # (D, D)
        weight = math.exp(-n_samples / (self.alpha_t * dim))
        W_global = (1 - weight) * W_global + weight * torch.eye(dim, device=device)

        feats_new_after_pred = features_before @ W_global
        feat_diffs = (features_after - features_before).norm(dim=1).mean().item()
        feat_diffs_pred = (features_after - feats_new_after_pred).norm(dim=1).mean().item()

        s = torch.linalg.svdvals(W_global)
        max_singular = s[0].item()
        min_singular = s[-1].item()

<<<<<<< HEAD
        print( 
            f"仿射变换矩阵（非对称）对角线元素均值：{W_global.diag().mean().item():.4f}，"
            f"融合权重：{weight:.4f}，样本数量：{n_samples}；"
            f"线性修正前差异：{feat_diffs:.4f}；修正后差异：{feat_diffs_pred:.4f}；"
            f"最大奇异值：{max_singular:.2f}；最小奇异值：{min_singular:.2f}")
         
=======
        print(
            f"仿射变换矩阵（非对称）对角线元素均值：{W_global.diag().mean().item():.4f}，"
            f"融合权重：{weight:.4f}，样本数量：{n_samples}；"
            f"线性修正前差异：{feat_diffs:.4f}；修正后差异：{feat_diffs_pred:.4f}；"
            f"最大奇异值：{max_singular:.2f}；最小奇异值：{min_singular:.2f}"
        )
>>>>>>> 4816499fc2b904e2d81571b705b8f392a7bd6601
        return W_global

    def _transform_stats_with_W(self, stats_dict: Dict[int, object], W: torch.Tensor) -> Dict[int, object]:
        """x' = x @ W  =>  μ' = μ @ W,  Σ' = W^T Σ W"""
        W = W.cpu()
        if stats_dict is None or len(stats_dict) == 0:
            return {}
        WT = W.t()
        out = {}
        for cid, stat in stats_dict.items():
            if isinstance(stat, MultiMeanGaussianStatistics):
                means = stat.means @ W           # (K, D)
<<<<<<< HEAD
                cov = WT @ stat.cov @ W +        # (D, D)
                new_stat = MultiMeanGaussianStatistics(means, cov, probs=stat.probs, reg=stat.reg)
            else:  # GaussianStatistics
                mean = stat.mean @ W             # (D,)
                cov = WT @ stat.cov @ W
                new_stat = GaussianStatistics(mean, cov, reg=stat.reg)
            out[cid] = new_stat
        return out
    


    # --------------------- Task-wise 协方差缓存 ---------------------
    def _get_task_cov_bank(self, features: torch.Tensor):
        """返回 (cluster_means, cluster_covs)，并缓存避免重复 KMeans。"""
        if self._task_cov_cache is not None:
            return self._task_cov_cache
        task_cluster_means, task_cluster_covs = self.compute_clusters(
            features, min_distance=0.5, compute_covariances=True
        )
        self._task_cov_cache = (task_cluster_means.to(features.device), task_cluster_covs.to(features.device))
        return self._task_cov_cache

    # --------------------- 统一构建统计（单/多均值 × 是否用 task-wise 协方差） ---------------------
    def _build_stats(self, features: torch.Tensor, labels: torch.Tensor,
                     multi_means: bool, use_task_cov: bool) -> Dict[int, object]:
        """
        multi_means: False => 单均值；True => 类内多均值（共享同一个类协方差）
        use_task_cov: False => 类内协方差由该类样本估计；True => 用 task-wise 聚类协方差按相似度加权得到类协方差
        """
        device = features.device
        unique_labels = torch.unique(labels)
        stats = {}

        if use_task_cov:
            task_means, task_covs = self._get_task_cov_bank(features)  # (T, D), (T, D, D)

        for lbl in unique_labels:
            mask = (labels == lbl)
            class_feats = features[mask]  # (N_c, D)

            # 1) 计算均值（单/多）
            if multi_means:
                # 多均值：只取中心，不计算每簇协方差（类内共享一个 Σ_c）
                # 使用类内 KMeans 个数为 args 中的 n_class_clusters
                kmeans = KMeans(n_clusters=self.n_clusters_per_class, n_init=20, init='k-means++').fit(
                    class_feats.cpu().numpy()
                )
                centers = self._enforce_min_distance(kmeans.cluster_centers_, min_dist=0.1)
                class_means = torch.from_numpy(centers).to(device=device, dtype=class_feats.dtype)  # (K, D)
            else:
                class_means = class_feats.mean(dim=0, keepdim=True)  # (1, D)

            # 2) 计算协方差（类内 or task-wise）
            if use_task_cov:
                query = class_means.mean(dim=0, keepdim=True)  # (1, D)
                sim = F.normalize(query, dim=1) @ F.normalize(task_means, dim=1).t()  # (1, T)
                attn = sim.squeeze(0).softmax(dim=-1)  # (T,)
                class_cov = sum(attn[i] * task_covs[i] for i in range(task_covs.size(0)))
            else:
                class_cov = torch.cov(class_feats.T)

            # 3) 打包
            cid = int(lbl.item())
            if multi_means:
                K = class_means.size(0)
                probs = torch.full((K,), 1.0 / K, device=device)
                stats[cid] = MultiMeanGaussianStatistics(class_means, class_cov, probs=probs)
            else:
                stats[cid] = GaussianStatistics(class_means, class_cov)  # ctor 会 squeeze 到 (D,)
        return stats

    # --------------------- 8 种分布一把生成 ---------------------
    def build_all_variants(self, task_id: int, model_before: nn.Module, model_after: nn.Module, data_loader):
        feats_before, feats_after, labels = self.extract_features_before_after(
            model_before, model_after, data_loader)

        # 初始化缓存噪声
        if self.cached_Z is None:
            self.cached_Z = torch.randn(50000, feats_after.size(1))

        # 线性补偿矩阵
        W = None
        if self.compensate and task_id > 0:
            aux_loader = self.get_aux_loader(self.args)
            feats_aux_before, feats_aux_after = self.extract_features_before_after_for_auxiliary_data(
                model_before, model_after, aux_loader)
            feats_b = torch.cat([feats_before, feats_aux_before], dim=0)
            feats_a = torch.cat([feats_after, feats_aux_after], dim=0)
            W = self.compute_linear_transform(feats_b, feats_a)

        self._task_cov_cache = None
        _ = self._get_task_cov_bank(feats_after)

        # 组合开关
        configs = []
        for multi_means in (False, True):
            for use_task_cov in (False, True):
                for use_linear in (False, True):
                    configs.append((multi_means, use_task_cov, use_linear))

        variants = {}
        for m, t, l in configs:
            key = f"mm{int(m)}_tc{int(t)}_lin{int(l)}"
            # 1) 在原空间构建
            base_stats = self._build_stats(
                features=feats_after, labels=labels,
                multi_means=m, use_task_cov=t
            )
            # 2) 线性变换
            if l and W is not None:
                stats = self._transform_stats_with_W(base_stats, W)
            else:
                stats = base_stats

            # 3) 协方差语义平滑（如果你希望只对部分变体做，可另设开关）
            # stats = compute_weighted_covariances(
            #     stats, temperature=self.cov_weighting_temperature, device=self.device
            # )
            
            variants[key] = stats

        self.variants = variants
        print(f"[INFO] Built {len(variants)} distribution variants: {list(variants.keys())}")
        return variants

=======
                cov = WT @ stat.cov @ W + 1e-3 * torch.eye(stat.cov.size(0), device=stat.cov.device)    # (D, D)
                new_stat = MultiMeanGaussianStatistics(means, cov, probs=stat.probs, reg=stat.reg)
            else:
                mean = stat.mean @ W             # (D,)
                cov = WT @ stat.cov @ W + 1e-3 * torch.eye(stat.cov.size(0), device=stat.cov.device) 
                new_stat = GaussianStatistics(mean, cov, reg=stat.reg)
            out[cid] = new_stat
        return out

    # --------------------- Task-wise 协方差缓存 ---------------------
    def _get_task_cov_bank(self, features: torch.Tensor):
        if self._task_cov_cache is not None:
            return self._task_cov_cache
        task_cluster_means, task_cluster_covs = self.compute_clusters(
            features, min_distance=0.5, compute_covariances=True)
        self._task_cov_cache = (task_cluster_means.to(features.device), task_cluster_covs.to(features.device))
        return self._task_cov_cache

    def _build_stats(self, features: torch.Tensor, labels: torch.Tensor,
                    multi_means: bool, use_task_cov: bool) -> Dict[int, object]:
        device = features.device
        unique_labels = torch.unique(labels)
        stats = {}

        if use_task_cov:
            task_means, task_covs = self._get_task_cov_bank(features)  # (T, D), (T, D, D)

        for lbl in unique_labels:
            mask = (labels == lbl)
            class_feats = features[mask]  # (N_c, D)

            if multi_means:
                kmeans = KMeans(
                    n_clusters=self.n_clusters_per_class,
                    n_init=20,
                    init='k-means++'
                ).fit(class_feats.cpu().numpy())

                # 取聚类中心并做最小间距约束（不改变簇数量）
                centers = self._enforce_min_distance(kmeans.cluster_centers_, min_dist=0.1)
                class_means = torch.from_numpy(centers).to(device=device, dtype=class_feats.dtype)  # (K, D)

                # 每簇样本计数 -> 概率（先验）
                # 注意：使用 minlength=K，保证形状与中心数一致
                K = class_means.size(0)
                counts = torch.bincount(torch.from_numpy(kmeans.labels_), minlength=K).to(device)
                # 若极端情况下某簇为空（counts.sum()>0 仍成立），做一个极小平滑以避免 0 概率
                probs = (counts.float() + 1e-12) / (counts.sum().float() + K * 1e-12)
            else:
                class_means = class_feats.mean(dim=0, keepdim=True)
                probs = None  # 单均值情形无需先验权重

            # 2) 协方差
            if use_task_cov:
                # 用类均值的平均作为 query 进行注意力加权
                query = class_means.mean(dim=0, keepdim=True)  # (1, D)
                sim = query @ task_means.t()
                # sim = F.normalize(query, dim=1) @ F.normalize(task_means, dim=1).t()  # (1, T)
                attn = (sim.squeeze(0) / 0.1).softmax(dim=-1)  # (T,)
                class_cov = sum(attn[i] * task_covs[i] for i in range(task_covs.size(0)))
            else:
                # 由该类样本直接估计协方差（必要时可加对角正则）
                # 若样本数 < 2，torch.cov 会报错或得到 NaN，可在上层保证或这里兜底
                if class_feats.size(0) >= 2:
                    class_cov = torch.cov(class_feats.T) 
                else:
                    # 兜底：用全局尺度的对角矩阵
                    d = class_feats.size(1)
                    class_cov = torch.eye(d, device=device, dtype=class_feats.dtype) * 1e-4

            # 3) 打包
            cid = int(lbl.item())
            if multi_means:
                # print(probs)
                stats[cid] = MultiMeanGaussianStatistics(class_means, class_cov, probs=probs)
            else:
                stats[cid] = GaussianStatistics(class_means, class_cov)  # ctor 会 squeeze 到 (D,)
        return stats

    # --------------------- 8 种分布一把生成 ---------------------
    def build_all_variants(self, task_id: int, model_before: nn.Module, model_after: nn.Module, data_loader):
        """
        生成并返回 8 份分布：
          multi_means ∈ {False(单均值), True(多均值-类内共享协方差)}
          task_cov    ∈ {False(类内协方差), True(task-wise 协方差)}
          linear      ∈ {False(原空间), True(线性补偿后空间)}
        命名形如：mm0_tc1_lin0, ...
        """
        feats_before, feats_after, labels = self.extract_features_before_after(
            model_before, model_after, data_loader)

        # 初始化缓存噪声
        if self.cached_Z is None:
            self.cached_Z = torch.randn(50000, feats_after.size(1))

        # 线性补偿矩阵
        if self.compensate and task_id > 0:
            aux_loader = self.get_aux_loader(self.args)
            feats_aux_before, feats_aux_after = self.extract_features_before_after_for_auxiliary_data(
                model_before, model_after, aux_loader)
            feats_b = torch.cat([feats_before, feats_aux_before], dim=0)
            feats_a = torch.cat([feats_after, feats_aux_after], dim=0)
            W = self.compute_linear_transform(feats_b, feats_a)

            if hasattr(self, "linear_transforms"):
                self.linear_transforms[task_id] = W.cpu()
            else:
                self.linear_transforms = {}
                self.linear_transforms[task_id] = W.cpu()

        # 刷新 task-wise 协方差缓存
        self._task_cov_cache = None
        self._get_task_cov_bank(feats_after)

        # 组合开关
        configs = []
        for multi_means in (False, True):
            for use_task_cov in (False, True):
                configs.append((multi_means, use_task_cov))

        variants = {}
        for m, t in configs:
            key = f"multi-mean:{m}_task-shared-cov:{t}"

            stats = self._build_stats(
                features=feats_after, labels=labels,
                multi_means=m, use_task_cov=t)
            
            variants[key] = stats

        if hasattr(self, "variants") and len(self.variants) > 0:
            if self.compensate and task_id > 0:
                for key, stats in self.variants.items():
                   if "compensate" in key:
                       self.variants[key] = self._transform_stats_with_W(stats, W)
                       
            for key, stats in variants.items():
                self.variants[key].update(variants[key])
                if self.compensate and task_id > 0:
                    new_key = f"{key}_compensate"
                    self.variants[new_key].update(variants[key])
        
        else:
            self.variants = variants
            if self.compensate:
                compensated_variants = {}
                for key, stats in variants.items():
                    new_key = f"{key}_compensate"
                    compensated_variants[new_key] = copy.deepcopy(stats)
                self.variants.update(compensated_variants)

        print(f"[INFO] Built {len(variants)} distribution variants: {list(variants.keys())}")
        return variants


>>>>>>> 4816499fc2b904e2d81571b705b8f392a7bd6601
    # --------------------- 分类器再训练（采样驱动） ---------------------
    def train_classifier_with_cached_samples(self, fc: nn.Module, stats: Dict[int, object],
                                             epochs: int = 6, use_weighted_cov: bool = False) -> nn.Module:
        """用各分布采样的特征对分类器做快速再训练"""
        epochs = int(epochs)
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
            # 为采样选择 L 或 L_weighted
            if isinstance(gauss, MultiMeanGaussianStatistics):
                class_mean = gauss.means.mean(dim=0).to(self.device)
                L_matrix = gauss.L_weighted.to(self.device) if use_weighted_cov and gauss.L_weighted is not None else gauss.L.to(self.device)
            else:
                class_mean = gauss.mean.to(self.device)
                L_matrix = gauss.L_weighted.to(self.device) if use_weighted_cov and gauss.L_weighted is not None else gauss.L.to(self.device)

            class_means.append(class_mean)

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
        class_mean_norm = class_means.norm(dim=-1, keepdim=True).mean(dim=0).squeeze()

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

                # 你可以切换为标准 CE 或 SCE
                # loss = symmetric_cross_entropy_loss(logits, tgt)
                loss = F.cross_entropy(logits, tgt)

                loss.backward()
                optimizer.step()
                losses += loss.item() * (end_idx - start_idx)

            loss_epoch = losses / num_samples
            if (epoch + 1) % 3 == 0:
                print(f"[INFO] Cached-sample classifier training: Epoch {epoch + 1}, Loss: {loss_epoch:.4f}")
            scheduler.step()

        return fc

    def refine_classifiers_from_variants(self, fc: nn.Module, epochs: int = 6, use_weighted_cov: bool = False) -> Dict[str, nn.Module]:
        """对 self.variants 里的每个分布各训练一个分类器"""
        assert hasattr(self, 'variants') and len(self.variants) > 0, "No variants found. Call build_all_variants first."
        out = {}
        for name, stats in self.variants.items():
            clf = self.train_classifier_with_cached_samples(fc, stats, epochs=epochs, use_weighted_cov=use_weighted_cov)
            out[name] = clf
        print(f"[INFO] Trained {len(out)} classifiers from variants.")
        return out

    # --------------------- 辅助数据加载 ---------------------
    def get_aux_loader(self, args):
        if self.aux_loader is not None:
            return self.aux_loader

        aux_dataset_type = args.get('aux_dataset', 'cifar10')
        num_samples = int(args.get('auxiliary_data_size', 5000))

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
<<<<<<< HEAD
                                 std=[0.229, 0.224, 0.225])
        ])
=======
                                 std=[0.229, 0.224, 0.225])])
>>>>>>> 4816499fc2b904e2d81571b705b8f392a7bd6601

        if aux_dataset_type == 'imagenet':
            if 'auxiliary_data_path' not in args:
                raise ValueError("必须提供 auxiliary_data_path")
            dataset = datasets.ImageFolder(args['auxiliary_data_path'], transform=transform)
        elif aux_dataset_type == 'cifar10':
            dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        elif aux_dataset_type == 'svhn':
            dataset = datasets.SVHN(root='./data', split='train', download=True, transform=transform)
        elif aux_dataset_type == 'flickr8k':
            dataset = datasets.ImageFolder(args['auxiliary_data_path'], transform=transform)
        else:
            raise ValueError(f"不支持的 aux_dataset_type: {aux_dataset_type}")

        indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
        train_subset = Subset(dataset, indices)

        self.aux_loader = DataLoader(train_subset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
        return self.aux_loader
<<<<<<< HEAD


# -----------------------------------------------------------------------------
# 使用示例（伪代码）
# -----------------------------------------------------------------------------
# args = dict(
#     n_class_clusters=3,
#     n_task_clusters=3,
#     cov_weighting_temperature=0.2,
#     alpha_t=1.0,
#     gamma_1=1e-4,
#     auxiliary_data_size=5000,
#     aux_dataset='cifar10',
#     compensate=True,
# )
# dc = Drift_Compensator(args)
# variants = dc.build_all_variants(task_id, model_before, model_after, data_loader)
# # 训练分类器
# classifiers = dc.refine_classifiers_from_variants(fc, epochs=6, use_weighted_cov=False)
=======
>>>>>>> 4816499fc2b904e2d81571b705b8f392a7bd6601
