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
            nn.SiLU(),
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
    
        X = F.normalize(features_before, dim=1)
        Y = F.normalize(features_after, dim=1)

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
                print(f"非线性变换训练 step {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
        
        self.is_trained = True
        return self.net
    
    def transform_features(self, features: torch.Tensor) -> torch.Tensor:
        """变换特征"""
        if not self.is_trained:
            raise ValueError("非线性变换器尚未训练")
        with torch.no_grad():
            normalized_features = F.normalize(features, dim=1)
            return self.net(normalized_features)
    
    def transform_stats(self, stats_dict: Dict[int, GaussianStatistics], 
                       n_samples: int = 1000) -> Dict[int, GaussianStatistics]:
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
        self.weaknonlinear_transforms = {}

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
        nonlinear_transform = None
        
        if self.compensate and task_id > 1:
            aux_loader = self.get_aux_loader(self.args)
            feats_aux_before, feats_aux_after, _ = self.extract_features_before_after(model_before, model_after, aux_loader)
            
            # 合并当前任务和辅助数据
            feats_before_combined = torch.cat([feats_before, feats_aux_before], dim=0)
            feats_after_combined = torch.cat([feats_after, feats_aux_after], dim=0)

            # 计算线性变换
            W = self.compute_linear_transform(feats_before_combined, feats_after_combined)
            self.linear_transforms[task_id] = W.cpu()
            
            # 计算非线性变换（如果启用）
            if self.use_nonlinear:
                nonlinear_transform = self.compute_nonlinear_transform(feats_before_combined, feats_after_combined)
                self.nonlinear_transforms[task_id] = nonlinear_transform

        # 构建基础统计
        stats = self._build_stats(features=feats_after, labels=labels)

        # 初始化变体字典
        if not hasattr(self, 'variants') or len(self.variants) == 0:
            self.variants = {
                "per-class": {},                    # 原始特征
                "per-class_compensate": {},         # 线性补偿
                "per-class_nonlinear": {},          # 非线性补偿（新增）
                "per-class_combined": {}            # 原始+线性补偿特征（新增）
            }

        # 更新原始分布
        self.variants["per-class"].update(stats)

        # 线性补偿更新
        if self.compensate and task_id > 1 and W is not None:
            # 线性补偿版本
            stats_compensated = self.transform_stats_with_W(self.variants['per-class_compensate'], W)
            self.variants["per-class_compensate"] = stats_compensated
            self.variants["per-class_compensate"].update(copy.deepcopy(stats))
            
            # 非线性补偿版本（新增）
            if nonlinear_transform is not None:
                stats_nonlinear = nonlinear_transform.transform_stats(self.variants["per-class"])
                self.variants["per-class_nonlinear"].update(stats_nonlinear)
                
            # 组合特征版本：原始特征 + 线性补偿特征（新增）
            self.variants["per-class_combined"].update(self._build_combined_stats(
                self.variants["per-class"], self.variants["per-class_compensate"]
            ))
        else:
            # 如果不补偿，所有版本都使用原始统计
            self.variants["per-class_compensate"].update(copy.deepcopy(stats))
            self.variants["per-class_nonlinear"].update(copy.deepcopy(stats))
            self.variants["per-class_combined"].update(copy.deepcopy(stats))

        print(f"[INFO] Built distribution variants: {list(self.variants.keys())}, num classes: {len(stats)}")
        return self.variants

    def _build_combined_stats(self, stats_original: Dict, stats_compensated: Dict) -> Dict:
        """构建组合特征版本的统计（原始特征 + 补偿特征）"""
        combined_stats = {}
        
        for cid in stats_original.keys():
            if cid not in stats_compensated:
                continue
                
            orig_stat = stats_original[cid]
            comp_stat = stats_compensated[cid]
            
            # 组合均值和协方差
            combined_mean = torch.cat([orig_stat.mean, comp_stat.mean])
            
            # 构建块对角协方差矩阵
            d = orig_stat.mean.size(0)
            combined_cov = torch.zeros(2*d, 2*d)
            combined_cov[:d, :d] = orig_stat.cov
            combined_cov[d:, d:] = comp_stat.cov
            # 可以添加跨协方差项，这里简单设为0
            
            combined_stats[cid] = GaussianStatistics(combined_mean, combined_cov, orig_stat.reg)
            
        return combined_stats

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

    # --------------------- 分类器再训练（采样驱动） ---------------------
    def train_classifier_with_cached_samples(self, fc: nn.Module, stats: Dict[int, object],
                                             epochs: int = 5, use_weighted_cov: bool = False) -> nn.Module:
        """用各分布采样的特征对分类器做快速再训练"""
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

    def refine_classifiers_from_variants(self, fc: nn.Module, epochs: int = 5, use_weighted_cov: bool = False) -> Dict[str, nn.Module]:
        """对 self.variants 里的每个分布各训练一个分类器"""
        assert hasattr(self, 'variants') and len(self.variants) > 0, "No variants found. Call build_all_variants first."
        out = {}
        for name, stats in self.variants.items():
            # 对于组合特征版本，需要调整分类器输入维度
            if "combined" in name:
                # 创建新的分类器来处理组合特征
                original_input_dim = fc.weight.size(1)
                combined_input_dim = 2 * original_input_dim  # 原始+补偿特征
                num_classes = fc.weight.size(0)
                
                combined_fc = nn.Linear(combined_input_dim, num_classes, bias=(fc.bias is not None))
                if fc.bias is not None:
                    combined_fc.bias.data = fc.bias.data.clone()
                # 初始化新权重：平均使用两个特征流的信息
                combined_fc.weight.data[:, :original_input_dim] = fc.weight.data * 0.5
                combined_fc.weight.data[:, original_input_dim:] = fc.weight.data * 0.5
                
                clf = self.train_classifier_with_cached_samples(combined_fc, stats, epochs=epochs, use_weighted_cov=use_weighted_cov)
            else:
                clf = self.train_classifier_with_cached_samples(fc, stats, epochs=epochs, use_weighted_cov=use_weighted_cov)
            out[name] = clf
        print(f"[INFO] Trained {len(out)} classifiers from variants.")
        return out

    # --------------------- 辅助数据加载 ---------------------
    def initialize_aux_loader(self, train_set):
        if self.aux_loader is not None:
            return self.aux_loader
        self.aux_loader = DataLoader(train_set, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
        return self.aux_loader
    
    def get_aux_loader(self, args):
        return self.aux_loader