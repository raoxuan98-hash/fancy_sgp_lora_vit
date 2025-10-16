# -*- coding: utf-8 -*-
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import logging


# ===========================
#   基础函数与结构体
# ===========================

def cholesky_manual_stable(matrix: torch.Tensor, reg: float = 1e-5) -> torch.Tensor:
    """稳定的Cholesky分解（保证正定）"""
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


class GaussianStatistics:
    """封装高斯统计：均值、协方差、Cholesky"""
    def __init__(self, mean: torch.Tensor, cov: torch.Tensor, reg: float = 1e-5):
        if mean.dim() == 2 and mean.size(0) == 1:
            mean = mean.squeeze(0)
        self.mean = mean
        self.cov = cov
        self.reg = reg
        self.L = cholesky_manual_stable(cov, reg=reg)

    def to(self, device):
        self.mean = self.mean.to(device)
        self.cov = self.cov.to(device)
        self.L = self.L.to(device)
        return self

    def sample(self, n_samples: int, cached_eps: torch.Tensor = None):
        device = self.mean.device
        d = self.mean.size(0)
        if cached_eps is None:
            eps = torch.randn(n_samples, d, device=device)
        else:
            eps = cached_eps.to(device)
            n_samples = eps.size(0)
        return self.mean.unsqueeze(0) + eps @ self.L.t()


# ===========================
#   弱非线性残差 MLP
# ===========================

class ResidMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.log_scale = nn.Parameter(torch.tensor(0.0))
        self.fc1 = nn.Linear(dim, dim, bias=False)
        self.fc1.weight.data = torch.eye(dim)
        self.fc2 = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.ReLU(),
            nn.Linear(dim, dim, bias=False),
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
    """训练弱非线性映射网络"""
    def __init__(self, input_dim: int):
        self.net = ResidMLP(input_dim)
        self.is_trained = False

    def train(self, features_before: torch.Tensor, features_after: torch.Tensor,
              epochs: int = 4000, lr: float = 0.001):
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
            pred = self.net(X[idx])
            loss = criterion(pred, Y[idx]) + 0.5 * self.net.reg_loss()
            loss.backward(); optimizer.step(); scheduler.step()
            if (epoch + 1) % 1000 == 0:
                logging.info(f"[WeakNonlinear] epoch {epoch+1}/{epochs}, loss={loss.item():.6f}")
        self.is_trained = True
        return self.net

    def transform_features(self, features: torch.Tensor):
        if not self.is_trained:
            raise ValueError("Nonlinear transform not trained yet.")
        with torch.no_grad():
            norms = features.norm(dim=1, keepdim=True)
            return self.net(features / norms) * norms

    def transform_stats(self, stats_dict, n_samples: int = 5000):
        """通过采样变换高斯统计量"""
        if not self.is_trained:
            raise ValueError("Nonlinear transform not trained yet.")
        transformed = {}
        device = next(self.net.parameters()).device
        for cid, stat in stats_dict.items():
            samples = stat.sample(n_samples).to(device)
            new_samples = self.transform_features(samples)
            new_mean = new_samples.mean(dim=0).cpu()
            new_cov = torch.cov(new_samples.T).cpu()
            transformed[cid] = GaussianStatistics(new_mean, new_cov, stat.reg)
        return transformed


# ===========================
#     SLDC 补偿器主体
# ===========================

class Drift_Compensator(object):
    def __init__(self, args):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.gamma = args.get("gamma", 1e-4)
        self.temp = args.get("temp", 1.0)
        self.compensate = args.get("compensate", True)
        self.use_nonlinear = args.get("use_weaknonlinear", True)
        self.args = args
        self.cached_Z = None
        self.linear_transforms = {}
        self.weaknonlinear_transforms = {}
        self.feature_dim = None
        self.aux_loader = None

    # -------------------------------------
    #   线性补偿矩阵估计 (α₁-SLDC)
    # -------------------------------------
    def compute_linear_transform(self, feats_before, feats_after, normalize=True):
        logging.info("[SLDC] Computing linear compensation matrix (α₁-SLDC)")
        device = self.device
        X = F.normalize(feats_before.to(device), dim=1) if normalize else feats_before
        Y = F.normalize(feats_after.to(device), dim=1) if normalize else feats_after
        n, d = X.size()
        XTX = X.T @ X + self.gamma * torch.eye(d, device=device)
        XTY = X.T @ Y
        W = torch.linalg.solve(XTX, XTY)
        weight = math.exp(-n / (self.temp * d))
        W = (1 - weight) * W + weight * torch.eye(d, device=device)
        return W

    # -------------------------------------
    #   弱非线性补偿 (α₂-SLDC)
    # -------------------------------------
    def compute_weaknonlinear_transform(self, feats_before, feats_after):
        logging.info("[SLDC] Training weak nonlinear compensator (α₂-SLDC)")
        transform = WeakNonlinearTransform(feats_before.size(1))
        transform.train(feats_before, feats_after)
        return transform

    # -------------------------------------
    #   基于线性矩阵变换统计量
    # -------------------------------------
    def transform_stats_with_W(self, stats_dict, W):
        if W is None: return {}
        W = W.cpu(); WT = W.t()
        out = {}
        for cid, stat in stats_dict.items():
            mean = stat.mean @ W
            cov = WT @ stat.cov @ W + 1e-3 * torch.eye(stat.cov.size(0))
            out[cid] = GaussianStatistics(mean, cov, stat.reg)
        return out

    # -------------------------------------
    #   构建高斯统计
    # -------------------------------------
    def _build_stats(self, feats, labels):
        feats, labels = feats.cpu(), labels.cpu()
        stats = {}
        for lbl in torch.unique(labels):
            Xc = feats[labels == lbl]
            mean = Xc.mean(dim=0)
            cov = torch.cov(Xc.T) if Xc.size(0) > 1 else torch.eye(Xc.size(1)) * 1e-4
            stats[int(lbl.item())] = GaussianStatistics(mean, cov)
        return stats

    # -------------------------------------
    #   主流程：构建各变体
    # -------------------------------------
    def build_all_variants(self, task_id, model_before, model_after, data_loader):
        feats_before, feats_after, labels = self.extract_features_before_after(model_before, model_after, data_loader)
        if self.feature_dim is None:
            self.feature_dim = feats_after.size(1)
        if self.cached_Z is None:
            self.cached_Z = torch.randn(50000, self.feature_dim)
        stats = self._build_stats(feats_after, labels)

        if not hasattr(self, "variants"):
            self.variants = {
                "alpha_1-SLDC": {},
                "alpha_1-SLDC + ADE": {},
                "alpha_2-SLDC": {},
                "alpha_2-SLDC + ADE": {},
            }

        if self.compensate and task_id > 1:
            W = self.compute_linear_transform(feats_before, feats_after)
            self.linear_transforms[task_id] = W.cpu()
            if self.use_nonlinear:
                T = self.compute_weaknonlinear_transform(feats_before, feats_after)
                self.weaknonlinear_transforms[task_id] = T

            # α₁: 线性补偿
            stats_lin = self.transform_stats_with_W(self.variants["alpha_1-SLDC"], W)
            stats_lin.update(copy.deepcopy(stats))
            self.variants["alpha_1-SLDC"] = stats_lin
            self.variants["alpha_1-SLDC + ADE"] = copy.deepcopy(stats_lin)

            # α₂: 弱非线性补偿
            if self.use_nonlinear:
                stats_nonlin = T.transform_stats(self.variants["alpha_2-SLDC"])
                stats_nonlin.update(copy.deepcopy(stats))
                self.variants["alpha_2-SLDC"] = stats_nonlin
                self.variants["alpha_2-SLDC + ADE"] = copy.deepcopy(stats_nonlin)

        else:
            for k in self.variants.keys():
                self.variants[k].update(copy.deepcopy(stats))

        logging.info(f"[INFO] Built SLDC variants for task {task_id}, num_classes={len(stats)}")
        return self.variants

    # -------------------------------------
    #   特征提取
    # -------------------------------------
    @torch.no_grad()
    def extract_features_before_after(self, model_before, model_after, loader):
        model_before, model_after = model_before.to(self.device), model_after.to(self.device)
        model_before.eval(); model_after.eval()
        feats_b, feats_a, labels = [], [], []
        for x, y in loader:
            x = x.to(self.device)
            feats_b.append(model_before(x).cpu())
            feats_a.append(model_after(x).cpu())
            labels.append(y)
        return torch.cat(feats_b), torch.cat(feats_a), torch.cat(labels)

    # -------------------------------------
    #   SGD分类器训练
    # -------------------------------------
    def train_classifier_with_cached_samples(self, fc, stats, epochs=5):
        device = self.device
        fc = copy.deepcopy(fc).to(device)
        opt = torch.optim.SGD(fc.parameters(), lr=0.01, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=0.001)
        cached_Z = self.cached_Z.to(device)
        samples_all, targets_all = [], []
        for cid, gauss in stats.items():
            m, L = gauss.mean.to(device), gauss.L.to(device)
            Z = cached_Z[cid * 1024:(cid + 1) * 1024]
            X = m + Z @ L.t()
            y = torch.full((X.size(0),), int(cid), device=device)
            samples_all.append(X); targets_all.append(y)
        X = torch.cat(samples_all); y = torch.cat(targets_all)
        for epoch in range(epochs):
            perm = torch.randperm(X.size(0), device=device)
            for i in range(0, X.size(0), 64):
                batch_idx = perm[i:i+64]
                xb, yb = X[batch_idx], y[batch_idx]
                opt.zero_grad()
                loss = F.cross_entropy(fc(xb), yb)
                loss.backward(); opt.step()
            sch.step()
        return fc

    # -------------------------------------
    #   ADE 数据加载器
    # -------------------------------------
    def initialize_aux_loader(self, train_set):
        self.aux_loader = DataLoader(train_set, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
        return self.aux_loader

    def get_aux_loader(self, args):
        return self.aux_loader
