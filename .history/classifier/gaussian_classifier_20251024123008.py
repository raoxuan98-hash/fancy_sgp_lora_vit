# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict
from compensator.gaussian_statistics import GaussianStatistics


class LinearLDAClassifier(nn.Module):
    """
    将 LDA 判别函数解析为一个线性层：logits = x @ W + b
    W_c = Σ⁻¹ μ_c
    b_c = -0.5 μ_cᵀ Σ⁻¹ μ_c + log π_c

    正则（固定为 spherical）:
    Σ_reg = (1-α)·Σ_global + α·(trace(Σ_global)/d)·I
    """
    def __init__(
        self,
        stats_dict,
        class_priors=None,
        lda_reg_alpha: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.lda_reg_alpha = lda_reg_alpha

        # === Step 1. 全局协方差 ===
        class_ids = sorted(stats_dict.keys())
        self.num_classes = len(class_ids)
        means = torch.stack([stats_dict[cid].mean for cid in class_ids]).to(self.device)
        covs = torch.stack([stats_dict[cid].cov for cid in class_ids]).to(self.device)
        global_cov = covs.mean(0)
        global_cov = 0.5 * (global_cov + global_cov.T)

        # === Step 2. spherical 正则 ===
        d = global_cov.size(0)
        cov_reg = (1.0 - self.lda_reg_alpha) * global_cov + self.lda_reg_alpha * torch.eye(d, device=self.device)

        # === Step 3. 计算逆矩阵 ===
        cov_reg = cov_reg + 1e-6 * torch.eye(d, device=self.device)
        cov_inv = torch.linalg.inv(cov_reg)

        # === Step 4. 权重 & 偏置解析解 ===
        priors = {cid: 1.0 / self.num_classes for cid in class_ids} if class_priors is None else class_priors
        W, b = [], []
        for i, cid in enumerate(class_ids):
            mu = means[i]
            w_c = cov_inv @ mu
            logpi = torch.log(torch.tensor(priors[cid], device=self.device))
            b_c = -0.5 * (mu @ cov_inv @ mu) + logpi
            W.append(w_c)
            b.append(b_c)
        W = torch.stack(W, dim=1)  # [D, C]
        b = torch.stack(b)         # [C]

        # === Step 5. 线性层承载 ===
        self.linear = nn.Linear(d, self.num_classes, bias=True)
        self.linear.weight.data = W.T.clone()
        self.linear.bias.data = b.clone()
        self.linear.requires_grad_(False)

    def forward(self, x):
        return self.linear(x)

    def predict(self, x):
        return torch.argmax(self.forward(x), dim=1)

    def predict_proba(self, x):
        return F.softmax(self.forward(x), dim=1)


class RegularizedGaussianDA(nn.Module):
    """
    仅 QDA（类别独立协方差）的解析判别分类器（去掉 LDA 分支）。

    正则：shrinkage + spherical 组合
      Σ_c_reg = α1·Σ_c + α2·Σ_global + α3·(trace(Σ_c)/d)·I,

    判别函数（对第 c 类）：
      g_c(x) = -0.5 (x-μ_c)^T Σ_c_reg^{-1} (x-μ_c) - 0.5 log|Σ_c_reg| + log π_c
    """
    def __init__(
        self,
        stats_dict: Dict[int, GaussianStatistics],
        class_priors: Dict[int, float] = None,
        qda_reg_alpha1: float = 1.0,
        qda_reg_alpha2: float = 1.0,
        qda_reg_alpha3: float = 1.0, 
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.class_ids = sorted(stats_dict.keys())
        self.num_classes = len(self.class_ids)
        self.qda_reg_alpha1 = float(qda_reg_alpha1)
        self.qda_reg_alpha2 = float(qda_reg_alpha2)
        self.qda_reg_alpha3 = float(qda_reg_alpha3)
        self.epsilon = 1e-4   # per-class稳健性
        self.epsilon_identity = 1e-6

        # 类先验
        if class_priors is None:
            priors_list = [1.0 / self.num_classes for _ in self.class_ids]
        else:
            priors_list = [class_priors[cid] for cid in self.class_ids]
        self.log_priors = nn.Parameter(torch.log(torch.tensor(priors_list, device=self.device)), requires_grad=False)  # [C]

        # 收集均值与协方差
        means = []
        covs = []
        for cid in self.class_ids:
            s = stats_dict[cid]
            means.append(s.mean.float().to(self.device))
            covs.append(s.cov.float().to(self.device))
        means = torch.stack(means)   # [C, D]
        covs = torch.stack(covs)     # [C, D, D]

        # 全局协方差（供 shrink 使用）
        global_cov = covs.mean(0)
        global_cov = 0.5 * (global_cov + global_cov.T)

        self.register_buffer("means", means, persistent=False)          # [C, D]
        self.register_buffer("global_cov", global_cov, persistent=False)

        # ---- QDA 正则：β·Σ_c + α1·Σ_global + α2·(tr(Σ_c)/d)·I ----
        C, D, _ = covs.shape
        sph = torch.eye(D, device=self.device).unsqueeze(0)
        a1, a2, a3 = self.qda_reg_alpha1, self.qda_reg_alpha2, self.qda_reg_alpha3
        covs_sym = 0.5 * (covs + covs.transpose(-1, -2))               # 数值对称化
        covs_reg = beta * covs_sym + a1 * global_cov.unsqueeze(0) + a2 * sph
        covs_reg = covs_reg + self.epsilon * torch.eye(D, device=self.device).unsqueeze(0)

        # ---- 预计算每类逆阵与 logdet（Cholesky优先，失败回退SVD） ----
        cov_invs = []
        logdets = []
        for c in range(C):
            cov = covs_reg[c]
            try:
                L = torch.linalg.cholesky(cov)
                inv = torch.cholesky_inverse(L)
                logdet = 2 * torch.sum(torch.log(torch.diag(L)))

            except Exception:
                U, S, Vh = torch.linalg.svd(cov)
                inv = U @ torch.diag(1.0 / torch.clamp(S, min=1e-6)) @ Vh
                logdet = torch.sum(torch.log(torch.clamp(S, min=1e-6)))
            cov_invs.append(inv)
            logdets.append(logdet)

        cov_invs = torch.stack(cov_invs)               # [C, D, D]
        logdets = torch.stack(logdets)                 # [C]

        self.register_buffer("cov_invs", cov_invs, persistent=False)
        self.register_buffer("logdets", logdets, persistent=False)

        logging.info(
            f"[INFO] RegularizedGaussianQDA initialized: {self.num_classes} classes, "
            f"qda_reg_alpha1={self.qda_reg_alpha1}, qda_reg_alpha2={self.qda_reg_alpha2}"
        )

    # ================= 判别函数（向量化） =================
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, D]
        返回：logits [B, C]
        """
        x = x.to(self.device)
        B, D = x.shape
        C = self.means.shape[0]

        # (x - mu_c)
        xc = x.unsqueeze(1) - self.means.unsqueeze(0)          # [B, C, D]

        # 计算 (x-μ)^T Σ^{-1} (x-μ)  —— 使用两步法避免高阶einsum易错
        # step1: v = (x-μ) @ Σ^{-1}  -> [B, C, D]
        v = torch.einsum("bcd,cde->bce", xc, self.cov_invs)
        # step2: (v * (x-μ)).sum(-1)  -> [B, C]
        maha = 0.5 * (v * xc).sum(dim=-1)                       # [B, C]

        logits = -maha - 0.5 * self.logdets.unsqueeze(0) + self.log_priors.unsqueeze(0)  # [B, C]
        return logits

    def predict(self, x: torch.Tensor):
        return torch.argmax(self.forward(x), dim=1)

    def predict_proba(self, x: torch.Tensor):
        return F.softmax(self.forward(x), dim=1)
