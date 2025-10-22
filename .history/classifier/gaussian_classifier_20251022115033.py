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
        trace_over_d = torch.trace(global_cov) / d
        cov_reg = (1.0 - self.lda_reg_alpha) * global_cov + self.lda_reg_alpha * trace_over_d * torch.eye(d, device=self.device)

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


class RegularizedGaussianClassifier(nn.Module):
    """
    基于高斯统计的解析判别分类器。
    - LDA（共享协方差）：固定 spherical 正则，强度 lda_reg_alpha
      Σ_shared_reg = (1-α)·Σ_global + α·(trace(Σ_global)/d)·I
    - QDA（类别独立协方差）：固定 shrinkage+spherical 组合正则，强度 qda_reg_alpha1, qda_reg_alpha2
      Σ_c_reg = β·Σ_c + α1·Σ_global + α2·(trace(Σ_c)/d)·I, 其中 β = max(1-α1-α2, 0)
    """
    def __init__(
        self,
        stats_dict: Dict[int, GaussianStatistics],
        class_priors: Dict[int, float] = None,
        mode: str = "qda",  # "lda" or "qda"
        lda_reg_alpha: float = 0.1,
        qda_reg_alpha1: float = 0.1,
        qda_reg_alpha2: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        assert mode in ("lda", "qda")
        self.mode = mode
        self.lda_reg_alpha = float(lda_reg_alpha)
        self.qda_reg_alpha1 = float(qda_reg_alpha1)  # shrink to global
        self.qda_reg_alpha2 = float(qda_reg_alpha2)  # spherical
        self.device = torch.device(device)
        self.class_ids = sorted(stats_dict.keys())
        self.num_classes = len(self.class_ids)
        self.epsilon = 1e-4
        self.epsilon_identity = 1e-6

        # 类先验
        if class_priors is None:
            self.priors = {cid: 1.0 / self.num_classes for cid in self.class_ids}
        else:
            self.priors = class_priors

        # 收集均值与协方差
        means = []
        covs = []
        for cid in self.class_ids:
            s = stats_dict[cid]
            means.append(s.mean.float().to(self.device))
            covs.append(s.cov.float().to(self.device))
        means = torch.stack(means)     # [C, D]
        covs = torch.stack(covs)       # [C, D, D]

        self.global_mean = nn.Parameter(means.mean(0), requires_grad=False)
        global_cov = covs.mean(0)
        global_cov = 0.5 * (global_cov + global_cov.T)
        self.global_cov = nn.Parameter(global_cov, requires_grad=False)

        # ---- 正则函数们 ----
        def spherical_reg(cov_mat: torch.Tensor, alpha: float) -> torch.Tensor:
            """(1-α)·cov_mat + α·(trace(cov_mat)/d)·I"""
            d = cov_mat.size(-1)
            trace_over_d = torch.trace(cov_mat) / d
            return (1.0 - alpha) * cov_mat + alpha * trace_over_d * torch.eye(d, device=cov_mat.device)

        def shrink_spherical_qda(class_cov: torch.Tensor) -> torch.Tensor:
            """
            QDA: β·Σ_c + α1·Σ_global + α2·(trace(Σ_c)/d)·I, β=max(1-α1-α2, 0)
            """
            a1 = self.qda_reg_alpha1
            a2 = self.qda_reg_alpha2
            beta = max(1.0 - a1 - a2, 0.0)
            d = class_cov.size(-1)
            trace_over_d = torch.trace(class_cov) / d
            sph = trace_over_d * torch.eye(d, device=class_cov.device)
            return beta * class_cov + a1 * self.global_cov + a2 * sph

        # ---- 对每个类别的协方差做正则、对称化与稳定化 ----
        covs_reg = []
        for cov in covs:
            if self.mode == "qda":
                cov_r = shrink_spherical_qda(0.5 * (cov + cov.T))
            else:
                # LDA时，先保持原始类协方差用于求 shared 之前的均值，再单独对 shared 做 spherical
                cov_r = 0.5 * (cov + cov.T)
            cov_r += self.epsilon * torch.eye(cov_r.size(-1), device=self.device)
            covs_reg.append(cov_r)
        covs_reg = torch.stack(covs_reg)

        # ---- 预计算每类（或共享）逆阵与 logdet ----
        invs = []
        logdets = []
        for cov in covs_reg:
            try:
                L = torch.linalg.cholesky(cov)
                inv = torch.cholesky_inverse(L)
                logdet = 2 * torch.sum(torch.log(torch.diag(L)))
            except Exception:
                U, S, Vh = torch.linalg.svd(cov)
                inv = U @ torch.diag(1.0 / torch.clamp(S, min=1e-12)) @ Vh
                logdet = torch.sum(torch.log(torch.clamp(S, min=1e-12)))
            invs.append(inv)
            logdets.append(logdet)
        invs = torch.stack(invs)
        logdets = torch.tensor(logdets, device=self.device)

        # 存储参数（逐类）
        self.means = nn.ParameterDict()
        self.covs = nn.ParameterDict()
        self.cov_invs = nn.ParameterDict()
        self.log_dets = nn.ParameterDict()
        for i, cid in enumerate(self.class_ids):
            key = str(cid)
            self.means[key] = nn.Parameter(means[i], requires_grad=False)
            self.covs[key] = nn.Parameter(covs_reg[i], requires_grad=False)
            self.cov_invs[key] = nn.Parameter(invs[i], requires_grad=False)
            self.log_dets[key] = nn.Parameter(logdets[i].unsqueeze(0), requires_grad=False)

        # ---- LDA 共享协方差的 spherical 正则 & 逆阵 ----
        if self.mode == "lda":
            shared_cov_raw = covs.mean(0)
            shared_cov_raw = 0.5 * (shared_cov_raw + shared_cov_raw.T)
            shared_cov = spherical_reg(shared_cov_raw, self.lda_reg_alpha)
            shared_cov = shared_cov + self.epsilon_identity * torch.eye(shared_cov.size(-1), device=self.device)
            try:
                L = torch.linalg.cholesky(shared_cov)
                shared_inv = torch.cholesky_inverse(L)
                shared_logdet = 2 * torch.sum(torch.log(torch.diag(L)))
            except Exception:
                U, S, Vh = torch.linalg.svd(shared_cov)
                shared_inv = U @ torch.diag(1.0 / torch.clamp(S, min=1e-12)) @ Vh
                shared_logdet = torch.sum(torch.log(torch.clamp(S, min=1e-12)))
            self.shared_cov = nn.Parameter(shared_cov, requires_grad=False)
            self.shared_inv = nn.Parameter(shared_inv, requires_grad=False)
            self.shared_logdet = nn.Parameter(shared_logdet.unsqueeze(0), requires_grad=False)

        logging.info(
            f"[INFO] RegularizedGaussianClassifier initialized: {self.num_classes} classes, "
            f"mode={mode}, lda_reg_alpha={self.lda_reg_alpha}, "
            f"qda_reg_alpha1={self.qda_reg_alpha1}, qda_reg_alpha2={self.qda_reg_alpha2}"
        )

    # ================= 判别函数 =================
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        batch = x.size(0)
        logits = torch.zeros(batch, self.num_classes, device=self.device)
        for i, cid in enumerate(self.class_ids):
            if self.mode == "qda":
                logits[:, i] = self._qda_discriminant(x, cid)
            else:
                logits[:, i] = self._lda_discriminant(x, cid)
        return logits

    def _qda_discriminant(self, x: torch.Tensor, class_id: int) -> torch.Tensor:
        cid = str(class_id)
        mu = self.means[cid]
        inv = self.cov_invs[cid]
        logdet = self.log_dets[cid]
        prior = torch.log(torch.tensor(self.priors[class_id], device=x.device))
        xc = x - mu.unsqueeze(0)
        maha = 0.5 * torch.sum(xc @ inv * xc, dim=1)
        return -maha - 0.5 * logdet + prior

    def _lda_discriminant(self, x: torch.Tensor, class_id: int) -> torch.Tensor:
        cid = str(class_id)
        mu = self.means[cid]
        inv = self.shared_inv
        prior = torch.log(torch.tensor(self.priors[class_id], device=x.device))
        t1 = x @ inv @ mu
        t2 = 0.5 * (mu @ inv @ mu)
        return t1 - t2 + prior

    # ================= 实用函数 =================
    def predict(self, x: torch.Tensor):
        return torch.argmax(self.forward(x), dim=1)

    def predict_proba(self, x: torch.Tensor):
        return F.softmax(self.forward(x), dim=1)
