# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict
from compensator.gaussian_statistics import GaussianStatistics


class RegularizedGaussianClassifier(nn.Module):
    """
    基于高斯统计构建的解析判别分类器。
    支持 LDA (共享协方差) 与 QDA (类别独立协方差)，并包含多种协方差正则化策略。
    """
    def __init__(self, 
                 stats_dict: Dict[int, GaussianStatistics],
                 class_priors: Dict[int, float] = None,
                 mode: str = "qda",
                 reg_alpha: float = 0.1,
                 reg_type: str = "shrinkage",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.mode = mode
        self.reg_alpha = reg_alpha
        self.reg_type = reg_type
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
        means = torch.stack(means)
        covs = torch.stack(covs)

        self.global_mean = nn.Parameter(means.mean(0), requires_grad=False)
        self.global_cov = nn.Parameter(covs.mean(0), requires_grad=False)

        # 协方差正则化函数
        def regularize(cov):
            d = cov.size(-1)
            if self.reg_type == "shrinkage":
                return (1 - self.reg_alpha) * cov + self.reg_alpha * self.global_cov
            elif self.reg_type == "diagonal":
                diag_cov = torch.diag_embed(torch.diagonal(cov, dim1=-2, dim2=-1))
                return (1 - self.reg_alpha) * cov + self.reg_alpha * diag_cov
            elif self.reg_type == "spherical":
                trace = torch.trace(cov) / d
                sph = trace * torch.eye(d, device=cov.device)
                return (1 - self.reg_alpha) * cov + self.reg_alpha * sph
            else:
                return cov

        # 协方差正则化 + 稳定化
        covs_reg = []
        for cov in covs:
            cov_r = regularize(cov)
            cov_r = 0.5 * (cov_r + cov_r.T)
            cov_r += self.epsilon * torch.eye(cov_r.size(-1), device=self.device)
            covs_reg.append(cov_r)
        covs_reg = torch.stack(covs_reg)

        # 安全求逆与 logdet
        invs = []
        logdets = []
        for cov in covs_reg:
            try:
                L = torch.linalg.cholesky(cov)
                inv = torch.cholesky_inverse(L)
                logdet = 2 * torch.sum(torch.log(torch.diag(L)))
            except Exception:
                U, S, Vh = torch.linalg.svd(cov)
                inv = U @ torch.diag(1.0 / S) @ Vh
                logdet = torch.sum(torch.log(S))
            invs.append(inv)
            logdets.append(logdet)
        invs = torch.stack(invs)
        logdets = torch.tensor(logdets, device=self.device)

        # 存储参数
        self.means = nn.ParameterDict()
        self.covs = nn.ParameterDict()
        self.cov_invs = nn.ParameterDict()
        self.log_dets = nn.ParameterDict()

        for i, cid in enumerate(self.class_ids):
            cid_str = str(cid)
            self.means[cid_str] = nn.Parameter(means[i], requires_grad=False)
            self.covs[cid_str] = nn.Parameter(covs_reg[i], requires_grad=False)
            self.cov_invs[cid_str] = nn.Parameter(invs[i], requires_grad=False)
            self.log_dets[cid_str] = nn.Parameter(logdets[i].unsqueeze(0), requires_grad=False)

        # LDA 共享协方差
        if self.mode == "lda":
            shared_cov = covs_reg.mean(0)
            shared_cov = 0.5 * (shared_cov + shared_cov.T)
            shared_cov += self.epsilon_identity * torch.eye(shared_cov.size(-1), device=self.device)
            try:
                L = torch.linalg.cholesky(shared_cov)
                shared_inv = torch.cholesky_inverse(L)
                shared_logdet = 2 * torch.sum(torch.log(torch.diag(L)))
            except:
                U, S, Vh = torch.linalg.svd(shared_cov)
                shared_inv = U @ torch.diag(1.0 / S) @ Vh
                shared_logdet = torch.sum(torch.log(S))
            self.shared_cov = nn.Parameter(shared_cov, requires_grad=False)
            self.shared_inv = nn.Parameter(shared_inv, requires_grad=False)
            self.shared_logdet = nn.Parameter(shared_logdet.unsqueeze(0), requires_grad=False)

        logging.info(f"[INFO] RegularizedGaussianClassifier initialized: {self.num_classes} classes, mode={mode}")

    # ======================================================
    #                    判别函数
    # ======================================================
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
        t2 = 0.5 * mu @ inv @ mu
        return t1 - t2 + prior

    # ======================================================
    #                    实用函数
    # ======================================================
    def predict(self, x: torch.Tensor):
        return torch.argmax(self.forward(x), dim=1)

    def predict_proba(self, x: torch.Tensor):
        return F.softmax(self.forward(x), dim=1)
