# classifier/linear_lda_classifier.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from stats.gaussian_statistics import GaussianStatistics


class LinearLDAClassifier(nn.Module):
    """
    将 LDA 判别函数解析为一个线性层：logits = x @ W + b
    W_c = Σ⁻¹ μ_c
    b_c = -0.5 μ_cᵀ Σ⁻¹ μ_c + log π_c
    """
    def __init__(self, 
                 stats_dict,
                 class_priors=None,
                 reg_alpha=0.1,
                 reg_type="shrinkage",
                 device="cuda"):
        super().__init__()
        self.device = torch.device(device)
        self.reg_alpha = reg_alpha
        self.reg_type = reg_type

        # === Step 1. 组装全局协方差 ===
        class_ids = sorted(stats_dict.keys())
        self.num_classes = len(class_ids)
        means = torch.stack([stats_dict[cid].mean for cid in class_ids]).to(self.device)
        covs = torch.stack([stats_dict[cid].cov for cid in class_ids]).to(self.device)
        global_cov = covs.mean(0)
        global_cov = 0.5 * (global_cov + global_cov.T)

        # === Step 2. 协方差正则化 ===
        d = global_cov.size(0)
        if self.reg_type == "shrinkage":
            cov_reg = (1 - self.reg_alpha) * global_cov + self.reg_alpha * torch.eye(d, device=self.device)
        elif self.reg_type == "diagonal":
            diag = torch.diag_embed(torch.diagonal(global_cov))
            cov_reg = (1 - self.reg_alpha) * global_cov + self.reg_alpha * diag
        elif self.reg_type == "spherical":
            trace = torch.trace(global_cov) / d
            cov_reg = (1 - self.reg_alpha) * global_cov + self.reg_alpha * trace * torch.eye(d, device=self.device)
        else:
            cov_reg = global_cov

        # === Step 3. 计算逆矩阵 ===
        cov_reg += 1e-6 * torch.eye(d, device=self.device)
        cov_inv = torch.linalg.inv(cov_reg)

        # === Step 4. 权重 & 偏置解析解 ===
        priors = {cid: 1.0 / self.num_classes for cid in class_ids} if class_priors is None else class_priors
        W = []
        b = []
        for i, cid in enumerate(class_ids):
            mu = means[i]
            w_c = cov_inv @ mu
            logpi = torch.log(torch.tensor(priors[cid], device=self.device))
            b_c = -0.5 * (mu @ cov_inv @ mu) + logpi
            W.append(w_c)
            b.append(b_c)
        W = torch.stack(W, dim=1)  # [D, C]
        b = torch.stack(b)         # [C]

        # === Step 5. 构建线性层 ===
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
