# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from compensator.gaussian_statistics import GaussianStatistics
from compensator.base_compensator import BaseCompensator

class ResidMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.log_scale = nn.Parameter(torch.tensor(0.0))
        self.fc1 = nn.Linear(dim, dim, bias=False)
        self.fc1.weight.data = torch.eye(dim)
        self.fc2 = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.ReLU(),
            nn.Linear(dim, dim, bias=False)
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

class WeakNonlinearCompensator(BaseCompensator):
    """弱非线性补偿器 (Residual MLP)"""
    def __init__(self, input_dim: int, device="cuda"):
        super().__init__(input_dim, device)
        self.net = ResidMLP(input_dim).to(self.device)

    def train(self, features_before: torch.Tensor, features_after: torch.Tensor, epochs: int = 4000, lr: float = 0.001):
        device = features_before.device
        self.net = self.net.to(device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-5)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr / 3)
        crit = nn.MSELoss()
        X = features_before
        Y = features_after
        for ep in range(epochs):
            idx = torch.randint(0, X.size(0), (64,), device=device)
            pred = self.net(X[idx])
            loss = crit(pred, Y[idx]) + 0.5 * self.net.reg_loss()
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
            if (ep + 1) % 2000 == 0:
                logging.info(f"[SLDC WeakNonlinearTransform] step {ep+1}/{epochs}, loss={loss.item():.6f}")
        self.is_trained = True

    @torch.no_grad()
    def transform_features(self, features: torch.Tensor) -> torch.Tensor:
        if not self.is_trained:
            raise ValueError("非线性变换器尚未训练。")
        return self.net(features)

    @torch.no_grad()
    def compensate(self, stats_dict, n_samples=5000):
        assert self.is_trained, "WeakNonlinearCompensator 尚未训练"
        device = self.device
        out = {}
        for cid, s in stats_dict.items():
            samples = s.sample(n_samples).to(device)
            transformed = self.net(samples)
            mu_new = transformed.mean(0).cpu()
            cov_new = torch.cov(transformed.T).cpu()
            out[cid] = GaussianStatistics(mu_new, cov_new, s.reg)
            
            # 清理临时变量
            del transformed, mu_new, cov_new
            
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return out
    

class RFFDriftCompensator(BaseCompensator):
    """
    基于 RFF 的残差预测补偿器：
      - 学习映射: phi(x_before) → drift = x_after - x_before
      - 补偿时: x_comp = x + W^T phi(x)
    """

    def __init__(
        self,
        input_dim: int,
        rff_dim: int = 2048,
        gamma: float = 1e-4,
        compensate_cov: bool = True,
        device="cuda"
    ):
        super().__init__(input_dim, device)
        self.rff_dim = rff_dim
        self.gamma = gamma
        self.compensate_cov = compensate_cov

        # RFF parameters (fixed)
        self.register_buffer('omega', torch.randn(rff_dim, input_dim, device=device))
        self.register_buffer('bias', torch.rand(rff_dim, device=device) * 2 * torch.pi)

        self.W = None  # shape: (rff_dim, input_dim), so that drift ≈ phi(x) @ W

    def _rff_map(self, x):
        """x: (..., d) → phi(x): (..., D)"""
        proj = x @ self.omega.t() + self.bias  # (..., D)
        phi = torch.cos(proj) * (2.0 / self.rff_dim) ** 0.5
        return phi

    def train(self, features_before, features_after):
        X = features_before.to(self.device)
        Y = features_after.to(self.device)
        drift = Y - X  # (N, d)

        # Normalize features (optional but recommended for RFF)
        X = F.normalize(X, dim=1)

        # Map to RFF space
        Phi_X = self._rff_map(X)  # (N, D)

        # Solve: Phi_X @ W ≈ drift  →  W = argmin ||Phi_X W - drift||^2 + γ||W||^2
        # (D, d) solution
        try:
            ATA = Phi_X.t() @ Phi_X + self.gamma * torch.eye(self.rff_dim, device=self.device)
            ATb = Phi_X.t() @ drift
            W = torch.linalg.solve(ATA, ATb)  # (D, d)
        except RuntimeError:
            W = torch.linalg.pinv(Phi_X) @ drift

        self.W = W
        self.is_trained = True
        return self.W

    @torch.no_grad()
    def compensate(
        self,
        stats_dict,
        n_samples=2000,
        chunk_size=512,
    ):
        assert self.is_trained, "RFFDriftCompensator 尚未训练"
        W = self.W  # (D, d) on device
        out = {}

        for cid, stat in stats_dict.items():
            mu = stat.mean.to(self.device)  # (d,)
            cov = stat.cov.to(self.device)  # (d, d)

            # --- 均值补偿 ---
            mu_norm = F.normalize(mu.unsqueeze(0), dim=1).squeeze(0)  # (d,)
            phi_mu = self._rff_map(mu_norm.unsqueeze(0))  # (1, D)
            drift_mu = (phi_mu @ W).squeeze(0)  # (d,)
            mu_new = mu + drift_mu  # (d,)

            if not self.compensate_cov:
                out[cid] = GaussianStatistics(mu_new.cpu(), cov.cpu(), stat.reg)
                continue

            # --- 协方差补偿：采样 + 残差预测 ---
            compensated_samples = []

            # 生成随机噪声（可复用，但每次类独立也可以）
            global_eps = torch.randn(n_samples, self.input_dim, device=self.device)

            for i in range(0, n_samples, chunk_size):
                end = min(i + chunk_size, n_samples)
                eps_chunk = global_eps[i:end]  # (chunk, d)

                # 从原始高斯分布采样
                samples = stat.sample(cached_eps=eps_chunk).to(self.device)  # (chunk, d)
                samples_norm = F.normalize(samples, dim=1)  # (chunk, d)

                # RFF 映射
                phi_samples = self._rff_map(samples_norm)  # (chunk, D)

                # 预测残差
                drift_pred = phi_samples @ W  # (chunk, d)

                # 补偿样本
                compensated_chunk = samples + drift_pred  # (chunk, d)
                compensated_samples.append(compensated_chunk.cpu())

            compensated_samples = torch.cat(compensated_samples, dim=0)  # (n_samples, d)
            mu_est = compensated_samples.mean(dim=0)
            cov_est = torch.cov(compensated_samples.T)
            mu_final = 0.9 * mu_est + 0.1 * mu_new.cpu()
            cov_final = 0.9 * cov_est + 0.1 * cov.cpu()
            out[cid] = GaussianStatistics(mu_final, cov_final, stat.reg)

        # 清理缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return out