
import torch
import torch.nn.functional as F
import logging
from compensator.gaussian_statistics import GaussianStatistics
from compensator.base_compensator import BaseCompensator

logger = logging.getLogger(__name__)


class RFFDriftCompensator(BaseCompensator):
    """基于核化注意力（FAVOR+）的语义漂移补偿器"""

    def __init__(
        self,
        input_dim: int,
        device="cuda",
        compensate_cov: bool = True,
        random_feature_dim=512,  # m in FAVOR+
        use_orthogonal=False,     # 是否使用正交随机特征（更稳定）
    ):
        super().__init__(input_dim, device)
        self.compensate_cov = compensate_cov
        self.random_feature_dim = random_feature_dim
        self.use_orthogonal = use_orthogonal
        self.drift_vectors = None
        self.features_before = None
        self.features_after = None
        self._omega = None  # 随机投影矩阵 (d, m)

    def _create_omega(self, d, m, device):
        """生成随机或正交的投影矩阵 omega ~ N(0, I)"""
        if self.use_orthogonal and d >= m:
            # 生成正交矩阵（QR 分解）
            random_mat = torch.randn(d, d, device=device)
            q, _ = torch.linalg.qr(random_mat)
            omega = q[:, :m]
        else:
            omega = torch.randn(d, m, device=device)
        # 缩放：FAVOR+ 要求 omega ~ N(0, I / tau^2)，但我们通过 temperature 外部控制
        return omega

    def train(self, features_before, features_after):
        self.features_before = features_before.to(self.device)
        self.features_after = features_after.to(self.device)
        self.drift_vectors = self.features_after - self.features_before
        d = features_before.shape[1]
        self._omega = self._create_omega(d, self.random_feature_dim, device=self.device)
        self.is_trained = True


    @torch.no_grad()
    def compensate(
        self,
        stats_dict,
        base_temperature=0.1,
        top_k=1000,
        n_samples=2000,
        chunk_size=512,
    ):
        assert self.is_trained, "KernelizedAttentionCompensator 尚未训练"

        out = {}
        fb = self.features_before  # [N, d]
        drift = self.drift_vectors
        fb_norm = F.normalize(fb, dim=1)
        N, d = fb.size()

        # 全局噪声，用于协方差补偿
        global_eps = torch.randn(n_samples, d, device=self.device)

        for cid, stat in stats_dict.items():
            mu = stat.mean.to(self.device)
            cov = stat.cov.to(self.device)
            # === 协方差补偿（复用 global_eps，分块）===
            compensated_samples = []
            for i in range(0, n_samples, chunk_size):
                end = min(i + chunk_size, n_samples)
                eps_chunk = global_eps[i:end]  # [chunk, d]

                samples = stat.sample(cached_eps=eps_chunk).to(self.device)  # [chunk, d]
                samples_norm = F.normalize(samples, dim=1)  # [chunk, d]

                # 对每个样本，计算其核化注意力（可并行）
                # 注意：这里 queries = samples_norm [B, d], keys = fb_norm [N, d]
                B = samples_norm.size(0)
                # 批量投影
                scale = 1.0 / base_temperature
                q_scaled = samples_norm * scale  # [B, d]
                k_scaled = fb_norm * scale       # [N, d]

                # 投影到随机特征空间
                q_proj = q_scaled @ self._omega  # [B, m]
                k_proj = k_scaled @ self._omega  # [N, m]

                phi_q_batch = torch.cat([torch.sin(q_proj), torch.cos(q_proj)], dim=-1)  # [B, 2m]
                phi_k = torch.cat([torch.sin(k_proj), torch.cos(k_proj)], dim=-1)      # [N, 2m]

                # 批量计算注意力：Attn[b, n] ∝ phi_q[b] · phi_k[n]
                k_sum = phi_k.sum(dim=0, keepdim=True)         # [1, 2m]
                denominator = phi_q_batch @ k_sum.t()          # [B, 1]
                numerator = phi_q_batch @ phi_k.t()            # [B, N]

                att = numerator / (denominator + 1e-8)         # [B, N]
                att = torch.clamp(att, min=0.0)
                att = att / (att.sum(dim=1, keepdim=True) + 1e-12)

                # top-k 稀疏化（可选）
                if top_k > 0 and top_k < N:
                    k = min(top_k, N)
                    top_vals, top_indices = torch.topk(att, k, dim=1, sorted=False)
                    mask = torch.zeros_like(att)
                    mask.scatter_(1, top_indices, top_vals)
                    att = mask / mask.sum(dim=1, keepdim=True).clamp(min=1e-12)

                drift_applied = torch.einsum('bn,nd->bd', att, drift)
                compensated_chunk = samples + drift_applied
                compensated_samples.append(compensated_chunk.cpu())

            compensated_samples = torch.cat(compensated_samples, dim=0)
            mu_new = 0.9 * compensated_samples.mean(dim=0) + 0.1 * mu.cpu()
            cov_new = 0.9 * torch.cov(compensated_samples.T) + 0.1 * cov.cpu()
            out[cid] = GaussianStatistics(mu_new, cov_new, stat.reg)

        return out