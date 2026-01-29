# compensator/kernelized_compensator.py

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
        random_feature_dim=512,
        use_orthogonal=False,
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
        return omega

    def train(self, features_before, features_after):
        self.features_before = features_before.to(self.device)
        self.features_after = features_after.to(self.device)
        self.drift_vectors = self.features_after - self.features_before
        self.is_trained = True
        self._omega = None  # 重置投影矩阵，将在第一次使用时创建

    @torch.no_grad()
    def _compute_kernel_attention(self, query, keys, temperature=0.1, top_k=None):
        """
        计算核化注意力权重
        
        Args:
            query: [B, d] 查询向量
            keys: [N, d] 键向量
            temperature: 温度参数
            top_k: 是否使用top-k稀疏化，None表示不使用
            
        Returns:
            attention: [B, N] 注意力权重矩阵
        """
        B, d = query.shape
        N, _ = keys.shape
        m = self.random_feature_dim
        
        # 确保投影矩阵存在
        if self._omega is None or self._omega.shape != (d, m):
            self._omega = self._create_omega(d, m, device=query.device)
        
        omega = self._omega
        
        # 缩放
        scale = temperature ** 0.5
        query_scaled = query / scale
        keys_scaled = keys / scale
        
        # 投影到随机特征空间
        q_proj = query_scaled @ omega  # [B, m]
        k_proj = keys_scaled @ omega   # [N, m]
        
        # 构造正交随机特征 (sin + cos)
        phi_q = torch.cat([torch.sin(q_proj), torch.cos(q_proj)], dim=-1)  # [B, 2m]
        phi_k = torch.cat([torch.sin(k_proj), torch.cos(k_proj)], dim=-1)  # [N, 2m]
        
        # 批量计算注意力
        k_sum = phi_k.sum(dim=0, keepdim=True)  # [1, 2m]
        denominator = phi_q @ k_sum.t()          # [B, 1]
        numerator = phi_q @ phi_k.t()            # [B, N]
        
        attention = numerator / (denominator + 1e-8)  # [B, N]
        attention = torch.clamp(attention, min=0.0)
        attention = attention / (attention.sum(dim=1, keepdim=True) + 1e-12)
        
        # top-k 稀疏化（可选）
        if top_k is not None and top_k > 0 and top_k < N:
            k = min(top_k, N)
            top_vals, top_indices = torch.topk(attention, k, dim=1, sorted=False)
            mask = torch.zeros_like(attention)
            mask.scatter_(1, top_indices, top_vals)
            attention = mask / mask.sum(dim=1, keepdim=True).clamp(min=1e-12)
        
        return attention

    @torch.no_grad()
    def compensate(
        self,
        stats_dict,
        temperature=0.1,
        top_k=1000,
        n_samples=2000,
    ):
        """
        补偿语义漂移
        
        Args:
            stats_dict: 类别ID到高斯统计量的字典
            temperature: 注意力温度参数
            top_k: 注意力稀疏化的top-k值
            n_samples: 采样数量
            
        Returns:
            out: 补偿后的高斯统计量字典
        """
        assert self.is_trained, "RFFDriftCompensator 尚未训练"

        out = {}
        fb = self.features_before  # [N, d]
        drift = self.drift_vectors  # [N, d]
        fb_norm = F.normalize(fb, dim=1)
        N, d = fb.size()

        # 预先生成全局噪声
        global_eps = torch.randn(n_samples, d, device=self.device)

        for cid, stat in stats_dict.items():
            # 一次性采样所有样本
            samples = stat.sample(cached_eps=global_eps).to(self.device)  # [n_samples, d]
            samples_norm = F.normalize(samples, dim=1)  # [n_samples, d]

            # 使用核化注意力计算权重
            attention = self._compute_kernel_attention(
                query=samples_norm,
                keys=fb_norm,
                temperature=temperature,
                top_k=top_k
            )  # [n_samples, N]

            # 应用漂移补偿
            drift_applied = torch.einsum('bn,nd->bd', attention, drift)
            compensated_samples = samples + drift_applied
            mu_new = compensated_samples.mean(dim=0)
            cov_new = torch.cov(compensated_samples.T)
            out[cid] = GaussianStatistics(mu_new, cov_new, stat.reg)

        return out
