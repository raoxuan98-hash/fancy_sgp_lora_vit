# compensator/sdc_compensator.py
import torch
from compensator.gaussian_statistics import GaussianStatistics
from compensator.base_compensator import BaseCompensator
import torch.nn.functional as F

class AttentionCompensator(BaseCompensator):
    """基于样本间距的语义漂移补偿器 (SDC)"""
    def __init__(self, input_dim: int, device="cuda", 
                 compensate_cov: bool = True):
        super().__init__(input_dim, device)
        self.compensate_cov = compensate_cov  # 控制是否补偿协方差
        self.drift_vectors = None
        self.features_before = None
        self.features_after = None

    def train(self, features_before, features_after):
        self.features_before = features_before.to(self.device)
        self.features_after = features_after.to(self.device)
        self.drift_vectors = self.features_after - self.features_before
        self.is_trained = True

    @torch.no_grad()
    def compensate(self, stats_dict, base_temperature=0.05, top_k=1000, n_samples=2000, chunk_size=512):
        assert self.is_trained, "SDC 尚未训练"
        out = {}
        fb = self.features_before  # assumed on device
        drift = self.drift_vectors
        fb_norm = F.normalize(fb, dim=1)
        N, d = fb.size()
        
        # ✅ 关键优化：预生成全局 eps（在 device 上）
        # Shape: [n_samples, d]
        global_eps = torch.randn(n_samples, d, device=self.device)

        for cid, stat in stats_dict.items():
            mu = stat.mean.to(self.device)
            cov = stat.cov.to(self.device)
            
            # --- 均值补偿 ---
            mu_norm = F.normalize(mu.unsqueeze(0), dim=1)
            similarities = torch.matmul(fb_norm, mu_norm.t()).squeeze(1)
            attention_weights = F.softmax(similarities / base_temperature, dim=0)
            
            if top_k > 0 and top_k < N:
                top_vals, top_indices = torch.topk(attention_weights, top_k, sorted=False)
                mask = torch.zeros_like(attention_weights)
                mask[top_indices] = top_vals
                attention_weights = mask / mask.sum().clamp(min=1e-12)
            
            drift_c = (attention_weights.unsqueeze(1) * drift).sum(dim=0)
            
            if not self.compensate_cov:
                out[cid] = GaussianStatistics((mu + drift_c).cpu(), cov.cpu(), stat.reg)
                continue

            # --- 协方差补偿：复用 global_eps，分块处理 ---
            compensated_samples = []
            for i in range(0, n_samples, chunk_size):
                end = min(i + chunk_size, n_samples)
                eps_chunk = global_eps[i:end]  # ✅ 复用预生成的噪声
                
                # 使用 GaussianStatistics 的 sample 方法（需支持 cached_eps）
                samples = stat.sample(cached_eps=eps_chunk)  # [chunk, d]
                
                # Normalization and attention
                samples_norm = F.normalize(samples, dim=1)
                sim = torch.matmul(samples_norm, fb_norm.t())
                att = F.softmax(sim / base_temperature, dim=1)
                
                if top_k > 0 and top_k < N:
                    k = min(top_k, N)
                    top_vals, top_indices = torch.topk(att, k, dim=1, sorted=False)
                    mask = torch.zeros_like(att)
                    mask.scatter_(1, top_indices, top_vals)
                    att = mask / mask.sum(dim=1, keepdim=True).clamp(min=1e-12)
                
                # Apply drift
                drift_applied = torch.einsum('bn,nd->bd', att, drift)
                compensated_chunk = samples + drift_applied
                compensated_samples.append(compensated_chunk.cpu())
            
            compensated_samples = torch.cat(compensated_samples, dim=0)
            mu_new = 0.8*compensated_samples.mean(dim=0) + 0.2 * mu
            cov_new = 0.8 * torch.cov(compensated_samples.T) + 0.2 * cov.cpu()
            out[cid] = GaussianStatistics(mu_new, cov_new, stat.reg)
        
        return out