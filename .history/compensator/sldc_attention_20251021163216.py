# compensator/sdc_compensator.py
import torch
from compensator.gaussian_statistics import GaussianStatistics
from compensator.base_compensator import BaseCompensator
import torch.nn.functional as F

class AttentionCompensator(BaseCompensator):
    """基于样本间距的语义漂移补偿器 (SDC)"""
    def __init__(self, input_dim: int, sigma: float = 1.0, device="cuda"):
        super().__init__(input_dim, device)
        self.sigma = sigma
        self.drift_vectors = None
        self.features_before = None
        self.features_after = None

    def train(self, features_before, features_after):
        self.features_before = features_before.to(self.device)
        self.features_after = features_after.to(self.device)
        self.drift_vectors = self.features_after - self.features_before
        self.is_trained = True

    @torch.no_grad()
    def compensate(self, stats_dict, base_temperature=0.1, top_k=500):
        assert self.is_trained, "SDC 尚未训练"
        out = {}
        fb = self.features_before
        drift = self.drift_vectors

        for cid, stat in stats_dict.items():
            mu = stat.mean.to(self.device)
            cov = stat.cov.to(self.device)
            fb_norm = F.normalize(fb, dim=1)
            mu_norm = F.normalize(mu.unsqueeze(0), dim=1)
            similarities = torch.matmul(fb_norm, mu_norm.t()).squeeze(1)  # [n_samples]
            
            # 1. 不中心化相似度
            # 2. 使用固定的温度参数
            temperature = base_temperature
            attention_weights = F.softmax(similarities / temperature, dim=0)
            
            # 3. 取top-k个最大的特征进行加权
            if top_k > 0 and top_k < len(attention_weights):
                top_k = min(top_k, len(attention_weights))
                top_indices = torch.topk(attention_weights, top_k).indices
                
                # 创建新的注意力权重，只保留top-k，其余置0
                mask = torch.zeros_like(attention_weights)
                mask[top_indices] = attention_weights[top_indices]
                
                # 重新归一化top-k权重
                mask = mask / mask.sum()
                attention_weights = mask
            
            drift_c = torch.sum(attention_weights.unsqueeze(1) * drift, dim=0)
            mu_new = mu + drift_c

            out[cid] = GaussianStatistics(mu_new.cpu(), cov.cpu())

        return out