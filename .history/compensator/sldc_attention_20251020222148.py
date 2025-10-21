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
    def compensate(self, stats_dict, base_temperature=0.1):
        assert self.is_trained, "SDC 尚未训练"
        out = {}
        fb = self.features_before
        drift = self.drift_vectors

        for cid, stat in stats_dict.items():
            mu = stat.mean.to(self.device)
            cov = stat.cov.to(self.device)
            fb_norm = F.normalize(fb, dim=1)
            mu_norm = F.normalize(mu.unsqueeze(0), dim=1)
            similarities = torch.matmul(fb_norm, mu_norm.t()).squeeze(1)  # [-1, 1]
            similarities = similarities - similarities.mean()
            scale = similarities.std().clamp(min=1e-6)
            temperature = base_temperature * scale
            attention_weights = F.softmax(similarities / temperature, dim=0)
            drift_c = torch.sum(attention_weights.unsqueeze(1) * drift, dim=0)
            mu_new = mu + drift_c

            out[cid] = GaussianStatistics(mu_new.cpu(), cov.cpu())

        return out

