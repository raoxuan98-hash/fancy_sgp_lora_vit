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
    def compensate(self, stats_dict, base_temperature=0.05, top_k=2000, n_samples=2000):
        assert self.is_trained, "SDC 尚未训练"
        out = {}
        fb = self.features_before
        drift = self.drift_vectors

        for cid, stat in stats_dict.items():
            mu = stat.mean.to(self.device)
            cov = stat.cov.to(self.device)
            
            # 计算注意力权重
            fb_norm = F.normalize(fb, dim=1)
            mu_norm = F.normalize(mu.unsqueeze(0), dim=1)
            similarities = torch.matmul(fb_norm, mu_norm.t()).squeeze(1)  # [n_samples]
            
            temperature = base_temperature
            attention_weights = F.softmax(similarities / temperature, dim=0)
            
            if top_k > 0 and top_k < len(attention_weights):
                top_k = min(top_k, len(attention_weights))
                top_indices = torch.topk(attention_weights, top_k).indices
                
                mask = torch.zeros_like(attention_weights)
                mask[top_indices] = attention_weights[top_indices]
                mask = mask / mask.sum()
                attention_weights = mask
            
            # 计算漂移向量
            drift_c = torch.sum(attention_weights.unsqueeze(1) * drift, dim=0)
            
            if self.compensate_cov:
                # 通过样本重计算来补偿协方差
                
                samples = stat.sample(n_samples).to(self.device)  # 从原始分布采样
                
                # 对每个样本计算其对应的注意力权重和漂移向量
                samples_norm = F.normalize(samples, dim=1)
                fb_norm_expanded = F.normalize(fb, dim=1)
                
                # 计算每个样本与所有训练特征的相似度
                sample_similarities = torch.matmul(samples_norm, fb_norm_expanded.t())  # [n_samples, n_train]
                
                # 对每个样本应用相同的注意力机制
                sample_attention_weights = F.softmax(sample_similarities / temperature, dim=1)
                
                if top_k > 0 and top_k < sample_attention_weights.size(1):
                    # 对每个样本保留top-k个最大的注意力权重
                    top_vals, top_indices = torch.topk(sample_attention_weights, top_k, dim=1)
                    mask = torch.zeros_like(sample_attention_weights)
                    mask.scatter_(1, top_indices, top_vals)
                    mask = mask / mask.sum(dim=1, keepdim=True)
                    sample_attention_weights = mask
                
                # 对每个样本应用对应的漂移
                compensated_samples = samples + torch.sum(
                    sample_attention_weights.unsqueeze(2) * drift.unsqueeze(0), dim=1
                )
                
                # 从补偿后的样本重新计算统计量
                mu_new = compensated_samples.mean(dim=0).cpu()
                cov_new = 0.9 * torch.cov(compensated_samples.T).cpu() + 0.1 * cov.cpu()

                del samples
                
            else:
                # 只补偿均值（保持原始协方差）
                mu_new = (mu + drift_c).cpu()
                cov_new = cov.cpu()

            out[cid] = GaussianStatistics(mu_new, cov_new, stat.reg)

        return out