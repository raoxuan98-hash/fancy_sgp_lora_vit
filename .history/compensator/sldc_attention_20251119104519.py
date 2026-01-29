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
    def compensate(self, stats_dict, base_temperature=0.05, top_k=2000, n_samples=2000, chunk_size=128):
        assert self.is_trained, "SDC 尚未训练"
        out = {}
        fb = self.features_before
        drift = self.drift_vectors
        d = fb.size(1)
        
        for cid, stat in stats_dict.items():
            mu = stat.mean.to(self.device)
            cov = stat.cov.to(self.device)
            
            # 均值补偿（保持不变）
            fb_norm = F.normalize(fb, dim=1)
            mu_norm = F.normalize(mu.unsqueeze(0), dim=1)
            similarities = torch.matmul(fb_norm, mu_norm.t()).squeeze(1)
            
            temperature = base_temperature
            attention_weights = F.softmax(similarities / temperature, dim=0)
            
            if top_k > 0 and top_k < len(attention_weights):
                top_k_current = min(top_k, len(attention_weights))
                top_indices = torch.topk(attention_weights, top_k_current).indices
                mask = torch.zeros_like(attention_weights)
                mask[top_indices] = attention_weights[top_indices]
                mask = mask / mask.sum()
                attention_weights = mask
            
            drift_c = torch.sum(attention_weights.unsqueeze(1) * drift, dim=0)
            
            if self.compensate_cov:
                # 分块处理样本
                compensated_samples = []
                
                for i in range(0, n_samples, chunk_size):
                    chunk_end = min(i + chunk_size, n_samples)
                    chunk_size_actual = chunk_end - i
                    
                    # 为当前分块生成随机数
                    cached_eps_chunk = torch.randn(chunk_size_actual, d, device=self.device)
                    samples_chunk = stat.sample(cached_eps=cached_eps_chunk).to(self.device)
                    
                    # 分块计算相似度
                    samples_norm_chunk = F.normalize(samples_chunk, dim=1)
                    fb_norm_expanded = F.normalize(fb, dim=1)
                    
                    sample_similarities_chunk = torch.matmul(samples_norm_chunk, fb_norm_expanded.t())
                    sample_attention_weights_chunk = F.softmax(sample_similarities_chunk / temperature, dim=1)
                    
                    if top_k > 0 and top_k < sample_attention_weights_chunk.size(1):
                        top_vals, top_indices = torch.topk(sample_attention_weights_chunk, 
                                                        min(top_k, sample_attention_weights_chunk.size(1)), 
                                                        dim=1)
                        mask = torch.zeros_like(sample_attention_weights_chunk)
                        mask.scatter_(1, top_indices, top_vals)
                        mask = mask / mask.sum(dim=1, keepdim=True)
                        sample_attention_weights_chunk = mask
                    
                    # 应用漂移
                    compensated_chunk = samples_chunk + torch.sum(
                        sample_attention_weights_chunk.unsqueeze(2) * drift.unsqueeze(0), dim=1
                    )
                    compensated_samples.append(compensated_chunk.cpu())
                    
                    # 及时清理GPU内存
                    del samples_chunk, compensated_chunk, sample_attention_weights_chunk
                
                # 合并所有分块
                compensated_samples = torch.cat(compensated_samples, dim=0)
                
                mu_new = compensated_samples.mean(dim=0).cpu()
                cov_new = 0.9 * torch.cov(compensated_samples.T).cpu() + 0.1 * cov.cpu()
                
            else:
                mu_new = (mu + drift_c).cpu()
                cov_new = cov.cpu()

            out[cid] = GaussianStatistics(mu_new, cov_new, stat.reg)
            
            # 清理临时变量
            del mu, cov, samples, compensated_samples, mu_new, cov_new
            
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return out