# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict
from compensator.gaussian_statistics import GaussianStatistics


def get_gpu_memory_info() -> Dict[str, float]:
    """获取当前GPU显存信息"""
    if not torch.cuda.is_available():
        return {"allocated": 0.0, "reserved": 0.0, "max_allocated": 0.0}
    
    return {
        "allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
        "reserved": torch.cuda.memory_reserved() / 1024**3,    # GB
        "max_allocated": torch.cuda.max_memory_allocated() / 1024**3  # GB
}


def log_memory_usage(operation_name: str, start_memory: Dict[str, float], end_memory: Dict[str, float]):
    """记录显存使用情况"""
    allocated_diff = end_memory["allocated"] - start_memory["allocated"]
    reserved_diff = end_memory["reserved"] - start_memory["reserved"]
    
    logging.info(
        f"[GPU Memory] {operation_name}: "
        f"Allocated={end_memory['allocated']:.2f}GB "
        f"(+{allocated_diff:.2f}GB), "
        f"Reserved={end_memory['reserved']:.2f}GB "
        f"(+{reserved_diff:.2f}GB), "
        f"Max={end_memory['max_allocated']:.2f}GB"
    )


class LinearLDAClassifier(nn.Module):
    def __init__(
        self,
        stats_dict,
        class_priors=None,
        lda_reg_alpha: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        init_device = torch.device(device)
        self.register_buffer("_device_indicator", torch.empty(0, device=init_device), persistent=False)
        self.lda_reg_alpha = lda_reg_alpha

        # === Step 1. 流式计算全局协方差（避免 stack 所有 cov）===
        class_ids = sorted(stats_dict.keys())
        self.num_classes = len(class_ids)

        # 获取维度 d（从第一个类的协方差推断）
        first_cid = class_ids[0]
        d = stats_dict[first_cid].cov.size(0)

        # 初始化累加器
        global_cov = torch.zeros(d, d, device=device)
        means_list = []

        # 流式遍历每个类
        for cid in class_ids:
            mu = stats_dict[cid].mean.to(device)
            Sigma = stats_dict[cid].cov.to(device)
            means_list.append(mu)
            global_cov += Sigma

        global_cov = global_cov / self.num_classes
        means = torch.stack(means_list)  # 仍需 means 用于后续计算

        # === Step 2. spherical 正则 ===
        cov_reg = (1.0 - self.lda_reg_alpha) * global_cov + self.lda_reg_alpha * torch.eye(d, device=device)

        # === Step 3. 计算逆矩阵 ===
        cov_reg = cov_reg + 1e-6 * torch.eye(d, device=device)
        cov_inv = torch.cholesky_inverse(cov_reg)

        # === Step 4. 权重 & 偏置解析解 ===
        priors = {cid: 1.0 / self.num_classes for cid in class_ids} if class_priors is None else class_priors
        W, b = [], []
        for i, cid in enumerate(class_ids):
            mu = means[i]
            w_c = cov_inv @ mu
            logpi = torch.log(torch.tensor(priors[cid], device=device))
            b_c = -0.5 * (mu @ cov_inv @ mu) + logpi
            W.append(w_c)
            b.append(b_c)
        W = torch.stack(W, dim=1)  # [D, C]
        b = torch.stack(b)         # [C]

        # === Step 5. 线性层承载 ===
        self.linear = nn.Linear(d, self.num_classes, bias=True)
        self.linear.weight.data = W.T.clone()
        self.linear.bias.data = b.clone()
        self.linear.requires_grad_(False)

    @property
    def device(self) -> torch.device:
        return self._device_indicator.device

    def forward(self, x):
        return self.linear(x)

    def predict(self, x):
        return torch.argmax(self.forward(x), dim=1)

    def predict_proba(self, x):
        return F.softmax(self.forward(x), dim=1)


class RegularizedGaussianDA(nn.Module):
    def __init__(
        self,
        stats_dict: Dict[int, GaussianStatistics],
        class_priors: Dict[int, float] = None,
        qda_reg_alpha1: float = 1.0,
        qda_reg_alpha2: float = 1.0,
        qda_reg_alpha3: float = 1.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        # 记录开始时的显存
        start_memory = get_gpu_memory_info()
        super().__init__()
        init_device = torch.device(device)
        self.register_buffer("_device_indicator", torch.empty(0, device=init_device), persistent=False)
        self.class_ids = sorted(stats_dict.keys())
        self.num_classes = len(self.class_ids)
        self.qda_reg_alpha1 = float(qda_reg_alpha1)
        self.qda_reg_alpha2 = float(qda_reg_alpha2)
        self.qda_reg_alpha3 = float(qda_reg_alpha3)
        self.epsilon = 1e-3   # per-class稳健性
        self.epsilon_identity = 1e-6

        # 类先验
        if class_priors is None:
            priors_list = [1.0 / self.num_classes for _ in self.class_ids]
        else:
            priors_list = [class_priors[cid] for cid in self.class_ids]
        device = self.device
        self.log_priors = nn.Parameter(torch.log(torch.tensor(priors_list, device=device)), requires_grad=False)  # [C]

        # 收集均值与协方差
        means = []
        covs = []
        for cid in self.class_ids:
            s = stats_dict[cid]
            means.append(s.mean.float().to(device))
            covs.append(s.cov.float().to(device))
        means = torch.stack(means)
        covs = torch.stack(covs)

        # 全局协方差（供 shrink 使用）
        global_cov = covs.mean(0)
        global_cov = 0.5 * (global_cov + global_cov.T)

        self.register_buffer("means", means, persistent=False)          # [C, D]
        self.register_buffer("global_cov", global_cov, persistent=False)

        C, D, _ = covs.shape
        sph = torch.eye(D, device=device).unsqueeze(0)
        a1, a2, a3 = self.qda_reg_alpha1, self.qda_reg_alpha2, self.qda_reg_alpha3
        covs_sym = 0.5 * (covs + covs.transpose(-1, -2))               # 数值对称化
        covs_reg = a1 * covs_sym + a2 * global_cov.unsqueeze(0) + a3 * sph
        covs_reg = covs_reg + self.epsilon * torch.eye(D, device=device).unsqueeze(0)

        # ---- 预计算每类逆阵与 logdet（Cholesky优先，失败回退SVD） ----
        cov_invs = []
        logdets = []
        for c in range(C):
            cov = covs_reg[c]
            try:
                L = torch.linalg.cholesky(cov)
                inv = torch.cholesky_inverse(L)
                logdet = 2 * torch.sum(torch.log(torch.diag(L)))

            except Exception:
                U, S, Vh = torch.linalg.svd(cov)
                inv = U @ torch.diag(1.0 / torch.clamp(S, min=1e-6)) @ Vh
                logdet = torch.sum(torch.log(torch.clamp(S, min=1e-6)))

            cov_invs.append(inv)
            logdets.append(logdet)

        cov_invs = torch.stack(cov_invs)
        logdets = torch.stack(logdets)

        self.register_buffer("cov_invs", cov_invs, persistent=False)
        self.register_buffer("logdets", logdets, persistent=False)

        # 清理中间变量以释放内存
        del means, covs, global_cov, covs_sym, covs_reg, cov_invs, logdets
        
        logging.info(
            f"[INFO] RegularizedGaussianQDA initialized: {self.num_classes} classes, "
            f"qda_reg_alpha1={self.qda_reg_alpha1}, qda_reg_alpha2={self.qda_reg_alpha2}, "
            f"qda_reg_alpha3={self.qda_reg_alpha3}")

    @property
    def device(self) -> torch.device:
        return self._device_indicator.device

    # ================= 判别函数（向量化） =================
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, D]
        返回：logits [B, C]
        """
        device = self.device
        x = x.to(device)
        B, D = x.shape
        C = self.means.shape[0]
        xc = x.unsqueeze(1) - self.means.unsqueeze(0)
        v = torch.einsum("bcd,cde->bce", xc, self.cov_invs)
        maha = 0.5 * (v * xc).sum(dim=-1)
        logits = -maha - 0.5 * self.logdets.unsqueeze(0) + self.log_priors.unsqueeze(0)
        return logits

    def predict(self, x: torch.Tensor):
        return torch.argmax(self.forward(x), dim=1)

    def predict_proba(self, x: torch.Tensor):
        return F.softmax(self.forward(x), dim=1)

# class LowRankGaussianDA(nn.Module):
#     def __init__(
#         self,
#         stats_dict: Dict[int, GaussianStatistics],
#         rank: int = 64,
#         class_priors = None,
#         qda_reg_alpha1: float = 1.0,  # Class-specific weight
#         qda_reg_alpha2: float = 1.0,  # Global weight
#         qda_reg_alpha3: float = 1.0,  # Identity weight (Spherical)
#         device: str = "cuda" if torch.cuda.is_available() else "cpu",
#     ):
#         super().__init__()
#         init_device = torch.device(device)
#         self.register_buffer("_device_indicator", torch.empty(0, device=init_device), persistent=False)
      
#         self.class_ids = sorted(stats_dict.keys())
#         self.num_classes = len(self.class_ids)
#         self.rank = rank
#         self.epsilon = 1e-4  # 数值稳定性
      
#         # === 1. 准备数据 ===
#         logging.info(f"[Init] Starting LowRankGaussianDA init on {device}. Rank={rank}")
      
#         # 收集均值和协方差到 Tensor (临时变量，用完即删)
#         means_list = []
#         covs_list = []
#         for cid in self.class_ids:
#             s = stats_dict[cid]
#             means_list.append(s.mean.float().to(init_device))
#             covs_list.append(s.cov.float().to(init_device))
          
#         means = torch.stack(means_list)      # [C, D]
#         covs = torch.stack(covs_list)        # [C, D, D]
#         C, D, _ = covs.shape
      
#         # === 2. 计算全局基矩阵 A 及其逆 A_inv ===
#         # A = alpha2 * Global + (alpha3 + epsilon) * I
#         global_cov = covs.mean(dim=0)
#         global_cov = 0.5 * (global_cov + global_cov.T) # 确保对称
      
#         # 构建基矩阵 A
#         A = qda_reg_alpha2 * global_cov + (qda_reg_alpha3 + self.epsilon) * torch.eye(D, device=init_device)
      
#         # 预计算 A 的逆和 LogDet
#         try:
#             L_A = torch.linalg.cholesky(A)
#             self.A_inv = torch.cholesky_inverse(L_A)
#             self.base_logdet = 2 * torch.sum(torch.log(torch.diag(L_A))) # log|A|
#         except Exception:
#             # 回退到普通求逆
#             self.A_inv = torch.linalg.inv(A)
#             self.base_logdet = torch.logdet(A)

#         # === 3. 批量低秩分解（向量化SVD） ===
#         logging.info(f"[Init] Performing batched low-rank SVD on {C} classes...")
        
#         # 3.1 对称化协方差矩阵
#         covs_sym = 0.5 * (covs + covs.transpose(-1, -2))
        
#         # 3.2 批量低秩SVD近似 (PyTorch 1.12+支持批量输入)
#         # 注意：svd_lowrank返回U, S, V，但对称矩阵只需U
#         U_batch, S_batch, _ = torch.svd_lowrank(
#             covs_sym, 
#             q=self.rank, 
#             niter=2  # 迭代次数，2通常足够
#         )
        
#         # 3.3 确保奇异值非负
#         S_batch = torch.clamp(S_batch, min=1e-7)
        
#         # 3.4 构建U_eff = sqrt(alpha1 * S) * U
#         scale = torch.sqrt(qda_reg_alpha1 * S_batch)  # [C, rank]
#         U_eff = U_batch * scale.unsqueeze(1)  # [C, D, rank] * [C, 1, rank] -> [C, D, rank]
        
#         # 3.5 计算 Woodbury 修正项
#         #   Ai_U = A_inv @ U_eff
#         Ai_U = torch.einsum('ij,cjk->cik', self.A_inv, U_eff)  # [D,D] @ [C,D,rank] -> [C,D,rank]
        
#         #   inner = U_eff^T @ Ai_U
#         inner = torch.einsum('cji,cjk->cik', U_eff, Ai_U)  # [C, rank, D] @ [C, D, rank] -> [C, rank, rank]
        
#         #   M = I + inner
#         M = torch.eye(self.rank, device=init_device).unsqueeze(0) + inner  # [C, rank, rank]
        
#         # 3.6 批量计算 M 的逆和 logdet
#         try:
#             # 尝试批量Cholesky分解（更稳定）
#             L_M = torch.linalg.cholesky(M)  # [C, rank, rank]
#             M_inv = torch.cholesky_inverse(L_M)  # [C, rank, rank]
            
#             # 计算 log|M| = 2 * sum(log(diag(L_M)))
#             diag_L = torch.diagonal(L_M, dim1=-2, dim2=-1)  # [C, rank]
#             logdet_correction = 2 * torch.sum(torch.log(diag_L + 1e-10), dim=-1)  # [C]
            
#         except Exception as e:
#             logging.warning(f"[Init] Batched Cholesky failed ({str(e)}), falling back to per-class inversion")
#             # 回退到逐类处理
#             M_inv_list = []
#             logdet_correction_list = []
            
#             for c in range(C):
#                 Mc = M[c]
#                 try:
#                     Lc = torch.linalg.cholesky(Mc)
#                     M_inv_c = torch.cholesky_inverse(Lc)
#                     logdet_c = 2 * torch.sum(torch.log(torch.diag(Lc) + 1e-10))
#                 except Exception:
#                     M_inv_c = torch.linalg.inv(Mc)
#                     logdet_c = torch.logdet(Mc + 1e-10 * torch.eye(self.rank, device=init_device))
                
#                 M_inv_list.append(M_inv_c)
#                 logdet_correction_list.append(logdet_c)
            
#             M_inv = torch.stack(M_inv_list)  # [C, rank, rank]
#             logdet_correction = torch.stack(logdet_correction_list)  # [C]
        
#         # === 4. 注册参数 (Buffer) ===
#         self.register_buffer("means", means)                             # [C, D]
#         self.register_buffer("global_A_inv", self.A_inv)                 # [D, D]
#         self.register_buffer("U_effs", U_eff)                            # [C, D, rank]
#         self.register_buffer("M_invs", M_inv)                            # [C, rank, rank]
#         self.register_buffer("logdet_correction", logdet_correction)    # [C]

#         # 先验概率
#         if class_priors is None:
#             priors_list = [1.0 / self.num_classes for _ in self.class_ids]
#         else:
#             priors_list = [class_priors[cid] for cid in self.class_ids]
#         self.log_priors = nn.Parameter(torch.log(torch.tensor(priors_list, device=init_device)), requires_grad=False)

#         # 清理显存
#         del covs, means_list, covs_list, U_batch, S_batch, U_eff, Ai_U, inner, M
#         torch.cuda.empty_cache()
        
#         logging.info(f"[Init] LowRankGaussianDA initialization completed. "
#                     f"Classes={C}, Dim={D}, Rank={rank}, "
#                     f"Memory saved vs full QDA: ~{(C * D * D * 4) / 1e9:.2f} GB")

#     @property
#     def device(self) -> torch.device:
#         return self._device_indicator.device

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         计算 Log-Likelihood (Logits)
#         利用 Woodbury 公式分解计算：
#         Mahalanobis = (x-u)^T [A^{-1} - A^{-1} U M^{-1} U^T A^{-1}] (x-u)
#         """
#         x = x.to(self.device)
#         B, D = x.shape
      
#         # 1. 计算中心化向量 y = x - mu
#         y = x.unsqueeze(1) - self.means.unsqueeze(0)  # [B, C, D]
      
#         # 2. 计算基投影 z = y @ A^{-1}
#         z = F.linear(y, self.global_A_inv)  # [B, C, D]
      
#         # 3. 第一项 (Base Term): y^T A^{-1} y = y . z
#         term1 = (y * z).sum(dim=-1)  # [B, C]
      
#         # 4. 计算 Woodbury 修正项
#         w = torch.einsum('bcd,cdr->bcr', z, self.U_effs)  # [B, C, rank]
#         Mw = torch.einsum('bcr,crk->bck', w, self.M_invs)  # [B, C, rank]
#         term2 = (w * Mw).sum(dim=-1)  # [B, C]
      
#         # 5. 组合马氏距离
#         maha_dist = term1 - term2  # [B, C]
      
#         # 6. 组合 LogDet
#         total_logdet = self.base_logdet + self.logdet_correction  # [C]
      
#         # 7. 最终 Logits
#         logits = -0.5 * maha_dist - 0.5 * total_logdet.unsqueeze(0) + self.log_priors.unsqueeze(0)  # [B, C]
      
#         return logits

#     def predict(self, x: torch.Tensor):
#         return torch.argmax(self.forward(x), dim=1)

#     def predict_proba(self, x: torch.Tensor):
#         return F.softmax(self.forward(x), dim=1)

class LowRankGaussianDA(nn.Module):
    def __init__(
        self,
        stats_dict: Dict[int, GaussianStatistics],
        rank: int = 64,
        class_priors = None,
        qda_reg_alpha1: float = 1.0,
        qda_reg_alpha2: float = 1.0,
        qda_reg_alpha3: float = 1.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 12 # 添加批次大小参数
    ):
        super().__init__()
        init_device = torch.device(device)
        self.register_buffer("_device_indicator", torch.empty(0, device=init_device), persistent=False)
        
        self.class_ids = sorted(stats_dict.keys())
        self.num_classes = len(self.class_ids)
        self.rank = rank
        self.batch_size = batch_size
        
        logging.info(f"[Init] Starting batched LowRankGaussianDA init on {device}. Rank={rank}, BatchSize={batch_size}")
        
        # === 1. 批次计算全局协方差 ===
        C = len(self.class_ids)
        D = stats_dict[self.class_ids[0]].mean.shape[0]
        
        logging.info(f"[Init] Data shape: {C} classes, {D} features")
        
        global_cov = torch.zeros((D, D), device=init_device)
        means_list = []
        
        # 分批处理协方差矩阵
        for i in range(0, C, batch_size):
            batch_cids = self.class_ids[i:i + batch_size]
            batch_covs = []
            
            for cid in batch_cids:
                s = stats_dict[cid]
                means_list.append(s.mean.float().to(init_device))
                batch_covs.append(s.cov.float().to(init_device))
            
            # 批次累加协方差
            batch_covs_tensor = torch.stack(batch_covs)  # [batch_size, D, D]
            global_cov += batch_covs_tensor.sum(dim=0)
            
            del batch_covs, batch_covs_tensor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logging.info(f"[Init] Processed batch {i//batch_size + 1}/{(C-1)//batch_size + 1}")
        
        global_cov = global_cov / C
        global_cov = 0.5 * (global_cov + global_cov.T)
        means = torch.stack(means_list)  # [C, D]
        
        # === 2. 计算全局基矩阵 A 及其逆 ===
        A = qda_reg_alpha2 * global_cov + qda_reg_alpha3 * torch.eye(D, device=init_device)
        
        try:
            L_A = torch.linalg.cholesky(A)
            A_inv = torch.cholesky_inverse(L_A)
            base_logdet = 2 * torch.sum(torch.log(torch.diag(L_A)))
        except Exception:
            A_inv = torch.linalg.inv(A)
            base_logdet = torch.logdet(A)
        
        try:
            U_eff = []
            for cid in self.class_ids:
                s = stats_dict[cid]
                U_eff.append(s.L[:, :self.rank])
            U_eff = torch.stack(U_eff).to(init_device)
            # print(U_eff.shape)
            logging.info(f"[Init] Precomputed low-rank matrices are available")
            
        except Exception as e:
            logging.info(f"[Init] Precomputed low-rank matrices not available, performing batched SVD: {e}")
            U_eff_list = []
            
            # 批次计算低秩SVD
            for i in range(0, C, batch_size):
                batch_cids = self.class_ids[i:i + batch_size]
                batch_cov = []

                for cid in batch_cids:
                    s = stats_dict[cid]
                    cov_c = s.cov.float().to(init_device)
                    batch_cov.append(cov_c)
                
                batch_cov = torch.stack(batch_cov)
                U_batch, S_batch, _ = torch.svd_lowrank(
                    batch_cov, q=self.rank, niter=2)
                S_batch = torch.clamp(S_batch, min=1e-7)
                
                scale = torch.sqrt(qda_reg_alpha1 * S_batch)  # [C, rank]
                U_eff = U_batch * scale.unsqueeze(1)  # [C, D, rank] * [C, 1, rank] -> [C, D, rank]
                U_eff_list.append(U_eff)

                logging.info(f"[Init] SVD batch {i//batch_size + 1}/{(C-1)//batch_size + 1}")
            
            U_eff = torch.cat(U_eff_list, dim=0)  # [C, D, rank]

        # === 4. 批次计算 Woodbury 修正项 ===
        M_inv_list = []
        logdet_correction_list = []
        
        for i in range(0, C, batch_size):
            batch_size_actual = min(batch_size, C - i)
            batch_U = U_eff[i:i + batch_size_actual]  # [batch_size, D, rank]
            batch_means = means[i:i + batch_size_actual]  # [batch_size, D]
            
            # 批次计算 Woodbury 修正项
            Ai_U_batch = torch.einsum('ij,bjk->bik', A_inv, batch_U)  # [batch_size, D, rank]
            inner_batch = torch.einsum('bji,bjk->bik', batch_U, Ai_U_batch)  # [batch_size, rank, rank]
            M_batch = torch.eye(self.rank, device=init_device).unsqueeze(0) + inner_batch  # [batch_size, rank, rank]
            
            # 批次计算 M 的逆和 logdet
            try:
                L_M_batch = torch.linalg.cholesky(M_batch)
                M_inv_batch = torch.cholesky_inverse(L_M_batch)
                diag_L = torch.diagonal(L_M_batch, dim1=-2, dim2=-1)
                logdet_batch = 2 * torch.sum(torch.log(diag_L + 1e-10), dim=-1)
            
            except Exception:
                logging.warning(f"[Init] Batched Cholesky failed for batch {i//batch_size + 1}, using per-element inversion")
                M_inv_batch = []
                logdet_batch = []
                for b in range(batch_size_actual):
                    try:
                        L_b = torch.linalg.cholesky(M_batch[b])
                        M_inv_b = torch.cholesky_inverse(L_b)
                        logdet_b = 2 * torch.sum(torch.log(torch.diag(L_b) + 1e-10))
                    except Exception:
                        M_inv_b = torch.linalg.inv(M_batch[b])
                        logdet_b = torch.logdet(M_batch[b] + 1e-10 * torch.eye(self.rank, device=init_device))
                    M_inv_batch.append(M_inv_b)
                    logdet_batch.append(logdet_b)
                M_inv_batch = torch.stack(M_inv_batch)
                logdet_batch = torch.stack(logdet_batch)
            
            M_inv_list.append(M_inv_batch)
            logdet_correction_list.append(logdet_batch)
            
            logging.info(f"[Init] Woodbury batch {i//batch_size + 1}/{(C-1)//batch_size + 1}")
        
        M_inv = torch.cat(M_inv_list, dim=0)  # [C, rank, rank]
        logdet_correction = torch.cat(logdet_correction_list, dim=0)  # [C]
        
        total_logdet = base_logdet + logdet_correction  # [C]
        
        # === 5. 预计算仿射部分参数 ===
        # 5.1 预计算 w_c = B^{-1} μ_c
        w_c = torch.einsum('ij,cj->ci', A_inv, means)  # [C, D]
        
        # 5.2 预计算 b_c = -0.5 μ_c^T B^{-1} μ_c - 0.5 log|Σ_c^reg| + log π_c
        if class_priors is None:
            priors_list = [1.0 / self.num_classes for _ in self.class_ids]
        else:
            priors_list = [class_priors[cid] for cid in self.class_ids]
        log_priors = torch.log(torch.tensor(priors_list, device=init_device))
        
        # 计算 -0.5 μ_c^T B^{-1} μ_c
        mahalanobis_const = -0.5 * torch.einsum('ci,ci->c', means, w_c)  # [C]
        
        # 计算完整的 b_c
        b_c = mahalanobis_const - 0.5 * total_logdet + log_priors  # [C]
        
        # === 6. 预计算投影矩阵 ===
        # 预计算 U_eff^T B^{-1} 用于快速计算 u_c
        # U_eff: [C, D, r], A_inv: [D, D] -> U_eff^T B^{-1}: [C, r, D]
        U_eff_T_B_inv = torch.einsum('cdr,dj->crj', U_eff, A_inv)  # [C, r, D]
        
        # 预计算 U_eff^T B^{-1} μ_c 用于快速计算 u_c 的常数部分
        U_eff_T_B_inv_mu = torch.einsum('crd,cd->cr', U_eff_T_B_inv, means)  # [C, r]
        
        # === 7. 注册预计算参数 ===
        # 仿射部分参数
        self.register_buffer("affine_weights", w_c)                    # [C, D]
        self.register_buffer("affine_biases", b_c)                     # [C]
        
        # 二次修正部分参数
        self.register_buffer("U_eff_T_B_inv", U_eff_T_B_inv)          # [C, r, D]
        self.register_buffer("U_eff_T_B_inv_mu", U_eff_T_B_inv_mu)    # [C, r]
        self.register_buffer("M_invs", M_inv)                         # [C, rank, rank]
        
        # 存储维度信息用于调试
        self.register_buffer("_feature_dim", torch.tensor(D))
        self.register_buffer("_rank", torch.tensor(rank))
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logging.info(f"[Init] OptimizedLowRankGaussianDA initialization completed. ")
        logging.info(f"[Init] Parameter shapes:")
        logging.info(f"[Init]   affine_weights: {tuple(w_c.shape)}")
        logging.info(f"[Init]   affine_biases: {tuple(b_c.shape)}")
        logging.info(f"[Init]   U_eff_T_B_inv: {tuple(U_eff_T_B_inv.shape)}")
        logging.info(f"[Init]   U_eff_T_B_inv_mu: {tuple(U_eff_T_B_inv_mu.shape)}")
        logging.info(f"[Init]   M_invs: {tuple(M_inv.shape)}")

    @property
    def device(self) -> torch.device:
        return self._device_indicator.device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        使用预计算参数的高效推理
        实现: g_c(x) = L_c(x) + Q_c(x)
        L_c(x) = w_c^T x + b_c
        Q_c(x) = 0.5 * u_c^T M_c^{-1} u_c
        u_c = U_eff^T B^{-1} (x - μ_c)
        """
        x = x.to(self.device)
        B, D = x.shape
        
        # 检查输入维度是否匹配
        expected_D = self._feature_dim.item()
        if D != expected_D:
            raise ValueError(f"Input feature dimension {D} does not match expected dimension {expected_D}. "
                           f"Please check your input data.")
        
        # === 1. 计算仿射部分 L_c(x) = w_c^T x + b_c ===
        # 使用预计算的权重和偏置
        # affine_weights: [C, D] -> 需要转置为 [D, C] 用于 F.linear
        affine_logits = F.linear(x, self.affine_weights, self.affine_biases)  # [B, C]
        
        # === 2. 计算二次修正部分 Q_c(x) = 0.5 * u_c^T M_c^{-1} u_c ===
        # 2.1 使用预计算的投影矩阵计算 u_c
        # u_c = U_eff^T B^{-1} x - U_eff^T B^{-1} μ_c
        # U_eff_T_B_inv: [C, r, D], x: [B, D] -> U_eff_T_B_inv_x: [B, C, r]
        U_eff_T_B_inv_x = torch.einsum('crd,bd->bcr', self.U_eff_T_B_inv, x)
        u_c = U_eff_T_B_inv_x - self.U_eff_T_B_inv_mu.unsqueeze(0)  # [B, C, r]
        
        # 2.2 计算二次形式
        # M_inv_u = M_c^{-1} u_c
        M_inv_u = torch.einsum('bcr,crk->bck', u_c, self.M_invs)  # [B, C, r]
        
        # u_c^T M_c^{-1} u_c = sum(u_c * M_inv_u, dim=-1)
        quadratic_terms = 0.5 * (u_c * M_inv_u).sum(dim=-1)  # [B, C]
        
        # === 3. 组合结果 ===
        logits = affine_logits + quadratic_terms  # [B, C]
        
        return logits

    def predict(self, x: torch.Tensor):
        return torch.argmax(self.forward(x), dim=1)

    def predict_proba(self, x: torch.Tensor):
        return F.softmax(self.forward(x), dim=1)
    
    def get_affine_classifier(self):
        """
        返回仅使用仿射部分的线性分类器
        用于分析或单独使用线性部分
        """
        affine_classifier = nn.Linear(self.affine_weights.size(1), self.affine_weights.size(0), bias=True)
        affine_classifier.weight.data = self.affine_weights.clone()
        affine_classifier.bias.data = self.affine_biases.clone()
        affine_classifier.requires_grad_(False)
        return affine_classifier

    def analyze_components(self, x: torch.Tensor):
        """
        分析仿射部分和二次修正部分的贡献
        返回各部分的详细分解
        """
        x = x.to(self.device)
        B, D = x.shape
        
        # 检查输入维度
        expected_D = self._feature_dim.item()
        if D != expected_D:
            raise ValueError(f"Input feature dimension {D} does not match expected dimension {expected_D}")
        
        # 计算仿射部分
        affine_logits = F.linear(x, self.affine_weights, self.affine_biases)
        
        # 计算二次修正部分
        U_eff_T_B_inv_x = torch.einsum('crd,bd->bcr', self.U_eff_T_B_inv, x)
        u_c = U_eff_T_B_inv_x - self.U_eff_T_B_inv_mu.unsqueeze(0)
        M_inv_u = torch.einsum('bcr,crk->bck', u_c, self.M_invs)
        quadratic_terms = 0.5 * (u_c * M_inv_u).sum(dim=-1)
        
        total_logits = affine_logits + quadratic_terms
        
        return {
            'affine_logits': affine_logits,      # 仿射部分输出
            'quadratic_terms': quadratic_terms,  # 二次修正部分
            'total_logits': total_logits,        # 总输出
            'affine_contribution': torch.softmax(affine_logits, dim=1),  # 仿射部分概率
            'total_probability': torch.softmax(total_logits, dim=1)      # 总概率
        }
    
    def get_parameter_info(self):
        """返回参数信息用于调试"""
        return {
            'feature_dim': self._feature_dim.item(),
            'rank': self._rank.item(),
            'num_classes': self.num_classes,
            'affine_weights_shape': tuple(self.affine_weights.shape),
            'affine_biases_shape': tuple(self.affine_biases.shape),
            'U_eff_T_B_inv_shape': tuple(self.U_eff_T_B_inv.shape),
            'U_eff_T_B_inv_mu_shape': tuple(self.U_eff_T_B_inv_mu.shape),
            'M_invs_shape': tuple(self.M_invs.shape),
        }

# # -*- coding: utf-8 -*-
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import logging
# from typing import Dict, Optional, List
# from compensator.gaussian_statistics import GaussianStatistics

# # 配置 Logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# def get_gpu_memory_info() -> Dict[str, float]:
#     """获取当前GPU显存信息"""
#     if not torch.cuda.is_available():
#         return {"allocated": 0.0, "reserved": 0.0, "max_allocated": 0.0}
    
#     return {
#         "allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
#         "reserved": torch.cuda.memory_reserved() / 1024**3,    # GB
#         "max_allocated": torch.cuda.max_memory_allocated() / 1024**3  # GB
#     }

# class LowRankGaussianDA(nn.Module):
#     def __init__(
#         self,
#         stats_dict: Dict[int, GaussianStatistics],
#         rank: int = 64,
#         class_priors: Optional[Dict[int, float]] = None,
#         qda_reg_alpha1: float = 1.0,  # Class-specific scale
#         qda_reg_alpha2: float = 1.0,  # Global covariance scale
#         qda_reg_alpha3: float = 1.0,  # Identity scale
#         device: str = "cuda" if torch.cuda.is_available() else "cpu",
#         batch_size: int = 12  # 初始化时的批处理大小
#     ):
#         super().__init__()
#         init_device = torch.device(device)
#         self.register_buffer("_device_indicator", torch.empty(0, device=init_device), persistent=False)
        
#         self.class_ids = sorted(stats_dict.keys())
#         self.num_classes = len(self.class_ids)
#         self.rank = rank
        
#         logging.info(f"[Init] Starting FastLowRankGaussianDA on {device}. Rank={rank}, Architecture=DualLinear")

#         # =========================================================
#         # Phase 1: 批量计算全局协方差 (Memory Efficient)
#         # =========================================================
#         # 从第一个样本推断维度
#         first_stats = stats_dict[self.class_ids[0]]
#         D = first_stats.mean.shape[0]
        
#         global_cov = torch.zeros((D, D), device=init_device)
#         means_list = []
        
#         # 分批累加
#         for i in range(0, self.num_classes, batch_size):
#             batch_cids = self.class_ids[i : i + batch_size]
#             batch_covs = []
#             for cid in batch_cids:
#                 s = stats_dict[cid]
#                 means_list.append(s.mean.float().to(init_device))
#                 batch_covs.append(s.cov.float().to(init_device))
            
#             # Stack后求和，比逐个加要快
#             batch_tensor = torch.stack(batch_covs)
#             global_cov += batch_tensor.sum(dim=0)
            
#             del batch_covs, batch_tensor
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()

#         global_cov = global_cov / self.num_classes
#         global_cov = 0.5 * (global_cov + global_cov.T) # 强制对称
#         means = torch.stack(means_list) # [C, D]

#         # =========================================================
#         # Phase 2: 计算全局基矩阵 A 的逆
#         # A = alpha2 * Global + alpha3 * I
#         # =========================================================
#         A = qda_reg_alpha2 * global_cov + qda_reg_alpha3 * torch.eye(D, device=init_device)
        
#         try:
#             L_A = torch.linalg.cholesky(A)
#             A_inv = torch.cholesky_inverse(L_A)
#             base_logdet = 2 * torch.sum(torch.log(torch.diag(L_A)))
#         except Exception:
#             logging.warning("[Init] Cholesky failed for global matrix A, falling back to SVD/Inv.")
#             A_inv = torch.linalg.inv(A)
#             base_logdet = torch.logdet(A)

#         # =========================================================
#         # Phase 3: 获取低秩基 U_eff (Batched SVD)
#         # =========================================================
#         U_eff_list = []
        
#         # 尝试直接使用 stats_dict 中的 L (如果存在且符合 rank)
#         try:
#             # 简单检查第一个是否可用
#             if hasattr(first_stats, 'L') and first_stats.L is not None:
#                 logging.info("[Init] Using precomputed Low-Rank factors from stats.")
#                 for cid in self.class_ids:
#                     U_eff_list.append(stats_dict[cid].L[:, :rank].to(init_device))
#                 U_eff = torch.stack(U_eff_list)
#             else:
#                 raise AttributeError("No precomputed L found.")
#         except Exception:
#             logging.info("[Init] Precomputed factors unavailable. Performing batched SVD...")
#             U_eff_list = []
#             for i in range(0, self.num_classes, batch_size):
#                 batch_cids = self.class_ids[i : i + batch_size]
#                 batch_covs = []
#                 for cid in batch_cids:
#                     batch_covs.append(stats_dict[cid].cov.float().to(init_device))
                
#                 batch_tensor = torch.stack(batch_covs) # [B, D, D]
#                 # torch.svd_lowrank 适合大矩阵提取少量特征值
#                 U_batch, S_batch, _ = torch.svd_lowrank(batch_tensor, q=rank, niter=2)
                
#                 S_batch = torch.clamp(S_batch, min=1e-7)
#                 scale = torch.sqrt(qda_reg_alpha1 * S_batch) # [B, rank]
#                 U_batch_eff = U_batch * scale.unsqueeze(1)   # [B, D, rank]
#                 U_eff_list.append(U_batch_eff)
                
#             U_eff = torch.cat(U_eff_list, dim=0) # [C, D, rank]

#         # =========================================================
#         # Phase 4: 计算 Woodbury 核心矩阵 M 及其逆
#         # M = I + U^T A^{-1} U
#         # =========================================================
#         M_inv_list = []
#         logdet_correction_list = []

#         for i in range(0, self.num_classes, batch_size):
#             # 获取当前 batch 数据
#             end_idx = min(i + batch_size, self.num_classes)
#             batch_U = U_eff[i:end_idx] # [B, D, r]
            
#             # M = I + U^T A^{-1} U
#             # 1. A_inv @ U -> [B, D, r]
#             Ai_U = torch.einsum('ij,bjk->bik', A_inv, batch_U)
#             # 2. U^T @ (A_inv @ U) -> [B, r, r]
#             inner = torch.einsum('bji,bjk->bik', batch_U, Ai_U)
#             # 3. Add Identity
#             M_batch = torch.eye(rank, device=init_device).unsqueeze(0) + inner
            
#             # Invert M
#             try:
#                 L_M = torch.linalg.cholesky(M_batch)
#                 M_inv_batch = torch.cholesky_inverse(L_M)
#                 # LogDet correction
#                 diag_L = torch.diagonal(L_M, dim1=-2, dim2=-1)
#                 logdet_batch = 2 * torch.sum(torch.log(diag_L + 1e-10), dim=-1)
#             except:
#                 # Fallback per item if batch fails
#                 M_inv_temp = []
#                 logdet_temp = []
#                 for k in range(M_batch.shape[0]):
#                     try:
#                         lk = torch.linalg.cholesky(M_batch[k])
#                         M_inv_temp.append(torch.cholesky_inverse(lk))
#                         logdet_temp.append(2 * torch.sum(torch.log(torch.diag(lk) + 1e-10)))
#                     except:
#                         M_inv_temp.append(torch.linalg.inv(M_batch[k]))
#                         logdet_temp.append(torch.logdet(M_batch[k]))
#                 M_inv_batch = torch.stack(M_inv_temp)
#                 logdet_batch = torch.stack(logdet_temp)

#             M_inv_list.append(M_inv_batch)
#             logdet_correction_list.append(logdet_batch)

#         M_invs = torch.cat(M_inv_list, dim=0)          # [C, r, r]
#         logdet_corrections = torch.cat(logdet_correction_list, dim=0) # [C]
#         total_logdets = base_logdet + logdet_corrections

#         # =========================================================
#         # Phase 5: 转换为 Linear 层参数 (核心加速步骤)
#         # =========================================================
        
#         # --- 5.1 LDA 线性部分 ---
#         # w_c = A^{-1} \mu_c
#         w_lda = torch.einsum('ij,cj->ci', A_inv, means) # [C, D]
        
#         # Priors
#         if class_priors is None:
#             priors = torch.tensor([1.0/self.num_classes]*self.num_classes, device=init_device)
#         else:
#             priors = torch.tensor([class_priors[cid] for cid in self.class_ids], device=init_device)
#         log_priors = torch.log(priors)
        
#         # b_c = -0.5 * \mu^T A^{-1} \mu - 0.5 * logdet + log_prior
#         maha_const = -0.5 * torch.einsum('cj,cj->c', means, w_lda)
#         b_lda = maha_const - 0.5 * total_logdets + log_priors

#         # 构建 LDA Layer
#         self.lda_layer = nn.Linear(D, self.num_classes, bias=True)
#         self.lda_layer.weight.data = w_lda # PyTorch Linear weight is [Out, In], so [C, D] matches
#         self.lda_layer.bias.data = b_lda
#         self.lda_layer.requires_grad_(False)

#         # --- 5.2 低秩投影部分 (Low-Rank Projection) ---
#         # 我们需要计算 u_c = U^T A^{-1} (x - \mu)
#         # 令 P = A^{-1} U [D, C, r] -> 转置后为投影矩阵 W_proj [C, r, D]
#         # 注意: (A^{-1} U)^T = U^T A^{-T} = U^T A^{-1} (A是轮换对称的)
        
#         # W_proj: [C, r, D]
#         # U_eff: [C, D, r], A_inv: [D, D]
#         # einsum: cdk, di -> cki (U^T @ A_inv)
#         W_proj_tensor = torch.einsum('cdk,di->cki', U_eff, A_inv) 
        
#         # 将 [C, r, D] 展平为 [C*r, D] 以放入单个 Linear 层
#         # 这样 x 通过 Linear 后输出 [B, C*r]，极大利用 GPU 并行能力
#         self.proj_out_dim = self.num_classes * rank
#         self.proj_layer = nn.Linear(D, self.proj_out_dim, bias=True)
        
#         self.proj_layer.weight.data = W_proj_tensor.reshape(-1, D)
        
#         # 预计算 Bias: b_proj = - W_proj @ \mu
#         # means: [C, D]
#         # W_proj: [C, r, D]
#         # einsum: crd, cd -> cr
#         bias_proj_tensor = -torch.einsum('crd,cd->cr', W_proj_tensor, means)
#         self.proj_layer.bias.data = bias_proj_tensor.reshape(-1)
        
#         self.proj_layer.requires_grad_(False)
        
#         # --- 5.3 注册剩余参数 ---
#         # M_invs 不能被线性层替代，保留为 Buffer
#         self.register_buffer("M_invs", M_invs) # [C, r, r]
        
#         # 清理
#         del U_eff, A_inv, global_cov, means, W_proj_tensor
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
            
#         logging.info("[Init] Initialization Complete.")

#     @property
#     def device(self) -> torch.device:
#         return self._device_indicator.device

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         加速推理:
#         logits = Linear_LDA(x) + 0.5 * Quadratic_Correction(Linear_Proj(x))
#         """
#         # x: [B, D]
#         B = x.shape[0]
        
#         # 1. LDA Linear Path (Extremely Fast) -> [B, C]
#         logits_lda = self.lda_layer(x)
        
#         # 2. Projection Linear Path (Extremely Fast) -> [B, C*r]
#         # 包含了 (x - mu) 的操作在 bias 中
#         u_flat = self.proj_layer(x)
        
#         # Reshape: [B, C*r] -> [B, C, r]
#         u_c = u_flat.view(B, self.num_classes, self.rank)
        
#         # 3. Quadratic Correction
#         # Q = u_c^T @ M^{-1} @ u_c
#         # M_invs: [C, r, r]
#         # u_c:    [B, C, r]
        
#         # Step 3.1: v = M^{-1} @ u_c
#         # 使用 einsum 处理 Batch 矩阵乘法
#         # crk, bck -> bcr (C: classes, r: rank_out, k: rank_in, b: batch)
#         v = torch.einsum('crk,bck->bcr', self.M_invs, u_c)
        
#         # Step 3.2: Dot product
#         # sum(u_c * v) over rank dimension -> [B, C]
#         quadratic_term = 0.5 * (u_c * v).sum(dim=-1)
        
#         # 4. Final Logits (Note: QDA quadratic term is subtracted in log-likelihood, 
#         # but here we follow the derivation:
#         # LogLikelihood = ... - 0.5 * (x-u)^T [A_inv - A_inv U M_inv U^T A_inv] (x-u)
#         #               = ... - 0.5 * (Base_Dist - Correction)
#         #               = ... - 0.5 * Base_Dist + 0.5 * Correction
#         # Our lda_layer computes (-0.5 * Base_Dist + log_pi + const).
#         # So we ADD the correction term.
#         return logits_lda + quadratic_term

#     def predict(self, x: torch.Tensor):
#         return torch.argmax(self.forward(x), dim=1)

#     def predict_proba(self, x: torch.Tensor):
#         return F.softmax(self.forward(x), dim=1)
    
#     def get_parameter_info(self):
#         """调试用"""
#         return {
#             "num_classes": self.num_classes,
#             "rank": self.rank,
#             "lda_weight_shape": self.lda_layer.weight.shape,
#             "proj_weight_shape": self.proj_layer.weight.shape,
#             "M_invs_shape": self.M_invs.shape
#         }