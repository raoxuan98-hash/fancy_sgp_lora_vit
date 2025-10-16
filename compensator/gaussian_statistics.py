
import torch
import torch.nn.functional as F

def cholesky_manual_stable(matrix: torch.Tensor, reg: float = 1e-5) -> torch.Tensor:
    """
    手动实现数值稳定的 Cholesky 分解，用于保障协方差矩阵正定性。
    """
    n = matrix.size(0)
    L = torch.zeros_like(matrix)
    reg_eye = reg * torch.eye(n, device=matrix.device, dtype=matrix.dtype)
    matrix = matrix + reg_eye
    for j in range(n):
        s_diag = torch.sum(L[j, :j] ** 2, dim=0)
        diag = matrix[j, j] - s_diag
        L[j, j] = torch.sqrt(torch.clamp(diag, min=1e-8))
        if j < n - 1:
            s_off = L[j + 1:, :j] @ L[j, :j]
            L[j + 1:, j] = (matrix[j + 1:, j] - s_off) / L[j, j]
    return L

class GaussianStatistics:
    """
    表示单个类别的高斯统计信息（均值 + 协方差 + 采样能力）
    """
    def __init__(self, mean: torch.Tensor, cov: torch.Tensor, reg: float = 1e-5):
        if mean.dim() == 2 and mean.size(0) == 1:
            mean = mean.squeeze(0)
        assert mean.dim() == 1, "GaussianStatistics.mean 必须是 1D 向量"
        self.mean = mean
        self.cov = cov
        self.reg = reg
        self.L = cholesky_manual_stable(cov, reg=reg)

    def to(self, device):
        """移动到指定设备"""
        self.mean = self.mean.to(device)
        self.cov = self.cov.to(device)
        self.L = self.L.to(device)
        return self

    def sample(self, n_samples: int = None, cached_eps: torch.Tensor = None, use_weighted_cov: bool = False) -> torch.Tensor:
        """
        从该高斯分布采样。
        """
        device = self.mean.device
        d = self.mean.size(0)
        if cached_eps is None:
            eps = torch.randn(n_samples, d, device=device)
        else:
            eps = cached_eps.to(device)
            n_samples = eps.size(0)
        L = self.L
        samples = self.mean.unsqueeze(0) + eps @ L.t()
        return samples
