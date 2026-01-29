import torch


def cholesky_stable(matrix: torch.Tensor, reg: float = 1e-5) -> torch.Tensor:
    if matrix.dim() == 3:
        batch_size, n, _ = matrix.shape
        reg_eye = reg * torch.eye(n, device=matrix.device, dtype=matrix.dtype)
        reg_eye = reg_eye.unsqueeze(0).repeat(batch_size, 1, 1)
        return torch.linalg.cholesky(matrix + reg_eye)
    
    elif matrix.dim() == 2:
        reg_eye = reg * torch.eye(matrix.size(0), device=matrix.device, dtype=matrix.dtype)
        return torch.linalg.cholesky(matrix + reg_eye)
    else:
        raise ValueError(f"不支持的矩阵维度: {matrix.dim()}。支持2D或3D张量。")

def cholesky_manual_stable(matrix: torch.Tensor, reg: float = 1e-5) -> torch.Tensor:
    """保留原始手动实现以供参考（已被优化的PyTorch版本替代）"""
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

def cholesky_stable_with_fallback(matrix: torch.Tensor, reg: float = 1e-5) -> torch.Tensor:
    """
    带自动回退的稳定Cholesky分解
    """
    try:
        # 首先尝试快速的原生实现
        return cholesky_stable(matrix, reg)
    
    except RuntimeError as e:
        if "cholesky" in str(e).lower():
            print(f"原生Cholesky失败，回退到手动实现: {e}")
            if matrix.dim() == 2:
                return cholesky_manual_stable(matrix, reg)
            else:
                # 对批量矩阵逐一处理
                batch_size, n, _ = matrix.shape
                results = []
                for i in range(batch_size):
                    try:
                        L = torch.linalg.cholesky(matrix[i] + reg * torch.eye(n, device=matrix.device))
                        results.append(L)
                    except RuntimeError:
                        # 如果仍然失败，使用手动实现
                        L_manual = cholesky_manual_stable(matrix[i], reg)
                        results.append(L_manual)
                return torch.stack(results)
        else:
            raise e

class GaussianStatistics:
    """Container for per-class Gaussian statistics."""

    def __init__(self, mean: torch.Tensor, cov: torch.Tensor, reg: float = 1e-4, cholesky = False):
        if mean.dim() == 2 and mean.size(0) == 1:
            mean = mean.squeeze(0)
        if mean.dim() != 1:
            raise AssertionError("GaussianStatistics.mean 必须是 1D 向量")

        self.mean = mean
        self.cov = cov
        self.reg = reg

        if cholesky:
            self.L = cholesky_stable_with_fallback(cov, reg=reg)
        else:
            self.L = None

    def to(self, device):
        """Move statistics to the requested device."""

        self.mean = self.mean.to(device)
        self.cov = self.cov.to(device)
        if self.L is not None:
            self.L = self.L.to(device)
        return self

    def sample(
        self,
        n_samples = None,
        cached_eps = None,
    ) -> torch.Tensor:
        """Draw samples from the Gaussian distribution."""

        if self.L is None:
            self.L = cholesky_stable_with_fallback(self.cov, reg=self.reg)

        device = self.mean.device
        d = self.mean.size(0)

        if cached_eps is None:
            if n_samples is None:
                raise ValueError("n_samples 必须在未提供 cached_eps 时给定")
            eps = torch.randn(n_samples, d, device=device)
        else:
            eps = cached_eps.to(device)
            n_samples = eps.size(0)

        samples = self.mean.unsqueeze(0) + eps @ self.L.t()
        return samples
