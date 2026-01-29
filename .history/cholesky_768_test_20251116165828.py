#!/usr/bin/env python3
"""
针对768维度大矩阵的cholesky分解性能测试
"""
import torch
import time
import numpy as np


def create_large_matrix(size=768, condition=1e3):
    """创建768维度的测试矩阵"""
    # 创建一个条件数适中的矩阵
    A = torch.randn(size, size)
    Q, R = torch.linalg.qr(A)
    # 创建对角线递减的对角矩阵以控制条件数
    diag = torch.logspace(0, -np.log10(condition), size)
    D = torch.diag(diag)
    matrix = Q @ D @ Q.T
    # 确保对称正定
    matrix = 0.5 * (matrix + matrix.T)
    return matrix


def cholesky_manual_stable(matrix: torch.Tensor, reg: float = 1e-5) -> torch.Tensor:
    """当前的手动实现"""
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


def cholesky_manual_optimized(matrix: torch.Tensor, reg: float = 1e-5) -> torch.Tensor:
    """优化的手动实现 - 针对大矩阵优化内存访问"""
    n = matrix.size(0)
    L = torch.zeros_like(matrix)
    reg_eye = reg * torch.eye(n, device=matrix.device, dtype=matrix.dtype)
    matrix = matrix + reg_eye

    for j in range(n):
        # 使用更高效的向量操作
        L_jj_prev = L[j, :j]
        s_diag = torch.dot(L_jj_prev, L_jj_prev)
        diag = matrix[j, j] - s_diag
        L[j, j] = torch.sqrt(torch.clamp(diag, min=1e-8))

        if j < n - 1:
            # 避免重复计算
            L_jj = L[j, :j]
            s_off = L[j + 1:, :j] @ L_jj
            L[j + 1:, j] = (matrix[j + 1:, j] - s_off) / L[j, j]

    return L


def cholesky_pytorch_builtin(matrix: torch.Tensor, reg: float = 1e-5) -> torch.Tensor:
    """使用PyTorch内置cholesky"""
    reg_eye = reg * torch.eye(matrix.size(0), device=matrix.device, dtype=matrix.dtype)
    return torch.linalg.cholesky(matrix + reg_eye)


def benchmark_large_matrix():
    """针对768维度矩阵的性能测试"""
    print("=== 768维度大矩阵性能测试 ===\n")
    
    matrix = create_large_matrix(768, condition=1e6)  # 更大的条件数以更真实
    device = matrix.device
    
    print(f"矩阵大小: {matrix.shape}")
    print(f"设备: {device}")
    print(f"内存占用: {matrix.element_size() * matrix.numel() / 1024**2:.2f} MB\n")
    
    # 测试手动实现
    start_time = time.time()
    for _ in range(5):
        L_manual = cholesky_manual_stable(matrix)
    manual_time = (time.time() - start_time) / 5
    
    # 测试优化的手动实现
    start_time = time.time()
    for _ in range(5):
        L_optimized = cholesky_manual_optimized(matrix)
    optimized_time = (time.time() - start_time) / 5
    
    # 测试PyTorch内置实现
    start_time = time.time()
    for _ in range(5):
        L_builtin = torch.linalg.cholesky(matrix + 1e-5 * torch.eye(768, device=device))
    builtin_time = (time.time() - start_time) / 5
    
    print(f"原始手动实现: {manual_time:.4f}s (基准)")
    print(f"优化手动实现: {optimized_time:.4f}s (加速比: {manual_time/optimized_time:.2f}x)")
    print(f"PyTorch内置:  {builtin_time:.4f}s (相对于手动的加速比: {manual_time/builtin_time:.2f}x)")
    
    # 检查结果一致性
    diff_manual = torch.max(torch.abs(L_manual - L_builtin)).item()
    diff_optimized = torch.max(torch.abs(L_optimized - L_builtin)).item()
    
    print(f"\n精度检查:")
    print(f"原始实现与内置差异: {diff_manual:.2e}")
    print(f"优化实现与内置差异: {diff_optimized:.2e}")
    
    # 检查是否满足Cholesky性质
    check_manual = torch.max(torch.abs(L_manual @ L_manual.t() - (matrix + 1e-5 * torch.eye(768, device=device)))).item()
    check_optimized = torch.max(torch.abs(L_optimized @ L_optimized.t() - (matrix + 1e-5 * torch.eye(768, device=device)))).item()
    check_builtin = torch.max(torch.abs(L_builtin @ L_builtin.t() - (matrix + 1e-5 * torch.eye(768, device=device)))).item()
    
    print(f"\nCholesky性质检查 (重构误差):")
    print(f"原始实现: {check_manual:.2e}")
    print(f"优化实现: {check_optimized:.2e}")
    print(f"内置实现: {check_builtin:.2e}")
    
    return manual_time, optimized_time, builtin_time


def test_different_sizes():
    """测试不同大小的性能对比"""
    print("\n=== 不同大小的性能对比 ===")
    
    sizes = [256, 512, 768, 1024]
    
    for size in sizes:
        print(f"\n矩阵大小: {size}x{size}")
        matrix = create_large_matrix(size, condition=1e4)
        
        # 只测试一次，避免耗时太长
        start_time = time.time()
        L_manual = cholesky_manual_stable(matrix)
        manual_time = time.time() - start_time
        
        start_time = time.time()
        L_pytorch = torch.linalg.cholesky(matrix + 1e-5 * torch.eye(size, device=matrix.device))
        pytorch_time = time.time() - start_time
        
        print(f"  手动实现: {manual_time:.4f}s")
        print(f"  PyTorch内置: {pytorch_time:.4f}s")
        print(f"  加速比: {manual_time/pytorch_time:.2f}x")


if __name__ == "__main__":
    manual_time, optimized_time, builtin_time = benchmark_large_matrix()
    test_different_sizes()
    
    print(f"\n=== 结论 ===")
    if manual_time < builtin_time * 1.1:
        print("对于768维度矩阵，手动实现仍然有优势")
        print("建议在gaussian_statistics.py中使用优化的手动实现")
    else:
        print("对于768维度矩阵，PyTorch内置实现更优")
        print("建议使用torch.linalg.cholesky替代手动实现")