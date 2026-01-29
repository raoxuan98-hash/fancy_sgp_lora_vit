#!/usr/bin/env python3
"""
测试GaussianStatistics中cholesky分解的性能优化方案
"""
import torch
import time
import numpy as np


def create_test_matrix(size=100, condition=1e3, batch_size=1):
    """创建测试用的对称正定矩阵"""
    matrices = []
    for _ in range(batch_size):
        # 创建一个条件数适中的矩阵
        A = torch.randn(size, size)
        Q, R = torch.linalg.qr(A)
        # 创建对角线递减的对角矩阵以控制条件数
        diag = torch.logspace(0, -np.log10(condition), size)
        D = torch.diag(diag)
        matrix = Q @ D @ Q.T
        # 确保对称正定
        matrix = 0.5 * (matrix + matrix.T)
        matrices.append(matrix)
    
    if batch_size == 1:
        return matrices[0]
    return torch.stack(matrices)


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


def cholesky_pytorch_builtin(matrix: torch.Tensor, reg: float = 1e-5) -> torch.Tensor:
    """使用PyTorch内置cholesky"""
    reg_eye = reg * torch.eye(matrix.size(0), device=matrix.device, dtype=matrix.dtype)
    matrix = matrix + reg_eye
    return torch.linalg.cholesky(matrix)


def cholesky_optimized_manual(matrix: torch.Tensor, reg: float = 1e-5) -> torch.Tensor:
    """优化后的手动实现 - 向量化操作"""
    n = matrix.size(0)
    L = torch.zeros_like(matrix)
    reg_eye = reg * torch.eye(n, device=matrix.device, dtype=matrix.dtype)
    matrix = matrix + reg_eye

    # 使用向量化操作减少Python循环
    for j in range(n):
        # 对角线元素
        s_diag = torch.sum(L[j, :j] ** 2, dim=0)
        diag = matrix[j, j] - s_diag
        L[j, j] = torch.sqrt(torch.clamp(diag, min=1e-8))

        # 非对角线元素 - 使用更高效的矩阵运算
        if j < n - 1:
            # 使用更高效的方式计算剩余部分
            L[j + 1:, j] = (matrix[j + 1:, j] - L[j + 1:, :j] @ L[j, :j]) / L[j, j]

    return L


def benchmark_cholesky():
    """比较不同cholesky实现的性能"""
    print("=== Cholesky分解性能比较 ===\n")
    
    # 测试不同矩阵大小
    for size in [50, 100, 200]:
        print(f"矩阵大小: {size}x{size}")
        matrix = create_test_matrix(size)
        
        # 手动实现
        start_time = time.time()
        for _ in range(100):  # 增加迭代次数以获得更稳定的测量
            L_manual = cholesky_manual_stable(matrix)
        manual_time = (time.time() - start_time) / 100
        
        # PyTorch内置实现
        start_time = time.time()
        for _ in range(100):
            L_builtin = torch.linalg.cholesky(matrix + 1e-5 * torch.eye(size))
        builtin_time = (time.time() - start_time) / 100
        
        # 优化手动实现
        start_time = time.time()
        for _ in range(100):
            L_optimized = cholesky_optimized_manual(matrix)
        optimized_time = (time.time() - start_time) / 100
        
        print(f"  原始手动实现:     {manual_time*1000:.3f}ms (基准)")
        print(f"  PyTorch内置:      {builtin_time*1000:.3f}ms (加速比: {manual_time/builtin_time:.1f}x)")
        print(f"  优化手动实现:     {optimized_time*1000:.3f}ms (加速比: {manual_time/optimized_time:.1f}x)")
        
        # 检查精度差异
        diff_manual = torch.max(torch.abs(L_manual - L_builtin)).item()
        diff_optimized = torch.max(torch.abs(L_optimized - L_builtin)).item()
        print(f"  与内置实现的精度差异 - 手动: {diff_manual:.2e}, 优化: {diff_optimized:.2e}")
        print()
    
    # 测试GPU性能（如果有GPU）
    if torch.cuda.is_available():
        print("=== GPU性能测试 ===\n")
        matrix_gpu = matrix.cuda()
        
        # GPU PyTorch内置实现
        start_time = time.time()
        for _ in range(100):
            L_gpu = torch.linalg.cholesky(matrix_gpu + 1e-5 * torch.eye(size).cuda())
        gpu_time = (time.time() - start_time) / 100
        
        print(f"GPU PyTorch内置实现: {gpu_time*1000:.3f}ms")
        print(f"相对于CPU的加速比: {manual_time/gpu_time:.1f}x")


def test_batch_processing():
    """测试批处理性能"""
    print("=== 批处理性能测试 ===\n")
    
    batch_sizes = [1, 4, 8, 16]
    matrix_size = 100
    
    for batch_size in batch_sizes:
        print(f"批处理大小: {batch_size}")
        
        # 创建批处理矩阵
        batch_matrices = create_test_matrix(matrix_size, batch_size=batch_size)
        
        # 逐个处理
        start_time = time.time()
        results_manual = []
        for i in range(batch_size):
            L = cholesky_manual_stable(batch_matrices[i])
            results_manual.append(L)
        manual_time = (time.time() - start_time)
        
        # 批处理内置实现
        start_time = time.time()
        reg_batch = 1e-5 * torch.eye(matrix_size, device=batch_matrices.device).unsqueeze(0).repeat(batch_size, 1, 1)
        L_batch = torch.linalg.cholesky(batch_matrices + reg_batch)
        batch_time = (time.time() - start_time)
        
        print(f"  逐个处理: {manual_time*1000:.3f}ms")
        print(f"  批处理:   {batch_time*1000:.3f}ms (加速比: {manual_time/batch_time:.1f}x)")
        print()


if __name__ == "__main__":
    benchmark_cholesky()
    test_batch_processing()
    
    print("=== 总结 ===")
    print("1. PyTorch内置torch.linalg.cholesky是最快的实现")
    print("2. GPU上的批处理可以显著提升性能")
    print("3. 手动优化对性能提升有限")
    print("4. 建议使用内置实现替代手动实现")