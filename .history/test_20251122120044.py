import torch
import time
import numpy as np

def cholesky_block_parallel(matrix: torch.Tensor, reg: float = 1e-5, block_size: int = 64) -> torch.Tensor:
    """
    分块并行Cholesky分解
    
    Args:
        matrix: 对称正定矩阵
        reg: 正则化参数
        block_size: 分块大小，影响并行度
        
    Returns:
        Cholesky下三角矩阵 L
    """
    n = matrix.size(0)
    L = torch.zeros_like(matrix)
    reg_eye = reg * torch.eye(n, device=matrix.device, dtype=matrix.dtype)
    matrix = matrix + reg_eye
    
    # 按块处理
    for k in range(0, n, block_size):
        k_end = min(k + block_size, n)
        
        # 处理对角块
        for j in range(k, k_end):
            # 对角线元素
            s_diag = torch.sum(L[j, :j] ** 2)
            diag = matrix[j, j] - s_diag
            L[j, j] = torch.sqrt(torch.clamp(diag, min=1e-8))
            
            # 同一块内的非对角元素
            if j < k_end - 1:
                j_range = slice(j + 1, k_end)
                # 修复：使用矩阵乘法而不是切片
                s_off = L[j_range, :j] @ L[j, :j].unsqueeze(1)
                L[j_range, j] = (matrix[j_range, j] - s_off.squeeze()) / L[j, j]
        
        # 并行处理剩余列
        if k_end < n:
            remaining_range = slice(k_end, n)
            
            # 批量计算s_off - 这里可以并行化
            for j in range(k, k_end):
                s_off = L[remaining_range, :j] @ L[j, :j].unsqueeze(1)
                L[remaining_range, j] = (matrix[remaining_range, j] - s_off.squeeze()) / L[j, j]
    
    return L

def cholesky_manual_stable(matrix: torch.Tensor, reg: float = 1e-5) -> torch.Tensor:
    """原始手动实现作为基准"""
    n = matrix.size(0)
    L = torch.zeros_like(matrix)
    reg_eye = reg * torch.eye(n, device=matrix.device, dtype=matrix.dtype)
    matrix = matrix + reg_eye

    for j in range(n):
        s_diag = torch.sum(L[j, :j] ** 2)
        diag = matrix[j, j] - s_diag
        L[j, j] = torch.sqrt(torch.clamp(diag, min=1e-8))

        if j < n - 1:
            s_off = L[j + 1:, :j] @ L[j, :j].unsqueeze(1)
            L[j + 1:, j] = (matrix[j + 1:, j] - s_off.squeeze()) / L[j, j]

    return L

def test_cholesky_implementations():
    """测试不同的Cholesky实现"""
    
    # 创建正定矩阵
    torch.manual_seed(42)
    
    sizes = [128, 256, 512, 768]
    block_sizes = [32, 64, 128]
    
    results = {}
    
    for size in sizes:
        print(f"\n测试矩阵大小: {size}x{size}")
        print("=" * 50)
        
        # 生成正定矩阵
        A = torch.randn(size, size, dtype=torch.float64)
        matrix = A @ A.T + 1e-3 * torch.eye(size, dtype=torch.float64)
        
        # 基准测试：PyTorch官方实现
        start = time.time()
        L_pytorch = torch.linalg.cholesky(matrix)
        time_pytorch = time.time() - start
        
        # 基准测试：原始手动实现
        start = time.time()
        L_manual = cholesky_manual_stable(matrix)
        time_manual = time.time() - start
        
        # 测试不同分块大小
        for block_size in block_sizes:
            if block_size > size:
                continue
                
            start = time.time()
            L_block = cholesky_block_parallel(matrix, block_size=block_size)
            time_block = time.time() - start
            
            # 验证精度
            error_pytorch = torch.norm(L_pytorch - L_block).item()
            reconstruction_error = torch.norm(matrix - L_block @ L_block.T).item()
            
            print(f"分块大小 {block_size:3d}: 时间={time_block:.4f}s, "
                  f"与PyTorch误差={error_pytorch:.2e}, 重构误差={reconstruction_error:.2e}")
            
            results[(size, block_size)] = {
                'time': time_block,
                'error_vs_pytorch': error_pytorch,
                'reconstruction_error': reconstruction_error
            }
        
        print(f"PyTorch官方: 时间={time_pytorch:.4f}s")
        print(f"手动实现:   时间={time_manual:.4f}s")
    
    return results

def test_parallelization_effect():
    """测试并行化效果"""
    
    size = 512
    torch.manual_seed(42)
    
    # 生成正定矩阵
    A = torch.randn(size, size)
    matrix = A @ A.T + 1e-3 * torch.eye(size)
    
    # 测试不同分块大小
    block_sizes = [16, 32, 64, 128, 256]
    
    print(f"\n并行化效果测试 (矩阵大小: {size}x{size})")
    print("=" * 60)
    
    for block_size in block_sizes:
        if block_size > size:
            continue
            
        # 预热
        _ = cholesky_block_parallel(matrix, block_size=block_size)
        
        # 计时
        times = []
        for _ in range(5):
            start = time.time()
            L = cholesky_block_parallel(matrix, block_size=block_size)
            torch.cuda.synchronize() if matrix.is_cuda else None
            times.append(time.time() - start)
        
        avg_time = np.mean(times[1:])  # 忽略第一次
        std_time = np.std(times[1:])
        
        # 验证结果
        reconstruction_error = torch.norm(matrix - L @ L.T).item()
        
        print(f"分块大小 {block_size:3d}: 平均时间={avg_time:.4f}s (±{std_time:.4f}), "
              f"重构误差={reconstruction_error:.2e}")

def test_gpu_performance():
    """测试GPU性能（如果有GPU）"""
    if not torch.cuda.is_available():
        print("未检测到GPU，跳过GPU测试")
        return
    
    size = 1024
    torch.manual_seed(42)
    
    # 在GPU上创建矩阵
    A = torch.randn(size, size, device='cuda')
    matrix = A @ A.T + 1e-3 * torch.eye(size, device='cuda')
    
    print(f"\nGPU性能测试 (矩阵大小: {size}x{size})")
    print("=" * 50)
    
    # 预热
    _ = torch.linalg.cholesky(matrix)
    _ = cholesky_block_parallel(matrix, block_size=64)
    
    # 测试PyTorch官方实现
    start = time.time()
    L_pytorch = torch.linalg.cholesky(matrix)
    torch.cuda.synchronize()
    time_pytorch = time.time() - start
    
    # 测试分块实现
    block_sizes = [64, 128, 256]
    for block_size in block_sizes:
        start = time.time()
        L_block = cholesky_block_parallel(matrix, block_size=block_size)
        torch.cuda.synchronize()
        time_block = time.time() - start
        
        error = torch.norm(L_pytorch - L_block).item()
        print(f"分块大小 {block_size}: 时间={time_block:.4f}s, "
              f"PyTorch时间={time_pytorch:.4f}s, 加速比={time_pytorch/time_block:.2f}x, "
              f"误差={error:.2e}")

if __name__ == "__main__":
    print("开始测试分块并行Cholesky分解")
    print("设备:", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    # CPU测试
    results = test_cholesky_implementations()
    
    # 并行化效果测试
    test_parallelization_effect()
    
    # GPU测试（如果可用）
    test_gpu_performance()
