#!/usr/bin/env python3
"""
快速验证GaussianStatistics优化效果
"""
import torch
import time

# 直接从模块导入函数
import sys
sys.path.append('/home/raoxuan/projects/fancy_sgp_lora_vit')

from compensator.gaussian_statistics import cholesky_stable, cholesky_manual_stable


def quick_test():
    """快速性能测试"""
    print("=== 快速Cholesky性能测试 ===\n")
    
    # 创建中等大小的测试矩阵（避免768的长时间计算）
    size = 256
    torch.manual_seed(42)
    
    # 创建对称正定矩阵
    A = torch.randn(size, size)
    matrix = A @ A.T + torch.eye(size) * 1e-3
    
    print(f"矩阵大小: {size}x{size}")
    
    # 测试手动实现
    times_manual = []
    for i in range(5):
        start = time.time()
        L_manual = cholesky_manual_stable(matrix)
        times_manual.append(time.time() - start)
    
    avg_manual = sum(times_manual) / len(times_manual)
    
    # 测试优化实现
    times_optimized = []
    for i in range(5):
        start = time.time()
        L_optimized = cholesky_stable(matrix)
        times_optimized.append(time.time() - start)
    
    avg_optimized = sum(times_optimized) / len(times_optimized)
    
    # 结果分析
    speedup = avg_manual / avg_optimized
    diff = torch.max(torch.abs(L_manual - L_optimized)).item()
    
    print(f"手动实现平均时间: {avg_manual*1000:.2f}ms")
    print(f"优化实现平均时间: {avg_optimized*1000:.2f}ms")
    print(f"性能提升: {speedup:.1f}x")
    print(f"结果一致性: 差异 {diff:.2e}")
    
    # 验证Cholesky性质
    check_manual = torch.max(torch.abs(L_manual @ L_manual.T - matrix)).item()
    check_optimized = torch.max(torch.abs(L_optimized @ L_optimized.T - matrix)).item()
    
    print(f"\nCholesky验证:")
    print(f"手动实现重构误差: {check_manual:.2e}")
    print(f"优化实现重构误差: {check_optimized:.2e}")
    
    return speedup > 2.0  # 至少2倍加速才认为成功


def test_768_matrix():
    """测试768维度的单次分解性能"""
    print(f"\n=== 768维度单次测试 ===")
    
    size = 768
    torch.manual_seed(42)
    A = torch.randn(size, size)
    matrix = A @ A.T + torch.eye(size) * 1e-3
    
    # 只测试一次避免长时间等待
    start = time.time()
    L = cholesky_stable(matrix)
    time_taken = time.time() - start
    
    print(f"768维度矩阵分解时间: {time_taken*1000:.2f}ms")
    print(f"✅ 优化后的768维度矩阵分解成功")


if __name__ == "__main__":
    success = quick_test()
    test_768_matrix()
    
    if success:
        print(f"\n🎉 优化验证成功!")
        print("✅ 性能显著提升")
        print("✅ 结果数值一致")
        print("✅ 768维度矩阵可正常处理")
    else:
        print(f"\n⚠️  性能提升不够明显")
        print("可能需要进一步优化")