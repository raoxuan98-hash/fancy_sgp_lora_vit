
# Rank参数控制修复总结

## 问题分析

### 原始问题
`sh/run_exp1_performance_surface_parallel.sh` 脚本中存在严重的 rank 参数控制问题：

1. **硬编码 rank 为 1**：
   - 原始脚本中所有实验配置的 rank 参数都被硬编码为 1
   - 但实验名称中却包含 "rank8"，造成命名与实际不符

2. **缺少多 rank 实验支持**：
   - 只测试了 rank=1 的单一配置
   - 无法评估不同 rank 值对模型性能的影响

3. **配置不够灵活**：
   - 需要手动修改多个地方的 rank 值
   - 扩展性差，难以添加新的 rank 值或模型

## 解决方案

### 1. 参数化配置
在脚本开头添加了集中化的配置区域：

```bash
# 要测试的rank值列表
RANKS=(8 16 32)

# 要测试的模型列表和对应的GPU
declare -A MODEL_GPU_MAP
MODEL_GPU_MAP["vit-b-p16"]="0"
