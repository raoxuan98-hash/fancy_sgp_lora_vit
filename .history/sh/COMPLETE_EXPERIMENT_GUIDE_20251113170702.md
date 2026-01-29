# 完整实验配置指南

## 概述

本指南详细说明了所有实验方法的参数配置和运行方式。根据实验方案，我们创建了包含所有参数变体的完整实验脚本，确保每个实验都运行3个随机种子以获得可靠的结果。

## 实验方法及参数变体

### 1. 基础LoRA (basic_lora)

**参数配置**:
```bash
LORA_TYPE="basic_lora"
VIT_TYPE="vit-b-p16-mocov3"
GAMMA_KD=0.0  # 不使用知识蒸馏
```

**运行脚本**: 
- 单独运行: `bash sh/main_experiments_basic_lora.sh`
- 完整实验: `bash sh/run_complete_experiments.sh` (包含在完整实验中)

### 2. LoRA + 知识蒸馏 (lora_kd)

#### 2.1 蒸馏权重 = 1.0

**参数配置**:
```bash
LORA_TYPE="basic_lora"
VIT_TYPE="vit-b-p16-mocov3"
GAMMA_KD=1.0
UPDATE_TEACHER_EACH_TASK=True
DISTILLATION_TRANSFORM="identity"
KD_TYPE="feat"
```

