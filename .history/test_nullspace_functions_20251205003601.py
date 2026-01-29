#!/usr/bin/env python3
"""
简化测试脚本：直接测试NullSpaceViT类添加的函数
"""

import sys
import torch
import torch.nn as nn
from types import SimpleNamespace

# 模拟一个简单的ViT模型结构
class MockBlock:
    def __init__(self):
        self.attn = SimpleNamespace()
        self.attn.qkv = nn.Linear(768, 3*768)
        self.mlp = SimpleNamespace()
        self.mlp.fc1 = nn.Linear(768, 3072)
        self.mlp.fc2 = nn.Linear(3072, 768)

class MockViT:
    def __init__(self):
        self.blocks = nn.ModuleList([MockBlock() for _ in range(12)])
        self.norm = nn.LayerNorm(768)
        self.feature_dim = 768

# 导入NullSpaceViT类
sys.path.append('.')
from lora import NullSpaceViT

def test_nullspace_functions():
    """测试NullSpaceViT类新添加的函数"""
    print("开始测试NullSpaceViT类的新函数...")
    
    try:
        # 创建模拟ViT模型
        mock_vit = MockViT()
        
