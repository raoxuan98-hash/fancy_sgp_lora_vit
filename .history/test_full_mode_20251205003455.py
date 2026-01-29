#!/usr/bin/env python3
"""
测试脚本：验证full模式下NullSpaceViT类的函数是否正确实现
"""

import sys
import torch
import torch.nn as nn
import timm
from types import SimpleNamespace

# 导入相关模块
from lora import NullSpaceViT
from utils.inc_net import get_vit

def test_nullspace_vit():
    """测试NullSpaceViT类是否正确实现了所需的函数"""
    print("开始测试NullSpaceViT类...")
    
    # 创建测试参数
    args = {
        'vit_type': 'vit-b-p16',
        'lora_type': 'full',
        'lora_rank': 4,
        'use_projection': True
    }
    
    try:
        # 测试get_vit函数是否能正确创建NullSpaceViT
        vit = get_vit(args, pretrained=False)
        print(f"✓ 成功创建NullSpaceViT模型: {type(vit)}")
        
        # 测试是否有所需的方法
        assert hasattr(vit, 'get_param_groups'), "❌ 缺少get_param_groups方法"
        print("✓ get_param_groups方法存在")
        
