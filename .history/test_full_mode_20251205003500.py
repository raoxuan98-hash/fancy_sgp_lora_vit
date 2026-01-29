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
        
        assert hasattr(vit, 'merge_lora_weights'), "❌ 缺少merge_lora_weights方法"
        print("✓ merge_lora_weights方法存在")
        
        assert hasattr(vit, 'finalize_without_lora'), "❌ 缺少finalize_without_lora方法"
        print("✓ finalize_without_lora方法存在")
        
        assert hasattr(vit, 'update_projection_matrices'), "❌ 缺少update_projection_matrices方法"
        print("✓ update_projection_matrices方法存在")
        
        # 测试get_param_groups方法
        param_groups = vit.get_param_groups()
        assert isinstance(param_groups, list), "❌ get_param_groups应返回列表"
        assert len(param_groups) > 0, "❌ get_param_groups返回的列表不应为空"
        print(f"✓ get_param_groups返回了{len(param_groups)}个参数组")
        
        # 测试merge_lora_weights方法
        try:
            vit.merge_lora_weights()
            print("✓ merge_lora_weights方法执行成功")
        except Exception as e:
            print(f"❌ merge_lora_weights方法执行失败: {e}")
            return False
        
        # 测试finalize_without_lora方法
        try:
            vit.finalize_without_lora()
            print("✓ finalize_without_lora方法执行成功")
        except Exception as e:
            print(f"❌ finalize_without_lora方法执行失败: {e}")
            return False
        
        # 测试前向传播
