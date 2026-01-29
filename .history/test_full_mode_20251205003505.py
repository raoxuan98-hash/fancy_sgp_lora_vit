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
        batch_size, channels, height, width = 2, 3, 224, 224
        dummy_input = torch.randn(batch_size, channels, height, width)
        
        try:
            with torch.no_grad():
                output = vit(dummy_input)
            assert output.shape[0] == batch_size, "❌ 输出批次大小不匹配"
            print(f"✓ 前向传播成功，输出形状: {output.shape}")
        except Exception as e:
            print(f"❌ 前向传播失败: {e}")
            return False
        
        # 测试update_projection_matrices方法
        try:
            # 创建虚拟的协方差矩阵
            covariances = {}
            for name in vit.get_module_names():
                # 获取模块的权重形状
                module = vit.lora_modules[name]
                if hasattr(module, 'weight'):
                    weight_shape = module.weight.shape
                    # 对于线性层，创建输入维度的协方差矩阵
                    if len(weight_shape) == 2:  # (out_features, in_features)
                        cov_shape = (weight_shape[1], weight_shape[1])
                    else:
                        cov_shape = (weight_shape[0], weight_shape[0])
                    covariances[name] = torch.randn(*cov_shape)
            
            vit.update_projection_matrices(covariances)
            print("✓ update_projection_matrices方法执行成功")
        except Exception as e:
            print(f"❌ update_projection_matrices方法执行失败: {e}")
            return False
        
        print("\n🎉 所有测试通过！NullSpaceViT类已正确实现所需的函数。")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_nullspace_vit()
    sys.exit(0 if success else 1)
