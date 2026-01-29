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
        
        # 创建NullSpaceViT实例
        nullspace_vit = NullSpaceViT(mock_vit, use_projection=True)
        print("✓ 成功创建NullSpaceViT实例")
        
        # 测试get_param_groups方法
        param_groups = nullspace_vit.get_param_groups()
        assert isinstance(param_groups, list), "❌ get_param_groups应返回列表"
        assert len(param_groups) > 0, "❌ get_param_groups返回的列表不应为空"
        print(f"✓ get_param_groups返回了{len(param_groups)}个参数组")
        
        # 测试merge_lora_weights方法
        try:
            nullspace_vit.merge_lora_weights()
            print("✓ merge_lora_weights方法执行成功")
        except Exception as e:
            print(f"❌ merge_lora_weights方法执行失败: {e}")
            return False
        
        # 测试finalize_without_lora方法
        try:
            nullspace_vit.finalize_without_lora()
            print("✓ finalize_without_lora方法执行成功")
        except Exception as e:
            print(f"❌ finalize_without_lora方法执行失败: {e}")
            return False
        
        # 测试update_projection_matrices方法
        try:
            # 创建虚拟的协方差矩阵
            covariances = {}
            for name in nullspace_vit.get_module_names():
                # 获取模块的权重形状
                module = nullspace_vit.lora_modules[name]
                if hasattr(module, 'weight'):
                    weight_shape = module.weight.shape
                    # 对于线性层，创建输入维度的协方差矩阵
                    if len(weight_shape) == 2:  # (out_features, in_features)
                        cov_shape = (weight_shape[1], weight_shape[1])
                    else:
                        cov_shape = (weight_shape[0], weight_shape[0])
                    covariances[name] = torch.randn(*cov_shape)
            
            nullspace_vit.update_projection_matrices(covariances)
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
    success = test_nullspace_functions()
    sys.exit(0 if success else 1)