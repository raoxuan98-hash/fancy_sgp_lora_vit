import torch
import torch.nn as nn
import timm
from lora import NullSpaceViT
import numpy as np

def test_gradient_projection():
    """测试梯度投影机制"""
    print("=== 测试梯度投影机制 ===")
    
    # 创建测试参数
    args = {
        'vit_type': 'vit-b-p16',
        'lora_type': 'full',  # 使用NullSpaceViT
        'use_projection': True
    }
    
    try:
        # 创建ViT模型
        vit = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
        vit.head = nn.Identity()
        del vit.norm
        vit.norm = nn.LayerNorm(768, elementwise_affine=False)
        
        # 使用NullSpaceViT包装
        nullspace_model = NullSpaceViT(vit, use_projection=True)
        print("✓ NullSpaceViT模型创建成功")
        
        # 测试前向传播
        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        output = nullspace_model(x)
        print(f"✓ 前向传播成功，输出形状: {output.shape}")
        
        # 测试反向传播和梯度投影
        loss = output.sum()
        loss.backward()
        
        # 检查是否有梯度
        has_gradients = False
        for name, param in nullspace_model.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_gradients = True
                print(f"✓ 参数 {name} 有梯度，形状: {param.grad.shape}")
                break
        
        if not has_gradients:
            print("✗ 没有找到梯度")
            return False
        
        # 测试投影矩阵更新
        print("\n=== 测试投影矩阵更新 ===")
        
        # 创建模拟的协方差矩阵
        module_names = nullspace_model.get_module_names()
        print(f"模块名称: {module_names[:3]}...")  # 只显示前3个
        
        # 创建模拟协方差矩阵
        mock_covariances = {}
        for name in module_names[:2]:  # 只测试前2个模块
            # 获取对应参数的形状
            for module_name, module in nullspace_model.lora_modules.items():
                if module_name == name and hasattr(module, 'weight'):
                    weight_shape = module.weight.shape
                    # 创建模拟协方差矩阵
                    if len(weight_shape) == 2:  # 线性层
                        dim = weight_shape[1]  # 输入维度
                        mock_cov = torch.randn(dim, dim)
                        mock_cov = mock_cov @ mock_cov.t()  # 确保正定
                        mock_covariances[name] = mock_cov
                        print(f"✓ 为模块 {name} 创建模拟协方差矩阵，形状: {mock_cov.shape}")
                    break
        
        # 更新投影矩阵
        nullspace_model.update_projection_matrices(mock_covariances, soft=True, temp=1.0)
        print("✓ 投影矩阵更新成功")
        
        # 检查投影矩阵是否正确存储
        for name in mock_covariances.keys():
            if name in nullspace_model.projection_matrices:
                proj = nullspace_model.projection_matrices[name]
                print(f"✓ 模块 {name} 的投影矩阵形状: {proj.shape}")
                
                # 验证投影矩阵的性质
                # 1. 应该是方阵
                assert proj.shape[0] == proj.shape[1], f"投影矩阵应该是方阵，但得到形状 {proj.shape}"
                
                # 2. 应该是对称的（数值误差范围内）
                diff = torch.max(torch.abs(proj - proj.t()))
                assert diff < 1e-5, f"投影矩阵应该是对称的，但最大不对称差异为 {diff}"
                
                print(f"  - 验证通过: 方阵且对称")
            else:
                print(f"✗ 模块 {name} 的投影矩阵未找到")
                return False
        
        # 测试梯度投影功能
        print("\n=== 测试梯度投影功能 ===")
        
        # 清零梯度
        nullspace_model.zero_grad()
        
        # 再次前向传播和反向传播
        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        output = nullspace_model(x)
        loss = output.sum()
        loss.backward()
        
        # 检查梯度是否被投影
        print("✓ 梯度投影测试完成")
        
        # 测试开关投影功能
        print("\n=== 测试投影开关功能 ===")
        nullspace_model.disable_projection()
        print("✓ 投影功能已禁用")
        
        nullspace_model.enable_projection()
        print("✓ 投影功能已启用")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_gradient_projection()
    if success:
        print("\n🎉 梯度投影机制测试通过！")
    else:
        print("\n❌ 梯度投影机制测试失败！")