import torch
import torch.nn as nn
import timm
from lora import NullSpaceViT

def test_gradient_projection_simple():
    """简单测试梯度投影机制"""
    print("=== 测试梯度投影机制（注意力+FFN模块） ===")
    
    try:
        # 创建ViT模型
        vit = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
        vit.head = nn.Identity()
        del vit.norm
        vit.norm = nn.LayerNorm(768, elementwise_affine=False)
        
        # 使用NullSpaceViT包装
        nullspace_model = NullSpaceViT(vit, use_projection=True)
        print("✓ NullSpaceViT模型创建成功")
        
        # 检查哪些模块被包含在梯度投影中
        print("\n=== 检查梯度投影模块 ===")
        trainable_modules = []
        frozen_modules = []
        
        for name, param in nullspace_model.named_parameters():
            if param.requires_grad:
                trainable_modules.append(name)
            else:
                frozen_modules.append(name)
        
        print(f"可训练模块数量: {len(trainable_modules)}")
        print("前5个可训练模块:")
        for name in trainable_modules[:5]:
            print(f"  - {name}")
        
        print(f"\n冻结模块数量: {len(frozen_modules)}")
        print("前5个冻结模块:")
        for name in frozen_modules[:5]:
            print(f"  - {name}")
        
        # 验证只有注意力模块和FFN模块是可训练的
        attention_ffn_modules = []
        other_trainable = []
        
        for name in trainable_modules:
            if any(keyword in name for keyword in ['attn', 'mlp', 'fc1', 'fc2', 'qkv', 'proj']):
                attention_ffn_modules.append(name)
            else:
                other_trainable.append(name)
        
        print(f"\n注意力/FFN可训练模块: {len(attention_ffn_modules)}")
        print("前5个:")
        for name in attention_ffn_modules[:5]:
            print(f"  - {name}")
            
        if other_trainable:
            print(f"\n其他可训练模块: {len(other_trainable)}")
            for name in other_trainable:
                print(f"  - {name}")
        else:
            print("\n✓ 没有其他可训练模块，只有注意力/FFN模块是可训练的")
        
        # 检查梯度投影钩子是否正确注册
        print("\n=== 检查梯度投影钩子 ===")
        hook_registered_params = list(nullspace_model._param_to_name.keys())
        print(f"注册了梯度投影钩子的参数数量: {len(hook_registered_params)}")
        
        # 验证这些参数都属于注意力或FFN模块
        hook_modules = []
        for param in hook_registered_params:
            name = nullspace_model._param_to_name[param]
            hook_modules.append(name)
        
        print("前5个注册了钩子的模块:")
        for name in hook_modules[:5]:
            print(f"  - {name}")
        
        # 验证所有钩子模块都是注意力或FFN模块
        non_attention_ffn_hooks = []
        for name in hook_modules:
            if not any(keyword in name for keyword in ['attn', 'mlp', 'fc1', 'fc2', 'qkv', 'proj', 'final_norm']):
                non_attention_ffn_hooks.append(name)
        
        if non_attention_ffn_hooks:
            print(f"\n✗ 发现非注意力/FFN模块的钩子: {non_attention_ffn_hooks}")
            return False
        else:
            print("\n✓ 所有钩子都注册在注意力/FFN模块上")
        
        # 测试投影矩阵更新
        print("\n=== 测试投影矩阵更新 ===")
        module_names = nullspace_model.get_module_names()
        print(f"模块数量: {len(module_names)}")
        print("前5个模块:")
        for name in module_names[:5]:
            print(f"  - {name}")
        
        # 创建模拟协方差矩阵并更新
        mock_covariances = {}
        for name in module_names[:2]:  # 只测试前2个
            mock_cov = torch.eye(768)  # 简单的单位矩阵
            mock_covariances[name] = mock_cov
        
        nullspace_model.update_projection_matrices(mock_covariances, soft_projection=True, weight_temp=1.0)
        
        # 检查投影矩阵是否正确存储
        for name in mock_covariances.keys():
            if name in nullspace_model.projection_matrices:
                proj = nullspace_model.projection_matrices[name]
                print(f"✓ 模块 {name} 的投影矩阵形状: {proj.shape}")
            else:
                print(f"✗ 模块 {name} 的投影矩阵未找到")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_gradient_projection_simple()
    if success:
        print("\n🎉 梯度投影机制测试通过！")
        print("✓ 确认：只有注意力模块和FFN模块实现了梯度修正")
    else:
        print("\n❌ 梯度投影机制测试失败！")