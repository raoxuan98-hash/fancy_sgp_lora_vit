import torch
import torch.nn as nn
import timm
from utils.inc_net import get_vit
from models.full_finetune import FullFinetuneViT

def test_full_finetune():
    """测试全参数微调实现"""
    print("=== 测试全参数微调实现 ===")
    
    # 创建测试参数
    args = {
        'vit_type': 'vit-b-p16',
        'lora_type': 'full_ft',
        'lora_rank': 4,
        'include_norm': False,
        'freeze_patch_embed': True,
        'finetune_layers': None  # 默认所有层
    }
    
    # 测试模型创建
    try:
        vit = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
        vit.head = nn.Identity()
        del vit.norm
        vit.norm = nn.LayerNorm(768, elementwise_affine=False)
        
        # 使用我们的FullFinetuneViT类
        full_ft_model = FullFinetuneViT(vit, 
                                       include_norm=args['include_norm'],
                                       freeze_patch_embed=args['freeze_patch_embed'],
                                       finetune_layers=args['finetune_layers'])
        
        print("✓ FullFinetuneViT模型创建成功")
        
        # 测试前向传播
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = full_ft_model(x)
        print(f"✓ 前向传播成功，输出形状: {output.shape}")
        
        # 测试参数统计
        trainable_params = full_ft_model.count_trainable_parameters()
        total_params = full_ft_model.count_total_parameters()
        efficiency = (trainable_params / total_params) * 100
        
        print(f"✓ 参数统计:")
        print(f"  总参数: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        print(f"  参数效率: {efficiency:.2f}%")
        
        # 测试接口一致性
        param_groups = full_ft_model.get_param_groups()
        module_names = full_ft_model.get_module_names()
        use_projection = full_ft_model.use_projection
        
        print(f"✓ 接口测试:")
        print(f"  参数组数量: {len(param_groups)}")
        print(f"  模块名称: {module_names[:3]}...")  # 只显示前3个
        print(f"  使用投影: {use_projection}")
        
        # 测试通过utils.inc_net创建
        print("\n=== 测试通过utils/inc_net创建 ===")
        vit_model = get_vit(args, pretrained=False)
        print("✓ 通过get_vit创建成功")
        
        # 测试参数获取
        if hasattr(vit_model, 'get_param_groups'):
            params = vit_model.get_param_groups()
            print(f"✓ 参数获取成功，数量: {len(params)}")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_full_finetune()
    if success:
        print("\n🎉 全参数微调实现测试通过！")
    else:
        print("\n❌ 全参数微调实现测试失败！")