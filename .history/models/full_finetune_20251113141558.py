import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Iterable, Optional, Tuple
from timm.models.vision_transformer import VisionTransformer as timm_ViT


class FullFinetuneViT(nn.Module):
    """
    全参数微调ViT模型
    与现有LoRA变体保持接口一致性，但允许所有模型参数参与训练
    """
    def __init__(
        self,
        vit_model: timm_ViT,
        include_norm: bool = True,
        freeze_patch_embed: bool = False):
        
        super().__init__()
        try:
            self.feature_dim = vit_model.embed_dim
        except:
            self.feature_dim = 768
        
        self.lora_vit = vit_model
        
        # 默认解冻所有参数
        for p in self.lora_vit.parameters():
            p.requires_grad = True
        
        # 可选：冻结patch embedding层（通常不需要微调）
        if freeze_patch_embed:
            for p in self.lora_vit.patch_embed.parameters():
                p.requires_grad = False
        
        # 可选：保持norm层冻结（根据需求调整）
        if not include_norm:
            for name, p in self.lora_vit.named_parameters():
                if "norm" in name:
                    p.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.lora_vit(x)
    
    def get_param_groups(self):
        """
        返回所有可训练参数，与现有LoRA接口保持一致
        """
        return [p for p in self.parameters() if p.requires_grad]
    
    def get_module_names(self):
        """
        返回模块名称列表，与现有LoRA接口保持一致
        """
        module_names = []
        for idx, blk in enumerate(self.lora_vit.blocks):
            module_names.append(f"block_{idx}_attn_qkv")
            module_names.append(f"block_{idx}_attn_proj")
            module_names.append(f"block_{idx}_mlp_fc1")
            module_names.append(f"block_{idx}_mlp_fc2")
        return module_names
    
    def finalize_without_lora(self) -> None:
        """
        与现有LoRA接口保持一致，但在全参数微调中不执行任何操作
        """
        pass
    
    def merge_lora_weights(self):
        """
        与现有LoRA接口保持一致，但在全参数微调中不执行任何操作
        """
        pass
    
    def update_projection_matrices(self, covariances: Dict[str, torch.Tensor]) -> None:
        """
        与现有LoRA接口保持一致，但在全参数微调中不执行任何操作
        """
        pass
    
    @property
    def use_projection(self):
        """
        与现有LoRA接口保持一致，全参数微调不使用投影
        """
        return False
    
    def count_trainable_parameters(self) -> int:
        """统计可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def count_total_parameters(self) -> int:
        """统计总参数数量"""
        return sum(p.numel() for p in self.parameters())
    
    def print_parameter_statistics(self) -> None:
        """打印参数统计信息"""
        trainable_params = self.count_trainable_parameters()
        total_params = self.count_total_parameters()
        
        print(f"=== 全参数微调模型统计 ===")
        print(f"总模型参数: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")
        
        # 计算参数效率
        efficiency = (trainable_params / total_params) * 100
        print(f"参数效率: {efficiency:.2f}%")