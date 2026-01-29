import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Iterable, Optional, Tuple
from timm.models.vision_transformer import VisionTransformer as timm_ViT


class FullFinetuneViT(nn.Module):
    """
    全参数微调ViT模型
    只微调注意力模块和FFN模块，与现有LoRA变体保持接口一致性
    """
    def __init__(
        self,
        vit_model: timm_ViT,
        include_norm: bool = False,
        freeze_patch_embed: bool = True,
        finetune_layers: Optional[Iterable[int]] = None):
        
        super().__init__()
        try:
            self.feature_dim = vit_model.embed_dim
        except:
            self.feature_dim = 768
        
        self.lora_vit = vit_model
        # 默认对所有block进行微调
        self.finetune_layers = set(finetune_layers) if finetune_layers is not None else set(range(len(vit_model.blocks)))
        
        # 首先冻结所有参数
        for p in self.lora_vit.parameters():
            p.requires_grad = False
        
        # 可选：冻结patch embedding层（通常不需要微调）
        if freeze_patch_embed:
            for p in self.lora_vit.patch_embed.parameters():
                p.requires_grad = False
        
        # 解冻指定层的注意力和FFN模块
        for idx, blk in enumerate(self.lora_vit.blocks):
            if idx not in self.finetune_layers:
                continue
                
            # 解冻注意力模块
            if hasattr(blk, 'attn') and hasattr(blk.attn, 'parameters'):
                for n, p in blk.attn.parameters():
                    p.requires_grad = True
            
            # 解冻FFN模块
            if hasattr(blk, 'mlp') and hasattr(blk.mlp, 'parameters'):
                for p in blk.mlp.parameters():
                    p.requires_grad = True
        
        # 可选：解冻norm层
        if include_norm:
            for name, p in self.lora_vit.named_parameters():
                if "norm" in name:
                    p.requires_grad = True
        
        # 存储可训练模块信息，用于接口一致性
        self.lora_modules = nn.ModuleDict()
        for idx, blk in enumerate(self.lora_vit.blocks):
            if idx not in self.finetune_layers:
                continue
                
            # 注意力模块
            if hasattr(blk, 'attn') and hasattr(blk.attn, 'qkv'):
                self.lora_modules[f"block_{idx}_attn_qkv"] = blk.attn.qkv
            if hasattr(blk, 'attn') and hasattr(blk.attn, 'proj'):
                self.lora_modules[f"block_{idx}_attn_proj"] = blk.attn.proj
            
            # FFN模块
            if hasattr(blk, 'mlp') and hasattr(blk.mlp, 'fc1'):
                self.lora_modules[f"block_{idx}_mlp_fc1"] = blk.mlp.fc1
            if hasattr(blk, 'mlp') and hasattr(blk.mlp, 'fc2'):
                self.lora_modules[f"block_{idx}_mlp_fc2"] = blk.mlp.fc2
    
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
        return list(self.lora_modules.keys())
    
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