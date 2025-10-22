import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Iterable, Optional, Tuple
from timm.models.vision_transformer import VisionTransformer as timm_ViT

# ==================== 原始 LoRA ====================
class OriginalLoRA(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        r: int):
        
        super().__init__()
        self.linear = linear
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.r = r

        # LoRA 参数
        self.alphas = 32
        self.lora_A = nn.Parameter(torch.zeros(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        if linear.bias is not None:
            self.bias = linear.bias
        else:
            self.register_buffer("bias", None)

        self.register_buffer("lora_active", torch.tensor(True))

    def forward(self, x):
        if self.lora_active:
            result = self.linear(x)
            result += (self.alphas / self.r) * (x @ self.lora_A.T @ self.lora_B.T)
            return result
        else:
            return self.linear(x)

    def merge_lora_weights(self, lora_active: bool = True) -> None:
        with torch.no_grad():
            lora_delta = (self.alphas / self.r) * (self.lora_B @ self.lora_A)
            self.linear.weight.data.add_(lora_delta.to(self.linear.weight.device))
            self.lora_B.data.zero_()
            self.lora_active = torch.tensor(lora_active)

# ==================== 原始 DoRA ====================
class OriginalDoRA(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        r: int):
        
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.r = r

        # LoRA 参数
        self.lora_A = nn.Parameter(torch.zeros(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # DoRA 参数：方向 + 幅度
        with torch.no_grad():
            weight = linear.weight.data
            weight_norm = weight.norm(p=2, dim=1, keepdim=True) + 1e-8
            self.weight_directions = nn.Parameter(
                weight / weight_norm, requires_grad=False)
            self.magnitude = nn.Parameter(weight_norm.clone(), requires_grad=True)

        if linear.bias is not None:
            self.bias = linear.bias
        else:
            self.register_buffer("bias", None)
            
        self.register_buffer("lora_active", torch.tensor(True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.lora_active:
            lora_delta = self.lora_B @ self.lora_A  # (out, in)
            adapted_weight = (self.weight_directions + lora_delta) * self.magnitude
        else:
            adapted_weight = self.weight_directions * self.magnitude
        return F.linear(x, adapted_weight, self.bias)

    def merge_lora_weights(self, lora_active: bool = True) -> None:
        with torch.no_grad():
            lora_delta = self.lora_B @ self.lora_A
            self.weight_directions.data.add_(lora_delta)
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            self.lora_B.data.zero_()
            self.lora_active = torch.tensor(lora_active)


class BaseLoRAViT(nn.Module):
    def __init__(
        self,
        vit_model: timm_ViT,
        r: int,
        lora_layer: Optional[Iterable[int]] = None,
        lora_class: type = OriginalDoRA,
        include_norm: bool = False):
        
        super().__init__()
        assert r > 0, "LoRA rank r must be positive"
        self.r = r
        try:
            self.feature_dim = vit_model.embed_dim
        except:
            self.feature_dim = 768

        self.lora_layer = (list(lora_layer) if lora_layer is not None 
                          else list(range(len(vit_model.blocks))))

        # 冻结原始参数
        for n, p in vit_model.named_parameters():
            if include_norm:
                if "norm" in n or "cls_token" in n:
                    p.requires_grad_(True)
            else:
                p.requires_grad = False

        self.lora_modules = nn.ModuleDict()

        for idx, blk in enumerate(vit_model.blocks):
            if idx not in self.lora_layer:
                continue

            # ---- QKV ----
            new_qkv = lora_class(blk.attn.qkv, r)
            blk.attn.qkv = new_qkv
            self.lora_modules[f"block_{idx}_attn_qkv"] = new_qkv

            # ---- MLP fc1 ----
            new_fc1 = lora_class(blk.mlp.fc1, r)
            blk.mlp.fc1 = new_fc1
            self.lora_modules[f"block_{idx}_mlp_fc1"] = new_fc1

            # ---- MLP fc2 ----
            new_fc2 = lora_class(blk.mlp.fc2, r)
            blk.mlp.fc2 = new_fc2
            self.lora_modules[f"block_{idx}_mlp_fc2"] = new_fc2

        self.lora_vit = vit_model
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """标准参数初始化"""
        for _, module in self.lora_modules.items():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                nn.init.kaiming_uniform_(module.lora_A, a=math.sqrt(5))
                nn.init.zeros_(module.lora_B)

    def reset_parameters_svd(self) -> None:
        """SVD-based 参数初始化"""
        for _, module in self.lora_modules.items():
            if isinstance(module, OriginalLoRA):
                W = module.linear.weight
            elif isinstance(module, OriginalDoRA):
                W = module.weight_directions

            _, _, Vh = torch.linalg.svd(W, full_matrices=False)
            module.lora_A.data = Vh[:self.r, :].clone()
            module.lora_B.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_vit(x)

    def get_module_names(self):
        return list(self.lora_modules.keys())

    def finalize_without_lora(self) -> None:
        """禁用LoRA，使用原始权重"""
        self.eval()
        for _, mod in self.lora_modules.items():
            mod.merge_lora_weights(lora_active=False)

    def merge_lora_weights(self):
        """合并LoRA权重到主干网络"""
        for _, mod in self.lora_modules.items():
            mod.merge_lora_weights()


class PlainLoRAViT(BaseLoRAViT):
    """原始LoRA ViT"""
    
    def __init__(
        self,
        vit_model: timm_ViT,
        r: int,
        lora_layer: Optional[Iterable[int]] = None,
        use_dora: bool = False,
        include_norm: bool = False):

        lora_class = OriginalDoRA if use_dora else OriginalLoRA

        super().__init__(
            vit_model=vit_model,
            r=r,
            lora_layer=lora_layer,
            lora_class=lora_class,
            include_norm=include_norm
        )


    def get_param_groups(self):
        """获取参数分组（用于优化器）"""
        params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                params.append(param)
        return params
    
    def update_projection_matrices(self):
        """更新投影矩阵（仅适用于SGP）"""
        pass

    @property
    def use_projection(self):
        return False
