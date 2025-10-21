# distiller/distiller.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import logging

def cosine_similarity_loss(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(x1, x2, dim=-1)).mean()

def feature_distillation_loss(teacher_feat: torch.Tensor, student_feat: torch.Tensor) -> torch.Tensor:
    return ((teacher_feat - student_feat) ** 2).mean()

class ResidMLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim),
            nn.GELU(),
            nn.Linear(mlp_ratio * dim, dim)
        )

    def forward(self, x):
        return x + self.mlp(x)

class Distiller(nn.Module):
    """
    A unified distillation module used by SubspaceLoRA.
    Handles teacher update, feature alignment, and loss computation.
    """

    def __init__(
        self,
        kd_type: str = "cos",
        gamma_kd: float = 0.0,
        update_teacher_each_task: bool = True,
        device: str = "cuda",
        feat_dim: int = None,
        transform: str = "weaknonlinear",  # "identity", "linear", "weaknonlinear"
        mlp_ratio: int = 4,
    ):
        super().__init__()
        self.device = device
        self.kd_type = kd_type
        self.gamma_kd = gamma_kd
        self.update_teacher_each_task = update_teacher_each_task

        # === Distillation head ===
        self.head = self._build_head(feat_dim, transform, mlp_ratio)
        self.loss_fn = self._get_loss_fn(kd_type)
        self.teacher: nn.Module = None  # Teacher model to be set externally
        logging.info(f"Distiller initialized: kd_type={kd_type}, gamma={gamma_kd}, transform_type={transform}")

    def _get_loss_fn(self, kd_type: str):
        if kd_type == "feat":
            return feature_distillation_loss
        elif kd_type == "cos":
            return cosine_similarity_loss
        else:
            raise ValueError(f"Unsupported kd_type = {kd_type}")

    def _build_head(self, feat_dim: int = None, transform: str = "identity", mlp_ratio: int = 4):
        if feat_dim is None:
            return None
        if transform == "identity":
            head = nn.Identity()
        if transform == 'linear':
            head = nn.Linear(feat_dim, feat_dim, bias=False)
            nn.init.eye_(head.weight)
        elif transform == 'weaknonlinear':
            head = ResidMLP(feat_dim, mlp_ratio)
        else:
            raise ValueError(f"Unsupported head transform = {transform}")
        return head.to(self.device)

    @torch.no_grad()
    def update_teacher(self, student_network: nn.Module):
        """Optionally update teacher network to the current student."""
        if self.update_teacher_each_task:
            self.teacher = copy.deepcopy(student_network).to(self.device)
            self.teacher.vit.finalize_without_lora()
            logging.info("✅ Teacher network updated after task (update_teacher_each_task=True).")
        else:
            logging.info("⚠️ Teacher is not updated (update_teacher_each_task=False).")

    def forward(self, inputs: torch.Tensor, student_features: torch.Tensor) -> torch.Tensor:
        """
        Compute KD loss given student features and inputs.
        If KD disabled, returns 0.0.
        """
        if self.gamma_kd <= 0.0 or self.teacher is None:
            return torch.tensor(0.0, device=self.device)

        with torch.no_grad():
            teacher_features = self.teacher.forward_features(inputs)
        student_features = self.head(student_features)
        
        kd_loss = self.loss_fn(student_features, teacher_features)
        return self.gamma_kd * kd_loss
