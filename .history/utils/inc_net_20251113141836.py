import copy
import torch
from torch import nn
from copy import deepcopy
import timm
# from lora import NullSpaceViT
from models.basic_lora import PlainLoRAViT
from models.sgp_lora import SGPLoRAViT, SGPLoRACLIPVisionTransformer
from transformers import CLIPModel, CLIPProcessor
from torchvision import transforms
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_vit(args, pretrained=False):
    name = args['vit_type']
    name = name.lower()
    rank = args['lora_rank']

    if name == 'vit-b-p16':
        vit = timm.create_model("vit_base_patch16_224", pretrained=pretrained, num_classes=0)

    elif name == 'vit-b-p16-mocov3':
        vit = timm.create_model('vit_base_patch16_224.', pretrained=False, num_classes=0)
        model_dict = torch.load('mocov3-vit-base-300ep.pth', weights_only=False)
        vit.load_state_dict(model_dict['model'], strict=True)
    
    elif name == 'vit-b-p16-dino':
        vit = timm.create_model('vit_base_patch16_224.dino', pretrained=pretrained, num_classes=0)

    elif name == 'vit-b-p16-mae':
        vit = timm.create_model('vit_base_patch16_224.mae', pretrained=pretrained, num_classes=0)
    
    elif name == 'vit-b-p16-clip':
        # For CLIP ViT, we use the same architecture as vit-b-p16 but with moco-v3 weights
        vit = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
        model_dict = torch.load('mocov3-vit-base-300ep.pth', weights_only=False)
        vit.load_state_dict(model_dict['model'], strict=True)

    else:
        raise ValueError(f'Model {name} not supported')
    
    vit.head = nn.Identity()
    del vit.norm
    vit.norm = nn.LayerNorm(768, elementwise_affine=False)
    # vit.norm = nn.LayerNorm(768, elementwise_affine=True)
    
    lora_type = args['lora_type']
    if lora_type == "full":
        return NullSpaceViT(vit, use_projection=args['use_projection'])
    
    elif lora_type == "full_ft":
        from models.full_finetune import FullFinetuneViT
        return FullFinetuneViT(vit, include_norm=args.get('include_norm', False),
                             freeze_patch_embed=args.get('freeze_patch_embed', True),
                             finetune_layers=args.get('finetune_layers', None))
    
    elif lora_type == "basic_lora":
        return PlainLoRAViT(vit, r=rank, include_norm=False)

    elif lora_type == "sgp_lora":
        return SGPLoRAViT(vit, r=rank, weight_temp=args['weight_temp'], use_soft_projection=True, weight_kind=args['weight_kind'], weight_p=args['weight_p'])
    
    elif lora_type == "nsp_lora":
        return SGPLoRAViT(vit, r=rank, weight_temp=args['weight_temp'], use_soft_projection=False, nsp_eps=args['nsp_eps'], nsp_weight=args['nsp_weight'])

    else:
        raise ValueError(f"LoRA type {lora_type} not supported")


def get_clip_model(args, train_mode="lora"):
    """
    train_mode: "lora" | "full" | "frozen"
    """
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    if train_mode == "frozen":
        for p in model.parameters():
            p.requires_grad = False
        # 不加 LoRA，完全冻结
        return model, processor

    elif train_mode == "full":
        for n, p in model.named_parameters():
            if "vision_model.encoder.layers" in n and ("self_attn" in n or "mlp" in n):
                p.requires_grad = True
            else:
                p.requires_grad = False
        return model, processor

    elif train_mode == "lora":
        for p in model.parameters():
            p.requires_grad = False

        rank = args['lora_rank']

        use_soft_projection = bool(args.get('sgp_soft_projection', True))

        model.vision_model = SGPLoRACLIPVisionTransformer(
            model.vision_model,
            r=rank,
            weight_temp=args['weight_temp'],
            use_soft_projection=use_soft_projection,
            weight_kind=args['weight_kind'],
            weight_p=args['weight_p'])
        
        return model, processor

    else:
        raise ValueError(f"Unsupported train_mode: {train_mode}")


class ContinualLinear(nn.Module):
    def __init__(self, embed_dim, nb_classes):
        super().__init__()
        self.embed_dim = embed_dim
        self.heads = nn.ModuleList([nn.Linear(embed_dim, nb_classes, bias=False)])
        self.head_weights = nn.Parameter(torch.ones(nb_classes))
        self.current_output_size = nb_classes

    def update(self, nb_classes):
        new_head = nn.Linear(self.embed_dim, nb_classes, bias=False)
        self.heads.append(new_head)
        new_head_weights = nn.Parameter(torch.ones(self.current_output_size + nb_classes))

        with torch.no_grad():
            new_head_weights[:self.current_output_size] = self.head_weights
            new_head_weights[self.current_output_size:] = 1.0
        
        self.head_weights = new_head_weights
        self.current_output_size += nb_classes

    def forward(self, x):
        outputs = [head(x) for head in self.heads]
        combined = torch.cat(outputs, dim=1)
        return combined * self.head_weights


class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()
        self.vit = get_vit(args, pretrained)
        self.fc = None

    def extract_vector(self, x):
        return self.vit(x)

    def forward(self, x):
        feat = self.vit(x)
        logits = self.fc(feat)
        return feat, logits
    
    def forward_features(self, x):
        return self.vit(x)
    
    def update_projection_matrices(self, covariances):
        self.vit.update_projection_matrices(covariances)
    
    @property
    def feature_dim(self):
        return self.vit.feature_dim

    def update_fc(self, nb_classes):
        if self.fc is None:
            self.fc = ContinualLinear(self.feature_dim, nb_classes)
        else:
            self.fc.update(nb_classes)

    def copy(self):
        return copy.deepcopy(self)


class CLIP_BaseNet(nn.Module):
    def __init__(self, args, train_mode="full"):
        super(CLIP_BaseNet, self).__init__()
        self.train_mode = train_mode

        self.model, self.processor = get_clip_model(args, train_mode=train_mode)

        self.valid_preprocess = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711])])

    def forward(self, img, text):
        x = self.model.get_image_features(img)
        y = self.model.get_text_features(text)
        return x, y

    def encode_image(self, img):
        return self.model.get_image_features(img)

    def encode_text(self, text):
        text_inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True)
        text_inputs = {k: v.to(self.model.device) for k, v in text_inputs.items()}
        text_features = self.model.get_text_features(**text_inputs)
        return text_features

    @property
    def feature_dim(self):
        return self.model.config.projection_dim  # CLIP 输出维度，通常是 512