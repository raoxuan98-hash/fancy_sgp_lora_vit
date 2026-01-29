# %%
import timm
import torch
import torch.nn as nn

def get_vit(vit_name = "vit-b-p16"):
    name = vit_name.lower()
    if name == 'vit-b-p16':
        vit = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
    elif name == 'vit-b-p16-mocov3':
        vit = timm.create_model('vit_base_patch16_224.', pretrained=False, num_classes=0)
        model_dict = torch.load('mocov3-vit-base-300ep.pth', weights_only=False)
        vit.load_state_dict(model_dict['model'], strict=True)
    elif name == 'vit-b-p16-dino':
        vit = timm.create_model('vit_base_patch16_224.dino', pretrained=True, num_classes=0)
    elif name == 'vit-b-p16-mae':
        vit = timm.create_model('vit_base_patch16_224.mae', pretrained=True, num_classes=0)
    elif name == 'vit-b-p16-clip':
        vit = timm.create_model("vit_base_patch16_clip_224.openai", pretrained=True, num_classes=0)
    else:
        raise ValueError(f'Model {name} not supported')
    vit.head = nn.Identity()
    del vit.norm
    vit.norm = nn.LayerNorm(768, elementwise_affine=False)
    return vit

# %%
cross_domain_datasets = ['resisc45', 'imagenet-r', 'caltech-101', 'dtd', 'fgvc-aircraft-2013b-variants102', 'food-101', 'mnist', 'oxford-flower-102', 'oxford-iiit-pets', 'cars196_224']
from utils.cross_domain_data_manager import CrossDomainDataManagerCore
dataset = CrossDomainDataManagerCore(cross_domain_datasets, False, 0, 32, 0)
# 使用训练数据构建分类器
train_subsets = dataset.get_subset(len(cross_domain_datasets)-1, source='train', cumulative=True, mode="train")
# 使用测试数据评估分类器
test_subsets = dataset.get_subset(len(cross_domain_datasets)-1, source='test', cumulative=True, mode="test")

# %%
from compensator.gaussian_statistics import GaussianStatistics
import tqdm

def extract_features_and_labels(model, test_loader):
    model.eval()
    features = []
    targets = []
    device = "cuda"
    model.to(device)
    
    with torch.no_grad():
        for batch in tqdm.tqdm(test_loader):
            inputs = batch[0]
            labels = batch[1]
            inputs = inputs.to(device)
            feats = model(inputs).cpu()
            features.append(feats)
            targets.append(labels.cpu())
    
    return torch.cat(features, dim=0), torch.cat(targets, dim=0)

vit = get_vit("vit-b-p16")
dataloader = torch.utils.data.DataLoader(all_subsets, batch_size=64, shuffle=False, num_workers=8)
cached_features, cached_labels = extract_features_and_labels(vit, dataloader)

# %%
def build_gaussian_statistics(features: torch.Tensor, labels: torch.Tensor):
    """为每个类别构建高斯统计量"""
    features = features.cpu()
    labels = labels.cpu()
    unique_labels = torch.unique(labels)
    
    stats = {}
    for lbl in tqdm.tqdm(unique_labels):
        mask = (labels == lbl)
        feats_class = features[mask]
        
        mu = feats_class.mean(0)
        if feats_class.size(0) >= 2:
            cov = torch.cov(feats_class.T) + torch.eye(feats_class.size(1)) * 1e-4
        else:
            cov = torch.eye(feats_class.size(1)) * 1e-4
        stats[int(lbl.item())] = GaussianStatistics(mu, cov)
    return stats


cached_stats = build_gaussian_statistics(cached_features, cached_labels)

# %%
from classifier.da_classifier_builder import QDAClassifierBuilder

def evaluate_qda_classifier(alpha1, stats, features, targets, device="cuda", batch_size=512):
    alpha2 = 0.95 - alpha1
    alpha3 = 0.05
    builder = QDAClassifierBuilder(
        qda_reg_alpha1=alpha1,
        qda_reg_alpha2=alpha2,
        qda_reg_alpha3=alpha3,
        device=device)
    
    classifier = builder.build(stats)
    classifier.to(device)
    
    classifier.eval()
    classifier_device = next(classifier.parameters()).device
    dataset = torch.utils.data.TensorDataset(features, targets)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0].to(classifier_device)
            logits = classifier(inputs)
            preds = torch.argmax(logits, dim=1)
            predictions.append(preds.cpu())

    accuracy = (torch.cat(predictions) == targets).float().mean().item()
    torch.cuda.empty_cache()
    return accuracy

acc = evaluate_qda_classifier(0.10, cached_stats, cached_features, cached_labels, device="cuda")
print(acc)


