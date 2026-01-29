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
from torch.utils.data import random_split, Dataset

cross_domain_datasets = ['resisc45', 'imagenet-r', 'caltech-101', 'dtd', 'fgvc-aircraft-2013b-variants102', 'food-101', 'mnist', 'oxford-flower-102', 'oxford-iiit-pets', 'cars196_224']
from utils.cross_domain_data_manager import CrossDomainDataManagerCore
dataset = CrossDomainDataManagerCore(cross_domain_datasets, False, 0, 32, 0)

# 直接获取训练集和测试集，避免数据泄露
subsets = dataset.get_subset(len(cross_domain_datasets)-1, source='train', cumulative=True, mode="test")

total_size = len(subsets)
split1_size = int(0.5 * total_size)
split2_size = total_size - split1_size
train_subsets, test_subsets = random_split(subsets, [split1_size, split2_size])
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
# 提取训练数据特征和标签
train_dataloader = torch.utils.data.DataLoader(train_subsets, batch_size=64, shuffle=False, num_workers=8)
train_features, train_labels = extract_features_and_labels(vit, train_dataloader)

# 提取测试数据特征和标签
test_dataloader = torch.utils.data.DataLoader(test_subsets, batch_size=64, shuffle=False, num_workers=8)
test_features, test_labels = extract_features_and_labels(vit, test_dataloader)
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


# 使用训练数据构建高斯统计量
train_stats = build_gaussian_statistics(train_features, train_labels)
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

# 使用测试数据评估分类器
# %%
# %%
import numpy as np
import matplotlib.pyplot as plt

def test_alpha1_range(alpha1_min=0, alpha1_max=0.95, num_points=20):
    # 生成alpha1值
    alpha1_values = np.linspace(alpha1_min, alpha1_max, num_points)
    accuracies = []
    
    print(f"开始测试{num_points}个alpha1值，范围从{alpha1_min}到{alpha1_max}")
    
    for i, alpha1 in enumerate(alpha1_values):
        print(f"测试进度: {i+1}/{num_points}, alpha1={alpha1:.4f}")
        acc = evaluate_qda_classifier(alpha1, train_stats, test_features, test_labels, device="cuda")
        accuracies.append(acc)
        print(f"准确率: {acc:.4f}")
    
    return alpha1_values, accuracies

def plot_alpha1_accuracy_curve(alpha1_values, accuracies, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(alpha1_values, accuracies, 'b-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Alpha1 值')
    plt.ylabel('准确率')
    plt.title('Alpha1 到准确率的函数曲线')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 找到最佳准确率及其对应的alpha1值
    max_acc_idx = np.argmax(accuracies)
    max_acc = accuracies[max_acc_idx]
    best_alpha1 = alpha1_values[max_acc_idx]
    
    # 标记最佳点
    plt.plot(best_alpha1, max_acc, 'ro', markersize=8)
    plt.annotate(f'最佳点: alpha1={best_alpha1:.3f}, acc={max_acc:.4f}', 
                 xy=(best_alpha1, max_acc), 
                 xytext=(best_alpha1+0.05, max_acc-0.02),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    
    plt.show()
    return best_alpha1, max_acc
# %%
# 执行测试并绘制曲线
alpha1_values, accuracies = test_alpha1_range(alpha1_min=0, alpha1_max=0.95, num_points=20)
best_alpha1, best_acc = plot_alpha1_accuracy_curve(alpha1_values, accuracies, save_path="alpha1_accuracy_curve.png")

print(f"最佳alpha1值: {best_alpha1:.4f}, 对应的准确率: {best_acc:.4f}")

# %%
