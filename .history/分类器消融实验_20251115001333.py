# %%
import timm
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import tqdm

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
subsets = dataset.get_subset(len(cross_domain_datasets)-1, source='train', cumulative=True, mode="test")

total_size = len(subsets)
split1_size = int(0.5 * total_size)
split2_size = total_size - split1_size

indices = list(range(total_size))
train_indices, test_indices = random_split(indices, [split1_size, split2_size])

from torch.utils.data import Subset
train_subsets = Subset(subsets, train_indices.indices)
test_subsets = Subset(subsets, test_indices.indices)
# %%
from compensator.gaussian_statistics import GaussianStatistics

def extract_features_and_labels(model, loader):
    model.eval()
    features = []
    targets = []
    dataset_ids = []
    device = "cuda"
    model.to(device)
    
    with torch.no_grad():
        for batch in tqdm.tqdm(loader):
            inputs = batch[0]
            labels = batch[1]
            inputs = inputs.to(device)
            feats = model(inputs).cpu()
            features.append(feats)
            targets.append(labels.cpu())
    

    targets_tensor = torch.cat(targets, dim=0)
    dataset_ids = infer_dataset_ids_from_labels(targets_tensor, dataset)
    return torch.cat(features, dim=0), torch.cat(targets, dim=0), torch.tensor(dataset_ids)

def infer_dataset_ids_from_labels(labels, dataset_manager):
    dataset_ids = []
    
    label_ranges = []
    for i in range(len(dataset_manager.datasets)):
        offset = dataset_manager.global_label_offset[i]
        num_classes = dataset_manager.datasets[i]['num_classes']
        label_ranges.append((offset, offset + num_classes - 1))
    
    # 为每个标签推断数据集ID
    for label in labels:
        label_item = label.item()
        for i, (start, end) in enumerate(label_ranges):
            if start <= label_item <= end:
                dataset_ids.append(i)
                break
        else:
            dataset_ids.append(0)
    return dataset_ids
# %%
adapt_backbone = True
vit = get_vit(vit_name = "vit-b-p16")
iterations = 1000

if adapt_backbone:
    print("开始适应网络主干...")
    total_classes = dataset.total_classes
    classifier = nn.Linear(768, total_classes).to("cuda")
    criterion = nn.CrossEntropyLoss()
    adapt_dataloader = torch.utils.data.DataLoader(train_subsets, batch_size=24, shuffle=True, num_workers=6)
    # 分别设置不同学习率
    optimizer = torch.optim.AdamW([
        {'params': vit.parameters(), 'lr': 1e-5},
        {'params': classifier.parameters(), 'lr': 1e-3}
    ])
    # EMA 参数
    ema_beta = 0.90  # 可调
    ema_loss = 0.0
    ema_acc = 0.0
    iteration = 0
    vit.train()
    vit.cuda()
    classifier.train()
    
    while iteration < iterations:
        for batch in adapt_dataloader:
            inputs = batch[0].to("cuda")
            labels = batch[1].to("cuda")
            optimizer.zero_grad()
            features = vit(inputs)
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 计算当前 batch 准确率
            pred = outputs.argmax(dim=1)
            acc = pred.eq(labels).float().mean().item()
            loss_val = loss.item()
            if iteration == 0:
                ema_loss = loss_val
                ema_acc = acc
            else:
                ema_loss = ema_beta * ema_loss + (1 - ema_beta) * loss_val
                ema_acc = ema_beta * ema_acc + (1 - ema_beta) * acc
            if iteration % 10 == 0:
                print(f"Iteration {iteration}, EMA Loss: {ema_loss:.4f}, EMA Acc: {ema_acc:.4f}")
            iteration += 1
            if iteration >= 500:
                break
    print("网络主干适应完成")
    vit.eval()
# %%

# 提取训练数据特征和标签
train_dataloader = torch.utils.data.DataLoader(train_subsets, batch_size=64, shuffle=False, num_workers=8)
train_features, train_labels, _ = extract_features_and_labels(vit, train_dataloader)

# 提取测试数据特征和标签
test_dataloader = torch.utils.data.DataLoader(test_subsets, batch_size=64, shuffle=False, num_workers=8)
test_features, test_labels, test_dataset_ids = extract_features_and_labels(vit, test_dataloader)

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

def evaluate_qda_classifier(alpha1, alpha3, stats, features, targets, dataset_ids, device="cuda", batch_size=512):
    alpha2 = 0.95 - alpha1
    alpha3 = alpha3
    builder = QDAClassifierBuilder(
        qda_reg_alpha1=alpha1,
        qda_reg_alpha2=alpha2,
        qda_reg_alpha3=alpha3,
        device=device)
    
    classifier = builder.build(stats)
    classifier.to(device)
    
    classifier.eval()
    classifier_device = next(classifier.parameters()).device
    dataset = torch.utils.data.TensorDataset(features, targets, dataset_ids)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_predictions = []
    all_targets = []
    all_dataset_ids = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0].to(classifier_device)
            all_targets.append(batch[1])
            all_dataset_ids.append(batch[2])
            logits = classifier(inputs)
            preds = torch.argmax(logits, dim=1)
            all_predictions.append(preds.cpu())
    
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    all_dataset_ids = torch.cat(all_dataset_ids)
    
    # 计算每个数据集的准确率，然后取平均
    unique_datasets = torch.unique(all_dataset_ids)
    dataset_accuracies = []
    
    for dataset_id in unique_datasets:
        mask = (all_dataset_ids == dataset_id)
        if mask.sum() > 0:  # 确保该数据集有样本
            dataset_correct = (all_predictions[mask] == all_targets[mask]).float().sum().item()
            dataset_total = mask.sum().item()
            dataset_acc = dataset_correct / dataset_total
            dataset_accuracies.append(dataset_acc)
            dataset_name = cross_domain_datasets[dataset_id.item()] if dataset_id.item() < len(cross_domain_datasets) else f"Unknown-{dataset_id.item()}"
            print(f"数据集 {dataset_name} (ID: {dataset_id.item()}): 准确率 {dataset_acc:.4f}, 样本数 {dataset_total}")
    
    # 计算所有数据集准确率的平均值
    dataset_wise_accuracy = np.mean(dataset_accuracies) if dataset_accuracies else 0.0
    
    torch.cuda.empty_cache()
    return dataset_wise_accuracy

def test_alpha1_range(alpha1_min=0, alpha1_max=0.95, num_points=20, log=False):
    """
    测试不同alpha1值对应的准确率
    
    参数:
        alpha1_min: alpha1的最小值
        alpha1_max: alpha1的最大值
        num_points: 测试点的数量
    
    返回:
        alpha1_values: 测试的alpha1值列表
        accuracies: 对应的准确率列表
    """
    # 生成alpha1值
    if log:
        alpha1_values = np.linspace
    else:
        alpha1_values = np.linspace(alpha1_min, alpha1_max, num_points)
    accuracies = []
    
    print(f"开始测试{num_points}个alpha1值，范围从{alpha1_min}到{alpha1_max}")
    
    for i, alpha1 in enumerate(alpha1_values):
        print(f"\n测试进度: {i+1}/{num_points}, alpha1={alpha1:.4f}")
        acc = evaluate_qda_classifier(alpha1, train_stats, test_features, test_labels, test_dataset_ids, device="cuda")
        accuracies.append(acc)
        print(f"Dataset-wise准确率: {acc:.4f}")
        print("-" * 50)
    
    return alpha1_values, accuracies

def plot_alpha1_accuracy_curve(alpha1_values, accuracies, save_path=None):
    """
    绘制alpha1到准确率的函数曲线
    
    参数:
        alpha1_values: alpha1值列表
        accuracies: 对应的准确率列表
        save_path: 图片保存路径，如果为None则不保存
    """
    plt.figure(figsize=(5, 4))
    plt.plot(alpha1_values, accuracies, 'b-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Alpha1')
    plt.ylabel('Dataset-wise Avg-Acc')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 找到最佳准确率及其对应的alpha1值
    max_acc_idx = np.argmax(accuracies)
    max_acc = accuracies[max_acc_idx]
    best_alpha1 = alpha1_values[max_acc_idx]
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    
    plt.show()
    return best_alpha1, max_acc

# %%
# 执行测试并绘制曲线
alpha1_values, accuracies = test_alpha1_range(alpha1_min=0, alpha1_max=0.4, num_points=10)
best_alpha1, best_acc = plot_alpha1_accuracy_curve(alpha1_values, accuracies, save_path="alpha1_dataset_wise_accuracy_curve.png")

print(f"\n最终结果:")
print(f"最佳alpha1值: {best_alpha1:.4f}")
print(f"对应的Dataset-wise准确率: {best_acc:.4f}")
# %%
