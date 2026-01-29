# %% matplotlib格式设置
import timm
import torch
import torch.nn as nn
import numpy as np
import matplotlib
# IEEE单栏风格设置：使用系统可用字体，避免字体警告
import matplotlib.font_manager as fm
import os

# 检查可用字体并设置合适的字体
available_fonts = [f.name for f in fm.fontManager.ttflist]
if 'Times New Roman' in available_fonts:
    serif_font = 'Times New Roman'
elif 'serif' in available_fonts:
    serif_font = 'serif'
else:
    # 使用系统中任何可用的serif字体
    serif_fonts = ['DejaVu Serif', 'Liberation Serif', 'Bitstream Vera Serif']
    serif_font = 'DejaVu Serif'  # 默认使用DejaVu Serif，这是Linux系统常用字体
    
matplotlib.rcParams['font.family'] = ['serif']
matplotlib.rcParams['font.serif'] = [serif_font]
matplotlib.rcParams['text.usetex'] = False  # 避免Type3字体问题
print(f"使用字体: {serif_font}")
matplotlib.rcParams['figure.figsize'] = (3.5, 2.5)  # IEEE单栏标准尺寸
matplotlib.rcParams['font.size'] = 9  # IEEE标准字体大小
matplotlib.rcParams['axes.linewidth'] = 0.5  # 细线宽
matplotlib.rcParams['lines.linewidth'] = 1.0  # 标准线宽
import matplotlib.pyplot as plt
import tqdm
import time

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
    # del vit.norm
    # vit.norm = nn.LayerNorm(768, elementwise_affine=False)
    return vit
# %%
from torch.utils.data import random_split, Dataset, Subset

num_shots = 128
model_name = "vit-b-p16-clip"

cross_domain_datasets = ['cifar100_224', 'cub200_224', 'resisc45', 'imagenet-r', 'caltech-101', 'dtd', 'fgvc-aircraft-2013b-variants102', 'food-101', 'mnist', 'oxford-flower-102', 'oxford-iiit-pets', 'cars196_224']
from utils.cross_domain_data_manager import CrossDomainDataManagerCore
dataset = CrossDomainDataManagerCore(cross_domain_datasets, shuffle=False, seed=0, num_shots=num_shots)
subsets = dataset.get_subset(len(cross_domain_datasets)-1, source='train', cumulative=True, mode="test")

# 直接使用random_split分割数据集
train_subsets, test_subsets = random_split(subsets, [0.5, 0.5])
# %%
import os
from compensator.gaussian_statistics import GaussianStatistics

def extract_features_and_labels(model, train_loader, test_loader, model_name, num_shots, iterations=None, cache_dir="cached_data/classifier_ablation"):
    # 将训练步数纳入缓存键，如果提供了iterations参数
    if iterations is not None:
        cache_key = f"{model_name}_{num_shots}_iter{iterations}_features_cache"
    else:
        cache_key = f"{model_name}_{num_shots}_features_cache"
    cache_path = os.path.join(cache_dir, cache_key)
    
    if (os.path.exists(cache_path + "_train_features.pt") and
        os.path.exists(cache_path + "_train_labels.pt") and
        os.path.exists(cache_path + "_train_dataset_ids.pt") and
        os.path.exists(cache_path + "_test_features.pt") and
        os.path.exists(cache_path + "_test_labels.pt") and
        os.path.exists(cache_path + "_test_dataset_ids.pt")):
        print(f"检测到缓存文件，直接加载特征和标签...")
        print(f"缓存键: {cache_key}")
        train_features = torch.load(cache_path + "_train_features.pt")
        train_labels = torch.load(cache_path + "_train_labels.pt")
        train_dataset_ids = torch.load(cache_path + "_train_dataset_ids.pt")
        
        test_features = torch.load(cache_path + "_test_features.pt")
        test_labels = torch.load(cache_path + "_test_labels.pt")
        test_dataset_ids = torch.load(cache_path + "_test_dataset_ids.pt")
        
        print(f"缓存加载完成:")
        print(f"  训练特征: {train_features.shape}")
        print(f"  训练标签: {train_labels.shape}")
        print(f"  测试特征: {test_features.shape}")
        print(f"  测试标签: {test_labels.shape}")
        
        return train_features, train_labels, train_dataset_ids, test_features, test_labels, test_dataset_ids
    
    print("未检测到缓存，开始提取特征...")
    
    model.eval()
    device = "cuda"
    model.to(device)
    
    # 提取训练特征
    print("提取训练特征...")
    train_features = []
    train_targets = []
    
    with torch.no_grad():
        for batch in tqdm.tqdm(train_loader):
            inputs = batch[0].to(device)
            labels = batch[1]
            feats = model(inputs).cpu()
            train_features.append(feats)
            train_targets.append(labels.cpu())
    
    train_features = torch.cat(train_features, dim=0)
    train_labels = torch.cat(train_targets, dim=0)
    train_dataset_ids = torch.tensor(infer_dataset_ids_from_labels(train_labels, dataset))
    
    # 提取测试特征
    print("提取测试特征...")
    test_features = []
    test_targets = []
    
    with torch.no_grad():
        for batch in tqdm.tqdm(test_loader):
            inputs = batch[0].to(device)
            labels = batch[1]
            feats = model(inputs).cpu()
            test_features.append(feats)
            test_targets.append(labels.cpu())
    
    test_features = torch.cat(test_features, dim=0)
    test_labels = torch.cat(test_targets, dim=0)
    test_dataset_ids = torch.tensor(infer_dataset_ids_from_labels(test_labels, dataset))
    
    # 保存缓存
    print("保存特征缓存...")
    os.makedirs(cache_dir, exist_ok=True)
    torch.save(train_features, cache_path + "_train_features.pt")
    torch.save(train_labels, cache_path + "_train_labels.pt")
    torch.save(train_dataset_ids, cache_path + "_train_dataset_ids.pt")
    torch.save(test_features, cache_path + "_test_features.pt")
    torch.save(test_labels, cache_path + "_test_labels.pt")
    torch.save(test_dataset_ids, cache_path + "_test_dataset_ids.pt")
    
    print(f"缓存已保存到: {cache_dir}")
    
    return train_features, train_labels, train_dataset_ids, test_features, test_labels, test_dataset_ids

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
vit = get_vit(vit_name = model_name)
iterations = 500

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

            if iteration >= iterations:
                break
    print("网络主干适应完成")
    vit.eval()
# %%
# 提取训练数据特征和标签
train_dataloader = torch.utils.data.DataLoader(train_subsets, batch_size=64, shuffle=False, num_workers=8)
test_dataloader = torch.utils.data.DataLoader(test_subsets, batch_size=64, shuffle=False, num_workers=8)
train_features, train_labels, train_dataset_ids, test_features, test_labels, test_dataset_ids = extract_features_and_labels(vit, train_dataloader, test_dataloader, model_name, num_shots=num_shots)
train_dataset_ids = torch.tensor(train_dataset_ids)
test_dataset_ids = torch.tensor(test_dataset_ids)
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
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

def evaluate_qda_classifier(alpha1, alpha2, alpha3, stats, features, targets, dataset_ids,
                           device="cuda", batch_size=512, custom_classifier=None):
    """
    评估分类器性能，支持自定义分类器
    """
    if custom_classifier is None:
        builder = QDAClassifierBuilder(
            qda_reg_alpha1=alpha1,
            qda_reg_alpha2=alpha2,
            qda_reg_alpha3=alpha3,
            device=device)
        
        classifier = builder.build(stats)
    else:
        classifier = custom_classifier
    
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
    
    # 计算每个数据集的准确率
    unique_datasets = torch.unique(all_dataset_ids)
    dataset_accuracies = []
    
    for dataset_id in unique_datasets:
        mask = (all_dataset_ids == dataset_id)
        if mask.sum() > 0:
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

def grid_search_alpha1_alpha2(alpha1_min=0, alpha1_max=1.0, alpha2_min=0, alpha2_max=1.0,
                           alpha1_points=10, alpha2_points=10):
    alpha1_values = np.linspace(alpha1_min, alpha1_max, alpha1_points)
    alpha2_values = np.linspace(alpha2_min, alpha2_max, alpha2_points)
    alpha3_fixed = 0.5
    
    # 初始化准确率矩阵
    accuracy_matrix = np.zeros((alpha1_points, alpha2_points))
    
    print(f"开始二维网格搜索: {alpha1_points} x {alpha2_points} = {alpha1_points * alpha2_points} 个组合")
    print(f"Alpha3固定为: {alpha3_fixed}")
    
    total_tests = alpha1_points * alpha2_points
    test_count = 0
    
    for i, alpha1 in enumerate(alpha1_values):
        for j, alpha2 in enumerate(alpha2_values):
            test_count += 1
            print(f"\n测试进度: {test_count}/{total_tests}, alpha1={alpha1:.4f}, alpha2={alpha2:.4f}")
            
            # 评估当前参数组合
            acc = evaluate_qda_classifier(alpha1, alpha2, alpha3_fixed, train_stats, test_features,
                                         test_labels, test_dataset_ids, device="cuda")
            accuracy_matrix[i, j] = acc
            print(f"Dataset-wise准确率: {acc:.4f}")
            print("-" * 50)
    
    return alpha1_values, alpha2_values, accuracy_matrix

# %%
# 执行二维网格搜索并绘制等高线图
alpha1_values, alpha2_values, accuracy_matrix = grid_search_alpha1_alpha2(
    alpha1_min=0, alpha1_max=4.0, alpha2_min=0, alpha2_max=4.0,
    alpha1_points=5, alpha2_points=5
)

# %%
def plot_alpha1_alpha2_contour(alpha1_values, alpha2_values, accuracy_matrix, save_path=None):
    plt.figure(figsize=(3.5, 2.5))  # IEEE单栏标准尺寸
    
    # 创建网格
    alpha1_grid, alpha2_grid = np.meshgrid(alpha1_values, alpha2_values)
    
    # 绘制等高线图
    contour = plt.contourf(alpha1_grid, alpha2_grid, accuracy_matrix.T, levels=20, cmap='viridis')
    cbar = plt.colorbar(contour, shrink=0.8)
    cbar.set_label('Dataset-wise Avg-Acc', fontsize=9)
    
    # 添加等高线
    contour_lines = plt.contour(alpha1_grid, alpha2_grid, accuracy_matrix.T, levels=10, colors='black', alpha=0.4)
    plt.clabel(contour_lines, inline=True, fontsize=7)
    
    # 找到最佳准确率及其对应的参数
    max_idx = np.unravel_index(np.argmax(accuracy_matrix), accuracy_matrix.shape)
    best_alpha1 = alpha1_values[max_idx[0]]
    best_alpha2 = alpha2_values[max_idx[1]]
    best_acc = accuracy_matrix[max_idx]
    
    # 标记最佳点
    plt.plot(best_alpha1, best_alpha2, 'r*', markersize=6, label=f'Best: ({best_alpha1:.3f}, {best_alpha2:.3f}) = {best_acc:.4f}')
    
    plt.xlabel('Alpha1', fontsize=9)
    plt.ylabel('Alpha2', fontsize=9)
    plt.title('Alpha1-Alpha2 Grid Search Results (Alpha3=0.2)', fontsize=9)
    plt.legend(fontsize=7, loc='best')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', fonttype=42)
        print(f"等高线图已保存到: {save_path}")
    
    plt.show()
    return best_alpha1, best_alpha2, best_acc

def plot_alpha1_accuracy_curve(alpha1_values, accuracies, save_path=None):
    plt.figure(figsize=(3.5, 2.5))  # IEEE单栏标准尺寸
    plt.plot(alpha1_values, accuracies, 'b-', linewidth=1.0, marker='o', markersize=3)
    plt.xlabel('Alpha1', fontsize=9)
    plt.ylabel('Dataset-wise Avg-Acc', fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 找到最佳准确率及其对应的alpha1值
    max_acc_idx = np.argmax(accuracies)
    max_acc = accuracies[max_acc_idx]
    best_alpha1 = alpha1_values[max_acc_idx]
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', fonttype=42)
        print(f"图表已保存到: {save_path}")
    
    plt.show()
    return best_alpha1, max_acc

# %%
best_alpha1, best_alpha2, best_acc = plot_alpha1_alpha2_contour(
    alpha1_values, alpha2_values, accuracy_matrix,
    save_path="实验结果保存/分类器消融实验/alpha1_alpha2_contour.png"
)

print(f"\n最终结果:")
print(f"最佳alpha1值: {best_alpha1:.4f}")
print(f"最佳alpha2值: {best_alpha2:.4f}")
print(f"对应的Dataset-wise准确率: {best_acc:.4f}")
print(f"固定的alpha3值: 0.2")
# %% 实验2: 参数敏感性分析
def run_experiment2_sensitivity_analysis(alpha1_range=(0.0, 5.0), alpha2_range=(0.0, 5.0),
                                       alpha_sum=3.0, fixed_points=21, save_path=None):
    """
    实验2: 参数敏感性分析
    1. 固定alpha1 + alpha2 = 常数
    2. 固定alpha1，变动alpha2
    3. 固定alpha2，变动alpha1
    """
    print("\n" + "="*50)
    print("实验2: 参数敏感性分析")
    print("="*50)
    
    results = {
        'fixed_sum': {'alpha1': [], 'alpha2': [], 'accuracy': []},
        'fixed_alpha1': {'alpha1': [], 'alpha2': [], 'accuracy': []},
        'fixed_alpha2': {'alpha1': [], 'alpha2': [], 'accuracy': []}
    }
    
    alpha3_fixed = 0.5
    
    # 1. 固定alpha1 + alpha2 = 常数
    print(f"\n1. 固定alpha1 + alpha2 = {alpha_sum}")
    alpha1_values = np.linspace(0, alpha_sum, fixed_points)
    
    for alpha1 in alpha1_values:
        alpha2 = alpha_sum - alpha1
        if alpha2 < 0:
            continue
        
        print(f"测试: alpha1={alpha1:.4f}, alpha2={alpha2:.4f}")
        
        acc = evaluate_qda_classifier(alpha1, alpha2, alpha3_fixed, train_stats, test_features,
                                     test_labels, test_dataset_ids, device="cuda")
        
        results['fixed_sum']['alpha1'].append(alpha1)
        results['fixed_sum']['alpha2'].append(alpha2)
        results['fixed_sum']['accuracy'].append(acc)
        
        print(f"准确率: {acc:.4f}")
    
    # 2. 固定alpha1，变动alpha2
    print(f"\n2. 固定alpha1，变动alpha2")
    fixed_alpha1 = 1.0
    alpha2_values = np.linspace(alpha2_range[0], alpha2_range[1], fixed_points)
    
    for alpha2 in alpha2_values:
        print(f"测试: alpha1={fixed_alpha1:.4f}, alpha2={alpha2:.4f}")
        
        acc = evaluate_qda_classifier(fixed_alpha1, alpha2, alpha3_fixed, train_stats, test_features,
                                     test_labels, test_dataset_ids, device="cuda")
        
        results['fixed_alpha1']['alpha1'].append(fixed_alpha1)
        results['fixed_alpha1']['alpha2'].append(alpha2)
        results['fixed_alpha1']['accuracy'].append(acc)
        
        print(f"准确率: {acc:.4f}")
    
    # 3. 固定alpha2，变动alpha1
    print(f"\n3. 固定alpha2，变动alpha1")
    fixed_alpha2 = 2.0
    alpha1_values = np.linspace(alpha1_range[0], alpha1_range[1], fixed_points)
    
    for alpha1 in alpha1_values:
        print(f"测试: alpha1={alpha1:.4f}, alpha2={fixed_alpha2:.4f}")
        
        acc = evaluate_qda_classifier(alpha1, fixed_alpha2, alpha3_fixed, train_stats, test_features,
                                     test_labels, test_dataset_ids, device="cuda")
        
        results['fixed_alpha2']['alpha1'].append(alpha1)
        results['fixed_alpha2']['alpha2'].append(fixed_alpha2)
        results['fixed_alpha2']['accuracy'].append(acc)
        
        print(f"准确率: {acc:.4f}")
    
    # 绘制敏感性分析图
    plot_experiment2_sensitivity(results, save_path)
    
    return results

def plot_experiment2_sensitivity(results, save_path=None):
    """绘制实验2敏感性分析图"""
    fig, axes = plt.subplots(1, 3, figsize=(3.5*3, 2.5))  # IEEE单栏尺寸适配
    
    # 1. 固定和
    ax = axes[0]
    ax.plot(results['fixed_sum']['alpha1'], results['fixed_sum']['accuracy'],
            'b-', linewidth=1.0, marker='o', markersize=3)
    ax.set_xlabel('Alpha1 (Alpha1+Alpha2=Constant)', fontsize=9)
    ax.set_ylabel('Dataset-wise Avg-Acc', fontsize=9)
    ax.set_title('Fixed Sum', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # 2. 固定alpha1
    ax = axes[1]
    ax.plot(results['fixed_alpha1']['alpha2'], results['fixed_alpha1']['accuracy'],
            'r-', linewidth=1.0, marker='s', markersize=3)
    ax.set_xlabel('Alpha2 (Alpha1=1.0)', fontsize=9)
    ax.set_ylabel('Dataset-wise Avg-Acc', fontsize=9)
    ax.set_title('Fixed Alpha1', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # 3. 固定alpha2
    ax = axes[2]
    ax.plot(results['fixed_alpha2']['alpha1'], results['fixed_alpha2']['accuracy'],
            'g-', linewidth=1.0, marker='^', markersize=3)
    ax.set_xlabel('Alpha1 (Alpha2=2.0)', fontsize=9)
    ax.set_ylabel('Dataset-wise Avg-Acc', fontsize=9)
    ax.set_title('Fixed Alpha2', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', fonttype=42)
        print(f"敏感性分析图已保存到: {save_path}")
    
    plt.show()

# %% 实验3: SGD分类器对比及baseline集成
def run_experiment3_classifier_comparison(sgd_epochs=5, sgd_lr=0.01, save_path=None):
    """
    实验3: SGD分类器对比及baseline集成
    对比RGDA、SGD、NCM、LDA等分类器
    """
    print("\n" + "="*50)
    print("实验3: SGD分类器对比及baseline集成")
    print("="*50)
    
    results = {}
    alpha3_fixed = 0.5
    
    # 1. RGDA分类器（不同参数）
    print("\n1. 测试RGDA分类器...")
    rgda_params = [
        (0.5, 0.5, "RGDA(0.5,0.5)"),
        (1.0, 2.0, "RGDA(1.0,2.0)"),
        (2.0, 3.0, "RGDA(2.0,3.0)"),
    ]
    
    for alpha1, alpha2, name in rgda_params:
        print(f"测试: {name}")
        
        acc = evaluate_qda_classifier(alpha1, alpha2, alpha3_fixed, train_stats, test_features,
                                     test_labels, test_dataset_ids, device="cuda")
        
        results[name] = acc
        print(f"准确率: {acc:.4f}")
    
    # 2. SGD分类器
    print("\n2. 测试SGD分类器...")
    
    from classifier.sgd_classifier_builder import SGDClassifierBuilder
    
    # 生成随机样本用于SGD训练
    cached_Z = torch.randn(1024, train_features.size(1))
    
    # 构建SGD分类器
    sgd_builder = SGDClassifierBuilder(
        cached_Z=cached_Z,
        device="cuda",
        epochs=sgd_epochs,
        lr=sgd_lr
    )
    
    classifier = sgd_builder.build(train_stats)
    acc = evaluate_qda_classifier(0, 0, 0, train_stats, test_features,
                                 test_labels, test_dataset_ids, device="cuda",
                                 custom_classifier=classifier)
    
    results['SGD'] = acc
    print(f"SGD准确率: {acc:.4f}")
    
    # 3. NCM baseline
    print("\n3. 测试NCM baseline...")
    
    from classifier.ncm_classifier import NCMClassifier
    
    ncm_classifier = NCMClassifier(train_stats).to("cuda")
    acc = evaluate_qda_classifier(0, 0, 0, train_stats, test_features,
                                 test_labels, test_dataset_ids, device="cuda",
                                 custom_classifier=ncm_classifier)
    
    results['NCM'] = acc
    print(f"NCM准确率: {acc:.4f}")
    
    # 4. LDA baseline (alpha1=0的RGDA变体)
    print("\n4. 测试LDA baseline...")
    
    from classifier.da_classifier_builder import LDAClassifierBuilder
    
    lda_builder = LDAClassifierBuilder(
        reg_alpha=0.3,
        device="cuda"
    )
    
    classifier = lda_builder.build(train_stats)
    acc = evaluate_qda_classifier(0, 0, 0, train_stats, test_features,
                                 test_labels, test_dataset_ids, device="cuda",
                                 custom_classifier=classifier)
    
    results['LDA'] = acc
    print(f"LDA准确率: {acc:.4f}")
    
    # 绘制对比图
    plot_experiment3_comparison(results, save_path)
    
    return results

def plot_experiment3_comparison(results, save_path=None):
    """绘制实验3对比图"""
    plt.figure(figsize=(3.5, 2.5))  # IEEE单栏标准尺寸
    
    classifiers = list(results.keys())
    accuracies = list(results.values())
    
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow']
    bars = plt.bar(classifiers, accuracies, color=colors[:len(classifiers)], alpha=0.7, width=0.6)
    
    # 添加数值标签
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{acc:.4f}', ha='center', va='bottom', fontsize=7)
    
    plt.xlabel('Classifier Type', fontsize=9)
    plt.ylabel('Dataset-wise Avg-Acc', fontsize=9)
    plt.title('Classifier Performance Comparison', fontsize=9)
    plt.xticks(rotation=45, fontsize=7)
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', fonttype=42)
        print(f"分类器对比图已保存到: {save_path}")
    
    plt.show()

# %% 实验4: 计算效率对比
def measure_classifier_efficiency(classifier_builder, stats, features, device="cuda"):
    """测量分类器构建和预测效率"""
    # 测量构建时间
    start_time = time.time()
    classifier = classifier_builder.build(stats)
    build_time = time.time() - start_time
    
    # 测量预测时间
    classifier.to(device)
    classifier.eval()
    
    # 使用一小部分测试数据测量预测时间
    sample_features = features[:1000].to(device)
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(10):  # 多次预测取平均
            _ = classifier(sample_features)
    predict_time = (time.time() - start_time) / 10
    
    torch.cuda.empty_cache()
    return build_time, predict_time

def run_experiment4_efficiency_comparison(save_path=None):
    """
    实验4: 计算效率对比
    对比不同分类器的构建时间和预测时间
    """
    print("\n" + "="*50)
    print("实验4: 计算效率对比")
    print("="*50)
    
    results = {}
    alpha3_fixed = 0.5
    
    # 1. RGDA分类器效率
    print("\n1. 测试RGDA分类器效率...")
    
    rgda_builder = QDAClassifierBuilder(
        qda_reg_alpha1=1.0,
        qda_reg_alpha2=2.0,
        qda_reg_alpha3=alpha3_fixed,
        device="cuda"
    )
    
    build_time, predict_time = measure_classifier_efficiency(
        rgda_builder, train_stats, test_features, device="cuda"
    )
    results['RGDA'] = {'build_time': build_time, 'predict_time': predict_time}
    
    print(f"RGDA构建时间: {build_time:.4f}s")
    print(f"RGDA预测时间: {predict_time:.4f}s")
    
    # 2. SGD分类器效率
    print("\n2. 测试SGD分类器效率...")
    
    from classifier.sgd_classifier_builder import SGDClassifierBuilder
    
    cached_Z = torch.randn(1024, train_features.size(1))
    sgd_builder = SGDClassifierBuilder(
        cached_Z=cached_Z,
        device="cuda",
        epochs=5,
        lr=0.01
    )
    
    build_time, predict_time = measure_classifier_efficiency(
        sgd_builder, train_stats, test_features, device="cuda"
    )
    results['SGD'] = {'build_time': build_time, 'predict_time': predict_time}
    
    print(f"SGD构建时间: {build_time:.4f}s")
    print(f"SGD预测时间: {predict_time:.4f}s")
    
    # 3. LDA分类器效率
    print("\n3. 测试LDA分类器效率...")
    
    from classifier.da_classifier_builder import LDAClassifierBuilder
    
    lda_builder = LDAClassifierBuilder(
        reg_alpha=0.3,
        device="cuda"
    )
    
    build_time, predict_time = measure_classifier_efficiency(
        lda_builder, train_stats, test_features, device="cuda"
    )
    results['LDA'] = {'build_time': build_time, 'predict_time': predict_time}
    
    print(f"LDA构建时间: {build_time:.4f}s")
    print(f"LDA预测时间: {predict_time:.4f}s")
    
    # 4. NCM分类器效率
    print("\n4. 测试NCM分类器效率...")
    
    from classifier.ncm_classifier import NCMClassifier
    
    start_time = time.time()
    ncm_classifier = NCMClassifier(train_stats).to("cuda")
    build_time = time.time() - start_time
    
    # 测量预测时间
    ncm_classifier.eval()
    sample_features = test_features[:1000].to("cuda")
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(10):
            _ = ncm_classifier(sample_features)
    predict_time = (time.time() - start_time) / 10
    
    results['NCM'] = {'build_time': build_time, 'predict_time': predict_time}
    
    print(f"NCM构建时间: {build_time:.4f}s")
    print(f"NCM预测时间: {predict_time:.4f}s")
    
    # 绘制效率对比图
    plot_experiment4_efficiency(results, save_path)
    
    return results

def plot_experiment4_efficiency(results, save_path=None):
    """绘制实验4效率对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(3.5*2, 2.5))  # IEEE单栏尺寸适配
    
    classifiers = list(results.keys())
    build_times = [results[c]['build_time'] for c in classifiers]
    predict_times = [results[c]['predict_time'] for c in classifiers]
    
    # 构建时间对比
    ax = axes[0]
    colors = ['blue', 'green', 'red', 'cyan']
    bars = ax.bar(classifiers, build_times, color=colors[:len(classifiers)], alpha=0.7, width=0.6)
    
    for bar, time_val in zip(bars, build_times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
               f'{time_val:.4f}s', ha='center', va='bottom', fontsize=7)
    
    ax.set_ylabel('Build Time (s)', fontsize=9)
    ax.set_title('Classifier Build Time', fontsize=9)
    ax.set_yscale('log')
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    ax.tick_params(axis='both', which='major', labelsize=7)
    
    # 预测时间对比
    ax = axes[1]
    bars = ax.bar(classifiers, predict_times, color=colors[:len(classifiers)], alpha=0.7, width=0.6)
    
    for bar, time_val in zip(bars, predict_times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.0001,
               f'{time_val:.4f}s', ha='center', va='bottom', fontsize=7)
    
    ax.set_ylabel('Predict Time (s)', fontsize=9)
    ax.set_title('Classifier Predict Time', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    ax.tick_params(axis='both', which='major', labelsize=7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', fonttype=42)
        print(f"效率对比图已保存到: {save_path}")
    
    plt.show()


# %% 运行所有实验
def run_all_ablation_experiments(output_dir="实验结果保存/分类器消融实验"):
    """运行所有消融实验"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("开始运行所有消融实验...")
    
    # 实验1: 性能曲面等高线图
    print("\n" + "="*60)
    print("运行实验1: 性能曲面等高线图")
    print("="*60)
    alpha1_values, alpha2_values, accuracy_matrix = grid_search_alpha1_alpha2(
        alpha1_min=0, alpha1_max=5.0, alpha2_min=0, alpha2_max=5.0,
        alpha1_points=11, alpha2_points=11  # 为了快速演示，使用较少的点
    )
    best_alpha1, best_alpha2, best_acc = plot_alpha1_alpha2_contour(
        alpha1_values, alpha2_values, accuracy_matrix,
        save_path=f"{output_dir}/exp1_contour.png"
    )
    
    # 实验2: 参数敏感性分析
    print("\n" + "="*60)
    print("运行实验2: 参数敏感性分析")
    print("="*60)
    exp2_results = run_experiment2_sensitivity_analysis(
        alpha1_range=(0.0, 5.0), alpha2_range=(0.0, 5.0),
        alpha_sum=3.0, fixed_points=11,
        save_path=f"{output_dir}/exp2_sensitivity.png"
    )
    
    # 实验3: 分类器对比
    print("\n" + "="*60)
    print("运行实验3: 分类器对比")
    print("="*60)
    exp3_results = run_experiment3_classifier_comparison(
        sgd_epochs=5, sgd_lr=0.01,
        save_path=f"{output_dir}/exp3_comparison.png"
    )
    
    # 实验4: 效率对比
    print("\n" + "="*60)
    print("运行实验4: 效率对比")
    print("="*60)
    exp4_results = run_experiment4_efficiency_comparison(
        save_path=f"{output_dir}/exp4_efficiency.png"
    )
    
    # 保存所有结果
    all_results = {
        'exp1': {
            'best_alpha1': best_alpha1,
            'best_alpha2': best_alpha2,
            'best_accuracy': best_acc,
            'alpha1_values': alpha1_values.tolist(),
            'alpha2_values': alpha2_values.tolist(),
            'accuracy_matrix': accuracy_matrix.tolist()
        },
        'exp2': exp2_results,
        'exp3': exp3_results,
        'exp4': exp4_results
    }
    
    import json
    with open(f"{output_dir}/all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n所有实验完成！结果已保存到 {output_dir}/")
    print(f"包含文件:")
    print(f"  - exp1_contour.png: 实验1性能曲面等高线图")
    print(f"  - exp2_sensitivity.png: 实验2参数敏感性分析图")
    print(f"  - exp3_comparison.png: 实验3分类器对比图")
    print(f"  - exp4_efficiency.png: 实验4效率对比图")
    print(f"  - all_results.json: 所有实验结果数据")

# %% 快速测试所有实验（使用较少的数据点）
def quick_test_all_experiments():
    """快速测试所有实验（用于验证代码正确性）"""
    print("开始快速测试所有实验...")
    
    # 使用较少的数据点进行快速测试
    output_dir = "./ablation_results_quick"
    
    # 实验1: 5x5网格
    print("\n快速测试实验1: 5x5网格...")
    alpha1_values, alpha2_values, accuracy_matrix = grid_search_alpha1_alpha2(
        alpha1_min=0, alpha1_max=2.0, alpha2_min=0, alpha2_max=2.0,
        alpha1_points=5, alpha2_points=5
    )
    plot_alpha1_alpha2_contour(alpha1_values, alpha2_values, accuracy_matrix)
    
    # 实验2: 5个点
    print("\n快速测试实验2: 参数敏感性...")
    run_experiment2_sensitivity_analysis(
        alpha1_range=(0.0, 2.0), alpha2_range=(0.0, 2.0),
        alpha_sum=2.0, fixed_points=5
    )
    
    # 实验3: 基础对比
    print("\n快速测试实验3: 分类器对比...")
    run_experiment3_classifier_comparison(sgd_epochs=2, sgd_lr=0.01)
    
    # 实验4: 效率测试
    print("\n快速测试实验4: 效率对比...")
    run_experiment4_efficiency_comparison()
    
    print("\n快速测试完成！")

# %% 主执行区域
if __name__ == "__main__":
    # 可以选择运行完整实验或快速测试
    # run_all_ablation_experiments()  # 完整实验
    quick_test_all_experiments()  # 快速测试
