from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
from torchvision import datasets, transforms

from models.base import BaseLearner
from classifier.classifier_builder import ClassifierReconstructor
from compensator.distribution_compensator import DistributionCompensator
from models.distillator import Distiller
from utils.inc_net import BaseNet
from lora import compute_covariances
import math

class GPUMemoryMonitor:
    """GPU显存监控器，用于跟踪和记录显存使用情况"""
    
    def __init__(self, enabled=True):
        self.enabled = enabled and torch.cuda.is_available()
        self.memory_history = []
        
    def log_memory(self, stage: str):
        """记录当前显存使用情况"""
        if not self.enabled:
            return
            
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
        
        self.memory_history.append({
            'stage': stage,
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'max_allocated_gb': max_allocated
        })
        
        logging.info(f"[GPU Memory] {stage}: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, Max={max_allocated:.2f}GB")
        
    def get_peak_memory(self) -> float:
        """获取峰值显存使用量（GB）"""
        if not self.enabled or not self.memory_history:
            return 0.0
        return max(entry['max_allocated_gb'] for entry in self.memory_history)
        
    def clear_history(self):
        """清空历史记录"""
        self.memory_history.clear()

class EMASmooth:
    def __init__(self, alpha=0.95):
        self.alpha = alpha
        self.value = None
    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * self.value + (1 - self.alpha) * new_value
        return self.value
    def get(self):
        return self.value if self.value is not None else 0.0

def symmetric_cross_entropy_loss(logits, targets, sce_a=0.5, sce_b=0.5):
    pred = F.softmax(logits, dim=1)
    pred = torch.clamp(pred, min=1e-7, max=1.0) 
    label_one_hot = torch.nn.functional.one_hot(targets, pred.size(1)).float().to(pred.device)
    label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
    ce_loss = -torch.sum(label_one_hot * torch.log(pred), dim=1).mean()
    rce_loss = -torch.sum(pred * torch.log(label_one_hot), dim=1).mean()
    total_loss = sce_a * ce_loss + sce_b * rce_loss
    return total_loss

@dataclass
class Timing:
    train: float = 0.0
    drift: float = 0.0
    total: float = 0.0

criterion = lambda logits, targets: symmetric_cross_entropy_loss(logits, targets, sce_a=0.5, sce_b=0.5)

class SubspaceLoRA(BaseLearner):
    def __init__(self, args: Dict[str, Any]) -> None:
        super().__init__(args)

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = BaseNet(args, pretrained=True).to(self._device)
        self.args = args

        self._timings: Timing = Timing()
        self.time_history: List[Dict[str, float]] = []

        self.batch_size: int = args["batch_size"]
        self.iterations: int = args["iterations"]
        self.warmup_steps: int = int(args["warmup_ratio"] * self.iterations)

        self.lrate: float = args["lrate"]
        self.weight_decay: float = args["weight_decay"]
        self.optimizer_type: str = args["optimizer"]
        kd_type = args["kd_type"]

        self.use_kd: bool = args["gamma_kd"] > 0.0
        self.gamma_kd: float = args["gamma_kd"] if self.use_kd else 1.0

        self.use_aux_for_kd: bool = args["use_aux_for_kd"]
        self.update_teacher_each_task: bool = args["update_teacher_each_task"]
        
        self.aux_loader = None
        self.aux_iter = None

        self.covariances: Dict[str, torch.Tensor] | None = None
        self.drift_compensator = DistributionCompensator(
            auxiliary_data_size=args["auxiliary_data_size"],
            compensator_types=args['compensator_types'])
        
        self.classifier_reconstructor = ClassifierReconstructor(
            device=self._device,
            lda_reg_alpha=args['lda_reg_alpha'],
            qda_reg_alpha1=args['qda_reg_alpha1'],
            qda_reg_alpha2=args['qda_reg_alpha2'],
            qda_reg_alpha3=args['qda_reg_alpha3'])
        
        # 获取特征维度，处理不同模型类型
        feat_dim = getattr(self.network.vit, 'feature_dim', 768)
        self.distillator = Distiller(
            kd_type=kd_type,
            gamma_kd=args["gamma_kd"],
            update_teacher_each_task=args["update_teacher_each_task"],
            device=self._device,
            feat_dim=feat_dim,
            transform=args["distillation_transform"])
        
        self.get_aux_loader(self.args)
        self.drift_compensator.set_auxiliary_loader(self.aux_loader)

        self.teacher_network = None
        self.prev_network = None
        self.seed: int = args["seed"]
        self.task_count: int = 0
        self.current_task_id = 0
        
        logging.info(f"Optimizer instantiated: lrate={self.lrate}, wd={self.weight_decay}, optimizer={self.optimizer_type}")

        self.loss_smoother = EMASmooth(alpha=0.98)
        self.kd_smoother = EMASmooth(alpha=0.98)
        self.acc_smoother = EMASmooth(alpha=0.98)
        
        # 初始化GPU显存监控器
        self.gpu_monitor = GPUMemoryMonitor(enabled=True)

    def save_checkpoint(self, prefix: str) -> None:
        """Save trainable parameters after the current task."""
        param_dict = {n: p.detach().cpu() for n, p in self.network.named_parameters() if p.requires_grad}
        payload = {"task": self.current_task_id, "model_state_dict": param_dict}
        path = f"{prefix}/after_task_{self.current_task_id}.pth"
        torch.save(payload, path)
        logging.info(f"Checkpoint saved to: {path}")

    def load_checkpoint(self, prefix: str) -> None:
        """Save trainable parameters after the current task."""
        path = f"{prefix}/after_task_{self.current_task_id}.pth"
        param_dict = torch.load(path)['model_state_dict']
        self.network.load_state_dict(param_dict, strict=False)
        logging.info(f"Checkpoint loaded from: {path}")

    def handle_drift_compensation(self) -> None:
        """Handle the drift compensation and update classifiers."""
        drift_start = time.time()
        
        # 记录显存使用情况
        self.gpu_monitor.log_memory(f"before_drift_compensation_task_{self.current_task_id}")
        
        self.drift_compensator.build_all_variants(
            self.current_task_id,
            self.prev_network.vit,
            self.network.vit,
            self.train_loader_test_mode)

        self._timings.drift = time.time() - drift_start
        
        # 记录显存使用情况
        self.gpu_monitor.log_memory(f"after_drift_compensation_task_{self.current_task_id}")

    def refine_classifiers(self):
        self.fc_dict = self.classifier_reconstructor.build_classifiers(
            self.drift_compensator.variants
        )

    def after_task(self) -> None:
        """Update class counters after finishing a task."""
        self._known_classes = self._total_classes
        self.update_projection_matrices()
        self.task_count += 1

    def _sample_test_set_by_class(self, test_set, data_manager, task_id, num_samples_per_class):
        """
        对test_set进行类别平衡采样
        
        Args:
            test_set: 原始测试数据集
            data_manager: 数据管理器
            task_id: 当前任务ID
            num_samples_per_class: 每个类别要采样的样本数量
            
        Returns:
            采样后的数据集
        """
        # 获取当前任务的类别范围
        task_start = data_manager.global_label_offset[task_id]
        task_end = task_start + data_manager.get_task_size(task_id)
        
        # 收集每个类别的样本索引
        class_indices = {}
        for idx in range(len(test_set)):
            _, label = test_set[idx]
            # 只考虑当前任务的类别
            if task_start <= label < task_end:
                if label not in class_indices:
                    class_indices[label] = []
                class_indices[label].append(idx)
        
        # 对每个类别进行采样
        sampled_indices = []
        for class_id, indices in class_indices.items():
            if len(indices) > 0:
                # 如果样本数量不足，则全部采样；否则随机采样指定数量
                if len(indices) <= num_samples_per_class:
                    sampled_indices.extend(indices)
                    logging.info(f"类别 {class_id}: 采样 {len(indices)} 个样本（不足 {num_samples_per_class} 个）")
                else:
                    np.random.shuffle(indices)
                    sampled_indices.extend(indices[:num_samples_per_class])
                    logging.info(f"类别 {class_id}: 采样 {num_samples_per_class} 个样本（总共 {len(indices)} 个）")
        
        logging.info(f"采样前测试集大小: {len(test_set)}, 采样后: {len(sampled_indices)}")
        
        # 创建采样的子集
        return Subset(test_set, sampled_indices)

    def incremental_train(self, data_manager) -> None:
        start_time = time.time()
        task_id = self.current_task_id
        task_size = data_manager.get_task_size(task_id)

        self.distillator.update_teacher(self.network)
        self.prev_network = copy.deepcopy(self.network).cpu()

        if hasattr(self.prev_network.vit, 'finalize_without_lora'):
            self.prev_network.vit.finalize_without_lora()
        
        if hasattr(self.network.vit, 'merge_lora_weights'):
            self.network.vit.merge_lora_weights()

        self._total_classes = self._known_classes + task_size
        self.current_task_id += 1
        self.topk = min(self._total_classes, 5)

        train_set = data_manager.get_subset(
            task=task_id, source="train", cumulative=False, mode="train")
        test_set = data_manager.get_subset(
            task=task_id, source="test", cumulative=True, mode="test")
        train_set_test_mode = data_manager.get_subset(
            task=task_id, source="train", cumulative=False, mode="test")

        self.train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True,
                                    num_workers=4, pin_memory=True, persistent_workers=False)
        
        self.test_loader = DataLoader(test_set, batch_size=self.batch_size * 4, shuffle=False,
                                    num_workers=4, pin_memory=True, persistent_workers=False)

        dataset_size = len(train_set_test_mode)
        max_samples = getattr(self, 'max_train_test_samples', 5000)
        sampler = None

        # if dataset_size > max_samples:
        #     indices = torch.randperm(dataset_size)[:max_samples].tolist()
        #     sampler = SubsetRandomSampler(indices)
        #     print(f"⚠️ Dataset too large ({dataset_size}), sampling {max_samples} examples for test-mode training set.")

        self.train_loader_test_mode = DataLoader(
            train_set_test_mode,
            batch_size=self.batch_size,
            shuffle=False if sampler else True,
            sampler=sampler,
            num_workers=3,
            pin_memory=True,
            persistent_workers=True)

        # === 初始化 DriftCompensator 所需的 loader，但不用于 KD ===
        self.network.update_fc(task_size)
        self.network.fc.to(self._device)

        try:
            logging.info(
                "System training on classes %d-%d (%s)",
                self._known_classes,
                self._total_classes,
                data_manager.dataset_name.lower())
        except:
            logging.info(
                "System training on classes %d-%d (%s)",
                self._known_classes,
                self._total_classes,
                data_manager.dataset_names[task_id].lower())

        if self.args['eval_only']:
            self.load_checkpoint(self.args["log_path"])
        else:
            self.print_parameter_statistics(task_id)
            self.system_training()
            # self.save_checkpoint(self.args["log_path"])

        self.handle_drift_compensation()
        self._timings.total = time.time() - start_time

        logging.info(
            "Task %d finished total: %.2f s | train: %.2f s | drift: %.2f s",
            self.current_task_id,
            self._timings.total,
            self._timings.train,
            self._timings.drift)

        
    def make_optimizer(
        self,
        lora_params: List[torch.nn.Parameter],
        fc_params: List[torch.nn.Parameter]) -> optim.Optimizer:

        """Create optimizer according to ``self.optimizer_type``."""
        distill_params = filter(lambda p: p.requires_grad, self.distillator.parameters())

        param_groups = [
            {"params": lora_params, "lr": self.lrate, "weight_decay": self.weight_decay},
            {"params": fc_params, "lr": 1e-3 if self.optimizer_type == "adamw" else 5e-3, "weight_decay": self.weight_decay},
            {"params": distill_params, "lr": 10 * self.lrate, "weight_decay": 3e-5},
        ]

        if self.optimizer_type == "sgd":
            optimizer = optim.SGD(param_groups, momentum=0.9)
        elif self.optimizer_type == "adamw":
            optimizer = optim.AdamW(param_groups)
        elif self.optimizer_type == "rmsprop":
            optimizer = optim.RMSprop(param_groups)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_type}")

        if self.warmup_steps > 0:
            def lora_lr_lambda(step):
                if step < self.warmup_steps:
                    return step / max(1, self.warmup_steps)
                else:
                    progress = (step - self.warmup_steps) / max(1, self.iterations - self.warmup_steps)
                    initial_lr = self.lrate
                    eta_min = getattr(self, 'eta_min', self.lrate * 0.3)
                    lr_ratio = eta_min / initial_lr
                    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                    return lr_ratio + cosine_decay * (1.0 - lr_ratio)
            
            def const_lr_lambda(step):
                return 1.0
            
            lr_lambdas = [lora_lr_lambda, const_lr_lambda, const_lr_lambda]
            # lr_lambdas = [const_lr_lambda, const_lr_lambda, const_lr_lambda]
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambdas, last_epoch=-1)
        return optimizer, scheduler


    def system_training(self) -> None:
        """Train ViT + new classifier head for ``self.epochs`` epochs."""
        fc_params = self.network.fc.parameters()
        # 处理不同模型类型的参数获取
        if hasattr(self.network.vit, 'get_param_groups'):
            lora_params = self.network.vit.get_param_groups()
        else:
            lora_params = [p for p in self.network.vit.parameters() if p.requires_grad]
        optimizer, scheduler = self.make_optimizer(
            lora_params,
            fc_params)
        
        start = time.time()
        self.network.train()
        
        step = 0
        done = False
        while True:
            for batch in self.train_loader:
                inputs, targets = batch[0], batch[1]
                loss, n_corr, kd_term = self.process_batch(inputs, targets, optimizer)

                batch_acc = n_corr / inputs.size(0)

                # 使用 EMASmooth 平滑
                smoothed_loss = self.loss_smoother.update(loss)
                smoothed_kd = self.kd_smoother.update(kd_term)
                smoothed_acc = self.acc_smoother.update(batch_acc)

                if (step + 1) % 50 == 0:
                    logging.info(
                        'step: %d, loss: %.4f, kd_loss: %.4f, acc: %.4f',
                        step, smoothed_loss, smoothed_kd, smoothed_acc)

                scheduler.step()
                step += 1

                if step == self.iterations:
                    done = True
                    break
            if done:
                break

        self._timings.train = time.time() - start

    def process_batch(self, inputs, targets, optimizer):
        inputs, targets = inputs.to(self._device), targets.to(self._device)
        
        # 调试：记录第一个batch的标签信息
        if not hasattr(self, '_debug_logged') and self.current_task_id == 0:
            logging.info(f"[TRAIN] Task {self.current_task_id} first batch debug:")
            logging.info(f"[TRAIN]   Targets range: {targets.min().item()}-{targets.max().item()}")
            logging.info(f"[TRAIN]   Targets unique (first 10): {targets.unique()[:10].tolist()}")
            logging.info(f"[TRAIN]   _known_classes: {self._known_classes}")
            logging.info(f"[TRAIN]   _total_classes: {self._total_classes}")
            logging.info(f"[TRAIN]   cross_domain: {self.args.get('cross_domain', False)}")
            self._debug_logged = True
        
        feats = self.network.forward_features(inputs)
 
        kd_term = self.distillator(inputs, feats)

        # === 分类损失 ===
        logits = self.network.fc(feats)
        
        new_targets_rel = torch.where(
            targets - self._known_classes >= 0,
            targets - self._known_classes, -100)
        new_logits = logits[:, self._known_classes:]
        sce = criterion(new_logits, new_targets_rel)

        loss = sce + kd_term

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            n_correct = (pred == targets).sum().item()

        kd_raw = kd_term.item() / self.gamma_kd

        return loss.item(), n_correct, kd_raw



    @staticmethod
    def norm_loss(t_feat: torch.Tensor, s_feat: torch.Tensor) -> torch.Tensor:
        """MSE between L2‑norms of teacher / student feature vectors."""
        t_norm = t_feat.norm(p=2, dim=1)
        s_norm = s_feat.norm(p=2, dim=1)
        return F.mse_loss(t_norm, s_norm)


    def evaluate(
        self,
        loader: DataLoader,
        fc_dict):
        """
        Evaluate model on test data.
        For cross-domain scenarios, compute accuracy per dataset and average across datasets.
        For regular scenarios, compute overall accuracy.
        """
        self.network.eval()
        
        # Check if this is cross-domain scenario
        is_cross_domain = self.args.get('cross_domain', False)
        
        if is_cross_domain:
            # For cross-domain: compute accuracy per dataset (task)
            result = self._evaluate_cross_domain(loader, fc_dict)
        else:
            # For regular scenarios: compute overall accuracy
            result = self._evaluate_regular(loader, fc_dict)
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return result
    
    def _evaluate_regular(self, loader: DataLoader, fc_dict):
        """Regular evaluation: compute overall accuracy across all samples"""
        self.network.eval()
        total = 0
        corrects = {}
        for name, fc in fc_dict.items():
            corrects[name] = 0
            fc.to(self._device)

        with torch.no_grad():
            for batch in loader:
                inputs = batch[0].to(self._device)
                targets = batch[1]
                
                feats = self.network.forward_features(inputs)
                for name, fc in fc_dict.items():
                    preds = fc(feats).argmax(dim=1).cpu()
                    corrects[name] += (preds == targets).sum().item()
                total += targets.size(0)
        
        for name, correct in corrects.items():
            corrects[name] = float(np.around(100 * correct / total, 2))
        return corrects
    
    def _evaluate_cross_domain(self, loader: DataLoader, fc_dict):
        """
        Cross-domain evaluation: compute accuracy per dataset and average across datasets.
        This ensures each dataset contributes equally to the final metric, regardless of sample size.
        Also computes class-wise average accuracy for cross-domain scenarios, both per-task and overall.
        """
        self.network.eval()
        
        # Get task information from data manager
        data_manager = self.data_manager
        num_tasks = data_manager.nb_tasks
        
        # Initialize per-task accuracy tracking
        task_corrects = {name: [0] * num_tasks for name in fc_dict.keys()}
        task_totals = [0] * num_tasks
        
        # 初始化每个类别的正确数和总数（跨域场景下按任务组织）
        per_class_corrects = {}
        per_class_totals = {}
        for name in fc_dict.keys():
            per_class_corrects[name] = {}
            per_class_totals[name] = {}

        for name, fc in fc_dict.items():
            fc.to(self._device)
        
        with torch.no_grad():
            for batch in loader:
                inputs = batch[0].to(self._device)
                targets = batch[1]
                
                # Only evaluate learned tasks (0 to current_task_id-1)
                for task_id in range(self.current_task_id):
                    task_start = data_manager.global_label_offset[task_id]
                    task_end = task_start + data_manager.get_task_size(task_id)
                    
                    # Create mask for samples belonging to this task
                    task_mask = (targets >= task_start) & (targets < task_end)
                    task_samples = task_mask.sum().item()
                    
                    if task_samples > 0:
                        task_inputs = inputs[task_mask]
                        task_targets = targets[task_mask]
                        
                        feats = self.network.forward_features(task_inputs)
                        for name, fc in fc_dict.items():
                            preds = fc(feats).argmax(dim=1).cpu()
                            task_corrects[name][task_id] += (preds == task_targets).sum().item()
                            
                            # 计算每个类别的准确度（相对于任务内的类别）
                            for class_id in range(task_start, task_end):
                                class_mask = (task_targets == class_id)
                                class_total = class_mask.sum().item()
                                if class_total > 0:
                                    class_correct = (preds[class_mask] == class_id).sum().item()
                                    # 使用全局类别ID作为键
                                    per_class_corrects[name][class_id] = per_class_corrects[name].get(class_id, 0) + class_correct
                                    per_class_totals[name][class_id] = per_class_totals[name].get(class_id, 0) + class_total
                        task_totals[task_id] += task_samples
        
        # Compute per-task accuracies for learned tasks only
        task_accuracies = {}
        for name in fc_dict.keys():
            task_accuracies[name] = []
            for task_id in range(self.current_task_id):  # Only compute for learned tasks
                if task_totals[task_id] > 0:
                    acc = 100 * task_corrects[name][task_id] / task_totals[task_id]
                    task_accuracies[name].append(float(np.around(acc, 2)))
        
        # 计算每个任务的class-wise平均准确度（新增功能）
        per_task_class_wise_accuracies = {}
        for name in fc_dict.keys():
            per_task_class_wise_accuracies[name] = []
            for task_id in range(self.current_task_id):
                task_start = data_manager.global_label_offset[task_id]
                task_end = task_start + data_manager.get_task_size(task_id)
                
                # 计算当前任务的class-wise准确度
                task_class_accuracies = []
                for class_id in range(task_start, task_end):
                    if class_id in per_class_totals[name] and per_class_totals[name][class_id] > 0:
                        acc = 100 * per_class_corrects[name][class_id] / per_class_totals[name][class_id]
                        task_class_accuracies.append(float(np.around(acc, 2)))
                
                # 计算当前任务的class-wise平均准确度
                if task_class_accuracies:
                    task_avg_class_acc = np.mean(task_class_accuracies)
                    per_task_class_wise_accuracies[name].append(float(np.around(task_avg_class_acc, 2)))
                else:
                    per_task_class_wise_accuracies[name].append(0.0)
        
        # 计算总体class-wise平均准确度（跨域场景）
        class_wise_accuracies = {}  # 存储每个变体的class-wise平均准确度
        for name in fc_dict.keys():
            class_accuracies = []
            for task_id in range(self.current_task_id):
                task_start = data_manager.global_label_offset[task_id]
                task_end = task_start + data_manager.get_task_size(task_id)
                
                for class_id in range(task_start, task_end):
                    if class_id in per_class_totals[name] and per_class_totals[name][class_id] > 0:
                        acc = 100 * per_class_corrects[name][class_id] / per_class_totals[name][class_id]
                        class_accuracies.append(float(np.around(acc, 2)))
            
            # 计算平均类别准确度
            if class_accuracies:
                avg_class_acc = np.mean(class_accuracies)
                class_wise_accuracies[name] = float(np.around(avg_class_acc, 2))
            else:
                class_wise_accuracies[name] = 0.0
        
        # Return average across learned tasks only
        results = {}
        for name in fc_dict.keys():
            if task_accuracies[name]:
                results[name] = float(np.around(np.mean(task_accuracies[name]), 2))
            else:
                results[name] = 0.0
        
        # 将class-wise准确度存储在单独的字典中，不混入主结果
        # 这些将在analyze_task_results中单独处理
        results["_class_wise_data"] = {
            "class_wise_accuracies": class_wise_accuracies,
            "per_task_class_wise_accuracies": per_task_class_wise_accuracies
        }
        
        return results
    
    def eval_task(self):
        num_samples_for_eval = self.args.get('num_samples_per_task_for_evaluation', 0)
        if num_samples_for_eval > 0:
            logging.info(f"🔍 Using sampled evaluation with {num_samples_for_eval} samples per task")
            
        # 记录评估前的显存使用情况
        self.gpu_monitor.log_memory(f"before_evaluation_task_{self.current_task_id}")
        
        # 记录评估开始时间
        eval_start_time = time.time()
        
        results = self.evaluate(
            self.test_loader,
            fc_dict=self.fc_dict)
        
        # 记录评估结束时间
        eval_end_time = time.time()
        eval_elapsed_time = eval_end_time - eval_start_time
        logging.info(f"[Time] Evaluation for task {self.current_task_id}: {eval_elapsed_time:.4f}s")
            
        # 记录评估后的显存使用情况
        self.gpu_monitor.log_memory(f"after_evaluation_task_{self.current_task_id}")
        
        if not hasattr(self, "all_task_results"):
            self.all_task_results: Dict[int, Dict[str, float]] = {}
        self.all_task_results[self.current_task_id] = results
        
        # 增量学习每个任务时记录每个数据集的准确度（新增功能）
        is_cross_domain = self.args.get('cross_domain', False)
        
        if is_cross_domain and self.current_task_id > 0:
            data_manager = self.data_manager
            logging.info(f"📊 任务 {self.current_task_id} 完成后的各数据集准确度:")
            
            # 记录每个变体的准确度
            for variant, accuracy in results.items():
                if isinstance(accuracy, (int, float)):
                    logging.info(f"  {variant:<20}: {accuracy:.2f}%")
        
        # 打印峰值显存使用情况
        peak_memory = self.gpu_monitor.get_peak_memory()
        logging.info(f"[GPU Memory] Peak memory usage up to task {self.current_task_id}: {peak_memory:.2f}GB")
        
        return results

    def update_projection_matrices(self):
        if hasattr(self.network.vit, 'use_projection') and self.network.vit.use_projection:
            if self.current_task_id >= 0:
                new_covs = compute_covariances(self.network.vit, self.train_loader_test_mode)
                # 将新的协方差矩阵移到CPU以节省GPU显存
                new_covs_cpu = {k: v.cpu() for k, v in new_covs.items()}
                
                if self.covariances is None:
                    self.covariances = new_covs_cpu
                else:
                    for k in self.covariances:
                        self.covariances[k] = 0.9 * self.covariances[k] + new_covs_cpu[k] + 1e-7 * torch.eye(self.covariances[k].size(0))
                # 只在需要时再将covariances移到GPU
                covariances_gpu = {k: v.to(self._device) for k, v in self.covariances.items()}
                self.network.update_projection_matrices(covariances_gpu)
                # 清理GPU上的临时covariances
                del covariances_gpu
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def loop(self, data_manager) -> Dict[str, Any]:
        self.data_manager = data_manager
        final_analysis: Dict[str, Any] | None = None
        for _ in range(data_manager.nb_tasks):
            self.incremental_train(data_manager)
            self.refine_classifiers()
            logging.info(f"Evaluating after task {self.current_task_id}...")
            self.eval_task()
            # 传入dataset_names参数用于cross-domain任务准确度显示
            dataset_names = getattr(self.data_manager, 'dataset_names', None)
            final_analysis = self.analyze_task_results(self.all_task_results, dataset_names)
            self.after_task()

        if final_analysis is None:
            final_analysis = self.analyze_task_results(self.all_task_results, dataset_names)

        combined_results: Dict[str, Any] = dict(final_analysis)
        combined_results["per_task_results"] = dict(self.all_task_results)
        return combined_results

    def analyze_task_results(self, all_task_results: Dict[int, Dict[str, Any]], dataset_names) -> Dict[str, Any]:
        if not all_task_results:
            logging.info("📊 Task evaluation results are empty. Nothing to analyze.")
            return {
                "last_task_id": None,
                "last_task_accuracies": {},
                "average_accuracies": {},
                "class_wise_accuracies": {},
                "average_class_wise_accuracies": {}}

        # Sort task IDs and get the last one
        task_ids = sorted(all_task_results.keys())
        last_task_id = task_ids[-1]

        variant_names = set()
        for task_dict in all_task_results.values():
            for key, value in task_dict.items():
                # 只添加数值类型的键，排除字典类型的数据
                if isinstance(value, (int, float)):
                    variant_names.add(key)
        variant_names = sorted(variant_names)

        # Compute final-task accuracies
        last_task_accuracies = {
            variant: all_task_results[last_task_id].get(variant, 0.0)
            for variant in variant_names
        }

        # Compute average accuracies across all tasks
        average_accuracies = {}
        for variant in variant_names:
            # 只获取数值类型的准确率，排除字典类型的数据（如class_wise_accuracies）
            accs = []
            for task_id in task_ids:
                task_value = all_task_results[task_id].get(variant, 0.0)
                if isinstance(task_value, (int, float)):
                    accs.append(task_value)
            
            if accs:
                average_accuracies[variant] = float(np.mean(accs))
            else:
                average_accuracies[variant] = 0.0

        # Extract class-wise accuracies from the last task (only for cross-domain)
        class_wise_accuracies = {}
        is_cross_domain = self.args.get('cross_domain', False)
        
        if is_cross_domain and '_class_wise_data' in all_task_results[last_task_id] and 'class_wise_accuracies' in all_task_results[last_task_id]['_class_wise_data']:
            class_wise_accuracies = all_task_results[last_task_id]['_class_wise_data']['class_wise_accuracies']
            logging.info("   ── Class-wise Average Accuracy (%) ──")
            for variant in variant_names:
                if variant in class_wise_accuracies:
                    logging.info(f"      {variant:<20} : {class_wise_accuracies[variant]:.2f}%")
        else:
            logging.info("   ── Class-wise Average Accuracy (%) ──")
            logging.info("      Not available (within-domain scenario)")

        # Compute average class-wise accuracies across all tasks (新增功能)
        average_class_wise_accuracies = {}
        if is_cross_domain:
            # 收集所有任务的class-wise准确度
            for variant in variant_names:
                all_task_class_wise_accs = []
                for task_id in task_ids:
                    if '_class_wise_data' in all_task_results[task_id] and 'per_task_class_wise_accuracies' in all_task_results[task_id]['_class_wise_data']:
                        per_task_accs = all_task_results[task_id]['_class_wise_data']['per_task_class_wise_accuracies']
                        if variant in per_task_accs and per_task_accs[variant]:
                            # 获取当前任务的class-wise平均准确度
                            task_class_wise_acc = per_task_accs[variant][-1]  # 最后一个值是当前任务的class-wise平均准确度
                            all_task_class_wise_accs.append(task_class_wise_acc)
                
                # 计算所有任务的class-wise平均准确度的平均值
                if all_task_class_wise_accs:
                    average_class_wise_accuracies[variant] = float(np.mean(all_task_class_wise_accs))
                else:
                    average_class_wise_accuracies[variant] = 0.0
            
            if average_class_wise_accuracies:
                logging.info("   ── Average Class-wise Accuracy Across Tasks (%) ──")
                for variant in variant_names:
                    if variant in average_class_wise_accuracies:
                        logging.info(f"      {variant:<20} : {average_class_wise_accuracies[variant]:.2f}%")

        # === Cross-domain per-task accuracies (learned tasks only) ===
        if is_cross_domain and task_ids:
            logging.info("   ── Cross-domain per-task accuracies (learned tasks only) ──")
            for variant in variant_names:
                # 从所有任务结果中收集该变体的每个任务准确度
                per_task_accuracies = []
                for task_id in task_ids:
                    task_acc = 0.0
                    task_count = 0
                    for tid in task_ids:
                        if tid <= task_id:  # 只计算到当前任务为止的准确度
                            task_result = all_task_results[tid]
                            if variant in task_result and isinstance(task_result[variant], (int, float)):
                                task_acc += task_result[variant]
                                task_count += 1
                    if task_count > 0:
                        avg_acc = task_acc / task_count
                        per_task_accuracies.append(avg_acc)
                    else:
                        per_task_accuracies.append(0.0)
                
                if per_task_accuracies:
                    logging.info(f"   Variant: {variant}")
                    for i, task_id in enumerate(task_ids):
                        dataset_name = dataset_names[task_id - 1] if dataset_names and task_id - 1 < len(dataset_names) else f"Task {task_id}"
                        if i < len(per_task_accuracies):
                            logging.info(f"     {dataset_name} (task {task_id}): {per_task_accuracies[i]:.2f}%")
                    
        # === Log Results in Structured Format ===
        logging.info(" Incremental Learning Evaluation Analysis:")
        logging.info(f"   Last Task ID: {last_task_id}")

        logging.info("   ── Average Accuracy Across Tasks (%) ──")
        for variant in variant_names:
            logging.info(f"      {variant:<20} : {average_accuracies[variant]:.2f}%")

        logging.info("  ── Final Task Accuracy (%) ──")
        for variant in variant_names:
            logging.info(f"      {variant:<20} : {last_task_accuracies[variant]:.2f}%")

        # Optional: Identify best variants and log summary
        best_last = max(last_task_accuracies.items(), key=lambda x: x[1])[0] if last_task_accuracies else None
        best_avg = max(average_accuracies.items(), key=lambda x: x[1])[0] if average_accuracies else None

        if best_last and best_avg:
            if best_last == best_avg:
                summary = f" Variant '{best_last}' is best in both final task and average performance."
            else:
                summary = f" Best in Final Task: '{best_last}' | Best Average: '{best_avg}'"
        else:
            summary = " No valid variants found for comparison."

        logging.info("   ── Summary ──")
        logging.info(f"      {summary}")

        # Return structured data for further use
        return {
            "last_task_id": last_task_id,
            "last_task_accuracies": last_task_accuracies,
            "average_accuracies": average_accuracies,
            "class_wise_accuracies": class_wise_accuracies,
            "average_class_wise_accuracies": average_class_wise_accuracies}
    
    def get_aux_loader(self, args):
        aux_dataset_type = args.get('aux_dataset', 'flickr8k')
        num_samples = int(args.get('auxiliary_data_size', 5000))

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

        if aux_dataset_type == 'imagenet':
            dataset = datasets.ImageFolder(args['auxiliary_data_path'] + '/ImageNet-2012/train', transform=transform)
        elif aux_dataset_type == 'cifar10':
            dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        elif aux_dataset_type == 'svhn':
            dataset = datasets.SVHN(root='./data', split='train', download=True, transform=transform)
        elif aux_dataset_type == 'flickr8k':
            dataset = datasets.ImageFolder(args['auxiliary_data_path'] + '/flickr8k', transform=transform)
        else:
            raise ValueError(f"不支持的 aux_dataset_type: {aux_dataset_type}")

        indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
        train_subset = Subset(dataset, indices)

        self.aux_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        self.aux_trainset = train_subset
        return self.aux_loader

    def count_trainable_parameters(self) -> Dict[str, int]:
        """统计各部分的训练参数数量"""
        param_counts = {}
        
        # 获取模型参数，处理不同模型类型
        if hasattr(self.network.vit, 'get_param_groups'):
            lora_params = self.network.vit.get_param_groups()
        else:
            # 对于全参数微调模型，获取所有可训练参数
            lora_params = [p for p in self.network.vit.parameters() if p.requires_grad]
        
        lora_count = sum(p.numel() for p in lora_params)
        param_counts["lora"] = lora_count
        
        # 分类头参数
        fc_count = sum(p.numel() for p in self.network.fc.parameters())
        param_counts["classifier"] = fc_count
        total_count = lora_count + fc_count
        param_counts["total"] = total_count
        
        return param_counts

    def count_total_parameters(self) -> int:
        """统计模型总参数数量（包括冻结参数）"""
        return sum(p.numel() for p in self.network.parameters())

    def print_parameter_statistics(self, task_id: int) -> None:
        """打印参数统计信息"""
        trainable_params = self.count_trainable_parameters()
        total_params = self.count_total_parameters()
        
        logging.info(f"=== 任务 {task_id} 参数统计 ===")
        logging.info(f"总模型参数: {total_params:,}")
        logging.info(f"可训练参数: {trainable_params['total']:,}")
        logging.info(f"  - LoRA参数: {trainable_params['lora']:,}")
        logging.info(f"  - 分类头参数: {trainable_params['classifier']:,}")
        
        # 计算参数效率
        efficiency = (trainable_params['total'] / total_params) * 100
        logging.info(f"参数效率: {efficiency:.2f}%")
