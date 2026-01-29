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
        
        self.drift_compensator.build_all_variants(
            self.current_task_id,
            self.prev_network.vit,
            self.network.vit,
            self.train_loader_test_mode)

        self._timings.drift = time.time() - drift_start

    def refine_classifiers(self):
        self.fc_dict = self.classifier_reconstructor.build_classifiers(
            self.drift_compensator.variants
        )

    def after_task(self) -> None:
        """Update class counters after finishing a task."""
        self._known_classes = self._total_classes
        self.update_projection_matrices()
        self.task_count += 1

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
                                    num_workers=3, pin_memory=True, persistent_workers=False)
        
        self.test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False,
                                    num_workers=3, pin_memory=True, persistent_workers=False)

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
            return self._evaluate_cross_domain(loader, fc_dict)
        else:
            # For regular scenarios: compute overall accuracy
            return self._evaluate_regular(loader, fc_dict)
    
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
        Cross-domain evaluation: compute class-wise accuracy and average across all classes.
        This ensures each class contributes equally to the final metric, regardless of sample size.
        """
        self.network.eval()
        
        # Get task information from data manager
        data_manager = self.data_manager
        num_tasks = data_manager.nb_tasks
        
        # DEBUG: 添加调试日志验证假设
        logging.info(f"[DEBUG] _evaluate_cross_domain called:")
        logging.info(f"[DEBUG]   Current task ID: {self.current_task_id}")
        logging.info(f"[DEBUG]   Total tasks in data manager: {num_tasks}")
        logging.info(f"[DEBUG]   Should evaluate tasks: 0 to {self.current_task_id - 1}")
        
        # Initialize per-class accuracy tracking for all learned classes
        # We need to track accuracy for each individual class across all tasks

        for name, fc in self.fc_dict.items():
            fc.to(self._device)

        class_corrects = {name: {} for name in fc_dict.keys()}
        class_totals = {}
        
        with torch.no_grad():
            for batch in loader:
                inputs = batch[0].to(self._device)
                targets = batch[1]
                feats = self.network.forward_features(inputs)
                for name, fc in fc_dict.items():
                    preds = fc(feats).argmax(dim=1).cpu()                    
                    # Process each sample to update per-class statistics
                    for pred, target in zip(preds, targets):
                        # Only consider classes that have been learned so far (0 to _total_classes-1)
                        # In cross-domain setting, we evaluate on cumulative test set with all learned classes
                        if target < self._total_classes:
                            # Initialize class tracking if needed
                            if target not in class_corrects[name]:
                                class_corrects[name][target] = 0
                            if target not in class_totals:
                                class_totals[target] = 0
                            
                            # Update statistics
                            class_totals[target] += 1
                            if pred == target:
                                class_corrects[name][target] += 1
        
        # Compute per-class accuracies
        class_accuracies = {}
        for name in fc_dict.keys():
            class_accuracies[name] = []
            for class_id in sorted(class_totals.keys()):
                if class_totals[class_id] > 0:
                    acc = 100 * class_corrects[name].get(class_id, 0) / class_totals[class_id]
                    class_accuracies[name].append(float(np.around(acc, 2)))
        
        # Log per-class accuracies summary
        logging.info("Cross-domain class-wise accuracies:")
        for name in fc_dict.keys():
            logging.info(f"  Variant: {name}")
            if class_accuracies[name]:
                avg_acc = np.mean(class_accuracies[name])
                min_acc = np.min(class_accuracies[name])
                max_acc = np.max(class_accuracies[name])
                logging.info(f"    Average across all classes: {avg_acc:.2f}%")
                logging.info(f"    Min class accuracy: {min_acc:.2f}%")
                logging.info(f"    Max class accuracy: {max_acc:.2f}%")
                logging.info(f"    Total classes evaluated: {len(class_accuracies[name])}")
            else:
                logging.info(f"    No classes evaluated")
        
        # Return average across all classes
        results = {}
        for name in fc_dict.keys():
            if class_accuracies[name]:
                results[name] = float(np.around(np.mean(class_accuracies[name]), 2))
            else:
                results[name] = 0.0
        
        return results

    def eval_task(self):
        num_samples_for_eval = self.args.get('num_samples_per_task_for_evaluation', 0)
        if num_samples_for_eval > 0:
            logging.info(f"🔍 Using sampled evaluation with {num_samples_for_eval} samples per task")
        results = self.evaluate(
            self.test_loader,
            fc_dict=self.fc_dict)
        if not hasattr(self, "all_task_results"):
            self.all_task_results: Dict[int, Dict[str, float]] = {}
        self.all_task_results[self.current_task_id] = results
        return results

    def update_projection_matrices(self):
        if hasattr(self.network.vit, 'use_projection') and self.network.vit.use_projection:
            if self.current_task_id >= 0:
                new_covs = compute_covariances(self.network.vit, self.train_loader_test_mode)
                if self.covariances is None:
                    self.covariances = new_covs
                else:
                    for k in self.covariances:
                        self.covariances[k] = 0.9 * self.covariances[k] + new_covs[k] + 1e-7 * torch.eye(self.covariances[k].size(0)).to(self.covariances[k].device)
                self.network.update_projection_matrices(self.covariances)

    def loop(self, data_manager) -> Dict[str, Any]:
        self.data_manager = data_manager
        final_analysis: Dict[str, Any] | None = None
        for _ in range(data_manager.nb_tasks):
            self.incremental_train(data_manager)
            self.refine_classifiers()
            logging.info(f"Evaluating after task {self.current_task_id}...")
            self.eval_task()
            final_analysis = self.analyze_task_results(self.all_task_results)
            self.after_task()

        if final_analysis is None:
            final_analysis = self.analyze_task_results(self.all_task_results)

        combined_results: Dict[str, Any] = dict(final_analysis)
        combined_results["per_task_results"] = dict(self.all_task_results)
        return combined_results

    def analyze_task_results(self, all_task_results: Dict[int, Dict[str, float]]) -> Dict[str, Any]:
        if not all_task_results:
            logging.info("📊 Task evaluation results are empty. Nothing to analyze.")
            return {
                "last_task_id": None,
                "last_task_accuracies": {},
                "average_accuracies": {}}

        # Sort task IDs and get the last one
        task_ids = sorted(all_task_results.keys())
        last_task_id = task_ids[-1]


        variant_names = set()
        for task_dict in all_task_results.values():
            variant_names.update(task_dict.keys())
        variant_names = sorted(variant_names)

        # Compute final-task accuracies
        last_task_accuracies = {
            variant: all_task_results[last_task_id].get(variant, 0.0)
            for variant in variant_names
        }

        # Compute average accuracies across all tasks
        average_accuracies = {}
        for variant in variant_names:
            accs = [all_task_results[task_id].get(variant, 0.0) for task_id in task_ids]
            average_accuracies[variant] = float(np.mean(accs))

        # === Log Results in Structured Format ===
        logging.info(" Incremental Learning Evaluation Analysis:")
        logging.info(f"   Last Task ID: {last_task_id}")

        logging.info("  ── Final Task Accuracy (%) ──")
        for variant in variant_names:
            logging.info(f"      {variant:<20} : {last_task_accuracies[variant]:.2f}%")

        logging.info("   ── Average Accuracy Across Tasks (%) ──")
        for variant in variant_names:
            logging.info(f"      {variant:<20} : {average_accuracies[variant]:.2f}%")

        # Optional: Identify best variants and log summary
        best_last = max(last_task_accuracies, key=last_task_accuracies.get)
        best_avg = max(average_accuracies, key=average_accuracies.get)

        if best_last == best_avg:
            summary = f" Variant '{best_last}' is best in both final task and average performance."
        else:
            summary = f" Best in Final Task: '{best_last}' | Best Average: '{best_avg}'"

        logging.info("   ── Summary ──")
        logging.info(f"      {summary}")

        # Return structured data for further use
        return {
            "last_task_id": last_task_id,
            "last_task_accuracies": last_task_accuracies,
            "average_accuracies": average_accuracies}
    
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
