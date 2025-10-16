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
from models.sldc_modules2 import Drift_Compensator
from utils.inc_net import BaseNet
from lora import compute_covariances
import math 
from lora import compute_covariances

class EMASmooth:
    def __init__(self, alpha=0.9):
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

def symmetric_cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    sce_a: float = 0.5,
    sce_b: float = 0.5) -> torch.Tensor:

    logsoftmax = F.log_softmax(logits, dim=1)
    softmax = logsoftmax.exp()

    oh = F.one_hot(targets, num_classes=logits.size(1)).float()
    oh = torch.clamp(oh, min=1e-4, max=1.0)

    ce  = -(oh * logsoftmax).sum(dim=1).mean()
    rce = -(softmax * oh).sum(dim=1).mean()
    return sce_a * ce + sce_b * rce

def feature_distillation_loss(
    teacher_feat: torch.Tensor, student_feat: torch.Tensor) -> torch.Tensor:
    return ((teacher_feat - student_feat) ** 2).mean()

def cosine_similarity_loss(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(x1, x2, dim=-1)).mean()

@dataclass
class Timing:
    train: float = 0.0
    drift: float = 0.0
    total: float = 0.0

class SubspaceLoRA(BaseLearner):
    def __init__(self, args: Dict[str, Any]) -> None:
        super().__init__(args)

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = BaseNet(args, pretrained=True).to(self._device)
        self.args = args

        self._timings: Timing = Timing()
        self.time_history: List[Dict[str, float]] = []

        self.sce_a: float = args["sce_a"]
        self.sce_b: float = args["sce_b"]

        self.batch_size: int = args["batch_size"]
        self.iterations: int = args["iterations"]
        self.warmup_steps: int = int(args["warmup_ratio"] * self.iterations)
        self.ca_epochs: int = args["ca_epochs"]
        self.lrate: float = args["lrate"]
        self.weight_decay: float = args["weight_decay"]
        self.optimizer_type: str = args["optimizer"]
        self.compensate: bool = args["compensate"]

        self.distill_head = None

        kd_type = args["kd_type"]

        if kd_type == "feat":
            self.kd_loss_fn = feature_distillation_loss
        elif kd_type == "cos":
            self.kd_loss_fn = cosine_similarity_loss
        else:
            raise ValueError(f"Unsupported kd_type = {kd_type}")

        self.use_feature_kd: bool = args["gamma_kd"] > 0.0
        self.gamma_kd: float = args["gamma_kd"] if self.use_feature_kd else 0.0
        self.gamma_norm: float = args["gamma_norm"] if self.use_feature_kd else 0.0
        self.use_aux_for_kd: bool = args["use_aux_for_kd"]

        self.update_teacher_each_task: bool = args["update_teacher_each_task"]
        
        self.aux_loader = None
        self.aux_iter = None

        self.covariances: Dict[str, torch.Tensor] | None = None
        self.drift_compensator = Drift_Compensator(args)

        self.teacher_network = None   # 用于 KD
        self.prev_network = None      # 用于漂移补偿（上一个任务结束后的模型）
        self.seed: int = args["seed"]
        self.task_count: int = 0
        self.current_task_id = 0
        
        logging.info(f"Optimizer instantiated: lrate={self.lrate}, wd={self.weight_decay}, optimizer={self.optimizer_type}")

        self.loss_smoother = EMASmooth(alpha=0.9)
        self.kd_smoother = EMASmooth(alpha=0.9)
        self.acc_smoother = EMASmooth(alpha=0.9)

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
        self.fc_dict = self.drift_compensator.refine_classifiers_from_variants(self.network.fc, self.ca_epochs)
        self.network.fc = next(iter(self.fc_dict.values()))

    def after_task(self) -> None:
        """Update class counters after finishing a task."""
        self._known_classes = self._total_classes
        self.update_projection_matrices()
        self.task_count += 1

    def incremental_train(self, data_manager) -> None:
        start_time = time.time()
        task_id = self.current_task_id
        task_size = data_manager.get_task_size(task_id)

        if self.use_feature_kd:
            if self.distill_head is None:
                self.initialize_distillation_head()

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
                                    num_workers=2, pin_memory=True, persistent_workers=False)
        self.test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False,
                                    num_workers=2, pin_memory=True, persistent_workers=False)

        dataset_size = len(train_set_test_mode)
        max_samples = getattr(self, 'max_train_test_samples', 5000)
        sampler = None
        if dataset_size > max_samples:
            indices = torch.randperm(dataset_size)[:max_samples].tolist()
            sampler = SubsetRandomSampler(indices)
            print(f"⚠️ Dataset too large ({dataset_size}), sampling {max_samples} examples for test-mode training set.")

        self.train_loader_test_mode = DataLoader(
            train_set_test_mode,
            batch_size=self.batch_size,
            shuffle=False if sampler else True,
            sampler=sampler,
            num_workers=3,
            pin_memory=True,
            persistent_workers=True)

        if self.use_feature_kd:
            if self.teacher_network is None:
                self.teacher_network = copy.deepcopy(self.network).to(self._device)
                self.teacher_network.vit.finalize_without_lora()
            
            elif self.update_teacher_each_task:
                self.teacher_network = copy.deepcopy(self.network).to(self._device)
                self.teacher_network.vit.finalize_without_lora()

        if self.compensate:
            self.prev_network = copy.deepcopy(self.network).cpu()
            self.prev_network.vit.finalize_without_lora()

        # === 初始化 DriftCompensator 所需的 loader，但不用于 KD ===
        self.get_aux_loader(self.args)
        self.drift_compensator.initialize_aux_loader(self.aux_trainset)
        self.network.update_fc(task_size)
        self.network.fc.to(self._device)

        logging.info(
            "System training on classes %d-%d (%s)",
            self._known_classes,
            self._total_classes,
            data_manager.dataset_name.lower())

        if self.args['eval_only']:
            self.load_checkpoint(self.args["log_path"])
        else:
            self.system_training()
            self.save_checkpoint(self.args["log_path"])

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
        distill_params = list(self.distill_head.parameters()) if self.distill_head is not None else []

        param_groups = [
            {"params": lora_params, "lr": self.lrate, "weight_decay": self.weight_decay},
            {"params": fc_params, "lr": 1e-3 if self.optimizer_type == "adamw" else 5e-3, "weight_decay": self.weight_decay},
            {"params": distill_params, "lr": 1e-3 if self.optimizer_type == "adamw" else 1e-2, "weight_decay": 1e-4},
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
                    eta_min = getattr(self, 'eta_min', self.lrate // 10)
                    lr_ratio = eta_min / initial_lr
                    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                    return lr_ratio + cosine_decay * (1.0 - lr_ratio)
            
            def const_lr_lambda(step):
                return 1.0
            
            lr_lambdas = [lora_lr_lambda, const_lr_lambda, const_lr_lambda]
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambdas, last_epoch=-1)
        return optimizer, scheduler


    def system_training(self) -> None:
        """Train ViT + new classifier head for ``self.epochs`` epochs."""
        fc_params = self.network.fc.parameters()
        lora_params = self.network.vit.get_param_groups()
        optimizer, scheduler = self.make_optimizer(
            lora_params,
            fc_params)
        
        start = time.time()
        self.network.train()
        
        step = 1
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
        kd_term = 0.0

        feats = self.network.forward_features(inputs)

        # === 知识蒸馏：仅当 teacher_network 存在且 KD 启用 ===
        if self.use_feature_kd and self.teacher_network is not None:
            with torch.no_grad():
                teacher_feats = self.teacher_network.forward_features(inputs)
                teacher_transformed = self.distill_head(teacher_feats)

            student_norm = F.normalize(feats, dim=-1)
            teacher_norm = F.normalize(teacher_transformed, dim=-1)
            kd_cos = 1.0 - (student_norm * teacher_norm).sum(dim=-1).mean()
            kd_term = self.gamma_kd * kd_cos

        # === 分类损失 ===
        logits = self.network.fc(feats)
        new_targets_rel = torch.where(
            targets - self._known_classes >= 0,
            targets - self._known_classes, -100)
        new_logits = logits[:, self._known_classes:]
        sce = symmetric_cross_entropy_loss(new_logits, new_targets_rel, self.sce_a, self.sce_b)

        loss = sce + kd_term

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            n_correct = (pred == targets).sum().item()

        kd_raw = (kd_term.item() / self.gamma_kd if isinstance(kd_term, torch.Tensor) and self.gamma_kd != 0 else float(kd_term))

        return loss.item(), n_correct, kd_raw



    @staticmethod
    def norm_loss(t_feat: torch.Tensor, s_feat: torch.Tensor) -> torch.Tensor:
        """MSE between L2‑norms of teacher / student feature vectors."""
        t_norm = t_feat.norm(p=2, dim=1)
        s_norm = s_feat.norm(p=2, dim=1)
        return F.mse_loss(t_norm, s_norm)
    
    def initialize_distillation_head(self, ratio=3, use_identity_linear=False):
        feat_dim = self.network.vit.feature_dim

        if use_identity_linear:
            # 创建一个简单的线性层，无偏置
            distill_head = nn.Linear(feat_dim, feat_dim, bias=False)
            # 初始化权重为单位矩阵
            nn.init.eye_(distill_head.weight)
            self.distill_head = distill_head.to(self._device)
        else:
            self.distill_head = nn.Sequential(
                nn.Linear(feat_dim, ratio * feat_dim, bias=False),
                nn.ReLU(),
                nn.Linear(ratio * feat_dim, feat_dim, bias=False)
            ).to(self._device)

        logging.info(f"Initialized new distillation head for task {self.current_task_id} "
                    f"(identity linear: {use_identity_linear})")

    def evaluate(
        self,
        loader: DataLoader,
        fc_dict):

        self.network.eval()
        total = 0
        corrects = {}
        for name, fc in fc_dict.items():
            corrects[name] = 0

        with torch.no_grad():
            for _, batch in enumerate(loader):
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

    def eval_task(self):
        results = self.evaluate(
            self.test_loader,
            fc_dict=self.fc_dict)
        if not hasattr(self, "all_task_results"):
            self.all_task_results: Dict[int, Dict[str, float]] = {}
        self.all_task_results[self.current_task_id] = results
        return results

    def update_projection_matrices(self):
        if self.current_task_id >= 0 and self.network.vit.use_projection:
            new_covs = compute_covariances(self.network.vit, self.train_loader_test_mode)
            if self.covariances is None:
                self.covariances = new_covs
            else:
                for k in self.covariances:
                    self.covariances[k] = 0.9 * self.covariances[k] + new_covs[k] + 1e-7 * torch.eye(self.covariances[k].size(0)).to(self.covariances[k].device)
            self.network.update_projection_matrices(self.covariances)

    def loop(self, data_manager) -> Dict[str, List[float | None]]:
        self.data_manager = data_manager
        for _ in range(data_manager.nb_tasks):
            self.incremental_train(data_manager)
            self.refine_classifiers()
            logging.info(f"Evaluating after task {self.current_task_id}...")
            self.eval_task()
            self.analyze_task_results(self.all_task_results)
            self.after_task()
        return self.all_task_results

    def analyze_task_results(self, all_task_results: Dict[int, Dict[str, float]]) -> Dict[str, Any]:
        """
        Analyze incremental learning evaluation results.
        Logs and returns final-task accuracy and average accuracy across tasks for each variant.
        """
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
            dataset = datasets.ImageFolder(args['auxiliary_data_path'] + '/imagenet', transform=transform)
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
