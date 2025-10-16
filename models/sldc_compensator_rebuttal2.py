# -*- coding: utf-8 -*-
import math
import copy
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
import logging

from compensator.gaussian_statistics import GaussianStatistics
from compensator.sldc_linear import LinearCompensator
from compensator.sldc_weaknonlinear import WeakNonlinearCompensator
from compensator.sldc_attention import SemanticDriftCompensator

from classifier.ls_classifier_builder import LeastSquaresClassifierBuilder
from classifier.sgd_classifier_builder import SGDClassifierBuilder
from classifier.da_classifier_builder import LDAClassifierBuilder, QDAClassifierBuilder

def symmetric_cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor,
                                 sce_a: float = 0.5, sce_b: float = 0.5) -> torch.Tensor:
    pred = F.softmax(logits, dim=1)
    pred = torch.clamp(pred, min=1e-7, max=1.0)

    label_one_hot = F.one_hot(targets, num_classes=pred.size(1)).float().to(pred.device)
    label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)

    ce_loss = -(label_one_hot * torch.log(pred)).sum(dim=1).mean()
    rce_loss = -(pred * torch.log(label_one_hot)).sum(dim=1).mean()
    return sce_a * ce_loss + sce_b * rce_loss


class Drift_Compensator(object):
    def __init__(self, args):
        # 设备 & 基本超参
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.auxiliary_data_size = args.get('auxiliary_data_size', 1024)
        self.args = args
        self.compensate = args.get('compensate', True)
        self.use_nonlinear = args.get('use_weaknonlinear', True)


        # 缓存 & 容器
        self.cached_Z = None
        self.aux_loader = None
        self.linear_transforms = {}
        self.linear_transforms_current_only = {}
        self.weaknonlinear_transforms = {}
        self.weaknonlinear_transforms_current_only = {}
        self.feature_dim = None

    @torch.no_grad()
    def extract_features_before_after(self, model_before, model_after, data_loader):
        """从同一批数据上抽取“前后模型”的特征用于补偿器估计。"""
        model_before, model_after = model_before.to(self.device), model_after.to(self.device)
        model_before.eval()
        model_after.eval()

        feats_before, feats_after, targets = [], [], []
        for batch in data_loader:
            inputs, batch_targets = batch[0], batch[1]
            inputs = inputs.to(self.device)
            feats_before.append(model_before(inputs).cpu())
            feats_after.append(model_after(inputs).cpu())
            targets.append(batch_targets)

        feats_before = torch.cat(feats_before)
        feats_after = torch.cat(feats_after)
        targets = torch.cat(targets)
        return feats_before, feats_after, targets

    def compute_linear_transform(self, features_before: torch.Tensor, features_after: torch.Tensor, normalize: bool = True):
        """使用通用 LinearCompensator 拟合前后特征之间的线性映射。"""

        logging.info("基于当前任务的前后特征构建线性补偿器(alpha_1-SLDC)")
        linear_comp = LinearCompensator(
            input_dim=features_before.size(1),
            gamma=self.gamma,
            temp=self.temp,
            device=self.device,
        )
        W_global = linear_comp.train(features_before, features_after, normalize=normalize)

        with torch.no_grad():
            fb = features_before.to(self.device)
            fa = features_after.to(self.device)
            preds = fb @ W_global
            feat_diffs = (fa - fb).norm(dim=1).mean().item()
            feat_diffs_pred = (fa - preds).norm(dim=1).mean().item()
            s = torch.linalg.svdvals(W_global)
            max_singular = s[0].item()
            min_singular = s[-1].item()
            weight = math.exp(-fb.size(0) / (self.temp * fb.size(1)))

        logging.info(
            f"仿射变换矩阵对角线元素均值：{W_global.diag().mean().item():.4f}，"
            f"融合权重：{weight:.4f}，样本数量：{fb.size(0)}；"
            f"线性修正前差异：{feat_diffs:.4f}；修正后差异：{feat_diffs_pred:.4f}；"
            f"最大奇异值：{max_singular:.2f}；最小奇异值：{min_singular:.2f}")

        return linear_comp

    def compute_weaknonlinear_transform(self, features_before: torch.Tensor, features_after: torch.Tensor):
        """基于前/后特征训练弱非线性变换器（Residual MLP）。"""

        logging.info("基于当前任务的前后特征构建弱非线性补偿器")
        transform = WeakNonlinearCompensator(
            input_dim=features_before.size(1),
            device=self.device,
        )
        transform.train(features_before.to(self.device), features_after.to(self.device))

        with torch.no_grad():
            fb = features_before.to(self.device)
            fa = features_after.to(self.device)
            transformed_features = transform.transform_features(fb)
            feat_diffs = (fa - fb).norm(dim=1).mean().item()
            feat_diffs_nonlinear = (fa - transformed_features).norm(dim=1).mean().item()
            logging.info(
                f"非线性修正前差异：{feat_diffs:.4f}；修正后差异：{feat_diffs_nonlinear:.4f}")

        return transform

    def semantic_drift_compensation(self, old_stats_dict: Dict[int, GaussianStatistics],
                                    features_before: torch.Tensor, features_after: torch.Tensor,
                                    labels: torch.Tensor, use_auxiliary: bool = False) -> Dict[int, GaussianStatistics]:
        if not old_stats_dict:
            return {}

        sdc = SemanticDriftCompensator(
            input_dim=features_before.size(1),
            sigma=1.0,
            device=self.device,
        )
        sdc.train(features_before.to(self.device), features_after.to(self.device))
        compensated_stats = sdc.compensate(old_stats_dict)

        logging.info(f"[SDC] Drift compensation applied on {len(compensated_stats)} classes (aux={use_auxiliary})")
        return compensated_stats


    def _build_stats(self, features: torch.Tensor, labels: torch.Tensor) -> Dict[int, object]:
        """仅使用类内协方差构建（μ, Σ）。"""
        features = features.cpu()
        labels = labels.cpu()

        unique_labels = torch.unique(labels)
        stats = {}

        for lbl in unique_labels:
            mask = (labels == lbl)
            class_feats = features[mask]
            class_mean = class_feats.mean(dim=0)

            if class_feats.size(0) >= 2:
                class_cov = torch.cov(class_feats.T)
            else:
                d = class_feats.size(1)
                class_cov = torch.eye(d, dtype=class_feats.dtype) * 1e-4

            cid = int(lbl.item())
            stats[cid] = GaussianStatistics(class_mean, class_cov)
        return stats

    # =========================
    #       DPCR: 核心新增
    # =========================
    @torch.no_grad()
    def _compute_class_projectors(self, feats_before: torch.Tensor, labels: torch.Tensor):
        """
        对每个类 c，用未中心化协方差 Φ_c = X_c^T X_c 的特征分解，取累计能量>=self.energy 的主成分子空间 V_r，
        得到类行空间近似投影基 V_r（投影为 V_r V_r^T）。
        返回 {cid: V_r}。
        """
        X = F.normalize(feats_before, dim=1).cpu()
        y = labels.cpu()
        proj_basis = {}

        for cid in torch.unique(y).tolist():
            mask = (y == cid)
            Xc = X[mask]  # [Nc, d]
            if Xc.size(0) == 0:
                continue

            # 未中心化协方差（行空间）
            Phi = Xc.T @ Xc  # [d, d], PSD
            evals, evecs = torch.linalg.eigh(Phi)  # 升序
            evals = torch.clamp(evals, min=0)

            if evals.sum() <= 0:
                # 极端退化：取单位基的前 r_cap 列
                V_r = torch.eye(Xc.size(1))[:, :min(self.r_cap, Xc.size(1))]  # [d, r]
            else:
                w = evals / (evals.sum() + 1e-12)
                # 从最大特征值开始累计
                w_rev = torch.flip(w, dims=[0])
                cumsum_rev = torch.cumsum(w_rev, dim=0)
                # 选择最小 k 使累计能量 >= self.energy
                idx = int((cumsum_rev >= self.energy).nonzero(as_tuple=False)[0].item() + 1)
                k = min(idx, self.r_cap, Xc.size(1))
                V_r = evecs[:, -k:]  # 取最大的 k 个特征向量
            proj_basis[int(cid)] = V_r.contiguous()
        return proj_basis

    @torch.no_grad()
    def _build_classwise_W(self, W: torch.Tensor, feats_before: torch.Tensor, labels: torch.Tensor):
        """给定全局 W，构造 {cid: W_c = W @ (V_r V_r^T)}。"""
        if W is None:
            return {}
        W = W.detach().cpu()
        basis = self._compute_class_projectors(feats_before, labels)  # {cid: V_r}
        W_dict = {}
        for cid, V_r in basis.items():
            W_c = (W @ (V_r @ V_r.T)).contiguous()  # [d, d]
            W_dict[cid] = W_c
        return W_dict

    def transform_stats_with_W_classwise(self, stats_dict: Dict[int, object], W_dict: Dict[int, torch.Tensor]):
        """
        类专属线性变换：x' = x @ W_c  =>  μ' = μ @ W_c, Σ' = W_c^T Σ W_c
        """
        if not stats_dict or not W_dict:
            return {}
        out = {}
        for cid, stat in stats_dict.items():
            Wc = W_dict.get(cid, None)
            if Wc is None:
                # 若缺少该类的 Wc，则保持原样（也可退回全局 W）
                out[cid] = GaussianStatistics(stat.mean.clone(), stat.cov.clone(), stat.reg)
                continue
            Wc = Wc.to(stat.mean.device)
            WTc = Wc.t()
            mean = stat.mean @ Wc
            cov  = WTc @ stat.cov @ Wc
            cov  = cov + 1e-6 * torch.eye(cov.size(0), device=cov.device, dtype=cov.dtype)  # 稳定化
            out[cid] = GaussianStatistics(mean, cov, stat.reg)
        return out
    # =========================
    #     DPCR: 核心新增结束
    # =========================

    def build_all_variants(self, task_id: int, model_before: nn.Module, model_after: nn.Module, data_loader):
        """构建各类"分布变体"统计量（含 DPCR / DPCR + ADE）。"""
        feats_before, feats_after, labels = self.extract_features_before_after(
            model_before, model_after, data_loader)

        # 设置特征维度
        if self.feature_dim is None:
            self.feature_dim = feats_after.size(1)

        if self.cached_Z is None:
            self.cached_Z = torch.randn(50000, feats_after.size(1))

        linear_global = None
        linear_current_only = None
        weaknonlinear_transform = None
        weaknonlinear_transform_current_only = None

        if self.compensate and task_id > 1:
            # —— 仅用当前任务拟合 W
            linear_current_only = self.compute_linear_transform(feats_before, feats_after)
            self.linear_transforms_current_only[task_id] = linear_current_only

            if self.use_nonlinear:
                weaknonlinear_transform_current_only = self.compute_weaknonlinear_transform(feats_before, feats_after)
                self.weaknonlinear_transforms_current_only[task_id] = weaknonlinear_transform_current_only

            # —— ADE：拼接辅助数据拟合更鲁棒的 W
            aux_loader = self.get_aux_loader(self.args)
            if aux_loader is not None:
                feats_aux_before, feats_aux_after, _ = self.extract_features_before_after(model_before, model_after, aux_loader)
                feats_before_combined = torch.cat([feats_before, feats_aux_before], dim=0)
                feats_after_combined = torch.cat([feats_after, feats_aux_after], dim=0)

                linear_global = self.compute_linear_transform(feats_before_combined, feats_after_combined)
                self.linear_transforms[task_id] = linear_global

                if self.use_nonlinear:
                    weaknonlinear_transform = self.compute_weaknonlinear_transform(feats_before_combined, feats_after_combined)
                    self.weaknonlinear_transforms[task_id] = weaknonlinear_transform
            else:
                # 如果没有ADE数据，使用当前任务的W
                linear_global = linear_current_only
                weaknonlinear_transform = weaknonlinear_transform_current_only

        # 基于"当前特征（after）"计算原始统计量
        stats = self._build_stats(features=feats_after, labels=labels)
      
        # 单位协方差版本（用于 LDC 系列）
        stats_unit_cov = {}
        for cid, stat in stats.items():
            d = stat.mean.size(0)
            unit_cov = torch.eye(d, device=stat.mean.device, dtype=stat.mean.dtype)
            stats_unit_cov[cid] = GaussianStatistics(stat.mean, unit_cov, stat.reg)

        # 初始化 variants 字典（新增：SeqFT + LDA / SeqFT + QDA）
        if not hasattr(self, 'variants') or len(self.variants) == 0:
            self.variants = {
                "SeqFT": {},
                "SeqFT without Cov": {},
                "SeqFT + LDA": {},          # 新增
                "SeqFT + QDA": {},          # 新增
                "alpha_1-SLDC + ADE": {},
                "alpha_2-SLDC + ADE": {},
                "alpha_1-SLDC": {},
                "alpha_2-SLDC": {},
                "LDC": {},
                "LDC + ADE": {},
                "SDC": {},
                "SDC + ADE": {},
                # 新增：DPCR（类行空间收缩）
                "DPCR": {},
                "DPCR + ADE": {},
            }

        # 直接更新"当前任务"的统计量
        self.variants["SeqFT"].update(copy.deepcopy(stats))
        self.variants["SeqFT without Cov"].update(copy.deepcopy(stats_unit_cov))
        # 新增：与 SeqFT 一致（真实协方差）
        self.variants["SeqFT + LDA"].update(copy.deepcopy(stats))
        self.variants["SeqFT + QDA"].update(copy.deepcopy(stats))

        if self.compensate and task_id > 1:
            # ============ 线性补偿（全局 W） ============
            if "alpha_1-SLDC + ADE" in self.variants and linear_global is not None:
                stats_compensated = self.transform_stats_with_W(self.variants['alpha_1-SLDC + ADE'], linear_global)
            else:
                stats_compensated = {}
            self.variants["alpha_1-SLDC + ADE"] = stats_compensated
            self.variants["alpha_1-SLDC + ADE"].update(copy.deepcopy(stats))

            if "alpha_1-SLDC" in self.variants and linear_current_only is not None:
                stats_compensated_current = self.transform_stats_with_W(self.variants['alpha_1-SLDC'], linear_current_only)
            else:
                stats_compensated_current = {}
            self.variants["alpha_1-SLDC"] = stats_compensated_current
            self.variants["alpha_1-SLDC"].update(copy.deepcopy(stats))

            # ============ 弱非线性补偿 ============
            if self.use_nonlinear and weaknonlinear_transform is not None:
                if "alpha_2-SLDC + ADE" in self.variants:
                    stats_weaknonlinear = weaknonlinear_transform.compensate(self.variants["alpha_2-SLDC + ADE"])
                else:
                    stats_weaknonlinear = {}
                self.variants["alpha_2-SLDC + ADE"] = stats_weaknonlinear
                self.variants["alpha_2-SLDC + ADE"].update(copy.deepcopy(stats))

            if self.use_nonlinear and weaknonlinear_transform_current_only is not None:
                if "alpha_2-SLDC" in self.variants:
                    stats_weaknonlinear_current = weaknonlinear_transform_current_only.compensate(self.variants["alpha_2-SLDC"])
                else:
                    stats_weaknonlinear_current = {}
                self.variants["alpha_2-SLDC"] = stats_weaknonlinear_current
                self.variants["alpha_2-SLDC"].update(copy.deepcopy(stats))

            # ============ LDC（单位协方差） ============
            if "LDC" in self.variants and linear_current_only is not None:
                stats_ldc_compensated = self.transform_stats_with_W(self.variants['LDC'], linear_current_only)
            else:
                stats_ldc_compensated = {}
            self.variants["LDC"] = stats_ldc_compensated
            self.variants["LDC"].update(copy.deepcopy(stats_unit_cov))

            if "LDC + ADE" in self.variants and linear_global is not None:
                stats_ldc_ade_compensated = self.transform_stats_with_W(self.variants['LDC + ADE'], linear_global)
            else:
                stats_ldc_ade_compensated = {}
            self.variants["LDC + ADE"] = stats_ldc_ade_compensated
            self.variants["LDC + ADE"].update(copy.deepcopy(stats_unit_cov))

            # ============ SDC ============
            if "SDC" in self.variants:
                stats_sdc_compensated = self.semantic_drift_compensation(
                    self.variants['SDC'], feats_before, feats_after, labels, use_auxiliary=False)
            else:
                stats_sdc_compensated = {}
            self.variants["SDC"] = stats_sdc_compensated
            self.variants["SDC"].update(copy.deepcopy(stats_unit_cov))

            if "SDC + ADE" in self.variants and aux_loader is not None:
                # 这里用 ADE 合并后的特征进行 SDC
                stats_sdc_ade_compensated = self.semantic_drift_compensation(
                    self.variants['SDC + ADE'], feats_before_combined, feats_after_combined, labels, use_auxiliary=True)
            else:
                stats_sdc_ade_compensated = {}
            self.variants["SDC + ADE"] = stats_sdc_ade_compensated
            self.variants["SDC + ADE"].update(copy.deepcopy(stats_unit_cov))


            logging.info("\n[DPCR] Performing real DPCR correction ...")
            dpcrr_corrected_stats = {}

            # 历史类来自 DPCR（上一轮构建时保存的旧类统计）
            if "DPCR" in self.variants and len(self.variants["DPCR"]) > 0 and linear_current_only is not None:
                old_stats = self.variants["DPCR"]
                W_current = linear_current_only.W.detach().to(self.device)
                for cid, st in old_stats.items():
                    mu_old = st.mean.to(self.device).float()
                    cov_old = st.cov.to(self.device).float()
                    M2 = cov_old
                    # U_r = self._principal_subspace_from_cov(M2)
                    P_ic = W_current  # @ (U_r @ U_r.T)

                    # 校正旧类统计到 θ_t 域
                    mu_hat = mu_old @ P_ic
                    cov_hat = P_ic.T @ cov_old @ P_ic
                    cov_hat = cov_hat + 1e-5 * torch.eye(cov_hat.size(0), device=self.device)

                    dpcrr_corrected_stats[cid] = GaussianStatistics(mu_hat.cpu(), cov_hat.cpu())

            merged_stats = copy.deepcopy(dpcrr_corrected_stats)
            merged_stats.update(copy.deepcopy(stats))
            self.variants["DPCR"] = merged_stats
            logging.info(f"[INFO] DPCR variant built with {len(merged_stats)} classes ")


            dpcrr_corrected_stats = {}
            if "DPCR + ADE" in self.variants and len(self.variants["DPCR + ADE"]) > 0 and linear_global is not None:
                old_stats = self.variants["DPCR + ADE"]
                W_global_matrix = linear_global.W.detach().to(self.device)
                for cid, st in old_stats.items():
                    mu_old = st.mean.to(self.device).float()
                    cov_old = st.cov.to(self.device).float()
                    M2 = cov_old
                    # U_r = self._principal_subspace_from_cov(M2)
                    P_ic = W_global_matrix  # @ (U_r @ U_r.T)

                    # 校正旧类统计到 θ_t 域
                    mu_hat = mu_old @ P_ic
                    cov_hat = P_ic.T @ cov_old @ P_ic
                    cov_hat = cov_hat + 1e-5 * torch.eye(cov_hat.size(0), device=self.device)

                    dpcrr_corrected_stats[cid] = GaussianStatistics(mu_hat.cpu(), cov_hat.cpu())

            merged_stats = copy.deepcopy(dpcrr_corrected_stats)
            merged_stats.update(copy.deepcopy(stats))
            self.variants["DPCR + ADE"] = merged_stats
            logging.info(f"[INFO] DPCRR + ADE variant built with {len(merged_stats)} classes ")            


        else:
            # 首任务或不做补偿：所有变体都先接入"当前统计量"以建立键空间
            self.variants["alpha_1-SLDC + ADE"].update(copy.deepcopy(stats))
            self.variants["alpha_2-SLDC + ADE"].update(copy.deepcopy(stats))
            self.variants["alpha_1-SLDC"].update(copy.deepcopy(stats))
            self.variants["alpha_2-SLDC"].update(copy.deepcopy(stats))
            self.variants["LDC"].update(copy.deepcopy(stats_unit_cov))
            self.variants["LDC + ADE"].update(copy.deepcopy(stats_unit_cov))
            self.variants["SDC"].update(copy.deepcopy(stats_unit_cov))
            self.variants["SDC + ADE"].update(copy.deepcopy(stats_unit_cov))
            self.variants["DPCR"].update(copy.deepcopy(stats))
            self.variants["DPCR + ADE"].update(copy.deepcopy(stats))
            # 新增：首任务也建立 SeqFT + LDA/QDA 的键空间（与 SeqFT 一致）
            self.variants["SeqFT + LDA"].update(copy.deepcopy(stats))
            self.variants["SeqFT + QDA"].update(copy.deepcopy(stats))

        logging.info(f"[INFO] Built distribution variants: {list(self.variants.keys())}, num classes: {len(stats)}")
      
        # 打印每个变体的类别数量
        for name, variant_stats in self.variants.items():
            logging.info(f"  {name}: {len(variant_stats)} classes")
      
        return self.variants

    @staticmethod
    def _principal_subspace_from_cov(Phi: torch.Tensor, energy: float = 0.95, r_cap: int = 512) -> torch.Tensor:
        """
        从非中心二阶矩(或任意PSD矩阵) Phi 中提取累计能量>=energy 的主成分子空间 U_r
        返回 U_r: [d, r]
        """
        d = Phi.size(0)
        # 数值稳定化
        Phi = 0.5 * (Phi + Phi.T)
        evals, evecs = torch.linalg.eigh(Phi)  # 升序
        evals = torch.clamp(evals, min=0)
        if torch.sum(evals) <= 0:
            k = min(r_cap, d)
            return torch.eye(d, device=Phi.device, dtype=Phi.dtype)[:, :k]

        # 按能量从大到小取前k个
        evals_desc, idx = torch.sort(evals, descending=True)
        evecs_desc = evecs[:, idx]
        cumsum = torch.cumsum(evals_desc, dim=0)
        ratio = cumsum / (evals_desc.sum() + 1e-12)
        k = int((ratio >= energy).nonzero(as_tuple=False)[0].item() + 1)
        k = min(k, r_cap, d)
        U_r = evecs_desc[:, :k].contiguous()
        return U_r


    def transform_stats_with_W(self, stats_dict: Dict[int, GaussianStatistics],
                               linear_comp: Optional[LinearCompensator]) -> Dict[int, GaussianStatistics]:
        """使用线性补偿器对高斯统计进行批量变换。"""

        if linear_comp is None or not stats_dict:
            return {}
        return linear_comp.compensate(stats_dict)

    def train_classifier_with_cached_samples(self, _: nn.Module, stats: Dict[int, GaussianStatistics],
                                             epochs: int = 5) -> nn.Module:
        """利用统一的 SGDClassifierBuilder 构建基于高斯采样的线性分类器。"""

        builder = SGDClassifierBuilder(
            cached_Z=self.cached_Z,
            device=self.device,
            epochs=epochs,
            lr=0.01,
        )
        model = builder.build(stats)
        logging.info(f"[INFO] SGD builder finished: {len(stats)} classes, epochs={epochs}")
        return model

    def refine_classifiers_from_variants(self, fc: nn.Module, epochs: int = 5,
                                        reg_alpha: float = 0.5, reg_type: str = "shrinkage") -> Dict[str, nn.Module]:
        """根据各分布变体构建相应的分类器。"""

        assert hasattr(self, 'variants') and len(self.variants) > 0, "No variants found. Call build_all_variants first."

        out: Dict[str, nn.Module] = {}
        lda_builder = LDAClassifierBuilder(reg_alpha=reg_alpha, reg_type=reg_type, device=self.device)
        qda_builder = QDAClassifierBuilder(reg_alpha=reg_alpha, reg_type=reg_type, device=self.device)

        for name, stats in self.variants.items():
            if len(stats) == 0:
                logging.info(f"[WARNING] Variant '{name}' has no statistics, skipping...")
                continue

            if name in ["LDC", "LDC + ADE", "SDC", "SDC + ADE"]:
                clf = self.train_classifier_with_cached_samples(fc, stats, epochs=epochs)
                out[name] = clf
                logging.info(f"[INFO] {name}: SGD with {len(stats)} classes")

            elif name in ["DPCR", "DPCR + ADE"]:
                ls_clf = self.train_least_squares_classifier(stats, reg_lambda=1e-3)
                out[name] = ls_clf
                logging.info(f"[INFO] {name}: LS with {len(stats)} classes")

            elif name in ["SeqFT without Cov", "SeqFT + LDA"]:
                clf = lda_builder.build(stats)
                out[name] = clf
                logging.info(f"[INFO] {name}: LDA with {len(stats)} classes")

            elif name == "SeqFT + QDA":
                clf = qda_builder.build(stats)
                out[name] = clf
                logging.info(f"[INFO] {name}: QDA with {len(stats)} classes")

            elif name == "SeqFT":
                clf = self.train_classifier_with_cached_samples(fc, stats, epochs=epochs)
                out[name] = clf
                logging.info(f"[INFO] {name}: SGD with {len(stats)} classes")

            elif name in ["alpha_1-SLDC", "alpha_2-SLDC", "alpha_1-SLDC + ADE", "alpha_2-SLDC + ADE"]:
                lda_clf = lda_builder.build(stats)
                qda_clf = qda_builder.build(stats)
                sgd_clf = self.train_classifier_with_cached_samples(fc, stats, epochs=epochs)

                out[f"{name} (LDA)"] = lda_clf
                out[f"{name} (QDA)"] = qda_clf
                out[f"{name} (SGD)"] = sgd_clf
                logging.info(f"[INFO] {name}: LDA/QDA/SGD with {len(stats)} classes")

            else:
                logging.info(f"[WARNING] Unknown variant '{name}', using SGD as default")
                clf = self.train_classifier_with_cached_samples(fc, stats, epochs=epochs)
                out[name] = clf

            with torch.no_grad():
                test_input = torch.randn(2, list(stats.values())[0].mean.size(0), device=self.device)
                if name in ["alpha_1-SLDC", "alpha_2-SLDC", "alpha_1-SLDC + ADE", "alpha_2-SLDC + ADE"]:
                    _ = out[f"{name} (LDA)"](test_input)
                    _ = out[f"{name} (QDA)"](test_input)
                    _ = out[f"{name} (SGD)"](test_input)
                else:
                    _ = out[name](test_input)

        logging.info(f"[INFO] Created {len(out)} classifiers from variants.")

        for key, value in out.items():
            out[key] = value.cpu()
        return out

    def train_least_squares_classifier(self, stats: Dict[int, GaussianStatistics], reg_lambda: float = 1e-3) -> nn.Module:
        """使用 LeastSquaresClassifierBuilder 构建最小二乘分类器。"""

        builder = LeastSquaresClassifierBuilder(
            cached_Z=self.cached_Z,
            reg_lambda=reg_lambda,
            device=self.device,
        )
        model = builder.build(stats)
        logging.info(f"[INFO] LS builder finished: {len(stats)} classes, λ={reg_lambda}")
        return model


    def initialize_aux_loader(self, train_set):
        """外部传入一个 DataSet（如 replay/额外数据），用于 ADE 拟合 W。"""
        if self.aux_loader is not None:
            return self.aux_loader
        self.aux_loader = DataLoader(train_set, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
        return self.aux_loader

    def get_aux_loader(self, args):
        """若未初始化，将返回 None；使用前请先调用 initialize_aux_loader。"""
        return self.aux_loader
