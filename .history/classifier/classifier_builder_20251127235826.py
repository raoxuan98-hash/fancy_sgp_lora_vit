# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import logging
import time
from typing import Dict, Union, List

from classifier.da_classifier_builder import LDAClassifierBuilder, QDAClassifierBuilder
from classifier.ls_classifier_builder import LeastSquaresClassifierBuilder
from classifier.sgd_classifier_builder import SGDClassifierBuilder


def log_time_usage(operation_name: str, start_time: float, end_time: float):
    """记录时间损耗情况"""
    elapsed_time = end_time - start_time
    logging.info(f"[Time] {operation_name}: {elapsed_time:.4f}s")


def get_gpu_memory_info() -> Dict[str, float]:
    """获取当前GPU显存信息"""
    if not torch.cuda.is_available():
        return {"allocated": 0.0, "reserved": 0.0, "max_allocated": 0.0}
    
    return {
        "allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
        "reserved": torch.cuda.memory_reserved() / 1024**3,    # GB
        "max_allocated": torch.cuda.max_memory_allocated() / 1024**3  # GB
    }


def log_memory_usage(operation_name: str, start_memory: Dict[str, float], end_memory: Dict[str, float]):
    """记录显存使用情况"""
    allocated_diff = end_memory["allocated"] - start_memory["allocated"]
    reserved_diff = end_memory["reserved"] - start_memory["reserved"]
    
    logging.info(
        f"[GPU Memory] {operation_name}: "
        f"Allocated={end_memory['allocated']:.2f}GB "
        f"(+{allocated_diff:.2f}GB), "
        f"Reserved={end_memory['reserved']:.2f}GB "
        f"(+{reserved_diff:.2f}GB), "
        f"Max={end_memory['max_allocated']:.2f}GB"
    )


class ClassifierReconstructor:
    """
    统一的分类器重构模块：
    输入各 variant 的高斯统计，输出 {variant_name: {classifier_type: nn.Module}}
    """
    def __init__(self, device="cuda",  **kwargs):
        self.device = device
        self.kwargs = kwargs
        self.cached_Z = None

        if 'lda_reg_alpha' in kwargs:
            self.lda_reg_alpha = kwargs['lda_reg_alpha']
            logging.info(f"[ClassifierReconstructor] LDA regularization alpha set to {self.lda_reg_alpha}")
        else:
            self.lda_reg_alpha = 0.4
        if all(k in kwargs for k in ('qda_reg_alpha1', 'qda_reg_alpha2', 'qda_reg_alpha3')):
            self.qda_reg_alpha1 = kwargs['qda_reg_alpha1']
            self.qda_reg_alpha2 = kwargs['qda_reg_alpha2']
            self.qda_reg_alpha3 = kwargs['qda_reg_alpha3']
            logging.info(
                "[ClassifierReconstructor] QDA regularization alphas set to %s, %s, %s",
                self.qda_reg_alpha1,
                self.qda_reg_alpha2,
                self.qda_reg_alpha3,
            )
        else:
            self.qda_reg_alpha1 = 0.25
            self.qda_reg_alpha2 = 0.25
            self.qda_reg_alpha3 = 0.25

    # def build_classifiers(self, variants: Dict[str, Dict[int, object]], classifier_type: Union[str, List[str]] = ["lda", "qda"]) -> Dict[str, Dict[str, nn.Module]]:
    def build_classifiers(self, variants: Dict[str, Dict[int, object]], classifier_type: Union[str, List[str]] = ["qda"], low_rank_cov_for_qda = False) -> Dict[str, Dict[str, nn.Module]]:
        if isinstance(classifier_type, str):
            classifier_types = [classifier_type]
        else:
            classifier_types = classifier_type
        
        out = {}
        for cls_type in classifier_types:
            classifier_builder = self._get_classifier_builder(variants, cls_type, low_rank_cov_for_qda)
            if classifier_builder is not None:  # 确保分类器构建器不是 None
                for name, stats in variants.items():
                    cls_name = name + " + " + cls_type.upper()
                    
                    # 记录构建前的显存和时间
                    start_memory = get_gpu_memory_info()
                    start_time = time.time()
                    out[cls_name] = classifier_builder.build(stats)
                    end_time = time.time()
                    end_memory = get_gpu_memory_info()
                    
                    # 记录时间损耗
                    log_time_usage(f"Classifier {cls_name} built", start_time, end_time)
                    
                    # 记录显存损耗
                    allocated_diff = end_memory["allocated"] - start_memory["allocated"]
                    reserved_diff = end_memory["reserved"] - start_memory["reserved"]
                    
                    logging.info(f"[GPU Memory] Classifier {cls_name} built: "
                               f"Allocated={end_memory['allocated']:.2f}GB "
                               f"(+{allocated_diff:.2f}GB), "
                               f"Reserved={end_memory['reserved']:.2f}GB "
                               f"(+{reserved_diff:.2f}GB)")
                    
                logging.info(f"[Classifier] Built classifiers for {len(out)} variants with types: {cls_type.upper()}")

        return out

    def _get_classifier_builder(self, variants, classifier_type):
        """根据分类器类型获取对应的构建器"""
        if classifier_type == "lda":
            return LDAClassifierBuilder(reg_alpha=self.lda_reg_alpha, device=self.device)

        elif classifier_type == "qda":
            return QDAClassifierBuilder(
                qda_reg_alpha1=self.qda_reg_alpha1,
                qda_reg_alpha2=self.qda_reg_alpha2,
                qda_reg_alpha3=self.qda_reg_alpha3,
                device=self.device,
            )

        elif classifier_type == "sgd":
            if self.cached_Z is None:
                for name, stats in variants.items():
                    if stats:  # 找到第一个非空的variant
                        feature_dim = list(stats.values())[0].mean.size(0)
                        self.cached_Z = torch.randn(1024, feature_dim)
                        break
            return SGDClassifierBuilder(cached_Z=self.cached_Z, device=self.device, max_steps=100, lr=5e-4)
        
        elif classifier_type == "ls":
            if self.cached_Z is None:
                # 初始化cached_Z
                for name, stats in variants.items():
                    if stats:
                        feature_dim = list(stats.values())[0].mean.size(0)
                        self.cached_Z = torch.randn(50000, feature_dim)
                        break
            return LeastSquaresClassifierBuilder(cached_Z=self.cached_Z, device=self.device, reg_lambda=1e-3)

        elif classifier_type == "ncm":
            return None  # 暂时不支持 NCM 分类器
        
        else:
            raise ValueError(f"Unsupported classifier type: {classifier_type}")
