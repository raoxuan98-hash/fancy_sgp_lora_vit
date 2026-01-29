# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import logging
from typing import Dict, Union, List

from classifier.da_classifier_builder import LDAClassifierBuilder, QDAClassifierBuilder
from classifier.ls_classifier_builder import LeastSquaresClassifierBuilder
from classifier.sgd_classifier_builder import SGDClassifierBuilder


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

    def build_classifiers(self, variants: Dict[str, Dict[int, object]], classifier_type: Union[str, List[str]] = ["lda", "qda", "sgd"]) -> Dict[str, Dict[str, nn.Module]]:
        """
        构建分类器
        
        Args:
            variants: 各variant的高斯统计信息
            classifier_type: 分类器类型，可以是字符串或字符串列表
            
        Returns:
            Dict[str, Dict[str, nn.Module]]: 外层key为variant名称，内层key为分类器类型
        """
        # 统一处理为列表形式
        if isinstance(classifier_type, str):
            classifier_types = [classifier_type]
        else:
            classifier_types = classifier_type
        
        out = {}

        # 为每种分类器类型构建分类器
        for cls_type in classifier_types:
            classifier_builder = self._get_classifier_builder(variants, cls_type)
            
            for name, stats in variants.items():
                cls_name = name + " + " + cls_type.upper()
                out[cls_name] = classifier_builder.build(stats)
                logging.info(f"[Classifier] {name}: {cls_type.upper()}")
        
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
                # 初始化cached_Z（这里需要根据实际情况调整）
                for name, stats in variants.items():
                    if stats:  # 找到第一个非空的variant
                        feature_dim = list(stats.values())[0].mean.size(0)
                        self.cached_Z = torch.randn(1024, feature_dim)
                        break
            return SGDClassifierBuilder(cached_Z=self.cached_Z, device=self.device, epochs=5, lr=0.01)
        
        elif classifier_type == "ls":
            if self.cached_Z is None:
                # 初始化cached_Z
                for name, stats in variants.items():
                    if stats:
                        feature_dim = list(stats.values())[0].mean.size(0)
                        self.cached_Z = torch.randn(50000, feature_dim)
                        break
            return LeastSquaresClassifierBuilder(cached_Z=self.cached_Z, device=self.device, reg_lambda=1e-3)
        
        else:
            raise ValueError(f"Unsupported classifier type: {classifier_type}")