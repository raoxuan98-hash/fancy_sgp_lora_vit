# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import logging
from typing import Dict

from classifier.da_classifier_builder import LDAClassifierBuilder, QDAClassifierBuilder
from classifier.ls_classifier_builder import LeastSquaresClassifierBuilder
from classifier.sgd_classifier_builder import SGDClassifierBuilder


class ClassifierReconstructor:
    """
    统一的分类器重构模块：
    输入各 variant 的高斯统计，输出 {variant_name: nn.Module}
    """
    def __init__(self, device="cuda", cached_Z=None):
        self.device = device
        self.cached_Z = cached_Z

    def build_classifiers(self, variants: Dict[str, Dict[int, object]], classifier_type="sgd", epochs=5,
                          reg_alpha=0.1, reg_type="shrinkage") -> Dict[str, nn.Module]:
        out = {}

        if classifier_type == "lda":
            classifier_builder = LDAClassifierBuilder(reg_alpha=reg_alpha, reg_type=reg_type, device=self.device)
        elif classifier_type == "qda":
            classifier_builder = QDAClassifierBuilder(reg_alpha=reg_alpha, reg_type=reg_type, device=self.device)
        elif classifier_type == "sgd":
            if self.cached_Z is None:
                for name, stats in variants.items():
                    feature_dim = list(stats.values())[0].mean.size(0)
                    break
                self.cached_Z = torch.randn(50000, feature_dim)
            classifier_builder = SGDClassifierBuilder(cached_Z=self.cached_Z, device=self.device, epochs=epochs, lr=0.01)
        elif classifier_type == "ls":
            classifier_builder = LeastSquaresClassifierBuilder(cached_Z=self.cached_Z, device=self.device, reg_lambda=1e-3)

        for name, stats in variants.items():
            if not stats:
                continue

            out[name] = classifier_builder.build(stats)
            logging.info(f"[Classifier] {name}: {classifier_type.upper()}")
        logging.info(f"[Classifier] Built {len(out)} classifiers.")
        return out
