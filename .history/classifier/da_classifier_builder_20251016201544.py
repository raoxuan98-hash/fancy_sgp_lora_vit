# classifier/gaussian_classifier_builder.py
import torch
from classifier.gaussian_classifier import RegularizedGaussianClassifier, LinearLDAClassifier
from classifier.base_classifier_builder import BaseClassifierBuilder

class LDAClassifierBuilder(BaseClassifierBuilder):
    def __init__(self, reg_alpha=0.1, reg_type="shrinkage", device="cuda"):
        self.reg_alpha = reg_alpha
        self.reg_type = reg_type
        self.device = device

    def build(self, stats_dict):
        priors = {cid: 1.0 / len(stats_dict) for cid in stats_dict}
        model = LinearLDAClassifier(
            stats_dict=stats_dict,
            class_priors=priors,
            reg_alpha=self.reg_alpha,
            reg_type=self.reg_type
        ).to(self.device)
        return model


class QDAClassifierBuilder(BaseClassifierBuilder):
    def __init__(self, reg_alpha=0.1, reg_type="shrinkage", device="cuda"):
        self.reg_alpha = reg_alpha
        self.reg_type = reg_type
        self.device = device

    def build(self, stats_dict):
        priors = {cid: 1.0 / len(stats_dict) for cid in stats_dict}
        model = RegularizedGaussianClassifier(
            stats_dict=stats_dict,
            class_priors=priors,
            mode="qda",
            reg_alpha=self.reg_alpha,
            reg_type=self.reg_type
        ).to(self.device)
        return model
