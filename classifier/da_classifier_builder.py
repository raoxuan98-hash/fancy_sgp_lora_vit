# classifier/gaussian_classifier_builder.py
import torch
from classifier.gaussian_classifier import RegularizedGaussianDA, LinearLDAClassifier
from classifier.base_classifier_builder import BaseClassifierBuilder

class LDAClassifierBuilder(BaseClassifierBuilder):
    def __init__(self, reg_alpha=0.3, device="cuda"):
        self.reg_alpha = reg_alpha
        self.device = device

    def build(self, stats_dict):
        priors = {cid: 1.0 / len(stats_dict) for cid in stats_dict}
        model = LinearLDAClassifier(
            stats_dict=stats_dict,
            class_priors=priors,
            lda_reg_alpha=self.reg_alpha
        ).to(self.device)
        return model


class QDAClassifierBuilder(BaseClassifierBuilder):
    def __init__(
        self,
        qda_reg_alpha1=0.2,
        qda_reg_alpha2=0.2,
        qda_reg_alpha3=0.2,
        device="cuda",
    ):
        self.qda_reg_alpha1 = qda_reg_alpha1
        self.qda_reg_alpha2 = qda_reg_alpha2
        self.qda_reg_alpha3 = qda_reg_alpha3
        self.device = device

    def build(self, stats_dict):
        priors = {cid: 1.0 / len(stats_dict) for cid in stats_dict}
        model = RegularizedGaussianDA(
            stats_dict=stats_dict,
            class_priors=priors,
            qda_reg_alpha1=self.qda_reg_alpha1,
            qda_reg_alpha2=self.qda_reg_alpha2,
            qda_reg_alpha3=self.qda_reg_alpha3,
        ).to(self.device)
        return model
