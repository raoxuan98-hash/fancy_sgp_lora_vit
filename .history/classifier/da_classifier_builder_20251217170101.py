# classifier/gaussian_classifier_builder.py
import torch
import time
import logging
from classifier.gaussian_classifier import RegularizedGaussianDA, LinearLDAClassifier, LowRankGaussianDA
from classifier.base_classifier_builder import BaseClassifierBuilder


def log_time_usage(operation_name: str, start_time: float, end_time: float):
    """记录时间损耗情况"""
    elapsed_time = end_time - start_time
    logging.info(f"[Time] {operation_name}: {elapsed_time:.4f}s")

class LDAClassifierBuilder(BaseClassifierBuilder):
    def __init__(self, reg_alpha=0.3, device="cuda"):
        self.reg_alpha = reg_alpha
        self.device = device

    def build(self, stats_dict):
        start_time = time.time()
        
        priors = {cid: 1.0 / len(stats_dict) for cid in stats_dict}
        model = LinearLDAClassifier(
            stats_dict=stats_dict,
            class_priors=priors,
            lda_reg_alpha=self.reg_alpha
        ).to(self.device)
        
        end_time = time.time()
        log_time_usage("LDA Classifier build", start_time, end_time)
        
        return model


class QDAClassifierBuilder(BaseClassifierBuilder):
    def __init__(
        self,
        qda_reg_alpha1=None,
        qda_reg_alpha2=None,
        qda_reg_alpha3=None,
        low_rank=True,
        rank = 64,
        device="cuda",
    ):
        self.qda_reg_alpha1 = qda_reg_alpha1
        self.qda_reg_alpha2 = qda_reg_alpha2
        self.qda_reg_alpha3 = qda_reg_alpha3
        self.device = device
        self.low_rank = low_rank
        self.rank = rank

    def build(self, stats_dict):
        start_time = time.time()
        
        priors = {cid: 1.0 / len(stats_dict) for cid in stats_dict}
        if self.low_rank:
            model = LowRankGaussianDA(
                stats_dict=stats_dict,
                class_priors=priors,
                rank = self.rank,
                qda_reg_alpha1=self.qda_reg_alpha1,
                qda_reg_alpha2=self.qda_reg_alpha2,
                qda_reg_alpha3=self.qda_reg_alpha3,
            ).to(self.device)
        else:
            model = RegularizedGaussianDA(
                stats_dict=stats_dict,
                class_priors=priors,
                qda_reg_alpha1=self.qda_reg_alpha1,
                qda_reg_alpha2=self.qda_reg_alpha2,
                qda_reg_alpha3=self.qda_reg_alpha3,
            ).to(self.device)
        
        end_time = time.time()
        log_time_usage("QDA Classifier build", start_time, end_time)
        
        return model
