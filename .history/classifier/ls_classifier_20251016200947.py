# classifier/ls_classifier_builder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from classifier.base_classifier_builder import BaseClassifierBuilder

class LeastSquaresClassifierBuilder(BaseClassifierBuilder):
    def __init__(self, cached_Z, reg_lambda=1e-3, device="cuda"):
        self.cached_Z = cached_Z
        self.device = device
        self.reg_lambda = reg_lambda

    def build(self, stats_dict):
        d = list(stats_dict.values())[0].mean.size(0)
        C = len(stats_dict)
        Z = self.cached_Z.to(self.device)
        Xs, Ys = [], []
        for cid, gs in stats_dict.items():
            mu, L = gs.mean.to(self.device), gs.L.to(self.device)
            X = mu + Z[:1024] @ L.t()
            y = torch.full((1024,), int(cid), device=self.device)
            Xs.append(X); Ys.append(y)
        X, Y = torch.cat(Xs), torch.cat(Ys)
        Xn = F.normalize(X, dim=1)
        Y_oh = F.one_hot(Y, num_classes=C).float()
        reg = self.reg_lambda * torch.eye(d, device=self.device)
        W = torch.linalg.solve(Xn.T @ Xn + reg, Xn.T @ Y_oh)
        model = nn.Sequential(nn.Linear(d, C, bias=False)).to(self.device)
        model[0].weight.data = W.T.clone()
        return model.cpu()
