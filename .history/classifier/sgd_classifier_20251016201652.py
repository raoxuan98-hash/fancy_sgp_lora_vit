# classifier/sgd_classifier_builder.py
import torch
import torch.nn as nn
from classifier.base_classifier_builder import BaseClassifierBuilder

class SGDClassifierBuilder(BaseClassifierBuilder):
    def __init__(self, cached_Z, device="cuda", epochs=5, lr=0.01):
        self.cached_Z = cached_Z
        self.device = device
        self.epochs = epochs
        self.lr = lr

    def build(self, stats_dict):
        fc = nn.Linear(list(stats_dict.values())[0].mean.size(0), len(stats_dict)).to(self.device)
        opt = torch.optim.SGD(fc.parameters(), lr=self.lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs, eta_min=self.lr/10)
        Z = self.cached_Z.to(self.device)

        # 构造样本
        samples, labels = [], []
        for cid, gs in stats_dict.items():
            mu, L = gs.mean.to(self.device), gs.L.to(self.device)
            X = mu + Z[:1024] @ L.t()
            y = torch.full((1024,), int(cid), device=self.device)
            samples.append(X); labels.append(y)
        X = torch.cat(samples); Y = torch.cat(labels)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        for e in range(self.epochs):
            perm = torch.randperm(X.size(0), device=self.device)
            for i in range(0, X.size(0), 64):
                idx = perm[i:i+64]
                opt.zero_grad()
                loss = criterion(fc(X[idx]), Y[idx])
                loss.backward(); opt.step()
            sch.step()

        return fc.cpu()

