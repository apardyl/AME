import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2):
        super().__init__()
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.bce = nn.BCELoss(reduction="none")
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, out, target, apply_sigmoid=True):
        if apply_sigmoid:
            ce_loss = self.bce_with_logits_loss(out, target)
        else:
            ce_loss = self.bce(out, target)
        pt = torch.exp(-ce_loss)

        pos = target == 1
        ce_loss[pos] *= self.alpha
        ce_loss[~pos] *= (1 - self.alpha)

        return torch.mean((1.0 - pt) ** self.gamma * ce_loss)
