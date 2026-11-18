import torch 
import torch.nn as nn

from torch.nn import functional as F

def ce_loss(logits, targets, class_weights=None, reduction='none'):
    if logits.shape == targets.shape:
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        if class_weights is not None:
            nll_loss = nll_loss * torch.matmul(targets, class_weights.unsqueeze(0).T)
        if reduction == 'none':
            return nll_loss
        else:
            return nll_loss.mean()
    else:
        log_pred = F.log_softmax(logits, dim=-1)
        if class_weights is not None:
            return F.nll_loss(log_pred, targets, weight=class_weights, reduction=reduction)
        else:
            return F.nll_loss(log_pred, targets, reduction=reduction)

class CELoss(nn.Module):
    def forward(self, logits, targets, reduction='none', class_weights=None):
        return ce_loss(logits, targets, class_weights, reduction)

