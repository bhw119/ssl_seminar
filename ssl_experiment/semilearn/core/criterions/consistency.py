import torch 
import torch.nn as nn 
from torch.nn import functional as F

from .cross_entropy import ce_loss

def consistency_loss(logits, targets, name='ce', mask=None):

    assert name in ['ce', 'mse', 'kl']
    if name == 'mse':
        probs = torch.softmax(logits, dim=-1)
        loss = F.mse_loss(probs, targets, reduction='none').mean(dim=1)
    elif name == 'kl':
        loss = F.kl_div(F.log_softmax(logits / 0.5, dim=-1), F.softmax(targets / 0.5, dim=-1), reduction='none')
        loss = torch.sum(loss * (1.0 - mask).unsqueeze(dim=-1).repeat(1, torch.softmax(logits, dim=-1).shape[1]), dim=1)
    else:
        loss = ce_loss(logits, targets, reduction='none')

    if mask is not None and name != 'kl':
        loss = loss * mask

    return loss.mean()

class ConsistencyLoss(nn.Module):
    def forward(self, logits, targets, name='ce', mask=None):
        return consistency_loss(logits, targets, name, mask)

class GCELoss(nn.Module):
    def __init__(self, num_classes, q=0.7):
        super(GCELoss, self).__init__()
        self.num_classes = num_classes
        self.q = q

    def forward(self, pred, labels, mask):
        pred = F.softmax(pred.float(), dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = (1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
        if mask is not None:
            loss = loss * mask
        return loss.mean()
