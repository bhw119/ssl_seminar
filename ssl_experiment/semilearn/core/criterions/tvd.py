import torch 
import torch.nn as nn 
from torch.nn import functional as F

def tailr_loss(logits, target, epsilon, min_weight, gamma, probs_model, reduction='none', reverse=False, mask=None): 
    probs_model = torch.nn.functional.softmax(logits, dim=-1)
    lprobs = probs_model.log()

    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)

    nll_loss = -lprobs.gather(dim=-1, index=target)

    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    weight_theta_hat = probs_model.gather(dim=-1, index=target)
    if reverse:
        weight_theta_hat = 1 - 0.5 * probs_model.gather(dim=-1, index=target)

    with torch.no_grad():
        weight_theta_hat = (weight_theta_hat.log() - (gamma + (1 - gamma) * weight_theta_hat).log()).exp()
        weight_theta_hat = torch.clamp(weight_theta_hat, min=min_weight, max=1.0)

    tailr_loss = weight_theta_hat * nll_loss
    if reduction != 'none':
        nll_loss = nll_loss.mean()
        smooth_loss = smooth_loss.mean()
        tailr_loss = tailr_loss.mean()

    eps_i = epsilon / (lprobs.size(-1))

    loss = (1. - epsilon) * tailr_loss + eps_i * smooth_loss

    if mask is not None:
        loss = loss * mask
        return loss.mean()
    return loss

class TVDLoss(nn.Module):
    def forward(self, logits, targets, epsilon=0.2, min_weight=0.001, gamma=0.8, probs_model=None, reduction='none', reverse=False, name='', mask=None):
        return tailr_loss(logits, targets, epsilon=epsilon, min_weight=min_weight, gamma=gamma, probs_model=probs_model, reduction=reduction, reverse=reverse, mask=mask)

