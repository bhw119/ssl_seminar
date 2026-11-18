import torch
import numpy as np
from semilearn.core.hooks import Hook

class MaskingHook(Hook):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
    
    def update(self, *args, **kwargs):
        pass
    
    @torch.no_grad()
    def masking(self, algorithm, 
                      logits_x_lb=None, logits_x_ulb=None, idx_lb=None, idx_ulb=None,
                      softmax_x_lb=True, softmax_x_ulb=True,
                      *args, **kwargs):
        raise NotImplementedError

class FixedThresholdingHook(MaskingHook):

    @torch.no_grad()
    def masking(self, algorithm, logits_x_ulb, softmax_x_ulb=True, *args, **kwargs):
        if softmax_x_ulb:
            probs_x_ulb = algorithm.compute_prob(logits_x_ulb.detach())
        else:
            probs_x_ulb = logits_x_ulb.detach()
        max_probs, _ = torch.max(probs_x_ulb, dim=-1)
        mask = max_probs.ge(algorithm.p_cutoff).to(max_probs.dtype)
        return mask

