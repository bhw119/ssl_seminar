import torch
from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool
import numpy as np

@ALGORITHMS.register('fixmatch')
class FixMatch(AlgorithmBase):

    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 
        self.init(T=args.T, p_cutoff=args.p_cutoff, hard_label=args.hard_label)

    
    def init(self, T, p_cutoff, hard_label=True):
        self.T = T
        self.p_cutoff = p_cutoff
        self.use_hard_label = hard_label
    
    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FixedThresholdingHook(), "MaskingHook")
        super().set_hooks()

    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s, x_ulb_s2=None, y_ulb=None):
        num_lb = y_lb.shape[0]
        logits_x_ulb_s2 = None

        with self.amp_cm():
            if self.use_cat:
                if x_ulb_s2 is not None:
                    inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s, x_ulb_s2))
                    outputs = self.model(inputs)
                    logits_x_lb = outputs['logits'][:num_lb]
                    logits_x_ulb_w, logits_x_ulb_s, logits_x_ulb_s2 = outputs['logits'][num_lb:].chunk(3)
                    feats_x_lb = outputs['feat'][:num_lb]
                    feats_x_ulb_w, feats_x_ulb_s, feats_x_ulb_s2 = outputs['feat'][num_lb:].chunk(3)
                else:
                    inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                    outputs = self.model(inputs)
                    logits_x_lb = outputs['logits'][:num_lb]
                    logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
                    feats_x_lb = outputs['feat'][:num_lb]
                    feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][num_lb:].chunk(2)

            else:
                outs_x_lb = self.model(x_lb) 
                logits_x_lb = outs_x_lb['logits']
                feats_x_lb = outs_x_lb['feat']
                outs_x_ulb_s = self.model(x_ulb_s)
                logits_x_ulb_s = outs_x_ulb_s['logits']
                feats_x_ulb_s = outs_x_ulb_s['feat']
                outs_x_ulb_s2 = None
                logits_x_ulb_s2 = None
                feats_x_ulb_s2 = None
                if x_ulb_s2 is not None:
                    outs_x_ulb_s2 = self.model(x_ulb_s2)
                    logits_x_ulb_s2 = outs_x_ulb_s2['logits']
                    feats_x_ulb_s2 = outs_x_ulb_s2['feat']
                with torch.no_grad():
                    outs_x_ulb_w = self.model(x_ulb_w)
                    logits_x_ulb_w = outs_x_ulb_w['logits']
                    feats_x_ulb_w = outs_x_ulb_w['feat']
            if x_ulb_s2 is not None:
                 feat_dict = {'x_lb':feats_x_lb, 'x_ulb_w':feats_x_ulb_w, 'x_ulb_s':feats_x_ulb_s, 'x_ulb_s2':feats_x_ulb_s2}
            else:
                feat_dict = {'x_lb':feats_x_lb, 'x_ulb_w':feats_x_ulb_w, 'x_ulb_s':feats_x_ulb_s} 

            sup_loss = self.ce_loss(logits=logits_x_lb, targets=y_lb, reduction='mean')
            
            probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())
            
            if self.registered_hook("DistAlignHook"):
                probs_x_ulb_w = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs_x_ulb_w.detach())

            
            if logits_x_ulb_s2 is not None:
                pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                logits=logits_x_ulb_w.detach(),
                                use_hard_label=self.use_hard_label,
                                T=self.T,
                                softmax=True)
                mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False)
                pseudo_label = torch.cat([pseudo_label, pseudo_label], dim=0)
                logits_x_ulb_s = torch.cat([logits_x_ulb_s, logits_x_ulb_s2], dim=0)
                mask = torch.cat([mask, mask], dim=0)
            else:
                mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False)
                pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                logits=probs_x_ulb_w,
                                use_hard_label=self.use_hard_label,
                                T=self.T,
                                softmax=True)
                

            unsup_loss = self.consistency_loss(logits_x_ulb_s,
                                               pseudo_label,
                                               name='ce',
                                               mask=mask)

            total_loss = sup_loss + self.lambda_u * unsup_loss

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict, y_ulb=y_ulb)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                         unsup_loss=unsup_loss.item(), 
                                         total_loss=total_loss.item(), 
                                         util_ratio=mask.float().mean().item())
        return out_dict, log_dict
        

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--p_cutoff', float, 0.95),
        ]
