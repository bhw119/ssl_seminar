import torch 
import numpy as np
from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.utils import SSL_Argument, str2bool

@ALGORITHMS.register('pimodel')
class PiModel(AlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None, **kwargs):
        super().__init__(args, net_builder, tb_log, logger, **kwargs)
        self.init(unsup_warm_up=args.unsup_warm_up)
    
    def init(self, unsup_warm_up=0.4):
        self.unsup_warm_up = unsup_warm_up 

    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s):
        with self.amp_cm():

            outs_x_lb = self.model(x_lb)
            logits_x_lb = outs_x_lb['logits']
            feats_x_lb = outs_x_lb['feat']            

            self.bn_controller.freeze_bn(self.model)
            outs_x_ulb_w = self.model(x_ulb_w)
            logits_x_ulb_w = outs_x_ulb_w['logits']
            feats_x_ulb_w = outs_x_ulb_w['feat']    
            outs_x_ulb_s = self.model(x_ulb_s)
            logits_x_ulb_s = outs_x_ulb_s['logits']
            feats_x_ulb_s = outs_x_ulb_s['feat']    
            self.bn_controller.unfreeze_bn(self.model)

            feat_dict = {'x_lb': feats_x_lb, 'x_ulb_w': feats_x_ulb_w, 'x_ulb_s': feats_x_ulb_s}

            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
            unsup_loss = self.consistency_loss(logits_x_ulb_s,
                                               self.compute_prob(logits_x_ulb_w.detach()),
                                               'mse')
            unsup_warmup = np.clip(self.it / (self.unsup_warm_up * self.num_train_iter),  a_min=0.0, a_max=1.0)
            total_loss = sup_loss + self.lambda_u * unsup_loss * unsup_warmup

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                         unsup_loss=unsup_loss.item(), 
                                         total_loss=total_loss.item())
        return out_dict, log_dict
        

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--unsup_warm_up', float, 0.4, 'warm up ratio for unsupervised loss'),
        ]
