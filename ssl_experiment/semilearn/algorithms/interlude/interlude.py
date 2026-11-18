import torch
import torch.nn.functional as F
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.fixmatch import FixMatch
from semilearn.algorithms.utils import SSL_Argument, str2bool

def _unwrap(m):
    return m.module if hasattr(m, 'module') else m

def _fusion_target(net):
    net = _unwrap(net)
    if hasattr(net, 'backbone'):
        net = net.backbone
    if hasattr(net, 'fusion_hook_module'):
        return net.fusion_hook_module
    if hasattr(net, 'blocks'):
        return net.blocks[-1]
    if hasattr(net, 'block3'):
        return net.block3
    raise ValueError('InterLUDE: cannot find a module to attach embedding fusion to')

@ALGORITHMS.register('interlude')
class InterLUDE(FixMatch):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)
        self.lambda_rel = args.lambda_rel
        self.fusion_strength = args.fusion_strength
        self.rel_T = args.rel_T
        self.fusion_at_eval = args.fusion_at_eval
        self._fuse_now = False
        target = _fusion_target(self.model)
        target.register_forward_pre_hook(self._fusion_pre_hook)
        if self.fusion_at_eval:
            _fusion_target(self.ema_model).register_forward_pre_hook(self._fusion_pre_hook_eval)

    def _fuse(self, x):
        if self.fusion_strength <= 0 or x.shape[0] < 2:
            return x
        return (1 - self.fusion_strength) * x + self.fusion_strength * torch.roll(x, shifts=-1, dims=0)

    def _fusion_pre_hook(self, module, inputs):
        if self._fuse_now or (self.fusion_at_eval and not module.training):
            return (self._fuse(inputs[0]),) + tuple(inputs[1:])
        return None

    def _fusion_pre_hook_eval(self, module, inputs):
        return (self._fuse(inputs[0]),) + tuple(inputs[1:])

    def train_step(self, x_lb, x_lb_s, y_lb, x_ulb_w, x_ulb_s, y_ulb=None):
        B = y_lb.shape[0]
        mu = x_ulb_w.shape[0] // B
        x_ulb_w, x_ulb_s = x_ulb_w[:B * mu], x_ulb_s[:B * mu]
        G = 2 + 2 * mu
        img_shape = x_lb.shape[1:]

        inputs = torch.cat([x_lb.unsqueeze(1), x_lb_s.unsqueeze(1),
                            x_ulb_w.view(B, mu, *img_shape), x_ulb_s.view(B, mu, *img_shape)], dim=1)
        inputs = inputs.flatten(0, 1)

        with self.amp_cm():
            self._fuse_now = True
            try:
                outputs = self.model(inputs)
            finally:
                self._fuse_now = False
            C = outputs['logits'].shape[-1]
            logits = outputs['logits'].view(B, G, C)
            feats = outputs['feat'].view(B, G, -1)
            logits_l_w, logits_l_s = logits[:, 0], logits[:, 1]
            logits_u_w = logits[:, 2:2 + mu].reshape(B * mu, C)
            logits_u_s = logits[:, 2 + mu:].reshape(B * mu, C)
            feat_dict = {'x_lb': feats[:, 0], 'x_ulb_w': feats[:, 2:2 + mu].reshape(B * mu, -1),
                         'x_ulb_s': feats[:, 2 + mu:].reshape(B * mu, -1)}

            sup_loss = self.ce_loss(logits_l_w, y_lb, reduction='mean')

            probs_u_w = self.compute_prob(logits_u_w.detach())
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_u_w, softmax_x_ulb=False)
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", logits=probs_u_w,
                                          use_hard_label=self.use_hard_label, T=self.T, softmax=False)
            unsup_loss = self.consistency_loss(logits_u_s, pseudo_label, 'ce', mask=mask)

            if self.lambda_rel > 0:
                l_diff = (torch.softmax(logits_l_w.detach().float() / self.rel_T, -1)
                          - torch.softmax(logits_l_s.detach().float() / self.rel_T, -1))
                u_diff = (torch.softmax(logits_u_w.float(), -1) - torch.softmax(logits_u_s.float(), -1))
                u_diff = u_diff.view(B, mu, C).mean(1)
                rel_loss = F.mse_loss(u_diff, l_diff, reduction='mean')
            else:
                rel_loss = torch.zeros((), device=logits.device)

            total_loss = sup_loss + self.lambda_u * unsup_loss + self.lambda_rel * rel_loss

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), unsup_loss=unsup_loss.item(),
                                         rel_loss=rel_loss.item(), total_loss=total_loss.item(),
                                         util_ratio=mask.float().mean().item())
        return out_dict, log_dict

    @staticmethod
    def get_argument():
        return FixMatch.get_argument() + [
            SSL_Argument('--lambda_rel', float, 1.0),
            SSL_Argument('--fusion_strength', float, 0.1),
            SSL_Argument('--rel_T', float, 1.0),
            SSL_Argument('--fusion_at_eval', str2bool, False),
        ]
