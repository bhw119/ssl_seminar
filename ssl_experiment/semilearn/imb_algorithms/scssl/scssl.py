import os
import torch
import torch.nn as nn
import numpy as np
from inspect import signature
import copy
from semilearn.core import ImbAlgorithmBase
from semilearn.core.utils import get_dataset, get_data_loader, send_model_cuda, IMB_ALGORITHMS
from semilearn.algorithms.utils import SSL_Argument
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, top_k_accuracy_score
from .utils import MyNet, UpdateBiasThresHook, EstimateDistHook, CheckpointHook

@IMB_ALGORITHMS.register('scssl')
class SCSSL(ImbAlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None, **kwargs):
        self.imb_init(t_step=args.t_step, coe=args.coe)

        super(SCSSL, self).__init__(args, net_builder, tb_log, logger, **kwargs)

        lb_class_dist = [0 for _ in range(self.num_classes)]
        for c in  self.dataset_dict['train_lb'].targets:
            lb_class_dist[c] += 1
        lb_class_dist = np.array(lb_class_dist, dtype=float)

        self.lb_class_dist = torch.from_numpy(np.min(lb_class_dist) / lb_class_dist)
        self.p_hat_lb = torch.from_numpy(lb_class_dist / lb_class_dist.sum()).float().cuda()
        self.p_hat_ulb = None

        self.minor_classes = torch.from_numpy(np.where(lb_class_dist < np.median(lb_class_dist))[0])

        self.lb_class_nums = lb_class_dist

        self.est_iters = 10240
        self.est_dist = None
        self.target_ulb_class_dist = None
        self.ulb_class_dist = None
        self.pseudo_label_list = None

        self.model = MyNet(self.model, num_classes=self.num_classes)
        self.ema_model = MyNet(self.ema_model, num_classes=self.num_classes)
        self.ema_model.load_state_dict(self.model.state_dict())
        self.optimizer, self.scheduler = self.set_optimizer()
 
        self.logits_record = torch.zeros(self.num_classes, self.num_classes)  
        self.logits_counts = torch.zeros(self.num_classes) 
        self.logits_record2 = torch.zeros(self.num_classes, self.num_classes) 
        self.logits_counts2 = torch.zeros(self.num_classes)  
        self.logits_counts2_debias = torch.zeros(self.num_classes)  
        self.logits_counts2_mask = torch.zeros(self.num_classes)
        self.logits_counts3 = torch.zeros(self.num_classes)
        self.logits_counts3_mask = torch.zeros(self.num_classes)

        self.class_weights = None

        self.head2_ulb_mask = True
        self.head3_ulb_mask = True
        self.head2_desc = False
        self.head3_desc = False
        self.head2_t = None
        self.head3_t = None
        self.tau1 = 2.0
        self.tau2 = 4.0
        self.uloss_scale = 2.0
        
        if args.dataset in ['cifar10', 'stl10']:
            if args.dataset in ['stl10']:
                self.tau1 = 1.0
                self.tau2 = 2.0
            self.dist_config = {
                None: (False, False, False, False, (0.95, 0.8), (0.95, 0.8)),
                'long': (True, True, False, True, (0.95, 0.95), (0.95, 0.8)),
                'long_inverse': (False, False, False, False, (0.95, 0.75), 'cd'),
                'uniform': (False, True, False, True, (0.95, 0.85), (0.95, 0.6)), 
                'gaussian':(True, True, False, True, (0.95, 0.95), (0.95, 0.8)),
                'gaussian_inverse': (True, True, False, True, (0.95, 0.95), (0.95, 0.8)),
            }
        elif args.dataset in ['imagenet127-32', 'imagenet127-64'] :
            self.tau1 = 0.15
            self.tau2 = 0.3
            self.dist_config = { 
                None: (False, False, False, False, (0.95, 0.8), (0.95, 0.8)),
                'long': (False, False, False, True, (0.95, 0.95), (0.95, 0.8)),
            }
        else:
            self.tau1 = 2.0
            self.tau2 = 4.0
            self.dist_config = {
                None: (False, False, False, False, (0.95, 0.95), (0.95, 0.95)),
                'long': (False, False, False, True, (0.95, 0.95), (0.95, 0.95)),
                'long_inverse': (False, False, False, True, (0.95, 0.90), (0.95, 0.6)),
                'uniform': (False, True, False, True, (0.95, 0.95), (0.95, 8)), 
                'gaussian':(True, True, False, True, (0.95, 0.95), (0.95, 0.8)),
                'gaussian_inverse': (True, True, False, True, (0.95, 0.95), (0.95, 0.8)),
            }

        self.bias_ema = torch.zeros(self.num_classes).cuda()
        self.bias_ema2 = torch.zeros(self.num_classes).cuda()
        self.bias_white = torch.zeros(self.num_classes).cuda()
        self.bias_white2 = torch.zeros(self.num_classes).cuda()

    def imb_init(self, t_step=0.01, coe=1.0):
        self.t_step = t_step
        self.coe = coe

    def set_hooks(self):
        super().set_hooks()
        self.register_hook(EstimateDistHook(), None, "HIGH")
        self.register_hook(CheckpointHook(), None, "HIGH")
        self.register_hook(UpdateBiasThresHook(), None, "HIGH")

    def process_batch(self, **kwargs):
        input_args = signature(super().train_step).parameters
        input_args = list(input_args.keys())
        return super().process_batch(input_args=input_args, **kwargs)
    
    def train_est(self):
        self.model.train()

        for epoch in range(0, int(self.est_iters/1024)):
            self.epoch = epoch
            if self.it >= self.est_iters:
                break
            self.call_hook("before_train_epoch")
            for data_lb, data_ulb in zip(self.loader_dict['train_lb'],
                                         self.loader_dict['train_ulb']):
                if self.it >= self.est_iters:
                    break
                self.call_hook("before_train_step")
                self.out_dict, self.log_dict = self.train_step(**self.process_batch(**data_lb, **data_ulb))
                self.call_hook("after_train_step")
                self.it += 1
            self.call_hook("after_train_epoch")

    def train_step(self, *args, **kwargs):
        with self.amp_cm():
            return self._scssl_train_step(*args, **kwargs)

    def _scssl_train_step(self, *args, **kwargs):
        
        out_dict, log_dict = super().train_step(*args, **kwargs)

        feats_x_lb = out_dict['feat']['x_lb']
        feats_x_ulb_w = out_dict['feat']['x_ulb_w']
        feats_x_ulb_s = out_dict['feat']['x_ulb_s']
        feats_x_ulb_s2 = out_dict['feat'].get('x_ulb_s2', None)
        if isinstance(feats_x_ulb_s, list):
            feats_x_ulb_s = feats_x_ulb_s[0]
         
        logits_x_lb = (self.model.module if hasattr(self.model, 'module') else self.model).aux_classifier(feats_x_lb)
        logits_x_ulb_s = (self.model.module if hasattr(self.model, 'module') else self.model).aux_classifier(feats_x_ulb_s) if feats_x_ulb_s2 is None else [(self.model.module if hasattr(self.model, 'module') else self.model).aux_classifier(feats_x_ulb_s), (self.model.module if hasattr(self.model, 'module') else self.model).aux_classifier(feats_x_ulb_s2)]
        with torch.no_grad():
            logits_x_ulb_w = (self.model.module if hasattr(self.model, 'module') else self.model).aux_classifier(feats_x_ulb_w)

        head2_loss = self.compute_head2_loss(
            logits_x_lb=logits_x_lb, 
            y_lb=kwargs['y_lb'],
            logits_x_ulb_w=logits_x_ulb_w,
            logits_x_ulb_s=logits_x_ulb_s,
            y_ulb=out_dict['y_ulb']
            )
        out_dict['loss'] += head2_loss 
        log_dict['train/2ndhead_loss'] = head2_loss.item()

        logits_x_lb_2 = (self.model.module if hasattr(self.model, 'module') else self.model).aux_classifier2(feats_x_lb)
        logits_x_ulb_s_2 = (self.model.module if hasattr(self.model, 'module') else self.model).aux_classifier2(feats_x_ulb_s) if feats_x_ulb_s2 is None else [(self.model.module if hasattr(self.model, 'module') else self.model).aux_classifier2(feats_x_ulb_s), (self.model.module if hasattr(self.model, 'module') else self.model).aux_classifier2(feats_x_ulb_s2)]
        with torch.no_grad():
            logits_x_ulb_w_2 = (self.model.module if hasattr(self.model, 'module') else self.model).aux_classifier2(feats_x_ulb_w)
        
        head3_loss = self.compute_head3_loss(
            logits_x_lb=logits_x_lb_2, 
            y_lb=kwargs['y_lb'],
            logits_x_ulb_w=logits_x_ulb_w_2,
            logits_x_ulb_s=logits_x_ulb_s_2
            )
        out_dict['loss'] += head3_loss 
        log_dict['train/3rdhead_loss'] = head3_loss.item()

        return out_dict, log_dict
    
    def evaluate(self, eval_dest='eval', out_key='logits', return_logits=False):
        results = {}  
        results.update(self.evaluate_debias())
        return results

    def update_dist_params(self):
        config = self.dist_config.get(self.est_dist, self.dist_config[None])

        (self.head2_ulb_mask, self.head3_ulb_mask,
        self.head2_desc, self.head3_desc,
        head2_t, head3_t) = config

        if type(head2_t) == str:
            self.head2_t = 1 - self.lb_class_dist
        else:
            self.head2_t = torch.full((self.num_classes,), head2_t[0])
            self.head2_t[self.minor_classes] = head2_t[1]

        if type(head3_t) == str:
            self.head3_t = 1 - self.lb_class_dist
        else:
            self.head3_t = torch.full((self.num_classes,), head3_t[0])
            self.head3_t[self.minor_classes] = head3_t[1]

    def compute_prob(self, logits, head=1):
        logits_x_ulb_w_detached = logits.detach().float().cpu()
        max_indices = torch.argmax(logits_x_ulb_w_detached, dim=1)
        bin_counts = torch.bincount(max_indices, minlength=self.logits_counts.size(0))
        if head == 1:
            self.logits_record.scatter_add_(0, max_indices.unsqueeze(1).expand(-1, logits_x_ulb_w_detached.size(1)), logits_x_ulb_w_detached)
            self.logits_counts += bin_counts
        elif head == 2:
            self.logits_record2.scatter_add_(0, max_indices.unsqueeze(1).expand(-1, logits_x_ulb_w_detached.size(1)), logits_x_ulb_w_detached)
            self.logits_counts2 += bin_counts
        elif head == 3:
            self.logits_counts3 += bin_counts
        elif head == 0:
            self.logits_counts2_debias += bin_counts
        else:
            pass
        return torch.softmax(logits.detach(), dim=-1)
    
    @staticmethod
    @torch.no_grad()
    def bernouli_mask(x):
        return torch.bernoulli(x.detach()).float()

    def compute_head2_loss(self, logits_x_lb, y_lb, logits_x_ulb_w, logits_x_ulb_s, y_ulb=None):
        if not isinstance(logits_x_ulb_s, list):
            logits_x_ulb_s = [logits_x_ulb_s]
        
        if not self.lb_class_dist.is_cuda:
            self.lb_class_dist = self.lb_class_dist.to(y_lb.device)
        if not self.head2_t.is_cuda:
            self.head2_t = self.head2_t.to(y_lb.device)
        

        head2_lb_loss = (self.ce_loss(logits_x_lb + self.tau1 * torch.log(self.p_hat_lb), y_lb, reduction='none',class_weights=self.class_weights)).mean()

        with torch.no_grad():
            self.model.eval()
            white = torch.ones((1, 3, self.args.img_size, self.args.img_size)).cuda(self.gpu)
            selected_head = 'logits' if self.est_dist is not None and self.est_dist == 'long' else 'logits_aux'
            biased_degree_aux = self.model(white)[selected_head][-1]
            self.model.train()
 
            self.compute_prob(logits_x_ulb_w-biased_degree_aux, head=0)

            probs_x_ulb_w = self.compute_prob(logits_x_ulb_w, head=2)

            max_probs, ys_ulb = torch.max(probs_x_ulb_w, dim=1)
                
            assert self.head2_t is not None
            mask_ulb_1 = max_probs.ge(self.head2_t.to(ys_ulb.device)[ys_ulb]).to(logits_x_ulb_w.dtype)

            beta = self.epoch / self.epochs
            if self.est_dist is not None and self.est_dist == 'long':
                beta = beta if beta < 0.75 else 0.75

            ulb_class_dist = 1 - beta * (1 - self.lb_class_dist)
            
            mask_ulb_2 = self.bernouli_mask(ulb_class_dist[ys_ulb])
            
            if self.head2_ulb_mask:
                mask_ulb = mask_ulb_1 * mask_ulb_2
            else:
                mask_ulb = mask_ulb_1
            
            indices = torch.where(mask_ulb.int() == 1)[0]
            self.logits_counts2_mask += torch.bincount(ys_ulb[indices].detach().cpu(), minlength=self.logits_counts.size(0))
 
        head2_ulb_loss = 0.0
        for logits_s in logits_x_ulb_s:
            head2_ulb_loss += (self.ce_loss(logits_s, ys_ulb, reduction='none', class_weights=self.class_weights) * mask_ulb).mean()
        
        head2_loss = head2_lb_loss + self.uloss_scale * head2_ulb_loss
        return head2_loss
    
    def compute_head3_loss(self, logits_x_lb, y_lb, logits_x_ulb_w, logits_x_ulb_s):
        if not isinstance(logits_x_ulb_s, list):
            logits_x_ulb_s = [logits_x_ulb_s]
        
        if not self.lb_class_dist.is_cuda:
            self.lb_class_dist = self.lb_class_dist.to(y_lb.device)
        if not self.head3_t.is_cuda:
            self.head3_t = self.head3_t.to(y_lb.device)

        head3_lb_loss = (self.ce_loss(logits_x_lb + self.tau2 * torch.log(self.p_hat_lb), y_lb, reduction='none',class_weights=self.class_weights)).mean()

        with torch.no_grad():
            probs_x_ulb_w = self.compute_prob(logits_x_ulb_w, head=3)
            max_probs, y_ulb = torch.max(probs_x_ulb_w, dim=1)
            
            assert self.head3_t is not None
            mask_ulb_1 = max_probs.ge(self.head3_t.to(y_ulb.device)[y_ulb]).to(logits_x_ulb_w.dtype)
            
            beta = self.epoch / self.epochs
            if self.est_dist is not None and self.est_dist == 'long':
                beta = beta if beta < 0.75 else 0.75
            ulb_class_dist = 1 - beta * (1 - self.lb_class_dist)
            mask_ulb_2 = self.bernouli_mask(ulb_class_dist[y_ulb])

            if self.head3_ulb_mask:
                mask_ulb = mask_ulb_1 * mask_ulb_2
            else:
                mask_ulb = mask_ulb_1

            indices = torch.where(mask_ulb.int() == 1)[0]
            self.logits_counts3_mask += torch.bincount(y_ulb[indices].detach().cpu(), minlength=self.logits_counts.size(0))
    
        head3_ulb_loss = 0.0
        for logits_s in logits_x_ulb_s:
            head3_ulb_loss += (self.ce_loss(logits_s, y_ulb, reduction='none', class_weights=self.class_weights) * mask_ulb).mean()
        
        head3_loss = head3_lb_loss + self.uloss_scale * head3_ulb_loss
        return head3_loss

    def evaluate_debias(self, out_key='logits_aux', eval_dest='eval', return_logits=False):
        self.model.eval()
        self.ema.apply_shadow()

        eval_loader = self.loader_dict[eval_dest]
        total_num = 0.0
        y_true = []
        feats = []
        y_pred_b1 = []
        y_pred_b2 = []
        y_pred_r1 = []
        y_pred_r2 = []

        y_logits_record2 = []
        
        with torch.no_grad():
            averaged_logits = self.logits_record / torch.where(self.logits_counts.unsqueeze(1) == 0, torch.tensor(1.0), self.logits_counts.unsqueeze(1))
            averaged_logits2 = self.logits_record2 / torch.where(self.logits_counts2.unsqueeze(1) == 0, torch.tensor(1.0), self.logits_counts2.unsqueeze(1))    
            averaged_logits2.fill_diagonal_(0)
            mean_head2 = torch.sum(averaged_logits2, dim=0).cuda() / (self.num_classes - 1)
            averaged_logits.fill_diagonal_(0)
            mean_head1 = torch.sum(averaged_logits, dim=0).cuda() / (self.num_classes - 1)
            self.bias_ema = 0.9 * mean_head1.cuda() + 0.1 * self.bias_ema
            self.bias_ema2 = 0.9 * mean_head2.cuda() + 0.1 * self.bias_ema2

            white = torch.ones((1, 3, self.args.img_size, self.args.img_size)).cuda()
            biased_degree = self.model(white, use_zero=False)['logits']
            biased_degree2 = self.model(white, use_zero=False)['logits_aux']

            self.bias_white = biased_degree[0]
            self.bias_white2 = biased_degree2[0]

            self.print_fn(f"bias h1: {biased_degree}")
            self.print_fn(f"bias h2: {biased_degree2}")

            for data in eval_loader:
                x = data['x_lb']
                y = data['y_lb']
                
                if isinstance(x, dict):
                    x = {k: v.cuda(self.gpu) for k, v in x.items()}
                else:
                    x = x.cuda(self.gpu)
                y = y.cuda(self.gpu)

                num_batch = y.shape[0]
                total_num += num_batch

                out_dict = self.model(x)
                logits = out_dict[out_key]
                feat = out_dict['feat']

                logits_b1 = logits - self.coe * biased_degree
                logits_b2 = logits - self.coe * biased_degree2

                y_true.extend(y.cpu().tolist())

                y_pred_b1.extend(torch.max(logits_b1, dim=-1)[1].cpu().tolist())
                y_pred_b2.extend(torch.max(logits_b2, dim=-1)[1].cpu().tolist())
                y_pred_r1.extend(torch.max(out_dict['logits'], dim=-1)[1].cpu().tolist())
                y_pred_r2.extend(torch.max(out_dict['logits_aux'], dim=-1)[1].cpu().tolist())

                feats.append(feat.cpu().numpy())

        y_true = np.array(y_true)
        y_pred_b1 = np.array(y_pred_b1)
        y_pred_b2 = np.array(y_pred_b2)

        feats = np.concatenate(feats)

        top1_b1 = accuracy_score(y_true, y_pred_b1)
        top1_b2 = accuracy_score(y_true, y_pred_b2)

        self.ema.restore()
        self.model.train()

        eval_dict = {'h1/top-1-acc': top1_b1, 'h2/top-1-acc': top1_b2,
                     'h1/raw-acc': accuracy_score(y_true, np.array(y_pred_r1)),
                     'h2/raw-acc': accuracy_score(y_true, np.array(y_pred_r2))}

        if return_logits:
            return eval_dict, y_logits_record2

        return eval_dict
    
    def set_dataset(self):
        dataset_dict = get_dataset(self.args, self.algorithm, self.args.dataset, self.args.num_labels, self.args.num_classes, self.args.data_dir, self.args.include_lb_to_ulb)
        dataset_dict['eval_ulb'] = copy.deepcopy(dataset_dict['train_ulb'])
        dataset_dict['eval_ulb'].is_ulb = False

        self.args.ulb_dest_len = len(dataset_dict['train_ulb']) if dataset_dict['train_ulb'] is not None else 0
        self.args.lb_dest_len = len(dataset_dict['train_lb'])
        self.print_fn("unlabeled data number: {}, labeled data number {}, unlabeled eval data number {}".format(self.args.ulb_dest_len, self.args.lb_dest_len, len(dataset_dict['eval_ulb'])))
        
        return dataset_dict
    
    def set_data_loader(self):
        loader_dict = super().set_data_loader()
        
        loader_dict['eval_ulb'] = get_data_loader(self.args,
                                                  self.dataset_dict['eval_ulb'],
                                                  self.args.eval_batch_size,
                                                  data_sampler=None,
                                                  shuffle=False,
                                                  num_workers=self.args.num_workers,
                                                  drop_last=False)
        return loader_dict

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--t_step', float, 0.005),
            SSL_Argument('--coe', float, 1.0),
            SSL_Argument('--est_dist', str, None),
        ]
