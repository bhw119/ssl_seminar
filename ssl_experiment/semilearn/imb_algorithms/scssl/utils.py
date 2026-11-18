import numpy as np
import torch
import torch.nn as nn
from semilearn.core.hooks import Hook
import os
from semilearn.core.utils import (
    get_net_builder,
    send_model_cuda,
)
from semilearn.datasets.utils import make_imbalance_data
from semilearn.imb_algorithms import get_imb_algorithm
from scipy.special import kl_div

def normalize_distribution(distribution):
    return distribution / np.sum(distribution)

class EstimateDistHook(Hook):
    def __init__(self) -> None:
        super().__init__()
        self.DIST_ZOO = {
        'long': lambda class_num, max_num, gamma: make_imbalance_data(num_classes=class_num, max_num_labels=max_num, gamma=gamma, dist_type='long_tail'),
        'long_inverse': lambda class_num, max_num, gamma: make_imbalance_data(num_classes=class_num, max_num_labels=max_num, gamma=-gamma, dist_type='long_tail'),
        'uniform': lambda class_num, max_num, gamma: make_imbalance_data(num_classes=class_num, max_num_labels=max_num, gamma=1, dist_type='long_tail'),
        'gaussian': lambda class_num, max_num, gamma: make_imbalance_data(num_classes=class_num, max_num_labels=max_num, gamma=1, dist_type='gaussian'),
        'gaussian_inverse': lambda class_num, max_num, gamma: make_imbalance_data(num_classes=class_num, max_num_labels=max_num, gamma=-1, dist_type='gaussian'),
        }
        self.search_points = [100, 200, 300]
    
    def find_closest_distribution(self, debias_ulb_nums, max_num=3000):
        if max_num == 'None':
            max_num = max(debias_ulb_nums)

        closest_distribution_name = None
        min_kl_divergence = float('inf')
        named_true_distribution = None
        param = None

        for name, dist_func in self.DIST_ZOO.items():
            if name in ['long', 'long_inverse']:
                for gm in self.search_points:
                    true_distribution = dist_func(class_num=len(debias_ulb_nums), max_num=max_num, gamma=gm)
                    kl_divergence = np.min((np.sum(kl_div(true_distribution, debias_ulb_nums)), np.sum(kl_div(debias_ulb_nums, true_distribution))))
                    if kl_divergence < min_kl_divergence:
                        min_kl_divergence = kl_divergence
                        closest_distribution_name = name
                        named_true_distribution = true_distribution
                        param = gm
            else:
                true_distribution = dist_func(class_num=len(debias_ulb_nums), max_num=max_num, gamma=None)
                kl_divergence = np.min((np.sum(kl_div(true_distribution, debias_ulb_nums)), np.sum(kl_div(debias_ulb_nums, true_distribution))))
                if kl_divergence < min_kl_divergence:
                    min_kl_divergence = kl_divergence
                    closest_distribution_name = name
                    named_true_distribution = true_distribution

        return closest_distribution_name, min_kl_divergence, named_true_distribution, param

    def before_run(self, algorithm):
        if algorithm.est_dist is None:
            _net_builder = get_net_builder(algorithm.args.net, algorithm.args.net_from_name)
            alg = get_imb_algorithm(algorithm.args, _net_builder, None, logger=None)
            alg.print_fn = algorithm.print_fn
            alg.model = send_model_cuda(algorithm.args, alg.model)
            alg.ema_model = send_model_cuda(algorithm.args, alg.ema_model, clip_batch=False)
            alg.call_hook("before_run", "TimerHook")
            alg.call_hook("before_run", "EMAHook")
            alg.call_hook("before_run",  "UpdateBiasThresHook")

            
            tmp_dist = algorithm.args.est_dist
            gamma=100
            if algorithm.args.dataset == 'stl10':
                algorithm.args.ulb_num_labels = 10500
            elif algorithm.args.dataset == 'imagenet127-32' or algorithm.args.dataset == 'imagenet127-64':
                algorithm.args.ulb_num_labels = 1281167 - algorithm.args.num_labels
            

            if tmp_dist == 'from_lb':
                n_ulb = float(algorithm.args.ulb_dest_len)
                named = algorithm.lb_class_nums / algorithm.lb_class_nums.sum() * n_ulb
                algorithm.est_dist = 'long'
                algorithm.ulb_class_dist = torch.tensor(named, dtype=torch.float32)
                algorithm.print_fn(f"Estimated Dist: from_lb (treated as 'long'); ulb counts {algorithm.ulb_class_dist}")
                algorithm.update_dist_params()
                return
            debias_ulb_nums = self.DIST_ZOO[tmp_dist](class_num=algorithm.num_classes, max_num=algorithm.args.ulb_num_labels, gamma=gamma)
            closest_dist_name, min_kl_divergence, named_true_dist = (tmp_dist, 0, self.DIST_ZOO[tmp_dist](class_num=algorithm.num_classes, max_num=algorithm.args.ulb_num_labels, gamma=gamma))
            algorithm.print_fn(f"Estimation Debias Counts: {debias_ulb_nums}")
            algorithm.print_fn(f"Estimated Dist: {closest_dist_name}/{min_kl_divergence}")
            algorithm.est_dist = closest_dist_name
            if algorithm.args.dataset == 'imagenet127-32' or algorithm.args.dataset == 'imagenet127-64':
                algorithm.ulb_class_dist = torch.from_numpy(algorithm.lb_class_nums)
            else:
                algorithm.ulb_class_dist = torch.from_numpy(np.array(named_true_dist) + algorithm.lb_class_nums).to(torch.float32) if algorithm.args.include_lb_to_ulb else torch.tensor(named_true_dist, dtype=torch.float32)
            algorithm.print_fn(f"Estimated Ulb Counts: {algorithm.ulb_class_dist}")

            algorithm.update_dist_params()
            algorithm.print_fn(f"Start Training... ")

class UpdateBiasThresHook(Hook):
    def effective_weights(self, y_cnt, true_cnt):
        true_cnt = self.scale_to_target_sum(true_cnt)
        weights = torch.ones_like(y_cnt, dtype=torch.float)
        less_mask = y_cnt < true_cnt
        greater_mask = y_cnt >= true_cnt

        weights[less_mask] += (true_cnt[less_mask] - y_cnt[less_mask]) / (true_cnt[less_mask] + 1)

        weights[greater_mask] -= (y_cnt[greater_mask] - true_cnt[greater_mask]) / (y_cnt[greater_mask] + 1)
        return weights
    
    def scale_to_target_sum(self, weights, target_sum=131072):
        current_sum = weights.sum()
        scale_factor = target_sum / current_sum
        scaled_weights = weights * scale_factor
        return scaled_weights
    
    def clear_counts(self, algorithm):
        algorithm.logits_record = torch.zeros(algorithm.num_classes, algorithm.num_classes) 
        algorithm.logits_counts = torch.zeros(algorithm.num_classes)
        algorithm.logits_record2 = torch.zeros(algorithm.num_classes, algorithm.num_classes) 
        algorithm.logits_counts2 = torch.zeros(algorithm.num_classes)  
        algorithm.logits_counts2_debias = torch.zeros(algorithm.num_classes)  
        algorithm.logits_counts2_mask = torch.zeros(algorithm.num_classes)
        algorithm.logits_counts3 = torch.zeros(algorithm.num_classes)
        algorithm.logits_counts3_mask = torch.zeros(algorithm.num_classes) 

    def before_run(self, algorithm):
        self.clear_counts(algorithm)

    def after_train_step(self, algorithm):
        if self.every_n_iters(algorithm, algorithm.num_eval_iter) or self.is_last_iter(algorithm):
            if algorithm.est_dist not in ['long']:
                used_bias = algorithm.bias_white2 if algorithm.args.dataset != 'stl10' else algorithm.bias_ema2
            else:
                used_bias = algorithm.bias_white if algorithm.args.dataset != 'stl10' else algorithm.bias_ema
            condition1 = (used_bias.cpu().abs() > 1) & (used_bias.cpu() < 0)

            if algorithm.head2_desc and algorithm.head2_t is not None:
                algorithm.head2_t[condition1] -= algorithm.t_step
            if algorithm.head3_desc and algorithm.head3_t is not None:
                if algorithm.epoch < 128:
                    algorithm.head3_t[condition1] -= algorithm.t_step
            
            if algorithm.head2_t is not None:
                algorithm.head2_t = torch.clamp(algorithm.head2_t, max=0.95, min=0.001)
            if algorithm.head3_t is not None:
                algorithm.head3_t = torch.clamp(algorithm.head3_t, max=0.95, min=0.001)

            algorithm.print_fn(f"Thresholds h2: {algorithm.head2_t}")
            algorithm.print_fn(f"Thresholds h3: {algorithm.head3_t}")

            if algorithm.epoch >= 10:
                algorithm.class_weights = self.effective_weights(algorithm.logits_counts2_debias, algorithm.ulb_class_dist).cuda()
                algorithm.class_weights = torch.clamp(algorithm.class_weights, max=3.0, min=1.0)
            algorithm.print_fn(f"Class weights: {algorithm.class_weights}")

            if not (algorithm.est_dist is None and algorithm.it == algorithm.est_iters-1):
                self.clear_counts(algorithm)

            

class CheckpointHook(Hook):
    def after_train_step(self, algorithm):
        if self.every_n_iters(algorithm, algorithm.num_eval_iter) or self.is_last_iter(algorithm):
            save_path = os.path.join(algorithm.save_dir, algorithm.save_name)
            prefix = '' if algorithm.est_dist is not None else 'est_'
            if (not algorithm.distributed) or \
               (algorithm.distributed and algorithm.rank % algorithm.ngpus_per_node == 0):
                algorithm.save_model(f'{prefix}latest_model.pth', save_path)

                if algorithm.it == algorithm.best_it:
                    algorithm.save_model(f'{prefix}model_best.pth', save_path)

class MyNet(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.num_features = backbone.num_features
        self.head3 = True

        self.aux_classifier = nn.Linear(self.backbone.num_features, num_classes)
        self.aux_classifier2 = nn.Linear(self.backbone.num_features, num_classes)
    
    def forward(self, x, **kwargs):
        results_dict = self.backbone(x, **kwargs)
        results_dict['logits_aux'] = self.aux_classifier(results_dict['feat'])
        if self.head3:
            results_dict['logits_aux2'] = self.aux_classifier2(results_dict['feat'])
        return results_dict

    def group_matcher(self, coarse=False):
        if hasattr(self.backbone, 'backbone'):
            matcher = self.backbone.backbone.group_matcher(coarse, prefix='backbone.backbone')
        else:
            matcher = self.backbone.group_matcher(coarse, prefix='backbone.')
        return matcher
