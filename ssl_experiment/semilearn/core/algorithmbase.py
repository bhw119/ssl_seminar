import os
import contextlib
import numpy as np
from inspect import signature
from collections import OrderedDict
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, top_k_accuracy_score

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from semilearn.core.hooks import Hook, get_priority, CheckpointHook, TimerHook, LoggingHook, DistSamplerSeedHook, ParamUpdateHook, EvaluationHook, EMAHook, WANDBHook, AimHook
from semilearn.core.utils import get_dataset, get_data_loader, get_optimizer, get_cosine_schedule_with_warmup, Bn_Controller
from semilearn.core.criterions import CELoss, ConsistencyLoss, GCELoss

class AlgorithmBase:
    def __init__(
        self,
        args,
        net_builder,
        tb_log=None,
        logger=None,
        **kwargs):
        
        self.args = args
        self.num_classes = args.num_classes
        self.ema_m = args.ema_m
        self.epochs = args.epoch
        self.num_train_iter = args.num_train_iter
        self.num_eval_iter = args.num_eval_iter
        self.num_log_iter = args.num_log_iter
        self.num_iter_per_epoch = int(self.num_train_iter // self.epochs)
        self.lambda_u = args.ulb_loss_ratio 
        self.use_cat = args.use_cat
        self.use_amp = args.amp
        self.clip_grad = args.clip_grad
        self.save_name = args.save_name
        self.save_dir = args.save_dir
        self.resume = args.resume
        self.algorithm = args.algorithm

        self.tb_log = tb_log
        self.print_fn = print if logger is None else logger.info
        self.ngpus_per_node = torch.cuda.device_count()
        self.loss_scaler = GradScaler()
        self.amp_cm = autocast if self.use_amp else contextlib.nullcontext
        self.gpu = args.gpu
        self.rank = args.rank
        self.distributed = args.distributed
        self.world_size = args.world_size

        self.it = 0
        self.start_epoch = 0
        self.best_eval_acc, self.best_it = 0.0, 0
        self.bn_controller = Bn_Controller()
        self.net_builder = net_builder
        self.ema = None

        self.dataset_dict = self.set_dataset()

        self.loader_dict = self.set_data_loader()

        self.model = self.set_model()
        self.ema_model = self.set_ema_model()

        self.optimizer, self.scheduler = self.set_optimizer()

        self.ce_loss = CELoss()
        self.consistency_loss = ConsistencyLoss()
        self.gce_loss = GCELoss(num_classes=self.num_classes)

        self._hooks = []
        self.hooks_dict = OrderedDict()
        self.set_hooks()

    def init(self, **kwargs):
        raise NotImplementedError
    

    def set_dataset(self):
        if self.rank != 0 and self.distributed:
            torch.distributed.barrier()
        dataset_dict = get_dataset(self.args, self.algorithm, self.args.dataset, self.args.num_labels, self.args.num_classes, self.args.data_dir, self.args.include_lb_to_ulb)
        if dataset_dict is None:
            return dataset_dict

        self.args.ulb_dest_len = len(dataset_dict['train_ulb']) if dataset_dict['train_ulb'] is not None else 0
        self.args.lb_dest_len = len(dataset_dict['train_lb'])
        self.print_fn("unlabeled data number: {}, labeled data number {}".format(self.args.ulb_dest_len, self.args.lb_dest_len))
        if self.rank == 0 and self.distributed:
            torch.distributed.barrier()
        return dataset_dict

    def set_data_loader(self):
        if self.dataset_dict is None:
            return
            
        self.print_fn("Create train and test data loaders")
        loader_dict = {}
        loader_dict['train_lb'] = get_data_loader(self.args,
                                                  self.dataset_dict['train_lb'],
                                                  self.args.batch_size,
                                                  data_sampler=self.args.train_sampler,
                                                  num_iters=self.num_train_iter,
                                                  num_epochs=self.epochs,
                                                  num_workers=self.args.num_workers,
                                                  distributed=self.distributed)

        loader_dict['train_ulb'] = get_data_loader(self.args,
                                                   self.dataset_dict['train_ulb'],
                                                   self.args.batch_size * self.args.uratio,
                                                   data_sampler=self.args.train_sampler,
                                                   num_iters=self.num_train_iter,
                                                   num_epochs=self.epochs,
                                                   num_workers=2 * self.args.num_workers,
                                                   distributed=self.distributed)

        loader_dict['eval'] = get_data_loader(self.args,
                                              self.dataset_dict['eval'],
                                              self.args.eval_batch_size,
                                              data_sampler=None,
                                              num_workers=self.args.num_workers,
                                              drop_last=False)
        
        if getattr(self.args, 'warm_up_iter', 0) and self.args.algorithm == 'cgmatch':
            loader_dict['warm_up_eval'] = get_data_loader(self.args,
                                                          self.dataset_dict['train_ulb'],
                                                          self.args.eval_batch_size,
                                                          data_sampler=None,
                                                          num_workers=2 * self.args.num_workers,
                                                          drop_last=False)

        if self.dataset_dict['test'] is not None:
            loader_dict['test'] =  get_data_loader(self.args,
                                                   self.dataset_dict['test'],
                                                   self.args.eval_batch_size,
                                                   data_sampler=None,
                                                   num_workers=self.args.num_workers,
                                                   drop_last=False)
        self.print_fn(f'[!] data loader keys: {loader_dict.keys()}')
        return loader_dict

    def set_optimizer(self):
        self.print_fn("Create optimizer and scheduler")
        optimizer = get_optimizer(self.model, self.args.optim, self.args.lr, self.args.momentum, self.args.weight_decay, self.args.layer_decay)
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    self.num_train_iter,
                                                    num_warmup_steps=self.args.num_warmup_iter)
        return optimizer, scheduler

    def set_model(self):
        model = self.net_builder(num_classes=self.num_classes, pretrained=self.args.use_pretrain, pretrained_path=self.args.pretrain_path)
        return model

    def set_ema_model(self):
        ema_model = self.net_builder(num_classes=self.num_classes)
        ema_model.load_state_dict(self.model.state_dict())
        return ema_model

    def set_hooks(self):
        self.register_hook(ParamUpdateHook(), None, "HIGHEST")
        self.register_hook(EMAHook(), None, "HIGH")
        self.register_hook(EvaluationHook(), None, "HIGH")
        self.register_hook(CheckpointHook(), None, "HIGH")
        self.register_hook(DistSamplerSeedHook(), None, "NORMAL")
        self.register_hook(TimerHook(), None, "LOW")
        self.register_hook(LoggingHook(), None, "LOWEST")
        if self.args.use_wandb:
            self.register_hook(WANDBHook(), None, "LOWEST")
        if self.args.use_aim:
            self.register_hook(AimHook(), None, "LOWEST")

    def process_batch(self, input_args=None, **kwargs):
        if input_args is None:
            input_args = signature(self.train_step).parameters
            input_args = list(input_args.keys())

        input_dict = {}

        for arg, var in kwargs.items():
            if not arg in input_args:
                continue
            
            if var is None:
                continue
            
            if isinstance(var, dict):
                var = {k: v.cuda(self.gpu) for k, v in var.items()}
            else:
                var = var.cuda(self.gpu)
            input_dict[arg] = var
        return input_dict
    

    def process_out_dict(self, out_dict=None, **kwargs):
        if out_dict is None:
            out_dict = {}

        for arg, var in kwargs.items():
            out_dict[arg] = var
        
        return out_dict

    def process_log_dict(self, log_dict=None, prefix='train', **kwargs):
        if log_dict is None:
            log_dict = {}

        for arg, var in kwargs.items():
            log_dict[f'{prefix}/' + arg] = var
        return log_dict

    def compute_prob(self, logits):
        return torch.softmax(logits, dim=-1)

    def train_step(self, idx_lb, x_lb, y_lb, idx_ulb, x_ulb_w, x_ulb_s):
        raise NotImplementedError

    def train(self):
        self.model.train()
        self.call_hook("before_run")

        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            
            if self.it >= self.num_train_iter:
                break
            
            self.call_hook("before_train_epoch")

            for data_lb, data_ulb in zip(self.loader_dict['train_lb'],
                                         self.loader_dict['train_ulb']):
                if self.it >= self.num_train_iter:
                    break
                self.call_hook("before_train_step")
                self.out_dict, self.log_dict = self.train_step(**self.process_batch(**data_lb, **data_ulb))
                self.call_hook("after_train_step")
                self.it += 1
            
            self.call_hook("after_train_epoch")

        self.call_hook("after_run")

    def evaluate(self, eval_dest='eval', out_key='logits', return_logits=False, pname=''):
        self.model.eval()
        self.ema.apply_shadow()

        eval_loader = self.loader_dict[eval_dest]
        prefix = eval_dest + '-' + pname
        total_loss = 0.0
        total_num = 0.0
        y_true = []
        feats = []
        y_pred = []
        y_probs = []
        y_logits = []
        y_logits_org = []
        with torch.no_grad():
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
                logits_org = out_dict['logits']
                feat = out_dict['feat']
                
                loss = F.cross_entropy(logits, y, reduction='mean', ignore_index=-1)
                y_true.extend(y.cpu().tolist())
                y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
                y_logits.append(logits.cpu().numpy())
                y_logits_org.append(logits_org.cpu().numpy())
                y_probs.extend(torch.softmax(logits, dim=-1).cpu().tolist())
                feats.append(feat.cpu().numpy())

                total_loss += loss.item() * num_batch
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_logits = np.concatenate(y_logits)
        y_logits_org = np.concatenate(y_logits_org)

        feats = np.concatenate(feats)

        top1 = accuracy_score(y_true, y_pred)
        top2 = top_k_accuracy_score(y_true, y_probs, k=2)
        top3 = top_k_accuracy_score(y_true, y_probs, k=3)

        top5 = top_k_accuracy_score(y_true, y_probs, k=5)
        balanced_top1 = balanced_accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        F1 = f1_score(y_true, y_pred, average='macro')

        unique_classes = np.unique(y_true)
        classwise_acc = []

        for class_label in unique_classes:
            class_mask = (y_true == class_label)
            if class_mask.sum() > 0: 
                class_accuracy = accuracy_score(y_true[class_mask], y_pred[class_mask])
                classwise_acc.append(class_accuracy)

        macro_avg_acc = np.mean(classwise_acc) if classwise_acc else 0

        cf_mat = confusion_matrix(y_true, y_pred, normalize='true')
        if self.num_classes >= 100:
            self.print_fn('confusion matrix diag:\n' + np.array_str(np.diag(cf_mat)))
        else:
            self.print_fn('confusion matrix:\n' + np.array_str(cf_mat))
        self.ema.restore()
        self.model.train()

        
        eval_dict = {prefix+'/loss': total_loss / total_num, prefix+'/top-1-acc': top1, prefix+'/top-2-acc': top2, prefix+'/top-3-acc': top3, prefix+'/top-5-acc': top5, 
                prefix+'/balanced_acc': balanced_top1, prefix+'/precision': precision, prefix+'/recall': recall, prefix+'/F1': F1, prefix+'/cls-avg-acc': macro_avg_acc}
        
        if return_logits:
            eval_dict[prefix+f'/{out_key}'] = y_logits
            eval_dict[prefix+'/logits'] = y_logits_org
            eval_dict[prefix+'/y_true'] = y_true
        return eval_dict

    def get_save_dict(self):
        save_dict = {
            'model': self.model.state_dict(),
            'ema_model': self.ema_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'loss_scaler': self.loss_scaler.state_dict(),
            'it': self.it + 1,
            'epoch': self.epoch + 1,
            'best_it': self.best_it,
            'best_eval_acc': self.best_eval_acc,
        }
        if self.scheduler is not None:
            save_dict['scheduler'] = self.scheduler.state_dict()
        return save_dict
    

    def save_model(self, save_name, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        save_filename = os.path.join(save_path, save_name)
        save_dict = self.get_save_dict()
        torch.save(save_dict, save_filename)
        self.print_fn(f"model saved: {save_filename}")

    def load_model(self, load_path):
        checkpoint = torch.load(load_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model'], strict=True)
        self.ema_model.load_state_dict(checkpoint['ema_model'], strict=True)
        self.loss_scaler.load_state_dict(checkpoint['loss_scaler'])
        self.it = checkpoint['it']
        self.start_epoch = checkpoint['epoch']
        self.epoch = self.start_epoch
        self.best_it = checkpoint['best_it']
        self.best_eval_acc = checkpoint['best_eval_acc']
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.scheduler is not None and 'scheduler' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.print_fn('Model loaded')
        return checkpoint

    def check_prefix_state_dict(self, state_dict):
        new_state_dict = dict()
        for key, item in state_dict.items():
            if key.startswith('module'):
                new_key = '.'.join(key.split('.')[1:])
            else:
                new_key = key
            new_state_dict[new_key] = item
        return new_state_dict

    def register_hook(self, hook, name=None, priority='NORMAL'):
        assert isinstance(hook, Hook)
        if hasattr(hook, 'priority'):
            raise ValueError('"priority" is a reserved attribute for hooks')
        priority = get_priority(priority)
        hook.priority = priority
        hook.name = name if name is not None else type(hook).__name__

        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if priority >= self._hooks[i].priority:
                self._hooks.insert(i + 1, hook)
                inserted = True
                break
        
        if not inserted:
            self._hooks.insert(0, hook)

        self.hooks_dict = OrderedDict()
        for hook in self._hooks:
            self.hooks_dict[hook.name] = hook
        

    def call_hook(self, fn_name, hook_name=None, *args, **kwargs):
        
        if hook_name is not None:
            return getattr(self.hooks_dict[hook_name], fn_name)(self, *args, **kwargs)
        
        for hook in self.hooks_dict.values():
            if hasattr(hook, fn_name):
                getattr(hook, fn_name)(self, *args, **kwargs)

    def registered_hook(self, hook_name):
        return hook_name in self.hooks_dict

    @staticmethod
    def get_argument():
        return {}

class ImbAlgorithmBase(AlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None, **kwargs):
        super().__init__(args, net_builder, tb_log, logger, **kwargs)
        
        self.lb_imb_ratio = self.args.lb_imb_ratio
        self.ulb_imb_ratio = self.args.ulb_imb_ratio
        self.imb_algorithm = self.args.imb_algorithm
    
    def imb_init(self, *args, **kwargs):
        pass 

    def set_optimizer(self):
        if 'vit' in self.args.net and self.args.dataset in ['cifar100', 'food101', 'semi_aves', 'semi_aves_out']:
            return super().set_optimizer() 
        elif self.args.dataset in ['imagenet', 'imagenet127']:
            return super().set_optimizer() 
        elif self.args.dataset in ['semi_inat2020', 'semi_inat2020_out'] and not getattr(self.args, 'orig_optim', False):
            return super().set_optimizer() 
        else:
            self.print_fn("Create optimizer and scheduler")
            optimizer = get_optimizer(self.model, self.args.optim, self.args.lr, self.args.momentum, self.args.weight_decay, self.args.layer_decay, bn_wd_skip=False)
            scheduler = None
            return optimizer, scheduler
