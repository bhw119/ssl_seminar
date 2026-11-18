from __future__ import print_function
import copy
import time
import numpy as np

import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from torch.utils.data.sampler import WeightedRandomSampler

from progress.bar import Bar as Bar

__all__ = ['classifier_warmup']

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class WeightEMA(object):
    def __init__(self, model, ema_model, lr, alpha=0.999, wd=True):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        if wd:
            self.wd = 0.02 * lr
        else:
            self.wd = 0.0

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            ema_param = ema_param.float()
            param = param.float()
            ema_param.mul_(self.alpha)
            ema_param.add_(param * one_minus_alpha)
            param.mul_(1 - self.wd)

def classifier_warmup(args, model, train_labeled_set, train_unlabeled_set, N_SAMPLES_PER_CLASS, num_class, gpu):

    epochs = args.cossl_tfe_warm_epoch
    lr = args.cossl_tfe_warm_lr
    ema_decay = args.cossl_tfe_warm_ema_decay
    weight_decay = args.cossl_tfe_warm_wd
    batch_size = args.cossl_tfe_warm_bs
    val_iteration = args.num_eval_iter

    labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=batch_size,
                                          shuffle=False, num_workers=0, drop_last=False)
    unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size=batch_size,
                                            shuffle=False, num_workers=0, drop_last=False)

    tfe_model = weight_imprint(args, copy.deepcopy(model), train_labeled_set, num_class, gpu)

    for param in model.parameters():
        param.requires_grad = False
    
    model.module.classifier.reset_parameters()
    for param in model.module.classifier.parameters():
        param.requires_grad = True

    model = model.cuda(gpu)
    ema_model = copy.deepcopy(model)
    for param in ema_model.parameters():
        param.detach_()
    ema_optimizer = WeightEMA(model, ema_model, lr, alpha=ema_decay, wd=False)

    wd_params, non_wd_params = [], []
    for name, param in model.module.classifier.named_parameters():
        if 'bn' in name or 'bias' in name:
            non_wd_params.append(param)
        else:
            wd_params.append(param)
    param_list = [{'params': wd_params, 'weight_decay': weight_decay}, {'params': non_wd_params, 'weight_decay': 0}]
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.module.classifier.parameters()) / 1000000.0))

    optimizer = optim.Adam(param_list, lr=1e-3)

    balanced_feature_set = TFE(args, labeled_trainloader, unlabeled_trainloader,
                               tfe_model, num_class, N_SAMPLES_PER_CLASS, gpu)
    balanced_feature_loader = data.DataLoader(balanced_feature_set, batch_size=batch_size,
                                              shuffle=True, num_workers=0, drop_last=True)

    for epoch in range(epochs):
        print('\nTFE-head-warming: Epoch: [%d | %d] LR: %f' % (epoch + 1, epochs, optimizer.param_groups[0]['lr']))

        classifier_train(args, balanced_feature_loader, model, optimizer, None, ema_optimizer, val_iteration, gpu)

    return model, ema_model

def TFE(args, labeled_loader, unlabeled_loader, tfe_model, num_classes, num_samples_per_class, gpu):

    tfe_model.eval()
    with torch.no_grad():
        for batch_idx, data_dict in enumerate(labeled_loader):
            inputs = data_dict['x_lb']
            targets = data_dict['y_lb']

            inputs = inputs.cuda(gpu)
            targets = targets.cuda(gpu)
            ret_dict = tfe_model(inputs)
            logits = ret_dict['logits']
            features = ret_dict['feat']
            cls_probs = torch.softmax(logits, dim=1)
            features = features.squeeze()
            if batch_idx == 0:
                labeled_feature_stack = features
                labeled_target_stack = targets
                labeled_cls_prob_stack = cls_probs
            else:
                labeled_feature_stack = torch.cat((labeled_feature_stack, features), 0)
                labeled_target_stack = torch.cat((labeled_target_stack, targets), 0)
                labeled_cls_prob_stack = torch.cat((labeled_cls_prob_stack, cls_probs), 0)
        for batch_idx, data_dict in enumerate(unlabeled_loader):
            inputs_w = data_dict['x_ulb_w'].cuda(gpu)
            if args.algorithm in ['remixmatch', 'comatch']:
                inputs_s = data_dict['x_ulb_s_0'].cuda(gpu)
            else:
                inputs_s = data_dict['x_ulb_s'].cuda(gpu)
            features = tfe_model(inputs_s)['feat']
            logits = tfe_model(inputs_w)['logits']
            cls_probs = torch.softmax(logits, dim=1)
            _, targets = torch.max(cls_probs, dim=1)
            features = features.squeeze()
            if batch_idx == 0:
                unlabeled_feature_stack = features
                unlabeled_target_stack = targets
                unlabeled_cls_prob_stack = cls_probs
            else:
                unlabeled_feature_stack = torch.cat((unlabeled_feature_stack, features), 0)
                unlabeled_target_stack = torch.cat((unlabeled_target_stack, targets), 0)
                unlabeled_cls_prob_stack = torch.cat((unlabeled_cls_prob_stack, cls_probs), 0)

        labeled_set_idx = torch.tensor(list(range(len(labeled_feature_stack))))
        labeled_set_per_cls_idx = [labeled_set_idx[(labeled_target_stack == i).to(labeled_set_idx.device)] for i in range(num_classes)]

        augment_features = []
        augment_targets = []
        for cls_id in range(num_classes):
            if num_samples_per_class[cls_id] < max(num_samples_per_class):

                augment_size = max(num_samples_per_class) - num_samples_per_class[cls_id]

                current_cls_feats = labeled_feature_stack[labeled_target_stack == cls_id]

                other_labeled_data_idx = np.concatenate(labeled_set_per_cls_idx[:cls_id] + labeled_set_per_cls_idx[cls_id + 1:], axis=0)
                other_cls_feats = torch.cat([labeled_feature_stack[other_labeled_data_idx], unlabeled_feature_stack], dim=0)
                other_cls_probs = torch.cat([labeled_cls_prob_stack[other_labeled_data_idx], unlabeled_cls_prob_stack], dim=0)

                assert len(other_cls_feats) == len(other_cls_probs)
                assert (len(other_cls_feats) + len(current_cls_feats)) == (len(labeled_feature_stack) + len(unlabeled_feature_stack))

                probs4current_cls = other_cls_probs[:, cls_id]
                sorted_probs, order = probs4current_cls.sort(descending=True)
                other_cls_feats = other_cls_feats[order]

                input_a_idx = np.random.choice(list(range(len(current_cls_feats))), augment_size, replace=True)
                input_b_idx = np.asarray(list(range(augment_size)))
                lambdas = np.random.beta(0.75, 0.75, size=augment_size)

                for l, a_idx, b_idx in zip(lambdas, input_a_idx, input_b_idx):
                    tfe_input = l * current_cls_feats[a_idx] + (1 - l) * other_cls_feats[b_idx]
                    tfe_target = torch.zeros((1, num_classes))
                    tfe_target[0, cls_id] = 1
                    augment_features.append(tfe_input.view(1, -1))
                    augment_targets.append(tfe_target)

        augment_features = torch.cat(augment_features, dim=0)
        augment_targets = torch.cat(augment_targets, dim=0).cuda(gpu)

        target_stack = torch.zeros(len(labeled_target_stack), num_classes).cuda(gpu).scatter_(1, labeled_target_stack.view(-1, 1), 1)
        new_feat_tensor = torch.cat([labeled_feature_stack, augment_features], dim=0)
        new_target_tensor = torch.cat([target_stack, augment_targets], dim=0)

    balanced_feature_set = data.dataset.TensorDataset(new_feat_tensor, new_target_tensor)
    return balanced_feature_set

def weight_imprint(args, model, labeled_set, num_classes, gpu):
    model = model.cuda(gpu)
    model.eval()

    labeledloader = data.DataLoader(labeled_set, batch_size=100, shuffle=False, num_workers=0, drop_last=False)

    with torch.no_grad():
        bar = Bar('Classifier weight imprinting...', max=len(labeledloader))

        for batch_idx, data_dict in enumerate(labeledloader):
            inputs = data_dict['x_lb']
            targets = data_dict['y_lb']

            inputs = inputs.cuda(gpu)
            features = model(inputs)['feat']
            output = features.squeeze()

            if batch_idx == 0:
                output_stack = output
                target_stack = targets
            else:
                output_stack = torch.cat((output_stack, output), 0)
                target_stack = torch.cat((target_stack, targets), 0)

            bar.suffix = '({batch}/{size}'.format(batch=batch_idx + 1, size=len(labeledloader))
            bar.next()
        bar.finish()

    
    new_weight = torch.zeros(num_classes, model.module.backbone.num_features)

    for i in range(num_classes):
        tmp = output_stack[target_stack == i].mean(0)
        new_weight[i] = tmp / tmp.norm(p=2)
    model.module.classifier = torch.nn.Linear(model.module.backbone.num_features, num_classes, bias=False).cuda(gpu)
    model.module.classifier.weight.data = new_weight.cuda(gpu)

    model.eval()
    return model

def classifier_train(args, labeled_trainloader, model, optimizer, scheduler, ema_optimizer, val_iteration, gpu):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    train_acc = AverageMeter()
    end = time.time()

    labeled_train_iter = iter(labeled_trainloader)

    model.eval()
    for batch_idx in range(val_iteration):
        try:
            inputs_x, targets_x = next(labeled_train_iter)
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x = next(labeled_train_iter)

        data_time.update(time.time() - end)

        inputs_x, targets_x = inputs_x.cuda(gpu), targets_x.cuda(gpu)

        outputs = model.module.classifier(inputs_x)

        loss = (-F.log_softmax(outputs, dim=1) * targets_x).sum(dim=1)
        loss = loss.mean()
        acc = (torch.argmax(outputs, dim=1) == torch.argmax(targets_x, dim=1)).float().sum() / len(targets_x)

        losses.update(loss.item(), inputs_x.size(0))
        train_acc.update(acc.item(), inputs_x.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()
        if scheduler is not None:
            scheduler.step()

        batch_time.update(time.time() - end)
        end = time.time()

    return losses.avg, train_acc.avg

def get_weighted_sampler(target_sample_rate, num_sample_per_class, target):
    assert len(num_sample_per_class) == len(np.unique(target))

    sample_weights = target_sample_rate / num_sample_per_class
    print(sample_weights)

    samples_weight = np.array([sample_weights[t] for t in target])

    return WeightedRandomSampler(samples_weight, len(samples_weight), True)

def make_imb_data(max_num, class_num, gamma):
    mu = np.power(1/abs(gamma), 1/(class_num - 1))
    class_num_list = []
    for i in range(class_num):
        if i == (class_num - 1):
            class_num_list.append(int(max_num / abs(gamma)))
        else:
            class_num_list.append(int(max_num * np.power(mu, i)))
    if gamma < 0:
        class_num_list = class_num_list[::-1]
    return list(class_num_list)
