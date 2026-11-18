import os
import random
import numpy as np
import torch
from torch.utils.data import sampler, DataLoader
import torch.distributed as dist
from io import BytesIO

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

def split_ssl_data(args, data, targets, num_classes,
                   lb_num_labels, ulb_num_labels=None,
                   lb_imbalance_ratio=1.0, ulb_imbalance_ratio=1.0,
                   lb_index=None, ulb_index=None, include_lb_to_ulb=True, load_exist=True):
    data, targets = np.array(data), np.array(targets)
    lb_idx, ulb_idx = sample_labeled_unlabeled_data(args, data, targets, num_classes, 
                                                    lb_num_labels, ulb_num_labels,
                                                    lb_imbalance_ratio, ulb_imbalance_ratio, load_exist=False)
    
    if lb_index is not None:
        lb_idx = lb_index
    if ulb_index is not None:
        ulb_idx = ulb_index

    if include_lb_to_ulb:
        ulb_idx = np.concatenate([lb_idx, ulb_idx], axis=0)
    
    return data[lb_idx], targets[lb_idx], data[ulb_idx], targets[ulb_idx]

def sample_labeled_data():
    pass

def sample_labeled_unlabeled_data(args, data, target, num_classes,
                                  lb_num_labels, ulb_num_labels=None,
                                  lb_imbalance_ratio=1.0, ulb_imbalance_ratio=1.0,
                                  load_exist=True):
    dump_dir = os.path.join(base_dir, 'data', args.dataset, 'labeled_idx')
    os.makedirs(dump_dir, exist_ok=True)

    if args.dist_type=='long_tail':
        lb_dump_path = os.path.join(dump_dir, f'lb_labels{args.num_labels}_{args.lb_imb_ratio}_seed{args.seed}_idx.npy')
        ulb_dump_path = os.path.join(dump_dir, f'ulb_labels{args.num_labels}_{args.ulb_imb_ratio}_seed{args.seed}_idx.npy')
    else:
        lb_dump_path = os.path.join(dump_dir, f'lb_labels{args.num_labels}_{args.lb_imb_ratio}_seed{args.seed}_idx.npy')
        ulb_dump_path = os.path.join(dump_dir, f'ulb_labels{args.num_labels}_{args.ulb_imb_ratio}_seed{args.seed}_{args.dist_type}_idx.npy')
                                     
    if os.path.exists(lb_dump_path) and os.path.exists(ulb_dump_path) and load_exist:
        lb_idx = np.load(lb_dump_path)
        ulb_idx = np.load(ulb_dump_path)
        return lb_idx, ulb_idx 

    
    if lb_imbalance_ratio == 1.0:
        assert lb_num_labels % num_classes == 0, "lb_num_labels must be dividable by num_classes in balanced setting"
        lb_samples_per_class = [int(lb_num_labels / num_classes)] * num_classes
    else:
        lb_samples_per_class = make_imbalance_data(lb_num_labels, num_classes, lb_imbalance_ratio)

    if ulb_imbalance_ratio == 1.0 and args.dist_type == 'long_tail':
        if ulb_num_labels is None or ulb_num_labels == 'None':
            pass
        else:
            assert ulb_num_labels % num_classes == 0, "ulb_num_labels must be dividable by num_classes in balanced setting"
            ulb_samples_per_class = [ulb_num_labels] * num_classes
    else:
        assert ulb_num_labels is not None, "ulb_num_labels must be set set in imbalanced setting"
        if args.dist_type != 'long_tail':
            print(f"{args.dist_type} distribution making...")
            ulb_samples_per_class = make_imbalance_data(ulb_num_labels, num_classes, ulb_imbalance_ratio, dist_type=args.dist_type)
        else:
            ulb_samples_per_class = make_imbalance_data(ulb_num_labels, num_classes, ulb_imbalance_ratio)

    lb_idx = []
    ulb_idx = []
    
    for c in range(num_classes):
        idx = np.where(target == c)[0]
        np.random.shuffle(idx)
        lb_idx.extend(idx[:lb_samples_per_class[c]])
        if ulb_num_labels is None or ulb_num_labels == 'None':
            ulb_idx.extend(idx[lb_samples_per_class[c]:])
        else:
            ulb_idx.extend(idx[lb_samples_per_class[c]:lb_samples_per_class[c]+ulb_samples_per_class[c]])
    
    if isinstance(lb_idx, list):
        lb_idx = np.asarray(lb_idx)
    if isinstance(ulb_idx, list):
        ulb_idx = np.asarray(ulb_idx)

    np.save(lb_dump_path, lb_idx)
    np.save(ulb_dump_path, ulb_idx)
    
    return lb_idx, ulb_idx

def make_imbalance_data(max_num_labels, num_classes, gamma, dist_type='long_tail'):
    if gamma == 1 and dist_type == 'long_tail':
        return [int(max_num_labels)] * num_classes
    class_num_list = []
    if dist_type == 'long_tail':
        mu = np.power(1/abs(gamma), 1/(num_classes - 1))
        for i in range(num_classes):
            if i == (num_classes - 1):
                class_num_list.append(int(max_num_labels / abs(gamma)))
            else:
                class_num_list.append(int(max_num_labels * np.power(mu, i)))
        if gamma < 0:
            class_num_list = class_num_list[::-1]
    else:
        mean = (num_classes - 1) / 2
        std_dev = num_classes / 6 

        for i in range(num_classes):
            prob = np.exp(-((i - mean) ** 2) / (2 * std_dev ** 2))
            if gamma == -1:
                prob = 1 - prob
            count = int(prob * max_num_labels / np.max([1 - np.exp(-((j - mean) ** 2) / (2 * std_dev ** 2)) for j in range(num_classes)]))
            class_num_list.append(count)
    return class_num_list

def get_collactor(args, net):
    if net == 'bert_base_uncased':
        from semilearn.datasets.collactors import get_bert_base_uncased_collactor
        collact_fn = get_bert_base_uncased_collactor(args.max_length)
    elif net == 'bert_base_cased':
        from semilearn.datasets.collactors import get_bert_base_cased_collactor
        collact_fn = get_bert_base_cased_collactor(args.max_length)
    elif net == 'wave2vecv2_base':
        from semilearn.datasets.collactors import get_wave2vecv2_base_collactor
        collact_fn = get_wave2vecv2_base_collactor(args.max_length_seconds, args.sample_rate)
    elif net == 'hubert_base':
        from semilearn.datasets.collactors import get_hubert_base_collactor
        collact_fn = get_hubert_base_collactor(args.max_length_seconds, args.sample_rate)
    else:
        collact_fn = None
    return collact_fn

def get_onehot(num_classes, idx):
    onehot = np.zeros([num_classes], dtype=np.float32)
    onehot[idx] += 1.0
    return onehot

def bytes_to_array(b: bytes) -> np.ndarray:
    np_bytes = BytesIO(b)
    return np.load(np_bytes, allow_pickle=True)

def random_subsample(wav: np.ndarray, max_length: float, sample_rate: int = 16000):
    sample_length = int(round(sample_rate * max_length))
    if len(wav) <= sample_length:
        return wav
    random_offset = random.randint(0, len(wav) - sample_length - 1)
    return wav[random_offset : random_offset + sample_length]
