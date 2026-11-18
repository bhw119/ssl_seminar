import csv
import os
import json
import math
import numpy as np
from torchvision.datasets import folder as dataset_parser
from torchvision.transforms import transforms
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation
from .datasetbase import BasicDataset

IMGNET_MEAN = (0.485, 0.456, 0.406)
IMGNET_STD = (0.229, 0.224, 0.225)

def _read_split(data_dir, split, check_exists=False):
    ann = os.path.join(data_dir, 'annotation', 'annotation')
    if not os.path.isdir(ann):
        ann = os.path.join(data_dir, 'annotation')
    with open(os.path.join(ann, f'anno_{split}.json')) as f:
        d = json.load(f)
    id2label = {a['image_id']: a['category_id'] for a in d['annotations']}
    sol = os.path.join(data_dir, 'solution.csv')
    if split == 'test' and os.path.exists(sol):
        with open(sol) as f:
            id2label = {int(r['Id']): int(r['Category'].split()[0]) for r in csv.DictReader(f)}
    paths, targets = [], []
    n_missing = 0
    nested = bool(d['images']) and os.path.exists(os.path.join(data_dir, d['images'][0]['file_name'].split('/')[0], d['images'][0]['file_name']))
    for im in d['images']:
        top = im['file_name'].split('/')[0]
        p = os.path.join(data_dir, top, im['file_name']) if nested else os.path.join(data_dir, im['file_name'])
        if check_exists and not os.path.exists(p):
            n_missing += 1
            continue
        paths.append(p)
        targets.append(int(id2label.get(im['id'], -1)))
    if n_missing:
        print(f'[semi_inat] {split}: skipped {n_missing} images missing on disk')
    print(f'[semi_inat] {split}: {len(paths)} images')
    return paths, targets

class SemiINatDataset(BasicDataset):
    def __init__(self, alg, paths, targets, num_classes, is_ulb=False, transform=None,
                 strong_transform=None, lb_strong=False):
        super().__init__(alg, paths, targets, num_classes, transform=transform, is_ulb=is_ulb,
                         strong_transform=strong_transform)
        self.loader = dataset_parser.default_loader
        self.lb_strong = lb_strong

    def __sample__(self, idx):
        return self.loader(self.data[idx]), self.targets[idx]

    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        if not self.is_ulb and self.lb_strong and self.strong_transform is not None:
            img, _ = self.__sample__(idx)
            out['x_lb_s'] = self.strong_transform(img)
        return out

def _transforms(args):
    img_size = args.img_size
    crop_ratio = args.crop_ratio
    resize = int(math.floor(img_size / crop_ratio))

    transform_weak = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMGNET_MEAN, IMGNET_STD)])
    transform_strong = transforms.Compose([
        transforms.Resize((resize, resize)),
        RandomResizedCropAndInterpolation((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 10),
        transforms.ToTensor(),
        transforms.Normalize(IMGNET_MEAN, IMGNET_STD)])
    transform_val = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMGNET_MEAN, IMGNET_STD)])
    return transform_weak, transform_strong, transform_val

def _ds_alg(args, alg):
    return 'scssl' if getattr(args, 'imb_algorithm', None) == 'scssl' else alg

def get_semi_inat_split(args, alg, data_dir, split):
    p, t = _read_split(data_dir, split)
    return SemiINatDataset(_ds_alg(args, alg), p, t, args.num_classes, False, _transforms(args)[2])

def get_semi_inat(args, alg, data_dir, ulb_split='in'):
    transform_weak, transform_strong, transform_val = _transforms(args)
    ds_alg = _ds_alg(args, alg)
    num_classes = args.num_classes

    lb_p, lb_t = _read_split(data_dir, 'l_train')
    ulb_p, ulb_t = _read_split(data_dir, 'u_train_in')
    if ulb_split == 'in_out':
        p2, t2 = _read_split(data_dir, 'u_train_out', check_exists=True)
        ulb_p += p2
        ulb_t += t2
    ev_p, ev_t = _read_split(data_dir, 'val')

    lb_dset = SemiINatDataset(ds_alg, lb_p, lb_t, num_classes, False, transform_weak, transform_strong,
                              lb_strong=(alg == 'interlude'))
    ulb_dset = SemiINatDataset(ds_alg, ulb_p, ulb_t, num_classes, True, transform_weak, transform_strong)
    eval_dset = SemiINatDataset(ds_alg, ev_p, ev_t, num_classes, False, transform_val)
    return lb_dset, ulb_dset, eval_dset
