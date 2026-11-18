import os
os.environ['TORCHVISION_IMAGE_BACKEND'] = 'accimage'
import gc
import copy
import json
import random
from torchvision.datasets import ImageFolder
from PIL import Image
from torchvision import transforms
import math
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation, str_to_interp_mode
from semilearn.datasets.cv_datasets.datasetbase import BasicDataset
from tqdm import tqdm 

mean, std = {}, {}
mean['imagenet'] = [0.48109809, 0.45747185, 0.40785507]
std['imagenet'] = [0.26040889, 0.2532126, 0.26820634]

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        return pil_loader(path)

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def default_loader(path):
    import torchvision
    try:
        torchvision.set_image_backend('accimage')
    except:
        pass
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

def get_imagenet(args, alg, name, num_labels, num_classes, data_dir='./data', include_lb_to_ulb=True):
    img_size = args.img_size
    crop_ratio = args.crop_ratio

    transform_weak = transforms.Compose([
        transforms.RandomCrop(img_size, padding=int(img_size / 8)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean['imagenet'], std['imagenet'])
    ])

    transform_medium = transforms.Compose([
        transforms.RandomCrop(img_size, padding=int(img_size / 8)),
        transforms.RandomHorizontalFlip(),
        RandAugment(1, 10),
        transforms.ToTensor(),
        transforms.Normalize(mean['imagenet'], std['imagenet'])
    ])

    transform_strong = transforms.Compose([
        transforms.RandomCrop(img_size, padding=int(img_size / 8)),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 10),
        transforms.ToTensor(),
        transforms.Normalize(mean['imagenet'], std['imagenet'])
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean['imagenet'], std['imagenet'])
    ])

    data_dir = os.path.join(data_dir, name.lower())

    dataset = ImagenetDataset(root=os.path.join(data_dir, "train"), transform=transform_weak, ulb=False, alg=alg)
    percentage = num_labels / len(dataset)
    
    print(f"Making lb dset / {percentage}...")
    lb_dset = ImagenetDataset(root=os.path.join(data_dir, "train"), transform=transform_weak, ulb=False, alg=alg, percentage=percentage)

    print(f"Making ulb dset...")
    ulb_dset = ImagenetDataset(root=os.path.join(data_dir, "train"), transform=transform_weak, alg=alg, ulb=True, percentage=1.0, medium_transform=transform_medium, strong_transform=transform_strong, include_lb_to_ulb=include_lb_to_ulb, lb_index=lb_dset.lb_idx)

    eval_dset = ImagenetDataset(root=os.path.join(data_dir, "val"), transform=transform_val, alg=alg, ulb=False, balance_eval=False)

    if alg == 'ede' or alg == 'scssl':
        import numpy as np
        num_classes = len(set(dataset.targets))
        num_data_per_cls = [0 for _ in range(num_classes)]
        for l in dataset.targets:
            num_data_per_cls[l] += 1
        num_labeled_data_per_cls = [int(np.around(n * percentage)) for n in num_data_per_cls]
        arr = np.array(num_labeled_data_per_cls)
        tar_index = np.argsort(-arr)
        tar_index = tar_index.tolist()
        for idx in range(len(lb_dset.targets)):
            lb_dset.targets[idx] = tar_index.index(lb_dset.targets[idx])
        for idx in range(len(ulb_dset.targets)):
            ulb_dset.targets[idx] = tar_index.index(ulb_dset.targets[idx])
        for idx in range(len(eval_dset.targets)):
            eval_dset.targets[idx] = tar_index.index(eval_dset.targets[idx])

    return lb_dset, ulb_dset, eval_dset
    

class ImagenetDataset(BasicDataset, ImageFolder):
    def __init__(self, root, transform, ulb, alg, medium_transform=None, strong_transform=None, percentage=-1, include_lb_to_ulb=True, lb_index=None, balance_eval=False):
        self.alg = alg
        self.is_ulb = ulb
        self.percentage = percentage
        self.transform = transform
        self.root = root
        self.include_lb_to_ulb = include_lb_to_ulb
        self.lb_index = lb_index
        self.balance_eval = balance_eval

        is_valid_file = None
        extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
        classes, class_to_idx = self.find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = default_loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.data = [s[0] for s in samples]
        self.targets = [s[1] for s in samples]

        self.medium_transform = medium_transform
        if self.medium_transform is None:
            if self.is_ulb:
                assert self.alg not in ['sequencematch'], f"alg {self.alg} requires strong augmentation"
        self.strong_transform = strong_transform
        if self.strong_transform is None:
            if self.is_ulb:
                assert self.alg not in ['fullysupervised', 'supervised', 'pseudolabel', 'vat', 'pimodel', 'meanteacher', 'mixmatch', 'refixmatch'], f"alg {self.alg} requires strong augmentation"

    def find_classes(self, directory: str):
        def natural_sort_key(s):
            import re
            return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
        classes = sorted((entry.name for entry in os.scandir(directory) if entry.is_dir()), key=natural_sort_key)
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __sample__(self, index):
        path = self.data[index]
        sample = self.loader(path)
        target = self.targets[index]
        return sample, target

    def make_dataset(
            self,
            directory,
            class_to_idx,
            extensions=None,
            is_valid_file=None,
    ):
        instances = []
        directory = os.path.expanduser(directory)
        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
        if extensions is not None:
            def is_valid_file(x: str) -> bool:
                return x.lower().endswith(extensions)
        
        lb_idx = {}
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                random.shuffle(fnames)
                if self.balance_eval:
                    fnames = fnames[:50]
                if self.percentage != -1:
                    fnames = fnames[:int(len(fnames) * self.percentage)]
                if self.percentage != -1 and self.is_ulb == False:
                    lb_idx[target_class] = fnames
                for fname in fnames:
                    if not self.include_lb_to_ulb:
                        if fname in self.lb_index[target_class]:
                            continue
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        item = path, class_index
                        instances.append(item)
        gc.collect()
        self.lb_idx = lb_idx
        return instances

