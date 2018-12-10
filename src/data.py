from pathlib import Path
from collections import Counter

import torch
import torch.utils.data
import cv2
import numpy as np

label_to_string = {
    0: 'Nucleoplasm',
    1: 'Nuclear membrane',
    2: 'Nucleoli',
    3: 'Nucleoli fibrillar center',
    4: 'Nuclear speckles',
    5: 'Nuclear bodies',
    6: 'Endoplasmic reticulum',
    7: 'Golgi apparatus',
    8: 'Peroxisomes',
    9: 'Endosomes',
    10: 'Lysosomes',
    11: 'Intermediate filaments',
    12: 'Actin filaments',
    13: 'Focal adhesion sites',
    14: 'Microtubules',
    15: 'Microtubule ends',
    16: 'Cytokinetic bridge',
    17: 'Mitotic spindle',
    18: 'Microtubule organizing center',
    19: 'Centrosome',
    20: 'Lipid droplets',
    21: 'Plasma membrane',
    22: 'Cell junctions',
    23: 'Mitochondria',
    24: 'Aggresome',
    25: 'Cytosol',
    26: 'Cytoplasmic bodies',
    27: 'Rods & rings'
}

class DataPaths:
    ROOT_DATA_PATH = Path("../data")
    TRAIN_IMAGES = Path(ROOT_DATA_PATH, "train")
    TRAIN_COMBINED_IMAGES = Path(ROOT_DATA_PATH, "train_combined")
    TRAIN_COMBINED_HPA_V18_IMAGES = Path(ROOT_DATA_PATH, "train_combined_HPAv18")
    TRAIN_ALL_COMBINED_IMAGES = Path(ROOT_DATA_PATH, "train_all_combined")
    TRAIN_LABELS = Path(ROOT_DATA_PATH, "train.csv")
    TRAIN_ALL_LABELS = Path(ROOT_DATA_PATH, "train_all.csv")
    TRAIN_HPA_V18_LABELS = Path(ROOT_DATA_PATH, "HPAv18RBGY_wodpl.csv")
    TEST_IMAGES = Path(ROOT_DATA_PATH, "test")
    TEST_COMBINED_IMAGES = Path(ROOT_DATA_PATH, "test_combined")


def open_rgby(path, with_image_wrapper=True): #a function that reads RGBY image
    colors = ['red','green','blue','yellow']
    flags = cv2.IMREAD_GRAYSCALE
    img_id = path.name[:36]
    img = [cv2.imread(f"{str(path)}_{color}.png", flags).astype(np.uint8) for color in colors]
    if with_image_wrapper:
        return Image(px=np.stack(img, axis=-1), name=img_id)
    else:
        return {
            "image": np.stack(img, axis=-1),
            "name": img_id
        }

def open_numpy(path, with_image_wrapper=True):
    img = np.load(path, allow_pickle=True).transpose(1, 2, 0)
    img_id = path.stem
    if with_image_wrapper:
        return Image(px=img, name=img_id)
    else:
        return {
            "image": img,
            "name": path.stem
        }

class Image:
    def __init__(self, px, name=None):
        self._px = px
        self._tensor = None
        self.name = name

    @property
    def px(self):
        return self._px

    @px.setter
    def px(self, px):
        self._px = px

    @property
    def tensor(self):
        return torch.from_numpy((self._px.astype(np.float32) / 255).transpose(2, 0, 1))

    def __getattr__(self, name):
        if name not in ["px", "tensor"]:
            return getattr(self.px, name)
        else:
            return getattr(self, name)


class ProteinClassificationDataset(torch.utils.data.Dataset):
    @staticmethod
    def collate_fn(batch):
        default_collate_batch = torch.utils.data.dataloader.default_collate(batch)
        x = default_collate_batch['input']
        #         torch.from_numpy([e['label'] for e in batch])
        y = {
            'name': [e['name'] for e in batch],
        }
        if batch[0].get('label') is not None:
            y['label'] = default_collate_batch['label']
        return x, y

    def __init__(self, inputs, open_image_fn=open_numpy, image_cached=False, augment_fn=None, normalize_fn=None, labels=None):
        self.inputs = inputs
        self.open_image_fn = open_image_fn
        self.image_cached = image_cached
        self.augment_fn = augment_fn
        self.normalize_fn = normalize_fn
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        ret = {}

        if self.image_cached:
            img = self.inputs[i]
        else:
            img = self.open_image_fn(self.inputs[i])

        if self.augment_fn is not None:
            img.px = self.augment_fn(img)
        img_tensor = img.tensor
        if self.normalize_fn is not None:
            img_tensor = self.normalize_fn(img_tensor)
        ret['input'] = img_tensor
        ret['name'] = img.name
        if self.labels is not None:
            ret['label'] = self.labels[i]
        return ret

def single_class_counter(labels, inv_proportions=True):
    flattened_classes = []
    for l in labels:
        flattened_classes.extend(l)
    cnt = Counter(flattened_classes)
    n_classes = sum(cnt.values())
    if inv_proportions:
        prop_cnt = {k: v/n_classes for k, v in cnt.items()}
        return sorted(prop_cnt.items())
    else:
        return sorted(cnt.items())