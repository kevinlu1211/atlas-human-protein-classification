from pathlib import Path

import torch
import torch.utils.data
import cv2
import numpy as np

class DataPaths:
    ROOT_DATA_PATH = Path("../data")
    TRAIN_IMAGES = Path(ROOT_DATA_PATH, "train")
    TRAIN_LABELS = Path(ROOT_DATA_PATH, "train.csv")
    TEST_IMAGES = Path(ROOT_DATA_PATH, "test")


def open_rgby(path): #a function that reads RGBY image
    colors = ['red','green','blue','yellow']
    flags = cv2.IMREAD_GRAYSCALE
    img_id = path.name[:36]
    img = [cv2.imread(f"{str(path)}_{color}.png", flags).astype(np.uint8) for color in colors]
    return Image(px=np.stack(img, axis=-1), name=img_id)


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

    def __init__(self, inputs, image_cached=False, augment_fn=None, normalize_fn=None, labels=None):
        self.inputs = inputs
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
            img = open_rgby(self.inputs[i])

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
