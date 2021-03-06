import uuid
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

import torch

from .data import DataPaths, label_to_string, string_to_label


def register_cmap():
    cdict1 = {'red': ((0.0, 0.0, 0.0),
                      (1.0, 0.0, 0.0)),

              'green': ((0.0, 0.0, 0.0),
                        (0.75, 1.0, 1.0),
                        (1.0, 1.0, 1.0)),

              'blue': ((0.0, 0.0, 0.0),
                       (1.0, 0.0, 0.0))}

    cdict2 = {'red': ((0.0, 0.0, 0.0),
                      (0.75, 1.0, 1.0),
                      (1.0, 1.0, 1.0)),

              'green': ((0.0, 0.0, 0.0),
                        (1.0, 0.0, 0.0)),

              'blue': ((0.0, 0.0, 0.0),
                       (1.0, 0.0, 0.0))}

    cdict3 = {'red': ((0.0, 0.0, 0.0),
                      (1.0, 0.0, 0.0)),

              'green': ((0.0, 0.0, 0.0),
                        (1.0, 0.0, 0.0)),

              'blue': ((0.0, 0.0, 0.0),
                       (0.75, 1.0, 1.0),
                       (1.0, 1.0, 1.0))}

    cdict4 = {'red': ((0.0, 0.0, 0.0),
                      (0.75, 1.0, 1.0),
                      (1.0, 1.0, 1.0)),

              'green': ((0.0, 0.0, 0.0),
                        (0.75, 1.0, 1.0),
                        (1.0, 1.0, 1.0)),

              'blue': ((0.0, 0.0, 0.0),
                       (1.0, 0.0, 0.0))}

    plt.register_cmap(name='greens', data=cdict1)
    plt.register_cmap(name='reds', data=cdict2)
    plt.register_cmap(name='blues', data=cdict3)
    plt.register_cmap(name='yellows', data=cdict4)


def plot_rgby(image, title=None):
    register_cmap()
    _, axs = plt.subplots(1, 4, figsize=(24, 16))
    axs = axs.flatten()
    if title is not None:
        axs[0].set_title(title)
    axs[0].imshow(image[:, :, 0], cmap='reds')
    axs[1].imshow(image[:, :, 1], cmap='greens')
    axs[2].imshow(image[:, :, 2], cmap='blues')
    axs[3].imshow(image[:, :, 3], cmap='yellows')
    plt.tight_layout()


def plot_rgb(image):
    plt.imshow(image[:, :, :3])


def get_image_with_id(image_id, with_image_wrapper=False):
    paths = list(DataPaths.TRAIN_COMBINED_IMAGES_HPAv18.glob("*")) + \
            list(DataPaths.TRAIN_COMBINED_IMAGES.glob("*")) + \
            list(DataPaths.TEST_COMBINED_IMAGES.glob("*"))
    path, = [p for p in paths if p.stem == image_id]
    if with_image_wrapper:
        return open_numpy(path, with_image_wrapper=True)
    else:
        return open_numpy(path, with_image_wrapper=False)['image']


def get_label_with_id(image_id):
    df = pd.read_csv(DataPaths.TRAIN_ALL_LABELS)
    return df[df['Id'] == image_id]


def convert_to_labels(class_labels):
    converted_labels = []
    for label in class_labels:
        if isinstance(label, str):
            converted_labels.append(string_to_label[label])
        elif isinstance(label, int):
            assert label <= 27
            converted_labels.append(label)
    return np.array(converted_labels)


def convert_to_string(class_labels):
    converted_labels = []
    for label in class_labels:
        if isinstance(label, str):
            converted_labels.append(label)
        else:
            converted_labels.append(label_to_string[label])
    return np.array(converted_labels)


def get_image_from_class(class_labels, n_samples=1):
    converted_labels = convert_to_labels(class_labels)
    df = pd.read_csv(DataPaths.TRAIN_LABELS_ALL_NO_DUPES)
    df['Sorted Target'] = df['Target'].map(lambda t: np.array(sorted([int(l) for l in t.split()])))
    filtered_df = df[df['Sorted Target'].map(lambda x: np.all(converted_labels == x))]
    if len(filtered_df) == 0:
        raise ValueError(f"No images found for {[label_to_string[label] for label in class_labels]}")
    samples = filtered_df.sample(n_samples)
    sampled_ids = samples['Id']
    sampled_images = [get_image_with_id(image_id) for image_id in sampled_ids.values]
    return sampled_images


def get_unique_classes(k=10):
    df = pd.read_csv(DataPaths.TRAIN_LABELS_ALL_NO_DUPES)
    df['Target'] = df['Target'].map(lambda t: np.array(sorted([int(l) for l in t.split()])))
    df['Target String'] = df['Target'].map(lambda t: tuple([label_to_string[l] for l in t]))
    unique_classes = [ts for ts in df['Target String'].unique()]
    if k == "ALL":
        return unique_classes
    return random.sample(unique_classes, k)


def open_rgby(path, with_image_wrapper=True):  # a function that reads RGBY image
    colors = ['red', 'green', 'blue', 'yellow']
    flags = cv2.IMREAD_GRAYSCALE
    img_id = Path(path).name
    img = [cv2.imread(f"{str(path)}_{color}.jpg", flags).astype(np.uint8) for color in colors]
    if with_image_wrapper:
        return Image(px=np.stack(img, axis=-1), name=img_id)
    else:
        return {
            "image": np.stack(img, axis=-1),
            "name": img_id
        }


def open_numpy(path, with_image_wrapper=True):
    img = np.load(path, allow_pickle=True)
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
        self.name = str(uuid.uuid4()) if name is None else name

    @property
    def px(self):
        return self._px

    @px.setter
    def px(self, px):
        self._px = px

    @property
    def tensor(self):
        return torch.from_numpy((self._px.astype(np.float32)).transpose(2, 0, 1))

    def __getattr__(self, name):
        if name not in ["px", "tensor"]:
            return getattr(self.px, name)
        else:
            return getattr(self, name)