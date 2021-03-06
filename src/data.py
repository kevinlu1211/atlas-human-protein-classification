from pathlib import Path
from collections import Counter

import torch
import torch.utils.data
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from pytorch_toolbox.utils import make_one_hot

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

string_to_label = {
    v: k for k, v in label_to_string.items()
}


class DataPaths:
    ROOT_DATA_PATH = Path(__file__).parent.parent / 'data'
    TRAIN_IMAGES = Path(ROOT_DATA_PATH, "train")
    TRAIN_IMAGES_HPAv18 = Path(ROOT_DATA_PATH, "train_HPAv18")
    TRAIN_COMBINED_IMAGES = Path(ROOT_DATA_PATH, "train_combined")
    TRAIN_COMBINED_IMAGES_HPAv18 = Path(ROOT_DATA_PATH, "train_combined_HPAv18")
    TRAIN_LABELS = Path(ROOT_DATA_PATH, "train.csv")

    TRAIN_LABELS_HPAv18 = Path(ROOT_DATA_PATH, "HPAv18RBGY_wodpl.csv")
    TRAIN_LABELS_ALL_NO_DUPES = Path(ROOT_DATA_PATH, "train_all_no_dupes.csv")
    TEST_IMAGES = Path(ROOT_DATA_PATH, "test")
    TEST_COMBINED_IMAGES = Path(ROOT_DATA_PATH, "test_combined")


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

    def __init__(self, inputs, open_image_fn=None, image_cached=False, augment_fn=None, normalize_fn=None,
                 labels=None, tta_fn=None):
        # TODO: refactor to get rid of the lazy import
        from src.image import open_numpy
        self.inputs = inputs
        self.open_image_fn = open_numpy if open_image_fn is None else open_image_fn
        self.image_cached = image_cached
        self.augment_fn = augment_fn
        self.normalize_fn = normalize_fn
        self.labels = labels
        self.tta_fn = tta_fn

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

        if self.tta_fn is not None:
            img_tensor = self.tta_fn(img_tensor)

        ret['input'] = img_tensor
        ret['name'] = img.name

        if self.labels is not None:
            ret['label'] = self.labels[i]
        return ret


def single_class_counter(labels, smooth=0, inv_proportions=True):
    flattened_classes = []
    for l in labels:
        flattened_classes.extend(l)
    cnt = Counter(flattened_classes)
    n_classes = sum(cnt.values())
    if inv_proportions:
        prop_cnt = {k: (v + smooth * n_classes) / n_classes for k, v in cnt.items()}
        return sorted(prop_cnt.items())
    else:
        return sorted(cnt.items())


def create_combined_training_examples(kaggle_labels_df, hpa_labels_df, threshold=0.02):
    include_below = threshold
    included_labels = []
    for label, proportion in sorted(single_class_counter(kaggle_labels_df['Target'].values), key=lambda x: x[1]):
        if proportion < include_below:
            included_labels.append(label)
    rare_labels_from_hpa_df = hpa_labels_df[
        hpa_labels_df['Target'].map(lambda x: len(set(x) & set(included_labels)) > 0)]
    combined_training_df = pd.concat([kaggle_labels_df, rare_labels_from_hpa_df])
    return combined_training_df


def create_image_label_set(image_paths, label_paths, n_classes=28):
    image_paths = sorted(image_paths, key=lambda p: p.stem)
    labels_df = pd.read_csv(label_paths)
    labels_df['Target'] = [[int(i) for i in s.split()] for s in labels_df['Target']]
    labels_df = labels_df.sort_values(["Id"], ascending=[True])
    assert np.all(np.array([p.stem for p in image_paths]) == labels_df["Id"])
    labels_one_hot = make_one_hot(labels_df['Target'], n_classes=n_classes)
    return image_paths, labels_df, labels_one_hot


def mean_proportion_class_weights(all_labels):
    all_weights = []
    label_proportions = single_class_counter(all_labels)
    weight_lookup = {label: 1 / prop for label, prop in label_proportions}
    for labels in all_labels:
        weights = np.array([weight_lookup[l] for l in labels]).mean()
        all_weights.append(weights)
    return all_weights


def uniform_weights(all_labels):
    return [1 for _ in all_labels]


def create_sample_weights(all_labels, method="MEAN"):
    all_weights = []
    label_proportions = single_class_counter(all_labels)
    weight_lookup = {label: 1 / prop for label, prop in label_proportions}
    for labels in all_labels:
        if method == "MEAN":
            weights = np.array([weight_lookup[l] for l in labels]).mean()
        elif method == "MAX":
            weights = np.array([weight_lookup[l] for l in labels]).max()
        all_weights.append(weights)
    return all_weights


def match_prediction_probs_with_labels(prediction_probs, sort=True):
    assert prediction_probs.shape == (1, 28)
    matched = {name: prob for name, prob in zip(label_to_string.values(), prediction_probs[0])}
    if sort:
        return sorted(matched.items(), key=lambda x: x[1], reverse=True)
    else:
        return matched


sampler_weight_lookup = {
    "mean_proportion_class_weights": mean_proportion_class_weights,
    "create_sample_weights": create_sample_weights,
    "uniform_weights": uniform_weights
}

dataset_lookup = {
    "ProteinClassificationDataset": ProteinClassificationDataset
}

split_method_lookup = {
    "ShuffleSplit": ShuffleSplit,
    "StratifiedShuffleSplit": StratifiedShuffleSplit,
    "MultilabelStratifiedShuffleSplit": MultilabelStratifiedShuffleSplit
}
