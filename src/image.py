import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

from .data import open_numpy, DataPaths, label_to_string, string_to_label

def register_cmap():
    cdict1 = {'red':   ((0.0,  0.0, 0.0),
                       (1.0,  0.0, 0.0)),

             'green': ((0.0,  0.0, 0.0),
                       (0.75, 1.0, 1.0),
                       (1.0,  1.0, 1.0)),

             'blue':  ((0.0,  0.0, 0.0),
                       (1.0,  0.0, 0.0))}

    cdict2 = {'red':   ((0.0,  0.0, 0.0),
                       (0.75, 1.0, 1.0),
                       (1.0,  1.0, 1.0)),

             'green': ((0.0,  0.0, 0.0),
                       (1.0,  0.0, 0.0)),

             'blue':  ((0.0,  0.0, 0.0),
                       (1.0,  0.0, 0.0))}

    cdict3 = {'red':   ((0.0,  0.0, 0.0),
                       (1.0,  0.0, 0.0)),

             'green': ((0.0,  0.0, 0.0),
                       (1.0,  0.0, 0.0)),

             'blue':  ((0.0,  0.0, 0.0),
                       (0.75, 1.0, 1.0),
                       (1.0,  1.0, 1.0))}

    cdict4 = {'red': ((0.0,  0.0, 0.0),
                       (0.75, 1.0, 1.0),
                       (1.0,  1.0, 1.0)),

             'green': ((0.0,  0.0, 0.0),
                       (0.75, 1.0, 1.0),
                       (1.0,  1.0, 1.0)),

             'blue':  ((0.0,  0.0, 0.0),
                       (1.0,  0.0, 0.0))}

    plt.register_cmap(name='greens', data=cdict1)
    plt.register_cmap(name='reds', data=cdict2)
    plt.register_cmap(name='blues', data=cdict3)
    plt.register_cmap(name='yellows', data=cdict4)

def plot_rgby(image):
    register_cmap()
    _, axs = plt.subplots(1, 4, figsize=(24, 16))
    axs = axs.flatten()
    axs[0].imshow(image[:, :, 0], cmap='reds')
    axs[1].imshow(image[:, :, 1], cmap='greens')
    axs[2].imshow(image[:, :, 2], cmap='blues')
    axs[3].imshow(image[:, :, 3], cmap='yellows')
    plt.tight_layout()

def plot_rgb(image):
    plt.imshow(image[:,:,:3])


def get_image_with_id(image_id):
    paths = list(DataPaths.TRAIN_COMBINED_IMAGES_HPAv18.glob("*")) + \
            list(DataPaths.TRAIN_COMBINED_IMAGES.glob("*")) + \
            list(DataPaths.TEST_COMBINED_IMAGES.glob("*"))
    path, = [p for p in paths if p.stem == image_id]
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
    if k is None:
        return unique_classes
    return random.sample(unique_classes, k)

