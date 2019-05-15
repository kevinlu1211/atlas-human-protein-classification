import numpy as np
import cv2
import os
import glob
from pathlib import Path

from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
from miniutils.progress_bar import parallel_progbar, progbar

from src.training.data import DataPaths

COLORS = ['red', 'green', 'blue', 'yellow']
KAGGLE_IMAGE_EXTENSION = "png"
HPAV18_IMAGE_EXTENSION = "jpg"


def read_channels(root_dir, img_id, img_extension="tif"):
    channels = []
    for color in COLORS:
        image_path = root_dir / f"{img_id}_{color}.{img_extension}"
        channel = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        channels.append(channel)
    channels = np.array(channels)
    assert channels.max() <= 255
    return channels.astype(np.uint8).transpose(1, 2, 0)


def get_image_ids_HPAv18(root_dir):
    image_file_paths = root_dir.glob(f"*.{HPAV18_IMAGE_EXTENSION}")
    img_ids = []
    for fp in progbar(image_file_paths):
        img_ids.append("_".join(fp.stem.split("_")[:-1]))
    img_ids = list(set(img_ids))
    return img_ids


def get_image_ids(root_dir):
    image_file_paths = root_dir.glob(f"*.{KAGGLE_IMAGE_EXTENSION}")
    img_ids = []
    for fp in progbar(image_file_paths):
        img_ids.append(fp.stem.split("_")[0])
    img_ids = list(set(img_ids))
    return img_ids


def make_image_path(root_dir, img_id):
    path = root_dir / f"{img_id}.npy"
    return path

def save_img(data):
    root_dir, save_dir, img_id, img_extension, size = data
    img_path = make_image_path(save_dir, img_id)
    channels = cv2.resize(read_channels(root_dir, img_id, img_extension), size)
    np.save(img_path, channels, allow_pickle=True)


print("Combining HPAv18 images")
root_dir = DataPaths.TRAIN_IMAGES_HPAv18
save_dir = DataPaths.TRAIN_COMBINED_IMAGES_HPAv18
save_dir.mkdir(parents=True, exist_ok=True)
print("Creating image paths")
img_ids = get_image_ids_HPAv18(root_dir)
SIZE = (1024, 1024)
n_samples = len(img_ids)
img_details = zip([root_dir] * n_samples,
                [save_dir] * n_samples,
                img_ids,
                [HPAV18_IMAGE_EXTENSION] * n_samples,
                [SIZE] * n_samples)
parallel_progbar(save_img, img_details)

print("Combining Kaggle training images")
root_dir = DataPaths.TRAIN_IMAGES
save_dir = DataPaths.TRAIN_COMBINED_IMAGES
save_dir.mkdir(parents=True, exist_ok=True)
print("Creating image paths")
img_ids = get_image_ids(root_dir)
SIZE = (1024, 1024)
n_samples = len(img_ids)
img_details = zip([root_dir] * n_samples,
                [save_dir] * n_samples,
                img_ids,
                [KAGGLE_IMAGE_EXTENSION] * n_samples,
                [SIZE] * n_samples)
parallel_progbar(save_img, img_details)

print("Combining Kaggle testing images")
root_dir = DataPaths.TEST_IMAGES
save_dir = DataPaths.TEST_COMBINED_IMAGES
save_dir.mkdir(parents=True, exist_ok=True)
print("Creating image paths")
img_ids = get_image_ids(root_dir)
SIZE = (1024, 1024)
n_samples = len(img_ids)
img_details = zip([root_dir] * n_samples,
                [save_dir] * n_samples,
                img_ids,
                [KAGGLE_IMAGE_EXTENSION] * n_samples,
                [SIZE] * n_samples)
parallel_progbar(save_img, img_details)