import sys

sys.path.append("..")

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import time
import pickle
from pathlib import Path
from functools import partial
import random
from collections import defaultdict

import cv2
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
import torch
import torch.utils.data
import torch.nn as nn

from src.data import DataPaths

from pytorch_toolbox.fastai import fastai
from pytorch_toolbox.utils.core import to_numpy
from pytorch_toolbox.fastai.fastai import vision
from pytorch_toolbox.vision.utils import normalize, denormalize, tensor2img
from pytorch_toolbox.fastai_extensions.loss import LossWrapper, FocalLoss, SoftF1Loss
from pytorch_toolbox.fastai_extensions.basic_data import DataBunch

# CONFIG_FILE = Path("../configs/cbam_resnet18.yml")
CONFIG_FILE = Path("../configs/iafoss_resnet34.yml")
# CONFIG_FILE = Path("../configs/gapnet.yml")
# CONFIG_FILE = Path("../configs/iafoss_resnet50.yml")
# CONFIG_FILE = Path("../configs/cbam_resnet50.yml")
# CONFIG_FILE = Path("../configs/cbam_resnet101.yml")

ROOT_SAVE_PATH = Path("/media/hd1/data/Kaggle/human-protein-image-classification/results")
SAVE_FOLDER_NAME = f"{CONFIG_FILE.stem}_{time.strftime('%Y%m%d-%H%M%S')}"
RESULTS_SAVE_PATH = ROOT_SAVE_PATH / SAVE_FOLDER_NAME
RESULTS_SAVE_PATH.mkdir(exist_ok=True, parents=True)

with CONFIG_FILE.open("r") as f:
    config = yaml.load(f)

with (RESULTS_SAVE_PATH / "config.yml").open('w') as yaml_file:
    yaml.dump(config, yaml_file, default_flow_style=False)

from pprint import pprint

pprint(config)


def extract_name_and_parameters(config, key):
    name = config[key].get('name')
    parameters = config[key].get('parameters', dict())
    return name, parameters


DEBUG = True

# 1. Generate the training data
from pytorch_toolbox.utils import make_one_hot

# Get paths for training
train_paths = sorted(list(DataPaths.TRAIN_ALL_COMBINED_IMAGES.glob("*")), key=lambda p: p.stem)
labels_df = pd.read_csv(DataPaths.TRAIN_ALL_LABELS)
test_paths = sorted(list(DataPaths.TEST_COMBINED_IMAGES.glob("*")), key=lambda p: p.stem)

# train_paths = sorted(list(DataPaths.TRAIN_COMBINED_HPA_V18_IMAGES.glob("*")), key=lambda p: p.stem)
# labels_df = pd.read_csv(DataPaths.TRAIN_HPA_V18_LABELS)
# test_paths = sorted(list(DataPaths.TEST_COMBINED_IMAGES.glob("*")), key=lambda p: p.stem)

# train_paths = sorted(list(DataPaths.TRAIN_COMBINED_IMAGES.glob("*")), key=lambda p: p.stem)
# labels_df = pd.read_csv((DataPaths.TRAIN_LABELS))
# test_paths = sorted(list(DataPaths.TEST_COMBINED_IMAGES.glob("*")), key=lambda p: p.stem)

labels_df['Target'] = [[int(i) for i in s.split()] for s in labels_df['Target']]
labels_df = labels_df.sort_values(["Id"], ascending=[True])
train_labels_one_hot = make_one_hot(labels_df['Target'])

assert np.all(np.array([p.stem for p in train_paths]) == labels_df["Id"])

if DEBUG:
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

    msss = partial(MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42).split, X=train_paths,
                                                                                               y=train_labels_one_hot)
    _, idx = next(iter(msss()))
    train_paths = np.array(train_paths)[idx]
    train_labels_one_hot = np.array(train_labels_one_hot)[idx]

# 2. Transformation / normalization of images
from pytorch_toolbox.vision.transforms import *


def albumentations_transform_wrapper(image, augment_fn):
    augmentation = augment_fn(image=image.px)
    return augmentation['image']


augment_fn_lookup = {
    "very_simple_aug": very_simple_aug,
    "simple_aug": simple_aug,
    "simple_aug_lower_prob": simple_aug_lower_prob,
    "resize_aug": resize_aug,
}

augment_fn_name, augment_fn_parameters = extract_name_and_parameters(config, "augment_fn")

augment_fn = partial(albumentations_transform_wrapper,
                     augment_fn=augment_fn_lookup[augment_fn_name](**augment_fn_parameters))

four_channel_image_net_stats = {
    'mean': [0.485, 0.456, 0.406, 0.485],
    'sd': [0.229, 0.224, 0.224, 0.229]
}
four_channel_image_net_normalize = partial(normalize, **four_channel_image_net_stats)
four_channel_image_net_denormalize = partial(denormalize, **four_channel_image_net_stats)

normalize_fn_lookup = {
    "four_channel_image_net_normalize": four_channel_image_net_normalize,
    "identity": lambda x: x
}

denormalize_fn_lookup = {
    "four_channel_image_net_denormalize": four_channel_image_net_denormalize,
    "identity": lambda x: x
}

normalize_fn_name, normalize_fn_parameters = extract_name_and_parameters(config, "normalize_fn")
normalize_fn = partial(normalize_fn_lookup[normalize_fn_name], **normalize_fn_parameters)
denormalize_fn_name, denormalize_fn_parameters = extract_name_and_parameters(config, "denormalize_fn")
denormalize_fn = partial(denormalize_fn_lookup[denormalize_fn_name], **denormalize_fn_parameters)

# 3. Create the splits
from collections import Counter
from pprint import pprint
from sklearn.model_selection import ShuffleSplit
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from src.data import single_class_counter

split_method_lookup = {
    "ShuffleSplit": ShuffleSplit,
    "MultilabelStratifiedShuffleSplit": MultilabelStratifiedShuffleSplit
}

split_method_name, split_method_parameters = extract_name_and_parameters(config, "split_method")

# Data splitting
split_method = partial(split_method_lookup[split_method_name](**split_method_parameters).split, X=train_paths,
                       y=train_labels_one_hot)

train_idx, val_idx = next(iter(split_method()))




print("Training distribution:")
pprint(single_class_counter(labels_df['Target'].iloc[train_idx].values))

print("Validation distribution:")
pprint(single_class_counter(labels_df['Target'].iloc[val_idx].values))

# 4. Create the data bunch which wraps our dataset
from pytorch_toolbox.fastai_extensions.basic_data import DeviceDataLoader
from src.data import ProteinClassificationDataset, open_numpy
import torch
import torch.utils.data
from torch.utils.data import WeightedRandomSampler


def create_data_bunch(train_ds, val_ds, test_ds, sampler=None, **data_bunch_parameters):
    data = DataBunch.create(train_ds, val_ds, test_ds,
                            collate_fn=train_ds.collate_fn,
                            sampler=sampler,
                            **data_bunch_parameters)

    return data


dataset_lookup = {
    "ProteinClassificationDataset": ProteinClassificationDataset
}
dataset_name, dataset_parameters = extract_name_and_parameters(config, 'dataset')
dataset = partial(dataset_lookup[dataset_name], **dataset_parameters)

sampler_lookup = {
    "WeightedRandomSampler": WeightedRandomSampler
}


def generate_weight_lookup(all_labels):
    label_proportions = single_class_counter(all_labels)
    weight_lookup = {label: 1 / prop for label, prop in label_proportions}
    return weight_lookup


def create_weights(all_labels):
    all_weights = []
    weight_lookup = generate_weight_lookup(all_labels)
    for labels in all_labels:
        weights = np.array([weight_lookup[l] for l in labels]).mean()
        all_weights.append(weights)
    return all_weights


weights = create_weights(labels_df['Target'].values[train_idx])
sampler_name, sampler_parameters = extract_name_and_parameters(config, 'sampler')
sampler = sampler_lookup.get(sampler_name)

if sampler is not None: sampler = sampler(weights=weights, num_samples=len(weights), **sampler_parameters)
train_ds = dataset(inputs=np.array(train_paths)[train_idx],
                   open_image_fn=open_numpy,
                   labels=np.array(train_labels_one_hot)[train_idx],
                   augment_fn=augment_fn,
                   normalize_fn=normalize_fn)
val_ds = dataset(inputs=np.array(train_paths)[val_idx],
                 open_image_fn=open_numpy,
                 labels=np.array(train_labels_one_hot)[val_idx],
                 normalize_fn=normalize_fn)
test_ds = dataset(inputs=np.array(test_paths),
                  open_image_fn=open_numpy,
                  normalize_fn=normalize_fn)

data = create_data_bunch(train_ds=train_ds,
                         val_ds=val_ds,
                         test_ds=test_ds,
                         sampler=sampler,
                         **config["data_bunch"].get("parameters", dict()))
from tqdm import tqdm

# cnt = Counter()
# for x, y in tqdm(data.train_dl):
#     for l in np.where(y['label'].cpu().data.numpy() == 1)[1]:
#         cnt[l] += 1
# print(cnt)

# 5. Initialize the model
import pytorch_toolbox.fastai.fastai as fastai
from pytorch_toolbox.fastai_extensions.basic_train import Learner
from pytorch_toolbox.utils import num_parameters
from src.models import *

model_lookup = {
    "resnet18_four_channel_input": resnet18_four_channel_input,
    "resnet34_four_channel_input": resnet34_four_channel_input,
    "resnet34_four_channel_input_one_fc": resnet34_four_channel_input_one_fc,
    "resnet50_four_channel_input": resnet50_four_channel_input,
    "cbam_resnet18_four_channel_input_one_fc": cbam_resnet18_four_channel_input_one_fc,
    "cbam_resnet34_four_channel_input": cbam_resnet34_four_channel_input,
    "cbam_resnet50_four_channel_input": cbam_resnet50_four_channel_input,
    "cbam_resnet50_four_channel_input_one_fc": cbam_resnet50_four_channel_input_one_fc,
    "cbam_resnet101_four_channel_input": cbam_resnet101_four_channel_input,
    "GapNet": GapNet
}

model_name, model_parameters = extract_name_and_parameters(config, "model")

model = model_lookup[model_name](**model_parameters)
# print(model)
# print(f"Number of parameters: {num_parameters(model, only_trainable=False)}")
# print(f"Number of trainable parameters: {num_parameters(model, only_trainable=True)}")

# TODO: figure out how create the layer groups
n_starting_layers = len(fastai.flatten_model(model[:6]))
n_middle_layers = len(fastai.flatten_model(model[6:9]))
n_head = len(fastai.flatten_model(model[9:]))
layer_groups = fastai.split_model_idx(model, [n_starting_layers, n_starting_layers + n_middle_layers])

# 6. Initialize the callbacks
from pytorch_toolbox.fastai_extensions.callbacks import NameExtractionTrainer, GradientClipping
from src.callbacks import OutputRecorder

callback_lookup = {
    "NameExtractionTrainer": NameExtractionTrainer
}

learner_callback_lookup = {
    "OutputRecorder": partial(OutputRecorder, save_path=RESULTS_SAVE_PATH,
                              save_img_fn=partial(tensor2img, denorm_fn=denormalize_fn)),
    "GradientClipping": GradientClipping
}
callbacks = []
for callback in config.get('callbacks', list()):
    name = callback['name']
    parameters = callback.get('parameters', dict())
    callbacks.append(callback_lookup[name](**parameters))

callback_fns = []
for callback_fn in config.get('callback_fns', list()):
    name = callback_fn['name']
    parameters = callback_fn.get('parameters', dict())
    callback_fns.append(partial(learner_callback_lookup[name], **parameters))

# 7. Initialize the loss func:
loss_lookup = {
    "FocalLoss": FocalLoss,
    "SoftF1Loss": SoftF1Loss
}

loss_funcs = []
for loss_func in config.get('loss_func', list()):
    name = loss_func['name']
    parameters = loss_func.get('parameters', dict())
    loss_funcs.append(loss_lookup[name](**parameters))

# 8. Define the metrics
from pytorch_toolbox.metrics import accuracy, f1_soft

metric_lookup = {
    "accuracy": accuracy,
    "f1_soft": f1_soft
}

metrics = []
for metric in config.get('metrics', list()):
    name = metric['name']
    parameters = metric.get('parameters', dict())
    metrics.append(partial(metric_lookup[name], **parameters))

# 9. Initialize the learner class
learner = Learner(data,
                  layer_groups=layer_groups,
                  model=model,
                  loss_func=LossWrapper(loss_funcs),
                  callbacks=callbacks,
                  callback_fns=callback_fns,
                  metrics=metrics)

# Now for the training scheme
from src.training import *

training_scheme_lookup = {
    "iafoss_training_scheme": iafoss_training_scheme,
    "training_scheme_1": training_scheme_1,
    "training_scheme_2": training_scheme_2,
    "training_scheme_3": training_scheme_3,
    "training_scheme_4": training_scheme_4,
    "training_scheme_gapnet": training_scheme_gapnet,
}

training_scheme_name, training_scheme_parameters = extract_name_and_parameters(config, "training_scheme")
training_scheme_lookup[training_scheme_name](learner=learner, **training_scheme_parameters)
learner.save(RESULTS_SAVE_PATH / 'model')
learner.load(RESULTS_SAVE_PATH / 'model')


class ResultRecorder(fastai.Callback):
    _order = -10

    def __init__(self):
        self.names = []
        self.prob_preds = []
        self.targets = []

    def on_batch_begin(self, last_input, last_target, train, **kwargs):
        if train:
            self.phase = 'TRAIN'
        else:
            label = last_target.get('label')
            if label is not None:
                self.phase = 'VAL'
            else:
                self.phase = 'TEST'
                #         inputs = tensor2img(last_input, denorm_fn=image_net_denormalize)
                #         self.inputs.extend(inputs)
        self.names.extend(last_target['name'])
        if self.phase == 'TRAIN' or self.phase == 'VAL':
            label = to_numpy(last_target['label'])
            self.targets.extend(label)

    def on_loss_begin(self, last_output, **kwargs):
        prob_pred = to_numpy(torch.sigmoid(last_output))
        self.prob_preds.extend(prob_pred)


res_recorder = ResultRecorder()
learner.predict_on_dl(dl=learner.data.valid_dl, callbacks=[res_recorder])

from sklearn.metrics import f1_score
import scipy.optimize as opt


def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))


def F1_soft(preds, targs, th=0.5, d=50.0):
    preds = sigmoid_np(d * (preds - th))
    targs = targs.astype(np.float)
    score = 2.0 * (preds * targs).sum(axis=0) / ((preds + targs).sum(axis=0) + 1e-6)
    return score


def fit_val(x, y):
    params = 0.5 * np.ones(28)
    wd = 1e-5
    error = lambda p: np.concatenate((F1_soft(x, y, p) - 1.0,
                                      wd * (p - 0.5)), axis=None)
    p, success = opt.leastsq(error, params)
    return p


pred_probs = np.stack(res_recorder.prob_preds)
targets = np.stack(res_recorder.targets)
print(targets.shape)
print(pred_probs.shape)

th = fit_val(pred_probs, targets)
th[th < 0.1] = 0.1
print('Thresholds: ', th)
print('F1 macro: ', f1_score(targets, pred_probs > th, average='macro'))
print('F1 macro (th = 0.5): ', f1_score(targets, pred_probs > 0.5, average='macro'))
print('F1 micro: ', f1_score(targets, pred_probs > th, average='micro'))

learner.load(RESULTS_SAVE_PATH / 'model')
res_recorder = ResultRecorder()
learner.predict_on_dl(dl=learner.data.test_dl, callbacks=[res_recorder])

names = np.stack(res_recorder.names)
pred_probs = np.stack(res_recorder.prob_preds)
print(names.shape)
print(pred_probs.shape)

predicted = []
for pred in tqdm_notebook(pred_probs):
    classes = [str(c) for c in np.where(pred > th)[0]]
    if len(classes) == 0:
        classes = [str(np.argmax(pred[0]))]
    predicted.append(" ".join(classes))

submission_df = pd.DataFrame({
    "Id": names,
    "Predicted": predicted
})

submission_df.to_csv(RESULTS_SAVE_PATH / "submission.csv", index=False)

with open(RESULTS_SAVE_PATH / "config.yml", 'w') as yaml_file:
    yaml.dump(config, yaml_file, default_flow_style=False)
