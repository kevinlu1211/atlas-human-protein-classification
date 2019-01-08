# coding: utf-8


import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

# In[3]:


import time
import pickle
from pathlib import Path
from functools import partial
import random
from collections import defaultdict, Counter
from pprint import pprint

import cv2
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
import torch
import torch.utils.data
from torch.utils.data import WeightedRandomSampler
import torch.nn as nn
from torchsummary import summary

sys.path.append("../..")
from src.data import DataPaths, create_image_label_set, make_one_hot, open_rgby
from src.data import ProteinClassificationDataset, open_numpy, mean_proportion_class_weights, dataset_lookup, \
    sampler_weight_lookup, split_method_lookup, single_class_counter
from src.models import model_lookup
from src.callbacks import OutputRecorder
from src.image import plot_rgby

import pytorch_toolbox.fastai.fastai as fastai
from pytorch_toolbox.utils.core import to_numpy

from pytorch_toolbox.fastai.fastai import vision
from pytorch_toolbox.vision.utils import normalize, denormalize, tensor2img
from pytorch_toolbox.vision import augment_fn_lookup, albumentations_transform_wrapper
from pytorch_toolbox.vision.utils import denormalize_fn_lookup, normalize_fn_lookup

from pytorch_toolbox.fastai.fastai.callbacks import CSVLogger
from pytorch_toolbox.fastai_extensions.basic_train import Learner
from pytorch_toolbox.fastai_extensions.loss import LossWrapper, FocalLoss, SoftF1Loss, loss_lookup
from pytorch_toolbox.fastai_extensions.basic_data import DataBunch
from pytorch_toolbox.fastai_extensions.callbacks import callback_lookup
from pytorch_toolbox.metrics import metric_lookup

# ### First load in the data

# In[4]:


# CONFIG_FILE = Path("../configs/resnet34_d.yml")
CONFIG_FILE = Path("../configs/se_resnext50_32x4d_image_patches.yml")
ROOT_SAVE_PATH = Path("/media/hd/Kaggle/human-protein-image-classification/results")
SAVE_FOLDER_NAME = f"{CONFIG_FILE.stem}_{time.strftime('%Y%m%d-%H%M%S')}"
RESULTS_SAVE_PATH = ROOT_SAVE_PATH / SAVE_FOLDER_NAME
RESULTS_SAVE_PATH.mkdir(exist_ok=True, parents=True)

with CONFIG_FILE.open("r") as f:
    config = yaml.load(f)


with (RESULTS_SAVE_PATH / "config.yml").open('w') as yaml_file:
    yaml.dump(config, yaml_file, default_flow_style=False)

from pprint import pprint

pprint(config)

# In[5]:


# In[6]:


def extract_name_and_parameters(config, key):
    name = config.get(key, dict()).get('name')
    parameters = config.get(key, dict()).get('parameters', dict())
    return name, parameters


# #### Load data

# In[7]:


train_label_paths = DataPaths.TRAIN_ALL_LABELS
train_image_paths = list(DataPaths.TRAIN_RGBY_IMAGES.glob("*")) + list(DataPaths.TRAIN_RGBY_IMAGES_HPA_V18.glob("*"))
img_ids = sorted(list(set([p.name.split("_crop")[0] for p in train_image_paths])))

# In[8]:


test_paths = list(DataPaths.TEST_COMBINED_IMAGES.glob("*"))

# #### Create training DataFrame

# In[9]:


labels_df = pd.read_csv(train_label_paths)
labels_df['Target'] = [[int(i) for i in s.split()] for s in labels_df['Target']]
labels_df = labels_df.sort_values(["Id"], ascending=[True])
assert np.all(np.array(img_ids) == labels_df["Id"])
labels_one_hot = make_one_hot(labels_df['Target'], n_classes=28)
print("labels_df.shape")
print(labels_df.shape)
print("len(labels_one_hot)")
print(len(labels_one_hot))

# Get the names for each patch

# In[10]:


img_ids_crops = sorted(list(set(["_".join(p.name.split("_")[:-1]) for p in train_image_paths])))

# In[11]:


lookup = {}
for i in tqdm_notebook(range(len(labels_df))):
    row = labels_df.iloc[i].to_dict()
    lookup[row['Id']] = row['Target']

# Create new df with oversampled patches for rare labels

# In[12]:


rows = []
for img_ids_crop in tqdm_notebook(img_ids_crops):
    base_id = img_ids_crop.split("_crop")[0]
    target = lookup[base_id]
    row = {
        'Id': img_ids_crop,
        'BaseId': base_id,
        'Target': target
    }
    rows.append(row)


# #### Create the data split

# In[13]:


def create_split_indices(config, train_paths, train_labels_one_hot):
    split_method_name, split_method_parameters = extract_name_and_parameters(config, "split_method")

    # Data splitting
    split_method = partial(split_method_lookup[split_method_name](**split_method_parameters).split, X=train_paths,
                           y=train_labels_one_hot)

    train_idx, val_idx = next(iter(split_method()))

    return train_idx, val_idx


train_idx, val_idx = create_split_indices(config, labels_df['Id'], labels_one_hot)
train_ids, val_ids = labels_df['Id'].values[train_idx], labels_df['Id'].values[val_idx]

# Now create the train / val rows

# In[14]:


train_ids_lookup = Counter(train_ids)
train_df_rows = []
val_df_rows = []
for row in rows:
    if train_ids_lookup.get(row['BaseId']) is None:
        val_df_rows.append({
            'Id': row['Id'],
            'Target': row['Target'],
            'BaseId': row['BaseId']
        })
    else:
        train_df_rows.append({
            'Id': row['Id'],
            'Target': row['Target']
        })

# Oversample rare classes for training set

# In[15]:


rares = [8, 9, 10, 15, 16, 17, 20, 24, 26, 27]
n_oversample = 2
oversampled_train_df_rows = []
for row in train_df_rows:
    target = row['Target']
    oversampled_train_df_rows.append(row)
    for t in target:
        if t in rares:
            for _ in range(n_oversample - 1):
                oversampled_train_df_rows.append(row)

# Create the training / val data frame

# In[16]:


train_df = pd.DataFrame(oversampled_train_df_rows)
val_df = pd.DataFrame(val_df_rows)



# Now create the paths to the images

# In[20]:


train_paths = np.array([str(DataPaths.TRAIN_RGBY_IMAGES / img_id) for img_id in train_df['Id'].values])
val_paths = np.array([str(DataPaths.TRAIN_RGBY_IMAGES / img_id) for img_id in val_df['Id'].values])



# In[21]:


train_labels_one_hot = make_one_hot(train_df['Target'], n_classes=28)
val_labels_one_hot = make_one_hot(val_df['Target'], n_classes=28)

# In[22]:


sample_img = open_rgby(train_paths[0])
plot_rgby(sample_img.px)


# #### Load augmentation functions

# In[23]:


def load_augmentation_functions(config):
    augment_fn_name, augment_fn_parameters = extract_name_and_parameters(config, "augment_fn")

    augment_fn = partial(albumentations_transform_wrapper,
                         augment_fn=augment_fn_lookup[augment_fn_name](**augment_fn_parameters))

    normalize_fn_name, normalize_fn_parameters = extract_name_and_parameters(config, "normalize_fn")
    normalize_fn = partial(normalize_fn_lookup[normalize_fn_name], **normalize_fn_parameters)
    denormalize_fn_name, denormalize_fn_parameters = extract_name_and_parameters(config, "denormalize_fn")
    denormalize_fn = partial(denormalize_fn_lookup[denormalize_fn_name], **denormalize_fn_parameters)
    return augment_fn, normalize_fn, denormalize_fn


# In[24]:


augment_fn, normalize_fn, denormalize_fn = load_augmentation_functions(config)

# In[25]:


# Uncomment to see the distribution of the dataset

# In[26]:


# print("Training distribution:")
# pprint(single_class_counter(labels_df['Target'].iloc[train_idx].values))

# print("Validation distribution:")
# pprint(single_class_counter(labels_df['Target'].iloc[val_idx].values))


# #### Create data bunch that wraps our dataset

# In[27]:


def create_sampler(config):
    sampler_weight_fn_name, sampler_weight_fn_parameters = extract_name_and_parameters(config, 'sample_weight_fn')
    if sampler_weight_fn_name is not None:
        sampler_weight_fn = partial(sampler_weight_lookup[sampler_weight_fn_name], **sampler_weight_fn_parameters)
        weights = np.array(sampler_weight_fn(labels_df['Target'].values[train_idx]))
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights))
        return sampler, weights
    else:
        sampler = None
        return sampler, None


def create_data_bunch(config, sampler, train_paths, val_paths, train_labels_one_hot, val_labels_one_hot,
                      test_paths=None):
    dataset_name, dataset_parameters = extract_name_and_parameters(config, 'dataset')
    dataset = partial(dataset_lookup[dataset_name], **dataset_parameters)

    train_ds = dataset(inputs=np.array(train_paths)[:100],
                       open_image_fn=open_rgby,
                       labels=np.array(train_labels_one_hot),
                       augment_fn=augment_fn,
                       normalize_fn=normalize_fn)
    val_ds = dataset(inputs=np.array(val_paths),
                     open_image_fn=open_rgby,
                     augment_fn=augment_fn_lookup["resize_aug"](384, 384),
                     labels=np.array(val_labels_one_hot),
                     normalize_fn=normalize_fn)
    test_ds = dataset(inputs=np.array(test_paths),
                      open_image_fn=open_rgby,
                      normalize_fn=normalize_fn)

    data = DataBunch.create(train_ds, val_ds, test_ds,
                            collate_fn=train_ds.collate_fn,
                            sampler=sampler,
                            **config["data_bunch"].get("parameters", dict()))
    return data


# Uncomment to see the distribution of the sampled data

# In[28]:


# sampler, weights = create_sampler(config)
# if sampler is not None:
#     label_cnt = Counter()
#     name_cnt = Counter()
#     n_samples = len(weights)
#     for idx in np.random.choice(train_idx, n_samples, p=weights / weights.sum()):
#         row = labels_df.iloc[idx]
#         labels = row['Target']
#         for l in labels:
#             label_cnt[l] += 1
#         name_cnt[row['Id']] += 1
#     print("Weighted sampled proportions:")
#     pprint(sorted({k: v / sum(label_cnt.values()) for k, v in label_cnt.items()}.items()))
#     # pprint(sorted({k: v for k, v in name_cnt.items()}.items(), key=lambda x: x[1]))
# else:
#     print("No weighted sampling")


# In[29]:


sampler, _ = create_sampler(config)
data = create_data_bunch(config, sampler, train_paths, val_paths, train_labels_one_hot, val_labels_one_hot, test_paths)

# In[32]:



# Uncomment to see a sample of the batch

# In[34]:


# x, _ = next(iter(data.train_dl))
# sample_x = x[1]
# plot_rgby(tensor2img(sample_x, denorm_fn=denormalize_fn))


# #### Initialize the model

# In[35]:


def create_model(config):
    model_name, model_parameters = extract_name_and_parameters(config, "model")
    model = model_lookup[model_name](**model_parameters)
    return model


# In[36]:


model = create_model(config)

# Uncomment to see model summary

# In[37]:


# x, _ = next(iter(data.train_dl))
# input_shape = x.shape[1:]
# summary(model.cuda(), input_shape)


# #### Initialize the callbacks

# In[38]:


learner_callback_lookup = {
    "OutputRecorder": partial(OutputRecorder, save_path=RESULTS_SAVE_PATH,
                              save_img_fn=partial(tensor2img, denorm_fn=denormalize_fn)),
    "CSVLogger": partial(CSVLogger, filename=str(RESULTS_SAVE_PATH / 'history')),
    "GradientClipping": fastai.GradientClipping,
}


# In[39]:


def create_callbacks(config):
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
    return callbacks, callback_fns


# In[40]:


callbacks, callback_fns = create_callbacks(config)


# In[41]:


# callbacks


# In[42]:


# callback_fns


# #### Create loss funcs

# In[43]:


def create_loss_funcs(config):
    loss_funcs = []
    for loss_func in config.get('loss_func', list()):
        name = loss_func['name']
        parameters = loss_func.get('parameters', dict())
        loss_funcs.append(loss_lookup[name](**parameters))
    return loss_funcs


# In[44]:


loss_funcs = create_loss_funcs(config)


# #### Create metrics

# In[45]:


def create_metrics(config):
    metrics = []
    for metric in config.get('metrics', list()):
        name = metric['name']
        parameters = metric.get('parameters', dict())
        metrics.append(partial(metric_lookup[name], **parameters))
    return metrics


# In[46]:


metrics = create_metrics(config)


# #### Create the Learner object

# In[47]:


def create_learner(config):
    sampler, _ = create_sampler(config)
    data = create_data_bunch(config, sampler, train_paths, val_paths, train_labels_one_hot, val_labels_one_hot,
                             test_paths)
    model = create_model(config)
    callbacks, callback_fns = create_callbacks(config)
    loss_funcs = create_loss_funcs(config)
    metrics = create_metrics(config)
    learner = Learner(data,
                      model=model,
                      loss_func=LossWrapper(loss_funcs),
                      callbacks=callbacks,
                      callback_fns=callback_fns,
                      # callback_fns=callback_fns + [LRPrinter],
                      metrics=metrics)
    return learner


# In[48]:

from src.training import training_scheme_lookup

learner = create_learner(config)
training_scheme_name, training_scheme_parameters = extract_name_and_parameters(config, "training_scheme")
training_scheme_lookup[training_scheme_name](learner=learner, **training_scheme_parameters)
learner.save(RESULTS_SAVE_PATH / 'model')
