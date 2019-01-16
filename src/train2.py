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
import logging
from collections import defaultdict, Counter
from pprint import pprint

import click
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
from src.transforms import augment_fn_lookup
from src.callbacks import OutputRecorder
from src.image import plot_rgby

import pytorch_toolbox.fastai.fastai as fastai
from pytorch_toolbox.utils.core import to_numpy
from pytorch_toolbox.fastai_extensions.vision.utils import denormalize_fn_lookup, normalize_fn_lookup, tensor2img
from pytorch_toolbox.fastai.fastai.callbacks import CSVLogger
from pytorch_toolbox.fastai_extensions.basic_train import Learner
from pytorch_toolbox.fastai_extensions.loss import LossWrapper, loss_lookup
from pytorch_toolbox.fastai_extensions.basic_data import DataBunch
from pytorch_toolbox.fastai_extensions.callbacks import callback_lookup
from pytorch_toolbox.fastai_extensions.metrics import metric_lookup
from pytorch_toolbox.pipeline import PipelineGraph


def set_logger(log_level):
    log_levels = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "NONSET": logging.NOTSET
    }
    logging.basicConfig(
        level=log_levels.get(log_level, logging.INFO),
    )


def load_training_data(root_image_paths, root_label_paths):
    X = sorted(list(Path(root_image_paths).glob("*")), key=lambda p: p.stem)
    labels_df = pd.read_csv(root_label_paths)
    labels_df['Target'] = [[int(i) for i in s.split()] for s in labels_df['Target']]
    labels_df = labels_df.sort_values(["Id"], ascending=[True])
    y = labels_df['Target'].values
    y_one_hot = make_one_hot(labels_df['Target'], n_classes=28)
    assert np.all(np.array([p.stem for p in X]) == labels_df["Id"])
    return np.array(X), np.array(y), np.array(y_one_hot)


def load_testing_data(root_image_paths):
    X = sorted(list(Path(root_image_paths).glob("*")), key=lambda p: p.stem)
    return np.array(X)


def create_data_bunch(train_idx, val_idx, train_X, train_y, test_X, train_ds, train_bs, val_ds, val_bs, test_ds,
                      test_bs, sampler, num_workers):
    sampler = sampler(y=train_y[train_idx])
    train_ds = train_ds(inputs=train_X[train_idx], labels=train_y[train_idx])
    val_ds = val_ds(inputs=train_X[val_idx], labels=train_y[val_idx])
    test_ds = test_ds(inputs=test_X),
    return DataBunch.create(train_ds, val_ds, test_ds,
                            train_bs=train_bs, val_bs=val_bs, test_bs=test_bs,
                            collate_fn=train_ds.collate_fn, sampler=sampler, num_workers=num_workers)


def create_sampler(y=None, sampler_fn=None):
    sampler = None
    if sampler_fn is not None:
        weights = np.array(sampler_fn(y))
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights))
    else:
        pass
    return sampler


def create_callbacks(callback_references):
    callbacks = []
    for cb_ref in callback_references:
        callbacks.append(cb_ref())
    return callbacks

def create_learner_callbacks(learner_callback_references):
    callback_fns = []
    for learn_cb_ref in learner_callback_references:
        try:
            callback_fns.append(learn_cb_ref())
        except TypeError:
            callback_fns.append(learn_cb_ref)
    return callback_fns


def create_learner(data, model_creator, callbacks_creator, callback_fns_creator, metrics, loss_funcs):
    model = model_creator()
    callbacks = callbacks_creator()
    callback_fns = callback_fns_creator()
    learner = Learner(data,
                      model=model,
                      loss_func=LossWrapper(loss_funcs),
                      metrics=metrics,
                      callbacks=callbacks,
                      callback_fns=callback_fns)
    return learner


def training_loop(create_learner, data_bunch_creator, data_splitter_iterable, **state_dict):
    for i, (train_idx, val_idx) in enumerate(data_splitter_iterable()):
        state_dict["current_fold"] = i
        data = data_bunch_creator(train_idx, val_idx)
        learner = create_learner(data)
        learner.lr_find()
        print("please?")


def create_time_stamped_save_path(save_path, **state_dict):
    current_time = state_dict.get("start_time")
    if current_time is None:
         current_time = f"{time.strftime('%Y%m%d-%H%M%S')}"
         state_dict["start_time"] = current_time
    current_fold = state_dict.get("current_fold")
    path = Path(save_path, current_time)
    if current_fold is not None:
        path = path / f"Fold_{current_fold}"
    return path



def create_output_recorder(save_path_creator, denormalize_fn):
    return partial(OutputRecorder, save_path=save_path_creator(), save_img_fn=partial(tensor2img, denormalize_fn=denormalize_fn))


def create_csv_logger(save_path_creator):
    return partial(CSVLogger, filename=str(save_path_creator() / 'history'))


learner_callback_lookup = {
    "create_output_recorder": create_output_recorder,
    "create_csv_logger": create_csv_logger,
    "GradientClipping": fastai.GradientClipping,
}

lookups = {
    **model_lookup,
    **dataset_lookup,
    **sampler_weight_lookup,
    **split_method_lookup,
    **augment_fn_lookup,
    **normalize_fn_lookup,
    **denormalize_fn_lookup,
    **loss_lookup,
    **callback_lookup,
    **metric_lookup,
    **sampler_weight_lookup,
    **learner_callback_lookup,
    "open_numpy": open_numpy,
    "load_training_data": load_training_data,
    "load_testing_data": load_testing_data,
    "create_data_bunch": create_data_bunch,
    "create_sampler": create_sampler,
    "create_learner": create_learner,
    "create_time_stamped_save_path": create_time_stamped_save_path,
    "create_callbacks": create_callbacks,
    'create_learner_callbacks': create_learner_callbacks,
    "training_loop": training_loop
}


@click.command()
@click.option('-cfg', '--config_file_path')
@click.option('-log-lvl', '--log_level', default="INFO")
def main(config_file_path, log_level):
    set_logger(log_level)
    with Path(config_file_path).open("r") as f:
        config = yaml.load(f)
    pipeline_graph = PipelineGraph.create_pipeline_graph_from_config(config)
    print(pipeline_graph.sorted_node_names)
    pipeline_graph.run_graph(reference_lookup=lookups)


if __name__ == '__main__':
    main()

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
# train_image_paths = list(DataPaths.TRAIN_RGBY_IMAGES.glob("*")) + list(DataPaths.TRAIN_RGBY_IMAGES_HPA_V18.glob("*"))
train_image_paths = pickle.load(open("train_image_paths.p", "rb"))
img_ids = sorted(list(set([p.name.split("_crop")[0] for p in train_image_paths])))
img_ids_lookup = Counter(img_ids)
# print(len(img_ids))

# #### Create training DataFrame

# In[9]:


labels_df = pd.read_csv(train_label_paths)
labels_df['Target'] = [[int(i) for i in s.split()] for s in labels_df['Target']]
labels_df = labels_df.sort_values(["Id"], ascending=[True])
labels_df = labels_df.loc[labels_df["Id"].map(lambda x: img_ids_lookup.get(x) is not None)]
# assert np.all(np.array(img_ids) == labels_df["Id"])
labels_one_hot = make_one_hot(labels_df['Target'], n_classes=28)
print("labels_df.shape")
print(labels_df.shape)
print("len(labels_one_hot)")
print(len(labels_one_hot))

# Get the names for each patch
# In[11]:

from tqdm import tqdm

lookup = {}
for i in tqdm(range(len(labels_df))):
    row = labels_df.iloc[i].to_dict()
    lookup[row['Id']] = row['Target']

# Create new df with oversampled patches for rare labels

# In[12]:


rows = []
for p in tqdm(train_image_paths):
    root_path = Path(p).parent
    img_ids_crop_with_color = "_".join(p.name.split("_"))
    img_ids_crop = "_".join(p.name.split("_")[:-1])
    base_id = img_ids_crop.split("_crop")[0]
    target = lookup[base_id]
    row = {
        'RootPath': root_path / img_ids_crop_with_color,
        'LoadPath': root_path / img_ids_crop,
        'Id': img_ids_crop,
        'BaseId': base_id,
        'Target': target
    }
    rows.append(row)


# for p in tqdm(train_image_paths):
#     root_path = Path(p).parent
#     img_ids_crop_with_color = "_".join(p.name.split("_"))
#     img_ids_crop = "_".join(p.name.split("_")[:-1])
#     load_ = "_".join(p.name.split("_"))
#     base_id = img_ids_crop.split("_crop")[0]
#     target = lookup[base_id]
#     row = {
#         'RootPath': root_path / img_ids_crop_with_color,
#         'LoadPath': root_path / img_ids_crop,
#         'Id': img_ids_crop,
#         'BaseId': base_id,
#         'Target': target
#     }
#     rows.append(row)


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
val_ids_lookup = Counter(val_ids)
train_df_rows = []
val_df_rows = []
for row in rows:
    train_id = train_ids_lookup.get(row['BaseId'])
    val_id = val_ids_lookup.get(row['BaseId'])
    if train_id is None and val_id is None:
        continue
    else:
        if train_id:
            train_df_rows.append(row)
        else:
            val_df_rows.append(row)

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
train_paths = np.array([root_path for root_path in train_df['LoadPath'].values])
train_labels_one_hot = make_one_hot(train_df['Target'], n_classes=28)

# Now create the paths to the images

# In[20]:

from collections import defaultdict

val_data = defaultdict(list)

sorted_val_df_rows = sorted(val_df_rows, key=lambda d: d['BaseId'])

for row in sorted_val_df_rows:
    data = {**row}
    val_data[row['BaseId']].append(row)

val_data = list(val_data.values())

# test_img_ids = sorted(list(set([p.name.split("_crop")[0] for p in test_image_paths])))
test_image_paths = DataPaths.TEST_RGBY_IMAGES.glob("*")
test_rows = []
for p in tqdm(test_image_paths):
    root_path = Path(p).parent
    img_ids_crop_with_color = "_".join(p.name.split("_"))
    img_ids_crop = "_".join(p.name.split("_")[:-1])
    base_id = img_ids_crop.split("_crop")[0]
    row = {
        'RootPath': root_path / img_ids_crop_with_color,
        'LoadPath': root_path / img_ids_crop,
        'Id': img_ids_crop,
        'BaseId': base_id,
    }
    test_rows.append(row)
test_data = defaultdict(list)
sorted_test_rows = sorted(test_rows, key=lambda d: d['BaseId'])

for row in sorted_test_rows:
    data = {**row}
    test_data[row['BaseId']].append(row)

for k, v in test_data.items():
    sorted_v = sorted(test_data[k], key=lambda x: x['LoadPath'])
    test_data[k] = sorted_v[120:]

test_data = list(test_data.values())
print("test data len is")
print(len(test_data))


# In[21]:


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


from src.data import ProteinClassificationDatasetForTest


def create_data_bunch(config, sampler, train_paths, train_labels_one_hot, val_data, test_data):
    dataset_name, dataset_parameters = extract_name_and_parameters(config, 'dataset')
    dataset = partial(dataset_lookup[dataset_name], **dataset_parameters)

    train_ds = dataset(inputs=np.array(train_paths),
                       open_image_fn=open_rgby,
                       labels=train_labels_one_hot,
                       augment_fn=augment_fn,
                       normalize_fn=normalize_fn)
    val_ds = ProteinClassificationDatasetForTest(data=np.array(val_data),
                                                 open_image_fn=open_rgby,
                                                 augment_fn=partial(albumentations_transform_wrapper,
                                                                    augment_fn=augment_fn_lookup["resize_aug"](p=1,
                                                                                                               height=384,
                                                                                                               width=384)),
                                                 normalize_fn=normalize_fn)
    test_ds = ProteinClassificationDatasetForTest(data=np.array(test_data),
                                                  open_image_fn=open_rgby,
                                                  augment_fn=partial(albumentations_transform_wrapper,
                                                                     augment_fn=augment_fn_lookup["resize_aug"](p=1,
                                                                                                                height=384,
                                                                                                                width=384)),
                                                  normalize_fn=normalize_fn)

    data = DataBunch.create(train_ds, val_ds, test_ds,
                            train_collate_fn=train_ds.collate_fn,
                            val_collate_fn=val_ds.collate_fn,
                            test_collate_fn=test_ds.collate_fn,
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
data = create_data_bunch(config, sampler, train_paths, train_labels_one_hot, val_data, test_data)


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

from src.callbacks import ImagePatchesPrediction, ImagePatchesConfidenceRecorder


def create_learner(config):
    sampler, _ = create_sampler(config)
    # data = create_data_bunch(config, sampler, train_paths, train_labels_one_hot, val_data, test_data)
    data = create_data_bunch(config, sampler, train_paths, train_labels_one_hot, val_data, test_data)
    model = create_model(config)
    callbacks, callback_fns = create_callbacks(config)
    loss_funcs = create_loss_funcs(config)
    metrics = create_metrics(config)
    learner = Learner(data,
                      model=model,
                      loss_func=LossWrapper(loss_funcs),
                      callbacks=callbacks,
                      callback_fns=callback_fns + [partial(ImagePatchesPrediction, save_path=RESULTS_SAVE_PATH),
                                                   partial(ImagePatchesConfidenceRecorder, save_path=RESULTS_SAVE_PATH)
                                                   ],
                      # callback_fns=callback_fns + [LRPrinter],
                      metrics=metrics)
    return learner


# In[48]:

from src.training import training_scheme_lookup

learner = create_learner(config)
# learner = learner.to_fp16()
learner.load_from_path(
    "/media/hd/Kaggle/human-protein-image-classification/results/saved_results/se_resnext50_32x4d_image_patches_20190111-060113/model_checkpoints/cycle_0_epoch_0.pth")


# training_scheme_name, training_scheme_parameters = extract_name_and_parameters(config, "training_scheme")
# training_scheme_lookup[training_scheme_name](learner=learner, **training_scheme_parameters)
# learner.save(RESULTS_SAVE_PATH / 'model')
# learner.predict_on_dl(learner.data.valid_dl, callback_fns=[partial(ImagePatchesPrediction, save_path=RESULTS_SAVE_PATH),
#                                                    partial(ImagePatchesConfidenceRecorder, save_path=RESULTS_SAVE_PATH)
#                                                    ])

class ResultRecorder(fastai.Callback):
    _order = -0.5

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
        print([last_target['name'][0].split("_crop")[0]])
        self.names.extend([last_target['name'][0].split("_crop")[0]])
        if self.phase == 'TRAIN' or self.phase == 'VAL':
            label = to_numpy(last_target['label'])
            self.targets.extend(label)

    def on_loss_begin(self, last_output, **kwargs):
        prob_pred = to_numpy(torch.sigmoid(last_output))
        self.prob_preds.extend(prob_pred)


# res_recorder = ResultRecorder()
# learner.predict_on_dl(dl=learner.data.valid_dl, callbacks=[res_recorder],
#                       callback_fns=[partial(ImagePatchesPrediction, save_path=RESULTS_SAVE_PATH),
#                                     ])

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


# pred_probs = np.stack(res_recorder.prob_preds)
# targets = np.stack(res_recorder.targets)
# print(targets.shape)
# print(pred_probs.shape)
#
# th = fit_val(pred_probs, targets)
# th[th < 0.1] = 0.1
# print('Thresholds: ', th)
# print('F1 macro: ', f1_score(targets, pred_probs > th, average='macro'))
# print('F1 macro (th = 0.5): ', f1_score(targets, pred_probs > 0.5, average='macro'))
# print('F1 micro: ', f1_score(targets, pred_probs > th, average='micro'))

learner.load_from_path(
    "/media/hd/Kaggle/human-protein-image-classification/results/saved_results/se_resnext50_32x4d_image_patches_20190111-060113/model_checkpoints/cycle_0_epoch_0.pth")
res_recorder = ResultRecorder()
learner.predict_on_dl(dl=learner.data.test_dl, callbacks=[res_recorder],
                      callback_fns=[partial(ImagePatchesPrediction, save_path=RESULTS_SAVE_PATH),
                                    ])

names = np.stack(res_recorder.names)
pred_probs = np.stack(res_recorder.prob_preds)
# print(names.shape)
# print(pred_probs.shape)
#
# predicted = []
# for pred in tqdm_notebook(pred_probs):
#     classes = [str(c) for c in np.where(pred > th)[0]]
#     if len(classes) == 0:
#         classes = [str(np.argmax(pred[0]))]
#     predicted.append(" ".join(classes))
#
# submission_df = pd.DataFrame({
#     "Id": names,
#     "Predicted": predicted
# })
#
# submission_df.to_csv(RESULTS_SAVE_PATH / "submission_optimal_threshold.csv", index=False)

predicted = []
for pred in tqdm_notebook(pred_probs):
    classes = [str(c) for c in np.where(pred > 0.5)[0]]
    if len(classes) == 0:
        classes = [str(np.argmax(pred[0]))]
    predicted.append(" ".join(classes))

submission_df = pd.DataFrame({
    "Id": names,
    "Predicted": predicted
})

submission_df.to_csv(RESULTS_SAVE_PATH / "submission_no_threshold.csv", index=False)
